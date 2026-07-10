"""
Private-chat admin-approval gate + support relay.

A brand-new private-chat user cannot use the bot until the admin approves
them. Every message an unapproved user sends is forwarded to the admin with
Approve/Reject buttons, and the admin can reply directly to a forwarded
message (Telegram's native "Reply") to have that reply relayed back to the
user — a lightweight helpdesk relay, no separate app needed.

Scope: PRIVATE chats only. Group access remains governed entirely by the
existing ALLOWED_CHAT_IDS / mention-required logic in bot.py; this module
doesn't touch that.

The admin is identified by username match (works immediately) and/or a
numeric ADMIN_USER_ID override (robust against username changes — see
config.py). The admin's chat_id (needed to actively PUSH messages to them,
which Telegram only allows once the bot has *received* at least one message
from that chat) is learned automatically the first time a message from the
admin identity arrives, and persisted from then on.

BOOTSTRAP GAP THIS MODULE HANDLES: if a regular user messages the bot
BEFORE the admin has ever sent the bot a single message (so the admin's
chat_id isn't known yet — no ADMIN_USER_ID configured), there is nowhere to
push that user's request to. Without special handling that request would be
silently dropped forever. Instead it's queued (queue_pending); the moment
the admin identity is next recognized, the whole queue is flushed to them
(see bot.py's AccessGateMiddleware) — so no early request is lost, it's
just delayed until the admin's first message.

STORAGE — the user/approval record is durable BUSINESS data (who's
approved, F.I.O, phone, audit trail), so it uses the three-tier fallback
described in db.py: PostgreSQL (DATABASE_URL) -> Redis (REDIS_URL) ->
in-memory. The relay-message-id -> user-chat-id map and the bootstrap
pending-queue are, by contrast, short-lived UI/session state (a relay only
matters until the admin acts on it) — those stay Redis/in-memory only, on
purpose; see db.py's module docstring for the full reasoning.
"""

import logging
from datetime import datetime, timezone

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

import db
import redis_client
from config import settings

logger = logging.getLogger(__name__)

_ADMIN_USERNAME_NORM = (settings.admin_username or "").lstrip("@").strip().lower()

_APPROVED_SET = "access:approved"
_DENIED_SET = "access:denied"
_ADMIN_CHAT_KEY = "access:admin_chat_id"
_RELAY_TTL_SECONDS = 60 * 60 * 24 * 7  # a relay mapping needn't outlive a week
_PENDING_LIST = "access:pending"
_MAX_PENDING = 500  # bound the queue in case the admin identity is never bootstrapped
_SEEN_SET = "access:seen"
_KNOWN_SET = "access:known"

# Post-approval onboarding: collect F.I.O + phone number before the user's
# messages reach the normal AI pipeline. States: "awaiting_fio" ->
# "awaiting_phone" -> "done". No entry / None means no onboarding pending
# (never approved yet, or approved before this feature existed).
_ONBOARD_STATE_KEY = "access:onboard:state"  # hash: user_id -> state


def _user_key(user_id: int) -> str:
    return f"access:user:{user_id}"


# --- in-memory fallback (bottom tier: no DATABASE_URL and no REDIS_URL) ---
_approved_mem: set[int] = set()
_denied_mem: set[int] = set()
_admin_chat_mem: int | None = None
_onboard_state_mem: dict[int, str] = {}
_pending_mem: list[tuple[int, int]] = []  # [(user_chat_id, message_id), ...]
_relay_mem: dict[int, int] = {}  # admin-side message_id -> user chat_id
_seen_mem: set[int] = set()
_users_mem: dict[int, dict] = {}  # user_id -> {full_name, username, first_seen, last_seen, message_count}


async def _pg() -> object | None:
    """Returns an initialised Postgres pool, or None to fall through to
    Redis/in-memory. Centralises the "is Postgres actually usable right
    now" check so every function below reads the same one-liner."""
    if await db.init_schema():
        return await db.get_pool()
    return None


def is_admin(user_id: int, username: str | None) -> bool:
    if settings.admin_user_id and user_id == settings.admin_user_id:
        return True
    uname_norm = (username or "").lstrip("@").strip().lower()
    return bool(_ADMIN_USERNAME_NORM) and uname_norm == _ADMIN_USERNAME_NORM


async def remember_admin_chat(chat_id: int) -> None:
    global _admin_chat_mem
    pool = await _pg()
    if pool is not None:
        await db.kv_set("admin_chat_id", str(chat_id))
        return
    client = redis_client.get_client()
    if client is None:
        _admin_chat_mem = chat_id
        return
    try:
        await client.set(_ADMIN_CHAT_KEY, str(chat_id))
    except Exception:  # noqa: BLE001
        logger.exception("Redis remember_admin_chat failed")
        _admin_chat_mem = chat_id


async def get_admin_chat_id() -> int | None:
    # A configured numeric ID is always authoritative and needs no learning —
    # private-chat chat_id == user_id on Telegram, so it doubles as the target.
    if settings.admin_user_id:
        return settings.admin_user_id
    pool = await _pg()
    if pool is not None:
        raw = await db.kv_get("admin_chat_id")
        return int(raw) if raw else _admin_chat_mem
    client = redis_client.get_client()
    if client is None:
        return _admin_chat_mem
    try:
        raw = await client.get(_ADMIN_CHAT_KEY)
        return int(raw) if raw else _admin_chat_mem
    except Exception:  # noqa: BLE001
        logger.exception("Redis get_admin_chat_id failed")
        return _admin_chat_mem


async def is_approved(user_id: int) -> bool:
    pool = await _pg()
    if pool is not None:
        try:
            async with pool.acquire() as conn:
                status = await conn.fetchval("SELECT status FROM users WHERE user_id = $1", user_id)
            return status == "approved"
        except Exception:  # noqa: BLE001
            logger.exception("Postgres is_approved failed")
            return user_id in _approved_mem
    client = redis_client.get_client()
    if client is None:
        return user_id in _approved_mem
    try:
        return bool(await client.sismember(_APPROVED_SET, user_id))
    except Exception:  # noqa: BLE001
        logger.exception("Redis is_approved failed")
        return user_id in _approved_mem


async def is_denied(user_id: int) -> bool:
    pool = await _pg()
    if pool is not None:
        try:
            async with pool.acquire() as conn:
                status = await conn.fetchval("SELECT status FROM users WHERE user_id = $1", user_id)
            return status == "denied"
        except Exception:  # noqa: BLE001
            logger.exception("Postgres is_denied failed")
            return user_id in _denied_mem
    client = redis_client.get_client()
    if client is None:
        return user_id in _denied_mem
    try:
        return bool(await client.sismember(_DENIED_SET, user_id))
    except Exception:  # noqa: BLE001
        logger.exception("Redis is_denied failed")
        return user_id in _denied_mem


async def mark_first_contact(user_id: int) -> bool:
    """Returns True the FIRST time this unapproved user is seen, False on
    every call after — lets the gate show the full "you need approval"
    explanation only once, then a short acknowledgment, so a back-and-forth
    with the admin (asking questions while waiting) doesn't repeat a wall of
    text on every single message."""
    pool = await _pg()
    if pool is not None:
        try:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "INSERT INTO seen_users (user_id) VALUES ($1) ON CONFLICT DO NOTHING", user_id,
                )
            return result.endswith("1")  # "INSERT 0 1" if a row was actually inserted, "INSERT 0 0" if not
        except Exception:  # noqa: BLE001
            logger.exception("Postgres mark_first_contact failed")
            if user_id in _seen_mem:
                return False
            _seen_mem.add(user_id)
            return True
    client = redis_client.get_client()
    if client is None:
        if user_id in _seen_mem:
            return False
        _seen_mem.add(user_id)
        return True
    try:
        added = await client.sadd(_SEEN_SET, user_id)
        return bool(added)
    except Exception:  # noqa: BLE001
        logger.exception("Redis mark_first_contact failed")
        if user_id in _seen_mem:
            return False
        _seen_mem.add(user_id)
        return True


async def approve(user_id: int, via: str = "") -> None:
    """`via` is an audit label ("button"/"command") stored alongside the
    decision timestamp — the admin reported a user showing up approved that
    they don't remember approving, and without when/how on record there is
    no way to reconstruct such an event after the fact."""
    now = datetime.now(timezone.utc)
    pool = await _pg()
    if pool is not None:
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO users (user_id, status, approved_at, approved_via, denied_at, denied_via)
                    VALUES ($1, 'approved', $2, $3, NULL, NULL)
                    ON CONFLICT (user_id) DO UPDATE SET
                        status = 'approved', approved_at = $2, approved_via = $3,
                        denied_at = NULL, denied_via = NULL
                    """,
                    user_id, now, via or "?",
                )
            return
        except Exception:  # noqa: BLE001
            logger.exception("Postgres approve failed")
    client = redis_client.get_client()
    if client is None:
        _approved_mem.add(user_id)
        _denied_mem.discard(user_id)
        rec = _users_mem.setdefault(user_id, {})
        rec["approved_at"] = now.isoformat()
        rec["approved_via"] = via or "?"
        rec.pop("denied_at", None)
        return
    try:
        async with client.pipeline(transaction=True) as pipe:
            pipe.sadd(_APPROVED_SET, user_id)
            pipe.srem(_DENIED_SET, user_id)
            pipe.hset(_user_key(user_id), mapping={"approved_at": now.isoformat(), "approved_via": via or "?"})
            pipe.hdel(_user_key(user_id), "denied_at", "denied_via")
            await pipe.execute()
    except Exception:  # noqa: BLE001
        logger.exception("Redis approve failed")
        _approved_mem.add(user_id)


async def deny(user_id: int, via: str = "") -> None:
    """Mark denied AND revoke any prior approval — deny() must be able to cut
    off someone who was previously approved, not just record-keep a rejection
    for someone who was never approved in the first place."""
    now = datetime.now(timezone.utc)
    pool = await _pg()
    if pool is not None:
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO users (user_id, status, denied_at, denied_via, approved_at, approved_via)
                    VALUES ($1, 'denied', $2, $3, NULL, NULL)
                    ON CONFLICT (user_id) DO UPDATE SET
                        status = 'denied', denied_at = $2, denied_via = $3,
                        approved_at = NULL, approved_via = NULL
                    """,
                    user_id, now, via or "?",
                )
            return
        except Exception:  # noqa: BLE001
            logger.exception("Postgres deny failed")
    client = redis_client.get_client()
    if client is None:
        _denied_mem.add(user_id)
        _approved_mem.discard(user_id)
        rec = _users_mem.setdefault(user_id, {})
        rec["denied_at"] = now.isoformat()
        rec["denied_via"] = via or "?"
        rec.pop("approved_at", None)
        return
    try:
        async with client.pipeline(transaction=True) as pipe:
            pipe.sadd(_DENIED_SET, user_id)
            pipe.srem(_APPROVED_SET, user_id)
            pipe.hset(_user_key(user_id), mapping={"denied_at": now.isoformat(), "denied_via": via or "?"})
            pipe.hdel(_user_key(user_id), "approved_at", "approved_via")
            await pipe.execute()
    except Exception:  # noqa: BLE001
        logger.exception("Redis deny failed")
        _denied_mem.add(user_id)
        _approved_mem.discard(user_id)


# --------------------------------------------------------------------------
# Post-approval onboarding: F.I.O + phone number collection
# --------------------------------------------------------------------------
async def set_onboarding_state(user_id: int, state: str) -> None:
    pool = await _pg()
    if pool is not None:
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO users (user_id, onboarding_state) VALUES ($1, $2)
                    ON CONFLICT (user_id) DO UPDATE SET onboarding_state = $2
                    """,
                    user_id, state,
                )
            return
        except Exception:  # noqa: BLE001
            logger.exception("Postgres set_onboarding_state failed")
    client = redis_client.get_client()
    if client is None:
        _onboard_state_mem[user_id] = state
        return
    try:
        await client.hset(_ONBOARD_STATE_KEY, str(user_id), state)
    except Exception:  # noqa: BLE001
        logger.exception("Redis set_onboarding_state failed")
        _onboard_state_mem[user_id] = state


async def get_onboarding_state(user_id: int) -> str | None:
    pool = await _pg()
    if pool is not None:
        try:
            async with pool.acquire() as conn:
                return await conn.fetchval("SELECT onboarding_state FROM users WHERE user_id = $1", user_id)
        except Exception:  # noqa: BLE001
            logger.exception("Postgres get_onboarding_state failed")
            return _onboard_state_mem.get(user_id)
    client = redis_client.get_client()
    if client is None:
        return _onboard_state_mem.get(user_id)
    try:
        return await client.hget(_ONBOARD_STATE_KEY, str(user_id))
    except Exception:  # noqa: BLE001
        logger.exception("Redis get_onboarding_state failed")
        return _onboard_state_mem.get(user_id)


async def save_profile_field(user_id: int, field: str, value: str) -> None:
    """Writes into the user's row (Postgres) / the same per-user hash
    record_activity() uses (Redis fallback), so list_users() picks up
    fio/phone automatically with no extra round-trip."""
    if field not in ("fio", "phone"):
        raise ValueError(f"unsupported profile field: {field!r}")
    pool = await _pg()
    if pool is not None:
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    f"""
                    INSERT INTO users (user_id, {field}) VALUES ($1, $2)
                    ON CONFLICT (user_id) DO UPDATE SET {field} = $2
                    """,
                    user_id, value,
                )
            return
        except Exception:  # noqa: BLE001
            logger.exception("Postgres save_profile_field failed")
    client = redis_client.get_client()
    if client is None:
        _users_mem.setdefault(user_id, {})[field] = value
        return
    try:
        await client.hset(_user_key(user_id), field, value)
    except Exception:  # noqa: BLE001
        logger.exception("Redis save_profile_field failed")
        _users_mem.setdefault(user_id, {})[field] = value


async def get_known_full_name(user_id: int) -> str | None:
    """The Telegram display name record_activity() already captured for this
    user (first message or later) — used to auto-fill F.I.O without asking,
    since Telegram exposes this name freely but NEVER the phone number
    (that requires the user's own explicit contact-share action, no way
    around it via the Bot API)."""
    pool = await _pg()
    if pool is not None:
        try:
            async with pool.acquire() as conn:
                name = await conn.fetchval("SELECT full_name FROM users WHERE user_id = $1", user_id)
            return name or None
        except Exception:  # noqa: BLE001
            logger.exception("Postgres get_known_full_name failed")
            return _users_mem.get(user_id, {}).get("full_name") or None
    client = redis_client.get_client()
    if client is None:
        name = _users_mem.get(user_id, {}).get("full_name")
        return name or None
    try:
        name = await client.hget(_user_key(user_id), "full_name")
        return name or None
    except Exception:  # noqa: BLE001
        logger.exception("Redis get_known_full_name failed")
        return _users_mem.get(user_id, {}).get("full_name") or None


# --------------------------------------------------------------------------
# Bootstrap-gap queue + admin-reply relay — short-lived UI/session state,
# NOT durable business data. Stays Redis/in-memory only; see db.py's
# module docstring for why these two kinds of state are treated differently.
# --------------------------------------------------------------------------
async def queue_pending(user_chat_id: int, message_id: int) -> None:
    """Stash a request that couldn't be forwarded because the admin's
    chat_id isn't known yet — see the bootstrap-gap note in the module
    docstring. Flushed by pop_all_pending() once the admin is recognized."""
    client = redis_client.get_client()
    entry = f"{user_chat_id}:{message_id}"
    if client is None:
        _pending_mem.append((user_chat_id, message_id))
        del _pending_mem[: max(0, len(_pending_mem) - _MAX_PENDING)]
        return
    try:
        async with client.pipeline(transaction=True) as pipe:
            pipe.rpush(_PENDING_LIST, entry)
            pipe.ltrim(_PENDING_LIST, -_MAX_PENDING, -1)
            await pipe.execute()
    except Exception:  # noqa: BLE001
        logger.exception("Redis queue_pending failed")
        _pending_mem.append((user_chat_id, message_id))
        del _pending_mem[: max(0, len(_pending_mem) - _MAX_PENDING)]


async def pop_all_pending() -> list[tuple[int, int]]:
    """Return and clear every queued (user_chat_id, message_id) pair."""
    client = redis_client.get_client()
    if client is None:
        out = list(_pending_mem)
        _pending_mem.clear()
        return out
    try:
        async with client.pipeline(transaction=True) as pipe:
            pipe.lrange(_PENDING_LIST, 0, -1)
            pipe.delete(_PENDING_LIST)
            results = await pipe.execute()
        out = []
        for entry in results[0]:
            cid_s, _, mid_s = entry.partition(":")
            if cid_s.lstrip("-").isdigit() and mid_s.isdigit():
                out.append((int(cid_s), int(mid_s)))
        return out
    except Exception:  # noqa: BLE001
        logger.exception("Redis pop_all_pending failed")
        out = list(_pending_mem)
        _pending_mem.clear()
        return out


async def link_relay(admin_message_id: int, user_chat_id: int) -> None:
    client = redis_client.get_client()
    if client is None:
        _relay_mem[admin_message_id] = user_chat_id
        return
    try:
        await client.setex(f"access:relay:{admin_message_id}", _RELAY_TTL_SECONDS, str(user_chat_id))
    except Exception:  # noqa: BLE001
        logger.exception("Redis link_relay failed")
        _relay_mem[admin_message_id] = user_chat_id


async def resolve_relay(admin_message_id: int) -> int | None:
    client = redis_client.get_client()
    if client is None:
        return _relay_mem.get(admin_message_id)
    try:
        raw = await client.get(f"access:relay:{admin_message_id}")
        return int(raw) if raw else _relay_mem.get(admin_message_id)
    except Exception:  # noqa: BLE001
        logger.exception("Redis resolve_relay failed")
        return _relay_mem.get(admin_message_id)


def access_keyboard(user_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ Ruxsat berish", callback_data=f"acc:a:{user_id}"),
        InlineKeyboardButton(text="❌ Rad etish", callback_data=f"acc:r:{user_id}"),
    ]])


# --------------------------------------------------------------------------
# User directory — lets the admin see who has talked to the bot, when, and
# how much, regardless of approval status (/users in bot.py).
# --------------------------------------------------------------------------
async def record_activity(user_id: int, username: str | None, full_name: str | None) -> None:
    """Update (creating on first contact) this user's activity record.
    Called on every private-chat message from a non-admin user — including
    already-approved ones — so /users reflects real usage, not just
    pending-request traffic."""
    now = datetime.now(timezone.utc)
    pool = await _pg()
    if pool is not None:
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO users (user_id, username, full_name, first_seen, last_seen, message_count)
                    VALUES ($1, $2, $3, $4, $4, 1)
                    ON CONFLICT (user_id) DO UPDATE SET
                        username = $2, full_name = $3, last_seen = $4,
                        message_count = users.message_count + 1
                    """,
                    user_id, username or None, full_name or None, now,
                )
            return
        except Exception:  # noqa: BLE001
            logger.exception("Postgres record_activity failed")
    now_iso = now.isoformat()
    client = redis_client.get_client()
    if client is None:
        rec = _users_mem.setdefault(user_id, {"first_seen": now_iso, "message_count": "0"})
        rec["username"] = username or ""
        rec["full_name"] = full_name or ""
        rec["last_seen"] = now_iso
        # Keep as str in-memory too — Redis hashes always return strings via
        # hgetall, and list_users()/the /users display shouldn't have to
        # care which backend served a given record.
        rec["message_count"] = str(int(rec.get("message_count", "0")) + 1)
        return
    try:
        async with client.pipeline(transaction=True) as pipe:
            pipe.hsetnx(_user_key(user_id), "first_seen", now_iso)
            pipe.hset(_user_key(user_id), mapping={
                "username": username or "", "full_name": full_name or "", "last_seen": now_iso,
            })
            pipe.hincrby(_user_key(user_id), "message_count", 1)
            pipe.sadd(_KNOWN_SET, user_id)
            await pipe.execute()
    except Exception:  # noqa: BLE001
        logger.exception("Redis record_activity failed")
        rec = _users_mem.setdefault(user_id, {"first_seen": now_iso, "message_count": "0"})
        rec["username"] = username or ""
        rec["full_name"] = full_name or ""
        rec["last_seen"] = now_iso
        rec["message_count"] = str(int(rec.get("message_count", "0")) + 1)


def _row_to_dict(row) -> dict:
    rec = dict(row)
    for ts_field in ("approved_at", "denied_at", "first_seen", "last_seen"):
        if rec.get(ts_field) is not None:
            rec[ts_field] = rec[ts_field].isoformat()
    rec["message_count"] = str(rec.get("message_count") or 0)
    return rec


async def list_users() -> list[dict]:
    """Every user who has ever messaged the bot (or been approved/denied
    directly), newest activity first, each with a 'status' of
    pending/approved/denied merged in."""
    pool = await _pg()
    if pool is not None:
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM users ORDER BY last_seen DESC NULLS LAST")
            return [_row_to_dict(r) for r in rows]
        except Exception:  # noqa: BLE001
            logger.exception("Postgres list_users failed")
            return []
    client = redis_client.get_client()
    if client is None:
        ids = set(_users_mem) | _approved_mem | _denied_mem
        out = []
        for uid in ids:
            rec = dict(_users_mem.get(uid, {}))
            rec["user_id"] = uid
            rec["status"] = "approved" if uid in _approved_mem else ("denied" if uid in _denied_mem else "pending")
            out.append(rec)
        out.sort(key=lambda r: r.get("last_seen", ""), reverse=True)
        return out
    try:
        known_raw, approved_raw, denied_raw = await client.smembers(_KNOWN_SET), \
            await client.smembers(_APPROVED_SET), await client.smembers(_DENIED_SET)
        approved_ids = {int(x) for x in approved_raw}
        denied_ids = {int(x) for x in denied_raw}
        all_ids = {int(x) for x in known_raw} | approved_ids | denied_ids
        out = []
        for uid in all_ids:
            raw = await client.hgetall(_user_key(uid))
            rec = dict(raw)
            rec["user_id"] = uid
            rec["status"] = "approved" if uid in approved_ids else ("denied" if uid in denied_ids else "pending")
            out.append(rec)
        out.sort(key=lambda r: r.get("last_seen", ""), reverse=True)
        return out
    except Exception:  # noqa: BLE001
        logger.exception("Redis list_users failed")
        return []
