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

Storage: Redis (approved/denied sets, learned admin chat_id, the
relay-message-id -> user-chat-id map used to route admin replies, and the
pending-request queue for the bootstrap gap above), with the same
in-memory fallback pattern as the rest of the codebase.
"""

import logging
from datetime import datetime, timezone

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

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


def _user_key(user_id: int) -> str:
    return f"access:user:{user_id}"


# --- in-memory fallback ---
_approved_mem: set[int] = set()
_denied_mem: set[int] = set()
_admin_chat_mem: int | None = None
_pending_mem: list[tuple[int, int]] = []  # [(user_chat_id, message_id), ...]
_relay_mem: dict[int, int] = {}  # admin-side message_id -> user chat_id
_seen_mem: set[int] = set()
_users_mem: dict[int, dict] = {}  # user_id -> {full_name, username, first_seen, last_seen, message_count}


def is_admin(user_id: int, username: str | None) -> bool:
    if settings.admin_user_id and user_id == settings.admin_user_id:
        return True
    uname_norm = (username or "").lstrip("@").strip().lower()
    return bool(_ADMIN_USERNAME_NORM) and uname_norm == _ADMIN_USERNAME_NORM


async def remember_admin_chat(chat_id: int) -> None:
    global _admin_chat_mem
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
    client = redis_client.get_client()
    if client is None:
        return user_id in _approved_mem
    try:
        return bool(await client.sismember(_APPROVED_SET, user_id))
    except Exception:  # noqa: BLE001
        logger.exception("Redis is_approved failed")
        return user_id in _approved_mem


async def mark_first_contact(user_id: int) -> bool:
    """Returns True the FIRST time this unapproved user is seen, False on
    every call after — lets the gate show the full "you need approval"
    explanation only once, then a short acknowledgment, so a back-and-forth
    with the admin (asking questions while waiting) doesn't repeat a wall of
    text on every single message."""
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


async def approve(user_id: int) -> None:
    client = redis_client.get_client()
    if client is None:
        _approved_mem.add(user_id)
        _denied_mem.discard(user_id)
        return
    try:
        async with client.pipeline(transaction=True) as pipe:
            pipe.sadd(_APPROVED_SET, user_id)
            pipe.srem(_DENIED_SET, user_id)
            await pipe.execute()
    except Exception:  # noqa: BLE001
        logger.exception("Redis approve failed")
        _approved_mem.add(user_id)


async def deny(user_id: int) -> None:
    """Mark denied AND revoke any prior approval — deny() must be able to cut
    off someone who was previously approved, not just record-keep a rejection
    for someone who was never approved in the first place."""
    client = redis_client.get_client()
    if client is None:
        _denied_mem.add(user_id)
        _approved_mem.discard(user_id)
        return
    try:
        async with client.pipeline(transaction=True) as pipe:
            pipe.sadd(_DENIED_SET, user_id)
            pipe.srem(_APPROVED_SET, user_id)
            await pipe.execute()
    except Exception:  # noqa: BLE001
        logger.exception("Redis deny failed")
        _denied_mem.add(user_id)
        _approved_mem.discard(user_id)


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
    now = datetime.now(timezone.utc).isoformat()
    client = redis_client.get_client()
    if client is None:
        rec = _users_mem.setdefault(user_id, {"first_seen": now, "message_count": "0"})
        rec["username"] = username or ""
        rec["full_name"] = full_name or ""
        rec["last_seen"] = now
        # Keep as str in-memory too — Redis hashes always return strings via
        # hgetall, and list_users()/the /users display shouldn't have to
        # care which backend served a given record.
        rec["message_count"] = str(int(rec.get("message_count", "0")) + 1)
        return
    try:
        async with client.pipeline(transaction=True) as pipe:
            pipe.hsetnx(_user_key(user_id), "first_seen", now)
            pipe.hset(_user_key(user_id), mapping={
                "username": username or "", "full_name": full_name or "", "last_seen": now,
            })
            pipe.hincrby(_user_key(user_id), "message_count", 1)
            pipe.sadd(_KNOWN_SET, user_id)
            await pipe.execute()
    except Exception:  # noqa: BLE001
        logger.exception("Redis record_activity failed")
        rec = _users_mem.setdefault(user_id, {"first_seen": now, "message_count": "0"})
        rec["username"] = username or ""
        rec["full_name"] = full_name or ""
        rec["last_seen"] = now
        # Keep as str in-memory too — Redis hashes always return strings via
        # hgetall, and list_users()/the /users display shouldn't have to
        # care which backend served a given record.
        rec["message_count"] = str(int(rec.get("message_count", "0")) + 1)


async def list_users() -> list[dict]:
    """Every user who has ever messaged the bot (or been approved/denied
    directly), newest activity first, each with a 'status' of
    pending/approved/denied merged in."""
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
