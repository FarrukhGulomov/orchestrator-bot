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

Storage: Redis (approved/denied sets, learned admin chat_id, and the
relay-message-id -> user-chat-id map used to route admin replies), with the
same in-memory fallback pattern as the rest of the codebase.
"""

import logging

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

import redis_client
from config import settings

logger = logging.getLogger(__name__)

_APPROVED_SET = "access:approved"
_DENIED_SET = "access:denied"
_ADMIN_CHAT_KEY = "access:admin_chat_id"
_RELAY_TTL_SECONDS = 60 * 60 * 24 * 7  # a relay mapping needn't outlive a week

_ADMIN_USERNAME_NORM = (settings.admin_username or "").lstrip("@").strip().lower()

# --- in-memory fallback ---
_approved_mem: set[int] = set()
_denied_mem: set[int] = set()
_admin_chat_mem: int | None = None
_relay_mem: dict[int, int] = {}  # admin-side message_id -> user chat_id


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
