"""
Quick actions — one-tap follow-ups attached under a substantive AI answer
in a private chat: turn the answer into a Word/PDF document, or into a
tracked task/reminder, without retyping anything into /proposal or
/addtask. Closes the "dead-end answer" flow gap: the pipeline produced
real content, and the obvious next steps were previously manual.

Same relay-store pattern as business_copilot/group_copilot: the sent
answer message's message_id keys a small payload (original request + the
answer body), Redis-backed with the usual in-memory fallback, TTL'd —
and ONE-SHOT: the entry is cleared after either button is used, so a
double-tap can't generate two documents (an LLM + file-render round) or
two duplicate tasks.
"""

import json
import logging

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

import redis_client

logger = logging.getLogger(__name__)

_TTL_SECONDS = 60 * 60 * 24 * 7

# --- in-memory fallback ---
_mem: dict[int, dict] = {}


def _key(message_id: int) -> str:
    return f"qa:{message_id}"


async def link(message_id: int, user_text: str, body: str) -> None:
    payload = json.dumps({"text": user_text[:1500], "body": body[:6000]})
    client = redis_client.get_client()
    if client is None:
        _mem[message_id] = json.loads(payload)
        return
    try:
        await client.setex(_key(message_id), _TTL_SECONDS, payload)
    except Exception:  # noqa: BLE001
        logger.exception("Redis quick_actions.link failed")
        _mem[message_id] = json.loads(payload)


async def resolve(message_id: int) -> dict | None:
    client = redis_client.get_client()
    if client is None:
        return _mem.get(message_id)
    try:
        raw = await client.get(_key(message_id))
        return json.loads(raw) if raw else _mem.get(message_id)
    except Exception:  # noqa: BLE001
        logger.exception("Redis quick_actions.resolve failed")
        return _mem.get(message_id)


async def clear(message_id: int) -> None:
    client = redis_client.get_client()
    if client is None:
        _mem.pop(message_id, None)
        return
    try:
        await client.delete(_key(message_id))
    except Exception:  # noqa: BLE001
        logger.exception("Redis quick_actions.clear failed")
        _mem.pop(message_id, None)


def keyboard(message_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="📄 Word/PDF hujjat", callback_data=f"qa:d:{message_id}"),
        InlineKeyboardButton(text="📋 Vazifa qilish", callback_data=f"qa:t:{message_id}"),
    ]])
