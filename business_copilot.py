"""
Telegram Business Bots integration — an AI copilot for the admin's OWN
personal Telegram chats, connected via Telegram's native "Business > Chat
automation" settings (a completely separate API surface from the rest of
this bot: business_connection / business_message updates, not the regular
message/private-chat flow access_control.py governs).

Flow:
  1. The admin connects this bot to specific personal chats via their own
     Telegram app (Settings > Business > Chatbots). Telegram then sends a
     `business_connection` update — cached here (save_connection).
  2. When a contact messages one of those connected chats, Telegram sends a
     `business_message` update. This module analyzes it with AI and drafts
     a suggested reply, then notifies the admin in their own normal DM with
     THIS bot (chat_id = BusinessConnection.user_chat_id) — never auto-sends
     anything on its own.
  3. The admin can tap "✅ Yuborish" to send the suggested reply into the
     business chat as themselves (business_connection_id makes Telegram
     attribute it to their account), or use Telegram's native "Reply" on
     the notification to send their OWN text instead — same relay pattern
     as access_control's admin-reply relay (see bot.py's
     AccessGateMiddleware, which checks both relay stores).

Only the connection's OWNER is ever notified, and only if they are the
configured admin (ADMIN_USERNAME/ADMIN_USER_ID) — this doesn't generalise
to arbitrary users connecting the bot to their own Business account.

Storage: Redis (connection cache, a short rolling per-chat history used for
reply context, and the relay map), with the same in-memory fallback
pattern as the rest of the codebase.
"""

import asyncio
import json
import logging
from collections import deque

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

import redis_client
from config import settings
from llm_clients import claude_generate_json

logger = logging.getLogger(__name__)

_HISTORY_CAP = 8
_HISTORY_TTL_SECONDS = 60 * 60 * 24 * 3
_RELAY_TTL_SECONDS = 60 * 60 * 24 * 7

# --- in-memory fallback ---
_conn_mem: dict[str, dict] = {}
_history_mem: dict[str, deque] = {}
_relay_mem: dict[int, dict] = {}  # admin-side message_id -> {conn, chat, reply}


def _conn_key(business_connection_id: str) -> str:
    return f"biz:conn:{business_connection_id}"


def _hist_key(business_connection_id: str, customer_chat_id: int) -> str:
    return f"biz:hist:{business_connection_id}:{customer_chat_id}"


def _relay_key(admin_message_id: int) -> str:
    return f"biz:relay:{admin_message_id}"


# --------------------------------------------------------------------------
# Connection cache
# --------------------------------------------------------------------------
async def save_connection(
    business_connection_id: str, owner_user_id: int, owner_chat_id: int, is_enabled: bool, can_reply: bool = True
) -> None:
    payload = json.dumps({
        "owner_user_id": owner_user_id, "owner_chat_id": owner_chat_id,
        "is_enabled": is_enabled, "can_reply": can_reply,
    })
    client = redis_client.get_client()
    if client is None:
        _conn_mem[business_connection_id] = json.loads(payload)
        return
    try:
        await client.set(_conn_key(business_connection_id), payload)
    except Exception:  # noqa: BLE001
        logger.exception("Redis save_connection failed")
        _conn_mem[business_connection_id] = json.loads(payload)


async def get_connection(business_connection_id: str) -> dict | None:
    client = redis_client.get_client()
    if client is None:
        return _conn_mem.get(business_connection_id)
    try:
        raw = await client.get(_conn_key(business_connection_id))
        return json.loads(raw) if raw else _conn_mem.get(business_connection_id)
    except Exception:  # noqa: BLE001
        logger.exception("Redis get_connection failed")
        return _conn_mem.get(business_connection_id)


# --------------------------------------------------------------------------
# Short rolling history per connected chat — gives the AI enough context for
# a coherent suggested reply without needing the full conversation.
# --------------------------------------------------------------------------
async def _append_history(business_connection_id: str, customer_chat_id: int, role: str, text: str) -> None:
    entry = json.dumps({"role": role, "text": text[:1000]})
    key = _hist_key(business_connection_id, customer_chat_id)
    client = redis_client.get_client()
    if client is None:
        dq = _history_mem.setdefault(key, deque(maxlen=_HISTORY_CAP))
        dq.append(entry)
        return
    try:
        async with client.pipeline(transaction=True) as pipe:
            pipe.rpush(key, entry)
            pipe.ltrim(key, -_HISTORY_CAP, -1)
            pipe.expire(key, _HISTORY_TTL_SECONDS)
            await pipe.execute()
    except Exception:  # noqa: BLE001
        logger.exception("Redis _append_history failed")
        dq = _history_mem.setdefault(key, deque(maxlen=_HISTORY_CAP))
        dq.append(entry)


async def _get_history(business_connection_id: str, customer_chat_id: int) -> list[dict]:
    key = _hist_key(business_connection_id, customer_chat_id)
    client = redis_client.get_client()
    if client is None:
        return [json.loads(e) for e in _history_mem.get(key, [])]
    try:
        raw = await client.lrange(key, 0, -1)
        return [json.loads(e) for e in raw]
    except Exception:  # noqa: BLE001
        logger.exception("Redis _get_history failed")
        return [json.loads(e) for e in _history_mem.get(key, [])]


async def append_contact_message(business_connection_id: str, customer_chat_id: int, text: str) -> None:
    await _append_history(business_connection_id, customer_chat_id, "them", text)


async def append_own_message(business_connection_id: str, customer_chat_id: int, text: str) -> None:
    await _append_history(business_connection_id, customer_chat_id, "me", text)


# --------------------------------------------------------------------------
# AI analysis + suggested reply
# --------------------------------------------------------------------------
_COPILOT_SYSTEM = """
You are a business-chat assistant helping a busy professional handle their
personal Telegram Business conversations. You're shown the recent exchange
with a contact and their latest message. Respond with ONLY this JSON, no
prose, no markdown fences:
{"analysis": "<one short line: intent/tone/urgency, in Uzbek>",
 "is_task": true|false,
 "suggested_reply": "<ready-to-send reply, SAME language as the contact's message>"}

is_task=true when the contact is asking for actual professional work to be
DONE — build/develop something, write a document, spec out a project,
quote a price/timeline, technical requirements, code, etc. — not just a
quick question or small talk.

When is_task=true, suggested_reply must be an HONEST holding reply only
(e.g. "tushunarli, jamoam bilan ko'rib chiqib tez orada aniq javob
beraman") — NEVER invent a commitment, timeline, price, or scope, since a
real answer needs the specialist team's actual input first, not a guess.
When is_task=false, suggested_reply is the real, ready-to-send answer.

Keep the suggested reply natural, professional, and concise — something a
real person would actually type, not a template or a chatbot-sounding line.
"""


async def analyze(business_connection_id: str, customer_chat_id: int, sender_name: str, latest_text: str) -> dict | None:
    history = await _get_history(business_connection_id, customer_chat_id)
    convo = "\n".join(
        f"{'Contact' if h['role'] == 'them' else 'You'}: {h['text']}" for h in history[-6:]
    )
    context = f"Recent exchange:\n{convo}\n\nContact ({sender_name})'s latest message: {latest_text}"
    try:
        raw = await asyncio.wait_for(
            claude_generate_json(_COPILOT_SYSTEM, [{"role": "user", "content": context}], max_tokens=400),
            timeout=settings.request_timeout,
        )
        data = json.loads(raw)
    except Exception:  # noqa: BLE001
        logger.exception("Business copilot analysis failed")
        return None
    if not isinstance(data, dict) or not str(data.get("suggested_reply", "")).strip():
        return None
    return data


# --------------------------------------------------------------------------
# Relay: admin-side notification message_id -> where/what to send if approved
# --------------------------------------------------------------------------
async def link_relay(
    admin_message_id: int, business_connection_id: str, customer_chat_id: int,
    suggested_reply: str, original_text: str = "",
) -> None:
    payload = json.dumps({
        "conn": business_connection_id, "chat": customer_chat_id,
        "reply": suggested_reply[:2000], "text": original_text[:2000],
    })
    client = redis_client.get_client()
    if client is None:
        _relay_mem[admin_message_id] = json.loads(payload)
        return
    try:
        await client.setex(_relay_key(admin_message_id), _RELAY_TTL_SECONDS, payload)
    except Exception:  # noqa: BLE001
        logger.exception("Redis link_relay (business) failed")
        _relay_mem[admin_message_id] = json.loads(payload)


async def resolve_relay(admin_message_id: int) -> dict | None:
    client = redis_client.get_client()
    if client is None:
        return _relay_mem.get(admin_message_id)
    try:
        raw = await client.get(_relay_key(admin_message_id))
        return json.loads(raw) if raw else _relay_mem.get(admin_message_id)
    except Exception:  # noqa: BLE001
        logger.exception("Redis resolve_relay (business) failed")
        return _relay_mem.get(admin_message_id)


async def clear_relay(admin_message_id: int) -> None:
    """Call after a relay has been acted on (sent or dismissed) — a real
    message to a real customer must never go out twice from a double-tap or
    a tap-then-reply on the same notification."""
    client = redis_client.get_client()
    if client is None:
        _relay_mem.pop(admin_message_id, None)
        return
    try:
        await client.delete(_relay_key(admin_message_id))
    except Exception:  # noqa: BLE001
        logger.exception("Redis clear_relay (business) failed")
        _relay_mem.pop(admin_message_id, None)


def suggestion_keyboard(admin_message_id: int, is_task: bool = False) -> InlineKeyboardMarkup:
    rows = [[
        InlineKeyboardButton(text="✅ Yuborish", callback_data=f"biz:s:{admin_message_id}"),
        InlineKeyboardButton(text="🚫 E'tiborsiz qoldirish", callback_data=f"biz:i:{admin_message_id}"),
    ]]
    if is_task:
        # Real work (BRD, code, a real quote) needs the specialist team, not
        # a guessed one-line reply — this triggers the full agent pipeline,
        # only after this explicit tap (same permission-gated pattern as
        # everywhere else this bot executes real work).
        rows.insert(0, [
            InlineKeyboardButton(text="🧠 Jamoa bilan ishlab chiqish", callback_data=f"biz:d:{admin_message_id}"),
        ])
    return InlineKeyboardMarkup(inline_keyboard=rows)
