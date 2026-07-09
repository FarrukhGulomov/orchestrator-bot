"""
Group mention copilot — different from this bot's regular group Q&A
feature (PROACTIVE_IN_GROUPS / REQUIRE_MENTION_IN_GROUPS, which answer the
GROUP directly when the BOT itself is addressed or the topic looks
relevant). This module watches for messages addressed to the ADMIN
SPECIFICALLY — someone @mentions them, or replies to a message the admin
themselves sent — in any group the bot is a plain member of (no admin
rights on the group needed), and privately notifies the admin with an AI
analysis + a ready suggested reply. The admin never has to actively watch
the group; they act from their own DM with the bot.

REQUIRES the bot's Telegram Privacy Mode to be DISABLED (BotFather ->
/setprivacy -> Disable) for groups it should watch — otherwise Telegram
only forwards messages that @mention/reply to the BOT itself, and a
message that merely mentions the ADMIN never reaches this bot at all.
See config.py's watch_group_mentions and .env.example.

Mirrors business_copilot.py's analyze + relay + suggestion-keyboard
design (same non-committal "is_task" holding reply for real work, same
Redis-backed relay with in-memory fallback) but the target is a GROUP
message (reply_to_message_id) instead of a Business chat
(business_connection_id).
"""

import asyncio
import json
import logging

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

import redis_client
from agents import get_agent
from config import settings
from llm_clients import claude_generate_json
from router import model_for

logger = logging.getLogger(__name__)

_RELAY_TTL_SECONDS = 60 * 60 * 24 * 7

# --- in-memory fallback ---
_relay_mem: dict[int, dict] = {}


def _relay_key(admin_message_id: int) -> str:
    return f"grp:relay:{admin_message_id}"


_COPILOT_SYSTEM = """
You are helping a busy professional handle messages addressed to them
personally inside a Telegram GROUP — someone @mentioned them, or replied
to a message they themselves sent earlier — since they can't watch every
group in real time. Respond with ONLY this JSON, no prose, no markdown
fences:
{"analysis": "<one short line: intent/tone/urgency, in Uzbek — this is for
  the professional's own eyes>",
 "is_task": true|false,
 "suggested_reply": "<ready-to-send reply — language rule below>"}

LANGUAGE RULE FOR suggested_reply — critical, follow strictly:
- Detect the sender's language from the GRAMMATICAL STRUCTURE of their
  message, not isolated words or script alone (Cyrillic serves both
  Uzbek and Russian).
- Russian grammar/connectors (e.g. "как", "у вас", "для", "мне", "нужно",
  "сделай") -> reply in RUSSIAN.
- Uzbek grammar/connectors (e.g. "qiling", "kerak", "menga", "bo'ladi",
  "-da", "-ni") -> reply in UZBEK.
- Purely English grammar -> reply in ENGLISH.
- Match the SENDER's latest message, never default to a fixed language.

is_task=true when the sender is asking for actual work to be DONE — a
decision, a document, a deadline commitment, technical work — not just a
quick question or small talk.
When is_task=true, suggested_reply is an HONEST holding reply only (e.g.
Uzbek: "ko'rib chiqib tez orada javob beraman"; Russian: "посмотрю и скоро
отвечу") — NEVER invent a decision, timeline, or commitment on the
professional's behalf.
When is_task=false, suggested_reply is the real, ready-to-send answer.

WRITE LIKE A HUMAN, NOT A BOT: short like real texting, react to the
specific thing said (not a generic acknowledgment), mirror the sender's
register (casual/formal), no stock customer-service phrasing, vary
phrasing across different replies — the group must never sense this was
AI-drafted if the admin sends it as-is.
"""


async def analyze(group_name: str, sender_name: str, quoted_text: str, latest_text: str) -> dict | None:
    context_lines = [f"Group: {group_name}"]
    if quoted_text:
        context_lines.append(f'This replies to a message you (the professional) sent earlier: "{quoted_text[:500]}"')
    context_lines.append(f"{sender_name}'s message addressed to you: {latest_text}")
    context = "\n".join(context_lines)

    # Fires only when the admin is actually addressed (not on every group
    # message), but still cheap/free-model routed like business_copilot's
    # analyze() — no reason to spend paid Claude quota on a quick triage.
    model, _label = model_for(get_agent("ba"), "low")
    try:
        raw = await asyncio.wait_for(
            claude_generate_json(_COPILOT_SYSTEM, [{"role": "user", "content": context}], model=model, max_tokens=400),
            timeout=settings.request_timeout,
        )
        data = json.loads(raw)
    except Exception:  # noqa: BLE001
        logger.exception("Group copilot analysis failed")
        return None
    if not isinstance(data, dict) or not str(data.get("suggested_reply", "")).strip():
        return None
    return data


# --------------------------------------------------------------------------
# Relay: admin-side notification message_id -> where/what to send if approved
# --------------------------------------------------------------------------
async def link_relay(
    admin_message_id: int, group_chat_id: int, reply_to_message_id: int,
    suggested_reply: str, original_text: str = "",
) -> None:
    payload = json.dumps({
        "chat": group_chat_id, "reply_to": reply_to_message_id,
        "reply": suggested_reply[:2000], "text": original_text[:2000],
    })
    client = redis_client.get_client()
    if client is None:
        _relay_mem[admin_message_id] = json.loads(payload)
        return
    try:
        await client.setex(_relay_key(admin_message_id), _RELAY_TTL_SECONDS, payload)
    except Exception:  # noqa: BLE001
        logger.exception("Redis link_relay (group) failed")
        _relay_mem[admin_message_id] = json.loads(payload)


async def resolve_relay(admin_message_id: int) -> dict | None:
    client = redis_client.get_client()
    if client is None:
        return _relay_mem.get(admin_message_id)
    try:
        raw = await client.get(_relay_key(admin_message_id))
        return json.loads(raw) if raw else _relay_mem.get(admin_message_id)
    except Exception:  # noqa: BLE001
        logger.exception("Redis resolve_relay (group) failed")
        return _relay_mem.get(admin_message_id)


async def clear_relay(admin_message_id: int) -> None:
    """Call after a relay has been acted on (sent or dismissed) — a real
    reply into a real group must never go out twice from a double-tap or a
    tap-then-reply on the same notification."""
    client = redis_client.get_client()
    if client is None:
        _relay_mem.pop(admin_message_id, None)
        return
    try:
        await client.delete(_relay_key(admin_message_id))
    except Exception:  # noqa: BLE001
        logger.exception("Redis clear_relay (group) failed")
        _relay_mem.pop(admin_message_id, None)


def suggestion_keyboard(admin_message_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ Guruhga yuborish", callback_data=f"grp:s:{admin_message_id}"),
        InlineKeyboardButton(text="🚫 E'tiborsiz qoldirish", callback_data=f"grp:i:{admin_message_id}"),
    ]])
