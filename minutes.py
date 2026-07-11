"""
Meeting minutes — the PM/BA's biggest daily time sink, automated.

/minutes <raw notes or transcript> (or reply to a message/voice transcript
with /minutes) → structured protocol: summary, decisions, action items with
owners and deadlines. Two explicit buttons then let the user — with their
permission, never automatically — turn action items into tracked reminder
tasks and append the decisions to the decision log.

The extracted batch is stashed (Redis, 1h TTL; in-memory fallback) under a
short id so the button callback_data stays within Telegram's 64-byte limit.
"""

import json
import logging
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta, timezone

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

import redis_client
import tasks
from config import settings
from llm_clients import generate_json
from router import model_for
from agents import get_agent

logger = logging.getLogger(__name__)

_BATCH_TTL_SECONDS = 3600
_MAX_INPUT_CHARS = 16000

_EXTRACT_SYSTEM = """
You turn raw meeting notes / a transcript into structured minutes for a
Senior Business Analyst. Input may be Uzbek (Latin or Cyrillic), Russian,
English, or mixed. Write ALL output text in the DOMINANT language of the
input (keep BA/IT terms in English).

Respond with ONLY this JSON, no prose, no markdown fences:
{"title": "<meeting topic, max 70 chars>",
 "participants": ["<name>", ...],
 "summary": "<3-6 sentences: what was discussed and concluded>",
 "decisions": ["<each firm decision made, one string each>", ...],
 "action_items": [
   {"title": "<imperative action, max 80 chars>",
    "owner": "<person responsible, or null>",
    "due_in_minutes": <integer minutes from NOW, or null if no deadline mentioned>,
    "priority": "low"|"medium"|"high"|"urgent"}, ...]}

RULES:
- decisions = things AGREED/DECIDED, not topics merely discussed.
- action_items = concrete follow-up work someone must do. Do not invent
  items that aren't in the notes. Empty lists are fine.
- due_in_minutes: resolve "ertaga", "juma kuni", "keyingi haftagacha" etc.
  against the CURRENT LOCAL TIME provided; null when no deadline is stated.
- participants: only names actually present in the notes.
"""


async def extract_minutes(raw_text: str) -> tuple[dict | None, str | None]:
    """Returns (data, error_message) — the real underlying error is surfaced
    to the admin's Telegram message on failure (same pattern as the
    copilots' analyze()), never just swallowed into server logs."""
    now = tasks.now_local()
    context = (
        f"CURRENT LOCAL TIME ({settings.timezone}): {now.strftime('%Y-%m-%d %H:%M')} "
        f"({now.strftime('%A')})\n\n--- MEETING NOTES ---\n{raw_text[:_MAX_INPUT_CHARS]}"
    )
    messages = [{"role": "user", "content": context}]
    # Meeting notes deserve the MAIN model (same reasoning as document
    # generation): the routing-tier default would truncate/garble long inputs.
    model, _label = model_for(get_agent("ba"), "high")
    try:
        data = await generate_json(_EXTRACT_SYSTEM, messages, model=model, max_tokens=4096)
    except Exception as exc:  # noqa: BLE001
        # In hybrid mode "high" is a Claude id — if THAT call dies (credits
        # exhausted, auth, outage), a protocol is still far more useful from
        # the free main model than no protocol at all. Degrade, don't fail.
        logger.warning("Minutes extraction failed on %s (%s) — falling back to %s",
                       model, str(exc)[:120], settings.or_main_model)
        try:
            data = await generate_json(
                _EXTRACT_SYSTEM, messages, model=settings.or_main_model, max_tokens=4096, retries=1,
            )
        except Exception as exc2:  # noqa: BLE001
            logger.exception("Minutes extraction failed on fallback model too")
            return None, f"{str(exc)[:150]} / fallback: {str(exc2)[:150]}"
    if not str(data.get("summary", "")).strip():
        return None, "Model protokol matnini ajratib bera olmadi (summary bo'sh)."
    return _sanitize(data), None


def _sanitize(data: dict) -> dict:
    """Coerce the LLM's JSON to the exact shape render_minutes/
    create_tasks_from_items expect. Models occasionally deviate from a
    requested schema (e.g. returning action_items as a flat list of strings,
    mirroring the neighboring decisions field which genuinely is a string
    list) — without this, that deviation crashes downstream .get() calls."""
    data["participants"] = [p for p in data.get("participants") or [] if isinstance(p, str)]
    data["decisions"] = [d for d in data.get("decisions") or [] if isinstance(d, str)]
    items = data.get("action_items") or []
    data["action_items"] = [it for it in items if isinstance(it, dict) and str(it.get("title", "")).strip()]
    return data


def render_minutes(data: dict) -> str:
    lines = [f"📝 Protokol: {data.get('title') or 'Uchrashuv'}"]
    participants = [str(p) for p in data.get("participants") or [] if str(p).strip()]
    if participants:
        lines.append(f"👥 Ishtirokchilar: {', '.join(participants[:12])}")
    lines.append(f"\n{data.get('summary', '')}")

    decisions = [str(d) for d in data.get("decisions") or [] if str(d).strip()]
    if decisions:
        lines.append("\n✅ Qarorlar:")
        lines.extend(f"{i}. {d}" for i, d in enumerate(decisions[:15], 1))

    items = data.get("action_items") or []
    if items:
        lines.append("\n📌 Action items:")
        for i, it in enumerate(items[:15], 1):
            owner = f" — {it['owner']}" if it.get("owner") else ""
            due = ""
            if it.get("due_in_minutes"):
                try:
                    due_local = (
                        datetime.now(timezone.utc) + timedelta(minutes=int(it["due_in_minutes"]))
                    ).astimezone(tasks.TZ)
                    due = f" (muddat: {due_local.strftime('%d-%m %H:%M')})"
                except (TypeError, ValueError):
                    pass
            lines.append(f"{i}. {it.get('title', '?')}{owner}{due}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Batch stash for the permission buttons
# --------------------------------------------------------------------------
_MAX_MEM_BATCHES = 200
_batch_mem: "OrderedDict[str, str]" = OrderedDict()


def _batch_key(batch_id: str) -> str:
    return f"minutes:batch:{batch_id}"


def _mem_put(batch_id: str, payload: str) -> None:
    # LRU-ish bound: evict only the OLDEST entries, never wipe everything —
    # Redis gives each batch its own 1h TTL, so the in-memory fallback must
    # not destroy a batch a user is about to act on just because unrelated
    # chats created a lot of other batches in the meantime.
    _batch_mem[batch_id] = payload
    _batch_mem.move_to_end(batch_id)
    while len(_batch_mem) > _MAX_MEM_BATCHES:
        _batch_mem.popitem(last=False)


async def stash_batch(chat_id: int, data: dict) -> str:
    batch_id = uuid.uuid4().hex[:10]
    payload = json.dumps({"chat_id": chat_id, "data": data})
    client = redis_client.get_client()
    if client is None:
        _mem_put(batch_id, payload)
        return batch_id
    try:
        await client.setex(_batch_key(batch_id), _BATCH_TTL_SECONDS, payload)
    except Exception:  # noqa: BLE001
        logger.exception("Redis stash_batch failed, using in-memory")
        _mem_put(batch_id, payload)
    return batch_id


async def resave_batch(batch_id: str, chat_id: int, data: dict) -> None:
    """Overwrite a stashed batch (used to persist the '_tasks_added' /
    '_decisions_saved' flags so a double-tapped button can't duplicate)."""
    payload = json.dumps({"chat_id": chat_id, "data": data})
    client = redis_client.get_client()
    if client is None:
        _mem_put(batch_id, payload)
        return
    try:
        await client.setex(_batch_key(batch_id), _BATCH_TTL_SECONDS, payload)
    except Exception:  # noqa: BLE001
        logger.exception("Redis resave_batch failed, using in-memory")
        _mem_put(batch_id, payload)


async def load_batch(batch_id: str) -> tuple[int, dict] | None:
    client = redis_client.get_client()
    payload = None
    if client is not None:
        try:
            payload = await client.get(_batch_key(batch_id))
        except Exception:  # noqa: BLE001
            logger.exception("Redis load_batch failed")
    if payload is None:
        payload = _batch_mem.get(batch_id)
    if not payload:
        return None
    try:
        obj = json.loads(payload)
        return int(obj["chat_id"]), obj["data"]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def minutes_keyboard(batch_id: str, n_items: int, n_decisions: int) -> InlineKeyboardMarkup | None:
    row = []
    if n_items:
        row.append(InlineKeyboardButton(
            text=f"➕ Vazifalarga qo'shish ({n_items})", callback_data=f"min:t:{batch_id}"
        ))
    if n_decisions:
        row.append(InlineKeyboardButton(
            text=f"💾 Qarorlarni saqlash ({n_decisions})", callback_data=f"min:d:{batch_id}"
        ))
    return InlineKeyboardMarkup(inline_keyboard=[row]) if row else None


async def create_tasks_from_items(chat_id: int, user_id: int, data: dict) -> list[tasks.Task]:
    """Turn a batch's action items into real reminder tasks. Runs only from
    the explicit '➕ Vazifalarga qo'shish' button — never automatically."""
    created: list[tasks.Task] = []
    for it in (data.get("action_items") or [])[:15]:
        title = str(it.get("title") or "").strip()
        if not title:
            continue
        minutes_out = it.get("due_in_minutes")
        try:
            m = max(1, min(int(minutes_out), 60 * 24 * 90)) if minutes_out else 24 * 60
        except (TypeError, ValueError):
            m = 24 * 60
        owner = str(it.get("owner") or "").strip()
        description = f"Mas'ul: {owner}" if owner else ""
        priority = str(it.get("priority") or "medium")
        task = await tasks.create_task(
            chat_id=chat_id,
            user_id=user_id,
            title=title,
            description=description,
            due_at=datetime.now(timezone.utc) + timedelta(minutes=m),
            complexity="medium",
            priority=priority,
        )
        created.append(task)
    return created
