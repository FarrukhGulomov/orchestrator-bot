"""
Daily task assistant — natural-language task/reminder creation, formatted
reminder delivery with action buttons, and (only after an explicit button
click — the user's permission) agent-assisted work on a task via the
existing specialist team.

Split from tasks.py: tasks.py is the pure storage + timing layer; this
module is presentation (Telegram text/keyboards), the LLM extraction step,
and the background reminder loop.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone

from aiogram import Bot
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

import tasks
from agents import AGENTS
from config import settings
from llm_clients import claude_generate_json

logger = logging.getLogger(__name__)

# Cheap pre-filter so natural-language task detection doesn't cost an LLM
# call on every single message — only messages containing one of these get
# classified. The LLM call itself still makes the real is_task decision, so
# this only controls cost, not accuracy.
_TRIGGER_WORDS = (
    "eslat", "vazifa", "topshiriq", "reja qil", "kunlik reja",
    "напомни", "напоминание", "задача", "напоминай",
    "remind", "reminder", "todo", "to-do",
)

_PRIORITY_EMOJI = {"low": "🟢", "medium": "🟡", "high": "🟠", "urgent": "🔴"}
_RECURRENCE_LABEL = {"none": "", "daily": " (har kuni)", "weekdays": " (ish kunlari)", "weekly": " (har hafta)"}


def looks_like_task(text: str) -> bool:
    low = text.lower()
    return any(w in low for w in _TRIGGER_WORDS)


_CLASSIFY_SYSTEM = f"""
You extract a reminder/task from a user's message for a Telegram task assistant.
The user writes in Uzbek, Russian, or English.

Respond with ONLY this JSON, no prose:
{{"is_task": true|false,
  "title": "<short imperative title, same language as user, max 70 chars>",
  "description": "<1-3 sentences of detail, or same as title if nothing extra>",
  "due_in_minutes": <integer minutes from NOW until the deadline/reminder time>,
  "complexity": "low"|"medium"|"high",
  "priority": "low"|"medium"|"high"|"urgent",
  "recurrence": "none"|"daily"|"weekdays"|"weekly",
  "suggested_agent": "<one of: {", ".join(AGENTS.keys())}, or null>"}}

RULES:
- is_task=true only if the user clearly wants something remembered/scheduled/reminded
  ("eslat", "eslatib tur", "vazifa qo'sh", "remind me", "напомни", a task with a
  time, or a recurring routine like "har kuni soat 9da"). Ordinary questions or
  requests for immediate help are NOT tasks — is_task=false for those.
- due_in_minutes: resolve relative time expressions ("bugun soat 15:00", "ertaga",
  "2 soatdan keyin", "har kuni 9:00") against the CURRENT LOCAL TIME given below.
  If NO time is mentioned at all, default to 180 (3 hours from now).
- complexity: estimate how much focused work the task itself needs (low = quick
  5-20 min item, medium = ~1-2 hours, high = 3+ hours / multi-step deliverable).
- priority: how important/urgent it reads as (deadlines, "muhim", "urgent", "zudlik
  bilan" -> high/urgent; casual reminders -> low/medium).
- recurrence: "daily" for "har kuni"/"every day", "weekdays" for "ish kunlari",
  "weekly" for "har hafta", else "none".
- suggested_agent: which specialist would help EXECUTE this task if the user later
  asks for help (e.g. "SQL hisobot tayyorlash" -> data_analyst). null if it's a
  purely personal reminder with no specialist angle.
"""


async def classify_task(user_text: str) -> dict | None:
    now = tasks.now_local()
    context = (
        f"CURRENT LOCAL TIME ({settings.timezone}): {now.strftime('%Y-%m-%d %H:%M')} "
        f"({now.strftime('%A')})\n\nMessage: {user_text}"
    )
    try:
        raw = await asyncio.wait_for(
            claude_generate_json(_CLASSIFY_SYSTEM, [{"role": "user", "content": context}], max_tokens=400),
            timeout=settings.request_timeout,
        )
        data = json.loads(raw)
    except Exception:  # noqa: BLE001
        logger.exception("Task classification failed")
        return None
    if not isinstance(data, dict) or not data.get("is_task"):
        return None
    return data


def _due_from_minutes(minutes: object) -> datetime:
    try:
        m = int(minutes)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        m = 180
    m = max(1, min(m, 60 * 24 * 90))  # clamp: 1 minute .. 90 days out
    return datetime.now(timezone.utc) + timedelta(minutes=m)


async def build_task_from_text(chat_id: int, user_id: int, user_text: str) -> tasks.Task | None:
    data = await classify_task(user_text)
    if data is None:
        return None
    due_at = _due_from_minutes(data.get("due_in_minutes"))
    agent_key = data.get("suggested_agent")
    if agent_key not in AGENTS:
        agent_key = None
    return await tasks.create_task(
        chat_id=chat_id,
        user_id=user_id,
        title=str(data.get("title") or user_text)[:200],
        description=str(data.get("description") or "")[:2000],
        due_at=due_at,
        complexity=str(data.get("complexity") or "medium"),
        priority=str(data.get("priority") or "medium"),
        recurrence=str(data.get("recurrence") or "none"),
        agent_key=agent_key,
    )


def format_confirmation(task: tasks.Task) -> str:
    due_local = datetime.fromisoformat(task.due_at).astimezone(tasks.TZ)
    remind_local = datetime.fromisoformat(task.remind_at).astimezone(tasks.TZ)
    lines = [
        f"✅ Vazifa qo'shildi{_RECURRENCE_LABEL.get(task.recurrence, '')}",
        f"📌 {task.title}",
    ]
    if task.description and task.description != task.title:
        lines.append(task.description)
    lines.append(f"🕒 Muddat: {due_local.strftime('%d-%m %H:%M')}")
    emoji = _PRIORITY_EMOJI.get(task.priority, "🟡")
    lines.append(f"🔔 Eslatma: {remind_local.strftime('%d-%m %H:%M')} dan boshlab ({emoji} {task.priority})")
    return "\n".join(lines)


def format_reminder(task: tasks.Task, is_final: bool) -> str:
    due_local = datetime.fromisoformat(task.due_at).astimezone(tasks.TZ)
    emoji = _PRIORITY_EMOJI.get(task.priority, "🟡")
    header = "⏰ OXIRGI ESLATMA" if is_final else "⏰ Vazifa vaqti keldi"
    lines = [f"{header} {emoji}", f"📌 {task.title}"]
    if task.description and task.description != task.title:
        lines.append(task.description)
    lines.append(f"🕒 Muddat: {due_local.strftime('%d-%m %H:%M')}")
    return "\n".join(lines)


def format_task_line(task: tasks.Task) -> str:
    due_local = datetime.fromisoformat(task.due_at).astimezone(tasks.TZ)
    emoji = _PRIORITY_EMOJI.get(task.priority, "🟡")
    rec = _RECURRENCE_LABEL.get(task.recurrence, "")
    return f"{emoji} `{task.id}` — {task.title} — {due_local.strftime('%d-%m %H:%M')}{rec}"


def confirmation_keyboard(task_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="🗑 Bekor qilish", callback_data=f"tsk:x:{task_id}"),
    ]])


def reminder_keyboard(task_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✅ Bajarildi", callback_data=f"tsk:d:{task_id}"),
            InlineKeyboardButton(text="⏰ +30 daq", callback_data=f"tsk:s:{task_id}"),
        ],
        [
            InlineKeyboardButton(text="🤖 Yordam so'rash", callback_data=f"tsk:w:{task_id}"),
            InlineKeyboardButton(text="❌ Bekor qilish", callback_data=f"tsk:x:{task_id}"),
        ],
    ])


# --------------------------------------------------------------------------
# Background reminder loop
# --------------------------------------------------------------------------
async def reminder_loop(bot: Bot) -> None:
    """Polls due reminders and sends them. Started once from main() alongside
    polling; runs for the lifetime of the process. Never raises — a single
    bad iteration is logged and the loop continues."""
    logger.info("Reminder loop started (poll every %ds).", settings.reminder_poll_seconds)
    while True:
        try:
            due = await tasks.pop_due()
            for task in due:
                await _fire(bot, task)
        except Exception:  # noqa: BLE001
            logger.exception("Reminder loop iteration failed (non-fatal, continuing)")
        await asyncio.sleep(settings.reminder_poll_seconds)


def _is_allowed(chat_id: int) -> bool:
    return not settings.allowed_chat_ids or chat_id in settings.allowed_chat_ids


async def _fire(bot: Bot, task: tasks.Task) -> None:
    if not _is_allowed(task.chat_id):
        # Chat was removed from ALLOWED_CHAT_IDS after this task was scheduled
        # (tasks can outlive a redeploy via Redis) — drop the reminder rather
        # than deliver to a chat that's no longer authorised.
        await tasks.advance_after_fire(task)
        return
    is_final = task.stage == "final"
    try:
        await bot.send_message(
            task.chat_id,
            format_reminder(task, is_final),
            reply_markup=reminder_keyboard(task.id),
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to send reminder for task=%s", task.id)
    await tasks.advance_after_fire(task)
