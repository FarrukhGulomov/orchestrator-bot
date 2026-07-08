"""
Daily rhythm for the PM/BA: morning digest, standup draft, weekly review.

Morning digest — an opt-in per-chat daily message ("/digest on", default
08:30 local) listing overdue tasks, today's tasks, and the next 7 days.
The loop ticks once a minute and sends when the chat's configured HH:MM has
passed and nothing was sent yet today — so a bot restarted/redeployed at
09:00 still delivers the 08:30 digest instead of skipping the day.

Standup (/standup) and weekly review (/week) are built deterministically
from the task store — no LLM call, so they're instant, free, and never
hallucinate work that didn't happen.

Config storage: Redis hash per chat + a registry set of enabled chats,
in-memory fallback like every other store in this codebase.
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from aiogram import Bot

import redis_client
import tasks
import task_assistant

logger = logging.getLogger(__name__)

DEFAULT_TIME = "08:30"
_TIME_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")

_REGISTRY = "digest:chats"


@dataclass
class DigestConfig:
    enabled: bool = False
    time: str = DEFAULT_TIME   # local HH:MM, zero-padded
    last_sent: str = ""        # local YYYY-MM-DD of the last delivered digest


# --- In-memory fallback --------------------------------------------------
_cfg_mem: dict[int, DigestConfig] = {}


def normalize_time(raw: str) -> str | None:
    """Validate 'H:MM'/'HH:MM' and return zero-padded 'HH:MM', or None."""
    m = _TIME_RE.match(raw.strip())
    if not m:
        return None
    return f"{int(m.group(1)):02d}:{m.group(2)}"


def _cfg_key(chat_id: int) -> str:
    return f"digest:cfg:{chat_id}"


async def get_config(chat_id: int) -> DigestConfig:
    client = redis_client.get_client()
    if client is None:
        return _cfg_mem.get(chat_id, DigestConfig())
    try:
        data = await client.hgetall(_cfg_key(chat_id))
        if not data:
            return DigestConfig()
        return DigestConfig(
            enabled=data.get("enabled") == "1",
            time=data.get("time") or DEFAULT_TIME,
            last_sent=data.get("last_sent") or "",
        )
    except Exception:  # noqa: BLE001
        logger.exception("Redis digest get_config failed")
        return _cfg_mem.get(chat_id, DigestConfig())


async def set_config(chat_id: int, cfg: DigestConfig) -> None:
    client = redis_client.get_client()
    if client is None:
        _cfg_mem[chat_id] = cfg
        return
    try:
        async with client.pipeline(transaction=True) as pipe:
            pipe.hset(_cfg_key(chat_id), mapping={
                "enabled": "1" if cfg.enabled else "0",
                "time": cfg.time,
                "last_sent": cfg.last_sent,
            })
            if cfg.enabled:
                pipe.sadd(_REGISTRY, str(chat_id))
            else:
                pipe.srem(_REGISTRY, str(chat_id))
            await pipe.execute()
    except Exception:  # noqa: BLE001
        logger.exception("Redis digest set_config failed, using in-memory")
        _cfg_mem[chat_id] = cfg


async def _enabled_chats() -> list[int]:
    client = redis_client.get_client()
    if client is None:
        return [cid for cid, c in _cfg_mem.items() if c.enabled]
    try:
        raw = await client.smembers(_REGISTRY)
        return [int(x) for x in raw]
    except Exception:  # noqa: BLE001
        logger.exception("Redis digest registry read failed")
        return [cid for cid, c in _cfg_mem.items() if c.enabled]


# --------------------------------------------------------------------------
# Content builders
# --------------------------------------------------------------------------
def _local_day_bounds(offset_days: int = 0) -> tuple[datetime, datetime]:
    """(start, end) of the local calendar day `offset_days` from today, in UTC."""
    day = tasks.now_local().date() + timedelta(days=offset_days)
    start_local = datetime.combine(day, datetime.min.time(), tzinfo=tasks.TZ)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def _fmt(t: tasks.Task, with_date: bool = False) -> str:
    due_local = datetime.fromisoformat(t.due_at).astimezone(tasks.TZ)
    stamp = due_local.strftime("%d-%m %H:%M") if with_date else due_local.strftime("%H:%M")
    emoji = task_assistant.PRIORITY_EMOJI.get(t.priority, "🟡")
    return f"{emoji} {stamp} — {t.title}"


async def build_digest(chat_id: int) -> str:
    today_start, today_end = _local_day_bounds(0)
    week_end = today_end + timedelta(days=7)

    # Fetch each status-bucket ONCE and filter locally — overdue/today/
    # upcoming are all views over the same pending set, and re-fetching per
    # view would be 3 redundant Redis round-trips per digest sent.
    pending = await tasks.list_tasks(chat_id, {"pending"})
    done = await tasks.list_tasks(chat_id, {"done"})

    overdue = tasks.overdue(pending)
    today = tasks.in_window(pending, today_start, today_end)
    # "Bugun" and "kechikkan" overlap for tasks due earlier today — show those
    # only in the overdue section so nothing appears twice.
    today = [t for t in today if t not in overdue]
    upcoming = tasks.in_window(pending, today_end, week_end)
    done_yesterday = tasks.completed_since_list(done, _local_day_bounds(-1)[0])
    done_yesterday = [
        t for t in done_yesterday
        if datetime.fromisoformat(t.completed_at) < today_start
    ]

    now_local = tasks.now_local()
    lines = [f"☀️ Kunlik reja — {now_local.strftime('%d-%m-%Y, %A')}"]

    if overdue:
        lines.append(f"\n⚠️ Kechikkan ({len(overdue)}):")
        lines.extend(_fmt(t, with_date=True) for t in overdue[:8])
    if today:
        lines.append("\n📅 Bugun:")
        lines.extend(_fmt(t) for t in today[:10])
    if upcoming:
        lines.append("\n🔜 Yaqin 7 kun:")
        lines.extend(_fmt(t, with_date=True) for t in upcoming[:5])
    if done_yesterday:
        lines.append(f"\n✅ Kecha bajarildi: {len(done_yesterday)} ta")

    if not (overdue or today or upcoming):
        lines.append("\nBugunga rejalashtirilgan vazifa yo'q. /addtask bilan qo'shishingiz mumkin.")
    else:
        lines.append("\n📋 Boshqarish: /tasks · Standup: /standup")
    return "\n".join(lines)


async def build_standup(chat_id: int) -> str:
    """Yesterday / today / blockers draft — paste-ready for the team standup."""
    today_start, today_end = _local_day_bounds(0)
    yesterday_start = _local_day_bounds(-1)[0]

    pending = await tasks.list_tasks(chat_id, {"pending"})
    done = tasks.completed_since_list(await tasks.list_tasks(chat_id, {"done"}), yesterday_start)
    today = tasks.in_window(pending, today_start, today_end)
    overdue = tasks.overdue(pending)
    today = [t for t in today if t not in overdue]

    lines = ["🗣 Standup draft:\n"]
    lines.append("Kecha:")
    if done:
        lines.extend(f"• {t.title} ✅" for t in done[:10])
    else:
        lines.append("• (bajarilgan vazifa qayd etilmagan)")
    lines.append("\nBugun:")
    if today:
        lines.extend(f"• {t.title}" for t in today[:10])
    else:
        lines.append("• (bugunga rejalashtirilgan vazifa yo'q)")
    lines.append("\nBlockers / kechikkan:")
    if overdue:
        lines.extend(f"• {t.title} (muddat: {datetime.fromisoformat(t.due_at).astimezone(tasks.TZ).strftime('%d-%m %H:%M')})" for t in overdue[:8])
    else:
        lines.append("• yo'q")
    return "\n".join(lines)


async def build_week(chat_id: int) -> str:
    """Last-7-days review: throughput + what's still open."""
    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)

    pending = await tasks.list_tasks(chat_id, {"pending"})
    done = tasks.completed_since_list(await tasks.list_tasks(chat_id, {"done"}), week_ago)
    overdue = tasks.overdue(pending)
    created = [
        t for t in (pending + done)
        if t.created_at and datetime.fromisoformat(t.created_at) >= week_ago
    ]

    lines = ["📊 Haftalik hisobot (oxirgi 7 kun):\n"]
    lines.append(f"• Qo'shilgan: {len(created)} ta")
    lines.append(f"• Bajarilgan: {len(done)} ta")
    lines.append(f"• Ochiq qolgan: {len(pending)} ta (shundan kechikkan: {len(overdue)})")
    if done:
        lines.append("\n✅ Bajarilganlar:")
        lines.extend(f"• {t.title}" for t in done[:15])
    if overdue:
        lines.append("\n⚠️ Kechikkanlar:")
        lines.extend(task_assistant.format_task_line(t) for t in overdue[:8])
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Background loop
# --------------------------------------------------------------------------
async def digest_loop(bot: Bot) -> None:
    """Delivers the morning digest. Started once from main(); never raises."""
    logger.info("Digest loop started (tick every 60s).")
    while True:
        try:
            await _tick(bot)
        except Exception:  # noqa: BLE001
            logger.exception("Digest loop tick failed (non-fatal, continuing)")
        await asyncio.sleep(60)


async def _tick(bot: Bot) -> None:
    now_local = tasks.now_local()
    hhmm = now_local.strftime("%H:%M")
    today = now_local.strftime("%Y-%m-%d")

    for chat_id in await _enabled_chats():
        if not task_assistant.is_allowed(chat_id):
            continue
        cfg = await get_config(chat_id)
        if not cfg.enabled or cfg.last_sent == today or cfg.time > hhmm:
            continue
        try:
            await bot.send_message(chat_id, await build_digest(chat_id))
        except Exception:  # noqa: BLE001
            logger.exception("Digest send failed for chat=%s", chat_id)
            continue
        cfg.last_sent = today
        await set_config(chat_id, cfg)
