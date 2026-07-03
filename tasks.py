"""
Daily task / reminder storage + the smart reminder-timing algorithm.

Pure data layer — no Telegram, no LLM calls (that's task_assistant.py).
Same Redis-backed / in-memory-fallback pattern as history.py and memory.py.

REMINDER TIMING ALGORITHM (the "kuchli algoritm" part):
Every task gets a PRIMARY reminder — not fired at the deadline, but early
enough to actually START and FINISH the estimated work, scaled by how
complex the task is (low/medium/high -> ~20/90/240 minutes of raw effort,
plus a 30% safety buffer). High/urgent-priority tasks additionally get a
FINAL nudge shortly before the deadline as a last call. Recurring tasks
(daily/weekdays/weekly) reschedule themselves after each firing.
"""

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import redis_client
from config import settings

logger = logging.getLogger(__name__)

try:
    TZ = ZoneInfo(settings.timezone)
except Exception:  # noqa: BLE001 — bad/missing tz data must not crash startup
    logger.warning("Invalid TIMEZONE=%r, falling back to Asia/Tashkent", settings.timezone)
    TZ = ZoneInfo("Asia/Tashkent")

STATUSES = {"pending", "done", "cancelled"}
RECURRENCES = {"none", "daily", "weekdays", "weekly"}
PRIORITIES = {"low", "medium", "high", "urgent"}
COMPLEXITIES = {"low", "medium", "high"}

# Estimated raw effort (minutes) by complexity — drives how much lead time
# the PRIMARY ("start working now") reminder gets before the deadline.
_EFFORT_MINUTES = {"low": 20, "medium": 90, "high": 240}
_LEAD_FACTOR = 1.3            # safety buffer on top of raw estimated effort
_LEAD_MIN_MINUTES = 15
_LEAD_MAX_MINUTES = 24 * 60   # never front-load a reminder more than a day out
_FINAL_NUDGE_MINUTES = 20     # last-call reminder before the deadline

_DUE_ZSET = "tasks:due"


@dataclass
class Task:
    id: str
    chat_id: int
    user_id: int
    title: str
    description: str
    due_at: str                    # ISO8601 UTC
    remind_at: str                 # ISO8601 UTC — primary "start now" reminder
    final_remind_at: str | None    # ISO8601 UTC — last-call reminder, or None
    complexity: str                # low|medium|high
    priority: str                  # low|medium|high|urgent
    recurrence: str                # none|daily|weekdays|weekly
    agent_key: str | None
    status: str = "pending"        # pending|done|cancelled
    stage: str = "primary"         # primary|final — which reminder fires next
    created_at: str = ""


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_local() -> datetime:
    return datetime.now(TZ)


def compute_reminders(
    due_at: datetime, complexity: str, priority: str, now: datetime | None = None
) -> tuple[datetime, datetime | None]:
    """Return (primary_reminder, final_reminder_or_None) for a task due at `due_at`."""
    now = now or _now_utc()
    effort = _EFFORT_MINUTES.get(complexity, 45)
    lead = max(_LEAD_MIN_MINUTES, min(int(effort * _LEAD_FACTOR), _LEAD_MAX_MINUTES))
    primary = due_at - timedelta(minutes=lead)
    if primary <= now:
        primary = now + timedelta(minutes=1)

    final = None
    if priority in ("high", "urgent"):
        candidate = due_at - timedelta(minutes=_FINAL_NUDGE_MINUTES)
        if candidate > primary + timedelta(minutes=10) and candidate > now:
            final = candidate
    return primary, final


def next_occurrence(due_at: datetime, recurrence: str) -> datetime:
    if recurrence == "daily":
        return due_at + timedelta(days=1)
    if recurrence == "weekly":
        return due_at + timedelta(weeks=1)
    if recurrence == "weekdays":
        nxt = due_at + timedelta(days=1)
        # Skip Sat/Sun by the USER'S local calendar date, not UTC — due_at is
        # stored in UTC, and for a fixed UTC+5 offset with no DST, any local
        # due time before 05:00 lands on the PREVIOUS UTC calendar date, so
        # nxt.weekday() alone would check the wrong day.
        while nxt.astimezone(TZ).weekday() >= 5:  # Sat=5, Sun=6
            nxt += timedelta(days=1)
        return nxt
    return due_at


# --------------------------------------------------------------------------
# In-memory fallback
# --------------------------------------------------------------------------
_store: dict[str, Task] = {}
_due_mem: dict[str, float] = {}
_index_mem: dict[int, set[str]] = {}


def _task_key(task_id: str) -> str:
    return f"task:{task_id}"


def _index_key(chat_id: int) -> str:
    return f"tasks:index:{chat_id}"


def _to_json(task: Task) -> str:
    return json.dumps(asdict(task))


def _from_json(raw: str) -> Task:
    return Task(**json.loads(raw))


async def _save(task: Task) -> None:
    client = redis_client.get_client()
    if client is None:
        _store[task.id] = task
        _index_mem.setdefault(task.chat_id, set()).add(task.id)
        return
    try:
        async with client.pipeline(transaction=True) as pipe:
            pipe.set(_task_key(task.id), _to_json(task))
            pipe.sadd(_index_key(task.chat_id), task.id)
            await pipe.execute()
    except Exception:  # noqa: BLE001
        logger.exception("Redis task save failed, falling back to in-memory")
        _store[task.id] = task
        _index_mem.setdefault(task.chat_id, set()).add(task.id)


async def _schedule(task_id: str, when: datetime) -> None:
    client = redis_client.get_client()
    score = when.timestamp()
    if client is None:
        _due_mem[task_id] = score
        return
    try:
        await client.zadd(_DUE_ZSET, {task_id: score})
    except Exception:  # noqa: BLE001
        logger.exception("Redis task schedule failed, falling back to in-memory")
        _due_mem[task_id] = score


async def _unschedule(task_id: str) -> None:
    client = redis_client.get_client()
    if client is None:
        _due_mem.pop(task_id, None)
        return
    try:
        await client.zrem(_DUE_ZSET, task_id)
    except Exception:  # noqa: BLE001
        logger.exception("Redis task unschedule failed")


async def create_task(
    chat_id: int,
    user_id: int,
    title: str,
    description: str,
    due_at: datetime,
    complexity: str,
    priority: str,
    recurrence: str = "none",
    agent_key: str | None = None,
) -> Task:
    complexity = complexity if complexity in COMPLEXITIES else "medium"
    priority = priority if priority in PRIORITIES else "medium"
    recurrence = recurrence if recurrence in RECURRENCES else "none"
    primary, final = compute_reminders(due_at, complexity, priority)

    task = Task(
        id=uuid.uuid4().hex[:10],
        chat_id=chat_id,
        user_id=user_id,
        title=title[:200],
        description=description[:2000],
        due_at=due_at.isoformat(),
        remind_at=primary.isoformat(),
        final_remind_at=final.isoformat() if final else None,
        complexity=complexity,
        priority=priority,
        recurrence=recurrence,
        agent_key=agent_key,
        created_at=_now_utc().isoformat(),
    )
    await _save(task)
    await _schedule(task.id, primary)
    return task


async def get_task(task_id: str) -> Task | None:
    client = redis_client.get_client()
    if client is None:
        return _store.get(task_id)
    try:
        raw = await client.get(_task_key(task_id))
        return _from_json(raw) if raw else None
    except Exception:  # noqa: BLE001
        logger.exception("Redis get_task failed")
        return _store.get(task_id)


async def list_tasks(chat_id: int, statuses: set[str] | None = None) -> list[Task]:
    statuses = statuses or {"pending"}
    client = redis_client.get_client()
    if client is None:
        ids = _index_mem.get(chat_id, set())
        out = [_store[i] for i in ids if i in _store]
    else:
        try:
            ids = await client.smembers(_index_key(chat_id))
            raws = await client.mget([_task_key(i) for i in ids]) if ids else []
            out = [_from_json(r) for r in raws if r]
        except Exception:  # noqa: BLE001
            logger.exception("Redis list_tasks failed")
            ids = _index_mem.get(chat_id, set())
            out = [_store[i] for i in ids if i in _store]
    out = [t for t in out if t.status in statuses]
    out.sort(key=lambda t: t.due_at)
    return out


async def set_status(task_id: str, status: str) -> Task | None:
    task = await get_task(task_id)
    if task is None:
        return None
    task.status = status
    await _save(task)
    if status != "pending":
        await _unschedule(task_id)
    return task


async def snooze(task_id: str, minutes: int) -> Task | None:
    task = await get_task(task_id)
    if task is None:
        return None
    when = _now_utc() + timedelta(minutes=minutes)
    task.remind_at = when.isoformat()
    task.stage = "primary"
    await _save(task)
    await _schedule(task_id, when)
    return task


async def pop_due(limit: int = 50) -> list[Task]:
    """Return tasks whose next reminder is due now, removing them from the
    schedule. Caller (task_assistant.advance_after_fire) decides whether to
    requeue a final nudge or the next recurrence."""
    now_ts = _now_utc().timestamp()
    client = redis_client.get_client()
    due_ids: list[str] = []
    if client is None:
        due_ids = [tid for tid, score in _due_mem.items() if score <= now_ts][:limit]
        for tid in due_ids:
            _due_mem.pop(tid, None)
    else:
        try:
            due_ids = await client.zrangebyscore(_DUE_ZSET, 0, now_ts, start=0, num=limit)
            if due_ids:
                await client.zrem(_DUE_ZSET, *due_ids)
        except Exception:  # noqa: BLE001
            logger.exception("Redis pop_due failed")
            return []

    out = []
    for tid in due_ids:
        t = await get_task(tid)
        if t and t.status == "pending":
            out.append(t)
    return out


async def advance_after_fire(task: Task) -> None:
    """Call after a reminder for `task` has been sent. Schedules the next
    reminder: the FINAL nudge, the next RECURRENCE, or nothing further."""
    if task.stage == "primary" and task.final_remind_at:
        task.stage = "final"
        await _save(task)
        await _schedule(task.id, datetime.fromisoformat(task.final_remind_at))
        return

    if task.recurrence != "none":
        due = datetime.fromisoformat(task.due_at)
        new_due = next_occurrence(due, task.recurrence)
        primary, final = compute_reminders(new_due, task.complexity, task.priority)
        task.due_at = new_due.isoformat()
        task.remind_at = primary.isoformat()
        task.final_remind_at = final.isoformat() if final else None
        task.stage = "primary"
        await _save(task)
        await _schedule(task.id, primary)
        return

    # One-off task, no more reminders queued — stays "pending" until the user
    # marks it done/cancelled, or re-schedules via snooze.
