"""
Daily task / reminder storage + the smart reminder-timing algorithm.

Pure data layer — no Telegram, no LLM calls (that's task_assistant.py).

STORAGE: three-tier fallback — PostgreSQL (DATABASE_URL) -> Redis
(REDIS_URL) -> in-memory. Tasks are durable business data (reminders a
real person is relying on to fire), so this follows the same tiering as
access_control/user_profile/decisions/memory — see db.py's module
docstring. The Postgres `tasks` table carries a `next_reminder_at` column
that always holds whichever reminder timestamp (primary or final) is next
due; this directly replaces the Redis version's separate `tasks:due` ZSET
"schedule index" — a plain indexed WHERE clause does the same job SQL-natively.

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

import db
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
    completed_at: str = ""         # ISO8601 UTC, set when status becomes "done"


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


async def _pg():
    if await db.init_schema():
        return await db.get_pool()
    return None


def _task_key(task_id: str) -> str:
    return f"task:{task_id}"


def _index_key(chat_id: int) -> str:
    return f"tasks:index:{chat_id}"


def _to_json(task: Task) -> str:
    return json.dumps(asdict(task))


def _from_json(raw: str) -> Task:
    return Task(**json.loads(raw))


def _dt(v: str | None) -> datetime | None:
    return datetime.fromisoformat(v) if v else None


def _row_to_task(row) -> Task:
    return Task(
        id=row["id"], chat_id=row["chat_id"], user_id=row["user_id"],
        title=row["title"], description=row["description"],
        due_at=row["due_at"].isoformat(), remind_at=row["remind_at"].isoformat(),
        final_remind_at=row["final_remind_at"].isoformat() if row["final_remind_at"] else None,
        complexity=row["complexity"], priority=row["priority"], recurrence=row["recurrence"],
        agent_key=row["agent_key"], status=row["status"], stage=row["stage"],
        created_at=row["created_at"].isoformat() if row["created_at"] else "",
        completed_at=row["completed_at"].isoformat() if row["completed_at"] else "",
    )


async def _pg_upsert(conn, task: Task, next_reminder_at: datetime | None) -> None:
    await conn.execute(
        """
        INSERT INTO tasks (
            id, chat_id, user_id, title, description, due_at, remind_at, final_remind_at,
            next_reminder_at, complexity, priority, recurrence, agent_key, status, stage,
            created_at, completed_at
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17)
        ON CONFLICT (id) DO UPDATE SET
            title = $4, description = $5, due_at = $6, remind_at = $7, final_remind_at = $8,
            next_reminder_at = $9, complexity = $10, priority = $11, recurrence = $12,
            agent_key = $13, status = $14, stage = $15, completed_at = $17
        """,
        task.id, task.chat_id, task.user_id, task.title, task.description,
        _dt(task.due_at), _dt(task.remind_at), _dt(task.final_remind_at),
        next_reminder_at, task.complexity, task.priority, task.recurrence,
        task.agent_key, task.status, task.stage,
        _dt(task.created_at) or _now_utc(), _dt(task.completed_at),
    )


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

    pool = await _pg()
    if pool is not None:
        try:
            async with pool.acquire() as conn:
                await _pg_upsert(conn, task, primary)
            return task
        except Exception:  # noqa: BLE001
            logger.exception("Postgres create_task failed, falling back to Redis/in-memory")

    await _save(task)
    await _schedule(task.id, primary)
    return task


async def get_task(task_id: str) -> Task | None:
    pool = await _pg()
    if pool is not None:
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow("SELECT * FROM tasks WHERE id = $1", task_id)
            return _row_to_task(row) if row else None
        except Exception:  # noqa: BLE001
            logger.exception("Postgres get_task failed")
            return _store.get(task_id)
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
    pool = await _pg()
    if pool is not None:
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM tasks WHERE chat_id = $1 AND status = ANY($2::text[]) ORDER BY due_at",
                    chat_id, list(statuses),
                )
            return [_row_to_task(r) for r in rows]
        except Exception:  # noqa: BLE001
            logger.exception("Postgres list_tasks failed")
            return []
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
    pool = await _pg()
    if pool is not None:
        try:
            completed_at = _now_utc() if status == "done" else None
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    UPDATE tasks SET status = $2, completed_at = COALESCE($3, completed_at),
                        next_reminder_at = CASE WHEN $2 != 'pending' THEN NULL ELSE next_reminder_at END
                    WHERE id = $1 RETURNING *
                    """,
                    task_id, status, completed_at,
                )
            return _row_to_task(row) if row else None
        except Exception:  # noqa: BLE001
            logger.exception("Postgres set_status failed")
            return None

    task = await get_task(task_id)
    if task is None:
        return None
    task.status = status
    if status == "done":
        task.completed_at = _now_utc().isoformat()
    await _save(task)
    if status != "pending":
        await _unschedule(task_id)
    return task


async def snooze(task_id: str, minutes: int) -> Task | None:
    when = _now_utc() + timedelta(minutes=minutes)
    pool = await _pg()
    if pool is not None:
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    UPDATE tasks SET remind_at = $2, stage = 'primary', next_reminder_at = $2
                    WHERE id = $1 RETURNING *
                    """,
                    task_id, when,
                )
            return _row_to_task(row) if row else None
        except Exception:  # noqa: BLE001
            logger.exception("Postgres snooze failed")
            return None

    task = await get_task(task_id)
    if task is None:
        return None
    task.remind_at = when.isoformat()
    task.stage = "primary"
    await _save(task)
    await _schedule(task_id, when)
    return task


async def pop_due(limit: int = 50) -> list[Task]:
    """Return tasks whose next reminder is due now, removing them from the
    schedule. Caller (task_assistant.advance_after_fire) decides whether to
    requeue a final nudge or the next recurrence."""
    pool = await _pg()
    if pool is not None:
        try:
            async with pool.acquire() as conn:
                # FOR UPDATE SKIP LOCKED: safe even if this ever ran from more
                # than one process — no two workers can claim the same task.
                rows = await conn.fetch(
                    """
                    UPDATE tasks SET next_reminder_at = NULL
                    WHERE id IN (
                        SELECT id FROM tasks
                        WHERE status = 'pending' AND next_reminder_at IS NOT NULL
                            AND next_reminder_at <= now()
                        ORDER BY next_reminder_at
                        LIMIT $1
                        FOR UPDATE SKIP LOCKED
                    )
                    RETURNING *
                    """,
                    limit,
                )
            return [_row_to_task(r) for r in rows]
        except Exception:  # noqa: BLE001
            logger.exception("Postgres pop_due failed")
            return []

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
        pool = await _pg()
        if pool is not None:
            try:
                async with pool.acquire() as conn:
                    await _pg_upsert(conn, task, datetime.fromisoformat(task.final_remind_at))
                return
            except Exception:  # noqa: BLE001
                logger.exception("Postgres advance_after_fire (final) failed")
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
        pool = await _pg()
        if pool is not None:
            try:
                async with pool.acquire() as conn:
                    await _pg_upsert(conn, task, primary)
                return
            except Exception:  # noqa: BLE001
                logger.exception("Postgres advance_after_fire (recurrence) failed")
        await _save(task)
        await _schedule(task.id, primary)
        return

    # One-off task, no more reminders queued — stays "pending" until the user
    # marks it done/cancelled, or re-schedules via snooze. (Postgres path:
    # next_reminder_at was already cleared by pop_due()'s UPDATE.)


# --------------------------------------------------------------------------
# Query helpers for digest / standup / weekly review
#
# Pure filters (overdue/in_window/completed_since_list) take an
# ALREADY-FETCHED task list so a caller building several views at once
# (digest = overdue + today + upcoming, all from the same pending set) pays
# for one list_tasks() round-trip instead of one per view. The async
# *_tasks wrappers remain for callers that only need a single view.
# --------------------------------------------------------------------------
def overdue(pending: list[Task], now: datetime | None = None) -> list[Task]:
    now = now or _now_utc()
    return [t for t in pending if datetime.fromisoformat(t.due_at) < now]


def in_window(pending: list[Task], start: datetime, end: datetime) -> list[Task]:
    """Tasks due within [start, end) — both UTC-aware."""
    return [t for t in pending if start <= datetime.fromisoformat(t.due_at) < end]


def completed_since_list(done: list[Task], since: datetime) -> list[Task]:
    """Tasks marked done at/after `since` (UTC-aware), oldest first."""
    out = [t for t in done if t.completed_at and datetime.fromisoformat(t.completed_at) >= since]
    out.sort(key=lambda t: t.completed_at)
    return out


async def overdue_tasks(chat_id: int) -> list[Task]:
    """Pending tasks whose deadline has already passed."""
    return overdue(await list_tasks(chat_id, {"pending"}))


async def due_in_window(chat_id: int, start: datetime, end: datetime) -> list[Task]:
    """Pending tasks due within [start, end) — both UTC-aware."""
    return in_window(await list_tasks(chat_id, {"pending"}), start, end)


async def completed_since(chat_id: int, since: datetime) -> list[Task]:
    """Tasks marked done at/after `since` (UTC-aware), oldest first."""
    return completed_since_list(await list_tasks(chat_id, {"done"}), since)
