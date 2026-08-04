"""
Shared PostgreSQL connection pool — the durable "source of truth" for
relational business data.

WHY POSTGRES ALONGSIDE REDIS (not instead of): the two stores solve
different problems. Redis is right for cheap, fast, expiring, high-churn
state (a relay mapping that only matters for 7 days, a rolling 10-turn
conversation window, a 1-hour meeting-minutes stash) — nobody needs to
query, join, or audit that data, and losing it on expiry is by design.
Postgres is right for data that IS the business record: who's approved,
what they're called, what tasks are due when, what was decided and by
whom. That data deserves real columns, indexes, and durability guarantees
instead of hand-rolled hash/set/list encodings — and Railway's managed
Postgres survives redeploys the same way managed Redis does, so this
isn't about reliability, it's about using the right tool for each kind of
data.

Modules that own durable business data (access_control, user_profile,
tasks, decisions, memory) use a THREE-TIER fallback, checked in this
order: PostgreSQL (if DATABASE_URL set) -> Redis (if REDIS_URL set) ->
in-memory. This means deploying this code before DATABASE_URL is wired up
on Railway changes nothing — existing Redis-backed data keeps working
exactly as before. The moment DATABASE_URL is set, init_schema() creates
the tables and migrate_from_redis() performs a ONE-TIME backfill of
whatever was in Redis, so already-approved users/tasks/decisions aren't
lost on cutover. Ephemeral modules (history, business_copilot,
group_copilot, quick_actions, minutes' batch stash, access_control's
relay/pending-queue) are UNCHANGED — they stay Redis/in-memory only, on
purpose, per the reasoning above.
"""

import logging
import os

from config import settings

logger = logging.getLogger(__name__)

_pool = None
_init_attempted = False
_schema_ready = False


async def get_pool():
    """Returns a shared asyncpg pool, or None if DATABASE_URL isn't
    configured or the pool failed to initialise. Safe to call from
    anywhere; never raises."""
    global _pool, _init_attempted
    if not settings.db_enabled:
        return None
    if _pool is not None:
        return _pool
    if _init_attempted:
        return None  # already tried and failed this run; don't retry every call
    _init_attempted = True
    try:
        import asyncpg

        _pool = await asyncpg.create_pool(
            settings.database_url, min_size=1, max_size=5, command_timeout=10,
        )
        logger.info("PostgreSQL pool initialised.")
    except Exception:  # noqa: BLE001
        logger.exception("Failed to initialise PostgreSQL pool — falling back to Redis/in-memory storage.")
        _pool = None
    return _pool


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS kv_store (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    user_id          BIGINT PRIMARY KEY,
    username         TEXT,
    full_name        TEXT,
    fio              TEXT,
    phone            TEXT,
    status           TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'denied')),
    onboarding_state TEXT,
    approved_at      TIMESTAMPTZ,
    approved_via     TEXT,
    denied_at        TIMESTAMPTZ,
    denied_via       TEXT,
    first_seen       TIMESTAMPTZ,
    last_seen        TIMESTAMPTZ,
    message_count    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_users_status ON users (status);
CREATE INDEX IF NOT EXISTS idx_users_last_seen ON users (last_seen DESC);

CREATE TABLE IF NOT EXISTS seen_users (
    user_id BIGINT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS user_profile_notes (
    id         SERIAL PRIMARY KEY,
    user_id    BIGINT NOT NULL,
    note       TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_profile_notes_user ON user_profile_notes (user_id, id);

CREATE TABLE IF NOT EXISTS tasks (
    id                TEXT PRIMARY KEY,
    chat_id           BIGINT NOT NULL,
    user_id           BIGINT NOT NULL,
    title             TEXT NOT NULL,
    description       TEXT NOT NULL DEFAULT '',
    due_at            TIMESTAMPTZ NOT NULL,
    remind_at         TIMESTAMPTZ NOT NULL,
    final_remind_at   TIMESTAMPTZ,
    next_reminder_at  TIMESTAMPTZ,
    complexity        TEXT NOT NULL DEFAULT 'medium',
    priority          TEXT NOT NULL DEFAULT 'medium',
    recurrence        TEXT NOT NULL DEFAULT 'none',
    agent_key         TEXT,
    status            TEXT NOT NULL DEFAULT 'pending',
    stage             TEXT NOT NULL DEFAULT 'primary',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at      TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_tasks_chat ON tasks (chat_id, status);
CREATE INDEX IF NOT EXISTS idx_tasks_due ON tasks (next_reminder_at) WHERE status = 'pending';

CREATE TABLE IF NOT EXISTS decisions (
    id         SERIAL PRIMARY KEY,
    chat_id    BIGINT NOT NULL,
    entry      TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_decisions_chat ON decisions (chat_id, id);

CREATE TABLE IF NOT EXISTS expenses (
    id         SERIAL PRIMARY KEY,
    user_id    BIGINT NOT NULL,
    chat_id    BIGINT NOT NULL,
    amount     NUMERIC(14, 2) NOT NULL,
    currency   TEXT NOT NULL DEFAULT 'UZS',
    category   TEXT NOT NULL DEFAULT 'boshqa',
    note       TEXT NOT NULL DEFAULT '',
    spent_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_expenses_user_date ON expenses (user_id, spent_at DESC);

CREATE TABLE IF NOT EXISTS memory_facts (
    id         SERIAL PRIMARY KEY,
    chat_id    BIGINT NOT NULL,
    fact       TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_memory_chat ON memory_facts (chat_id, id);

-- One row per LLM API call (see telemetry.py). Written on every reply, so
-- it grows faster than every other table here — hence the narrow columns
-- and the two indexes matching the only two access patterns: "this user's
-- spend today" (budget check, hot path) and "everything in a window"
-- (/xarajatai report).
--
-- provider is the one that ACTUALLY answered, which is not necessarily the
-- one the router preferred — that distinction is the whole point of
-- recording fallback_position.
CREATE TABLE IF NOT EXISTS llm_calls (
    id                SERIAL PRIMARY KEY,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    provider          TEXT NOT NULL,
    model             TEXT NOT NULL,
    tier              TEXT NOT NULL DEFAULT 'main',
    input_tokens      INTEGER NOT NULL DEFAULT 0,
    output_tokens     INTEGER NOT NULL DEFAULT 0,
    cost_usd          NUMERIC(12, 6) NOT NULL DEFAULT 0,
    latency_ms        INTEGER NOT NULL DEFAULT 0,
    ok                BOOLEAN NOT NULL DEFAULT TRUE,
    fallback_position INTEGER NOT NULL DEFAULT 0,
    user_id           BIGINT,
    chat_id           BIGINT,
    agent_key         TEXT,
    error             TEXT
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_user_day ON llm_calls (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_llm_calls_created ON llm_calls (created_at DESC);

-- Audit trail for meeting_attendee.py sessions (bot joining a live
-- Meet/Zoom/Teams call). `disclosed` records whether the mandatory
-- in-meeting chat announcement was confirmed sent — see that module's
-- docstring for why audio capture never starts unless this is true.
CREATE TABLE IF NOT EXISTS meeting_sessions (
    id                TEXT PRIMARY KEY,
    chat_id           BIGINT NOT NULL,
    user_id           BIGINT NOT NULL,
    platform          TEXT NOT NULL,
    meeting_url       TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'starting',
    disclosed         BOOLEAN NOT NULL DEFAULT FALSE,
    started_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at          TIMESTAMPTZ,
    error             TEXT,
    transcript_chars  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_meeting_sessions_chat ON meeting_sessions (chat_id, started_at DESC);
"""


async def init_schema() -> bool:
    """Create tables if they don't exist yet. Idempotent — safe to call on
    every startup. Returns True if Postgres is usable, False otherwise."""
    global _schema_ready
    pool = await get_pool()
    if pool is None:
        return False
    if _schema_ready:
        return True
    try:
        async with pool.acquire() as conn:
            await conn.execute(_SCHEMA_SQL)
        _schema_ready = True
        logger.info("PostgreSQL schema ready.")
        return True
    except Exception:  # noqa: BLE001
        logger.exception("Failed to initialise PostgreSQL schema")
        return False


_MIGRATIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "migrations")

_MIGRATIONS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename   TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


async def run_migrations() -> int:
    """Apply any migrations/*.sql file not yet recorded in
    schema_migrations, in filename order, one transaction per file — see
    migrations/README.md for the convention and why this exists alongside
    (not instead of) init_schema()'s CREATE TABLE IF NOT EXISTS baseline.

    Safe to call on every startup: already-applied files are skipped, and
    a missing migrations/ directory or no pending files is a normal no-op,
    not an error. Returns the count of newly-applied migrations."""
    pool = await get_pool()
    if pool is None:
        return 0
    if not os.path.isdir(_MIGRATIONS_DIR):
        return 0

    files = sorted(f for f in os.listdir(_MIGRATIONS_DIR) if f.endswith(".sql"))
    if not files:
        return 0

    try:
        async with pool.acquire() as conn:
            await conn.execute(_MIGRATIONS_SCHEMA_SQL)
            applied = {
                r["filename"] for r in await conn.fetch("SELECT filename FROM schema_migrations")
            }

            newly_applied = 0
            for filename in files:
                if filename in applied:
                    continue
                path = os.path.join(_MIGRATIONS_DIR, filename)
                with open(path, "r", encoding="utf-8") as f:
                    sql = f.read()
                async with conn.transaction():
                    await conn.execute(sql)
                    await conn.execute(
                        "INSERT INTO schema_migrations (filename) VALUES ($1)", filename,
                    )
                logger.info("Applied migration: %s", filename)
                newly_applied += 1
            return newly_applied
    except Exception:  # noqa: BLE001
        logger.exception("Migration run failed — see migrations/README.md")
        return 0


async def kv_get(key: str) -> str | None:
    pool = await get_pool()
    if pool is None:
        return None
    try:
        async with pool.acquire() as conn:
            return await conn.fetchval("SELECT value FROM kv_store WHERE key = $1", key)
    except Exception:  # noqa: BLE001
        logger.exception("db.kv_get failed for key=%s", key)
        return None


async def kv_set(key: str, value: str) -> None:
    pool = await get_pool()
    if pool is None:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO kv_store (key, value) VALUES ($1, $2) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                key, value,
            )
    except Exception:  # noqa: BLE001
        logger.exception("db.kv_set failed for key=%s", key)


async def migrate_from_redis() -> None:
    """ONE-TIME backfill: if Postgres's `users` table is empty but Redis
    has access-control data, copy it over so cutting over to Postgres
    doesn't reset every already-approved user back to 'pending'. Guarded
    by a kv_store flag so it only ever runs once per database, and is a
    no-op (fast) on every startup after that. Safe to call even when
    Redis was never configured (falls through: 0 users copied)."""
    pool = await get_pool()
    if pool is None:
        return
    already_done = await kv_get("migrated_from_redis")
    if already_done:
        return

    import redis_client
    client = redis_client.get_client()
    if client is None:
        await kv_set("migrated_from_redis", "true")  # nothing to migrate, don't recheck every boot
        return

    try:
        known_raw, approved_raw, denied_raw = (
            await client.smembers("access:known"),
            await client.smembers("access:approved"),
            await client.smembers("access:denied"),
        )
        all_ids = {int(x) for x in known_raw} | {int(x) for x in approved_raw} | {int(x) for x in denied_raw}
        if not all_ids:
            await kv_set("migrated_from_redis", "true")
            return

        approved_ids = {int(x) for x in approved_raw}
        denied_ids = {int(x) for x in denied_raw}
        copied = 0
        async with pool.acquire() as conn:
            for uid in all_ids:
                raw = await client.hgetall(f"access:user:{uid}")
                onboard_state = await client.hget("access:onboard:state", str(uid))
                status = "approved" if uid in approved_ids else ("denied" if uid in denied_ids else "pending")

                def _parse_ts(v):
                    if not v:
                        return None
                    try:
                        from datetime import datetime
                        return datetime.fromisoformat(v)
                    except ValueError:
                        return None

                await conn.execute(
                    """
                    INSERT INTO users (
                        user_id, username, full_name, fio, phone, status, onboarding_state,
                        approved_at, approved_via, denied_at, denied_via,
                        first_seen, last_seen, message_count
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
                    ON CONFLICT (user_id) DO NOTHING
                    """,
                    uid, raw.get("username") or None, raw.get("full_name") or None,
                    raw.get("fio") or None, raw.get("phone") or None, status, onboard_state,
                    _parse_ts(raw.get("approved_at")), raw.get("approved_via") or None,
                    _parse_ts(raw.get("denied_at")), raw.get("denied_via") or None,
                    _parse_ts(raw.get("first_seen")), _parse_ts(raw.get("last_seen")),
                    int(raw.get("message_count") or 0),
                )
                copied += 1

                notes = await client.lrange(f"user_profile:{uid}", 0, -1)
                for note in notes:
                    await conn.execute(
                        "INSERT INTO user_profile_notes (user_id, note) VALUES ($1, $2)", uid, note,
                    )

        admin_chat = await client.get("access:admin_chat_id")
        if admin_chat:
            await kv_set("admin_chat_id", admin_chat)

        seen_raw = await client.smembers("access:seen")
        if seen_raw:
            async with pool.acquire() as conn:
                for uid_s in seen_raw:
                    await conn.execute(
                        "INSERT INTO seen_users (user_id) VALUES ($1) ON CONFLICT DO NOTHING",
                        int(uid_s),
                    )

        logger.info("Migrated %d user(s) from Redis to PostgreSQL.", copied)
    except Exception:  # noqa: BLE001
        logger.exception("Redis -> PostgreSQL migration failed (will retry next startup)")
        return  # don't set the flag — retry on next boot

    await kv_set("migrated_from_redis", "true")
