"""
Decision log — a dated, append-only record of project decisions.

Answers the classic BA/PM question "kim, qachon, nima qaror qilgan edi?"
without digging through chat history. Distinct from memory.py: memory holds
durable *facts* that get injected into every agent prompt; the decision log
is a human-readable *audit trail* the user consults explicitly via
/decisions. Entries are dated and never silently deduplicated — superseding
decisions belong in the log too ("changed X back to Y" is itself a decision).

STORAGE: three-tier fallback — PostgreSQL (DATABASE_URL) -> Redis
(REDIS_URL) -> in-memory (no TTL — decisions are durable until
/cleardecisions). See db.py's module docstring.
"""

import logging

import db
import redis_client
import tasks

logger = logging.getLogger(__name__)

MAX_DECISIONS = 200

# --- In-memory fallback --------------------------------------------------
_store: dict[int, list[str]] = {}


async def _pg():
    if await db.init_schema():
        return await db.get_pool()
    return None


def _key(chat_id: int) -> str:
    return f"decisions:{chat_id}"


def _entry(text: str) -> str:
    stamp = tasks.now_local().strftime("%d-%m-%Y")
    return f"{stamp} — {text}"


async def add_decision(chat_id: int, text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return False
    entry = _entry(text[:500])

    pool = await _pg()
    if pool is not None:
        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(
                        "INSERT INTO decisions (chat_id, entry) VALUES ($1, $2)", chat_id, entry,
                    )
                    await conn.execute(
                        """
                        DELETE FROM decisions WHERE id IN (
                            SELECT id FROM decisions WHERE chat_id = $1
                            ORDER BY id DESC OFFSET $2
                        )
                        """,
                        chat_id, MAX_DECISIONS,
                    )
            return True
        except Exception:  # noqa: BLE001
            logger.exception("Postgres add_decision failed")
            return False

    client = redis_client.get_client()
    if client is None:
        entries = _store.setdefault(chat_id, [])
        entries.append(entry)
        if len(entries) > MAX_DECISIONS:
            del entries[0 : len(entries) - MAX_DECISIONS]
        return True
    try:
        async with client.pipeline(transaction=True) as pipe:
            pipe.rpush(_key(chat_id), entry)
            pipe.ltrim(_key(chat_id), -MAX_DECISIONS, -1)
            await pipe.execute()
        return True
    except Exception:  # noqa: BLE001
        logger.exception("Redis add_decision failed")
        return False


async def get_decisions(chat_id: int) -> list[str]:
    pool = await _pg()
    if pool is not None:
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT entry FROM decisions WHERE chat_id = $1 ORDER BY id", chat_id,
                )
            return [r["entry"] for r in rows]
        except Exception:  # noqa: BLE001
            logger.exception("Postgres get_decisions failed")
            return []
    client = redis_client.get_client()
    if client is None:
        return list(_store.get(chat_id, []))
    try:
        return await client.lrange(_key(chat_id), 0, -1)
    except Exception:  # noqa: BLE001
        logger.exception("Redis get_decisions failed")
        return []


async def clear_decisions(chat_id: int) -> int:
    pool = await _pg()
    if pool is not None:
        try:
            async with pool.acquire() as conn:
                result = await conn.execute("DELETE FROM decisions WHERE chat_id = $1", chat_id)
            return int(result.split()[-1])
        except Exception:  # noqa: BLE001
            logger.exception("Postgres clear_decisions failed")
            return 0
    client = redis_client.get_client()
    if client is None:
        return len(_store.pop(chat_id, []))
    try:
        n = await client.llen(_key(chat_id))
        if n:
            await client.delete(_key(chat_id))
        return n
    except Exception:  # noqa: BLE001
        logger.exception("Redis clear_decisions failed")
        return 0
