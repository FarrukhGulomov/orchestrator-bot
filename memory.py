"""
Lightweight project memory — a realistic, honest version of "self-learning".

IMPORTANT — what this is NOT: this bot cannot retrain or fine-tune any
underlying LLM. Real model self-training needs a training pipeline, GPU
infrastructure, and labeled data — none of which a Telegram bot has access
to. Claiming otherwise would be dishonest.

What this DOES do, which is the realistic and genuinely useful interpretation
of "the bot learns from conversations": durable facts about the team's
project (project name, confirmed tech-stack decisions, budget/timeline
figures, established constraints/conventions) are extracted from
consequential exchanges and stored per chat. Every agent's prompt then gets
these facts injected as context — so answers get progressively better
grounded in THIS team's actual project over time, without the user needing
to repeat context every message. That's real, measurable improvement in
answer quality; it just isn't "the model learned new weights".

STORAGE: three-tier fallback — PostgreSQL (DATABASE_URL) -> Redis
(REDIS_URL) -> in-memory. Facts have NO TTL — they're durable until
explicitly cleared with /forget. See db.py's module docstring for why
this differs from the codebase's purely-ephemeral Redis usage elsewhere.
"""

import logging

import db
import redis_client

logger = logging.getLogger(__name__)

MAX_FACTS = 40

# --- In-memory fallback --------------------------------------------------
_store: dict[int, list[str]] = {}


async def _pg():
    if await db.init_schema():
        return await db.get_pool()
    return None


def _words(text: str) -> set[str]:
    return {w for w in text.lower().split() if w}


def _is_near_duplicate(fact: str, existing: list[str]) -> bool:
    """True if `fact` is an exact match OR shares >= 80% of its words with an
    already-stored fact — stops trivially-reworded duplicates accumulating."""
    if fact in existing:
        return True
    new_words = _words(fact)
    if not new_words:
        return True
    for old in existing:
        old_words = _words(old)
        if not old_words:
            continue
        overlap = len(new_words & old_words)
        if overlap / len(new_words) >= 0.8 or overlap / len(old_words) >= 0.8:
            return True
    return False


def _mem_key(chat_id: int) -> str:
    return f"mem:{chat_id}"


async def get_memory(chat_id: int) -> list[str]:
    pool = await _pg()
    if pool is not None:
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT fact FROM memory_facts WHERE chat_id = $1 ORDER BY id", chat_id,
                )
            return [r["fact"] for r in rows]
        except Exception:  # noqa: BLE001
            logger.exception("Postgres get_memory failed")
            return list(_store.get(chat_id, []))
    client = redis_client.get_client()
    if client is None:
        return list(_store.get(chat_id, []))
    try:
        return await client.lrange(_mem_key(chat_id), 0, -1)
    except Exception:  # noqa: BLE001
        logger.exception("Redis get_memory failed, returning empty for this call")
        return []


async def add_memory(chat_id: int, fact: str) -> bool:
    """Add a fact if it's non-empty and not a near-duplicate. Returns True if added."""
    fact = (fact or "").strip()
    if not fact:
        return False

    pool = await _pg()
    if pool is not None:
        try:
            async with pool.acquire() as conn:
                existing = [r["fact"] for r in await conn.fetch(
                    "SELECT fact FROM memory_facts WHERE chat_id = $1 ORDER BY id", chat_id,
                )]
                if _is_near_duplicate(fact, existing):
                    return False
                async with conn.transaction():
                    await conn.execute(
                        "INSERT INTO memory_facts (chat_id, fact) VALUES ($1, $2)", chat_id, fact,
                    )
                    await conn.execute(
                        """
                        DELETE FROM memory_facts WHERE id IN (
                            SELECT id FROM memory_facts WHERE chat_id = $1
                            ORDER BY id DESC OFFSET $2
                        )
                        """,
                        chat_id, MAX_FACTS,
                    )
            return True
        except Exception:  # noqa: BLE001
            logger.exception("Postgres add_memory failed")
            return False

    client = redis_client.get_client()
    if client is None:
        facts = _store.setdefault(chat_id, [])
        if _is_near_duplicate(fact, facts):
            return False
        facts.append(fact)
        if len(facts) > MAX_FACTS:
            del facts[0 : len(facts) - MAX_FACTS]
        return True

    try:
        existing = await client.lrange(_mem_key(chat_id), 0, -1)
        if _is_near_duplicate(fact, existing):
            return False
        async with client.pipeline(transaction=True) as pipe:
            pipe.rpush(_mem_key(chat_id), fact)
            pipe.ltrim(_mem_key(chat_id), -MAX_FACTS, -1)
            await pipe.execute()
        return True
    except Exception:  # noqa: BLE001
        logger.exception("Redis add_memory failed")
        return False


async def clear_memory(chat_id: int) -> int:
    """Clear all stored facts for this chat. Returns how many were removed."""
    pool = await _pg()
    if pool is not None:
        try:
            async with pool.acquire() as conn:
                result = await conn.execute("DELETE FROM memory_facts WHERE chat_id = $1", chat_id)
            return int(result.split()[-1])
        except Exception:  # noqa: BLE001
            logger.exception("Postgres clear_memory failed")
            return 0
    client = redis_client.get_client()
    if client is None:
        facts = _store.pop(chat_id, [])
        return len(facts)
    try:
        n = await client.llen(_mem_key(chat_id))
        if n:
            await client.delete(_mem_key(chat_id))
        return n
    except Exception:  # noqa: BLE001
        logger.exception("Redis clear_memory failed")
        return 0
