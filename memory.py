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

Backed by Redis when REDIS_URL is configured (recommended for Railway —
without it, this is wiped on every redeploy). Falls back to an in-memory
dict with identical behaviour when Redis isn't configured or a call fails.
Unlike history.py, facts have NO TTL in Redis — they're meant to be durable
until explicitly cleared with /forget, not to expire from inactivity.
"""

import logging

import redis_client

logger = logging.getLogger(__name__)

MAX_FACTS = 40

# --- In-memory fallback --------------------------------------------------
_store: dict[int, list[str]] = {}


def _mem_key(chat_id: int) -> str:
    return f"mem:{chat_id}"


async def get_memory(chat_id: int) -> list[str]:
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

    client = redis_client.get_client()
    if client is None:
        facts = _store.setdefault(chat_id, [])
        if fact in facts:
            return False
        facts.append(fact)
        if len(facts) > MAX_FACTS:
            del facts[0 : len(facts) - MAX_FACTS]
        return True

    try:
        existing = await client.lrange(_mem_key(chat_id), 0, -1)
        if fact in existing:
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
