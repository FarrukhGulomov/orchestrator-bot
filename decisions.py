"""
Decision log — a dated, append-only record of project decisions.

Answers the classic BA/PM question "kim, qachon, nima qaror qilgan edi?"
without digging through chat history. Distinct from memory.py: memory holds
durable *facts* that get injected into every agent prompt; the decision log
is a human-readable *audit trail* the user consults explicitly via
/decisions. Entries are dated and never silently deduplicated — superseding
decisions belong in the log too ("changed X back to Y" is itself a decision).

Same storage pattern as memory.py: Redis list per chat (no TTL — decisions
are durable until /cleardecisions), in-memory dict fallback.
"""

import logging

import redis_client
import tasks

logger = logging.getLogger(__name__)

MAX_DECISIONS = 200

# --- In-memory fallback --------------------------------------------------
_store: dict[int, list[str]] = {}


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
    client = redis_client.get_client()
    if client is None:
        return list(_store.get(chat_id, []))
    try:
        return await client.lrange(_key(chat_id), 0, -1)
    except Exception:  # noqa: BLE001
        logger.exception("Redis get_decisions failed")
        return []


async def clear_decisions(chat_id: int) -> int:
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
