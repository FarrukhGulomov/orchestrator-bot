"""
Conversation history store, scoped per (chat_id, agent_key).

Backed by Redis when REDIS_URL is configured (recommended for Railway —
without it, history is wiped on every redeploy, which defeats the point of
not having to re-explain context every time). Falls back to an in-memory
dict with identical behaviour when Redis isn't configured or a call fails.

WHY SCOPED PER AGENT (this fixes a real bug):
Each agent has a different persona/system prompt. If every agent shared one
history queue per chat, switching agents mid-conversation could leak the
PREVIOUS agent's context into the NEW agent's prompt — and the model tends to
keep following that prior context over the freshly-injected system prompt.
Observed symptom: a message classified as SOC / BUG returned a Product
Manager-style answer, because the shared history still carried an earlier
PM/idea conversation. Scoping history by agent_key keeps each role's thread
clean while the SAME agent still keeps context across its own turns.

Also tracks the (agent_key, request_type_key) of the last successfully
answered turn per chat. The router uses this to keep routing short,
content-free follow-ups ("ok", "continue", "davom et", "zo'r") to the SAME
agent instead of re-guessing from a message that has no real signal.

All public functions are async — both the Redis-backed and in-memory paths
use the same interface so callers never need to know which is active.
"""

import json
import logging
from collections import defaultdict, deque

import redis_client
from config import settings

logger = logging.getLogger(__name__)

# --- In-memory fallback --------------------------------------------------
_store: dict[tuple[int, str], deque] = defaultdict(
    lambda: deque(maxlen=settings.history_turns * 2)
)
_last_route_mem: dict[int, tuple[str, str]] = {}


def _hist_key(chat_id: int, agent_key: str) -> str:
    return f"hist:{chat_id}:{agent_key}"


def _keys_set_key(chat_id: int) -> str:
    return f"hist:keys:{chat_id}"


def _lastroute_key(chat_id: int) -> str:
    return f"lastroute:{chat_id}"


async def get_history(chat_id: int, agent_key: str) -> list[dict]:
    client = redis_client.get_client()
    if client is None:
        return list(_store[(chat_id, agent_key)])
    try:
        raw = await client.lrange(_hist_key(chat_id, agent_key), 0, -1)
        return [json.loads(r) for r in raw]
    except Exception:  # noqa: BLE001
        logger.exception("Redis get_history failed, returning empty for this call")
        return []


async def append(chat_id: int, agent_key: str, role: str, content: str) -> None:
    entry = {"role": role, "content": content}
    client = redis_client.get_client()
    if client is None:
        _store[(chat_id, agent_key)].append(entry)
        return
    try:
        key = _hist_key(chat_id, agent_key)
        cap = settings.history_turns * 2
        ttl = settings.redis_history_ttl_seconds
        async with client.pipeline(transaction=True) as pipe:
            pipe.rpush(key, json.dumps(entry))
            pipe.ltrim(key, -cap, -1)
            pipe.expire(key, ttl)
            pipe.sadd(_keys_set_key(chat_id), agent_key)
            pipe.expire(_keys_set_key(chat_id), ttl)
            await pipe.execute()
    except Exception:  # noqa: BLE001
        logger.exception("Redis append failed, falling back to in-memory for this entry")
        _store[(chat_id, agent_key)].append(entry)


async def reset(chat_id: int) -> None:
    """Clear every agent's thread (and the last-route hint) for this chat."""
    client = redis_client.get_client()
    if client is None:
        for key in [k for k in list(_store.keys()) if k[0] == chat_id]:
            _store.pop(key, None)
        _last_route_mem.pop(chat_id, None)
        return
    try:
        agent_keys = await client.smembers(_keys_set_key(chat_id))
        keys_to_delete = [_hist_key(chat_id, ak) for ak in agent_keys]
        keys_to_delete += [_keys_set_key(chat_id), _lastroute_key(chat_id)]
        if keys_to_delete:
            await client.delete(*keys_to_delete)
    except Exception:  # noqa: BLE001
        logger.exception("Redis reset failed")


async def get_last_route(chat_id: int) -> tuple[str, str] | None:
    client = redis_client.get_client()
    if client is None:
        return _last_route_mem.get(chat_id)
    try:
        raw = await client.get(_lastroute_key(chat_id))
        if not raw:
            return None
        agent_key, _, type_key = raw.partition("|")
        return (agent_key, type_key)
    except Exception:  # noqa: BLE001
        logger.exception("Redis get_last_route failed")
        return None


async def set_last_route(chat_id: int, agent_key: str, request_type_key: str) -> None:
    client = redis_client.get_client()
    if client is None:
        _last_route_mem[chat_id] = (agent_key, request_type_key)
        return
    try:
        await client.set(
            _lastroute_key(chat_id),
            f"{agent_key}|{request_type_key}",
            ex=settings.redis_history_ttl_seconds,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Redis set_last_route failed")
