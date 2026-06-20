"""
Shared lazy Redis client.

Used by history.py and memory.py for persistent storage when REDIS_URL is
configured (recommended for Railway — without it, conversation history and
project memory are wiped on every redeploy). Both modules fall back to an
in-memory dict with identical behaviour when this returns None, so the bot
works fine without Redis too — it just forgets everything on restart.
"""

import logging

from config import settings

logger = logging.getLogger(__name__)

_client = None
_init_attempted = False


def get_client():
    """Returns a shared redis.asyncio.Redis instance, or None if REDIS_URL
    isn't configured or the client failed to initialise. Safe to call from
    anywhere; never raises."""
    global _client, _init_attempted
    if not settings.redis_enabled:
        return None
    if _client is not None:
        return _client
    if _init_attempted:
        return None  # already tried and failed this run; don't retry every call
    _init_attempted = True
    try:
        import redis.asyncio as redis_async

        _client = redis_async.from_url(
            settings.redis_url, decode_responses=True, socket_connect_timeout=5
        )
        logger.info("Redis client initialised (persistent history + memory enabled).")
    except Exception:  # noqa: BLE001
        logger.exception("Failed to initialise Redis client — falling back to in-memory storage.")
        _client = None
    return _client
