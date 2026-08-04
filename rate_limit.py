"""
Per-user burst rate limiting — a short sliding window on how many messages
one user can send, independent of telemetry.py's long-window daily token
BUDGET.

WHY BOTH EXIST: budget caps total spend over a day (a product/cost
decision, ships disabled). This caps burst — N messages in a short window
— so a stuck client retry-looping, or someone hammering the bot, can't
fire a pile of concurrent LLM calls before the daily counter even catches
up. It's abuse protection, not a spend control, so it ships enabled by
default (see config.py).

ALGORITHM: sliding-window log. Each check records "now" and counts how
many timestamps fall within the last WINDOW_SECONDS; over the limit ->
blocked, with retry_after computed from the oldest timestamp still in
window. Approximate under concurrent writes (no cross-process locking) is
an accepted tradeoff — this is an abuse guard, not a billing meter, and
occasionally letting one extra message through is a fine failure mode.

STORAGE: Redis (sorted set: score = timestamp) when configured, else an
in-memory dict — this is short-lived session state, not a business
record, so it deliberately skips the Postgres tier (see db.py's module
docstring on when NOT to reach for durable storage).
"""

import logging
import time

import redis_client
from config import settings

logger = logging.getLogger(__name__)

_KEY_PREFIX = "ratelimit:"

# In-memory fallback: user_id -> [timestamps]. Unbounded per-user list would
# leak memory across a long-running process; pruned to the window on every
# check, so it can only ever hold WINDOW_SECONDS worth of activity per user.
_mem: dict[int, list[float]] = {}


async def _check_redis(client, user_id: int, now: float) -> tuple[bool, float]:
    key = f"{_KEY_PREFIX}{user_id}"
    window_start = now - settings.rate_limit_window_seconds
    try:
        pipe = client.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zrange(key, 0, 0, withscores=True)
        pipe.zcard(key)
        _removed, oldest, count = await pipe.execute()

        if count >= settings.rate_limit_max_per_window:
            oldest_ts = oldest[0][1] if oldest else now
            retry_after = max(0.0, oldest_ts + settings.rate_limit_window_seconds - now)
            return False, retry_after

        pipe = client.pipeline()
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, settings.rate_limit_window_seconds + 1)
        await pipe.execute()
        return True, 0.0
    except Exception:  # noqa: BLE001 — a broken rate limiter must not block real traffic
        logger.exception("Redis rate limit check failed — allowing the request")
        return True, 0.0


def _check_memory(user_id: int, now: float) -> tuple[bool, float]:
    window_start = now - settings.rate_limit_window_seconds
    timestamps = [t for t in _mem.get(user_id, []) if t > window_start]

    if len(timestamps) >= settings.rate_limit_max_per_window:
        _mem[user_id] = timestamps
        retry_after = max(0.0, timestamps[0] + settings.rate_limit_window_seconds - now)
        return False, retry_after

    timestamps.append(now)
    _mem[user_id] = timestamps
    return True, 0.0


async def check(user_id: int) -> tuple[bool, int]:
    """(allowed, retry_after_seconds). Fails OPEN — a broken limiter must
    never be the reason a real message doesn't get answered."""
    if not settings.rate_limit_enabled or not user_id:
        return True, 0
    now = time.time()  # wall clock, not monotonic — Redis's clock is a separate process

    try:
        client = redis_client.get_client()
        if client is not None:
            allowed, retry_after = await _check_redis(client, user_id, now)
        else:
            allowed, retry_after = _check_memory(user_id, now)
    except Exception:  # noqa: BLE001 — see the fail-open contract above
        logger.exception("Rate limit check failed — allowing the request")
        return True, 0
    return allowed, int(retry_after) + (1 if retry_after > int(retry_after) else 0)
