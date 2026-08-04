"""
LLM call telemetry — what every AI call cost, how long it took, and which
provider actually served it.

WHY THIS EXISTS: the bot answers every message with a top-tier paid model
across a seven-provider failover chain, and before this module nothing
recorded a single token. There was no way to answer "what did last week
cost", "which provider is actually carrying the traffic", or "is one user
burning the budget" — the first signal of a runaway loop would have been
the invoice.

WHAT IT RECORDS: one row per API call — the provider that ACTUALLY
answered (not the one the router preferred), model, tier, tokens in/out,
estimated cost, latency, success, and its position in the failover chain.
Position 0 means the first choice worked; anything higher means a
provider ahead of it failed, which is the signal that a key is dead or
rate-limited.

COST FIGURES ARE ESTIMATES. They come from a static table in config.py,
not from any billing API, and vendors reprice without notice. Everything
this module renders for a human says so.

TWO HARD RULES, both about never letting bookkeeping break the product:
  1. record() must never raise and never block a reply. Recording failure
     is logged and swallowed — losing a telemetry row is strictly better
     than losing the user's answer.
  2. The budget cap ships DISABLED (DAILY_USER_TOKEN_BUDGET=0). A cap set
     carelessly locks someone out of their own assistant, and the admin is
     never capped regardless.

STORAGE: PostgreSQL when DATABASE_URL is set, otherwise a bounded
in-memory ring buffer. Deliberately NOT Redis-tiered like tasks/decisions:
this is high-volume append-only analytics, not business records, and
hand-encoding it into Redis lists would buy nothing. Without Postgres you
still get same-process totals (enough for budget enforcement between
restarts); you just don't get history across redeploys.

ATTRIBUTION: llm_clients.py calls record() from deep inside the failover
path, where the Telegram user/chat is not in scope. Rather than thread
user_id through a dozen signatures, bot.py sets a contextvar per update
(see set_call_context) and this module reads it — async-safe, and calls
made outside a handler (startup, background loops) simply record NULL.
"""

import asyncio
import contextvars
import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import db
from config import estimate_cost_usd, settings

logger = logging.getLogger(__name__)

# Bounded so a long-lived process without Postgres can't grow without limit.
_MEM_MAX = 5000
_mem: deque = deque(maxlen=_MEM_MAX)


@dataclass(frozen=True)
class CallContext:
    user_id: int | None = None
    chat_id: int | None = None
    agent_key: str | None = None


_EMPTY = CallContext()
_ctx: contextvars.ContextVar[CallContext] = contextvars.ContextVar("llm_call_ctx", default=_EMPTY)


def set_call_context(user_id: int | None, chat_id: int | None, agent_key: str | None = None) -> None:
    """Attribute subsequent LLM calls in this async task to a user/chat."""
    _ctx.set(CallContext(user_id=user_id, chat_id=chat_id, agent_key=agent_key))


def set_agent(agent_key: str | None) -> None:
    """Refine the current context once the router has chosen an agent,
    keeping whatever user/chat was already attributed."""
    cur = _ctx.get()
    _ctx.set(CallContext(user_id=cur.user_id, chat_id=cur.chat_id, agent_key=agent_key))


def get_call_context() -> CallContext:
    return _ctx.get()


def clear_call_context() -> None:
    _ctx.set(_EMPTY)


async def _pg():
    if await db.init_schema():
        return await db.get_pool()
    return None


def extract_usage(response: object) -> tuple[int, int]:
    """(input_tokens, output_tokens) from a raw provider response.

    Anthropic exposes usage.input_tokens/output_tokens; every
    OpenAI-compatible provider (OpenAI, Gemini, Grok, DeepSeek, Kimi,
    OpenRouter) uses usage.prompt_tokens/completion_tokens. Some free
    OpenRouter models omit usage entirely — hence the total-only fallback
    and the (0, 0) default rather than an exception.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0

    def _int(*names: str) -> int:
        for n in names:
            v = getattr(usage, n, None)
            if isinstance(v, (int, float)):
                return int(v)
        return 0

    inp = _int("input_tokens", "prompt_tokens")
    out = _int("output_tokens", "completion_tokens")
    if inp == 0 and out == 0:
        # Nothing itemised — attribute a reported total to output, which is
        # the more expensive side, so an estimate errs high rather than low.
        out = _int("total_tokens")
    return inp, out


async def record(
    provider: str,
    model: str,
    tier: str = "main",
    input_tokens: int = 0,
    output_tokens: int = 0,
    latency_ms: int = 0,
    ok: bool = True,
    fallback_position: int = 0,
    error: str | None = None,
) -> None:
    """Record one LLM call. Never raises — see rule 1 in the module docstring."""
    if not settings.telemetry_enabled:
        return
    try:
        ctx = _ctx.get()
        cost = estimate_cost_usd(model, input_tokens, output_tokens)
        row = {
            "created_at": datetime.now(timezone.utc),
            "provider": provider,
            "model": model,
            "tier": tier,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "cost_usd": cost,
            "latency_ms": int(latency_ms),
            "ok": bool(ok),
            "fallback_position": int(fallback_position),
            "user_id": ctx.user_id,
            "chat_id": ctx.chat_id,
            "agent_key": ctx.agent_key,
            "error": (error or "")[:300] or None,
        }
        _mem.append(row)

        pool = await _pg()
        if pool is None:
            return
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO llm_calls (
                    created_at, provider, model, tier, input_tokens, output_tokens,
                    cost_usd, latency_ms, ok, fallback_position, user_id, chat_id,
                    agent_key, error
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
                """,
                row["created_at"], row["provider"], row["model"], row["tier"],
                row["input_tokens"], row["output_tokens"], row["cost_usd"],
                row["latency_ms"], row["ok"], row["fallback_position"],
                row["user_id"], row["chat_id"], row["agent_key"], row["error"],
            )
    except Exception:  # noqa: BLE001 — bookkeeping must never break a reply
        logger.exception("telemetry.record failed (call itself was unaffected)")


def record_soon(**kwargs) -> None:
    """Fire-and-forget record() for synchronous call sites. The done-callback
    is what stops an exception here from vanishing into an unretrieved task."""
    if not settings.telemetry_enabled:
        return
    try:
        task = asyncio.get_running_loop().create_task(record(**kwargs))
        task.add_done_callback(_log_task_exception)
    except RuntimeError:
        # No running loop (import time, sync context) — the in-memory ring
        # buffer would be the only thing we could still write, and a call
        # made outside the event loop isn't user traffic worth recording.
        pass


def _log_task_exception(task: "asyncio.Task") -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.warning("telemetry background write failed: %s", exc)


# --- Budget ----------------------------------------------------------------
async def tokens_used_today(user_id: int) -> int:
    """Total tokens this user has spent since UTC midnight. Hot path — runs
    before every AI reply when a cap is configured."""
    if not user_id:
        return 0
    start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        pool = await _pg()
        if pool is not None:
            async with pool.acquire() as conn:
                val = await conn.fetchval(
                    "SELECT COALESCE(SUM(input_tokens + output_tokens), 0) "
                    "FROM llm_calls WHERE user_id = $1 AND created_at >= $2",
                    user_id, start,
                )
                return int(val or 0)
    except Exception:  # noqa: BLE001
        logger.exception("tokens_used_today query failed — falling back to in-memory")

    return sum(
        r["input_tokens"] + r["output_tokens"]
        for r in _mem
        if r["user_id"] == user_id and r["created_at"] >= start
    )


async def over_budget(user_id: int) -> tuple[bool, int, int]:
    """(is_over, used, limit). Always (False, 0, 0) when no cap is set.

    Fails OPEN: if the usage query itself breaks, the user is not blocked.
    Refusing to answer because bookkeeping is broken would be the worse
    failure — the cap is a cost guardrail, not a security control.
    """
    if not settings.budget_enforced or not user_id:
        return False, 0, 0
    limit = settings.daily_user_token_budget
    try:
        used = await tokens_used_today(user_id)
    except Exception:  # noqa: BLE001
        logger.exception("over_budget check failed — allowing the request")
        return False, 0, limit
    return used >= limit, used, limit


# --- Reporting -------------------------------------------------------------
async def _rows_since(since: datetime) -> list[dict]:
    try:
        pool = await _pg()
        if pool is not None:
            async with pool.acquire() as conn:
                records = await conn.fetch(
                    "SELECT created_at, provider, model, tier, input_tokens, output_tokens, "
                    "cost_usd, latency_ms, ok, fallback_position, user_id, agent_key "
                    "FROM llm_calls WHERE created_at >= $1 ORDER BY created_at DESC",
                    since,
                )
                return [dict(r) for r in records]
    except Exception:  # noqa: BLE001
        logger.exception("telemetry query failed — falling back to in-memory")
    return [r for r in _mem if r["created_at"] >= since]


def _sum_group(rows: list[dict], key: str) -> list[tuple[str, int, float, int]]:
    """[(group, calls, cost, tokens)] sorted by cost desc."""
    agg: dict[str, list] = {}
    for r in rows:
        k = str(r.get(key) or "—")
        slot = agg.setdefault(k, [0, 0.0, 0])
        slot[0] += 1
        slot[1] += float(r["cost_usd"] or 0)
        slot[2] += int(r["input_tokens"] or 0) + int(r["output_tokens"] or 0)
    return sorted(
        ((k, v[0], v[1], v[2]) for k, v in agg.items()),
        key=lambda x: x[2], reverse=True,
    )


async def spend_report(days: int = 7) -> dict:
    """Aggregates for /xarajatai: totals, and breakdowns by day, user,
    provider and agent."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = await _rows_since(since)
    total_cost = sum(float(r["cost_usd"] or 0) for r in rows)
    total_tokens = sum(int(r["input_tokens"] or 0) + int(r["output_tokens"] or 0) for r in rows)
    failures = sum(1 for r in rows if not r["ok"])
    fallbacks = sum(1 for r in rows if int(r["fallback_position"] or 0) > 0)

    by_day: dict[str, list] = {}
    for r in rows:
        k = r["created_at"].strftime("%Y-%m-%d")
        slot = by_day.setdefault(k, [0, 0.0])
        slot[0] += 1
        slot[1] += float(r["cost_usd"] or 0)

    return {
        "days": days,
        "calls": len(rows),
        "cost": total_cost,
        "tokens": total_tokens,
        "failures": failures,
        "fallbacks": fallbacks,
        "by_day": sorted(((k, v[0], v[1]) for k, v in by_day.items()), reverse=True),
        "by_user": _sum_group(rows, "user_id")[:10],
        "by_provider": _sum_group(rows, "provider"),
        "by_agent": _sum_group(rows, "agent_key")[:10],
        "persistent": settings.db_enabled,
    }


async def provider_health(hours: int = 24) -> list[tuple[str, int, float, int]]:
    """[(provider, calls, success_rate, median_latency_ms)] — surfaced in
    /status so a quietly-failing key is visible before it matters."""
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    rows = await _rows_since(since)
    agg: dict[str, list] = {}
    for r in rows:
        slot = agg.setdefault(str(r["provider"]), [0, 0, []])
        slot[0] += 1
        if r["ok"]:
            slot[1] += 1
        slot[2].append(int(r["latency_ms"] or 0))

    out = []
    for provider, (calls, ok, lats) in agg.items():
        lats.sort()
        median = lats[len(lats) // 2] if lats else 0
        out.append((provider, calls, (ok / calls * 100) if calls else 0.0, median))
    return sorted(out, key=lambda x: x[1], reverse=True)
