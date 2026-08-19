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
from llm_clients import claude_generate_fast

logger = logging.getLogger(__name__)

MAX_DECISIONS = 200

# Cheap pre-filter, same shape as task_assistant.looks_like_task /
# expenses.looks_like_expense — a settled, past-tense decision has a
# fairly distinctive vocabulary (unlike a plain statement of fact or a
# request), so this rarely collides with the task/expense/memory triggers
# that run in the same message pipeline.
_TRIGGER_WORDS = (
    # Uzbek
    "qaror qildik", "qaror qildim", "qaror bo'ldi", "kelishildik",
    "kelishib oldik", "shunday deb kelishdik", "qaror qabul qil",
    # Russian
    "решили", "договорились", "приняли решение", "решение принято",
    # English
    "we decided", "we've decided", "decided to", "agreed to",
    "it's settled", "final decision",
)


def looks_like_decision(text: str) -> bool:
    low = (text or "").lower()
    if len(low) > 500:  # a long paragraph isn't a one-line decision statement
        return False
    return any(w in low for w in _TRIGGER_WORDS)


_EXTRACT_SYSTEM = """
The user's message states a decision that was made (about their project,
team, or work) — not a question, not a task to do later, not money spent.
Extract it as ONE dated-log entry.

Respond with ONLY the entry text (imperative/past statement, same language
as the user, max 200 chars, no prefix/quotes) — e.g. "Reliz dushanba kuniga
ko'chirildi" or "Решили использовать PostgreSQL вместо MongoDB".

If the message does NOT actually state a settled decision (it's a question,
a plan still being discussed, a task assignment, or anything else), respond
with exactly NONE.
"""


async def extract_decision(text: str) -> str | None:
    """LLM extraction behind the looks_like_decision() pre-filter — mirrors
    task_assistant/memory's "NONE or the extracted text" pattern. Never
    raises; a failure here should just fall through to the normal chat
    reply, not surface an error for what was maybe just a stray keyword
    match."""
    try:
        raw = await claude_generate_fast(
            _EXTRACT_SYSTEM,
            [{"role": "user", "content": text[:2000]}],
            temperature=0.0,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Decision extraction failed (non-fatal)")
        return None
    entry = (raw or "").strip()
    if not entry or entry.upper() == "NONE" or len(entry) > 300:
        return None
    return entry

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
