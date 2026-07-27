"""
Personal expense tracking — the highest-frequency everyday-assistant
feature, and one that needs no external service at all.

WHY THIS ONE: the bot's daily-value problem is a FREQUENCY problem. Most
of its capabilities fire occasionally (a proposal, a meeting protocol);
money is spent every single day. Capturing that in one line — "taksiga 30
ming" — gives a reason to open the bot daily, and produces something the
user can't easily get otherwise: an honest monthly breakdown of where
their money actually went.

Zero setup: no API key, no OAuth. It uses the Postgres database that is
already configured, plus the fast/free model for parsing.

STORAGE: PostgreSQL ONLY — deliberately no in-memory fallback. Everywhere
else in this codebase a missing backend degrades to memory, but silently
losing someone's financial records (on the next redeploy, with no warning)
is worse than not offering the feature: the user would keep logging
expenses and only discover months later that the history is gone. So when
DATABASE_URL is unset, this module reports itself unavailable and says why.

Structure mirrors tasks.py/task_assistant.py: a data layer, then the LLM
extraction that turns free text into a row, then formatting. Kept in one
module because it is a fraction of their size.
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation

import db
import tasks  # for TZ / now_local — expense dates are local-calendar dates
from llm_clients import generate_json

logger = logging.getLogger(__name__)

# A fixed, small category set. Free-form categories would fragment the
# monthly report into dozens of near-duplicates ("taksi" / "taxi" /
# "transport"), which is exactly what makes hand-kept expense lists useless.
CATEGORIES = [
    "oziq-ovqat",      # food, groceries, cafe
    "transport",       # taxi, fuel, public transport
    "uy",              # rent, utilities, household
    "sog'liq",         # medicine, doctor
    "kiyim",           # clothing
    "ko'ngilochar",    # entertainment, eating out for fun
    "aloqa",           # phone, internet, subscriptions
    "ta'lim",          # courses, books
    "sovg'a",          # gifts, family support
    "ish",             # work-related spending
    "boshqa",
]

_CATEGORY_EMOJI = {
    "oziq-ovqat": "🍽", "transport": "🚕", "uy": "🏠", "sog'liq": "💊",
    "kiyim": "👕", "ko'ngilochar": "🎬", "aloqa": "📱", "ta'lim": "📚",
    "sovg'a": "🎁", "ish": "💼", "boshqa": "📦",
}

# Cheap pre-filter before spending an LLM call: a real expense mention
# always carries a NUMBER plus either a money word or a spending verb.
_MONEY_WORDS = (
    "so'm", "som", "sum", "ming", "mln", "million", "dollar", "usd", "$",
    "yevro", "eur", "rubl",
)
_SPEND_WORDS = (
    "ketdi", "sarfladim", "sarfladik", "to'ladim", "toladim", "tuladim",
    "oldim", "sotib", "xarajat", "chiqim", "harajat", "потратил", "купил",
    "заплатил",
)


def available() -> bool:
    """Expenses require a real database — see the module docstring."""
    from config import settings
    return settings.db_enabled


def looks_like_expense(text: str) -> bool:
    """Keyword/number pre-filter — keeps the LLM extraction off the ~99% of
    messages that are plainly not expense mentions."""
    low = (text or "").lower()
    if not any(ch.isdigit() for ch in low):
        return False
    if len(low) > 200:  # a long paragraph that happens to contain a number isn't a spend log
        return False
    return any(w in low for w in _MONEY_WORDS) or any(w in low for w in _SPEND_WORDS)


_EXTRACT_SYSTEM = f"""
You extract a PERSONAL EXPENSE from a short message (Uzbek, Russian, or
English). The user is logging money they ALREADY SPENT.

Respond with ONLY this JSON, no prose, no markdown fences:
{{"is_expense": true|false,
  "amount": <number, in the base currency unit — see the multiplier rule>,
  "currency": "UZS"|"USD"|"EUR"|"RUB",
  "category": "<one of: {", ".join(CATEGORIES)}>",
  "note": "<what it was for, 1-4 words, in the user's own language>"}}

MULTIPLIER RULE (critical — Uzbek speech shortens amounts):
- "30 ming" / "30к" / "30 тыс" = 30000
- "1.5 mln" / "1,5 million" = 1500000
- "30 000" / "30000" = 30000
Return the FULL numeric value, never the shortened form.

CURRENCY: default "UZS" when unmarked (so'm/sum/ming). Use "USD" only for
an explicit $/dollar/USD mention.

is_expense=false when the message is:
- a FUTURE or PLANNED payment ("ertaga 500 ming to'lashim kerak",
  "obunani uzaytirish kerak") — that is a task, not a spend record
- a question about money ("bu qancha turadi?", "narxi qancha?")
- income, a price quote, a salary figure, or any number that is not the
  user's own completed spending
When in doubt, return false — a wrongly recorded expense is worse than a
missed one, because the user has to find and delete it.
"""


async def extract(text: str) -> dict | None:
    """Free text -> a normalised expense dict, or None if this isn't one."""
    try:
        data = await generate_json(
            _EXTRACT_SYSTEM, [{"role": "user", "content": text[:500]}], max_tokens=250, retries=1,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Expense extraction failed")
        return None
    if not data.get("is_expense"):
        return None
    try:
        amount = Decimal(str(data.get("amount", 0)))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if amount <= 0 or amount >= Decimal("1000000000000"):
        return None
    category = str(data.get("category") or "boshqa")
    if category not in CATEGORIES:
        category = "boshqa"
    currency = str(data.get("currency") or "UZS").upper()[:4]
    return {
        "amount": amount,
        "currency": currency,
        "category": category,
        "note": str(data.get("note") or "")[:120],
    }


# --------------------------------------------------------------------------
# Data layer
# --------------------------------------------------------------------------
async def _pool():
    if await db.init_schema():
        return await db.get_pool()
    return None


async def add(user_id: int, chat_id: int, amount: Decimal, currency: str,
              category: str, note: str) -> int | None:
    pool = await _pool()
    if pool is None:
        return None
    try:
        async with pool.acquire() as conn:
            return await conn.fetchval(
                """
                INSERT INTO expenses (user_id, chat_id, amount, currency, category, note, spent_at)
                VALUES ($1, $2, $3, $4, $5, $6, now())
                RETURNING id
                """,
                user_id, chat_id, amount, currency, category, note,
            )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to save expense for user=%s", user_id)
        return None


async def delete(user_id: int, expense_id: int) -> bool:
    """user_id is part of the WHERE clause on purpose — an id alone must
    never let one user delete another's record."""
    pool = await _pool()
    if pool is None:
        return False
    try:
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM expenses WHERE id = $1 AND user_id = $2", expense_id, user_id,
            )
        return result.endswith("1")
    except Exception:  # noqa: BLE001
        logger.exception("Failed to delete expense id=%s", expense_id)
        return False


async def summary(user_id: int, since: datetime, until: datetime) -> dict | None:
    """Totals per (currency, category) plus the most recent entries."""
    pool = await _pool()
    if pool is None:
        return None
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT currency, category, SUM(amount) AS total, COUNT(*) AS n
                FROM expenses
                WHERE user_id = $1 AND spent_at >= $2 AND spent_at < $3
                GROUP BY currency, category
                ORDER BY total DESC
                """,
                user_id, since, until,
            )
            recent = await conn.fetch(
                """
                SELECT id, amount, currency, category, note, spent_at
                FROM expenses
                WHERE user_id = $1 AND spent_at >= $2 AND spent_at < $3
                ORDER BY spent_at DESC LIMIT 10
                """,
                user_id, since, until,
            )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to build expense summary for user=%s", user_id)
        return None
    return {
        "by_category": [dict(r) for r in rows],
        "recent": [dict(r) for r in recent],
    }


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------
def fmt_amount(amount: Decimal, currency: str) -> str:
    """1234567 UZS -> '1 234 567 so'm'. Thousands separated by spaces, which
    is how amounts are written locally; no decimals for whole sums."""
    q = Decimal(amount)
    whole = q.quantize(Decimal("1")) if q == q.to_integral_value() else q.quantize(Decimal("0.01"))
    text = f"{whole:,}".replace(",", " ")
    unit = {"UZS": "so'm", "USD": "$", "EUR": "€", "RUB": "₽"}.get(currency, currency)
    return f"{text} {unit}"


def period_bounds(period: str) -> tuple[datetime, datetime, str]:
    """(since, until, label) for 'bugun' | 'hafta' | 'oy' (default)."""
    now = tasks.now_local()
    start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow = start_of_today + timedelta(days=1)
    if period == "bugun":
        return start_of_today, tomorrow, "Bugun"
    if period == "hafta":
        return start_of_today - timedelta(days=now.weekday()), tomorrow, "Shu hafta"
    return start_of_today.replace(day=1), tomorrow, "Shu oy"


def render_confirmation(amount: Decimal, currency: str, category: str, note: str, expense_id: int) -> str:
    emoji = _CATEGORY_EMOJI.get(category, "📦")
    line = f"{emoji} Yozildi: {fmt_amount(amount, currency)} — {category}"
    if note:
        line += f" ({note})"
    return line + f"\n_O'chirish: /xarajatochir {expense_id}_"


def render_summary(data: dict, label: str) -> str:
    by_cat = data.get("by_category") or []
    if not by_cat:
        return f"{label}: xarajat yozilmagan.\n\nYozish uchun shunchaki yozing: \"taksiga 30 ming\""

    lines = [f"💸 {label} xarajatlar\n"]
    totals: dict[str, Decimal] = {}
    for row in by_cat:
        cur = row["currency"]
        totals[cur] = totals.get(cur, Decimal(0)) + Decimal(row["total"])
        emoji = _CATEGORY_EMOJI.get(row["category"], "📦")
        lines.append(
            f"{emoji} {row['category']}: {fmt_amount(Decimal(row['total']), cur)} ({row['n']} ta)"
        )
    lines.append("")
    for cur, total in totals.items():
        lines.append(f"*Jami: {fmt_amount(total, cur)}*")

    recent = data.get("recent") or []
    if recent:
        lines.append("\nOxirgilari:")
        for r in recent[:5]:
            when = r["spent_at"].astimezone(tasks.TZ).strftime("%d-%m")
            note = f" — {r['note']}" if r["note"] else ""
            lines.append(
                f"`{r['id']}` {when} · {fmt_amount(Decimal(r['amount']), r['currency'])} "
                f"· {r['category']}{note}"
            )
    return "\n".join(lines)
