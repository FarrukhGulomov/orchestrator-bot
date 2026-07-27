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
import re
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


_SHORTHAND_NUM_RE = re.compile(r"(\d[\d\s]*(?:[.,]\d+)?)")


def parse_amount_shorthand(text: str) -> Decimal | None:
    """Deterministic (no LLM call) amount parser for shorthand like '3 mln',
    '30 ming', '3000000', '3 000 000'. Used where a bare number is expected
    on its own (e.g. /byudjet) — an LLM round-trip would be wasteful for
    something this mechanical, and less reliable than a direct parse."""
    low = (text or "").lower()
    m = _SHORTHAND_NUM_RE.search(low)
    if not m:
        return None
    raw = m.group(1).strip().replace(" ", "").replace(",", ".")
    try:
        num = Decimal(raw)
    except InvalidOperation:
        return None
    tail = low[m.end():m.end() + 15]
    if "mln" in tail or "million" in tail or "млн" in tail or "миллион" in tail:
        num *= 1_000_000
    elif "ming" in tail or "минг" in tail or "тыс" in tail:
        num *= 1_000
    return num if num > 0 else None


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


async def search_notes(user_id: int, query: str, limit: int = 8) -> list[dict]:
    """Substring search over a user's own expense notes/categories — backs
    the cross-feature /qidir command. Scoped by user_id like delete()."""
    pool = await _pool()
    if pool is None or not query.strip():
        return []
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, amount, currency, category, note, spent_at
                FROM expenses
                WHERE user_id = $1 AND (note ILIKE $2 OR category ILIKE $2)
                ORDER BY spent_at DESC LIMIT $3
                """,
                user_id, f"%{query.strip()}%", limit,
            )
        return [dict(r) for r in rows]
    except Exception:  # noqa: BLE001
        logger.exception("Failed to search expenses for user=%s", user_id)
        return []


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


# --------------------------------------------------------------------------
# Monthly budget + proactive alerts
#
# A single overall monthly limit per user (not per-category — a category
# breakdown is for the report, a limit is a simple "am I okay?" signal, and
# a per-category budget UI is a lot of surface for a marginal gain here).
# Stored in kv_store since it is exactly one scalar per user, not a table.
# UZS only: the categories/report already treat UZS as the default currency
# and mixing currencies into one limit comparison would be meaningless.
# --------------------------------------------------------------------------
_ALERT_THRESHOLDS = (100, 80)  # checked highest-first so a big jump alerts once, at the higher milestone


def _budget_key(user_id: int) -> str:
    return f"budget:{user_id}"


def _alert_key(user_id: int, month_label: str) -> str:
    return f"budget_alert:{user_id}:{month_label}"


async def get_budget(user_id: int) -> Decimal | None:
    raw = await db.kv_get(_budget_key(user_id))
    if not raw:
        return None
    try:
        return Decimal(raw)
    except InvalidOperation:
        return None


async def set_budget(user_id: int, amount: Decimal) -> None:
    await db.kv_set(_budget_key(user_id), str(amount))


async def month_to_date_uzs_total(user_id: int) -> Decimal:
    since, until, _ = period_bounds("oy")
    data = await summary(user_id, since, until)
    if not data:
        return Decimal(0)
    total = Decimal(0)
    for row in data["by_category"]:
        if row["currency"] == "UZS":
            total += Decimal(row["total"])
    return total


async def check_budget_alert(user_id: int) -> str | None:
    """Call after logging an expense. Returns an alert message the FIRST
    time this calendar month that spend crosses 80%/100% of the budget, and
    None every other time — so a user who logs 10 expenses past the limit
    gets nudged once, not 10 times."""
    budget = await get_budget(user_id)
    if not budget or budget <= 0:
        return None
    spent = await month_to_date_uzs_total(user_id)
    pct = int(spent / budget * 100)

    month_label = tasks.now_local().strftime("%Y-%m")
    alert_raw = await db.kv_get(_alert_key(user_id, month_label))
    already_notified = int(alert_raw) if alert_raw else 0

    for threshold in _ALERT_THRESHOLDS:
        if pct >= threshold and already_notified < threshold:
            await db.kv_set(_alert_key(user_id, month_label), str(threshold))
            if threshold >= 100:
                return (
                    f"🔴 Bu oy byudjetdan oshib ketdingiz: {fmt_amount(spent, 'UZS')} / "
                    f"{fmt_amount(budget, 'UZS')} ({pct}%)."
                )
            return (
                f"🟠 Bu oy byudjetning {pct}% i sarflandi: {fmt_amount(spent, 'UZS')} / "
                f"{fmt_amount(budget, 'UZS')}."
            )
    return None


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
