"""
Personal expense-tracking commands — /xarajat, /xarajatlar, /xarajatochir,
/byudjet — plus the shared parse-and-record helper the natural-language
auto-capture path in bot.py also calls.

First extraction out of bot.py (engineering audit, Phase 2: "decompose
bot.py incrementally, one handler group per PR, behaviour-preserving").
Picked as the first slice because it's fully self-contained — these
commands only touch expenses.py, never the routing/agent machinery that
makes the rest of bot.py harder to split apart safely.

Registered into the bot via bot.py's dp.include_router(router) — an
aiogram Router scoped to this module, not the global Dispatcher, is what
actually breaks the "every handler lives on one shared object" coupling
that produced the 3,600-line bot.py in the first place. Every future
handler group extraction should follow the same shape: its own Router,
included once from bot.py.

Nothing about the LOGIC below changed from what lived in bot.py — this is
a move, not a rewrite.
"""

from aiogram import Router
from aiogram.filters import Command, CommandObject
from aiogram.types import Message

import expenses
from bot_utils import is_allowed, send_long

router = Router(name="expenses")


async def try_log_expense(message: Message, uid: int, chat_id: int, text: str) -> bool:
    """Parse and store a spend mention. Returns True if it was recorded (so
    the caller skips the normal AI pipeline). Any failure returns False and
    the message is answered normally — a missed capture is a small loss, a
    hijacked conversation is a big one."""
    parsed = await expenses.extract(text)
    if not parsed:
        return False
    expense_id = await expenses.add(
        uid, chat_id, parsed["amount"], parsed["currency"], parsed["category"], parsed["note"],
    )
    if expense_id is None:
        return False
    await message.answer(
        expenses.render_confirmation(
            parsed["amount"], parsed["currency"], parsed["category"], parsed["note"], expense_id,
        ),
        parse_mode="Markdown",
    )
    alert = await expenses.check_budget_alert(uid)
    if alert:
        await message.answer(alert)
    return True


@router.message(Command("xarajat", "expense"))
async def cmd_expense(message: Message, command: CommandObject) -> None:
    """Explicit expense entry — the same parser as the automatic capture,
    for when the user wants to be sure it's recorded as a spend."""
    chat_id = message.chat.id
    if not is_allowed(chat_id):
        return
    if not expenses.available():
        await message.answer(
            "💸 Xarajat hisobi uchun PostgreSQL kerak (moliyaviy yozuvlar "
            "restartda yo'qolmasligi uchun).\nRailway'da DATABASE_URL qo'shing."
        )
        return
    text = (command.args or "").strip()
    if not text:
        await message.answer(
            "Xarajatni yozing:\n/xarajat taksiga 30 ming\n\n"
            "Yoki shunchaki oddiy xabar sifatida yozavering — o'zim tushunaman.\n"
            "Hisobot: /xarajatlar"
        )
        return
    uid = message.from_user.id if message.from_user else 0
    if not await try_log_expense(message, uid, chat_id, text):
        await message.answer(
            "Buni xarajat sifatida tushunolmadim. Masalan: /xarajat obed 45 ming"
        )


@router.message(Command("xarajatlar", "expenses"))
async def cmd_expenses(message: Message, command: CommandObject) -> None:
    chat_id = message.chat.id
    if not is_allowed(chat_id):
        return
    if not expenses.available():
        await message.answer(
            "💸 Xarajat hisobi uchun PostgreSQL kerak — Railway'da DATABASE_URL qo'shing."
        )
        return
    period = (command.args or "").strip().lower() or "oy"
    since, until, label = expenses.period_bounds(period)
    uid = message.from_user.id if message.from_user else 0
    data = await expenses.summary(uid, since, until)
    if data is None:
        await message.answer("⚠️ Hisobotni olishda xatolik. Keyinroq urinib ko'ring.")
        return
    await send_long(message, expenses.render_summary(data, label))


@router.message(Command("xarajatochir", "delexpense"))
async def cmd_delete_expense(message: Message, command: CommandObject) -> None:
    chat_id = message.chat.id
    if not is_allowed(chat_id):
        return
    arg = (command.args or "").strip().split()[0] if command.args else ""
    if not arg.isdigit():
        await message.answer("O'chirish uchun ID yozing: /xarajatochir 42\n(ID lar /xarajatlar da)")
        return
    uid = message.from_user.id if message.from_user else 0
    if await expenses.delete(uid, int(arg)):
        await message.answer("🗑 O'chirildi.")
    else:
        await message.answer("Bunday yozuv topilmadi (yoki u sizniki emas).")


@router.message(Command("byudjet", "budget"))
async def cmd_budget(message: Message, command: CommandObject) -> None:
    """Set or view a monthly spending limit. Alerts fire automatically from
    the expense-capture path (try_log_expense) once 80%/100% is crossed."""
    chat_id = message.chat.id
    if not is_allowed(chat_id):
        return
    if not expenses.available():
        await message.answer(
            "💸 Byudjet uchun PostgreSQL kerak — Railway'da DATABASE_URL qo'shing."
        )
        return
    uid = message.from_user.id if message.from_user else 0
    arg = (command.args or "").strip()
    if arg:
        amount = expenses.parse_amount_shorthand(arg)
        if not amount:
            await message.answer("Byudjet miqdorini yozing, masalan: /byudjet 3 mln")
            return
        await expenses.set_budget(uid, amount)
        await message.answer(f"✅ Oylik byudjet o'rnatildi: {expenses.fmt_amount(amount, 'UZS')}")
        return
    budget = await expenses.get_budget(uid)
    if not budget:
        await message.answer(
            "Oylik byudjet o'rnatilmagan.\nMasalan: /byudjet 3 mln — 80% va 100% da ogohlantiraman."
        )
        return
    spent = await expenses.month_to_date_uzs_total(uid)
    pct = int(spent / budget * 100) if budget else 0
    remaining = budget - spent
    lines = [
        f"💰 Oylik byudjet: {expenses.fmt_amount(budget, 'UZS')}",
        f"💸 Sarflandi: {expenses.fmt_amount(spent, 'UZS')} ({pct}%)",
    ]
    if remaining >= 0:
        lines.append(f"✅ Qoldi: {expenses.fmt_amount(remaining, 'UZS')}")
    else:
        lines.append(f"🔴 Oshib ketdi: {expenses.fmt_amount(-remaining, 'UZS')}")
    await message.answer("\n".join(lines))
