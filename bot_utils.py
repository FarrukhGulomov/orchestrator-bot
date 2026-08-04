"""
Telegram-facing helpers with no dependency on bot.py's own state — the
first extraction out of bot.py (see the engineering audit's Phase 2:
"decompose bot.py incrementally, one handler group per PR"). Every future
handlers/*.py module needs these, so they had to move somewhere both
bot.py and those modules can import without a circular dependency; this
is that somewhere.

Behavior-preserving: nothing here changed, only moved. bot.py imports
these under their original names so every existing call site is untouched.
"""

from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Message

from config import settings

TELEGRAM_LIMIT = 4096


def is_allowed(chat_id: int) -> bool:
    return not settings.allowed_chat_ids or chat_id in settings.allowed_chat_ids


def split_message(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, current = [], ""
    for line in text.split("\n"):
        while len(line) > limit:
            if current:
                parts.append(current)
                current = ""
            parts.append(line[:limit])
            line = line[limit:]
        if len(current) + len(line) + 1 > limit:
            if current:  # never emit an empty chunk (Telegram rejects empty text)
                parts.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        parts.append(current)
    return parts


async def send_long(message: Message, text: str, reply_mode: bool = False) -> Message | None:
    """Send a (possibly long) reply, splitting at TELEGRAM_LIMIT. If reply_mode,
    the first chunk quotes the original message so group context is clear.
    Returns the LAST sent Message — callers can attach an inline keyboard to
    it (quick actions) via edit_reply_markup."""
    chunks = split_message(text, TELEGRAM_LIMIT)
    sent: Message | None = None
    for i, chunk in enumerate(chunks):
        use_reply = reply_mode and i == 0
        try:
            if use_reply:
                sent = await message.reply(chunk, parse_mode="Markdown")
            else:
                sent = await message.answer(chunk, parse_mode="Markdown")
        except TelegramBadRequest:
            if use_reply:
                sent = await message.reply(chunk, parse_mode=None)
            else:
                sent = await message.answer(chunk, parse_mode=None)
    return sent
