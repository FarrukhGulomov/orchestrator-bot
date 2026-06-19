"""
Very small in-memory conversation store, keyed by Telegram chat id.

Keeps the last N turns so ROUTE A (analysis) agents get useful context. This is
intentionally simple (a dict in RAM): history is lost on restart. For production,
swap _store for Redis or a database behind the same get/append interface.
"""

from collections import defaultdict, deque

from config import settings

# chat_id -> deque of {"role": "user"|"assistant", "content": str}
_store: dict[int, deque] = defaultdict(
    lambda: deque(maxlen=settings.history_turns * 2)
)


def get_history(chat_id: int) -> list[dict]:
    return list(_store[chat_id])


def append(chat_id: int, role: str, content: str) -> None:
    _store[chat_id].append({"role": role, "content": content})


def reset(chat_id: int) -> None:
    _store.pop(chat_id, None)
