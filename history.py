"""
Conversation history store, scoped per (chat_id, agent_key).

WHY SCOPED PER AGENT (this fixes a real bug):
Each agent has a different persona/system prompt. If every agent shared one
history queue per chat, switching agents mid-conversation could leak the
PREVIOUS agent's context into the NEW agent's prompt — and the model tends to
keep following that prior context over the freshly-injected system prompt.
Observed symptom: a message classified as SOC / BUG returned a Product
Manager-style answer (user interviews, personas), because the shared history
still carried an earlier PM/idea conversation. Scoping history by agent_key
keeps each role's thread clean while the SAME agent still keeps context
across its own turns.

Also tracks the (agent_key, request_type_key) of the last successfully
answered turn per chat. The router uses this to keep routing short,
content-free follow-ups ("ok", "continue", "davom et", "zo'r") to the SAME
agent instead of re-guessing from a message that has no real signal.

In-memory only (lost on restart). For production, swap the dict-based stores
for Redis/a database behind the same function signatures.
"""

from collections import defaultdict, deque

from config import settings

# (chat_id, agent_key) -> deque of {"role": "user"|"assistant", "content": str}
_store: dict[tuple[int, str], deque] = defaultdict(
    lambda: deque(maxlen=settings.history_turns * 2)
)

# chat_id -> (agent_key, request_type_key) of the last answered turn
_last_route: dict[int, tuple[str, str]] = {}


def get_history(chat_id: int, agent_key: str) -> list[dict]:
    return list(_store[(chat_id, agent_key)])


def append(chat_id: int, agent_key: str, role: str, content: str) -> None:
    _store[(chat_id, agent_key)].append({"role": role, "content": content})


def reset(chat_id: int) -> None:
    """Clear every agent's thread (and the last-route hint) for this chat."""
    for key in [k for k in list(_store.keys()) if k[0] == chat_id]:
        _store.pop(key, None)
    _last_route.pop(chat_id, None)


def get_last_route(chat_id: int) -> tuple[str, str] | None:
    return _last_route.get(chat_id)


def set_last_route(chat_id: int, agent_key: str, request_type_key: str) -> None:
    _last_route[chat_id] = (agent_key, request_type_key)
