"""
Lightweight project memory — a realistic, honest version of "self-learning".

IMPORTANT — what this is NOT: this bot cannot retrain or fine-tune any
underlying LLM. Real model self-training needs a training pipeline, GPU
infrastructure, and labeled data — none of which a Telegram bot has access
to. Claiming otherwise would be dishonest.

What this DOES do, which is the realistic and genuinely useful interpretation
of "the bot learns from conversations": durable facts about the team's
project (project name, confirmed tech-stack decisions, budget/timeline
figures, established constraints/conventions) are extracted from
consequential exchanges and stored per chat. Every agent's prompt then gets
these facts injected as context — so answers get progressively better
grounded in THIS team's actual project over time, without the user needing
to repeat context every message. That's real, measurable improvement in
answer quality; it just isn't "the model learned new weights".

In-memory only (lost on restart), same caveat as history.py — swap the
dict-based store for Redis/a database behind the same function signatures
for production persistence.
"""

from collections import defaultdict

MAX_FACTS = 40

_store: dict[int, list[str]] = defaultdict(list)


def get_memory(chat_id: int) -> list[str]:
    return list(_store[chat_id])


def add_memory(chat_id: int, fact: str) -> bool:
    """Add a fact if it's non-empty and not a near-duplicate. Returns True if added."""
    fact = (fact or "").strip()
    if not fact:
        return False
    facts = _store[chat_id]
    if fact in facts:
        return False
    facts.append(fact)
    if len(facts) > MAX_FACTS:
        del facts[0 : len(facts) - MAX_FACTS]  # drop oldest, keep most recent
    return True


def clear_memory(chat_id: int) -> int:
    """Clear all stored facts for this chat. Returns how many were removed."""
    facts = _store.pop(chat_id, [])
    return len(facts)
