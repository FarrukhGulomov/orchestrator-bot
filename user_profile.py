"""
Per-user profile memory — durable notes about an individual approved user,
separate from memory.py's per-CHAT project facts (tech stack, budget,
decisions). Where memory.py answers "what do we know about this project",
this answers "what do we know about this PERSON" — their role, communication
style, technical level, recurring concerns — so the AI team recognizes a
returning user instead of treating every message like a stranger's first
contact.

Scope: PRIVATE-chat, admin-approved, non-admin users only (see bot.py's
_profile_eligible / AccessGateMiddleware) — deliberately not applied to
groups or to the admin's own messages.

Keyed by Telegram user_id (stable across chats, unlike memory.py's
chat_id — which happens to equal user_id in private chats too, but the
name here is honest about what this store is actually about: a person,
not a room).

Backed by Redis when REDIS_URL is configured, with the same in-memory
fallback pattern as the rest of the codebase. No TTL — notes are meant to
accumulate durably until explicitly cleared.
"""

import logging

import redis_client

logger = logging.getLogger(__name__)

MAX_NOTES = 30

# --- In-memory fallback ---
_store: dict[int, list[str]] = {}


def _words(text: str) -> set[str]:
    return {w for w in text.lower().split() if w}


def _is_near_duplicate(note: str, existing: list[str]) -> bool:
    """Same dedup heuristic as memory.py: exact match, or >= 80% word
    overlap with an already-stored note — stops trivially-reworded
    observations piling up ("javoblarni qisqa yozadi" said five different
    ways)."""
    if note in existing:
        return True
    new_words = _words(note)
    if not new_words:
        return True
    for old in existing:
        old_words = _words(old)
        if not old_words:
            continue
        overlap = len(new_words & old_words)
        if overlap / len(new_words) >= 0.8 or overlap / len(old_words) >= 0.8:
            return True
    return False


def _key(user_id: int) -> str:
    return f"user_profile:{user_id}"


async def get_profile(user_id: int) -> list[str]:
    client = redis_client.get_client()
    if client is None:
        return list(_store.get(user_id, []))
    try:
        return await client.lrange(_key(user_id), 0, -1)
    except Exception:  # noqa: BLE001
        logger.exception("Redis get_profile failed, returning empty for this call")
        return []


async def add_profile_note(user_id: int, note: str) -> bool:
    """Add a note if it's non-empty and not a near-duplicate. Returns True if added."""
    note = (note or "").strip()
    if not note:
        return False

    client = redis_client.get_client()
    if client is None:
        notes = _store.setdefault(user_id, [])
        if _is_near_duplicate(note, notes):
            return False
        notes.append(note)
        if len(notes) > MAX_NOTES:
            del notes[0 : len(notes) - MAX_NOTES]
        return True

    try:
        existing = await client.lrange(_key(user_id), 0, -1)
        if _is_near_duplicate(note, existing):
            return False
        async with client.pipeline(transaction=True) as pipe:
            pipe.rpush(_key(user_id), note)
            pipe.ltrim(_key(user_id), -MAX_NOTES, -1)
            await pipe.execute()
        return True
    except Exception:  # noqa: BLE001
        logger.exception("Redis add_profile_note failed")
        return False


async def clear_profile(user_id: int) -> int:
    """Clear all stored notes for this user. Returns how many were removed."""
    client = redis_client.get_client()
    if client is None:
        notes = _store.pop(user_id, [])
        return len(notes)
    try:
        n = await client.llen(_key(user_id))
        if n:
            await client.delete(_key(user_id))
        return n
    except Exception:  # noqa: BLE001
        logger.exception("Redis clear_profile failed")
        return 0
