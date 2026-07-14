"""
Cross-channel conversation memory, keyed by Telegram user_id — the single
place where "what has been said to/by this specific PERSON, regardless of
which channel they used" lives.

Why this exists: a real person can reach the admin through THREE separate
surfaces this bot handles — the main bot's private chat (AI answers
directly), the admin's own Telegram Business chat (AI drafts a reply, the
admin sends it as themselves), and a group mention (same, but in a group).
Before this module, those three were fully isolated: the AI answering a
user directly in the main bot chat had no idea the admin had personally
already promised them something in a Business chat, and vice versa — a
real risk of the bot contradicting the admin, or the admin's suggested
reply contradicting what the bot already told that person. Every channel
now appends to and reads from this ONE store per user_id, so answers stay
consistent no matter which door the person walked through.

NOT a replacement for user_profile.py (durable BEHAVIORAL notes — "this
person is terse, technical") or memory.py (durable PROJECT facts). This is
a bounded ROLLING window of the actual recent exchange, because staying
consistent needs the real wording ("biz 15-iyulgacha deb aytdik"), not a
lossy summary of it.

STORAGE: Redis (rolling list per user_id, capped, TTL'd — same pattern as
history.py's conversation windows), in-memory fallback. This is
short-lived cross-channel CONTEXT, not a durable business record, so it
deliberately does NOT go to Postgres — see db.py's module docstring for
that distinction.
"""

import json
import logging
from collections import deque

import redis_client

logger = logging.getLogger(__name__)

_CAP = 12
_TTL_SECONDS = 60 * 60 * 24 * 14  # 2 weeks: long enough to matter for a slow-moving deal, short enough to stay relevant

_mem: dict[int, deque] = {}

_SPEAKER_LABELS = {"admin": "Admin", "user": "Mijoz/Foydalanuvchi", "bot": "Bot (AI)"}
_CHANNEL_LABELS = {"business": "shaxsiy(Business) chat", "group": "guruh", "main_bot": "bot bilan to'g'ridan-to'g'ri chat"}


def _key(user_id: int) -> str:
    return f"shared_ctx:{user_id}"


async def append(user_id: int, speaker: str, text: str, channel: str) -> None:
    """speaker: 'admin' | 'user' | 'bot'. channel: 'business' | 'group' |
    'main_bot' — kept alongside the text so injected context reads
    naturally ("[shaxsiy chat] Admin: ...") instead of an unlabeled blob."""
    text = (text or "").strip()
    if not text:
        return
    entry = json.dumps({"speaker": speaker, "text": text[:800], "channel": channel})
    client = redis_client.get_client()
    if client is None:
        _mem.setdefault(user_id, deque(maxlen=_CAP)).append(entry)
        return
    try:
        key = _key(user_id)
        async with client.pipeline(transaction=True) as pipe:
            pipe.rpush(key, entry)
            pipe.ltrim(key, -_CAP, -1)
            pipe.expire(key, _TTL_SECONDS)
            await pipe.execute()
    except Exception:  # noqa: BLE001
        logger.exception("Redis shared_context.append failed")
        _mem.setdefault(user_id, deque(maxlen=_CAP)).append(entry)


async def get_recent(user_id: int) -> list[dict]:
    client = redis_client.get_client()
    if client is None:
        return [json.loads(e) for e in _mem.get(user_id, [])]
    try:
        raw = await client.lrange(_key(user_id), 0, -1)
        return [json.loads(e) for e in raw]
    except Exception:  # noqa: BLE001
        logger.exception("Redis shared_context.get_recent failed")
        return [json.loads(e) for e in _mem.get(user_id, [])]


def render_for_prompt(entries: list[dict], exclude_channel: str | None = None) -> str:
    """Render entries as a labelled transcript block for injection into an
    LLM prompt. `exclude_channel` skips entries FROM the channel that's
    about to answer (it already has its own in-channel history — this
    block is specifically for what happened on the OTHER channels)."""
    rows = [e for e in entries if not exclude_channel or e.get("channel") != exclude_channel]
    if not rows:
        return ""
    lines = []
    for e in rows[-8:]:
        who = _SPEAKER_LABELS.get(e.get("speaker"), e.get("speaker", "?"))
        ch = _CHANNEL_LABELS.get(e.get("channel"), e.get("channel", ""))
        lines.append(f"[{ch}] {who}: {e.get('text', '')}")
    return "\n".join(lines)
