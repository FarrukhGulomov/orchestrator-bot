"""
Live web search — the bot's window onto the present.

WHY THIS IS THE FIRST "GENERAL ASSISTANT" FEATURE: without it every answer
comes from the model's frozen training data, so anything time-sensitive
("dollar kursi", "ob-havo", "bu texnologiya haqida nima deyishadi",
"raqiblar narxi qancha") is either refused or — worse — answered
confidently from stale memory. That single gap is what keeps the bot a
professional-advice tool rather than an everyday assistant.

Provider: Tavily (api.tavily.com) — a search API built for LLM
consumption: it returns clean text snippets plus its own short synthesized
answer, instead of raw HTML that would then need scraping. Free tier is
1000 searches/month with no credit card. Deliberately NOT OpenRouter's
web-search plugin, which bills ~$0.02 per request — this bot's owner is
cost-sensitive and has hit paid-quota walls before.

Config-gated exactly like github_integration/railway_integration: with no
TAVILY_API_KEY the module reports itself disabled and every caller carries
on normally (answers just aren't web-grounded). Failures return a real
error string rather than raising, so the reason reaches the user instead
of the server log — the same error-surfacing contract used by the
copilots and /minutes.
"""

import asyncio
import logging

import requests

from config import settings

logger = logging.getLogger(__name__)

_ENDPOINT = "https://api.tavily.com/search"
_TIMEOUT = 20

# Per-result snippet cap. Tavily snippets are already short, but a handful
# of long ones would still crowd out the conversation history in the
# agent's context window.
_MAX_SNIPPET_CHARS = 700
_MAX_RESULTS = 5


def enabled() -> bool:
    return bool(settings.tavily_api_key)


def _search_sync(query: str, max_results: int) -> dict:
    resp = requests.post(
        _ENDPOINT,
        json={
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",   # 1 credit; "advanced" costs 2 for marginal gain here
            "include_answer": "basic",  # Tavily's own short synthesis — cheap and often enough on its own
        },
        headers={
            "Authorization": f"Bearer {settings.tavily_api_key}",
            "Content-Type": "application/json",
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


async def search(query: str, max_results: int = _MAX_RESULTS) -> tuple[dict | None, str | None]:
    """Returns (payload, error_message). payload has:
        {"answer": str, "results": [{"title", "url", "content"}, ...]}
    Never raises — a search failure must degrade the answer, not kill it."""
    query = (query or "").strip()
    if not query:
        return None, "Bo'sh so'rov."
    if not enabled():
        return None, "Internetdan izlash sozlanmagan (TAVILY_API_KEY kerak)."

    try:
        raw = await asyncio.to_thread(_search_sync, query, max_results)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Web search failed for query=%.80s", query)
        return None, f"Izlashda xatolik: {str(exc)[:200]}"

    results = []
    for item in (raw.get("results") or [])[:max_results]:
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        content = str(item.get("content") or "").strip()[:_MAX_SNIPPET_CHARS]
        if title or content:
            results.append({"title": title, "url": url, "content": content})

    answer = str(raw.get("answer") or "").strip()
    if not results and not answer:
        return None, "Natija topilmadi."
    return {"answer": answer, "results": results}, None


def render_for_prompt(payload: dict) -> str:
    """Format search output as a context block for injection into an
    agent's system prompt. Includes source URLs so the agent can cite what
    it relied on — an ungrounded "the rate is X" is worse than no answer,
    since the user can't check it."""
    if not payload:
        return ""
    lines = [
        "\nLIVE WEB SEARCH RESULTS (fetched seconds ago — for anything "
        "time-sensitive TRUST THESE over your training data, and cite the "
        "source URL for any figure/date/claim you take from them. If they "
        "don't actually answer the question, say so plainly instead of "
        "guessing):",
    ]
    if payload.get("answer"):
        lines.append(f"Qisqa xulosa: {payload['answer']}")
    for i, r in enumerate(payload.get("results") or [], 1):
        lines.append(f"{i}. {r.get('title', '')} — {r.get('url', '')}\n   {r.get('content', '')}")
    return "\n".join(lines) + "\n"


def render_for_user(payload: dict) -> str:
    """Human-readable rendering for the explicit /search command."""
    lines = []
    if payload.get("answer"):
        lines.append(f"🔎 {payload['answer']}\n")
    for i, r in enumerate(payload.get("results") or [], 1):
        title = r.get("title") or r.get("url") or "—"
        lines.append(f"{i}. {title}\n{r.get('url', '')}")
    return "\n".join(lines) if lines else "Natija topilmadi."
