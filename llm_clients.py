"""
Claude (Anthropic) client wrappers.

All agent calls use the Anthropic SDK. Two tiers:
  - claude_generate()      — full model (Sonnet) for all agent responses
  - claude_generate_fast() — fast model (Haiku) for routing/classification

Retry logic: Claude returns HTTP 529 (Overloaded) during demand spikes.
_claude_sync retries up to 3 times with exponential backoff (2s → 4s → 8s)
before giving up — same pattern as any production API client.
"""

import asyncio
import base64
import logging
import time

from config import settings

logger = logging.getLogger(__name__)

_client = None

# Retry config for Claude 529 Overloaded errors
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2  # seconds; doubles each attempt: 2 → 4 → 8


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def _is_overloaded(exc: Exception) -> bool:
    """True for Claude 529 Overloaded and 503 Service Unavailable."""
    msg = str(exc)
    return "529" in msg or "overloaded" in msg.lower() or "503" in msg


def _claude_sync(
    model: str,
    system: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> str:
    client = _get_client()
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=model,
                system=system,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (resp.content[0].text if resp.content else "").strip()
        except Exception as exc:
            last_exc = exc
            if _is_overloaded(exc) and attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "Claude overloaded (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1, _MAX_RETRIES, delay, str(exc)[:120],
                )
                time.sleep(delay)
                continue
            raise

    raise last_exc  # type: ignore[misc]


async def claude_generate(
    system: str,
    messages: list[dict],
    temperature: float = 0.4,
    model: str | None = None,
) -> str:
    """Generate a full agent response using the configured main Claude model."""
    m = model or settings.claude_model
    return await asyncio.to_thread(
        _claude_sync, m, system, messages, temperature, settings.max_output_tokens
    )


async def claude_generate_fast(
    system: str,
    messages: list[dict],
    temperature: float = 0.0,
) -> str:
    """Quick call using the fast (Haiku) model — for routing and classification."""
    return await asyncio.to_thread(
        _claude_sync,
        settings.claude_fast_model,
        system,
        messages,
        temperature,
        1024,
    )


def _claude_describe_file_sync(data: bytes, mime_type: str, instruction: str) -> str:
    """Multimodal call: image/PDF bytes + text instruction. Used for file analysis."""
    client = _get_client()

    if mime_type.startswith("image/"):
        ext_map = {
            "image/jpeg": "image/jpeg",
            "image/png": "image/png",
            "image/gif": "image/gif",
            "image/webp": "image/webp",
        }
        media_type = ext_map.get(mime_type, "image/jpeg")
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.standard_b64encode(data).decode("utf-8"),
                },
            },
            {"type": "text", "text": instruction},
        ]
    else:
        content = [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": base64.standard_b64encode(data).decode("utf-8"),
                },
            },
            {"type": "text", "text": instruction},
        ]

    resp = client.messages.create(
        model=settings.claude_model,
        system="You are a senior business analyst. Extract and summarize the key information from this file.",
        messages=[{"role": "user", "content": content}],
        temperature=0.2,
        max_tokens=4096,
    )
    return (resp.content[0].text if resp.content else "").strip()


async def claude_describe_file(data: bytes, mime_type: str, instruction: str) -> str:
    return await asyncio.to_thread(_claude_describe_file_sync, data, mime_type, instruction)


def _claude_json_sync(system: str, messages: list[dict]) -> str:
    """Force JSON output via prefilling the assistant response with '{'."""
    client = _get_client()
    messages_with_prefix = list(messages) + [{"role": "assistant", "content": "{"}]
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=settings.claude_fast_model,
                system=system,
                messages=messages_with_prefix,
                temperature=0.0,
                max_tokens=512,
            )
            raw = (resp.content[0].text if resp.content else "").strip()
            return "{" + raw if not raw.startswith("{") else raw
        except Exception as exc:
            last_exc = exc
            if _is_overloaded(exc) and attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning("Claude JSON overloaded, retry %d in %ds", attempt + 1, delay)
                time.sleep(delay)
                continue
            raise

    raise last_exc  # type: ignore[misc]


async def claude_generate_json(system: str, messages: list[dict]) -> str:
    return await asyncio.to_thread(_claude_json_sync, system, messages)
