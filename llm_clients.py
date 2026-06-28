"""
Unified LLM client — OpenRouter (free) or Claude (Anthropic).

Provider priority (checked at runtime):
  1. OPENROUTER_API_KEY is set → use OpenRouter free models (default)
  2. ANTHROPIC_API_KEY is set  → use Claude (Sonnet/Haiku)
  3. Neither set               → startup validation warns, calls will fail

Public API (same regardless of provider):
  claude_generate()      — main model, deep analysis / agent responses
  claude_generate_fast() — fast model, routing / classification
  claude_describe_file() — vision / PDF file understanding
  claude_generate_json() — structured JSON output

Retry logic: both providers are retried up to 3x with exponential backoff
(2s → 4s → 8s) on overload/503/429 errors.
"""

import asyncio
import base64
import logging
import time

from config import settings

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE = 2  # seconds

# --------------------------------------------------------------------------
# Lazy singletons
# --------------------------------------------------------------------------
_or_client = None
_claude_client = None


def _get_or_client():
    global _or_client
    if _or_client is None:
        from openai import OpenAI
        _or_client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": settings.openrouter_referer,
                "X-Title": settings.openrouter_title,
            },
        )
    return _or_client


def _get_claude_client():
    global _claude_client
    if _claude_client is None:
        import anthropic
        _claude_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _claude_client


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc)
    return any(code in msg for code in ("529", "503", "429", "overloaded", "rate limit"))


# --------------------------------------------------------------------------
# OpenRouter (OpenAI-compatible)
# --------------------------------------------------------------------------
def _or_sync(model: str, system: str, messages: list[dict], temperature: float, max_tokens: int) -> str:
    client = _get_or_client()
    full = [{"role": "system", "content": system}] + messages
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=full,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            last_exc = exc
            if _is_retryable(exc) and attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE * (2 ** attempt)
                logger.warning("OpenRouter overloaded (attempt %d), retry in %ds: %s",
                               attempt + 1, delay, str(exc)[:100])
                time.sleep(delay)
                # Try fallback model on second retry
                if attempt == 1 and model != settings.or_auto_model:
                    model = settings.or_auto_model
                continue
            raise

    raise last_exc  # type: ignore[misc]


def _or_vision_sync(data: bytes, mime_type: str, instruction: str) -> str:
    client = _get_or_client()
    # Vision via OpenRouter — use a model that supports images
    vision_model = "google/gemini-2.0-flash-exp:free"

    if mime_type.startswith("image/"):
        b64 = base64.standard_b64encode(data).decode()
        content = [
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
            {"type": "text", "text": instruction},
        ]
    else:
        # For PDFs and other docs, send as text extraction instruction
        content = [{"type": "text", "text": f"{instruction}\n\n[File content in base64 — {mime_type}]"}]

    resp = client.chat.completions.create(
        model=vision_model,
        messages=[
            {"role": "system", "content": "You are a senior business analyst. Analyse files thoroughly."},
            {"role": "user", "content": content},
        ],
        temperature=0.2,
        max_tokens=4096,
    )
    return (resp.choices[0].message.content or "").strip()


# --------------------------------------------------------------------------
# Claude (Anthropic)
# --------------------------------------------------------------------------
def _claude_sync(model: str, system: str, messages: list[dict], temperature: float, max_tokens: int) -> str:
    client = _get_claude_client()
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=model, system=system, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            return (resp.content[0].text if resp.content else "").strip()
        except Exception as exc:
            last_exc = exc
            if _is_retryable(exc) and attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE * (2 ** attempt)
                logger.warning("Claude overloaded (attempt %d), retry in %ds", attempt + 1, delay)
                time.sleep(delay)
                continue
            raise

    raise last_exc  # type: ignore[misc]


def _claude_vision_sync(data: bytes, mime_type: str, instruction: str) -> str:
    client = _get_claude_client()
    if mime_type.startswith("image/"):
        ext_map = {"image/jpeg": "image/jpeg", "image/png": "image/png",
                   "image/gif": "image/gif", "image/webp": "image/webp"}
        content = [
            {"type": "image", "source": {
                "type": "base64",
                "media_type": ext_map.get(mime_type, "image/jpeg"),
                "data": base64.standard_b64encode(data).decode(),
            }},
            {"type": "text", "text": instruction},
        ]
    else:
        content = [
            {"type": "document", "source": {
                "type": "base64", "media_type": mime_type,
                "data": base64.standard_b64encode(data).decode(),
            }},
            {"type": "text", "text": instruction},
        ]
    resp = client.messages.create(
        model=settings.claude_model,
        system="You are a senior business analyst. Extract and summarize the key information from this file.",
        messages=[{"role": "user", "content": content}],
        temperature=0.2, max_tokens=4096,
    )
    return (resp.content[0].text if resp.content else "").strip()


# --------------------------------------------------------------------------
# Public API — provider-transparent
# --------------------------------------------------------------------------
async def claude_generate(
    system: str,
    messages: list[dict],
    temperature: float = 0.4,
    model: str | None = None,
) -> str:
    """Main model response. Dispatches to OpenRouter or Claude."""
    if settings.provider == "openrouter":
        m = model or settings.or_main_model
        return await asyncio.to_thread(
            _or_sync, m, system, messages, temperature, settings.max_output_tokens
        )
    m = model or settings.claude_model
    return await asyncio.to_thread(
        _claude_sync, m, system, messages, temperature, settings.max_output_tokens
    )


async def claude_generate_fast(
    system: str,
    messages: list[dict],
    temperature: float = 0.0,
) -> str:
    """Fast model for routing/classification."""
    if settings.provider == "openrouter":
        return await asyncio.to_thread(
            _or_sync, settings.or_fast_model, system, messages, temperature, 1024
        )
    return await asyncio.to_thread(
        _claude_sync, settings.claude_fast_model, system, messages, temperature, 1024
    )


async def claude_describe_file(data: bytes, mime_type: str, instruction: str) -> str:
    """Vision/PDF file understanding."""
    if settings.provider == "openrouter":
        return await asyncio.to_thread(_or_vision_sync, data, mime_type, instruction)
    return await asyncio.to_thread(_claude_vision_sync, data, mime_type, instruction)


def _or_json_sync(system: str, messages: list[dict]) -> str:
    client = _get_or_client()
    full = [{"role": "system", "content": system + "\nRespond with ONLY valid JSON, no markdown, no prose."}] + messages
    resp = client.chat.completions.create(
        model=settings.or_fast_model,
        messages=full,
        temperature=0.0,
        max_tokens=512,
    )
    raw = (resp.choices[0].message.content or "").strip()
    # Strip markdown code fences if model wraps JSON in ```json ... ```
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def _claude_json_sync(system: str, messages: list[dict]) -> str:
    client = _get_claude_client()
    msgs = list(messages) + [{"role": "assistant", "content": "{"}]
    resp = client.messages.create(
        model=settings.claude_fast_model, system=system,
        messages=msgs, temperature=0.0, max_tokens=512,
    )
    raw = (resp.content[0].text if resp.content else "").strip()
    return "{" + raw if not raw.startswith("{") else raw


async def claude_generate_json(system: str, messages: list[dict]) -> str:
    """Structured JSON output for router classification."""
    if settings.provider == "openrouter":
        return await asyncio.to_thread(_or_json_sync, system, messages)
    return await asyncio.to_thread(_claude_json_sync, system, messages)
