"""
OpenRouter integration — unified free model pool for the Fintech Orchestrator.

Why OpenRouter in addition to Gemini/Groq/GLM:
  - Single API key → access to 300+ models including 26+ free (:free) models
  - Automatic smart routing (`openrouter/free`) picks the best available free
    model for each request at zero cost
  - Eliminates the Groq 6000-TPM 413 error completely (free OR models have
    much more generous limits)
  - `deepseek/deepseek-r1:free` for compliance/risk reasoning
  - `qwen/qwen3-coder:free` for code generation
  - `meta-llama/llama-4-scout:free` for fast chat
  - Built-in fallback: OpenRouter retries across providers automatically

Rate limits (free tier as of June 2026):
  50 req/day, 20 req/min per model. `openrouter/free` counts against the
  selected underlying model's quota, not a shared pool.

API: OpenAI-compatible — uses the `openai` package (already installed).
Required headers: HTTP-Referer and X-Title (per OpenRouter's spec).

Recommended OpenRouter free models (verify at openrouter.ai/models — `:free`
availability changes):
  - openrouter/free          auto-picks best available free model
  - meta-llama/llama-4-scout:free    fast, low-latency chat
  - meta-llama/llama-4-maverick:free 128K ctx, vision support
  - deepseek/deepseek-r1:free        reasoning (compliance/risk analysis)
  - qwen/qwen3-coder:free            coding-focused
  - google/gemini-2.0-flash-exp:free Google's free flash model
  - nvidia/llama-3.1-nemotron-ultra-253b-v1:free  1M ctx (while promo lasts)
"""

import asyncio
import logging

from config import settings

logger = logging.getLogger(__name__)

_or_client = None


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


def _or_sync(model: str, system: str, messages: list[dict], temperature: float) -> str:
    client = _get_or_client()
    full = [{"role": "system", "content": system}] + messages
    resp = client.chat.completions.create(
        model=model,
        messages=full,
        temperature=temperature,
        max_tokens=settings.max_output_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


async def or_generate(
    model: str,
    system: str,
    messages: list[dict],
    temperature: float = 0.3,
) -> str:
    """Generate via OpenRouter. Falls back to `openrouter/free` if the given
    model ID triggers a 404 (no longer free or renamed)."""
    try:
        return await asyncio.to_thread(_or_sync, model, system, messages, temperature)
    except Exception as exc:
        msg = str(exc)
        if ("404" in msg or "not found" in msg.lower()) and model != "openrouter/free":
            logger.warning(
                "OpenRouter model '%s' not found/free, falling back to openrouter/free. %s",
                model, msg[:200]
            )
            return await asyncio.to_thread(
                _or_sync, "openrouter/free", system, messages, temperature
            )
        raise


async def or_health_check() -> tuple[bool, str]:
    """Probe OpenRouter with a minimal request. Returns (ok, message)."""
    try:
        result = await asyncio.wait_for(
            or_generate("openrouter/free", "You are a helpful assistant.", [{"role": "user", "content": "Hi"}]),
            timeout=20,
        )
        return True, f"OK (len={len(result)})"
    except Exception as exc:
        return False, f"OpenRouter probe failed: {str(exc)[:200]}"
