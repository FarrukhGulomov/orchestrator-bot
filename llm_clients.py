"""
Unified LLM client with multi-provider failover: Claude, ChatGPT (OpenAI),
Gemini, Grok (xAI), DeepSeek, Kimi (Moonshot), and OpenRouter.

Every provider is fully optional (works with just one key set). For every
call, claude_generate()/claude_generate_fast()/claude_generate_json() try
each CONFIGURED provider in PROVIDER_PRIORITY order (see config.py) until
one succeeds — a dead/invalid/quota-exhausted key on one provider doesn't
stop the bot, it just falls through to the next one. See
_generate_with_failover() / _generate_json_with_failover().

Public API (provider-transparent):
  claude_generate()      — main-tier model: agent responses, deep analysis
  claude_generate_fast() — fast-tier model: routing, classification, memory
  claude_describe_file() — vision/PDF understanding (Claude/OpenRouter only
                            for now — see its docstring)
  claude_generate_json() — structured JSON (router, docgen)

Retry logic: 3x exponential backoff (2s → 4s → 8s) on 503/429/529 PER
PROVIDER, before moving on to the next provider in the chain.

LANGUAGE-DRIVEN PREFERENCE: an Uzbek message jumps Gemini to the front of
the attempt order regardless of PROVIDER_PRIORITY, if GEMINI_API_KEY is
configured — see _looks_uzbek(). Reported directly by the bot's operator:
Gemini answers Uzbek noticeably more fluently than the other providers.
This only ever REORDERS the first attempt; the full configured chain
(including whatever PROVIDER_PRIORITY/an explicit preferred model would
have tried) still runs afterward if Gemini is unavailable or fails, so it
never narrows the failover safety net — just biases who answers first.
"""

import asyncio
import base64
import contextvars
import io
import json
import logging
import time

from config import settings

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE = 2  # seconds

# Gemini answers Uzbek noticeably more fluently than the other configured
# providers (reported directly by the bot's operator) — worth a quality-
# driven preference, not just whatever PROVIDER_PRIORITY says. This is a
# best-effort LANGUAGE SIGNAL, not real language ID: good enough to pick a
# preferred provider, not to gate a feature. A false positive/negative just
# means an ordinary provider answers instead of the preferred one — never a
# broken reply, since the full failover chain still runs either way.
_UZBEK_CYRILLIC_ONLY = ("ў", "қ", "ғ", "ҳ")  # not in the standard Russian alphabet
_UZBEK_WORDS = (
    "bo'l", "bo'ladi", "qil", "qiladi", "uchun", "bilan", "kerak", "ekan",
    "rahmat", "salom", "yordam", "qanday", "nima", "qachon", "bugun",
    "ertaga", "sizga", "menga", "bizga", "iltimos", "tushun", "haqida",
    "o'zbek", "yaxshi", "albatta", "javob", "savol",
)


def _looks_uzbek(text: str) -> bool:
    low = (text or "").lower()
    if any(ch in low for ch in _UZBEK_CYRILLIC_ONLY):
        return True
    return any(w in low for w in _UZBEK_WORDS)

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


def _is_model_unavailable(exc: Exception) -> bool:
    msg = str(exc)
    return "404" in msg or "No endpoints found" in msg or "model not found" in msg.lower()


def _capture_usage(usage_out: dict | None, response: object, model: str) -> None:
    """Fill an optional caller-supplied dict with token counts from a raw
    provider response.

    An out-parameter rather than a changed return type on purpose: these
    sync functions have other callers (the vision path calls _or_sync
    directly), and threading a tuple through all of them would be a wide,
    risky diff for a bookkeeping feature. Passing None — the default — keeps
    the original behaviour exactly.

    `model` is recorded here rather than by the caller because _or_sync can
    swap to the auto-router mid-retry; this captures what actually ran.
    """
    if usage_out is None:
        return
    try:
        import telemetry

        inp, out = telemetry.extract_usage(response)
        usage_out["input_tokens"] = inp
        usage_out["output_tokens"] = out
        usage_out["model"] = model
    except Exception:  # noqa: BLE001 — never let bookkeeping break a call
        logger.debug("usage capture failed", exc_info=True)


# --------------------------------------------------------------------------
# OpenRouter (OpenAI-compatible)
# --------------------------------------------------------------------------
def _or_sync(
    model: str, system: str, messages: list[dict], temperature: float, max_tokens: int,
    usage_out: dict | None = None,
) -> str:
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
            _capture_usage(usage_out, resp, model)
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            last_exc = exc
            if _is_model_unavailable(exc) and model != settings.or_auto_model:
                logger.warning("OpenRouter model unavailable (%s), switching to auto fallback", model)
                model = settings.or_auto_model
                continue
            if _is_retryable(exc) and attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE * (2 ** attempt)
                logger.warning("OpenRouter overloaded (attempt %d), retry in %ds: %s",
                               attempt + 1, delay, str(exc)[:100])
                time.sleep(delay)
                if attempt == 1 and model != settings.or_auto_model:
                    model = settings.or_auto_model
                continue
            raise

    raise last_exc  # type: ignore[misc]


def _extract_pdf_text(data: bytes) -> str:
    """Best-effort local PDF text extraction (used when the provider has no
    native PDF understanding, i.e. OpenRouter). Returns "" for scans/images."""
    try:
        from pypdf import PdfReader
    except Exception:  # noqa: BLE001 — missing/broken optional dep must not crash
        logger.warning("pypdf unavailable — cannot extract PDF text locally")
        return ""
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for page in reader.pages[:50]:
            pages.append(page.extract_text() or "")
        return "\n".join(pages).strip()
    except Exception:  # noqa: BLE001 — corrupt PDF must not crash the caller
        logger.exception("Local PDF text extraction failed")
        return ""


def _or_vision_sync(data: bytes, mime_type: str, instruction: str) -> str:
    client = _get_or_client()
    vision_model = settings.or_main_model

    if mime_type.startswith("image/"):
        b64 = base64.standard_b64encode(data).decode()
        content = [
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
            {"type": "text", "text": instruction},
        ]
    elif mime_type == "application/pdf":
        # OpenRouter chat models have no native PDF input — extract the text
        # locally and analyse that instead of silently sending a placeholder.
        text = _extract_pdf_text(data)
        if not text:
            raise ValueError(
                "Bu PDF matnini ajratib bo'lmadi (skan yoki rasm bo'lishi mumkin). "
                "PDF skan tahlili uchun Claude provider (ANTHROPIC_API_KEY) kerak."
            )
        return _or_sync(
            vision_model,
            "You are a senior business analyst. Analyse files thoroughly.",
            [{"role": "user", "content": f"{instruction}\n\n--- PDF content ---\n{text[:30000]}"}],
            0.2,
            4096,
        )
    else:
        raise ValueError(
            f"OpenRouter provider bu fayl turini ({mime_type}) tahlil qila olmaydi. "
            "Rasm yoki PDF yuboring, yoki Claude provider (ANTHROPIC_API_KEY) sozlang."
        )

    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
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
        except Exception as exc:
            last_exc = exc
            if _is_retryable(exc) and attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE * (2 ** attempt)
                logger.warning("OpenRouter vision overloaded (attempt %d), retry in %ds",
                               attempt + 1, delay)
                time.sleep(delay)
                continue
            raise

    raise last_exc  # type: ignore[misc]


# --------------------------------------------------------------------------
# Claude (Anthropic)
# --------------------------------------------------------------------------
def _claude_sync(
    model: str, system: str, messages: list[dict], temperature: float, max_tokens: int,
    usage_out: dict | None = None,
) -> str:
    client = _get_claude_client()
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=model, system=system, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            _capture_usage(usage_out, resp, model)
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
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=settings.claude_model,
                system="You are a senior business analyst. Extract and summarize the key information from this file.",
                messages=[{"role": "user", "content": content}],
                temperature=0.2, max_tokens=4096,
            )
            return (resp.content[0].text if resp.content else "").strip()
        except Exception as exc:
            last_exc = exc
            if _is_retryable(exc) and attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE * (2 ** attempt)
                logger.warning("Claude vision overloaded (attempt %d), retry in %ds", attempt + 1, delay)
                time.sleep(delay)
                continue
            raise

    raise last_exc  # type: ignore[misc]


# --------------------------------------------------------------------------
# Multi-provider failover (ChatGPT, Gemini, Grok, DeepSeek, Kimi, plus the
# Claude/OpenRouter clients above) — every provider here speaks the OpenAI
# chat.completions schema except Claude (native Anthropic API, handled
# separately above). If one provider's key is missing, invalid, out of
# quota, or erroring, the NEXT configured provider in PROVIDER_PRIORITY
# answers instead — the bot keeps working as long as ONE key is still good.
# --------------------------------------------------------------------------
_KNOWN_PROVIDERS = ("claude", "openai", "gemini", "grok", "deepseek", "kimi", "openrouter")

_COMPAT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "grok": "https://api.x.ai/v1",
    "deepseek": "https://api.deepseek.com",
    "kimi": "https://api.moonshot.ai/v1",
}

_compat_clients: dict[str, object] = {}


def _get_compat_client(key: str, api_key: str, base_url: str):
    if key not in _compat_clients:
        from openai import OpenAI
        _compat_clients[key] = OpenAI(api_key=api_key, base_url=base_url)
    return _compat_clients[key]


def _compat_sync(
    key: str, base_url: str, api_key: str, model: str,
    system: str, messages: list[dict], temperature: float, max_tokens: int,
    usage_out: dict | None = None,
) -> str:
    client = _get_compat_client(key, api_key, base_url)
    full = [{"role": "system", "content": system}] + messages
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model, messages=full, temperature=temperature, max_tokens=max_tokens,
            )
            _capture_usage(usage_out, resp, model)
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            last_exc = exc
            if _is_retryable(exc) and attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE * (2 ** attempt)
                logger.warning("%s overloaded (attempt %d), retry in %ds: %s",
                               key, attempt + 1, delay, str(exc)[:100])
                time.sleep(delay)
                continue
            raise
    raise last_exc  # type: ignore[misc]


def _compat_json_sync(
    key: str, base_url: str, api_key: str, model: str,
    system: str, messages: list[dict], max_tokens: int,
    usage_out: dict | None = None,
) -> str:
    client = _get_compat_client(key, api_key, base_url)
    full = [{"role": "system", "content": system + "\nRespond with ONLY valid JSON, no markdown, no prose."}] + messages
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model, messages=full, temperature=0.0, max_tokens=max_tokens,
            )
            _capture_usage(usage_out, resp, model)
            raw = (resp.choices[0].message.content or "").strip()
            return _strip_json_fences(raw)
        except Exception as exc:
            last_exc = exc
            if _is_retryable(exc) and attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE * (2 ** attempt)
                logger.warning("%s JSON call overloaded (attempt %d), retry in %ds: %s",
                               key, attempt + 1, delay, str(exc)[:100])
                time.sleep(delay)
                continue
            raise
    raise last_exc  # type: ignore[misc]


def _provider_api_key(key: str) -> str:
    return {
        "claude": settings.anthropic_api_key,
        "openai": settings.openai_api_key,
        "gemini": settings.gemini_api_key,
        "grok": settings.xai_api_key,
        "deepseek": settings.deepseek_api_key,
        "kimi": settings.kimi_api_key,
        "openrouter": settings.openrouter_api_key,
    }.get(key, "")


def _provider_models(key: str) -> tuple[str, str, str]:
    """(main_model, fast_model, display_label) for a provider key."""
    return {
        "claude": (settings.claude_model, settings.claude_fast_model, "Claude"),
        "openai": (settings.openai_model, settings.openai_model, settings.openai_model_label),
        "gemini": (settings.gemini_model, settings.gemini_fast_model, settings.gemini_model_label),
        "grok": (settings.grok_model, settings.grok_model, settings.grok_model_label),
        "deepseek": (settings.deepseek_model, settings.deepseek_model, settings.deepseek_model_label),
        "kimi": (settings.kimi_model, settings.kimi_model, settings.kimi_model_label),
        "openrouter": (settings.or_main_model, settings.or_fast_model, "OpenRouter"),
    }[key]


def _configured_chain() -> list[str]:
    """Provider keys in PROVIDER_PRIORITY order, filtered to ones with an
    API key set. Any configured-but-unlisted provider (someone added a key
    without touching PROVIDER_PRIORITY) is appended at the end so it still
    gets used rather than silently ignored."""
    order = [p.strip().lower() for p in settings.provider_priority.split(",") if p.strip()]
    chain: list[str] = []
    for key in order:
        if key in _KNOWN_PROVIDERS and _provider_api_key(key) and key not in chain:
            chain.append(key)
    for key in _KNOWN_PROVIDERS:
        if key not in chain and _provider_api_key(key):
            chain.append(key)
    return chain


def provider_chain_status() -> list[tuple[str, str, bool]]:
    """(provider_key, label, configured) for every known provider, in the
    active PROVIDER_PRIORITY order — used by /status for diagnostics."""
    order = [p.strip().lower() for p in settings.provider_priority.split(",") if p.strip()]
    seen: list[str] = [k for k in order if k in _KNOWN_PROVIDERS]
    for key in _KNOWN_PROVIDERS:
        if key not in seen:
            seen.append(key)
    return [(key, _provider_models(key)[2], bool(_provider_api_key(key))) for key in seen]


def _call_main_sync(
    key: str, model: str, system: str, messages: list[dict], temperature: float, max_tokens: int,
    usage_out: dict | None = None,
) -> str:
    if key == "claude":
        return _claude_sync(model, system, messages, temperature, max_tokens, usage_out)
    if key == "openrouter":
        return _or_sync(model, system, messages, temperature, max_tokens, usage_out)
    return _compat_sync(
        key, _COMPAT_BASE_URLS[key], _provider_api_key(key), model, system, messages,
        temperature, max_tokens, usage_out,
    )


def _call_json_sync(
    key: str, model: str, system: str, messages: list[dict], max_tokens: int,
    usage_out: dict | None = None,
) -> str:
    if key == "claude":
        return _claude_json_sync(system, messages, model, max_tokens, usage_out)
    if key == "openrouter":
        return _or_json_sync(system, messages, model, max_tokens, usage_out)
    return _compat_json_sync(
        key, _COMPAT_BASE_URLS[key], _provider_api_key(key), model, system, messages,
        max_tokens, usage_out,
    )


# Which (provider_key, model) actually answered the most recent
# claude_generate()/claude_generate_json() call in THIS async task — set on
# every successful attempt in _attempt(). Not necessarily the provider
# router.model_for() preferred: failover may have moved on. Exists so a
# caller can correct a model-id label AFTER the fact instead of guessing it
# beforehand (see bot.py's _answer_with_agent, which fixes route.model_label
# to reflect who actually answered rather than who was merely asked first).
_last_provider: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar(
    "last_llm_provider", default=None
)


def last_used_provider() -> tuple[str, str] | None:
    """(provider_key, model) for the most recent successful call, or None
    before any call has completed in this task."""
    return _last_provider.get()


def provider_label(key: str) -> str:
    """Display label for a provider key, e.g. 'ChatGPT (GPT-4.1)'."""
    return _provider_models(key)[2] if key in _KNOWN_PROVIDERS else key


async def _attempt(
    call, key: str, model: str, tier: str, position: int, *args,
) -> str:
    """Run one provider attempt, timing it and recording the outcome either
    way. `call` is _call_main_sync or _call_json_sync.

    Recording lives here — one place, wrapping every provider attempt — so
    a FAILED attempt is recorded too. That's what makes "provider 0 is
    quietly 401-ing and provider 1 is silently carrying the traffic"
    visible instead of invisible.
    """
    import telemetry

    usage: dict = {}
    started = time.monotonic()
    try:
        result = await asyncio.to_thread(call, key, model, *args, usage)
    except Exception as exc:
        await telemetry.record(
            provider=key, model=usage.get("model") or model, tier=tier,
            latency_ms=int((time.monotonic() - started) * 1000),
            ok=False, fallback_position=position, error=str(exc)[:300],
        )
        raise
    await telemetry.record(
        provider=key, model=usage.get("model") or model, tier=tier,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        latency_ms=int((time.monotonic() - started) * 1000),
        ok=True, fallback_position=position,
    )
    _last_provider.set((key, usage.get("model") or model))
    return result


async def _generate_with_failover(
    system: str, messages: list[dict], temperature: float, max_tokens: int,
    tier: str, preferred_model: str | None = None,
) -> str:
    chain = _configured_chain()
    if not chain:
        raise RuntimeError("Hech qanday AI provider sozlanmagan (API key topilmadi).")

    tried: set[str] = set()
    last_exc: Exception | None = None

    # Uzbek gets a quality-driven preference for Gemini (see _looks_uzbek),
    # tried BEFORE the router's own preferred_model — a fluent answer in
    # the user's language matters more here than the general "best model"
    # ranking. This only ever REORDERS the first attempt; every provider,
    # including whatever preferred_model would have been, is still tried
    # afterward if Gemini is unavailable or fails, so it never narrows the
    # failover safety net.
    if "gemini" in chain and _looks_uzbek(" ".join(str(m.get("content", "")) for m in messages)):
        main_model, fast_model, _label = _provider_models("gemini")
        try:
            return await _attempt(
                _call_main_sync, "gemini", main_model if tier == "main" else fast_model,
                tier, 0, system, messages, temperature, max_tokens,
            )
        except Exception as exc:
            last_exc = exc
            logger.warning("Gemini (Uzbek preference) failed, falling back: %s", str(exc)[:200])
            tried.add("gemini")

    # An explicit model id (e.g. from router.model_for()) is honored as the
    # next attempt if its provider is in the chain, before falling through
    # to the standard priority order.
    if preferred_model:
        preferred_key = "openrouter" if "/" in preferred_model else "claude"
        if preferred_key in chain and preferred_key not in tried:
            try:
                return await _attempt(
                    _call_main_sync, preferred_key, preferred_model, tier, len(tried),
                    system, messages, temperature, max_tokens,
                )
            except Exception as exc:
                last_exc = exc
                logger.warning("Preferred provider '%s' failed, falling back: %s",
                                preferred_key, str(exc)[:200])
                tried.add(preferred_key)

    for key in chain:
        if key in tried:
            continue
        main_model, fast_model, _label = _provider_models(key)
        model = main_model if tier == "main" else fast_model
        try:
            result = await _attempt(
                _call_main_sync, key, model, tier, len(tried),
                system, messages, temperature, max_tokens,
            )
            if tried:
                logger.info("Answered via fallback provider '%s' (tier=%s)", key, tier)
            return result
        except Exception as exc:
            last_exc = exc
            logger.warning("Provider '%s' failed (tier=%s), trying next: %s", key, tier, str(exc)[:200])
            tried.add(key)

    raise last_exc  # type: ignore[misc]


async def _generate_json_with_failover(
    system: str, messages: list[dict], max_tokens: int,
    tier: str, preferred_model: str | None = None,
) -> str:
    chain = _configured_chain()
    if not chain:
        raise RuntimeError("Hech qanday AI provider sozlanmagan (API key topilmadi).")

    tried: set[str] = set()
    last_exc: Exception | None = None

    # See the identical block in _generate_with_failover for why Uzbek
    # jumps the queue for Gemini specifically, and why this only reorders
    # rather than narrows the failover chain.
    if "gemini" in chain and _looks_uzbek(" ".join(str(m.get("content", "")) for m in messages)):
        main_model, fast_model, _label = _provider_models("gemini")
        try:
            return await _attempt(
                _call_json_sync, "gemini", main_model if tier == "main" else fast_model,
                tier, 0, system, messages, max_tokens,
            )
        except Exception as exc:
            last_exc = exc
            logger.warning("Gemini (Uzbek preference) failed (JSON), falling back: %s", str(exc)[:200])
            tried.add("gemini")

    if preferred_model:
        preferred_key = "openrouter" if "/" in preferred_model else "claude"
        if preferred_key in chain and preferred_key not in tried:
            try:
                return await _attempt(
                    _call_json_sync, preferred_key, preferred_model, tier, len(tried),
                    system, messages, max_tokens,
                )
            except Exception as exc:
                last_exc = exc
                logger.warning("Preferred provider '%s' failed (JSON), falling back: %s",
                                preferred_key, str(exc)[:200])
                tried.add(preferred_key)

    for key in chain:
        if key in tried:
            continue
        main_model, fast_model, _label = _provider_models(key)
        model = main_model if tier == "main" else fast_model
        try:
            result = await _attempt(
                _call_json_sync, key, model, tier, len(tried),
                system, messages, max_tokens,
            )
            if tried:
                logger.info("Answered via fallback provider '%s' (tier=%s, JSON)", key, tier)
            return result
        except Exception as exc:
            last_exc = exc
            logger.warning("Provider '%s' failed (tier=%s, JSON), trying next: %s", key, tier, str(exc)[:200])
            tried.add(key)

    raise last_exc  # type: ignore[misc]


# --------------------------------------------------------------------------
# Public API — provider-transparent
# --------------------------------------------------------------------------
async def claude_generate(
    system: str,
    messages: list[dict],
    temperature: float = 0.4,
    model: str | None = None,
) -> str:
    """Main model: agent responses, deep analysis, document generation.

    Tries every configured provider in PROVIDER_PRIORITY order (main-tier
    model each) until one succeeds — see _generate_with_failover. `model`,
    if given (e.g. from router.model_for()), is tried first.
    """
    return await _generate_with_failover(
        system, messages, temperature, settings.max_output_tokens, tier="main", preferred_model=model,
    )


async def claude_generate_fast(
    system: str,
    messages: list[dict],
    temperature: float = 0.0,
) -> str:
    """Fast model: routing, classification, memory extraction, relevance checks.

    Same failover chain as claude_generate(), but using each provider's
    fast/cheap model tier — quality matters less here than for real answers.
    """
    return await _generate_with_failover(system, messages, temperature, 1024, tier="fast")


async def claude_describe_file(data: bytes, mime_type: str, instruction: str) -> str:
    """Vision/PDF file understanding.

    NOTE: not yet part of the multi-provider failover chain — image/PDF
    content-block formats differ enough across providers (Claude's native
    blocks vs. OpenAI-style image_url vs. Gemini's own conventions) that
    wiring all of them up is a separate piece of work. Still just Claude
    (native, best accuracy) with an OpenRouter fallback:

    claude/hybrid → Claude (native PDF + vision)
    openrouter    → OpenRouter vision model (images) + pypdf (PDFs)
    """
    if settings.anthropic_api_key:
        return await asyncio.to_thread(_claude_vision_sync, data, mime_type, instruction)
    if settings.openrouter_api_key:
        return await asyncio.to_thread(_or_vision_sync, data, mime_type, instruction)
    raise RuntimeError(
        "Fayl tahlili uchun ANTHROPIC_API_KEY yoki OPENROUTER_API_KEY kerak "
        "(rasm/PDF tahlili hozircha boshqa provayderlar orqali ishlamaydi)."
    )


def _strip_json_fences(raw: str) -> str:
    # Strip markdown code fences if model wraps JSON in ```json ... ```
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def _repair_json_control_chars(raw: str) -> str:
    """Best-effort repair for the single most common way a free/weaker model
    breaks otherwise-valid JSON: emitting a LITERAL newline/tab inside a
    string value (e.g. a multi-line suggested_reply) instead of escaping it
    as \\n/\\t. json.loads rejects that outright with "Unterminated string
    starting at...", which every JSON-consuming caller (router, business/
    group copilot, minutes, docgen) would otherwise have to catch and guess
    about individually. Walks the text tracking whether we're inside a
    string literal (honoring backslash-escapes and quote boundaries) and
    re-escapes any raw control character found there — structurally valid
    JSON is returned byte-for-byte unchanged."""
    out = []
    in_string = False
    escaped = False
    for ch in raw:
        if not in_string:
            if ch == '"':
                in_string = True
            out.append(ch)
            continue
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            in_string = False
            out.append(ch)
            continue
        if ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        else:
            out.append(ch)
    return "".join(out)


def _repair_unescaped_quotes(raw: str) -> str:
    """Second-tier repair, only tried if control-char repair alone still
    doesn't parse: escapes a `"` that looks like it's INSIDE a string value
    rather than closing it — e.g. suggested_reply: "u \"aka\" dedi" written
    by the model as suggested_reply: "u "aka" dedi" (forgot to escape the
    inner quotes). A quote is treated as a genuine closing quote only if,
    after optional whitespace, the next character is a JSON structural one
    (, } ] :) or end of input; otherwise it's escaped and scanning continues
    as still-inside-the-string. Not a full JSON grammar — a heuristic that
    recovers the common "model quoted a phrase without escaping it" case."""
    out = []
    in_string = False
    escaped = False
    n = len(raw)
    i = 0
    while i < n:
        ch = raw[i]
        if not in_string:
            if ch == '"':
                in_string = True
            out.append(ch)
            i += 1
            continue
        if escaped:
            out.append(ch)
            escaped = False
            i += 1
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            i += 1
            continue
        if ch == '"':
            j = i + 1
            while j < n and raw[j] in " \t\r\n":
                j += 1
            if j >= n or raw[j] in ",}]:":
                in_string = False
                out.append(ch)
            else:
                out.append('\\"')
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def parse_llm_json(raw: str):
    """Best-effort JSON parsing for LLM output that's SUPPOSED to be JSON
    but, especially from free/weaker models routed through openrouter/free
    (a different model on every call, with varying JSON discipline), isn't
    always quite valid. Every caller that parses structured LLM output
    (router, business/group copilot, minutes, docgen, task classification)
    should use this instead of bare json.loads() for that reason.

    Tries progressively more aggressive repairs and raises the ORIGINAL
    json.JSONDecodeError (not a repair-stage one) if nothing works, so an
    error message shown to the admin describes the real problem instead of
    an artifact of a failed repair attempt."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError as first_exc:
        stage1 = _repair_json_control_chars(raw)
        try:
            return json.loads(stage1)
        except json.JSONDecodeError:
            pass
        stage2 = _repair_unescaped_quotes(stage1)
        try:
            return json.loads(stage2)
        except json.JSONDecodeError:
            pass
        raise first_exc


def _or_json_sync(
    system: str, messages: list[dict], model: str, max_tokens: int,
    usage_out: dict | None = None,
) -> str:
    client = _get_or_client()
    full = [{"role": "system", "content": system + "\nRespond with ONLY valid JSON, no markdown, no prose."}] + messages
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=full,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            _capture_usage(usage_out, resp, model)
            raw = (resp.choices[0].message.content or "").strip()
            return _strip_json_fences(raw)
        except Exception as exc:
            last_exc = exc
            if _is_model_unavailable(exc) and model != settings.or_auto_model:
                logger.warning("OpenRouter JSON model unavailable (%s), switching to auto fallback", model)
                model = settings.or_auto_model
                continue
            if _is_retryable(exc) and attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE * (2 ** attempt)
                logger.warning("OpenRouter JSON call overloaded (attempt %d), retry in %ds: %s",
                               attempt + 1, delay, str(exc)[:100])
                time.sleep(delay)
                if attempt == 1 and model != settings.or_auto_model:
                    model = settings.or_auto_model
                continue
            raise

    raise last_exc  # type: ignore[misc]


def _claude_json_sync(
    system: str, messages: list[dict], model: str, max_tokens: int,
    usage_out: dict | None = None,
) -> str:
    client = _get_claude_client()
    msgs = list(messages) + [{"role": "assistant", "content": "{"}]
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=model, system=system,
                messages=msgs, temperature=0.0, max_tokens=max_tokens,
            )
            _capture_usage(usage_out, resp, model)
            raw = (resp.content[0].text if resp.content else "").strip()
            return "{" + raw if not raw.startswith("{") else raw
        except Exception as exc:
            last_exc = exc
            if _is_retryable(exc) and attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE * (2 ** attempt)
                logger.warning("Claude JSON call overloaded (attempt %d), retry in %ds", attempt + 1, delay)
                time.sleep(delay)
                continue
            raise

    raise last_exc  # type: ignore[misc]


async def claude_generate_json(
    system: str,
    messages: list[dict],
    model: str | None = None,
    max_tokens: int = 512,
) -> str:
    """Structured JSON output.

    Defaults to fast model + small token budget (router classification).
    Pass larger max_tokens for document generation to avoid mid-JSON truncation.

    hybrid/openrouter → OpenRouter fast free model (routing is free)
    claude            → Claude Haiku

    An explicit model id (docgen/minutes pass model_for()'s "high" result
    when JSON quality matters) is tried first via the main-tier failover
    chain; no model → fast-tier chain (router classification etc).
    """
    tier = "main" if model else "fast"
    raw = await _generate_json_with_failover(system, messages, max_tokens, tier=tier, preferred_model=model)
    # Applied unconditionally: a no-op for already-valid JSON (its string
    # literals never contain raw control chars), and fixes the single most
    # common way a free/weaker model breaks otherwise-valid JSON — see
    # _repair_json_control_chars's docstring.
    return _repair_json_control_chars(raw)


async def generate_json(
    system: str,
    messages: list[dict],
    model: str | None = None,
    max_tokens: int = 512,
    timeout: float | None = None,
    retries: int = 1,
) -> dict:
    """High-level structured-JSON helper: calls claude_generate_json, parses
    the result with parse_llm_json() (which already repairs common
    formatting slips — raw control chars, unescaped inner quotes), AND —
    since openrouter/free can land on a DIFFERENT, sometimes flaky, free
    model on every single call — retries the WHOLE round-trip (a fresh LLM
    call, not just re-parsing the same broken text) up to `retries` extra
    times if the response still isn't valid JSON after repair. A malformed
    or truncated response is usually a one-off draw from the free-model
    pool; the next call very likely lands on a model that actually follows
    the "respond with ONLY JSON" instruction.

    Raises the LAST parse error if every attempt is exhausted — callers
    should catch and handle exactly as they would a bare claude_generate_json
    + json.loads failure."""
    timeout = timeout if timeout is not None else settings.request_timeout
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        raw = await asyncio.wait_for(
            claude_generate_json(system, messages, model=model, max_tokens=max_tokens),
            timeout=timeout,
        )
        try:
            data = parse_llm_json(raw)
        except json.JSONDecodeError as exc:
            last_exc = exc
            logger.warning(
                "generate_json: attempt %d/%d returned unparseable JSON (%s) — %s",
                attempt + 1, retries + 1, exc, "retrying" if attempt < retries else "giving up",
            )
            continue
        if not isinstance(data, dict):
            last_exc = ValueError(f"expected a JSON object, got {type(data).__name__}")
            continue
        return data
    raise last_exc  # type: ignore[misc]
