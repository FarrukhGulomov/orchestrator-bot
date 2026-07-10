"""
Unified LLM client — OpenRouter (free), Claude (Anthropic), or Hybrid.

Provider modes (auto-detected from config.settings.provider):
  hybrid     — BOTH keys set: fast/routing → OpenRouter, main/vision → Claude
  openrouter — Only OR key: all calls use free models
  claude     — Only Anthropic key: all calls use Claude Sonnet/Haiku

Public API (provider-transparent):
  claude_generate()      — main model: agent responses, deep analysis
  claude_generate_fast() — fast model: routing, classification, memory
  claude_describe_file() — vision/PDF understanding
  claude_generate_json() — structured JSON (router, docgen)

Retry logic: 3x exponential backoff (2s → 4s → 8s) on 503/429/529.
"""

import asyncio
import base64
import io
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


def _is_model_unavailable(exc: Exception) -> bool:
    msg = str(exc)
    return "404" in msg or "No endpoints found" in msg or "model not found" in msg.lower()


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
# Public API — provider-transparent
# --------------------------------------------------------------------------
async def claude_generate(
    system: str,
    messages: list[dict],
    temperature: float = 0.4,
    model: str | None = None,
) -> str:
    """Main model: agent responses, deep analysis, document generation.

    hybrid — routes PER CALL based on the model id's shape: an OpenRouter id
    (always contains "/", e.g. from model_for()'s low-complexity branch —
    simple questions, saving Claude usage) goes to OpenRouter; a bare Claude
    id (e.g. "claude-sonnet-4-6", from high-complexity work) goes to Claude.
    claude     → always Claude.
    openrouter → always OpenRouter (free).
    """
    if settings.provider == "openrouter":
        m = model or settings.or_main_model
        return await asyncio.to_thread(
            _or_sync, m, system, messages, temperature, settings.max_output_tokens
        )
    if settings.provider == "hybrid" and model and "/" in model:
        return await asyncio.to_thread(
            _or_sync, model, system, messages, temperature, settings.max_output_tokens
        )
    # hybrid with a Claude id (or no explicit model), or claude-only.
    m = model or settings.claude_model
    return await asyncio.to_thread(
        _claude_sync, m, system, messages, temperature, settings.max_output_tokens
    )


async def claude_generate_fast(
    system: str,
    messages: list[dict],
    temperature: float = 0.0,
) -> str:
    """Fast model: routing, classification, memory extraction, relevance checks.

    hybrid/openrouter → OpenRouter fast free model
    claude            → Claude Haiku
    """
    if settings.provider in ("openrouter", "hybrid"):
        return await asyncio.to_thread(
            _or_sync, settings.or_fast_model, system, messages, temperature, 1024
        )
    return await asyncio.to_thread(
        _claude_sync, settings.claude_fast_model, system, messages, temperature, 1024
    )


async def claude_describe_file(data: bytes, mime_type: str, instruction: str) -> str:
    """Vision/PDF file understanding.

    hybrid/claude → Claude (native PDF + vision, better accuracy)
    openrouter    → OpenRouter vision model (images) + pypdf (PDFs)
    """
    if settings.provider in ("claude", "hybrid"):
        return await asyncio.to_thread(_claude_vision_sync, data, mime_type, instruction)
    return await asyncio.to_thread(_or_vision_sync, data, mime_type, instruction)


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


def _or_json_sync(system: str, messages: list[dict], model: str, max_tokens: int) -> str:
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


def _claude_json_sync(system: str, messages: list[dict], model: str, max_tokens: int) -> str:
    client = _get_claude_client()
    msgs = list(messages) + [{"role": "assistant", "content": "{"}]
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=model, system=system,
                messages=msgs, temperature=0.0, max_tokens=max_tokens,
            )
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

    In hybrid mode an EXPLICIT Claude model id must go to the Claude API —
    docgen/minutes pass model_for()'s result, which is a Claude id in hybrid;
    sending it to OpenRouter 404s and silently degrades to openrouter/auto.
    OpenRouter ids always contain "/" (vendor/model); Claude ids never do.
    """
    if settings.provider == "hybrid" and model and "/" not in model:
        raw = await asyncio.to_thread(_claude_json_sync, system, messages, model, max_tokens)
    elif settings.provider in ("openrouter", "hybrid"):
        m = model or settings.or_fast_model
        raw = await asyncio.to_thread(_or_json_sync, system, messages, m, max_tokens)
    else:
        m = model or settings.claude_fast_model
        raw = await asyncio.to_thread(_claude_json_sync, system, messages, m, max_tokens)
    # Applied unconditionally: a no-op for already-valid JSON (its string
    # literals never contain raw control chars), and fixes the single most
    # common way a free/weaker model breaks otherwise-valid JSON — see
    # _repair_json_control_chars's docstring.
    return _repair_json_control_chars(raw)
