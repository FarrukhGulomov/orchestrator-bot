"""
Claude (Anthropic) client wrappers.

All agent calls use the Anthropic SDK. Two tiers:
  - claude_generate()      — full model (Sonnet) for all agent responses
  - claude_generate_fast() — fast model (Haiku) for routing/classification

Message format used across the app is provider-neutral:
    messages = [{"role": "user"|"assistant", "content": "..."}, ...]
"""

import asyncio
import base64
import logging

from config import settings

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def _claude_sync(
    model: str,
    system: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> str:
    client = _get_client()
    resp = client.messages.create(
        model=model,
        system=system,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.content[0].text if resp.content else "").strip()


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
        # For non-image files (PDF etc.), send as base64 document
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
    # Claude doesn't have a native JSON mode — prefix the assistant turn to force JSON
    messages_with_prefix = list(messages) + [{"role": "assistant", "content": "{"}]
    resp = client.messages.create(
        model=settings.claude_fast_model,
        system=system,
        messages=messages_with_prefix,
        temperature=0.0,
        max_tokens=512,
    )
    raw = (resp.content[0].text if resp.content else "").strip()
    # Re-attach the opening brace we used as a prefix
    return "{" + raw if not raw.startswith("{") else raw


async def claude_generate_json(system: str, messages: list[dict]) -> str:
    return await asyncio.to_thread(_claude_json_sync, system, messages)
