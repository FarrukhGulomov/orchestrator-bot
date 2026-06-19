"""
Thin async wrappers around the Gemini (google-genai) and Groq SDKs.

Both SDKs are synchronous, so calls are pushed to a thread with asyncio.to_thread
to avoid blocking the aiogram event loop.

Message format used across the app is provider-neutral:
    messages = [{"role": "user"|"assistant", "content": "..."}, ...]
"""

import asyncio
import logging

from config import settings

logger = logging.getLogger(__name__)

# --- Lazy singletons ------------------------------------------------------
_gemini_client = None
_groq_client = None


def _get_gemini():
    global _gemini_client
    if _gemini_client is None:
        from google import genai  # imported lazily so the app starts without it

        _gemini_client = genai.Client(api_key=settings.gemini_api_key)
    return _gemini_client


def _get_groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq

        _groq_client = Groq(api_key=settings.groq_api_key)
    return _groq_client


# --- Gemini (ROUTE A) -----------------------------------------------------
def _gemini_sync(model: str, system: str, messages: list[dict]) -> str:
    from google.genai import types

    client = _get_gemini()
    contents = []
    for m in messages:
        role = "model" if m["role"] == "assistant" else "user"
        contents.append(
            types.Content(role=role, parts=[types.Part.from_text(text=m["content"])])
        )

    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.4,
            max_output_tokens=2048,
        ),
    )
    return (resp.text or "").strip()


async def gemini_generate(model: str, system: str, messages: list[dict]) -> str:
    return await asyncio.to_thread(_gemini_sync, model, system, messages)


def _gemini_describe_file_sync(data: bytes, mime_type: str, instruction: str) -> str:
    """One-off multimodal call: an image/PDF/etc. + an instruction, no chat
    history. Used by file_processing.py to turn uploaded files into text that
    then flows through the normal agent pipeline."""
    from google.genai import types

    client = _get_gemini()
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=data, mime_type=mime_type),
                types.Part.from_text(text=instruction),
            ],
        )
    ]
    resp = client.models.generate_content(
        model=settings.analysis_model,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=4096,
        ),
    )
    return (resp.text or "").strip()


async def gemini_describe_file(data: bytes, mime_type: str, instruction: str) -> str:
    return await asyncio.to_thread(_gemini_describe_file_sync, data, mime_type, instruction)


# --- Groq / Llama (ROUTE B + router) --------------------------------------
def _groq_sync(
    model: str, system: str, messages: list[dict], temperature: float, json_mode: bool
) -> str:
    client = _get_groq()
    full = [{"role": "system", "content": system}] + messages
    kwargs = {
        "model": model,
        "messages": full,
        "temperature": temperature,
        "max_tokens": 2048,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


async def groq_generate(
    model: str,
    system: str,
    messages: list[dict],
    temperature: float = 0.3,
    json_mode: bool = False,
) -> str:
    return await asyncio.to_thread(
        _groq_sync, model, system, messages, temperature, json_mode
    )


def _groq_transcribe_sync(data: bytes, filename: str) -> str:
    client = _get_groq()
    resp = client.audio.transcriptions.create(
        model=settings.whisper_model,
        file=(filename, data),
        response_format="text",
    )
    # response_format="text" -> SDK returns a plain string; other formats
    # return an object with a .text attribute. Handle both defensively.
    return resp if isinstance(resp, str) else getattr(resp, "text", str(resp))


async def groq_transcribe(data: bytes, filename: str) -> str:
    return await asyncio.to_thread(_groq_transcribe_sync, data, filename)
