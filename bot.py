"""
Senior Business Analyst AI Orchestrator — Telegram bot entrypoint.

Powered exclusively by Claude (Anthropic). No other AI providers required.

Flow per message:
  1. Access control.
  2. In groups: optionally require @mention or reply.
  3. Router (Claude Haiku) classifies → (agent, request_type, complexity, model).
  4. Agent persona + request-type addendum answers with Claude Sonnet.
  5. Multi-agent collaboration when the router flags it (parallel calls, synthesized reply).
  6. GitHub issue/PR filed for actionable types (if configured).

Commands:
  /idea /task /bug /improve — force request type
  /kickoff — whole team responds in parallel (PM, BA, Data Analyst, Process Analyst)
  /proposal — deliver as Word + PDF document
  /remember /memory /forget — project memory management
  /logs — Railway deployment log analysis (if configured)
  /readfile — pull current file from GitHub into conversation
  /reset — clear chat history
  /status — show configuration health

Run:  python bot.py
"""

import asyncio
import logging
import re

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandObject
from aiogram.types import BufferedInputFile, Message

import document_generation as docgen
import github_integration
import history
import memory
import railway_integration
import redis_client
from agents import Agent, TEAM_MEMORY_HEADER, get_agent
from config import settings
from file_processing import SUPPORTED_SUMMARY, extract, transcribe_audio
from llm_clients import claude_generate, claude_generate_fast
from request_types import REQUEST_TYPES, get_request_type
from router import Route, classify, model_for

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("orchestrator")

dp = Dispatcher()

TELEGRAM_LIMIT = 4096

# Default cross-functional team for /kickoff: core BA team
KICKOFF_ROLES = ["ba", "pm", "data_analyst", "process_analyst"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _is_allowed(chat_id: int) -> bool:
    return not settings.allowed_chat_ids or chat_id in settings.allowed_chat_ids


def _should_answer(message: Message, bot_username: str) -> bool:
    if message.chat.type == "private":
        return True
    if not settings.require_mention_in_groups:
        return True
    if (
        message.reply_to_message
        and message.reply_to_message.from_user
        and message.reply_to_message.from_user.is_bot
    ):
        return True
    if bot_username and f"@{bot_username}".lower() in (message.text or "").lower():
        return True
    return False


def _strip_mention(text: str, bot_username: str) -> str:
    if bot_username:
        text = text.replace(f"@{bot_username}", "").replace(f"@{bot_username.lower()}", "")
    return text.strip()


async def _build_system_prompt(
    chat_id: int, agent: Agent, addendum: str = ""
) -> str:
    facts = await memory.get_memory(chat_id)
    memory_block = (
        TEAM_MEMORY_HEADER + "\n".join(f"- {f}" for f in facts) + "\n\n" if facts else ""
    )
    parts = [memory_block + agent.system]
    if addendum:
        parts.append(addendum)
    return "\n".join(parts)


def _header(route: Route) -> str:
    if not settings.show_metadata_header:
        return ""
    if route.request_type.key == "question":
        return f"_{route.agent.display_name} · {route.model_label}_\n\n"
    return (
        f"**[{route.agent.display_name}]** · {route.model_label} · "
        f"{route.request_type.label}\n\n"
    )


async def _answer_with_agent(route: Route, system_prompt: str, messages: list[dict]) -> str:
    """All agents use Claude — model choice (Sonnet vs Haiku) is in route.model."""
    return await claude_generate(system_prompt, messages, model=route.model)


async def _maybe_create_tickets(route: Route, user_text: str, body: str) -> str:
    if not route.request_type.creates_ticket or not settings.github_enabled:
        return ""

    labels = [l for l in [route.request_type.github_label, route.agent.key] if l]
    issue_body = f"**Original request:**\n{user_text}\n\n---\n\n{body}"
    issue_url = await github_integration.create_issue(route.title, issue_body, labels)

    pr_url = None
    if route.agent.key in ("system_analyst", "tech_consultant"):
        files = github_integration.extract_files(body)
        if files:
            pr_url = await github_integration.create_implementation_pr(
                route.title, issue_body, files, labels
            )

    lines = []
    if issue_url:
        lines.append(f"📋 Issue: {issue_url}")
    if pr_url:
        lines.append(f"🔀 Draft PR: {pr_url}")
    return ("\n\n" + "\n".join(lines)) if lines else ""


async def _send_long(message: Message, text: str) -> None:
    for chunk in _split(text, TELEGRAM_LIMIT):
        try:
            await message.answer(chunk, parse_mode="Markdown")
        except TelegramBadRequest:
            await message.answer(chunk, parse_mode=None)


def _split(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, current = [], ""
    for line in text.split("\n"):
        while len(line) > limit:
            if current:
                parts.append(current)
                current = ""
            parts.append(line[:limit])
            line = line[limit:]
        if len(current) + len(line) + 1 > limit:
            parts.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        parts.append(current)
    return parts


async def _send_document_deliverable(message: Message, route: Route, user_text: str) -> None:
    chat_id = message.chat.id
    status = await message.answer("📄 Hujjatni tayyorlayapman...")

    try:
        content = await asyncio.wait_for(
            docgen.generate_proposal_content(route.agent, user_text),
            timeout=settings.request_timeout,
        )
        docx_bytes = docgen.render_docx(content)
        pdf_bytes = docgen.render_pdf(content)
    except Exception as exc:
        logger.exception("Document generation failed")
        await status.edit_text(f"⚠️ Hujjatni tayyorlashda xatolik: {exc}")
        return

    try:
        await status.delete()
    except TelegramBadRequest:
        pass

    title = content.get("title") or "Hujjat"
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", title)[:50].strip("_") or "hujjat"
    caption = f"📄 {title}"
    if content.get("intro"):
        caption += "\n\n" + content["intro"][:900]

    await message.answer_document(
        BufferedInputFile(docx_bytes, filename=f"{slug}.docx"), caption=caption
    )
    await message.answer_document(BufferedInputFile(pdf_bytes, filename=f"{slug}.pdf"))

    await history.append(chat_id, route.agent.key, "user", user_text)
    await history.append(chat_id, route.agent.key, "assistant", f"[Hujjat tayyorlandi: {title}]")
    await history.set_last_route(chat_id, route.agent.key, route.request_type.key)


_CAPABILITY_OVERVIEW_SYSTEM = """
You are the BA AI team, answering in Telegram. The user asked what you can do.

Write 2-3 SHORT sentences, like a quick reply from a colleague. Cover the basics:
auto-routing to the right specialist (BA, Data/BI, Financial, Market, Process,
Engineering), /kickoff for team brainstorm, /proposal for Word/PDF output.

STRICT FORMAT RULES — no exceptions:
- Write as flowing prose. NO lists, NO bullet points, NO headers.
- NO markdown: no *, no **, no #, no -.
- Under 70 words total.
- Match the user's language (Uzbek / Russian / English).

WRONG: "Mening jamoamda quyidagi mutaxassislar bor:\n* Data Analyst: ..."
RIGHT: "Talablar, SQL, dashboard, moliyaviy model, bozor tahlili — nima kerak bo'lsa shunchaki yozing, o'zim to'g'ri mutaxassisga yo'naltiraman. Rasmiy hujjat kerak bo'lsa /proposal, butun jamoa bilan birga ishlash uchun /kickoff."
"""

_CAPABILITY_FALLBACK_UZ = (
    "Men senior biznes analitik ishlaringiz uchun butun jamoani birlashtiraman: "
    "talablar va user story yozish, SQL va KPI tahlili, Power BI/Tableau dashboard, "
    "moliyaviy modellashtirish, bozor tahlili, BPMN jarayon xaritalash, BRD/FRS "
    "hujjatlar, loyiha rejalashtirish va texnologiya maslahat. Shunchaki savolingizni "
    "yozing — o'zim to'g'ri mutaxassisga yo'naltirilaman."
)


async def _send_capability_overview(message: Message, bot: Bot, route: Route, user_text: str) -> None:
    chat_id = message.chat.id
    await bot.send_chat_action(chat_id, ChatAction.TYPING)
    try:
        body = await asyncio.wait_for(
            claude_generate_fast(
                _CAPABILITY_OVERVIEW_SYSTEM,
                [{"role": "user", "content": user_text}],
                temperature=0.5,
            ),
            timeout=settings.request_timeout,
        )
        body = body.strip() or _CAPABILITY_FALLBACK_UZ
    except Exception:
        logger.exception("Capability overview generation failed")
        body = _CAPABILITY_FALLBACK_UZ

    await _send_long(message, body)


# --------------------------------------------------------------------------
# Core pipeline
# --------------------------------------------------------------------------
async def _process(
    message: Message,
    bot: Bot,
    user_text: str | None,
    forced_type: str | None,
    forced_document: bool | None = None,
) -> None:
    chat_id = message.chat.id
    if not _is_allowed(chat_id):
        return

    if not user_text or not user_text.strip():
        await message.answer(
            "Matn yozing, masalan:\n"
            "/task Mijozlar bazasini tahlil qilish uchun SQL so'rov yozing"
        )
        return
    user_text = user_text.strip()

    await bot.send_chat_action(chat_id, ChatAction.TYPING)

    last = await history.get_last_route(chat_id)
    route = await classify(
        user_text,
        last_agent=last[0] if last else None,
        last_type=last[1] if last else None,
    )
    if forced_type:
        route.request_type = get_request_type(forced_type)
    if forced_document:
        route.wants_document = True

    if route.is_capability_question and not forced_type and not forced_document:
        logger.info("chat=%s -> capability overview", chat_id)
        await _send_capability_overview(message, bot, route, user_text)
        return

    if route.wants_document:
        logger.info("chat=%s -> agent=%s DOCUMENT", chat_id, route.agent.key)
        await _send_document_deliverable(message, route, user_text)
        return

    logger.info(
        "chat=%s -> agent=%s model=%s type=%s collaborators=%s",
        chat_id, route.agent.key, route.model, route.request_type.key, route.collaborators,
    )

    if route.collaborators:
        body = await _run_collaborative_answer(route, user_text, chat_id)
        await history.append(chat_id, route.agent.key, "user", user_text)
        await history.append(chat_id, route.agent.key, "assistant", body)
        await history.set_last_route(chat_id, route.agent.key, route.request_type.key)
        footer = await _maybe_create_tickets(route, user_text, body)
        await _send_long(message, _header(route) + body + footer)
        if route.request_type.creates_ticket:
            asyncio.create_task(_maybe_extract_memory(chat_id, user_text, body))
        return

    system_prompt = await _build_system_prompt(chat_id, route.agent, route.request_type.addendum)
    msgs = await history.get_history(chat_id, route.agent.key) + [
        {"role": "user", "content": user_text}
    ]

    try:
        body = await asyncio.wait_for(
            _answer_with_agent(route, system_prompt, msgs),
            timeout=settings.request_timeout,
        )
    except asyncio.TimeoutError:
        await message.answer("⏱ Model vaqtida javob bermadi. Qaytadan urinib ko'ring.")
        return
    except Exception as exc:
        logger.exception("Generation failed")
        msg = str(exc)
        if "529" in msg or "overloaded" in msg.lower() or "503" in msg:
            await message.answer(
                "⏳ Claude hozir juda band (overloaded). "
                "30 soniya kutib, qaytadan yuboring."
            )
        elif "401" in msg or "authentication" in msg.lower():
            await message.answer("⚠️ ANTHROPIC_API_KEY noto'g'ri yoki muddati o'tgan.")
        else:
            await message.answer(f"⚠️ Xatolik yuz berdi: {exc}")
        return

    if not body:
        body = "(bo'sh javob)"

    await history.append(chat_id, route.agent.key, "user", user_text)
    await history.append(chat_id, route.agent.key, "assistant", body)
    await history.set_last_route(chat_id, route.agent.key, route.request_type.key)

    footer = await _maybe_create_tickets(route, user_text, body)
    await _send_long(message, _header(route) + body + footer)
    if route.request_type.creates_ticket:
        asyncio.create_task(_maybe_extract_memory(chat_id, user_text, body))


# --------------------------------------------------------------------------
# /kickoff — whole BA team responds together
# --------------------------------------------------------------------------
async def _kickoff_agent_call(agent_key: str, user_text: str, chat_id: int) -> dict:
    agent: Agent = get_agent(agent_key)
    system_prompt = await _build_system_prompt(chat_id, agent, REQUEST_TYPES["task"].addendum)
    model, label = model_for(agent, "high")
    try:
        body = await asyncio.wait_for(
            claude_generate(system_prompt, [{"role": "user", "content": user_text}], model=model),
            timeout=settings.request_timeout,
        )
        return {"agent": agent, "model_label": label, "body": body or "(bo'sh javob)", "ok": True}
    except Exception as exc:
        logger.exception("Kickoff sub-call failed for agent=%s", agent_key)
        return {
            "agent": agent,
            "model_label": label,
            "body": f"⚠️ Bu rol javob bera olmadi: {exc}",
            "ok": False,
        }


_SYNTHESIS_SYSTEM = """
You are the lead voice presenting the team's combined answer to a Senior Business
Analyst, via their Telegram bot. You are given the original request and several
specialists' individual input. Merge them into ONE cohesive, complete answer —
NOT a sequence of separately labelled sections. Resolve overlaps, drop redundancy,
keep the most useful concrete specifics (real numbers, real steps, real templates),
and present it as if one excellent senior BA lead is speaking for the whole team.
Match the dominant language of the ORIGINAL request (Uzbek/Russian/English;
keep professional/business terms in English). Same human-tone rules apply:
length matches the substance, no boilerplate openers, no excessive markdown —
but use structure (headings/bullets) when the combined content genuinely needs it.
Make it self-contained and directly useful.
"""


async def _run_collaborative_answer(route: Route, user_text: str, chat_id: int) -> str:
    roles = [route.agent.key] + [k for k in route.collaborators if k != route.agent.key]
    roles = list(dict.fromkeys(roles))[:4]

    results = await asyncio.gather(*[_kickoff_agent_call(key, user_text, chat_id) for key in roles])

    for r in results:
        if r["ok"]:
            await history.append(chat_id, r["agent"].key, "user", user_text)
            await history.append(chat_id, r["agent"].key, "assistant", r["body"])

    ok_results = [r for r in results if r["ok"]]
    if not ok_results:
        return "Kechirasiz, jamoadan javob ololmadim. Birozdan keyin urinib ko'ring."

    contributions = "\n\n".join(
        f"--- {r['agent'].display_name}'s input ---\n{r['body']}" for r in ok_results
    )
    synthesis_input = f"ORIGINAL REQUEST:\n{user_text}\n\n{contributions}"

    try:
        body = await asyncio.wait_for(
            claude_generate(
                _SYNTHESIS_SYSTEM,
                [{"role": "user", "content": synthesis_input}],
                model=settings.claude_model,
            ),
            timeout=settings.request_timeout,
        )
        body = (body or "").strip()
    except Exception:
        logger.exception("Synthesis failed, using primary agent answer")
        body = ""

    if not body:
        primary = next((r for r in ok_results if r["agent"].key == route.agent.key), ok_results[0])
        body = primary["body"]

    return body


_MEMORY_EXTRACT_SYSTEM = """
Given this exchange between a Senior Business Analyst and their AI team, is there
ONE durable fact worth remembering for future conversations about this same project?
Examples: project name, confirmed tech stack, budget figure, established naming
convention, firm decision or constraint, key stakeholder name/role.
Do NOT record vague chit-chat, opinions, or anything not clearly decided/stated.
If yes: respond with ONLY that one fact, under 150 characters, in the same language.
If no durable fact: respond with exactly NONE.
"""


async def _maybe_extract_memory(chat_id: int, user_text: str, body: str) -> None:
    try:
        raw = await claude_generate_fast(
            _MEMORY_EXTRACT_SYSTEM,
            [{"role": "user", "content": f"REQUEST:\n{user_text}\n\nANSWER:\n{body[:2000]}"}],
            temperature=0.0,
        )
        fact = (raw or "").strip()
        if fact and fact.upper() != "NONE" and len(fact) < 300:
            await memory.add_memory(chat_id, fact)
    except Exception:
        logger.exception("Memory extraction failed (non-fatal)")


def _kickoff_header(results: list[dict]) -> str:
    if not settings.show_metadata_header:
        return ""
    agent_names = ", ".join(r["agent"].display_name for r in results)
    return f"**[BA TEAM KICKOFF]** — {agent_names}\n\n"


@dp.message(Command("kickoff"))
async def cmd_kickoff(message: Message, bot: Bot, command: CommandObject) -> None:
    chat_id = message.chat.id
    if not _is_allowed(chat_id):
        return

    user_text = (command.args or "").strip()
    if not user_text:
        await message.answer(
            "Butun BA jamoasini jalb qilish uchun so'rovni yozing:\n"
            "/kickoff Yangi mijoz onboarding jarayonini loyihalash"
        )
        return

    await bot.send_chat_action(chat_id, ChatAction.TYPING)
    results = await asyncio.gather(
        *[_kickoff_agent_call(key, user_text, chat_id) for key in KICKOFF_ROLES]
    )

    parts = [_kickoff_header(results)]
    for r in results:
        parts.append(f"#### {r['agent'].display_name}\n\n{r['body']}\n\n---\n")
        await history.append(chat_id, r["agent"].key, "user", user_text)
        await history.append(chat_id, r["agent"].key, "assistant", r["body"])

    await history.set_last_route(chat_id, "ba", "idea")

    footer = ""
    if settings.github_enabled:
        labels = ["idea"] + [r["agent"].key for r in results]
        title = user_text[:70] or "BA Team kickoff"
        issue_body = f"**Original request:**\n{user_text}\n\n---\n\n" + "\n\n".join(
            f"## {r['agent'].display_name}\n\n{r['body']}" for r in results
        )
        issue_url = await github_integration.create_issue(title, issue_body, labels)
        if issue_url:
            footer = f"\n\n📋 Issue: {issue_url}"

    await _send_long(message, "".join(parts) + footer)


# --------------------------------------------------------------------------
# File handling (images, documents — Claude handles natively)
# --------------------------------------------------------------------------
async def _handle_file(
    message: Message,
    bot: Bot,
    file_id: str,
    filename: str | None,
    mime_type: str | None,
    file_size: int | None,
    caption: str | None,
) -> None:
    chat_id = message.chat.id
    if not _is_allowed(chat_id):
        return

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if file_size and file_size > max_bytes:
        await message.answer(
            f"⚠️ Fayl juda katta ({file_size // (1024*1024)} MB). "
            f"Maksimal: {settings.max_file_size_mb} MB."
        )
        return

    status = await message.answer("📎 Faylni o'qiyapman...")
    try:
        buf = await bot.download(file_id)
        data = buf.read() if hasattr(buf, "read") else bytes(buf)
    except Exception as exc:
        logger.exception("File download failed")
        await status.edit_text(f"⚠️ Faylni yuklab olishda xatolik: {exc}")
        return

    extracted, kind_or_error = await extract(filename, mime_type, data)
    if extracted is None:
        await status.edit_text(kind_or_error)
        return

    label = filename or kind_or_error
    if caption and caption.strip():
        combined = f"{caption.strip()}\n\n--- Biriktirilgan fayl: {label} ---\n{extracted}"
    else:
        combined = (
            f"Quyidagi fayl yuborildi: {label}.\n"
            "Biznes analitik nuqtai nazaridan asosiy ma'lumotlarni tahlil qiling "
            "va eng foydali xulosa yoki keyingi qadamni taklif qiling.\n\n"
            f"--- Fayl tarkibi ---\n{extracted}"
        )

    try:
        await status.delete()
    except TelegramBadRequest:
        pass

    await _process(message, bot, combined, forced_type=None)


@dp.message(F.photo)
async def handle_photo(message: Message, bot: Bot) -> None:
    photo = message.photo[-1]
    await _handle_file(
        message, bot, photo.file_id, "photo.jpg", "image/jpeg", photo.file_size, message.caption
    )


@dp.message(F.document)
async def handle_document(message: Message, bot: Bot) -> None:
    doc = message.document
    await _handle_file(
        message, bot, doc.file_id, doc.file_name, doc.mime_type, doc.file_size, message.caption
    )


@dp.message(F.voice)
async def handle_voice(message: Message, bot: Bot) -> None:
    voice = message.voice
    chat_id = message.chat.id
    if not _is_allowed(chat_id):
        return

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if voice.file_size and voice.file_size > max_bytes:
        await message.answer("⚠️ Ovozli xabar juda katta.")
        return

    status = await message.answer("🎙 Eshityapman...")
    try:
        buf = await bot.download(voice.file_id)
        data = buf.read() if hasattr(buf, "read") else bytes(buf)
        transcript = await transcribe_audio(data, "voice.ogg")
    except Exception as exc:
        logger.exception("Voice transcription failed")
        await status.edit_text(f"⚠️ Ovozni tushunishda xatolik: {exc}")
        return

    user_text = transcript.strip() if transcript else ""
    if not user_text:
        await status.edit_text("⚠️ Ovozda matn aniqlanmadi.")
        return

    try:
        await status.delete()
    except TelegramBadRequest:
        pass

    await _process(message, bot, user_text, forced_type=None)


@dp.message(F.audio)
async def handle_audio(message: Message, bot: Bot) -> None:
    audio = message.audio
    await _handle_file(
        message, bot, audio.file_id, audio.file_name or "audio.mp3",
        audio.mime_type or "audio/mpeg", audio.file_size, message.caption,
    )


# --------------------------------------------------------------------------
# Command handlers
# --------------------------------------------------------------------------
@dp.message(Command("start", "help"))
async def cmd_start(message: Message) -> None:
    if not _is_allowed(message.chat.id):
        return
    await message.answer(
        "Salom! Senior BA AI jamoasi ishga tayyor.\n\n"
        "Nima kerak bo'lsa shunchaki yozing — BA, Data/BI, Financial, Market, "
        "Process, Engineering va boshqa mutaxassislarga o'zim yo'naltiraman.\n\n"
        "Asosiy buyruqlar:\n"
        "/kickoff — jamoa bilan birgalikda (BA + PM + Data + Process)\n"
        "/proposal — Word + PDF hujjat tayyorlash\n"
        "/remember <fakt> — loyiha ma'lumotini yodlash\n"
        "/memory — yodlangan faktlar\n"
        "/reset — suhbatni tozalash\n"
        "/status — tizim holati\n\n"
        f"Fayllar: {SUPPORTED_SUMMARY}\n"
        f"`chat_id: {message.chat.id}`",
        parse_mode="Markdown",
    )


@dp.message(Command("reset"))
async def cmd_reset(message: Message) -> None:
    if not _is_allowed(message.chat.id):
        return
    await history.reset(message.chat.id)
    await message.answer("Kontekst tozalandi. ✅")


@dp.message(Command("remember"))
async def cmd_remember(message: Message, command: CommandObject) -> None:
    if not _is_allowed(message.chat.id):
        return
    fact = (command.args or "").strip()
    if not fact:
        await message.answer("Faktni yozing:\n/remember Loyiha: EduConnect, stack: FastAPI + React")
        return
    await memory.add_memory(message.chat.id, fact)
    await message.answer("Yodda saqlandi. ✅ (`/memory` — ko'rish)", parse_mode="Markdown")


@dp.message(Command("memory"))
async def cmd_memory(message: Message) -> None:
    if not _is_allowed(message.chat.id):
        return
    facts = await memory.get_memory(message.chat.id)
    if not facts:
        await message.answer("Hozircha hech narsa yodda saqlanmagan.")
        return
    lines = "\n".join(f"{i+1}. {f}" for i, f in enumerate(facts))
    await _send_long(message, f"📌 Loyiha haqida faktlar:\n\n{lines}")


@dp.message(Command("forget"))
async def cmd_forget(message: Message) -> None:
    if not _is_allowed(message.chat.id):
        return
    n = await memory.clear_memory(message.chat.id)
    await message.answer(f"O'chirildi: {n} ta fakt. ✅")


@dp.message(Command("status"))
async def cmd_status(message: Message) -> None:
    if not _is_allowed(message.chat.id):
        return

    lines = ["**AI Orchestrator Status**\n"]
    if settings.provider == "openrouter":
        lines.append(f"**Provider:** ✅ OpenRouter (bepul)")
        lines.append(f"**Asosiy model:** {settings.or_main_model_label} (`{settings.or_main_model}`)")
        lines.append(f"**Tez model (routing):** {settings.or_fast_model_label}")
    elif settings.provider == "claude":
        lines.append(f"**Provider:** ✅ Claude (Anthropic)")
        lines.append(f"**Asosiy model:** {settings.claude_model_label}")
        lines.append(f"**Tez model:** {settings.claude_fast_model_label}")
    else:
        lines.append("**Provider:** ❌ Sozlanmagan (OPENROUTER_API_KEY yoki ANTHROPIC_API_KEY kerak)")
    redis_status = "✅ Persistent" if settings.redis_enabled else "⚠️ In-memory (restart da yo'oladi)"
    lines.append(f"\n**Redis:** {redis_status}")
    lines.append(f"**GitHub:** {'✅ ' + settings.github_repo if settings.github_enabled else '❌ Sozlanmagan'}")
    lines.append(f"**Railway logs:** {'✅' if settings.railway_enabled else '❌ Sozlanmagan'}")

    from agents import AGENTS
    lines.append(f"\n**Faol agentlar ({len(AGENTS)} ta):**")
    for key, agent in AGENTS.items():
        lines.append(f"  - {agent.display_name}")

    await _send_long(message, "\n".join(lines))


@dp.message(Command("loops", "loops"))
async def cmd_loops(message: Message) -> None:
    await cmd_status(message)


@dp.message(Command("id"))
async def cmd_id(message: Message) -> None:
    await message.answer(f"chat_id: `{message.chat.id}`", parse_mode="Markdown")


@dp.message(Command("idea"))
async def cmd_idea(message: Message, bot: Bot, command: CommandObject) -> None:
    await _process(message, bot, command.args, forced_type="idea")


@dp.message(Command("task"))
async def cmd_task(message: Message, bot: Bot, command: CommandObject) -> None:
    await _process(message, bot, command.args, forced_type="task")


@dp.message(Command("bug"))
async def cmd_bug(message: Message, bot: Bot, command: CommandObject) -> None:
    await _process(message, bot, command.args, forced_type="bug")


@dp.message(Command("improve"))
async def cmd_improve(message: Message, bot: Bot, command: CommandObject) -> None:
    await _process(message, bot, command.args, forced_type="improvement")


@dp.message(Command("proposal"))
async def cmd_proposal(message: Message, bot: Bot, command: CommandObject) -> None:
    await _process(message, bot, command.args, forced_type=None, forced_document=True)


@dp.message(Command("logs"))
async def cmd_logs(message: Message, bot: Bot) -> None:
    chat_id = message.chat.id
    if not _is_allowed(chat_id):
        return
    if not settings.railway_enabled:
        await message.answer(
            "Railway integratsiyasi sozlanmagan.\n"
            "RAILWAY_API_TOKEN qo'shing."
        )
        return

    status_msg = await message.answer("📜 Server loglarini olyapman...")
    deployment = await railway_integration.get_latest_deployment()
    if not deployment:
        await status_msg.edit_text("⚠️ Railway'dan ma'lumot olib bo'lmadi.")
        return

    logs = await railway_integration.get_recent_logs(limit=300)
    if logs is None:
        await status_msg.edit_text("⚠️ Loglarni olib bo'lmadi.")
        return
    if not logs:
        await status_msg.edit_text(f"Deployment status: {deployment.get('status')}. Loglar bo'sh.")
        return

    log_text = "\n".join(
        f"[{entry.get('severity', '')}] {entry.get('message', '')}" for entry in logs
    )[:12000]

    agent = get_agent("tech_consultant")
    system_prompt = await _build_system_prompt(
        chat_id, agent,
        "VAZIFA: quyidagi production server loglarini tahlil qiling. "
        "Xatolar bormi? Har birining sababi va tuzatish yo'lini ko'rsating.",
    )
    try:
        analysis = await asyncio.wait_for(
            claude_generate(
                system_prompt,
                [{"role": "user", "content": f"Status: {deployment.get('status')}\n\nLoglar:\n{log_text}"}],
            ),
            timeout=settings.request_timeout,
        )
    except Exception as exc:
        logger.exception("Log analysis failed")
        await status_msg.edit_text(f"⚠️ Loglarni tahlil qilishda xatolik: {exc}")
        return

    try:
        await status_msg.delete()
    except TelegramBadRequest:
        pass

    await _send_long(message, f"Status: {deployment.get('status')}\n\n{analysis}")


@dp.message(Command("readfile"))
async def cmd_readfile(message: Message, bot: Bot, command: CommandObject) -> None:
    chat_id = message.chat.id
    if not _is_allowed(chat_id):
        return
    path = (command.args or "").strip()
    if not path:
        await message.answer("Fayl yo'lini ko'rsating:\n/readfile requirements.md")
        return
    if not settings.github_enabled:
        await message.answer("GitHub integratsiyasi sozlanmagan (GITHUB_TOKEN / GITHUB_REPO).")
        return

    status_msg = await message.answer(f"📂 {path} o'qiyapman...")
    content = await github_integration.get_file_content(path)
    if content is None:
        await status_msg.edit_text(f"⚠️ Fayl topilmadi: {path}")
        return
    try:
        await status_msg.delete()
    except TelegramBadRequest:
        pass

    combined = (
        f"`{path}` faylining hozirgi holati:\n\n```\n" + content[:12000] + "\n```\n\n"
        "Yuqoridagi faylga asoslanib so'rovingizga javob bering."
    )
    await _process(message, bot, combined, forced_type=None)


# --------------------------------------------------------------------------
# Free-text handler
# --------------------------------------------------------------------------
@dp.message(F.text & ~F.text.startswith("/"))
async def handle_text(message: Message, bot: Bot) -> None:
    chat_id = message.chat.id
    if not _is_allowed(chat_id):
        return

    me = await bot.me()
    if not _should_answer(message, me.username or ""):
        return

    user_text = _strip_mention(message.text or "", me.username or "")
    if not user_text:
        return

    await _process(message, bot, user_text, forced_type=None)


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------
async def main() -> None:
    problems = settings.validate()
    for p in problems:
        logger.warning("CONFIG: %s", p)

    if settings.redis_enabled:
        if redis_client.get_client() is not None:
            logger.info("Redis: persistent storage ENABLED.")
        else:
            logger.warning("Redis URL set but client failed — falling back to in-memory.")
    else:
        logger.info("Redis: in-memory only (lost on restart). Set REDIS_URL to persist.")

    if settings.provider == "openrouter":
        logger.info("Provider: OpenRouter (free) — main=%s fast=%s",
                    settings.or_main_model, settings.or_fast_model)
    elif settings.provider == "claude":
        logger.info("Provider: Claude (Anthropic) — main=%s fast=%s",
                    settings.claude_model, settings.claude_fast_model)
    else:
        logger.warning("No AI provider configured! Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY.")

    if settings.github_enabled:
        logger.info("GitHub: repo=%s auto_pr=%s", settings.github_repo, settings.github_auto_pr)
    else:
        logger.info("GitHub: disabled.")

    if not settings.bot_token:
        raise SystemExit("BOT_TOKEN is required.")

    bot = Bot(
        token=settings.bot_token,
        default=DefaultBotProperties(parse_mode=None),
    )
    logger.info("Starting polling…")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
