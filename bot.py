"""
Master Orchestrator Telegram bot — entrypoint.

Flow per message:
  1. Access control: ignore chats not in ALLOWED_CHAT_IDS (if the list is set).
  2. In groups, optionally require an @mention or a reply to the bot.
  3. Router classifies -> (agent, request_type, complexity, title, model).
  4. The agent persona + request-type addendum answers with the chosen model,
     using that agent's own scoped chat history. For actionable types
     (task/bug/improvement/idea) the agent is instructed to DELIVER the
     implementation, not just discuss it.
  5. The bot deterministically prepends the metadata header so the backend
     can always parse it (the model never writes it itself).
  6. If GitHub integration is configured, actionable requests are filed as a
     GitHub Issue; if the agent is on the code route and produced file-marked
     code blocks and GITHUB_AUTO_PR=true, a draft PR is also opened.

Explicit commands /idea /task /bug /improve force the request type instead of
relying on the classifier (agent + model are still auto-routed).

/kickoff <text> fans the SAME request out to a FIXED cross-functional team
(PM, Product Designer, Backend, QA) and returns all four raw answers
side-by-side — an explicit "show me everyone's individual take" tool.

AUTOMATIC MULTI-AGENT COLLABORATION: separately from /kickoff, the router
itself decides per-message whether a request genuinely needs more than one
specialist (e.g. planning a feature needs PM + engineering input). When it
does, the bot automatically consults the relevant DYNAMIC set of roles in
parallel and SYNTHESIZES their input into one cohesive answer — the user
never has to ask for each role separately or know any commands exist. Most
messages are still single-domain and use the fast single-agent path; this
only engages when the classifier flags it.

PROJECT MEMORY ("self-learning", honestly described): durable facts about
the project (name, tech-stack decisions, budget figures, established
constraints) are auto-extracted from consequential exchanges and injected
into every agent's prompt going forward — see memory.py for what this
actually is and, importantly, what it is NOT (no model weights are ever
retrained; this is prompt-injected context, persisted via Redis when
configured, see history.py/memory.py/redis_client.py).

REAL DEVOPS ACCESS (read-only by design):
/logs — fetches the latest deployment status + recent runtime logs from this
bot's own Railway service (railway_integration.py) and has DevOps/SOC/Tech
Lead actually analyze them for real errors, not guess from a description.
/readfile <path> — pulls a file's CURRENT content from the configured GitHub
repo (github_integration.py) into the conversation, so REFINEMENT/BUG
requests are based on what's actually there, not generated blind. Combined
with the existing GITHUB_AUTO_PR flow, this is the "pull" half of
push/pull — "push" already existed via draft PR creation.
Deliberately NOT implemented: a chat command that directly triggers a
Railway redeploy or runs arbitrary commands on the server. The safe path is
already in place — a change becomes a draft PR, a human reviews and merges
it, Railway auto-deploys from the GitHub-linked service. A command that
skips that review step is a real risk for a production fintech system; see
railway_integration.py's docstring for the full reasoning.

/proposal <text> (also auto-detected by the classifier as wants_document) —
for requests that should be delivered as a polished Word + PDF file instead
of a chat reply (e.g. a commercial proposal / cost estimate), since pasting
that as raw chat text defeats the point of a shareable deliverable.

VOICE NOTE IN -> VOICE NOTE OUT: a Telegram voice note is transcribed, routed
and answered exactly like text, but the reply is generated in a short,
spoken-conversation style. For Russian/English the reply is sent back as
audio (Gemini TTS). For Uzbek, Gemini TTS has been confirmed by real testing
to produce the WRONG language (Kazakh) even with an explicit language hint —
most likely the model was never trained on real Uzbek speech — so for
Uzbek the reply is sent as TEXT instead of risking wrong-language audio.
See TTS_UNSUPPORTED_LANGUAGES in config.py to adjust this if that changes.

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
from llm_clients import gemini_generate, gemini_text_to_speech, glm_generate, glm_health_check, groq_generate

# Set to False at runtime if the GLM health-check at startup finds the model
# inaccessible (e.g. 404 / subscription required). Prevents error noise on
# every subsequent request — Route C falls back to Groq silently until the
# operator fixes the GLM config and restarts the bot.
_glm_runtime_ok: bool = True
from request_types import REQUEST_TYPES, get_request_type
from router import Route, classify, model_for

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("orchestrator")

dp = Dispatcher()

TELEGRAM_LIMIT = 4096

# Default cross-functional team for /kickoff: vision + design + core build + quality.
KICKOFF_ROLES = ["pm", "product_designer", "backend", "qa"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _is_allowed(chat_id: int) -> bool:
    # Empty allow-list => allow everything.
    return not settings.allowed_chat_ids or chat_id in settings.allowed_chat_ids


def _should_answer(message: Message, bot_username: str) -> bool:
    """In private chats: always. In groups: depends on mention/reply settings."""
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


async def _build_system_prompt(chat_id: int, agent: Agent, addendum: str = "") -> str:
    """Agent persona + project memory (if any) + request-type addendum.

    Centralised so every entry point (single-agent, collaborative synthesis,
    /kickoff, voice mode) gets the same memory-aware prompt instead of each
    building it ad hoc.
    """
    facts = await memory.get_memory(chat_id)
    memory_block = (
        TEAM_MEMORY_HEADER + "\n".join(f"- {f}" for f in facts) + "\n\n" if facts else ""
    )
    parts = [memory_block + agent.system]
    if addendum:
        parts.append(addendum)
    return "\n".join(parts)


def _header(route: Route) -> str:
    # Hidden by default (settings.show_metadata_header=False): replies should
    # read like they came from a person on the team, not visibly from a
    # multi-agent AI system. Routing info still goes to logs either way.
    if not settings.show_metadata_header:
        return ""
    # Casual, non-actionable chat gets a tiny tag instead of the full
    # ticket-style block — the heavy header is reserved for requests that
    # actually become tracked work, so everyday Q&A doesn't feel bureaucratic.
    if route.request_type.key == "question":
        return f"_{route.agent.display_name} · {route.model_label}_\n\n"
    if settings.metadata_format == "simple":
        return (
            f"[ACTIVE_AGENT]: {route.agent.display_name}\n"
            f"[ROUTED_MODEL]: {route.model_label}\n"
            f"[REQUEST_TYPE]: {route.request_type.label}\n\n"
        )
    # default: structured **[X_ORCHESTRATOR_METADATA]** block
    return (
        "**[X_ORCHESTRATOR_METADATA]**\n"
        f"- **CLASSIFICATION:** {route.request_type.classification_code}\n"
        f"- **ASSIGNED_AGENT:** {route.agent.display_name}\n"
        f"- **ROUTED_MODEL:** {route.model_label}\n\n"
        "---\n"
        f"### 🛠️ {route.agent.display_name} Response & Implementation\n\n"
    )


async def _answer_with_agent(route: Route, system_prompt: str, messages: list[dict]) -> str:
    """Dispatch to the right LLM based on the route's model selection.

    ROUTE A (Gemini): route.model is a Gemini model name.
    ROUTE B (Groq):   route.model is a Llama model on Groq, low-complexity.
    ROUTE C (GLM):    route.model is the configured GLM model via OpenAI-compat API.
                      Falls back to Groq if _glm_runtime_ok is False (startup 404).
    """
    global _glm_runtime_ok
    if route.agent.route == "A":
        return await gemini_generate(route.model, system_prompt, messages)
    # For technical routes, check if this is the GLM model
    if settings.glm_enabled and _glm_runtime_ok and route.model == settings.glm_model:
        return await glm_generate(route.model, system_prompt, messages, temperature=0.3)
    # Default: Groq / Llama (Route B) — also the fallback if GLM is down
    groq_model = (
        route.model if route.model != settings.glm_model
        else settings.code_model_fast
    )
    return await groq_generate(groq_model, system_prompt, messages, temperature=0.3)


async def _maybe_create_tickets(route: Route, user_text: str, body: str) -> str:
    """File a GitHub issue (and maybe a draft PR) for actionable request types.

    Returns a footer string with links, or "" if nothing was created
    (GitHub not configured, or this is a plain question).
    """
    if not route.request_type.creates_ticket or not settings.github_enabled:
        return ""

    labels = [l for l in [route.request_type.github_label, route.agent.key] if l]
    issue_body = f"**Original request:**\n{user_text}\n\n---\n\n{body}"

    issue_url = await github_integration.create_issue(route.title, issue_body, labels)

    pr_url = None
    if route.agent.route == "B":
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
    """Send text, splitting on Telegram's 4096-char limit.

    Tries legacy Markdown first (so code blocks render), falls back to plain text
    if Telegram rejects the markup.
    """
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
    """Generate a polished Word + PDF document instead of a chat reply, for
    requests that should be a shareable file (commercial proposal, cost
    estimate, formal report). Skips ticket-filing/GitHub by design — this is
    a deliverable handed directly to the user, not a tracked work item."""
    chat_id = message.chat.id
    status = await message.answer("📄 Hujjatni tayyorlayapman...")

    try:
        content = await asyncio.wait_for(
            docgen.generate_proposal_content(route.agent, user_text),
            timeout=settings.request_timeout,
        )
        docx_bytes = docgen.render_docx(content)
        pdf_bytes = docgen.render_pdf(content)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Document generation failed")
        await status.edit_text(f"⚠️ Hujjatni tayyorlashda xatolik: {exc}")
        return

    try:
        await status.delete()
    except TelegramBadRequest:
        pass

    title = content.get("title") or "Taklif"
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", title)[:50].strip("_") or "Taklif"
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
You are answering on behalf of the WHOLE in-house AI team in this Telegram
bot — not a single specialist. The user is asking what you/the team can do.
Give a SHORT, warm, human-sounding answer, not a bureaucratic role list.
Briefly mention the team covers product/business work (requirements,
roadmap, user stories, test plans, UX/UI design) AND engineering work
(backend, frontend, infrastructure, security, code review) — then 1-2
sentences on how to use it: just describe what's needed and it routes to
the right person automatically, /kickoff pulls in the whole team at once,
/proposal returns a ready Word/PDF document. Keep it under ~80 words, plain
conversational prose, no headers, no bullet list, no markdown decoration.
Match the dominant language of the user's message (Uzbek/Russian/English;
keep IT/product terms in English as usual).
"""

_CAPABILITY_FALLBACK_UZ = (
    "Jamoam mahsulot va texnik ishlarning deyarli barchasini qamrab oladi: "
    "talablar va backlog, user story va test rejalari, UX/UI dizayn, backend/"
    "frontend kod, infratuzilma va xavfsizlik. Shunchaki nima kerakligini "
    "yozing — kerakli odamga avtomatik yo'naltiraman, yoki /kickoff bilan "
    "butun jamoani bir vaqtda jalb qiling."
)


async def _send_capability_overview(message: Message, bot: Bot, route: Route, user_text: str) -> None:
    """Meta-questions ('what can you do') must NOT be answered by a single
    narrow specialist — that's the bug this fixes. Short, whole-team answer,
    no header (consistent with hide-metadata default)."""
    chat_id = message.chat.id
    await bot.send_chat_action(chat_id, ChatAction.TYPING)
    try:
        body = await asyncio.wait_for(
            groq_generate(
                settings.router_model,
                _CAPABILITY_OVERVIEW_SYSTEM,
                [{"role": "user", "content": user_text}],
                temperature=0.5,
            ),
            timeout=settings.request_timeout,
        )
        body = body.strip() or _CAPABILITY_FALLBACK_UZ
    except Exception:  # noqa: BLE001
        logger.exception("Capability overview generation failed, using fallback")
        body = _CAPABILITY_FALLBACK_UZ

    await _send_long(message, body)


# --------------------------------------------------------------------------
# Core pipeline (shared by free-text messages and /idea /task /bug /improve)
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
            "Напишите текст после команды, например:\n"
            "/bug при логине падает 500 на /api/auth"
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
        logger.info("chat=%s -> capability overview (whole team)", chat_id)
        await _send_capability_overview(message, bot, route, user_text)
        return

    if route.wants_document:
        logger.info(
            "chat=%s -> agent=%s DOCUMENT deliverable (Word+PDF)",
            chat_id, route.agent.key,
        )
        await _send_document_deliverable(message, route, user_text)
        return

    logger.info(
        "chat=%s -> agent=%s route=%s model=%s type=%s collaborators=%s",
        chat_id,
        route.agent.key,
        route.agent.route,
        route.model,
        route.request_type.key,
        route.collaborators,
    )

    if route.collaborators:
        # Request needs more than one perspective — gather the team
        # automatically instead of making the user ask for each role.
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
            _answer_with_agent(route, system_prompt, msgs), timeout=settings.request_timeout
        )
    except asyncio.TimeoutError:
        await message.answer("⏱ Модель не успела ответить вовремя. Попробуйте ещё раз.")
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("Generation failed")
        await message.answer(f"⚠️ Ошибка при обращении к модели: {exc}")
        return

    if not body:
        body = "(пустой ответ от модели)"

    await history.append(chat_id, route.agent.key, "user", user_text)
    await history.append(chat_id, route.agent.key, "assistant", body)
    await history.set_last_route(chat_id, route.agent.key, route.request_type.key)

    footer = await _maybe_create_tickets(route, user_text, body)
    await _send_long(message, _header(route) + body + footer)
    if route.request_type.creates_ticket:
        asyncio.create_task(_maybe_extract_memory(chat_id, user_text, body))


# --------------------------------------------------------------------------
# /kickoff — whole team responds together
# --------------------------------------------------------------------------
async def _kickoff_agent_call(agent_key: str, user_text: str, chat_id: int) -> dict:
    """Run one team member's reply (used by both /kickoff and automatic
    multi-agent collaboration). Never raises — returns an error note in
    'body' instead, so one failing teammate doesn't blank out the others.
    Routes to Gemini (A), GLM-5.2 (C for high complexity), or Groq (B)."""
    agent: Agent = get_agent(agent_key)
    system_prompt = await _build_system_prompt(chat_id, agent, REQUEST_TYPES["task"].addendum)
    model, label = model_for(agent, "high")
    try:
        # Build a minimal Route-like object and reuse _answer_with_agent so
        # the Route C (GLM) logic is in one place, not duplicated here.
        from dataclasses import dataclass as _dc
        from request_types import REQUEST_TYPES as _RT

        _mini_route = type("_R", (), {
            "agent": agent,
            "model": model,
        })()
        body = await asyncio.wait_for(
            _answer_with_agent(_mini_route, system_prompt, [{"role": "user", "content": user_text}]),  # type: ignore[arg-type]
            timeout=settings.request_timeout,
        )
        return {"agent": agent, "model_label": label, "body": body or "(пустой ответ)", "ok": True}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Kickoff sub-call failed for agent=%s", agent_key)
        return {
            "agent": agent,
            "model_label": label,
            "body": f"⚠️ Не удалось получить ответ от этой роли: {exc}",
            "ok": False,
        }


_SYNTHESIS_SYSTEM = """
You are the lead voice presenting the team's COMBINED answer to the user, in
this Telegram-based AI product team for a fintech company. You are given the
original request and several specialists' individual input below. Merge
them into ONE cohesive, complete answer — NOT a sequence of separately
labelled sections, not a meeting transcript. Resolve overlaps, drop
redundancy, keep the most useful concrete specifics from each specialist
(real numbers, real next steps, real code where given), and present it as
if one excellent, senior, friendly product lead is speaking on behalf of
the whole team. Match the dominant language of the ORIGINAL request (Uzbek/
Russian/English; keep IT/product terms in English as usual). Same human-tone
rules as always apply: length matches the substance, no boilerplate openers,
no excessive markdown decoration — but use structure (headings/bullets)
where the combined content genuinely needs it (e.g. a real multi-phase
plan). This may become a tracked ticket, so make it self-contained.
"""


async def _run_collaborative_answer(route: Route, user_text: str, chat_id: int) -> str:
    """Automatic multi-agent collaboration: fan out to the primary agent +
    its collaborators in parallel, then synthesize ONE combined answer.
    Each contributor's own exchange is still saved to ITS OWN history scope
    (same isolation as /kickoff), so their individual context keeps growing
    even though the user only ever sees the synthesized result."""
    roles = [route.agent.key] + [k for k in route.collaborators if k != route.agent.key]
    roles = list(dict.fromkeys(roles))[:4]  # dedupe, cap total team size at 4

    results = await asyncio.gather(*[_kickoff_agent_call(key, user_text, chat_id) for key in roles])

    for r in results:
        if r["ok"]:
            await history.append(chat_id, r["agent"].key, "user", user_text)
            await history.append(chat_id, r["agent"].key, "assistant", r["body"])

    ok_results = [r for r in results if r["ok"]]
    if not ok_results:
        return "Kechirasiz, jamoadan javob ololmadim, birozdan keyin qaytadan urinib ko'ring."

    contributions = "\n\n".join(
        f"--- {r['agent'].display_name}'s input ---\n{r['body']}" for r in ok_results
    )
    synthesis_input = f"ORIGINAL REQUEST:\n{user_text}\n\n{contributions}"

    try:
        body = await asyncio.wait_for(
            gemini_generate(
                settings.analysis_model,
                _SYNTHESIS_SYSTEM,
                [{"role": "user", "content": synthesis_input}],
            ),
            timeout=settings.request_timeout,
        )
        body = (body or "").strip()
    except Exception:  # noqa: BLE001
        logger.exception("Synthesis failed, falling back to primary agent's raw answer")
        body = ""

    if not body:
        primary = next((r for r in ok_results if r["agent"].key == route.agent.key), ok_results[0])
        body = primary["body"]

    return body


_MEMORY_EXTRACT_SYSTEM = """
Given this exchange between a user and an AI product team, is there ONE
durable fact worth remembering for future conversations about this same
project — e.g. the project's name, a confirmed tech-stack choice, a
budget/timeline figure, an established naming convention, a firm decision
or constraint? Do NOT record vague chit-chat, opinions, or anything not
clearly decided/stated as fact.
If yes: respond with ONLY that one fact, under 150 characters, in the same
language as the exchange.
If there is no durable fact: respond with exactly NONE.
"""


async def _maybe_extract_memory(chat_id: int, user_text: str, body: str) -> None:
    """Runs in the background after a response is already sent — never
    blocks or risks the user-facing reply."""
    try:
        raw = await groq_generate(
            settings.router_model,
            _MEMORY_EXTRACT_SYSTEM,
            [{"role": "user", "content": f"REQUEST:\n{user_text}\n\nANSWER:\n{body[:2000]}"}],
            temperature=0.0,
        )
        fact = (raw or "").strip()
        if fact and fact.upper() != "NONE" and len(fact) < 300:
            await memory.add_memory(chat_id, fact)
    except Exception:  # noqa: BLE001
        logger.exception("Memory extraction failed (non-fatal)")


def _kickoff_header(results: list[dict]) -> str:
    # Per-role section markers ("#### 🛠️ {name}") stay regardless — without
    # them a combined multi-answer message is unreadable, and that's the
    # whole point of /kickoff. Only the routing-metadata block is gated.
    if not settings.show_metadata_header:
        return ""
    agent_names = ", ".join(r["agent"].display_name for r in results)
    model_labels = ", ".join(sorted({r["model_label"] for r in results}))
    if settings.metadata_format == "simple":
        return (
            f"[ACTIVE_AGENTS]: {agent_names}\n"
            f"[ROUTED_MODELS]: {model_labels}\n"
            "[REQUEST_TYPE]: 🧩 Team Kickoff\n\n"
        )
    return (
        "**[X_ORCHESTRATOR_METADATA]**\n"
        "- **CLASSIFICATION:** TEAM KICKOFF\n"
        f"- **ASSIGNED_AGENTS:** {agent_names}\n"
        f"- **ROUTED_MODELS:** {model_labels}\n\n"
        "---\n"
        "### 🧩 Team Kickoff Response\n\n"
    )


@dp.message(Command("kickoff"))
async def cmd_kickoff(message: Message, bot: Bot, command: CommandObject) -> None:
    chat_id = message.chat.id
    if not _is_allowed(chat_id):
        return

    user_text = (command.args or "").strip()
    if not user_text:
        await message.answer(
            "Butun jamoani jalb qilish uchun g'oyani yozing, masalan:\n"
            "/kickoff Ota-onalar va o'quv markazlarini bog'lovchi platforma"
        )
        return

    await bot.send_chat_action(chat_id, ChatAction.TYPING)

    results = await asyncio.gather(
        *[_kickoff_agent_call(key, user_text, chat_id) for key in KICKOFF_ROLES]
    )

    parts = [_kickoff_header(results)]
    for r in results:
        parts.append(f"#### 🛠️ {r['agent'].display_name}\n\n{r['body']}\n\n---\n")
        await history.append(chat_id, r["agent"].key, "user", user_text)
        await history.append(chat_id, r["agent"].key, "assistant", r["body"])

    # Anchor follow-ups (e.g. "ok continue") back to PM by default after a kickoff.
    await history.set_last_route(chat_id, "pm", "idea")

    footer = ""
    if settings.github_enabled:
        labels = ["idea"] + [r["agent"].key for r in results]
        title = user_text[:70] or "Team kickoff"
        issue_body = f"**Original request:**\n{user_text}\n\n---\n\n" + "\n\n".join(
            f"## {r['agent'].display_name}\n\n{r['body']}" for r in results
        )
        issue_url = await github_integration.create_issue(title, issue_body, labels)

        pr_url = None
        backend_result = next(
            (r for r in results if r["agent"].key == "backend" and r["ok"]), None
        )
        if backend_result:
            files = github_integration.extract_files(backend_result["body"])
            if files:
                pr_url = await github_integration.create_implementation_pr(
                    title, issue_body, files, labels
                )

        lines = []
        if issue_url:
            lines.append(f"📋 Issue: {issue_url}")
        if pr_url:
            lines.append(f"🔀 Draft PR: {pr_url}")
        footer = ("\n\n" + "\n".join(lines)) if lines else ""

    await _send_long(message, "".join(parts) + footer)


# --------------------------------------------------------------------------
# Voice notes — transcribe, route, answer in spoken style, reply with audio
# --------------------------------------------------------------------------
_VOICE_MODE_ADDENDUM = """
VOICE REPLY MODE.
This answer will be converted to speech and sent back as a spoken audio
reply — the user is talking to you, not reading a document. Answer the way
you'd actually talk: short, natural sentences, conversational. NO headers,
NO bullet lists, NO markdown formatting, NO code blocks, NO file-marker
conventions. If the request genuinely needs code or a long structured
deliverable, say briefly that you've prepared it and they'll see the full
details in writing — do not try to read code or a table aloud. Keep the
whole answer well under what would take ~45 seconds to say out loud.

LANGUAGE: reply in ONLY Uzbek, Russian, or English — whichever best matches
what the user actually spoke. Never use any other language for this reply,
even if the transcript looks like it might be in a different language; pick
whichever of these three is the closest match.
"""

# Voice replies are restricted to exactly these three languages, end to end:
# the agent is instructed (above) to only ever answer in one of them, and the
# TTS call is given an EXPLICIT language_code from this same fixed set.
# NOTE: even with an explicit language_code, Gemini TTS has been confirmed
# (by real testing) to still produce the wrong language for Uzbek (Kazakh
# audio) — most likely the underlying model was simply never trained on real
# Uzbek speech, so no parameter fixes it. See tts_unsupported_languages in
# config.py: languages in that set skip TTS entirely and get a text reply.
_VOICE_LANGUAGE_CODES = {"uz": "uz-UZ", "ru": "ru-RU", "en": "en-US"}
_DEFAULT_VOICE_LANGUAGE = "uz"  # team's primary language; safe fallback on ambiguity

_LANGUAGE_DETECT_SYSTEM = """
Identify the dominant language of the given text. Respond with ONLY one of
these three codes, nothing else, no punctuation: uz, ru, en.
uz = Uzbek, ru = Russian, en = English.
The text may contain mixed IT/business loanwords in English even when the
dominant language is Uzbek or Russian — judge by the dominant language, not
individual loanwords. If genuinely ambiguous or it looks like some other
language entirely, pick whichever of uz/ru/en is the closest match — never
answer with anything other than one of these three codes.
"""


async def _detect_voice_language(text: str) -> str:
    try:
        raw = await groq_generate(
            settings.router_model,
            _LANGUAGE_DETECT_SYSTEM,
            [{"role": "user", "content": text[:1000]}],
            temperature=0.0,
        )
        code = raw.strip().lower()[:2]
        if code in _VOICE_LANGUAGE_CODES:
            return code
    except Exception:  # noqa: BLE001
        logger.exception("Voice language detection failed, using default")
    return _DEFAULT_VOICE_LANGUAGE


async def _handle_voice_note(
    message: Message,
    bot: Bot,
    file_id: str,
    mime_type: str | None,
    file_size: int | None,
    caption: str | None,
) -> None:
    chat_id = message.chat.id
    if not _is_allowed(chat_id):
        return

    if not settings.voice_replies_enabled:
        # Feature off -> behave like any other inbound file (transcribe,
        # answer as normal text).
        await _handle_file(message, bot, file_id, "voice.ogg", mime_type, file_size, caption)
        return

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if file_size and file_size > max_bytes:
        await message.answer(
            f"⚠️ Ovozli xabar juda katta ({file_size // (1024*1024)} MB). "
            f"Maksimal hajm: {settings.max_file_size_mb} MB."
        )
        return

    status = await message.answer("🎙 Eshityapman...")
    try:
        buf = await bot.download(file_id)
        data = buf.read() if hasattr(buf, "read") else bytes(buf)
        transcript = await transcribe_audio(data, "voice.ogg")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Voice transcription failed")
        await status.edit_text(f"⚠️ Ovozli xabarni tushunishda xatolik: {exc}")
        return

    user_text = f"{caption.strip()}\n\n{transcript}" if caption and caption.strip() else transcript
    user_text = (user_text or "").strip()
    if not user_text:
        await status.edit_text("⚠️ Ovozli xabarda matn aniqlanmadi, qaytadan urinib ko'ring.")
        return

    last = await history.get_last_route(chat_id)
    route = await classify(
        user_text,
        last_agent=last[0] if last else None,
        last_type=last[1] if last else None,
    )
    # Voice mode keeps things simple by design: always a short spoken answer
    # from the routed agent (no automatic multi-agent collaboration here —
    # a voice note is a quick spoken exchange, not a planning session).
    # Document-deliverable / capability-overview / GitHub-ticket paths are
    # intentionally not wired in either.
    system_prompt = await _build_system_prompt(chat_id, route.agent, _VOICE_MODE_ADDENDUM)
    msgs = await history.get_history(chat_id, route.agent.key) + [
        {"role": "user", "content": user_text}
    ]

    await bot.send_chat_action(chat_id, ChatAction.RECORD_VOICE)
    try:
        body = await asyncio.wait_for(
            _answer_with_agent(route, system_prompt, msgs), timeout=settings.request_timeout
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Voice-mode generation failed")
        await status.edit_text(f"⚠️ Javob tayyorlashda xatolik: {exc}")
        return
    body = (body or "").strip() or "Kechirasiz, javob topa olmadim."
    lang_code = await _detect_voice_language(body)

    if lang_code in settings.tts_unsupported_languages:
        # Known-bad combination (Uzbek -> Gemini TTS produces Kazakh audio,
        # confirmed by real testing, even with an explicit language_code).
        # Sending wrong-language audio is worse than just sending text — so
        # for these languages, skip the TTS attempt entirely.
        try:
            await status.delete()
        except TelegramBadRequest:
            pass
        await history.append(chat_id, route.agent.key, "user", user_text)
        await history.append(chat_id, route.agent.key, "assistant", body)
        await history.set_last_route(chat_id, route.agent.key, route.request_type.key)
        await _send_long(message, body)
        return

    try:
        audio_bytes = await asyncio.wait_for(
            gemini_text_to_speech(body, _VOICE_LANGUAGE_CODES[lang_code]),
            timeout=settings.request_timeout,
        )
    except Exception:  # noqa: BLE001
        # TTS failing (e.g. unsupported language) must not lose the answer —
        # fall back to a normal text reply instead.
        logger.exception("TTS failed, falling back to text reply")
        try:
            await status.delete()
        except TelegramBadRequest:
            pass
        await history.append(chat_id, route.agent.key, "user", user_text)
        await history.append(chat_id, route.agent.key, "assistant", body)
        await history.set_last_route(chat_id, route.agent.key, route.request_type.key)
        await _send_long(message, body)
        return

    try:
        await status.delete()
    except TelegramBadRequest:
        pass

    await history.append(chat_id, route.agent.key, "user", user_text)
    await history.append(chat_id, route.agent.key, "assistant", body)
    await history.set_last_route(chat_id, route.agent.key, route.request_type.key)

    caption_text = body if len(body) <= 1000 else body[:1000] + "…"
    await message.answer_audio(
        BufferedInputFile(audio_bytes, filename="javob.wav"), caption=caption_text
    )


# --------------------------------------------------------------------------
# File understanding — photo / document / voice / audio all funnel here
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
            f"Maksimal hajm: {settings.max_file_size_mb} MB."
        )
        return

    status = await message.answer("📎 Faylni o'qiyapman...")
    try:
        buf = await bot.download(file_id)
        data = buf.read() if hasattr(buf, "read") else bytes(buf)
    except Exception as exc:  # noqa: BLE001
        logger.exception("File download failed")
        await status.edit_text(f"⚠️ Faylni yuklab olishda xatolik: {exc}")
        return

    extracted, kind_or_error = await extract(filename, mime_type, data)
    if extracted is None:
        # kind_or_error is actually the user-facing error/explanation message here.
        await status.edit_text(kind_or_error)
        return

    label = filename or kind_or_error
    if caption and caption.strip():
        combined = f"{caption.strip()}\n\n--- Biriktirilgan fayl: {label} ---\n{extracted}"
    else:
        combined = (
            f"Foydalanuvchi izohsiz quyidagi faylni yubordi: {label}.\n"
            "Ushbu fayl tarkibini o'z roli nuqtai nazaridan tahlil qiling va "
            "eng foydali xulosa yoki keyingi qadamni taklif qiling.\n\n"
            f"--- Fayl tarkibi ---\n{extracted}"
        )

    try:
        await status.delete()
    except TelegramBadRequest:
        pass

    await _process(message, bot, combined, forced_type=None)


@dp.message(F.photo)
async def handle_photo(message: Message, bot: Bot) -> None:
    photo = message.photo[-1]  # largest size
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
    await _handle_voice_note(
        message, bot, voice.file_id, voice.mime_type or "audio/ogg",
        voice.file_size, message.caption,
    )


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
        "Master Orchestrator онлайн.\n"
        "Напишите задачу на UZ / RU / EN — я выберу подходящего агента "
        "(PM, BA, System Analyst, QA, Product Designer, Backend, Frontend, "
        "DevOps, SOC, Tech Lead) и оптимальную модель. Если запрос требует "
        "нескольких ролей сразу — сам соберу нужных и пришлю один общий "
        "ответ, без отдельных команд для каждой роли.\n\n"
        "Можно присылать файлы — прочитаю и обработаю как обычное сообщение: "
        f"{SUPPORTED_SUMMARY}.\n\n"
        "Голосовое сообщение — отвечу тоже голосом, коротко и по-человечески.\n\n"
        "Тип запроса можно задать явно:\n"
        "/idea <текст> — идея / предложение\n"
        "/task <текст> — задача на реализацию\n"
        "/bug <текст> — баг, нужен фикс\n"
        "/improve <текст> — доработка / рефакторинг\n"
        "/kickoff <текст> — все 4 ключевые роли сразу, каждая отдельно\n"
        "/proposal <текст> — готовый документ (Word+PDF), например коммерческое "
        "предложение или смета ресурсов\n"
        "Без команды — отвечу как на обычный вопрос, без тикета.\n\n"
        "/remember <факт> — запомнить факт о проекте\n"
        "/memory — показать, что запомнено\n"
        "/forget — забыть всё\n"
        "/logs — реальные логи и статус деплоя сервера (Railway), DevOps "
        "проанализирует ошибки\n"
        "/readfile <путь> — подтянуть текущий код файла из GitHub перед "
        "доработкой\n"
        "/reset — очистить контекст диалога\n"
        f"chat_id: `{message.chat.id}`",
        parse_mode="Markdown",
    )


@dp.message(Command("reset"))
async def cmd_reset(message: Message) -> None:
    if not _is_allowed(message.chat.id):
        return
    await history.reset(message.chat.id)
    await message.answer("Контекст очищен. ✅")


@dp.message(Command("remember"))
async def cmd_remember(message: Message, command: CommandObject) -> None:
    if not _is_allowed(message.chat.id):
        return
    fact = (command.args or "").strip()
    if not fact:
        await message.answer("Eslab qolish kerak bo'lgan faktni yozing:\n/remember Loyiha nomi: EduConnect")
        return
    await memory.add_memory(message.chat.id, fact)
    await message.answer("Yodda saqlandi. ✅ (`/memory` — ro'yxatni ko'rish)", parse_mode="Markdown")


@dp.message(Command("memory"))
async def cmd_memory(message: Message) -> None:
    if not _is_allowed(message.chat.id):
        return
    facts = await memory.get_memory(message.chat.id)
    if not facts:
        await message.answer("Hozircha hech narsa yodda saqlanmagan.")
        return
    lines = "\n".join(f"{i+1}. {f}" for i, f in enumerate(facts))
    await _send_long(message, f"📌 Loyiha haqida saqlangan faktlar:\n\n{lines}")


@dp.message(Command("forget"))
async def cmd_forget(message: Message) -> None:
    if not _is_allowed(message.chat.id):
        return
    n = await memory.clear_memory(message.chat.id)
    await message.answer(f"O'chirildi: {n} ta fakt. ✅")


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
    """Real bug-hunting on the live server: fetch the latest Railway
    deployment status + runtime logs and have DevOps actually analyze them,
    instead of answering a 'check the logs' question from guesswork."""
    chat_id = message.chat.id
    if not _is_allowed(chat_id):
        return
    if not settings.railway_enabled:
        await message.answer(
            "Railway integratsiyasi sozlanmagan.\n"
            "RAILWAY_API_TOKEN qo'shing (Railway → Account Settings → Tokens). "
            "Project/Environment/Service ID odatda avtomatik mavjud bo'ladi "
            "(Railway o'zi shu botning konteyneriga joylaydi)."
        )
        return

    status_msg = await message.answer("📜 Server loglarini olyapman...")
    deployment = await railway_integration.get_latest_deployment()
    if not deployment:
        await status_msg.edit_text(
            "⚠️ Railway'dan ma'lumot olib bo'lmadi — token yoki "
            "Project/Environment/Service ID noto'g'ri bo'lishi mumkin."
        )
        return

    logs = await railway_integration.get_recent_logs(limit=300)
    if logs is None:
        await status_msg.edit_text("⚠️ Loglarni olib bo'lmadi.")
        return
    if not logs:
        try:
            await status_msg.edit_text(f"Deployment status: {deployment.get('status')}. Loglar hozircha bo'sh.")
        except TelegramBadRequest:
            pass
        return

    log_text = "\n".join(
        f"[{entry.get('severity', '')}] {entry.get('message', '')}" for entry in logs
    )[:12000]

    agent = get_agent("devops")
    system_prompt = await _build_system_prompt(
        chat_id,
        agent,
        "VAZIFA: quyidagi production server loglarini diqqat bilan tahlil "
        "qiling. Xatolar (error/exception/crash/traceback) bormi? Bo'lsa, "
        "har birining ehtimoliy sababini va aniq tuzatish yo'lini ko'rsating. "
        "Agar muhim xato yo'q bo'lsa, shuni qisqa tasdiqlang.",
    )
    model, _label = model_for(agent, "high")
    try:
        analysis = await asyncio.wait_for(
            groq_generate(
                model,
                system_prompt,
                [
                    {
                        "role": "user",
                        "content": f"Deployment status: {deployment.get('status')}\n\nLoglar:\n{log_text}",
                    }
                ],
                temperature=0.3,
            ),
            timeout=settings.request_timeout,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Log analysis failed")
        await status_msg.edit_text(f"⚠️ Loglarni tahlil qilishda xatolik: {exc}")
        return

    try:
        await status_msg.delete()
    except TelegramBadRequest:
        pass

    header = "" if not settings.show_metadata_header else "🛠️ DevOps Engineer — Server Log Analysis\n\n"
    await _send_long(message, f"{header}Status: {deployment.get('status')}\n\n{analysis}")


@dp.message(Command("readfile"))
async def cmd_readfile(message: Message, bot: Bot, command: CommandObject) -> None:
    """The 'pull' half of push/pull: brings a file's CURRENT content from the
    configured GitHub repo into the conversation, so a follow-up refinement/
    bug-fix request is grounded in what's actually there."""
    chat_id = message.chat.id
    if not _is_allowed(chat_id):
        return
    path = (command.args or "").strip()
    if not path:
        await message.answer("Fayl yo'lini ko'rsating, masalan:\n/readfile bot.py")
        return
    if not settings.github_enabled:
        await message.answer("GitHub integratsiyasi sozlanmagan (GITHUB_TOKEN / GITHUB_REPO).")
        return

    status_msg = await message.answer(f"📂 {path} o'qiyapman...")
    content = await github_integration.get_file_content(path)
    if content is None:
        await status_msg.edit_text(f"⚠️ Fayl topilmadi yoki o'qib bo'lmadi: {path}")
        return
    try:
        await status_msg.delete()
    except TelegramBadRequest:
        pass

    combined = (
        f"GitHub repodagi `{path}` faylining HOZIRGI holati (quyidagi kodga "
        "asoslanib javob bering; agar refactor/fix so'ralsa, shu faylning "
        "to'liq yangilangan versiyasini bering, file-marker konvensiyasi "
        "bilan):\n\n```\n" + content[:12000] + "\n```"
    )
    await _process(message, bot, combined, forced_type=None)


# --------------------------------------------------------------------------
# Free-text handler (classifier decides the request type)
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
            logger.info("Persistent storage ENABLED via Redis — history and memory survive restarts.")
        else:
            logger.warning(
                "REDIS_URL is set but the client failed to initialise — "
                "falling back to in-memory storage (lost on restart). Check the URL/credentials."
            )
    else:
        logger.info(
            "Persistent storage DISABLED (no REDIS_URL) — history and memory are in-memory "
            "only and will be lost on every restart/redeploy. Add Railway's Redis database "
            "and set REDIS_URL to persist across deploys."
        )

    if settings.glm_enabled:
        global _glm_runtime_ok
        ok, msg = await glm_health_check()
        _glm_runtime_ok = ok
        if ok:
            logger.info("Route C (GLM) ENABLED: model=%s base_url=%s", settings.glm_model, settings.glm_base_url)
        else:
            logger.warning("Route C (GLM) DISABLED at startup: %s", msg)
    else:
        logger.info("Route C (GLM) disabled (GLM_API_KEY not set) — Route B (Groq) used for all code tasks.")

    if settings.github_enabled:
        logger.info(
            "GitHub integration ENABLED for repo=%s (auto_pr=%s)",
            settings.github_repo,
            settings.github_auto_pr,
        )
    else:
        logger.info("GitHub integration disabled (set GITHUB_TOKEN and GITHUB_REPO to enable).")

    if not settings.bot_token:
        raise SystemExit("BOT_TOKEN is required. See .env.example.")

    bot = Bot(
        token=settings.bot_token,
        default=DefaultBotProperties(parse_mode=None),
    )
    logger.info("Starting polling…")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
