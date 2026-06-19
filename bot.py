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

/kickoff <text> fans the SAME request out to a fixed cross-functional team
(PM, Product Designer, Backend, QA) in parallel and returns all four answers
in one message — for when the user wants the whole team to respond together,
not a single department. This exists because every agent is otherwise scoped
to its own role: without it, asking one agent to "prepare this with your
team" had nowhere to go except telling the user to go find people — even
though the rest of the team already lives in this same bot.

Run:  python bot.py
"""

import asyncio
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandObject
from aiogram.types import Message

import github_integration
import history
from agents import Agent, get_agent
from config import settings
from llm_clients import gemini_generate, groq_generate
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


def _header(route: Route) -> str:
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
    if route.agent.route == "A":
        return await gemini_generate(route.model, system_prompt, messages)
    return await groq_generate(route.model, system_prompt, messages, temperature=0.3)


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


# --------------------------------------------------------------------------
# Core pipeline (shared by free-text messages and /idea /task /bug /improve)
# --------------------------------------------------------------------------
async def _process(message: Message, bot: Bot, user_text: str | None, forced_type: str | None) -> None:
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

    last = history.get_last_route(chat_id)
    route = await classify(
        user_text,
        last_agent=last[0] if last else None,
        last_type=last[1] if last else None,
    )
    if forced_type:
        route.request_type = get_request_type(forced_type)

    logger.info(
        "chat=%s -> agent=%s route=%s model=%s type=%s",
        chat_id,
        route.agent.key,
        route.agent.route,
        route.model,
        route.request_type.key,
    )

    system_prompt = route.agent.system
    if route.request_type.addendum:
        system_prompt = system_prompt + "\n" + route.request_type.addendum

    msgs = history.get_history(chat_id, route.agent.key) + [
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

    history.append(chat_id, route.agent.key, "user", user_text)
    history.append(chat_id, route.agent.key, "assistant", body)
    history.set_last_route(chat_id, route.agent.key, route.request_type.key)

    footer = await _maybe_create_tickets(route, user_text, body)
    await _send_long(message, _header(route) + body + footer)


# --------------------------------------------------------------------------
# /kickoff — whole team responds together
# --------------------------------------------------------------------------
async def _kickoff_agent_call(agent_key: str, user_text: str) -> dict:
    """Run one team member's reply for /kickoff. Never raises — returns an
    error note in 'body' instead, so one failing teammate doesn't blank out
    the others."""
    agent: Agent = get_agent(agent_key)
    # Force an actual deliverable (not discussion) from every teammate, same
    # as the TASK request type does for normal single-agent routing.
    system_prompt = agent.system + "\n" + REQUEST_TYPES["task"].addendum
    model, label = model_for(agent, "high")
    try:
        if agent.route == "A":
            body = await asyncio.wait_for(
                gemini_generate(model, system_prompt, [{"role": "user", "content": user_text}]),
                timeout=settings.request_timeout,
            )
        else:
            body = await asyncio.wait_for(
                groq_generate(
                    model,
                    system_prompt,
                    [{"role": "user", "content": user_text}],
                    temperature=0.3,
                ),
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


def _kickoff_header(results: list[dict]) -> str:
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
        *[_kickoff_agent_call(key, user_text) for key in KICKOFF_ROLES]
    )

    parts = [_kickoff_header(results)]
    for r in results:
        parts.append(f"#### 🛠️ {r['agent'].display_name}\n\n{r['body']}\n\n---\n")
        history.append(chat_id, r["agent"].key, "user", user_text)
        history.append(chat_id, r["agent"].key, "assistant", r["body"])

    # Anchor follow-ups (e.g. "ok continue") back to PM by default after a kickoff.
    history.set_last_route(chat_id, "pm", "idea")

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
        "DevOps, SOC, Tech Lead) и оптимальную модель, а также определю тип "
        "запроса.\n\n"
        "Тип запроса можно задать явно:\n"
        "/idea <текст> — идея / предложение\n"
        "/task <текст> — задача на реализацию\n"
        "/bug <текст> — баг, нужен фикс\n"
        "/improve <текст> — доработка / рефакторинг\n"
        "/kickoff <текст> — вся команда сразу (PM + Designer + Backend + QA)\n"
        "Без команды — отвечу как на обычный вопрос, без тикета.\n\n"
        "/reset — очистить контекст диалога\n"
        f"chat_id: `{message.chat.id}`",
        parse_mode="Markdown",
    )


@dp.message(Command("reset"))
async def cmd_reset(message: Message) -> None:
    if not _is_allowed(message.chat.id):
        return
    history.reset(message.chat.id)
    await message.answer("Контекст очищен. ✅")


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
