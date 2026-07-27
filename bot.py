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
  /agents — list all available specialists
  /idea /task /bug /improve — force request type
  /kickoff — whole team responds in parallel (PM, BA, Data Analyst, Process Analyst)
  /proposal — deliver as Word + PDF document
  /addtask — add a daily task/reminder (also auto-detected from plain messages)
  /tasks — list active tasks; /done /canceltask — mark complete/cancelled
  /digest — opt-in daily morning plan; /standup /week — status drafts
  /minutes — meeting notes → structured minutes + action items → tasks
  /decision /decisions — dated project decision log
  /remember /memory /forget — project memory management
  /logs — Railway deployment log analysis (if configured)
  /readfile — pull current file from GitHub into conversation
  /reset — clear chat history
  /status — show configuration health
  /users — admin: see everyone who's messaged the bot, approve/deny them

Business copilot (automatic, no command): if the admin connects this bot to
their own personal chats via Telegram's Settings > Business > Chatbots,
incoming messages there get AI-analyzed with a suggested reply sent to the
admin's own DM with the bot — see business_copilot.py.

Group mention copilot (automatic, no command): add the bot as a plain
member to any group (no admin rights needed); if someone @mentions the
admin or replies to the admin's own message there, it's AI-analyzed with a
suggested reply sent privately to the admin's DM — see group_copilot.py
(requires Telegram Privacy Mode disabled for the bot, see .env.example).

Run:  python bot.py
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime

from aiogram import BaseMiddleware, Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandObject
from aiogram.types import (
    BotCommand,
    BotCommandScopeChat,
    BufferedInputFile,
    BusinessConnection,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    ReplyParameters,
)

import access_control
import business_copilot
import decisions as decisions_store
import digest
import monthly_report
import expenses
import document_generation as docgen
import github_integration
import db
import group_copilot
import history
import memory
import quick_actions
import shared_context
import user_profile
import minutes as minutes_mod
import railway_integration
import redis_client
import task_assistant
import tasks
import web_search
from agents import (
    Agent,
    TEAM_MEMORY_HEADER,
    agent_key_by_name,
    agent_label,
    get_agent,
    persona,
)
from config import settings
from file_processing import SUPPORTED_SUMMARY, extract, transcribe_audio
from llm_clients import claude_generate, claude_generate_fast, claude_generate_json, parse_llm_json
from request_types import REQUEST_TYPES, get_request_type
from router import Route, classify, model_for

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("orchestrator")

dp = Dispatcher()

TELEGRAM_LIMIT = 4096

# --------------------------------------------------------------------------
# Post-approval onboarding — collect phone number before an approved user's
# messages reach the normal AI pipeline. F.I.O is NOT asked for — Telegram
# already gives every message a first_name/last_name (captured into
# full_name by record_activity() on first contact), so that's used directly
# as F.I.O. The phone number has no such shortcut: Telegram's Bot API never
# exposes a user's phone number to a bot unless the user explicitly shares
# it via a contact-share action, so that's the one thing still asked for.
# --------------------------------------------------------------------------
def _phone_request_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="📱 Raqamni yuborish", request_contact=True)]],
        resize_keyboard=True, one_time_keyboard=True,
    )


async def _start_onboarding(bot: Bot, target_user_id: int) -> None:
    """Call right after approve() — auto-fills F.I.O from the Telegram name
    already on file (no need to ask) and asks only for the phone number.
    Shared by /approve and the ✅ button so both paths behave identically."""
    full_name = await access_control.get_known_full_name(target_user_id)
    if full_name:
        await access_control.save_profile_field(target_user_id, "fio", full_name)
    await access_control.set_onboarding_state(target_user_id, "awaiting_phone")
    try:
        await bot.send_message(
            target_user_id,
            "✅ Sizga botdan foydalanish uchun ruxsat berildi!\n\n"
            "Iltimos, telefon raqamingizni yuboring — pastdagi tugmani bosing "
            "(yoki /skip yozing).",
            reply_markup=_phone_request_keyboard(),
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to send onboarding prompt to user=%s", target_user_id)


async def _notify_admin_of_phone(bot: Bot, user_id: int, fio: str | None, phone: str) -> None:
    admin_chat_id = await access_control.get_admin_chat_id()
    if not admin_chat_id:
        return
    try:
        # No parse_mode on purpose: fio is user-controlled free text and any
        # markdown metacharacters in it would make Telegram reject the send.
        await bot.send_message(
            admin_chat_id,
            f"📇 {fio or 'Noma’lum'} (ID: {user_id}) telefon raqamini yubordi: {phone}",
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to notify admin of new phone for user=%s", user_id)


async def _handle_onboarding_step(message: Message, uid: int) -> bool:
    """If this approved user still has a pending onboarding step, consume
    this message as that step's answer and return True (caller must NOT
    fall through to the normal AI pipeline for this message). Returns False
    if there's no onboarding pending (already done, or never started —
    users approved before this feature existed)."""
    # A contact share is unambiguous — accept it ANY time it arrives, not
    # only while state is exactly "awaiting_phone". If the internal state
    # ever desyncs (e.g. in-memory onboarding progress wiped by a redeploy
    # when REDIS_URL isn't configured, or a stale reply-keyboard button
    # tapped after the flow already moved on), a shared phone number must
    # never be silently dropped — that was the concrete bug reported: the
    # user shared their contact, the bot showed no reaction at all, and
    # /users never got the number, because content_type=contact matches no
    # other handler in this file and was falling through into the void.
    contact = message.contact
    # Contact.user_id is OPTIONAL per the Bot API spec and is not always
    # populated even for a legitimate self-share via the request_contact
    # button (client/version-dependent) — trust it whenever it's absent,
    # only reject an EXPLICIT mismatch (someone manually attaching a
    # different contact card via the paperclip menu instead of tapping
    # the button).
    if contact and contact.user_id in (None, 0, uid):
        fio = await access_control.get_known_full_name(uid)
        await access_control.save_profile_field(uid, "phone", contact.phone_number)
        await access_control.set_onboarding_state(uid, "done")
        await message.answer(
            "✅ Ma'lumotlaringiz saqlandi.",
            reply_markup=ReplyKeyboardRemove(),
        )
        await _notify_admin_of_phone(message.bot, uid, fio, contact.phone_number)
        await _send_welcome_tour(message)
        return True

    state = await access_control.get_onboarding_state(uid)
    if state not in ("awaiting_fio", "awaiting_phone"):
        return False

    if (message.text or "").strip().lower().startswith("/skip"):
        await access_control.set_onboarding_state(uid, "done")
        await message.answer(
            "Yaxshi, o'tkazib yubordik.",
            reply_markup=ReplyKeyboardRemove(),
        )
        await _send_welcome_tour(message)
        return True

    # "awaiting_fio" only exists as a stale state for anyone caught mid-flow
    # by this change (F.I.O is no longer asked for) — auto-fill from
    # Telegram's name and fall straight into asking for the phone, ignoring
    # whatever text they just sent rather than leaving them stuck.
    if state == "awaiting_fio":
        full_name = await access_control.get_known_full_name(uid)
        if full_name:
            await access_control.save_profile_field(uid, "fio", full_name)
        await access_control.set_onboarding_state(uid, "awaiting_phone")
        await message.answer(
            "Iltimos, telefon raqamingizni yuboring — pastdagi tugmani bosing "
            "(yoki /skip yozing).",
            reply_markup=_phone_request_keyboard(),
        )
        return True

    if state == "awaiting_phone":
        await message.answer(
            "Iltimos, pastdagi \"📱 Raqamni yuborish\" tugmasini bosing (yoki /skip yozing)."
        )
        return True

    return False


# --------------------------------------------------------------------------
# Activation: welcome tour + one-tap daily digest
#
# The single biggest reason a newly-approved user stops opening the bot is
# that onboarding used to end with a bare "endi savolingizni yozing" — no
# concrete picture of what daily value looks like, no recurring touchpoint.
# This card fixes both: four copy-able real use cases, plus ONE TAP to turn
# on the morning digest (the retention hook — a reason the bot messages
# THEM every day, not the other way around).
# --------------------------------------------------------------------------
WELCOME_TOUR_TEXT = (
    "🚀 Har kuni foyda beradigan imkoniyatlar:\n\n"
    "1️⃣ Savol-javob — shunchaki yozing:\n"
    "\"Mijozlar oqimi kamaydi, sabablarini qanday tahlil qilay?\"\n\n"
    "2️⃣ Vazifa/eslatma — oddiy tilda:\n"
    "\"ertaga 15:00 da hisobotni topshirishim kerak\" → o'zim eslataman\n"
    "\"ijara puli 5-avgustgacha to'lash kerak\" → to'lov muddatini ham "
    "kuzataman, har oy takrorlansa \"har oy\" deb yozing\n\n"
    "3️⃣ Tayyor hujjat (Word + PDF):\n"
    "/proposal CRM joriy etish bo'yicha taklif\n\n"
    "4️⃣ Uchrashuv protokoli — yig'ilish yozuvini /minutes bilan yuboring → "
    "qarorlar va vazifalar avtomatik ajratiladi\n\n"
    "5️⃣ Xarajat hisobi — \"taksiga 30 ming\" deb yozing → yozib boraman, "
    "/xarajatlar bilan oylik hisobot. /byudjet 3 mln bilan chegara qo'ying — "
    "80% va 100% da o'zim ogohlantiraman\n\n"
    "☀️ Har kuni ertalab kunlik reja (vazifalaringiz, muddatlar, eslatmalar) "
    "olib turishni xohlaysizmi?"
)


def _welcome_tour_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="☀️ Ha, har kuni kelsin", callback_data="tour:d"),
        InlineKeyboardButton(text="Hozircha kerak emas", callback_data="tour:x"),
    ]])


async def _send_welcome_tour(message: Message) -> None:
    try:
        await message.answer(WELCOME_TOUR_TEXT, reply_markup=_welcome_tour_keyboard())
    except Exception:  # noqa: BLE001 — the tour is a bonus, never break onboarding over it
        logger.exception("Failed to send welcome tour")


# --------------------------------------------------------------------------
# Admin-approval gate (private chats only) — runs before every other handler
# --------------------------------------------------------------------------
async def _forward_request_to_admin(
    bot: Bot, admin_chat_id: int, user_chat_id: int, message_id: int, info_text: str
) -> None:
    """Forward one access request to the admin (message content + an info
    card with Approve/Reject buttons), linking both as valid relay targets
    for the admin's native Telegram "Reply". Shared by the live path
    (_handle_unapproved) and the queued-backlog flush (_deliver_pending_list)."""
    fwd = None
    try:
        fwd = await bot.forward_message(admin_chat_id, user_chat_id, message_id)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to forward access request (user chat=%s) to admin", user_chat_id)

    try:
        info = await bot.send_message(
            admin_chat_id, info_text, reply_markup=access_control.access_keyboard(user_chat_id)
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to send access request card (user chat=%s) to admin", user_chat_id)
        return

    if fwd is not None:
        await access_control.link_relay(fwd.message_id, user_chat_id)
    await access_control.link_relay(info.message_id, user_chat_id)


async def _handle_unapproved(message: Message) -> None:
    """An unapproved, non-admin private-chat user messaged the bot: tell them
    to contact the admin, and relay their message (with Approve/Reject
    buttons) to the admin — or, if the admin's chat isn't known YET (admin
    hasn't bootstrapped by messaging the bot even once), queue it so it's
    delivered the moment the admin is next recognized instead of being lost."""
    bot = message.bot
    handle = (settings.admin_username or "admin").lstrip("@")
    uid_for_seen = message.from_user.id if message.from_user else message.chat.id

    # Full bilingual explanation only on the first-ever message from this
    # user; after that, a short acknowledgment — so a real back-and-forth
    # with the admin (asking questions while waiting for approval) doesn't
    # repeat a wall of text every time. They can still freely message the
    # admin either way; only the OTHER bot features stay gated until approved.
    if await access_control.mark_first_contact(uid_for_seen):
        await message.answer(
            f"🔒 Botdan foydalanish uchun admin (@{handle}) ruxsati kerak. "
            f"Ruxsat berilishini kuting. Savol yoki murojaatingiz bo'lsa, shu yerga "
            f"yozavering — xabaringiz to'g'ridan-to'g'ri adminga yetkaziladi.\n\n"
            f"🔒 Для использования бота нужно разрешение администратора (@{handle}). "
            f"Дождитесь одобрения. Если есть вопрос или просьба к администратору — "
            f"просто напишите здесь, сообщение будет передано напрямую."
        )
    else:
        await message.answer(
            f"✅ Xabaringiz @{handle} ga yuborildi. / Ваше сообщение отправлено @{handle}."
        )

    admin_chat_id = await access_control.get_admin_chat_id()
    if not admin_chat_id or bot is None:
        logger.info(
            "Unapproved user=%s messaged before admin was recognized — queued for delivery.",
            message.from_user.id if message.from_user else "?",
        )
        await access_control.queue_pending(message.chat.id, message.message_id)
        return

    user = message.from_user
    name = user.full_name if user else "Noma'lum"
    uname = f"@{user.username}" if user and user.username else "(username yo'q)"
    uid = user.id if user else 0
    info_text = (
        f"👤 {name} {uname} (ID: {uid}) botdan foydalanishga ruxsat so'rayapti.\n\n"
        "Ruxsat berish uchun tugmani bosing, yoki shu xabarga (yoki yuqoridagi "
        "forward qilingan xabarga) Reply qilib to'g'ridan-to'g'ri javob yozing."
    )
    await _forward_request_to_admin(bot, admin_chat_id, message.chat.id, message.message_id, info_text)


async def _notify_denied(bot: Bot, target_id: int) -> None:
    """Tell a user their access was denied/revoked — shared by the ❌ button
    and /deny so BOTH paths inform the user (the /deny path used to stay
    silent, leaving the user to keep messaging a bot that ignored them)."""
    handle = (settings.admin_username or "admin").lstrip("@")
    try:
        await bot.send_message(
            target_id,
            f"❌ Afsuski, sizga botdan foydalanish ruxsati berilmadi. "
            f"Savollaringiz bo'lsa admin (@{handle}) bilan bog'laning.\n\n"
            f"❌ К сожалению, вам не предоставлен доступ к боту. "
            f"По вопросам обращайтесь к администратору (@{handle}).",
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to notify denied user=%s", target_id)


async def _handle_denied(message: Message) -> None:
    """A DENIED user messaged again. Don't pretend their message was passed
    on (the old flow answered 'xabaringiz adminга yuborildi' — misleading)
    and don't spam the admin with requests they already rejected: state the
    denial plainly with a direct contact route instead."""
    handle = (settings.admin_username or "admin").lstrip("@")
    await message.answer(
        f"❌ Sizga botdan foydalanish uchun ruxsat berilmagan. "
        f"Savollaringiz bo'lsa admin (@{handle}) bilan to'g'ridan-to'g'ri bog'laning.\n\n"
        f"❌ Вам не предоставлен доступ к боту. По вопросам обращайтесь "
        f"к администратору (@{handle}) напрямую."
    )


async def _deliver_pending_list(bot: Bot, admin_chat_id: int, pending: list[tuple[int, int]]) -> None:
    """Flush access requests that were queued while the admin's chat_id
    wasn't known yet. `pending` must already be popped (see
    access_control.pop_all_pending) — this function only delivers."""
    if not pending:
        return
    try:
        await bot.send_message(admin_chat_id, f"📥 Kutib turgan {len(pending)} ta ruxsat so'rovi bor edi:")
    except Exception:  # noqa: BLE001
        logger.exception("Failed to send pending-backlog notice to admin")
    for user_chat_id, message_id in pending:
        info_text = f"👤 ID: {user_chat_id} — botdan foydalanishga ruxsat so'ragan (kutib turgan so'rov)."
        await _forward_request_to_admin(bot, admin_chat_id, user_chat_id, message_id, info_text)


async def _relay_admin_reply(admin_message: Message, target_chat_id: int) -> None:
    """Admin replied (native Telegram "Reply") to a relayed user message —
    forward that reply back to the original user."""
    bot = admin_message.bot
    try:
        if admin_message.text:
            await bot.send_message(target_chat_id, f"💬 Admin javobi:\n\n{admin_message.text}")
        else:
            await bot.send_message(target_chat_id, "💬 Admin sizga javob yubordi:")
            await bot.copy_message(target_chat_id, admin_message.chat.id, admin_message.message_id)
        await admin_message.reply("✅ Yuborildi.")
    except Exception:  # noqa: BLE001
        logger.exception("Failed to relay admin reply to chat=%s", target_chat_id)
        await admin_message.reply("⚠️ Yuborib bo'lmadi (foydalanuvchi botni bloklagan bo'lishi mumkin).")


async def _relay_business_reply(admin_message: Message, relay_msg_id: int, relay: dict) -> None:
    """Admin replied to a business-copilot notification with their OWN text
    (instead of tapping the suggested-reply button) — send that text into
    the connected Business chat as themselves."""
    bot = admin_message.bot
    text = admin_message.text or admin_message.caption
    if not text:
        await admin_message.reply("⚠️ Faqat matn yuborish mumkin.")
        return
    try:
        await bot.send_message(relay["chat"], text, business_connection_id=relay["conn"])
        await business_copilot.append_own_message(relay["conn"], relay["chat"], text)
        await business_copilot.clear_relay(relay_msg_id)
        await admin_message.reply("✅ Yuborildi.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to relay admin's custom business-chat reply")
        await admin_message.reply(f"⚠️ Yuborib bo'lmadi: {str(exc)[:200]}")


async def _relay_group_reply(admin_message: Message, relay_msg_id: int, relay: dict) -> None:
    """Admin replied to a group-mention notification with their OWN text
    (instead of tapping the suggested-reply button) — send that text into
    the group, as a reply to the original message that addressed them."""
    bot = admin_message.bot
    text = admin_message.text or admin_message.caption
    if not text:
        await admin_message.reply("⚠️ Faqat matn yuborish mumkin.")
        return
    try:
        await bot.send_message(
            relay["chat"], text,
            reply_parameters=ReplyParameters(message_id=relay["reply_to"]),
        )
    except TelegramBadRequest:
        # Original message may have been deleted / too old to reply to —
        # still deliver the admin's answer into the group, just not threaded.
        try:
            await bot.send_message(relay["chat"], text)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to relay admin's custom group reply (fallback also failed)")
            await admin_message.reply(f"⚠️ Yuborib bo'lmadi: {str(exc)[:200]}")
            return
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to relay admin's custom group reply")
        await admin_message.reply(f"⚠️ Yuborib bo'lmadi: {str(exc)[:200]}")
        return
    await group_copilot.clear_relay(relay_msg_id)
    if relay.get("sender_id"):
        await shared_context.append(relay["sender_id"], "admin", text, "group")
    await admin_message.reply("✅ Guruhga yuborildi.")


class AccessGateMiddleware(BaseMiddleware):
    """Gates PRIVATE chats behind admin approval. Group access is untouched
    — it stays governed by ALLOWED_CHAT_IDS / mention-required logic below."""

    async def __call__(self, handler, event: Message, data):
        if event.chat.type != "private" or not event.from_user:
            return await handler(event, data)

        # /id must ALWAYS work, for EVERYONE, regardless of approval state.
        # It's the bootstrap utility: a pending user needs it to hand their
        # numeric ID to the admin for /approve, and the admin needs it to
        # discover their OWN numeric ID for ADMIN_USER_ID — which is a
        # chicken-and-egg case if their Telegram account has no public
        # @username, or one that doesn't match ADMIN_USERNAME, since is_admin()
        # can't recognize them yet either. Without this exemption /id was
        # silently swallowed by the unapproved-user flow (or the onboarding
        # nag) instead of ever reaching cmd_id — this was the reported bug.
        # Reveals nothing sensitive: Telegram already exposes chat_id to the
        # client itself.
        cmd = (event.text or "").split()[0].split("@")[0].lower() if event.text else ""
        if cmd == "/id":
            return await handler(event, data)

        uid = event.from_user.id
        uname = event.from_user.username

        if access_control.is_admin(uid, uname):
            previously_known = bool(await access_control.get_admin_chat_id())
            await access_control.remember_admin_chat(event.chat.id)

            if event.reply_to_message:
                target = await access_control.resolve_relay(event.reply_to_message.message_id)
                if target is not None:
                    await _relay_admin_reply(event, target)
                    return
                biz_relay = await business_copilot.resolve_relay(event.reply_to_message.message_id)
                if biz_relay is not None:
                    await _relay_business_reply(event, event.reply_to_message.message_id, biz_relay)
                    return
                grp_relay = await group_copilot.resolve_relay(event.reply_to_message.message_id)
                if grp_relay is not None:
                    await _relay_group_reply(event, event.reply_to_message.message_id, grp_relay)
                    return

            bot = event.bot
            if bot is not None:
                pending = await access_control.pop_all_pending()
                if pending:
                    await _deliver_pending_list(bot, event.chat.id, pending)
                elif not previously_known:
                    try:
                        await bot.send_message(
                            event.chat.id,
                            "✅ Siz admin sifatida aniqlandingiz. Endi foydalanuvchilarning "
                            "ruxsat so'rovlari shu yerga keladi.",
                        )
                    except Exception:  # noqa: BLE001
                        logger.exception("Failed to send admin-bootstrap confirmation")
            return await handler(event, data)

        # Track activity for every regular (non-admin) user, approved or not,
        # so /users shows real usage — not just pending-request traffic.
        await access_control.record_activity(uid, uname, event.from_user.full_name)

        if await access_control.is_approved(uid):
            if await _handle_onboarding_step(event, uid):
                return
            return await handler(event, data)

        if await access_control.is_denied(uid):
            await _handle_denied(event)
            return

        await _handle_unapproved(event)
        return


# MUST be OUTER middleware. An inner middleware (dp.message.middleware)
# only runs when some handler's filters match the message — and content
# types with no registered handler (contact shares, stickers, locations…)
# would bypass the gate entirely: an onboarding contact share got zero
# reaction (the reported bug), and unapproved users' unmatched messages
# were never relayed to the admin. Outer middleware runs for EVERY
# private message before any filter, closing both holes.
dp.message.outer_middleware(AccessGateMiddleware())

# Hard cap on combined user input (typed text + quoted file/reply context)
# before it reaches the router/agents — protects token budget and free-tier
# quota from unbounded pasted content.
MAX_USER_TEXT = 20000

# Default cross-functional team for /kickoff: core BA team
KICKOFF_ROLES = ["ba", "pm", "data_analyst", "process_analyst"]

# Proactive group mode: tracks last proactive reply time per chat to avoid spam
_proactive_last: dict[int, float] = {}

# Cost protection: one in-flight LLM pipeline per chat. A second message while
# the first is still generating gets a polite "wait" instead of spawning more
# parallel chains/collaborations (each one is multiple LLM calls).
_in_flight: set[int] = set()

# Grouped agent catalogue — used by /agents (discoverability) and /status.
# Keys must match agents.AGENTS; anything missing there is skipped gracefully.
AGENT_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("BA & Analysis", [
        ("ba", "biznes talablar, user story, GAP tahlil"),
        ("data_analyst", "SQL, KPI, ma'lumotlar tahlili"),
        ("bi_analyst", "Power BI/Tableau dashboard va hisobotlar"),
        ("process_analyst", "BPMN, AS-IS/TO-BE jarayon xaritalari"),
        ("financial_analyst", "P&L, moliyaviy model, ROI/NPV"),
        ("market_analyst", "bozor tahlili, SWOT, raqobat"),
        ("requirements_engineer", "BRD, FRS, NFR hujjatlar"),
        ("data_governance", "data quality, MDM, GDPR"),
    ]),
    ("Product & Delivery", [
        ("pm", "roadmap, backlog, prioritizatsiya, OKR"),
        ("system_analyst", "arxitektura, integratsiya, API kontraktlar"),
        ("qa", "test strategiya, test case, UAT"),
        ("project_manager", "WBS, Gantt, RAID, sprint rejalashtirish"),
        ("tech_consultant", "vendor tanlash, build-vs-buy, texnologiya maslahat"),
    ]),
    ("Engineering", [
        ("backend", "server kod, API, ma'lumotlar bazasi"),
        ("frontend", "React/Vue komponentlar, UI kod"),
        ("devops", "Docker, Kubernetes, CI/CD, cloud"),
        ("product_designer", "UX/UI, wireframe, dizayn tizimi"),
    ]),
    ("Maxsus mutaxassislar", [
        ("translator", "professional tarjima EN ↔ RU ↔ UZ"),
        ("diagram", "Mermaid BPMN/UML/ER diagrammalar"),
        ("technical_analyst", "API/Swagger, JSON/XML, log tahlili"),
        ("jira", "Jira Epic/Story/Task/Bug formatlash"),
    ]),
    ("Bank sohasi", [
        ("credit_conveyor", "kredit konveyer, KYC/AML, qaror mexanizmi"),
        ("core_banking", "hisoblar, tranzaksiyalar, GL/posting"),
        ("integration", "REST/SOAP/Kafka/ESB integratsiya"),
        ("scoring", "kredit skoring, PD/LGD/EAD, fraud"),
        ("insurance", "kredit sug'urtasi, polis, premiya hisobi"),
    ]),
]

_RELEVANCE_SYSTEM = """
You are a relevance filter for a Senior Business Analyst AI team in a Telegram group.
Decide whether the message below needs a response from the team.

Reply with ONLY this JSON — no prose, no markdown:
{"relevant": true}  or  {"relevant": false}

Respond TRUE if the message:
- Asks a professional question (business analysis, requirements, data, SQL, KPIs,
  dashboards, financial modeling, market research, architecture, APIs, DevOps, QA)
- Describes a problem or challenge needing help
- Requests analysis, a document, a plan, or expert advice
- Discusses specifications, integrations, process design, or delivery

Respond FALSE if:
- Casual chat, greetings, emojis only, jokes
- Short ack: "ok", "ha", "rahmat", "спасибо", "got it", "+1", "👍"
- Off-topic personal or social conversation
- The question is clearly already fully answered in the same message
"""


async def _check_group_relevance(text: str) -> bool:
    """Return True if this group message is relevant enough to respond to proactively."""
    if len(text.strip()) < 15:
        return False
    try:
        raw = await asyncio.wait_for(
            claude_generate_json(
                _RELEVANCE_SYSTEM,
                [{"role": "user", "content": text[:800]}],
            ),
            timeout=10,
        )
        data = parse_llm_json(raw)
        return bool(data.get("relevant", False))
    except Exception:
        return False


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _is_allowed(chat_id: int) -> bool:
    return not settings.allowed_chat_ids or chat_id in settings.allowed_chat_ids


async def _callback_still_authorized(callback: CallbackQuery) -> bool:
    """Defense in depth for private-chat callback buttons (tsk:/min:): the
    AccessGateMiddleware only runs on inbound messages, so a user who was
    approved when a button was created but has since been /deny-revoked
    would otherwise keep working the button forever — buttons don't get
    retroactively disabled. Re-check per-user approval at click time. Group
    callbacks stay governed by _is_allowed only, matching the message-level
    gate's scope (access_control doesn't apply to groups)."""
    if not callback.message or callback.message.chat.type != "private":
        return True
    uid = callback.from_user.id if callback.from_user else 0
    uname = callback.from_user.username if callback.from_user else None
    if access_control.is_admin(uid, uname):
        return True
    return await access_control.is_approved(uid)


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


def _mentions_admin(message: Message) -> bool:
    """True if this GROUP message is addressed to the ADMIN specifically —
    they're @mentioned (or text-mentioned, which works even without a
    public username), or the message replies to something the admin
    themselves sent earlier in this group. Powers group_copilot.py; see
    its docstring for the Privacy Mode prerequisite."""
    replied = message.reply_to_message
    if replied and replied.from_user and access_control.is_admin(
        replied.from_user.id, replied.from_user.username
    ):
        return True

    text = message.text or message.caption or ""
    entities = message.entities or message.caption_entities or []
    for ent in entities:
        if ent.type == "mention":
            handle = text[ent.offset: ent.offset + ent.length].lstrip("@")
            if access_control.is_admin(0, handle):
                return True
        elif ent.type == "text_mention" and ent.user:
            if access_control.is_admin(ent.user.id, ent.user.username):
                return True
    return False


async def _handle_group_mention(message: Message, bot: Bot, user_text: str) -> None:
    """A group message addressed the admin specifically — AI-analyze it
    and privately notify the admin with a suggested reply (see
    group_copilot.py). Never posts anything in the group on its own."""
    admin_chat_id = await access_control.get_admin_chat_id()
    if not admin_chat_id:
        return  # admin not bootstrapped yet (hasn't messaged the bot even once) — nothing to notify

    sender = message.from_user
    sender_name = sender.full_name if sender else "Noma'lum"
    group_name = message.chat.title or "Guruh"

    quoted_text = ""
    replied = message.reply_to_message
    if replied:
        quoted_text = (replied.text or replied.caption or "").strip()

    sender_id = sender.id if sender else None
    data, error = await group_copilot.analyze(group_name, sender_name, quoted_text, user_text, sender_id=sender_id)
    if data is None:
        try:
            await bot.send_message(
                admin_chat_id,
                f"👥 \"{group_name}\" guruhida {sender_name} sizga yozdi:\n\n{user_text}\n\n"
                f"⚠️ AI tahlili muvaffaqiyatsiz — o'zingiz javob yozing.\n"
                f"Sabab: {error or 'nomaʼlum xatolik'}\n"
                "(Agar bu tez-tez takrorlansa, /status bilan provider holatini tekshiring.)",
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to notify admin of group mention (analysis failed)")
        return

    analysis = data.get("analysis", "")
    suggested = data.get("suggested_reply", "")
    is_task = bool(data.get("is_task", False))
    action_line = (
        "🧠 Bu jiddiy vazifaga o'xshaydi — quyidagi taklif faqat vaqtinchalik "
        "javob, haqiqiy javobni o'zingiz yozing."
        if is_task else
        "Yuborish uchun tugmani bosing, yoki shu xabarga Reply qilib o'zingiz "
        "yozgan javobni yuboring."
    )
    card = (
        f"👥 \"{group_name}\" guruhida {sender_name} sizga yozdi:\n\n\"{user_text}\"\n\n"
        f"🧠 Tahlil: {analysis}\n\n"
        f"💬 Taklif qilingan javob:\n{suggested}\n\n{action_line}"
    )
    try:
        sent = await bot.send_message(admin_chat_id, card)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to send group-mention notification to admin")
        return

    await group_copilot.link_relay(
        sent.message_id, message.chat.id, message.message_id, suggested, user_text, sender_id=sender_id,
    )
    try:
        await sent.edit_reply_markup(reply_markup=group_copilot.suggestion_keyboard(sent.message_id))
    except Exception:  # noqa: BLE001
        logger.exception("Failed to attach group-copilot suggestion buttons")


USER_PROFILE_HEADER = (
    "WHAT YOU KNOW ABOUT THIS SPECIFIC PERSON (from earlier conversations — "
    "use it to recognize them and tailor tone/detail accordingly, but NEVER "
    "mention out loud that you're consulting stored notes about them):\n"
)

SHARED_CONTEXT_HEADER = (
    "WHAT THE ADMIN HAS PERSONALLY TOLD THIS SAME PERSON ELSEWHERE (their own "
    "Business chat or a group — a price, deadline, or commitment stated there "
    "is BINDING: never contradict it. Never reveal you're aware of another "
    "channel — just be consistent, as one team naturally would):\n"
)


async def _build_system_prompt(
    chat_id: int, agent: Agent, addendum: str = "", user_id: int | None = None
) -> str:
    facts = await memory.get_memory(chat_id)
    memory_block = (
        TEAM_MEMORY_HEADER + "\n".join(f"- {f}" for f in facts) + "\n\n" if facts else ""
    )
    profile_block = ""
    shared_block = ""
    if user_id is not None:
        notes = await user_profile.get_profile(user_id)
        if notes:
            profile_block = USER_PROFILE_HEADER + "\n".join(f"- {n}" for n in notes) + "\n\n"
        other_channels = shared_context.render_for_prompt(
            await shared_context.get_recent(user_id), exclude_channel="main_bot",
        )
        if other_channels:
            shared_block = SHARED_CONTEXT_HEADER + other_channels + "\n\n"
    parts = [memory_block + profile_block + shared_block + agent.system]
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


def _signature(agent: Agent, others: list[str] | None = None) -> str:
    """Who is speaking — '👩‍💼 Nodira · Senior Business Analyst'. Distinct from
    _header (a debug metadata line, off by default): this is the persona
    identity that makes the roster feel like a team, on by default.

    `others` names the specialists who also contributed to a synthesized
    answer, so a panel reply doesn't misrepresent itself as one person's
    work."""
    if not settings.show_agent_signature:
        return ""
    line = agent_label(agent)
    if others:
        names = ", ".join(persona(k)[0] for k in others)
        line += f" (+ {names})"
    return f"{line}\n\n"


def _agent_from_name_prefix(text: str) -> tuple[str | None, str]:
    """Address a specialist by name: 'Nodira, buni ko'rib chiq' ->
    ('ba', 'buni ko'rib chiq'). Returns (agent_key_or_None, remaining_text).

    Deterministic and free — no extra LLM call — and it beats the
    classifier when the user has explicitly named who they want. Only the
    FIRST token is considered, so a message that merely mentions the name
    later ("...buni Nodira aytdi") routes normally."""
    stripped = (text or "").lstrip()
    if not stripped:
        return None, text
    first, _, rest = stripped.partition(" ")
    # Tolerate "Nodira," / "Nodira:" / "@Nodira" — the natural ways people
    # address someone in chat.
    candidate = first.strip(",:;!?").lstrip("@")
    key = agent_key_by_name(candidate)
    if not key:
        return None, text
    remaining = rest.strip()
    # "Nodira" alone is a greeting, not a task — don't strip it to nothing.
    return (key, remaining) if remaining else (key, text)


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


async def _send_long(message: Message, text: str, reply_mode: bool = False) -> Message | None:
    """Send a (possibly long) reply, splitting at TELEGRAM_LIMIT. If reply_mode,
    the first chunk quotes the original message so group context is clear.
    Returns the LAST sent Message — callers can attach an inline keyboard to
    it (quick actions) via edit_reply_markup."""
    chunks = _split(text, TELEGRAM_LIMIT)
    sent: Message | None = None
    for i, chunk in enumerate(chunks):
        use_reply = reply_mode and i == 0
        try:
            if use_reply:
                sent = await message.reply(chunk, parse_mode="Markdown")
            else:
                sent = await message.answer(chunk, parse_mode="Markdown")
        except TelegramBadRequest:
            if use_reply:
                sent = await message.reply(chunk, parse_mode=None)
            else:
                sent = await message.answer(chunk, parse_mode=None)
    return sent


# Only offer quick actions under answers with enough substance to be worth
# turning into a document/task — short conversational replies would just
# collect button clutter.
_QUICK_ACTION_MIN_CHARS = 300


async def _offer_quick_actions(message: Message, sent: Message | None, user_text: str, body: str) -> None:
    """Attach one-tap follow-up buttons (Word/PDF, task) under a substantive
    answer in a PRIVATE chat — see quick_actions.py. Group answers are left
    clean, and short replies aren't worth converting."""
    if sent is None or message.chat.type != "private" or len(body) < _QUICK_ACTION_MIN_CHARS:
        return
    await quick_actions.link(sent.message_id, user_text, body)
    try:
        await sent.edit_reply_markup(reply_markup=quick_actions.keyboard(sent.message_id))
    except Exception:  # noqa: BLE001 — cosmetic; the answer itself already reached the user
        logger.exception("Failed to attach quick-action buttons")


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
            if current:  # never emit an empty chunk (Telegram rejects empty text)
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
            docgen.generate_proposal_content(route.agent, user_text, web_context=route.web_context),
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
- Language: detect from grammar, NOT from BA/IT keywords. Uzbek grammar → Uzbek.
  Russian grammar → Russian. DEFAULT to UZBEK when unsure. Never default to English.

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
    reply_mode: bool = False,
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

    # Hard input cap: combined text (message + quoted file + reply context)
    # can exceed the file extractor's own limit — trim before any LLM call.
    truncation_notice = ""
    if len(user_text) > MAX_USER_TEXT:
        logger.info("chat=%s input truncated from %d to %d chars", chat_id, len(user_text), MAX_USER_TEXT)
        user_text = user_text[:MAX_USER_TEXT]
        truncation_notice = (
            f"\n\n✂️ _Eslatma: matn juda uzun bo'lgani uchun birinchi "
            f"{MAX_USER_TEXT} belgisi tahlil qilindi._"
        )

    # Cost protection: don't stack a second multi-call pipeline on top of a
    # still-running one for the same chat.
    if chat_id in _in_flight:
        await message.answer("⏳ Oldingi so'rovingiz hali ishlanmoqda. Javobni kuting, keyin yuboring.")
        return
    _in_flight.add(chat_id)
    try:
        await _process_inner(message, bot, user_text, forced_type, forced_document, reply_mode, truncation_notice)
    finally:
        _in_flight.discard(chat_id)


async def _process_inner(
    message: Message,
    bot: Bot,
    user_text: str,
    forced_type: str | None,
    forced_document: bool | None,
    reply_mode: bool,
    truncation_notice: str,
) -> None:
    chat_id = message.chat.id
    await bot.send_chat_action(chat_id, ChatAction.TYPING)

    # Per-user profile memory (see user_profile.py): only for private-chat,
    # admin-approved, non-admin users, and only for substantive turns — a
    # bare "ha"/"ok" carries no signal worth an extraction call for.
    profile_uid = message.from_user.id if _profile_eligible(message) else None
    profile_worth_extracting = bool(profile_uid) and len(user_text) >= 12

    # Explicit name-addressing wins over the classifier: if the user wrote
    # "Nodira, ..." they've already chosen the specialist, so don't let a
    # probabilistic router override them (see _agent_from_name_prefix).
    named_key, stripped_text = _agent_from_name_prefix(user_text)
    if named_key:
        user_text = stripped_text

    last = await history.get_last_route(chat_id)
    route = await classify(
        user_text,
        last_agent=named_key or (last[0] if last else None),
        last_type=last[1] if last else None,
    )
    if named_key:
        route.agent = get_agent(named_key)
        # route.model is left as classified: model_for() picks purely on
        # complexity, not on which agent answers, so the classifier's
        # choice is still the right one for this request.
        # A named specialist answers personally — a chain/panel would put
        # other voices in front of the one the user actually asked for.
        route.execution_chain = []
        route.collaborators = []
    if forced_type:
        route.request_type = get_request_type(forced_type)
    if forced_document:
        route.wants_document = True

    if route.is_capability_question and not forced_type and not forced_document:
        logger.info("chat=%s -> capability overview", chat_id)
        await _send_capability_overview(message, bot, route, user_text)
        return

    # Live web grounding: the classifier flagged this as depending on
    # current facts (rates, weather, news, "latest") — fetch real results
    # BEFORE any agent runs, and hang them off the route so every downstream
    # path (single agent, collaborators, chain) picks them up via the
    # addendum. Failure is non-fatal: the agent then answers from its own
    # knowledge, exactly as before this feature existed.
    if route.needs_web and web_search.enabled():
        payload, search_err = await web_search.search(route.web_query or user_text)
        if payload:
            route.web_context = web_search.render_for_prompt(payload)
            logger.info("chat=%s -> web search OK for %.60s", chat_id, route.web_query)
        else:
            logger.warning("chat=%s -> web search failed: %s", chat_id, search_err)

    if route.wants_document:
        logger.info("chat=%s -> agent=%s DOCUMENT", chat_id, route.agent.key)
        await _send_document_deliverable(message, route, user_text)
        return

    if route.execution_chain:
        # Chain has its own log — the generic agent/model line below would be
        # misleading here (it shows only the primary agent, not the pipeline).
        logger.info(
            "chat=%s -> CHAIN %s type=%s",
            chat_id, route.execution_chain, route.request_type.key,
        )
        # Telegram's typing indicator only lasts ~5s; a 2-4 minute chain needs
        # a visible, updating status message so the user knows work is ongoing.
        status = await message.answer(
            f"🔄 Jamoam bilan ishlayapman... (0/{len(route.execution_chain)})"
        )

        async def _chain_progress(done: int, total: int, agent_name: str) -> None:
            try:
                await status.edit_text(
                    f"🔄 Jamoam bilan ishlayapman... {agent_name} tugatdi ({done}/{total})"
                )
            except TelegramBadRequest:
                pass

        body = await _run_sequential_chain(
            route, user_text, chat_id, progress=_chain_progress, user_id=profile_uid
        )
        try:
            await status.delete()
        except TelegramBadRequest:
            pass
        # Follow-ups should route to the LAST specialist in the chain — its
        # history holds the final, most complete state of the work.
        last_agent_key = route.execution_chain[-1]
        await history.set_last_route(chat_id, last_agent_key, route.request_type.key)
        footer = await _maybe_create_tickets(route, user_text, body)
        sent = await _send_long(message, _header(route) + body + footer + truncation_notice, reply_mode=reply_mode)
        await _offer_quick_actions(message, sent, user_text, body)
        if route.request_type.creates_ticket:
            asyncio.create_task(_maybe_extract_memory(chat_id, user_text, body))
        if profile_worth_extracting:
            asyncio.create_task(_maybe_extract_user_profile(profile_uid, user_text, body))
        if profile_uid:
            asyncio.create_task(_log_shared_context(profile_uid, user_text, body))
        return

    logger.info(
        "chat=%s -> agent=%s model=%s type=%s collaborators=%s",
        chat_id, route.agent.key, route.model, route.request_type.key, route.collaborators,
    )

    if route.execution_chain:
        logger.info(
            "chat=%s -> CHAIN %s", chat_id, route.execution_chain
        )
        body = await _run_sequential_chain(route, user_text, chat_id)
        await history.set_last_route(chat_id, route.agent.key, route.request_type.key)
        footer = await _maybe_create_tickets(route, user_text, body)
        await _send_long(message, _header(route) + body + footer, reply_mode=reply_mode)
        if route.request_type.creates_ticket:
            asyncio.create_task(_maybe_extract_memory(chat_id, user_text, body))
        return

    if route.collaborators:
        n = 1 + len(route.collaborators)
        status = await message.answer(f"🔄 Jamoam bilan ishlayapman... ({n} mutaxassis parallel)")
        body = await _run_collaborative_answer(route, user_text, chat_id, user_id=profile_uid)
        try:
            await status.delete()
        except TelegramBadRequest:
            pass
        await history.append(chat_id, route.agent.key, "user", user_text)
        await history.append(chat_id, route.agent.key, "assistant", body)
        await history.set_last_route(chat_id, route.agent.key, route.request_type.key)
        footer = await _maybe_create_tickets(route, user_text, body)
        sent = await _send_long(
            message,
            _header(route) + _signature(route.agent, route.collaborators) + body + footer + truncation_notice,
            reply_mode=reply_mode,
        )
        await _offer_quick_actions(message, sent, user_text, body)
        if route.request_type.creates_ticket:
            asyncio.create_task(_maybe_extract_memory(chat_id, user_text, body))
        if profile_worth_extracting:
            asyncio.create_task(_maybe_extract_user_profile(profile_uid, user_text, body))
        if profile_uid:
            asyncio.create_task(_log_shared_context(profile_uid, user_text, body))
        return

    system_prompt = await _build_system_prompt(
        chat_id, route.agent, route.request_type.addendum + route.web_context, user_id=profile_uid,
    )
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
                "⏳ Model hozir juda band (overloaded). "
                "30 soniya kutib, qaytadan yuboring."
            )
        elif "401" in msg or "authentication" in msg.lower():
            await message.answer("⚠️ API kaliti noto'g'ri yoki muddati o'tgan (provider sozlamalarini tekshiring).")
        else:
            await message.answer(f"⚠️ Xatolik yuz berdi: {msg[:300]}")
        return

    if not body:
        body = "(bo'sh javob)"

    await history.append(chat_id, route.agent.key, "user", user_text)
    await history.append(chat_id, route.agent.key, "assistant", body)
    await history.set_last_route(chat_id, route.agent.key, route.request_type.key)

    footer = await _maybe_create_tickets(route, user_text, body)
    sent = await _send_long(
        message,
        _header(route) + _signature(route.agent) + body + footer + truncation_notice,
        reply_mode=reply_mode,
    )
    await _offer_quick_actions(message, sent, user_text, body)
    if route.request_type.creates_ticket:
        asyncio.create_task(_maybe_extract_memory(chat_id, user_text, body))
    if profile_worth_extracting:
        asyncio.create_task(_maybe_extract_user_profile(profile_uid, user_text, body))
    if profile_uid:
        asyncio.create_task(_log_shared_context(profile_uid, user_text, body))


# --------------------------------------------------------------------------
# /kickoff — whole BA team responds together
# --------------------------------------------------------------------------
async def _kickoff_agent_call(
    agent_key: str, user_text: str, chat_id: int, user_id: int | None = None,
    web_context: str = "",
) -> dict:
    agent: Agent = get_agent(agent_key)
    # Intentionally always the "task" addendum: kickoff/collaborative calls ask
    # each specialist for a concrete deliverable regardless of request type.
    system_prompt = await _build_system_prompt(
        chat_id, agent, REQUEST_TYPES["task"].addendum + web_context, user_id=user_id,
    )
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


async def _run_collaborative_answer(
    route: Route, user_text: str, chat_id: int, user_id: int | None = None
) -> str:
    roles = [route.agent.key] + [k for k in route.collaborators if k != route.agent.key]
    roles = list(dict.fromkeys(roles))[:4]

    results = await asyncio.gather(
        *[
            _kickoff_agent_call(key, user_text, chat_id, user_id=user_id, web_context=route.web_context)
            for key in roles
        ]
    )

    for r in results:
        # Collaborators keep their own contribution in their thread. The
        # PRIMARY agent's thread gets the final SYNTHESIZED answer appended by
        # the caller (_process) — appending its raw draft here too would store
        # the same user turn twice and pollute its history.
        if r["ok"] and r["agent"].key != route.agent.key:
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
                model=route.model,
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


async def _run_sequential_chain(
    route: Route, user_text: str, chat_id: int, progress=None, user_id: int | None = None
) -> str:
    """Run agents in sequence, each specialist building on the previous output.

    `progress` is an optional async callback (done, total, agent_display_name)
    invoked after each agent finishes — used to keep the user informed during
    multi-minute chains (Telegram's typing indicator expires after ~5s)."""
    chain = route.execution_chain
    outputs: list[tuple[str, str]] = []

    for agent_key in chain:
        agent = get_agent(agent_key)
        system_prompt = await _build_system_prompt(
            chat_id, agent, route.request_type.addendum + route.web_context, user_id=user_id,
        )
        model, _ = model_for(agent, "high")

        if outputs:
            prev_work = "\n\n".join(
                f"=== {name} ===\n{body}" for name, body in outputs
            )
            content = (
                f"Original request:\n{user_text}\n\n"
                f"Previous specialists' work:\n{prev_work}\n\n"
                "Your task: add your specialist contribution building on the above."
            )
        else:
            content = user_text

        try:
            body = await asyncio.wait_for(
                claude_generate(system_prompt, [{"role": "user", "content": content}], model=model),
                timeout=settings.request_timeout,
            )
            body = (body or "").strip() or "(bo'sh javob)"
        except Exception as exc:
            logger.exception("Sequential chain agent=%s failed", agent_key)
            body = f"⚠️ {agent.display_name}: {exc}"

        outputs.append((agent_label(agent), body))
        await history.append(chat_id, agent_key, "user", user_text)
        await history.append(chat_id, agent_key, "assistant", body)

        if progress is not None:
            try:
                await progress(len(outputs), len(chain), agent.display_name)
            except Exception:  # noqa: BLE001 — a status-update hiccup must not kill the chain
                logger.exception("Chain progress callback failed (non-fatal)")

    if not outputs:
        return "(zanjir bo'sh)"
    return "\n\n---\n\n".join(f"### {name}\n\n{body}" for name, body in outputs)


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


_PROFILE_EXTRACT_SYSTEM = """
Given this one exchange between the AI team and an individual private-chat
user, is there ONE durable observation worth remembering about the USER
THEMSELVES for future conversations — their role/position, technical level,
communication style (terse vs detailed, formal vs casual), a recurring
concern or interest area, decision-making style, or similar? This is about
WHO THEY ARE as a person, not project facts (tracked separately elsewhere)
and not the literal content of this one message.

Do NOT record: project/company facts, a restatement of their question, or
anything speculative, judgmental, or unprofessional about their character.
Keep it a respectful, factual, behavioral observation.

If yes: respond with ONLY that one observation, under 150 characters,
ALWAYS in Uzbek regardless of what language the conversation was in (these
notes are for the admin's own reading, and for the AI to reuse next time).
If no durable observation: respond with exactly NONE.
"""


async def _log_shared_context(user_id: int, user_text: str, body: str) -> None:
    """Feed this main-bot exchange into the cross-channel store (see
    shared_context.py) so Business/Group Copilot suggestions for this same
    person later stay consistent with what the bot already told them
    directly. Verbatim, not LLM-extracted — unlike user_profile notes, this
    needs the actual wording to be useful for consistency checks."""
    await shared_context.append(user_id, "user", user_text, "main_bot")
    await shared_context.append(user_id, "bot", body, "main_bot")


async def _maybe_extract_user_profile(user_id: int, user_text: str, body: str) -> None:
    try:
        raw = await claude_generate_fast(
            _PROFILE_EXTRACT_SYSTEM,
            [{"role": "user", "content": f"USER'S MESSAGE:\n{user_text}\n\nTEAM'S ANSWER:\n{body[:1500]}"}],
            temperature=0.0,
        )
        note = (raw or "").strip()
        if note and note.upper() != "NONE" and len(note) < 300:
            await user_profile.add_profile_note(user_id, note)
    except Exception:
        logger.exception("User profile extraction failed (non-fatal)")


def _profile_eligible(message: Message) -> bool:
    """Per-user profile memory only applies to PRIVATE-chat, admin-approved,
    non-admin users (see user_profile.py docstring) — never the admin's own
    messages, and never groups."""
    return (
        message.chat.type == "private"
        and message.from_user is not None
        and not access_control.is_admin(message.from_user.id, message.from_user.username)
    )


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
    # Long parallel run — typing indicator expires in ~5s, keep a visible status.
    status = await message.answer(
        f"🔄 Jamoa ishlayapti ({len(KICKOFF_ROLES)} mutaxassis parallel)..."
    )
    results = await asyncio.gather(
        *[_kickoff_agent_call(key, user_text, chat_id) for key in KICKOFF_ROLES]
    )
    try:
        await status.delete()
    except TelegramBadRequest:
        pass

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
        "Process, Engineering, tarjima, diagramma, Jira va bank sohasi "
        "(kredit konveyer, core banking, skoring, integratsiya, sug'urta) "
        "mutaxassislariga o'zim yo'naltiraman.\n\n"
        "Asosiy buyruqlar:\n"
        "/agents — barcha mutaxassislar ro'yxati\n"
        "/kickoff — jamoa bilan birgalikda (BA + PM + Data + Process)\n"
        "/proposal — Word + PDF hujjat tayyorlash\n"
        "/addtask — vazifa/eslatma qo'shish (oddiy yozuvdan ham avtomatik aniqlayman)\n"
        "/tasks — faol vazifalar ro'yxati (/done, /canceltask bilan boshqarish)\n"
        "/digest on — har kuni ertalab kun rejasi\n"
        "/standup — kecha/bugun/blockers draft · /week — haftalik hisobot\n"
        "/minutes — uchrashuv yozuvidan protokol + action itemlar\n"
        "/decision — qarorlar jurnaliga yozish · /decisions — ko'rish\n"
        "/idea /task /bug /improve — so'rov turini majburlash\n"
        "/remember <fakt> — loyiha ma'lumotini yodlash\n"
        "/memory — yodlangan faktlar\n"
        "/reset — suhbatni tozalash\n"
        "/status — tizim holati\n\n"
        f"Fayllar: {SUPPORTED_SUMMARY}\n"
        f"`chat_id: {message.chat.id}`",
        parse_mode="Markdown",
    )


@dp.message(Command("agents", "team"))
async def cmd_agents(message: Message) -> None:
    """Discoverability: list every specialist so users know what to ask for."""
    if not _is_allowed(message.chat.id):
        return
    from agents import AGENTS

    lines = ["👥 **Jamoa — barcha mutaxassislar:**"]
    listed: set[str] = set()
    for group_name, members in AGENT_GROUPS:
        rows = []
        for key, blurb in members:
            agent = AGENTS.get(key)
            if agent is None:
                continue
            listed.add(key)
            rows.append(f"• {agent_label(agent)} — {blurb}")
        if rows:
            lines.append(f"\n**{group_name}:**")
            lines.extend(rows)
    # Future-proofing: any registered agent not in the catalogue still shows up.
    extra = [a for k, a in AGENTS.items() if k not in listed]
    if extra:
        lines.append("\n**Boshqa:**")
        lines.extend(f"• {agent_label(a)}" for a in extra)
    lines.append(
        "\nShunchaki savolingizni yozing — o'zim to'g'ri mutaxassisga yo'naltiraman.\n"
        "Yoki to'g'ridan-to'g'ri ismi bilan murojaat qiling: \"Nodira, talablarni yozib ber\"."
    )
    await _send_long(message, "\n".join(lines))


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
    if settings.provider == "hybrid":
        lines.append("**Provider:** ✅ Hybrid (OpenRouter + Claude)")
        lines.append(f"**Murakkab ish (hujjat/kod/tahlil):** {settings.claude_model_label} (Claude Sonnet)")
        lines.append(f"**Oddiy savollar + routing:** {settings.or_fast_model_label} (OpenRouter, bepul)")
        lines.append(f"**Vision/PDF:** Claude (native tahlil)")
    elif settings.provider == "openrouter":
        lines.append("**Provider:** ✅ OpenRouter (bepul)")
        lines.append(f"**Asosiy model:** {settings.or_main_model_label} (`{settings.or_main_model}`)")
        lines.append(f"**Tez model (routing):** {settings.or_fast_model_label}")
    elif settings.provider == "claude":
        lines.append("**Provider:** ✅ Claude (Anthropic)")
        lines.append(f"**Asosiy model:** {settings.claude_model_label}")
        lines.append(f"**Tez model:** {settings.claude_fast_model_label}")
    else:
        lines.append("**Provider:** ❌ Sozlanmagan (OPENROUTER_API_KEY yoki ANTHROPIC_API_KEY kerak)")
    db_status = "✅ Ulangan (users/tasks/decisions/memory)" if settings.db_enabled else "⚠️ Sozlanmagan (Redis/in-memory fallback)"
    lines.append(f"\n**PostgreSQL:** {db_status}")
    redis_status = "✅ Persistent" if settings.redis_enabled else "⚠️ In-memory (restart da yo'oladi)"
    lines.append(f"**Redis:** {redis_status}")
    lines.append(f"**GitHub:** {'✅ ' + settings.github_repo if settings.github_enabled else '❌ Sozlanmagan'}")
    lines.append(f"**Railway logs:** {'✅' if settings.railway_enabled else '❌ Sozlanmagan'}")
    lines.append(f"**Ovozli xabar (Groq Whisper):** {'✅' if settings.groq_enabled else '❌ Sozlanmagan (GROQ_API_KEY kerak)'}")
    lines.append(f"**Internetdan izlash (Tavily):** {'✅' if web_search.enabled() else '❌ Sozlanmagan (TAVILY_API_KEY kerak)'}")

    from agents import AGENTS
    lines.append(f"\n**Faol agentlar ({len(AGENTS)} ta):**")
    listed: set[str] = set()
    for group_name, members in AGENT_GROUPS:
        # Persona names only — the compact comma list would be unreadable
        # with full role titles; /agents shows name + role + blurb.
        names = [persona(k)[0] for k, _ in members if k in AGENTS]
        listed.update(k for k, _ in members if k in AGENTS)
        if names:
            lines.append(f"  _{group_name}:_ " + ", ".join(names))
    extra_names = [persona(k)[0] for k, _a in AGENTS.items() if k not in listed]
    if extra_names:
        lines.append("  _Boshqa:_ " + ", ".join(extra_names))
    lines.append("\nTo'liq ro'yxat va tavsiflar: /agents")

    await _send_long(message, "\n".join(lines))


@dp.message(Command("loops"))
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
# Daily task reminders
# --------------------------------------------------------------------------
@dp.message(Command("addtask"))
async def cmd_addtask(message: Message, command: CommandObject) -> None:
    chat_id = message.chat.id
    if not _is_allowed(chat_id):
        return
    text = (command.args or "").strip()
    if not text:
        await message.answer(
            "Vazifa matnini yozing, masalan:\n"
            "/addtask Ertaga soat 15:00 gacha mijoz uchun hisobot tayyorlash"
        )
        return

    status = await message.answer("🧠 Vazifani tahlil qilyapman...")
    uid = message.from_user.id if message.from_user else 0
    task = await task_assistant.build_task_from_text(chat_id, uid, text)
    if task is None:
        await status.edit_text(
            "⚠️ Vazifani aniqlab bo'lmadi. Nima va qachon kerakligini aniqroq yozib ko'ring."
        )
        return
    try:
        await status.delete()
    except TelegramBadRequest:
        pass
    await message.answer(
        task_assistant.format_confirmation(task),
        reply_markup=task_assistant.confirmation_keyboard(task.id),
    )


@dp.message(Command("tasks", "todo"))
async def cmd_tasks(message: Message) -> None:
    chat_id = message.chat.id
    if not _is_allowed(chat_id):
        return
    pending = await tasks.list_tasks(chat_id, {"pending"})
    if not pending:
        await message.answer("Hozircha faol vazifalar yo'q. /addtask bilan qo'shing.")
        return
    lines = ["📋 **Faol vazifalar:**\n"]
    lines.extend(task_assistant.format_task_line(t) for t in pending[:20])
    lines.append("\n✅ Bajarish: /done <id>   ❌ Bekor qilish: /canceltask <id>")
    await _send_long(message, "\n".join(lines))


@dp.message(Command("done"))
async def cmd_done(message: Message, command: CommandObject) -> None:
    if not _is_allowed(message.chat.id):
        return
    task_id = (command.args or "").strip().split()[0] if command.args else ""
    if not task_id:
        await message.answer("Vazifa ID sini yozing: /done <id>  (ID lar /tasks da ko'rsatilgan)")
        return
    task = await tasks.set_status(task_id, "done")
    if task is None or task.chat_id != message.chat.id:
        await message.answer("⚠️ Bunday vazifa topilmadi.")
        return
    await message.answer(f"✅ Bajarildi deb belgilandi: {task.title}")


@dp.message(Command("canceltask"))
async def cmd_canceltask(message: Message, command: CommandObject) -> None:
    if not _is_allowed(message.chat.id):
        return
    task_id = (command.args or "").strip().split()[0] if command.args else ""
    if not task_id:
        await message.answer("Vazifa ID sini yozing: /canceltask <id>")
        return
    task = await tasks.set_status(task_id, "cancelled")
    if task is None or task.chat_id != message.chat.id:
        await message.answer("⚠️ Bunday vazifa topilmadi.")
        return
    await message.answer(f"❌ Bekor qilindi: {task.title}")


# --------------------------------------------------------------------------
# PM/BA daily workflow: digest, standup, weekly review, minutes, decisions
# --------------------------------------------------------------------------
@dp.message(Command("digest"))
async def cmd_digest(message: Message, command: CommandObject) -> None:
    chat_id = message.chat.id
    if not _is_allowed(chat_id):
        return
    arg = (command.args or "").strip().lower()
    cfg = await digest.get_config(chat_id)

    if not arg:
        state = f"✅ yoqilgan, har kuni {cfg.time} da" if cfg.enabled else "❌ o'chirilgan"
        await message.answer(
            f"☀️ Kunlik reja (digest): {state}\n\n"
            "/digest on — yoqish (standart 08:30)\n"
            "/digest 07:45 — vaqtni o'zgartirish\n"
            "/digest off — o'chirish\n"
            "/digest now — hozir ko'rish"
        )
        return
    if arg == "now":
        await _send_long(message, await digest.build_digest(chat_id))
        return
    if arg in ("on", "yoq", "вкл"):
        cfg.enabled = True
        await digest.set_config(chat_id, cfg)
        await message.answer(f"☀️ Kunlik reja yoqildi — har kuni soat {cfg.time} da yuboraman.")
        return
    if arg in ("off", "o'chir", "выкл"):
        cfg.enabled = False
        await digest.set_config(chat_id, cfg)
        await message.answer("Kunlik reja o'chirildi.")
        return
    t = digest.normalize_time(arg)
    if t:
        cfg.time = t
        cfg.enabled = True
        await digest.set_config(chat_id, cfg)
        await message.answer(f"☀️ Kunlik reja endi har kuni soat {t} da keladi.")
        return
    await message.answer("Tushunmadim. /digest on | off | now | HH:MM")


@dp.message(Command("standup"))
async def cmd_standup(message: Message) -> None:
    if not _is_allowed(message.chat.id):
        return
    await _send_long(message, await digest.build_standup(message.chat.id))


@dp.message(Command("week"))
async def cmd_week(message: Message) -> None:
    if not _is_allowed(message.chat.id):
        return
    await _send_long(message, await digest.build_week(message.chat.id))


@dp.message(Command("hisobot", "report"))
async def cmd_report(message: Message) -> None:
    """Downloadable Word+PDF monthly snapshot: open/completed tasks,
    decisions, and expense-by-category totals. Deterministic — no LLM
    call, reuses document_generation's renderer with a hand-built content
    dict instead of an LLM-generated one (same pattern as /week's
    from-the-store-not-the-model philosophy)."""
    chat_id = message.chat.id
    if not _is_allowed(chat_id):
        return
    uid = message.from_user.id if message.from_user else 0
    status = await message.answer("📄 Oylik hisobotni tayyorlayapman...")
    try:
        content = await monthly_report.build_report_content(chat_id, uid)
        docx_bytes = docgen.render_docx(content)
        pdf_bytes = docgen.render_pdf(content)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Monthly report generation failed")
        await status.edit_text(f"⚠️ Hisobotni tayyorlashda xatolik: {exc}")
        return
    try:
        await status.delete()
    except TelegramBadRequest:
        pass
    title = content.get("title") or "Hisobot"
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", title)[:50].strip("_") or "hisobot"
    await message.answer_document(
        BufferedInputFile(docx_bytes, filename=f"{slug}.docx"),
        caption=f"📄 {title}\n\n{content.get('intro', '')}"[:1000],
    )
    await message.answer_document(BufferedInputFile(pdf_bytes, filename=f"{slug}.pdf"))


@dp.message(Command("decision"))
async def cmd_decision(message: Message, command: CommandObject) -> None:
    if not _is_allowed(message.chat.id):
        return
    text = (command.args or "").strip()
    if not text:
        await message.answer(
            "Qaror matnini yozing:\n/decision Integratsiya REST orqali, Kafka keyingi bosqichda"
        )
        return
    if await decisions_store.add_decision(message.chat.id, text):
        await message.answer("💾 Qarorlar jurnaliga yozildi. (/decisions — ko'rish)")
    else:
        await message.answer("⚠️ Saqlab bo'lmadi, qaytadan urinib ko'ring.")


@dp.message(Command("decisions"))
async def cmd_decisions(message: Message) -> None:
    if not _is_allowed(message.chat.id):
        return
    entries = await decisions_store.get_decisions(message.chat.id)
    if not entries:
        await message.answer("Qarorlar jurnali bo'sh. /decision <matn> bilan yozing.")
        return
    lines = "\n".join(f"{i}. {e}" for i, e in enumerate(entries, 1))
    await _send_long(message, f"📖 Qarorlar jurnali:\n\n{lines}")


@dp.message(Command("cleardecisions"))
async def cmd_cleardecisions(message: Message) -> None:
    if not _is_allowed(message.chat.id):
        return
    n = await decisions_store.clear_decisions(message.chat.id)
    await message.answer(f"O'chirildi: {n} ta qaror. ✅")


@dp.message(Command("minutes", "protokol"))
async def cmd_minutes(message: Message, command: CommandObject) -> None:
    chat_id = message.chat.id
    if not _is_allowed(chat_id):
        return
    text = (command.args or "").strip()
    if not text and message.reply_to_message:
        text = (message.reply_to_message.text or message.reply_to_message.caption or "").strip()
    if not text:
        await message.answer(
            "Uchrashuv yozuvlarini yuboring:\n"
            "/minutes <uchrashuv matni yoki transkript>\n"
            "yoki yozuvli xabarga reply qilib /minutes deb yozing."
        )
        return

    # Same cost-protection guard as the main pipeline: this is a single-shot
    # but large (4096-token, main-model) call — spamming /minutes shouldn't
    # stack several of those concurrently for one chat.
    if chat_id in _in_flight:
        await message.answer("⏳ Oldingi so'rovingiz hali ishlanmoqda. Javobni kuting, keyin yuboring.")
        return
    _in_flight.add(chat_id)
    try:
        status = await message.answer("📝 Protokol tayyorlayapman...")
        # No outer wait_for: extract_minutes makes up to TWO model calls
        # (primary + free-model fallback), each already bounded by
        # generate_json's own request_timeout — an outer cap at ONE
        # request_timeout would kill the fallback exactly when it's needed.
        try:
            data, error = await minutes_mod.extract_minutes(text)
        except asyncio.TimeoutError:
            data, error = None, "so'rov vaqti tugadi (timeout)"
        if data is None:
            await status.edit_text(
                "⚠️ Protokolni tuzib bo'lmadi. Matnni tekshirib, qaytadan yuboring.\n"
                f"Sabab: {error or 'nomaʼlum xatolik'}"
            )
            return
        try:
            await status.delete()
        except TelegramBadRequest:
            pass

        items = data["action_items"]
        decs = data["decisions"]
        keyboard = None
        if items or decs:
            batch_id = await minutes_mod.stash_batch(chat_id, data)
            keyboard = minutes_mod.minutes_keyboard(batch_id, len(items), len(decs))

        body = minutes_mod.render_minutes(data)
        # Keyboard must ride on the LAST chunk; _send_long doesn't support it,
        # so send the body first, then attach buttons to a short follow-up if long.
        if keyboard and len(body) <= TELEGRAM_LIMIT:
            try:
                await message.answer(body, reply_markup=keyboard)
            except TelegramBadRequest:
                await message.answer(body[:TELEGRAM_LIMIT], reply_markup=keyboard)
            return
        await _send_long(message, body)
        if keyboard:
            await message.answer("👇 Protokol bo'yicha amallar:", reply_markup=keyboard)
    finally:
        _in_flight.discard(chat_id)


@dp.callback_query(F.data.startswith("min:"))
async def handle_minutes_callback(callback: CallbackQuery, bot: Bot) -> None:
    if not callback.data or not callback.message:
        await callback.answer()
        return
    chat_id = callback.message.chat.id
    if not _is_allowed(chat_id):
        await callback.answer()
        return
    if not await _callback_still_authorized(callback):
        await callback.answer("Ruxsatingiz bekor qilingan.", show_alert=True)
        return
    try:
        _, action, batch_id = callback.data.split(":", 2)
    except ValueError:
        await callback.answer()
        return

    loaded = await minutes_mod.load_batch(batch_id)
    if loaded is None or loaded[0] != chat_id:
        await callback.answer("Bu protokol eskirgan (1 soatdan oshgan).", show_alert=True)
        return
    _, data = loaded

    if action == "t":  # create tasks from action items — explicit user permission
        if data.get("_tasks_added"):
            await callback.answer("Vazifalar allaqachon qo'shilgan.", show_alert=True)
            return
        uid = callback.from_user.id if callback.from_user else 0
        created = await minutes_mod.create_tasks_from_items(chat_id, uid, data)
        data["_tasks_added"] = True
        await minutes_mod.resave_batch(batch_id, chat_id, data)
        await callback.answer(f"{len(created)} ta vazifa qo'shildi ✅")
        if created:
            lines = ["➕ Protokoldan qo'shilgan vazifalar:\n"]
            lines.extend(task_assistant.format_task_line(t) for t in created)
            lines.append("\n📋 Boshqarish: /tasks")
            await bot.send_message(chat_id, "\n".join(lines))
        return

    if action == "d":  # save decisions to the decision log
        if data.get("_decisions_saved"):
            await callback.answer("Qarorlar allaqachon saqlangan.", show_alert=True)
            return
        saved = 0
        for d in (data.get("decisions") or [])[:15]:
            if await decisions_store.add_decision(chat_id, str(d)):
                saved += 1
        data["_decisions_saved"] = True
        await minutes_mod.resave_batch(batch_id, chat_id, data)
        await callback.answer(f"{saved} ta qaror jurnalga yozildi 💾")
        return

    await callback.answer()


@dp.callback_query(F.data.startswith("acc:"))
async def handle_access_callback(callback: CallbackQuery, bot: Bot) -> None:
    if not callback.data or not callback.message:
        await callback.answer()
        return
    admin_uid = callback.from_user.id if callback.from_user else 0
    admin_uname = callback.from_user.username if callback.from_user else None
    if not access_control.is_admin(admin_uid, admin_uname):
        await callback.answer("Bu tugma faqat admin uchun.", show_alert=True)
        return
    try:
        _, action, target_s = callback.data.split(":", 2)
        target_id = int(target_s)
    except ValueError:
        await callback.answer()
        return

    if action == "a":
        await access_control.approve(target_id, via="button")
        logger.info("ACCESS AUDIT: user=%s APPROVED via button by admin=%s", target_id, admin_uid)
        await callback.answer("Ruxsat berildi ✅")
        try:
            old = callback.message.text or ""
            await callback.message.edit_text(old + "\n\n✅ RUXSAT BERILDI", reply_markup=None)
        except Exception:  # noqa: BLE001 — best-effort cosmetic edit (stale/inaccessible message, etc.)
            pass
        await _start_onboarding(bot, target_id)
        return

    if action == "r":
        await access_control.deny(target_id, via="button")
        logger.info("ACCESS AUDIT: user=%s DENIED via button by admin=%s", target_id, admin_uid)
        await callback.answer("Rad etildi")
        try:
            old = callback.message.text or ""
            await callback.message.edit_text(old + "\n\n❌ RAD ETILDI", reply_markup=None)
        except Exception:  # noqa: BLE001 — best-effort cosmetic edit (stale/inaccessible message, etc.)
            pass
        await _notify_denied(bot, target_id)
        return

    await callback.answer()


@dp.message(Command("approve"))
async def cmd_approve(message: Message, command: CommandObject) -> None:
    uid = message.from_user.id if message.from_user else 0
    uname = message.from_user.username if message.from_user else None
    if not access_control.is_admin(uid, uname):
        return  # silently ignore — don't reveal admin-only commands to others
    arg = (command.args or "").strip().split()[0] if command.args else ""
    if not arg.isdigit():
        await message.answer("Foydalanuvchi ID sini yozing: /approve <user_id>")
        return
    target = int(arg)
    await access_control.approve(target, via="command")
    logger.info("ACCESS AUDIT: user=%s APPROVED via /approve by admin=%s", target, uid)
    await message.answer(f"✅ {target} ga ruxsat berildi.")
    await _start_onboarding(message.bot, target)


@dp.message(Command("deny"))
async def cmd_deny(message: Message, command: CommandObject) -> None:
    uid = message.from_user.id if message.from_user else 0
    uname = message.from_user.username if message.from_user else None
    if not access_control.is_admin(uid, uname):
        return
    arg = (command.args or "").strip().split()[0] if command.args else ""
    if not arg.isdigit():
        await message.answer("Foydalanuvchi ID sini yozing: /deny <user_id>")
        return
    target = int(arg)
    await access_control.deny(target, via="command")
    logger.info("ACCESS AUDIT: user=%s DENIED via /deny by admin=%s", target, uid)
    await message.answer(f"❌ {target} rad etildi.")
    await _notify_denied(message.bot, target)


@dp.message(Command("users"))
async def cmd_users(message: Message) -> None:
    uid = message.from_user.id if message.from_user else 0
    uname = message.from_user.username if message.from_user else None
    if not access_control.is_admin(uid, uname):
        return  # silently ignore — admin-only, don't reveal it to others

    users = await access_control.list_users()
    if not users:
        await message.answer("Hozircha hech kim botga yozmagan.")
        return

    status_emoji = {"approved": "✅", "denied": "❌", "pending": "⏳"}
    lines = [f"👥 Foydalanuvchilar ({len(users)} ta):\n"]
    for u in users[:60]:
        emoji = status_emoji.get(u.get("status"), "⏳")
        name = u.get("full_name") or "Noma'lum"
        uname_disp = f"@{u['username']}" if u.get("username") else "(username yo'q)"
        last_seen = (u.get("last_seen") or "")[:16].replace("T", " ")
        count = u.get("message_count") or "0"
        line = (
            f"{emoji} {name} {uname_disp} — ID: `{u['user_id']}` — {count} xabar"
            + (f" — oxirgi: {last_seen}" if last_seen else "")
        )
        if u.get("fio"):
            line += f"\n   👤 F.I.O: {u['fio']}"
        if u.get("phone"):
            line += f"\n   📱 {u['phone']}"
        # Audit trail: when/how the decision happened ("tugma" = the inline
        # ✅/❌ button, "buyruq" = /approve or /deny) — so an approval the
        # admin doesn't remember making can be traced instead of debated.
        via_label = {"button": "tugma", "command": "buyruq"}
        if u.get("status") == "approved" and u.get("approved_at"):
            ts = u["approved_at"][:16].replace("T", " ")
            line += f"\n   🕐 Ruxsat: {ts} ({via_label.get(u.get('approved_via'), '?')})"
        elif u.get("status") == "denied" and u.get("denied_at"):
            ts = u["denied_at"][:16].replace("T", " ")
            line += f"\n   🕐 Rad: {ts} ({via_label.get(u.get('denied_via'), '?')})"
        lines.append(line)
    lines.append(
        "\n⏳ kutilmoqda · ✅ ruxsat berilgan · ❌ rad etilgan\n"
        "Ruxsat berish: /approve <id>   Bekor qilish: /deny <id>   "
        "Batafsil: /whois <id>"
    )
    await _send_long(message, "\n".join(lines))


@dp.message(Command("whois"))
async def cmd_whois(message: Message, command: CommandObject) -> None:
    """Deep-dive on ONE user: full profile — who they are, not just that
    they exist. Complements /users (the list) with what the AI team has
    actually learned about this specific person over time (see
    user_profile.py) — this is the answer to "userni kimligini bilish"."""
    uid = message.from_user.id if message.from_user else 0
    uname = message.from_user.username if message.from_user else None
    if not access_control.is_admin(uid, uname):
        return

    arg = (command.args or "").strip().split()[0] if command.args else ""
    if not arg.isdigit():
        await message.answer("Foydalanuvchi ID sini yozing: /whois <user_id>")
        return
    target = int(arg)

    users = {u["user_id"]: u for u in await access_control.list_users()}
    u = users.get(target)
    if not u:
        await message.answer(f"ID {target} bo'yicha hech qanday yozuv topilmadi.")
        return

    status_label = {"approved": "✅ ruxsat berilgan", "denied": "❌ rad etilgan", "pending": "⏳ kutilmoqda"}
    lines = [
        f"👤 {u.get('full_name') or 'Noma’lum'} "
        + (f"@{u['username']}" if u.get("username") else "(username yo'q)"),
        f"ID: `{target}`",
        f"Holat: {status_label.get(u.get('status'), '?')}",
    ]
    if u.get("fio"):
        lines.append(f"F.I.O: {u['fio']}")
    if u.get("phone"):
        lines.append(f"📱 Telefon: {u['phone']}")
    if u.get("message_count"):
        lines.append(f"Xabarlar soni: {u['message_count']}")
    if u.get("first_seen"):
        lines.append(f"Birinchi murojaat: {u['first_seen'][:16].replace('T', ' ')}")
    if u.get("last_seen"):
        lines.append(f"Oxirgi murojaat: {u['last_seen'][:16].replace('T', ' ')}")
    via_label = {"button": "tugma", "command": "buyruq"}
    if u.get("status") == "approved" and u.get("approved_at"):
        lines.append(
            f"🕐 Ruxsat berilgan: {u['approved_at'][:16].replace('T', ' ')} "
            f"({via_label.get(u.get('approved_via'), '?')})"
        )
    elif u.get("status") == "denied" and u.get("denied_at"):
        lines.append(
            f"🕐 Rad etilgan: {u['denied_at'][:16].replace('T', ' ')} "
            f"({via_label.get(u.get('denied_via'), '?')})"
        )

    notes = await user_profile.get_profile(target)
    if notes:
        lines.append("\n🧠 Jamoa bu foydalanuvchi haqida bilib olganlari:")
        lines.extend(f"• {n}" for n in notes)
    else:
        lines.append("\n🧠 Bu foydalanuvchi haqida hali profil yozuvlari yo'q.")

    await _send_long(message, "\n".join(lines))


@dp.message(Command("stats"))
async def cmd_stats(message: Message) -> None:
    """Admin: one-screen usage overview aggregated from data the bot
    already records (/users' per-user activity) — no new tracking."""
    uid = message.from_user.id if message.from_user else 0
    uname = message.from_user.username if message.from_user else None
    if not access_control.is_admin(uid, uname):
        return

    users = await access_control.list_users()
    if not users:
        await message.answer("Hozircha hech qanday foydalanuvchi faoliyati yo'q.")
        return

    by_status = {"approved": 0, "denied": 0, "pending": 0}
    total_messages = 0
    with_phone = 0
    for u in users:
        by_status[u.get("status", "pending")] = by_status.get(u.get("status", "pending"), 0) + 1
        try:
            total_messages += int(u.get("message_count") or 0)
        except (TypeError, ValueError):
            pass
        if u.get("phone"):
            with_phone += 1

    def _count(u: dict) -> int:
        try:
            return int(u.get("message_count") or 0)
        except (TypeError, ValueError):
            return 0

    top = sorted(users, key=_count, reverse=True)[:5]
    lines = [
        "📊 Bot statistikasi\n",
        f"👥 Jami foydalanuvchilar: {len(users)}",
        f"✅ Ruxsat berilgan: {by_status.get('approved', 0)}",
        f"⏳ Kutilmoqda: {by_status.get('pending', 0)}",
        f"❌ Rad etilgan: {by_status.get('denied', 0)}",
        f"💬 Jami xabarlar: {total_messages}",
        f"📱 Telefon raqami bor: {with_phone}",
        "",
        "🏆 Eng faol foydalanuvchilar:",
    ]
    for u in top:
        if _count(u) == 0:
            continue
        name = u.get("full_name") or u.get("fio") or str(u["user_id"])
        lines.append(f"• {name} — {_count(u)} xabar")
    lines.append("")
    lines.append(
        f"🤖 Provider: {settings.provider} · 🗄 PostgreSQL: {'✅' if settings.db_enabled else '⚠️'} "
        f"· 💾 Redis: {'✅' if settings.redis_enabled else '⚠️ in-memory'}"
    )
    await _send_long(message, "\n".join(lines))


@dp.message(Command("broadcast"))
async def cmd_broadcast(message: Message, command: CommandObject) -> None:
    """Admin: send an announcement to every APPROVED user (bot updates,
    downtime notices, etc.) — previously required DMing each person."""
    uid = message.from_user.id if message.from_user else 0
    uname = message.from_user.username if message.from_user else None
    if not access_control.is_admin(uid, uname):
        return

    text = (command.args or "").strip()
    if not text:
        await message.answer(
            "E'lon matnini yozing:\n/broadcast Bot yangilandi — endi guruh "
            "mention'lari ham qo'llab-quvvatlanadi."
        )
        return

    users = await access_control.list_users()
    targets = [u["user_id"] for u in users if u.get("status") == "approved"]
    if not targets:
        await message.answer("Ruxsat berilgan foydalanuvchilar yo'q.")
        return

    sent_n, failed_n = 0, 0
    for target in targets:
        try:
            await message.bot.send_message(target, f"📢 Admin e'loni:\n\n{text}")
            sent_n += 1
        except Exception:  # noqa: BLE001 — user may have blocked the bot; keep going
            failed_n += 1
        # Stay well under Telegram's ~30 msg/s bot-wide send limit.
        await asyncio.sleep(0.05)
    await message.answer(f"📢 Yuborildi: {sent_n} ta · Yetkazilmadi: {failed_n} ta")


# --------------------------------------------------------------------------
# Business copilot — Telegram's native "Business > Chat automation" feature.
# Separate update types (business_connection / business_message), unrelated
# to the normal message flow above; see business_copilot.py for the design.
# --------------------------------------------------------------------------
def _business_can_reply(connection: BusinessConnection) -> bool:
    # can_reply moved under `rights` in newer Bot API versions but the
    # top-level field is kept for backward compatibility — prefer rights
    # when present, default True only if Telegram gave us neither.
    if connection.rights is not None:
        return bool(connection.rights.can_reply)
    if connection.can_reply is not None:
        return bool(connection.can_reply)
    return True


@dp.business_connection()
async def handle_business_connection(connection: BusinessConnection, bot: Bot) -> None:
    owner_id = connection.user.id
    owner_uname = connection.user.username
    if not access_control.is_admin(owner_id, owner_uname):
        logger.warning(
            "business_connection from non-admin user=%s — ignoring (only the "
            "configured admin's Business account is supported).", owner_id,
        )
        return
    can_reply = _business_can_reply(connection)
    await business_copilot.save_connection(
        connection.id, owner_id, connection.user_chat_id, connection.is_enabled, can_reply
    )
    logger.info(
        "Business connection %s %s (can_reply=%s) for admin=%s",
        connection.id, "enabled" if connection.is_enabled else "disabled", can_reply, owner_id,
    )
    if connection.is_enabled and not can_reply:
        try:
            await bot.send_message(
                connection.user_chat_id,
                "⚠️ Business Copilot ulandi, lekin \"Javob yozish\" (Reply to messages) "
                "ruxsati berilmagan — men xabarlarni tahlil qila olaman, lekin taklif "
                "qilingan javobni mijozga yubora olmayman.\n\n"
                "Tuzatish: Telegram → Sozlamalar → Business → Chatbots → botni tanlang → "
                "ruxsatlar ro'yxatida \"Javob yozish\"ni yoqing.",
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to send can_reply warning to admin")


@dp.business_message()
async def handle_business_message(message: Message, bot: Bot) -> None:
    conn_id = message.business_connection_id
    if not conn_id:
        return

    conn = await business_copilot.get_connection(conn_id)
    if conn is None:
        try:
            bc = await bot.get_business_connection(conn_id)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to fetch business connection %s", conn_id)
            return
        if not access_control.is_admin(bc.user.id, bc.user.username):
            return
        can_reply = _business_can_reply(bc)
        await business_copilot.save_connection(bc.id, bc.user.id, bc.user_chat_id, bc.is_enabled, can_reply)
        conn = {
            "owner_user_id": bc.user.id, "owner_chat_id": bc.user_chat_id,
            "is_enabled": bc.is_enabled, "can_reply": can_reply,
        }

    sender = message.from_user
    text = message.text or message.caption or "(media xabar)"

    if sender and sender.id == conn["owner_user_id"]:
        # The admin's own outgoing message in a connected chat (sent from
        # their phone, not via this bot) — just keep it in context, don't
        # analyze/notify about their own words.
        await business_copilot.append_own_message(conn_id, message.chat.id, text)
        return

    if not conn.get("is_enabled", True):
        return

    sender_name = sender.full_name if sender else "Noma'lum"
    await business_copilot.append_contact_message(conn_id, message.chat.id, text)

    owner_chat_id = conn["owner_chat_id"]
    data, error = await business_copilot.analyze(conn_id, message.chat.id, sender_name, text)
    if data is None:
        try:
            await bot.send_message(
                owner_chat_id,
                f"💼 {sender_name} sizga yozdi:\n\n{text}\n\n⚠️ AI tahlili muvaffaqiyatsiz — o'zingiz javob yozing.\n"
                f"Sabab: {error or 'nomaʼlum xatolik'}\n"
                "(Agar bu tez-tez takrorlansa, /status bilan provider holatini tekshiring.)",
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to notify admin of business message (analysis failed)")
        return

    analysis = data.get("analysis", "")
    suggested = data.get("suggested_reply", "")
    is_task = bool(data.get("is_task", False))
    can_reply = conn.get("can_reply", True)
    if not can_reply:
        action_line = (
            "⚠️ Ushbu ulanishda \"Javob yozish\" ruxsati yo'q — men bu javobni sizning "
            "nomingizdan yubora olmayman. Uni o'zingiz qo'lda yuboring."
        )
    elif is_task:
        action_line = (
            "🧠 Bu jiddiy vazifaga o'xshaydi — tez javob shunchaki xabar oldi deb "
            "bildiradi. Haqiqiy javob (BRD, texnik reja va h.k.) uchun \"Jamoa "
            "bilan ishlab chiqish\"ni bosing."
        )
    else:
        action_line = (
            "Yuborish uchun tugmani bosing, yoki shu xabarga Reply qilib o'zingiz "
            "yozgan javobni yuboring."
        )
    card = (
        f"💼 {sender_name} sizga yozdi:\n\n\"{text}\"\n\n"
        f"🧠 Tahlil: {analysis}\n\n"
        f"💬 Taklif qilingan javob:\n{suggested}\n\n{action_line}"
    )
    try:
        sent = await bot.send_message(owner_chat_id, card)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to send business-copilot notification to admin")
        return

    if not can_reply:
        return  # nothing to link/attach — the buttons would only fail too

    await business_copilot.link_relay(sent.message_id, conn_id, message.chat.id, suggested, text)
    try:
        await sent.edit_reply_markup(reply_markup=business_copilot.suggestion_keyboard(sent.message_id, is_task))
    except Exception:  # noqa: BLE001
        logger.exception("Failed to attach business-copilot suggestion buttons")


@dp.callback_query(F.data.startswith("biz:"))
async def handle_business_copilot_callback(callback: CallbackQuery, bot: Bot) -> None:
    if not callback.data or not callback.message:
        await callback.answer()
        return
    admin_uid = callback.from_user.id if callback.from_user else 0
    admin_uname = callback.from_user.username if callback.from_user else None
    if not access_control.is_admin(admin_uid, admin_uname):
        await callback.answer("Faqat admin uchun.", show_alert=True)
        return
    try:
        _, action, msg_id_s = callback.data.split(":", 2)
        relay_msg_id = int(msg_id_s)
    except ValueError:
        await callback.answer()
        return

    relay = await business_copilot.resolve_relay(relay_msg_id)
    if relay is None:
        await callback.answer("Bu so'rov eskirgan.", show_alert=True)
        return

    if action == "i":
        await business_copilot.clear_relay(relay_msg_id)
        await callback.answer("E'tiborsiz qoldirildi")
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except Exception:  # noqa: BLE001
            pass
        return

    if action == "s":
        try:
            await bot.send_message(relay["chat"], relay["reply"], business_connection_id=relay["conn"])
            await business_copilot.append_own_message(relay["conn"], relay["chat"], relay["reply"])
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to send business-copilot suggested reply")
            # Telegram caps callback-answer text at 200 chars — leave headroom for the prefix.
            await callback.answer(f"⚠️ Yuborib bo'lmadi: {str(exc)[:150]}", show_alert=True)
            return
        await business_copilot.clear_relay(relay_msg_id)
        await callback.answer("Yuborildi ✅")
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except Exception:  # noqa: BLE001
            pass
        return

    if action == "d":  # "Jamoa bilan ishlab chiqish" — run the full agent pipeline
        customer_text = relay.get("text") or ""
        if not customer_text:
            await callback.answer("Original xabar topilmadi.", show_alert=True)
            return
        await callback.answer("🧠 Jamoa ishlayapti, biroz kuting...")
        try:
            # Remove the "develop" button so a double-tap can't re-run the
            # whole (multi-LLM-call) pipeline while the first run is still
            # in flight or already answered; "Yuborish"/"E'tiborsiz" stay.
            await callback.message.edit_reply_markup(
                reply_markup=business_copilot.suggestion_keyboard(relay_msg_id, is_task=False)
            )
        except Exception:  # noqa: BLE001
            pass

        admin_chat_id = callback.message.chat.id
        route = await classify(customer_text)
        system_prompt = await _build_system_prompt(admin_chat_id, route.agent, route.request_type.addendum)
        try:
            body = await asyncio.wait_for(
                _answer_with_agent(route, system_prompt, [{"role": "user", "content": customer_text}]),
                timeout=settings.request_timeout,
            )
            body = (body or "").strip() or "(bo'sh javob)"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Business-copilot team development failed")
            await bot.send_message(admin_chat_id, f"⚠️ Jamoa javob bera olmadi: {str(exc)[:200]}")
            return

        full_text = f"🧠 {route.agent.display_name} tayyorladi:\n\n{body}"
        if len(full_text) > TELEGRAM_LIMIT:
            full_text = full_text[: TELEGRAM_LIMIT - 40] + "\n\n✂️ (davomi qisqartirildi)"
        try:
            sent = await bot.send_message(admin_chat_id, full_text)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to send team-developed answer to admin")
            return
        await business_copilot.link_relay(sent.message_id, relay["conn"], relay["chat"], body, customer_text)
        try:
            await sent.edit_reply_markup(reply_markup=business_copilot.suggestion_keyboard(sent.message_id, is_task=False))
        except Exception:  # noqa: BLE001
            logger.exception("Failed to attach send buttons to team-developed answer")
        return

    await callback.answer()


@dp.callback_query(F.data.startswith("grp:"))
async def handle_group_copilot_callback(callback: CallbackQuery, bot: Bot) -> None:
    if not callback.data or not callback.message:
        await callback.answer()
        return
    admin_uid = callback.from_user.id if callback.from_user else 0
    admin_uname = callback.from_user.username if callback.from_user else None
    if not access_control.is_admin(admin_uid, admin_uname):
        await callback.answer("Faqat admin uchun.", show_alert=True)
        return
    try:
        _, action, msg_id_s = callback.data.split(":", 2)
        relay_msg_id = int(msg_id_s)
    except ValueError:
        await callback.answer()
        return

    relay = await group_copilot.resolve_relay(relay_msg_id)
    if relay is None:
        await callback.answer("Bu so'rov eskirgan.", show_alert=True)
        return

    if action == "i":
        await group_copilot.clear_relay(relay_msg_id)
        await callback.answer("E'tiborsiz qoldirildi")
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except Exception:  # noqa: BLE001
            pass
        return

    if action == "s":
        try:
            await bot.send_message(
                relay["chat"], relay["reply"],
                reply_parameters=ReplyParameters(message_id=relay["reply_to"]),
            )
        except TelegramBadRequest:
            # Original message may have been deleted / too old to reply to.
            try:
                await bot.send_message(relay["chat"], relay["reply"])
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to send group-copilot suggested reply (fallback also failed)")
                await callback.answer(f"⚠️ Yuborib bo'lmadi: {str(exc)[:150]}", show_alert=True)
                return
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to send group-copilot suggested reply")
            await callback.answer(f"⚠️ Yuborib bo'lmadi: {str(exc)[:150]}", show_alert=True)
            return
        await group_copilot.clear_relay(relay_msg_id)
        if relay.get("sender_id"):
            await shared_context.append(relay["sender_id"], "admin", relay["reply"], "group")
        await callback.answer("Guruhga yuborildi ✅")
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except Exception:  # noqa: BLE001
            pass
        return

    await callback.answer()


@dp.callback_query(F.data.startswith("tour:"))
async def handle_tour_callback(callback: CallbackQuery, bot: Bot) -> None:
    """Welcome-tour buttons: one-tap daily-digest opt-in (the retention
    hook) or dismiss — see _send_welcome_tour."""
    if not callback.data or not callback.message:
        await callback.answer()
        return
    if not await _callback_still_authorized(callback):
        await callback.answer("Sizda botdan foydalanish ruxsati yo'q.", show_alert=True)
        return
    action = callback.data.split(":", 1)[1]
    chat_id = callback.message.chat.id

    if action == "d":
        cfg = await digest.get_config(chat_id)
        cfg.enabled = True
        await digest.set_config(chat_id, cfg)
        await callback.answer("Yoqildi ☀️")
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except Exception:  # noqa: BLE001
            pass
        await bot.send_message(
            chat_id,
            f"☀️ Kunlik reja yoqildi — har kuni soat {cfg.time} da yuboraman.\n"
            "O'zgartirish: /digest HH:MM · o'chirish: /digest off",
        )
        return

    if action == "x":
        await callback.answer("Xo'p. Keyin xohlasangiz: /digest on")
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except Exception:  # noqa: BLE001
            pass
        return

    await callback.answer()


async def _try_log_expense(message: Message, uid: int, chat_id: int, text: str) -> bool:
    """Parse and store a spend mention. Returns True if it was recorded (so
    the caller skips the normal AI pipeline). Any failure returns False and
    the message is answered normally — a missed capture is a small loss, a
    hijacked conversation is a big one."""
    parsed = await expenses.extract(text)
    if not parsed:
        return False
    expense_id = await expenses.add(
        uid, chat_id, parsed["amount"], parsed["currency"], parsed["category"], parsed["note"],
    )
    if expense_id is None:
        return False
    await message.answer(
        expenses.render_confirmation(
            parsed["amount"], parsed["currency"], parsed["category"], parsed["note"], expense_id,
        ),
        parse_mode="Markdown",
    )
    alert = await expenses.check_budget_alert(uid)
    if alert:
        await message.answer(alert)
    return True


@dp.message(Command("xarajat", "expense"))
async def cmd_expense(message: Message, command: CommandObject) -> None:
    """Explicit expense entry — the same parser as the automatic capture,
    for when the user wants to be sure it's recorded as a spend."""
    chat_id = message.chat.id
    if not _is_allowed(chat_id):
        return
    if not expenses.available():
        await message.answer(
            "💸 Xarajat hisobi uchun PostgreSQL kerak (moliyaviy yozuvlar "
            "restartda yo'qolmasligi uchun).\nRailway'da DATABASE_URL qo'shing."
        )
        return
    text = (command.args or "").strip()
    if not text:
        await message.answer(
            "Xarajatni yozing:\n/xarajat taksiga 30 ming\n\n"
            "Yoki shunchaki oddiy xabar sifatida yozavering — o'zim tushunaman.\n"
            "Hisobot: /xarajatlar"
        )
        return
    uid = message.from_user.id if message.from_user else 0
    if not await _try_log_expense(message, uid, chat_id, text):
        await message.answer(
            "Buni xarajat sifatida tushunolmadim. Masalan: /xarajat obed 45 ming"
        )


@dp.message(Command("xarajatlar", "expenses"))
async def cmd_expenses(message: Message, command: CommandObject) -> None:
    chat_id = message.chat.id
    if not _is_allowed(chat_id):
        return
    if not expenses.available():
        await message.answer(
            "💸 Xarajat hisobi uchun PostgreSQL kerak — Railway'da DATABASE_URL qo'shing."
        )
        return
    period = (command.args or "").strip().lower() or "oy"
    since, until, label = expenses.period_bounds(period)
    uid = message.from_user.id if message.from_user else 0
    data = await expenses.summary(uid, since, until)
    if data is None:
        await message.answer("⚠️ Hisobotni olishda xatolik. Keyinroq urinib ko'ring.")
        return
    await _send_long(message, expenses.render_summary(data, label))


@dp.message(Command("xarajatochir", "delexpense"))
async def cmd_delete_expense(message: Message, command: CommandObject) -> None:
    chat_id = message.chat.id
    if not _is_allowed(chat_id):
        return
    arg = (command.args or "").strip().split()[0] if command.args else ""
    if not arg.isdigit():
        await message.answer("O'chirish uchun ID yozing: /xarajatochir 42\n(ID lar /xarajatlar da)")
        return
    uid = message.from_user.id if message.from_user else 0
    if await expenses.delete(uid, int(arg)):
        await message.answer("🗑 O'chirildi.")
    else:
        await message.answer("Bunday yozuv topilmadi (yoki u sizniki emas).")


@dp.message(Command("byudjet", "budget"))
async def cmd_budget(message: Message, command: CommandObject) -> None:
    """Set or view a monthly spending limit. Alerts fire automatically from
    the expense-capture path (_try_log_expense) once 80%/100% is crossed."""
    chat_id = message.chat.id
    if not _is_allowed(chat_id):
        return
    if not expenses.available():
        await message.answer(
            "💸 Byudjet uchun PostgreSQL kerak — Railway'da DATABASE_URL qo'shing."
        )
        return
    uid = message.from_user.id if message.from_user else 0
    arg = (command.args or "").strip()
    if arg:
        amount = expenses.parse_amount_shorthand(arg)
        if not amount:
            await message.answer("Byudjet miqdorini yozing, masalan: /byudjet 3 mln")
            return
        await expenses.set_budget(uid, amount)
        await message.answer(f"✅ Oylik byudjet o'rnatildi: {expenses.fmt_amount(amount, 'UZS')}")
        return
    budget = await expenses.get_budget(uid)
    if not budget:
        await message.answer(
            "Oylik byudjet o'rnatilmagan.\nMasalan: /byudjet 3 mln — 80% va 100% da ogohlantiraman."
        )
        return
    spent = await expenses.month_to_date_uzs_total(uid)
    pct = int(spent / budget * 100) if budget else 0
    remaining = budget - spent
    lines = [
        f"💰 Oylik byudjet: {expenses.fmt_amount(budget, 'UZS')}",
        f"💸 Sarflandi: {expenses.fmt_amount(spent, 'UZS')} ({pct}%)",
    ]
    if remaining >= 0:
        lines.append(f"✅ Qoldi: {expenses.fmt_amount(remaining, 'UZS')}")
    else:
        lines.append(f"🔴 Oshib ketdi: {expenses.fmt_amount(-remaining, 'UZS')}")
    await message.answer("\n".join(lines))


@dp.message(Command("qidir", "find"))
async def cmd_find(message: Message, command: CommandObject) -> None:
    """Substring search across everything this chat has ACCUMULATED —
    vazifalar, qarorlar, xotira, xarajatlar — one place to find something
    already stored instead of scrolling chat history. Distinct from
    /search|/izla, which fetches fresh facts from the live internet."""
    chat_id = message.chat.id
    if not _is_allowed(chat_id):
        return
    query = (command.args or "").strip()
    if not query:
        await message.answer("Nimani qidirayotganingizni yozing:\n/qidir ijara")
        return
    low = query.lower()
    uid = message.from_user.id if message.from_user else 0
    sections: list[str] = []

    all_tasks = await tasks.list_tasks(chat_id, {"pending", "done", "cancelled"})
    task_hits = [
        t for t in all_tasks
        if low in t.title.lower() or low in t.description.lower()
    ][:8]
    if task_hits:
        sections.append("📋 Vazifalar:\n" + "\n".join(task_assistant.format_task_line(t) for t in task_hits))

    dec_hits = [d for d in await decisions_store.get_decisions(chat_id) if low in d.lower()][:8]
    if dec_hits:
        sections.append("📖 Qarorlar:\n" + "\n".join(f"• {d}" for d in dec_hits))

    mem_hits = [m for m in await memory.get_memory(chat_id) if low in m.lower()][:8]
    if mem_hits:
        sections.append("🧠 Xotira:\n" + "\n".join(f"• {m}" for m in mem_hits))

    if expenses.available():
        exp_hits = await expenses.search_notes(uid, query)
        if exp_hits:
            exp_lines = [
                f"`{r['id']}` {r['spent_at'].astimezone(tasks.TZ).strftime('%d-%m')} · "
                f"{expenses.fmt_amount(r['amount'], r['currency'])} · {r['category']}"
                + (f" ({r['note']})" if r["note"] else "")
                for r in exp_hits
            ]
            sections.append("💸 Xarajatlar:\n" + "\n".join(exp_lines))

    if not sections:
        await message.answer(f"🔎 \"{query}\" bo'yicha hech narsa topilmadi.")
        return
    await _send_long(message, f"🔎 \"{query}\" bo'yicha natijalar:\n\n" + "\n\n".join(sections))


@dp.message(Command("search", "izla"))
async def cmd_search(message: Message, command: CommandObject) -> None:
    """Explicit live web search. Normal messages get searched automatically
    when the classifier decides they need current facts (see
    router.needs_web); this is the manual override for when the user KNOWS
    they want fresh sources."""
    if not _is_allowed(message.chat.id):
        return
    query = (command.args or "").strip()
    if not query:
        await message.answer("Nimani izlashimni yozing:\n/search dollar kursi bugun")
        return
    if not web_search.enabled():
        await message.answer(
            "🔎 Internetdan izlash hozircha sozlanmagan.\n"
            "Yoqish uchun: tavily.com da bepul kalit oling (oyiga 1000 so'rov, "
            "karta shart emas) va Railway'da TAVILY_API_KEY qo'shing."
        )
        return

    status = await message.answer("🔎 Izlayapman...")
    payload, error = await web_search.search(query)
    if payload is None:
        await status.edit_text(f"⚠️ Izlab bo'lmadi.\nSabab: {error or 'nomaʼlum xatolik'}")
        return
    try:
        await status.delete()
    except TelegramBadRequest:
        pass
    await _send_long(message, web_search.render_for_user(payload))


@dp.message(Command("examples", "tour"))
async def cmd_examples(message: Message) -> None:
    """Re-open the welcome tour anytime — also the activation path for
    users approved BEFORE the tour existed."""
    if not _is_allowed(message.chat.id):
        return
    await _send_welcome_tour(message)


@dp.callback_query(F.data.startswith("qa:"))
async def handle_quick_action_callback(callback: CallbackQuery, bot: Bot) -> None:
    """One-tap follow-ups under an AI answer: 📄 render it as Word/PDF, or
    📋 turn the original request into a tracked task — see quick_actions.py."""
    if not callback.data or not callback.message:
        await callback.answer()
        return
    if not await _callback_still_authorized(callback):
        await callback.answer("Sizda botdan foydalanish ruxsati yo'q.", show_alert=True)
        return
    try:
        _, action, msg_id_s = callback.data.split(":", 2)
        qa_msg_id = int(msg_id_s)
    except ValueError:
        await callback.answer()
        return

    entry = await quick_actions.resolve(qa_msg_id)
    if entry is None:
        await callback.answer("Bu tugma eskirgan.", show_alert=True)
        return

    # ONE-SHOT: clear + strip the buttons up front so a double-tap can't
    # start a second document render or a duplicate task while the first
    # is still in flight.
    await quick_actions.clear(qa_msg_id)
    try:
        await callback.message.edit_reply_markup(reply_markup=None)
    except Exception:  # noqa: BLE001
        pass

    chat_id = callback.message.chat.id

    if action == "d":
        await callback.answer("📄 Hujjat tayyorlanmoqda...")
        status = await bot.send_message(chat_id, "📄 Hujjatni tayyorlayapman...")
        doc_prompt = (
            f"Foydalanuvchining asl so'rovi:\n{entry['text']}\n\n"
            f"Jamoa tayyorlagan javob — hujjatning asosiy mazmuni shu bo'lsin, "
            f"mazmunni o'zgartirmasdan professional hujjat ko'rinishiga keltir:\n{entry['body']}"
        )
        try:
            content = await asyncio.wait_for(
                docgen.generate_proposal_content(get_agent("ba"), doc_prompt),
                timeout=settings.request_timeout,
            )
            docx_bytes = docgen.render_docx(content)
            pdf_bytes = docgen.render_pdf(content)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Quick-action document generation failed")
            try:
                await status.edit_text(f"⚠️ Hujjatni tayyorlashda xatolik: {str(exc)[:200]}")
            except Exception:  # noqa: BLE001
                pass
            return
        try:
            await status.delete()
        except TelegramBadRequest:
            pass
        title = content.get("title") or "Hujjat"
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", title)[:50].strip("_") or "hujjat"
        await bot.send_document(
            chat_id, BufferedInputFile(docx_bytes, filename=f"{slug}.docx"), caption=f"📄 {title}"
        )
        await bot.send_document(chat_id, BufferedInputFile(pdf_bytes, filename=f"{slug}.pdf"))
        return

    if action == "t":
        await callback.answer("📋 Vazifa yaratilmoqda...")
        uid = callback.from_user.id if callback.from_user else 0
        task = await task_assistant.build_task_from_text(chat_id, uid, entry["text"])
        if task is None:
            await bot.send_message(
                chat_id,
                "⚠️ Vazifani avtomatik aniqlab bo'lmadi. /addtask bilan qo'lda "
                "qo'shishingiz mumkin.",
            )
            return
        await bot.send_message(
            chat_id,
            task_assistant.format_confirmation(task),
            reply_markup=task_assistant.confirmation_keyboard(task.id),
        )
        return

    await callback.answer()


@dp.callback_query(F.data.startswith("tsk:"))
async def handle_task_callback(callback: CallbackQuery, bot: Bot) -> None:
    if not callback.data or not callback.message:
        await callback.answer()
        return
    chat_id = callback.message.chat.id
    if not _is_allowed(chat_id):
        await callback.answer()
        return
    if not await _callback_still_authorized(callback):
        await callback.answer("Ruxsatingiz bekor qilingan.", show_alert=True)
        return
    try:
        _, action, task_id = callback.data.split(":", 2)
    except ValueError:
        await callback.answer()
        return

    task = await tasks.get_task(task_id)
    if task is None or task.chat_id != chat_id:
        await callback.answer("Vazifa topilmadi (eskirgan bo'lishi mumkin).", show_alert=True)
        return

    # Buttons stay on the sent message after it's resolved (Telegram keeps a
    # message's reply_markup unless a new one is explicitly set), so guard
    # every action except "done"/"cancel" themselves against a task that was
    # already resolved by an earlier click.
    if action in ("s", "w") and task.status != "pending":
        await callback.answer("Bu vazifa allaqachon yakunlangan.", show_alert=True)
        return

    if action == "d":  # done
        updated = await tasks.complete_occurrence(task_id)
        await callback.answer("Bajarildi deb belgilandi ✅")
        if updated is not None and task.recurrence != "none" and updated.status == "pending":
            # Recurring task: this occurrence is done but the reminder LIVES
            # ON for next time — must not read as "finished" the way a
            # one-off task's done-click does.
            streak_note = ""
            if task.recurrence == "daily":
                streak = await tasks.bump_daily_streak(task_id)
                if streak >= 2:
                    streak_note = f" 🔥 {streak} kunlik seriya!"
            next_due = datetime.fromisoformat(updated.due_at).astimezone(tasks.TZ)
            text = (
                f"✅ Bajarildi: {task.title}{streak_note}\n"
                f"🔁 Keyingisi: {next_due.strftime('%d-%m %H:%M')}"
            )
        else:
            text = f"✅ Bajarildi: {task.title}"
        try:
            await callback.message.edit_text(text, reply_markup=None)
        except Exception:  # noqa: BLE001 — best-effort cosmetic edit (stale/inaccessible message, etc.)
            pass
        return

    if action == "x":  # cancel
        await tasks.set_status(task_id, "cancelled")
        await callback.answer("Bekor qilindi")
        try:
            await callback.message.edit_text(f"❌ Bekor qilindi: {task.title}", reply_markup=None)
        except Exception:  # noqa: BLE001 — best-effort cosmetic edit (stale/inaccessible message, etc.)
            pass
        return

    if action == "s":  # snooze 30 min
        await tasks.snooze(task_id, 30)
        await callback.answer("30 daqiqaga kechiktirildi ⏰")
        try:
            await callback.message.edit_reply_markup(reply_markup=task_assistant.reminder_keyboard(task_id))
        except Exception:  # noqa: BLE001 — best-effort cosmetic edit (stale/inaccessible message, etc.)
            pass
        return

    if action == "w":  # work / discuss — only runs after this explicit button click
        await callback.answer("Boshladim 🤖")
        agent_hint = f" (mutaxassis: {task.agent_key})" if task.agent_key else ""
        prompt = (
            f"Quyidagi vazifani bajarishda yordam bering{agent_hint}:\n"
            f"Sarlavha: {task.title}\n"
            f"Tafsilot: {task.description or '—'}"
        )
        await _process(callback.message, bot, prompt, forced_type="task")
        return

    await callback.answer()


# --------------------------------------------------------------------------
# Free-text handler
# --------------------------------------------------------------------------
@dp.message(F.text & ~F.text.startswith("/"))
async def handle_text(message: Message, bot: Bot) -> None:
    chat_id = message.chat.id
    if not _is_allowed(chat_id):
        return

    me = await bot.me()
    username = me.username or ""
    is_direct = _should_answer(message, username)

    user_text = _strip_mention(message.text or "", username)
    if not user_text:
        return

    # If the user replied to a non-bot message, include the context:
    # - if the replied message has a FILE → download & extract it
    # - otherwise → prepend the quoted text
    if message.reply_to_message:
        replied = message.reply_to_message
        is_replied_to_bot = replied.from_user and replied.from_user.is_bot
        if not is_replied_to_bot:
            doc = replied.document
            photo = replied.photo
            if doc:
                max_bytes = settings.max_file_size_mb * 1024 * 1024
                if not doc.file_size or doc.file_size <= max_bytes:
                    try:
                        buf = await bot.download(doc.file_id)
                        data = buf.read() if hasattr(buf, "read") else bytes(buf)
                        extracted, _kind = await extract(doc.file_name, doc.mime_type, data)
                        if extracted:
                            label = doc.file_name or "fayl"
                            user_text = (
                                f"{user_text}\n\n"
                                f"--- Fayl: {label} ---\n{extracted}"
                            )
                    except Exception:
                        logger.exception("Failed to extract document from replied message")
            elif photo:
                try:
                    biggest = photo[-1]
                    buf = await bot.download(biggest.file_id)
                    data = buf.read() if hasattr(buf, "read") else bytes(buf)
                    extracted, _kind = await extract("photo.jpg", "image/jpeg", data)
                    if extracted:
                        user_text = f"{user_text}\n\n--- Rasm tahlili ---\n{extracted}"
                except Exception:
                    logger.exception("Failed to extract photo from replied message")
            else:
                quoted = (replied.text or replied.caption or "").strip()
                if quoted:
                    user_text = f'[Iqtibos: "{quoted[:600]}"]\n{user_text}'

    if is_direct:
        # Natural-language task/reminder detection — private chats only (avoids
        # misfiring in group business discussions). Cheap keyword pre-filter
        # first; the LLM call itself makes the real is_task decision.
        if message.chat.type == "private" and task_assistant.looks_like_task(user_text):
            uid = message.from_user.id if message.from_user else 0
            task = await task_assistant.build_task_from_text(chat_id, uid, user_text)
            if task is not None:
                await message.answer(
                    task_assistant.format_confirmation(task),
                    reply_markup=task_assistant.confirmation_keyboard(task.id),
                )
                return
        # Natural-language expense capture ("taksiga 30 ming"). Runs AFTER
        # task detection on purpose: "ertaga 500 ming to'lashim kerak" is a
        # future obligation (a task), not money already spent — and the
        # extractor is instructed to reject planned payments too, so the two
        # can't both claim the same message.
        if (
            message.chat.type == "private"
            and expenses.available()
            and expenses.looks_like_expense(user_text)
        ):
            uid = message.from_user.id if message.from_user else 0
            if await _try_log_expense(message, uid, chat_id, user_text):
                return
        # @mention, reply-to-bot, or private chat → respond normally
        await _process(message, bot, user_text, forced_type=None)
        return

    # Group mention copilot: this message wasn't addressed to the BOT, but
    # check whether it's addressed to the ADMIN specifically (mention/reply)
    # — see group_copilot.py. Independent of proactive/mention-required
    # settings below, which govern the bot answering the GROUP directly;
    # this always notifies the admin privately instead, never the group.
    if (
        settings.watch_group_mentions
        and message.chat.type != "private"
        and message.from_user
        and not access_control.is_admin(message.from_user.id, message.from_user.username)
        and _mentions_admin(message)
    ):
        await _handle_group_mention(message, bot, user_text)
        return

    # Proactive group mode: analyse all messages and join in when relevant
    if not settings.proactive_in_groups:
        return
    if message.chat.type == "private":
        return

    # Cooldown: don't respond more than once per N seconds per group
    now = time.monotonic()
    if now - _proactive_last.get(chat_id, 0) < settings.proactive_cooldown_seconds:
        return

    # Fast relevance check
    if not await _check_group_relevance(user_text):
        return

    logger.info("chat=%s PROACTIVE match: %.60s", chat_id, user_text)
    _proactive_last[chat_id] = now
    await _process(message, bot, user_text, forced_type=None, reply_mode=True)


# Registered LAST so every known Command() handler above matches first:
# typo'd / unknown commands get a help pointer instead of dead silence.
@dp.message(F.text.startswith("/"))
async def handle_unknown_command(message: Message, bot: Bot) -> None:
    if not _is_allowed(message.chat.id):
        return
    cmd = (message.text or "").split()[0]
    if message.chat.type != "private":
        # In groups, only react if the command explicitly targets THIS bot —
        # otherwise we'd answer other bots' commands with noise.
        if "@" not in cmd:
            return
        me = await bot.me()
        if cmd.rpartition("@")[2].lower() != (me.username or "").lower():
            return
    await message.answer(
        f"Noma'lum buyruq: {cmd.split('@')[0]}\n"
        "Buyruqlar ro'yxati: /help · Mutaxassislar: /agents"
    )


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------
async def main() -> None:
    problems = settings.validate()
    for p in problems:
        logger.warning("CONFIG: %s", p)

    if settings.db_enabled:
        if await db.init_schema():
            logger.info("PostgreSQL: durable storage ENABLED for users/tasks/decisions/memory.")
            await db.migrate_from_redis()
        else:
            logger.warning("DATABASE_URL set but PostgreSQL init failed — falling back to Redis/in-memory.")
    else:
        logger.info(
            "PostgreSQL: not configured — users/tasks/decisions/memory fall back to Redis/in-memory. "
            "Set DATABASE_URL for durable storage (strongly recommended)."
        )

    if settings.redis_enabled:
        client = redis_client.get_client()
        if client is not None:
            # from_url() is lazy — actually verify connectivity so the startup
            # log tells the truth about whether persistence is working.
            try:
                await client.ping()
                logger.info("Redis: persistent storage ENABLED (ping OK).")
            except Exception:  # noqa: BLE001
                logger.warning("Redis URL set but PING failed — history/memory calls will fall back to in-memory.")
        else:
            logger.warning("Redis URL set but client failed — falling back to in-memory.")
    else:
        logger.info("Redis: in-memory only (lost on restart). Set REDIS_URL to persist.")

    if settings.provider == "hybrid":
        logger.info(
            "Provider: HYBRID — complex work=%s | simple questions/routing=%s (free) | vision=Claude",
            settings.claude_model, settings.or_fast_model,
        )
    elif settings.provider == "openrouter":
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
    await _register_command_menu(bot)
    asyncio.create_task(task_assistant.reminder_loop(bot))
    asyncio.create_task(digest.digest_loop(bot))
    logger.info("Starting polling…")
    await dp.start_polling(bot)


_USER_COMMANDS: list[tuple[str, str]] = [
    ("start", "Bot haqida va asosiy buyruqlar"),
    ("examples", "Nima qila olaman — misollar bilan"),
    ("search", "Internetdan izlash"),
    ("xarajat", "Xarajat yozish"),
    ("xarajatlar", "Xarajatlar hisoboti"),
    ("byudjet", "Oylik byudjet o'rnatish/ko'rish"),
    ("qidir", "Vazifa/qaror/xotira/xarajat qidirish"),
    ("hisobot", "Oylik hisobot (Word + PDF)"),
    ("agents", "Barcha mutaxassislar ro'yxati"),
    ("kickoff", "Jamoa bilan birgalikda ishlash"),
    ("proposal", "Word + PDF hujjat tayyorlash"),
    ("addtask", "Vazifa/eslatma qo'shish"),
    ("tasks", "Faol vazifalar ro'yxati"),
    ("digest", "Har kuni ertalab kun rejasi"),
    ("standup", "Standup draft"),
    ("week", "Haftalik hisobot"),
    ("minutes", "Uchrashuv protokoli"),
    ("decisions", "Qarorlar jurnali"),
    ("remember", "Loyiha faktini yodlash"),
    ("memory", "Yodlangan faktlar"),
    ("reset", "Suhbatni tozalash"),
    ("status", "Tizim holati"),
    ("id", "Chat ID ni ko'rish"),
]

_ADMIN_COMMANDS: list[tuple[str, str]] = [
    ("users", "Foydalanuvchilar ro'yxati"),
    ("whois", "Bitta foydalanuvchi haqida to'liq"),
    ("stats", "Bot statistikasi"),
    ("broadcast", "Barcha approved userlarga e'lon"),
    ("approve", "Foydalanuvchiga ruxsat berish"),
    ("deny", "Ruxsatni bekor qilish"),
]


async def _register_command_menu(bot: Bot) -> None:
    """Populate Telegram's command menu (the [/] button) so users discover
    features without memorizing the /start wall of text. Admin-only commands
    are scoped to the admin's own chat (requires ADMIN_USER_ID) so regular
    users never see them in their menu."""
    try:
        await bot.set_my_commands([BotCommand(command=c, description=d) for c, d in _USER_COMMANDS])
    except Exception:  # noqa: BLE001 — menu is cosmetic, never block startup on it
        logger.exception("Failed to register the command menu (non-fatal)")
        return
    if settings.admin_user_id:
        try:
            await bot.set_my_commands(
                [BotCommand(command=c, description=d) for c, d in _USER_COMMANDS + _ADMIN_COMMANDS],
                scope=BotCommandScopeChat(chat_id=settings.admin_user_id),
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to register the admin command menu (non-fatal)")


if __name__ == "__main__":
    asyncio.run(main())
