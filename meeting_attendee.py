"""
Autonomous meeting attendee — joins a live Google Meet / Zoom / Microsoft
Teams call as a guest via browser automation (Playwright), captures the
call's audio, and turns it into the same structured minutes /minutes
already produces from pasted text.

CONSENT IS NOT OPTIONAL. Recording people without their knowledge is
illegal in a lot of places (two-party-consent US states, GDPR, and
similar rules elsewhere) and the liability for that lands on whoever
operates this bot, not on the code. So the join pipeline is a hard gate,
not a best-effort courtesy:

    join meeting -> announce in the meeting chat -> ONLY THEN start audio
    capture. If the announcement can't be confirmed sent (selectors
    changed, chat panel didn't open, permissions denied — meeting UIs
    break automation constantly), the bot leaves immediately WITHOUT
    ever starting the recorder. There is no configuration flag to skip
    the announcement — see _announce() and _run_session() below.

The `playwright` Python package ships in requirements.txt (small — it does
NOT pull the Chromium binary itself). Two things still can't be handled
from inside this process and need a one-time deploy-side setup:
  - The Chromium *browser binary* — downloaded automatically on first use
    (see _ensure_chromium()) the first time someone runs /uchrashuv, so no
    manual `playwright install` step is needed. Costs ~1-2 minutes on the
    very first call after each deploy (Railway's filesystem is ephemeral
    across deploys); every call after that is instant.
  - OS-level shared libraries Chromium links against (Debian: libnss3,
    libatk-bridge2.0-0, libgbm1, ...) and ffmpeg/PulseAudio for audio
    capture — these need root+apt at BUILD time, which this process
    cannot reach at runtime. The Dockerfile installs these (Railway builds
    from it directly whenever it's present at the repo root — see that
    file's own top comment for why an earlier version of this docstring
    pointed at a nixpacks env var instead, which was never actually in
    the build path).
Everything above is probed/attempted lazily at call time, never at import
time, so a deployment that never uses this feature is unaffected.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from aiogram.types import BufferedInputFile

import db
import file_processing
import minutes as minutes_mod
from config import settings

logger = logging.getLogger(__name__)

# How long a single ffmpeg segment chunk is, for Whisper transcription —
# keeps each chunk comfortably under Groq's free-tier upload size limit
# regardless of how long the meeting runs.
_CHUNK_SECONDS = 600
_JOIN_TIMEOUT_MS = 45_000
_ANNOUNCE_TIMEOUT_MS = 15_000


class MeetingPlatform(str, Enum):
    GOOGLE_MEET = "google_meet"
    ZOOM = "zoom"
    TEAMS = "teams"
    UNKNOWN = "unknown"


_PLATFORM_PATTERNS = (
    (re.compile(r"meet\.google\.com", re.I), MeetingPlatform.GOOGLE_MEET),
    (re.compile(r"zoom\.us", re.I), MeetingPlatform.ZOOM),
    (re.compile(r"teams\.(microsoft|live)\.com", re.I), MeetingPlatform.TEAMS),
)


def detect_platform(url: str) -> MeetingPlatform:
    for pattern, platform in _PLATFORM_PATTERNS:
        if pattern.search(url or ""):
            return platform
    return MeetingPlatform.UNKNOWN


def normalize_url(url: str) -> str:
    """Adds the https:// scheme when it's missing. People paste meeting
    links the way chat apps render them — "meet.google.com/abc-defg-hij",
    no scheme — and Playwright's page.goto() rejects a schemeless URL
    outright ("Cannot navigate to invalid URL"), which surfaced as a
    confusing "couldn't find the UI elements" failure with a blank
    about:blank screenshot: the browser never navigated anywhere at all."""
    url = (url or "").strip()
    # Strip surrounding angle brackets/quotes some clients add on paste.
    url = url.strip("<>\"'")
    if url and not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = f"https://{url}"
    return url


def disclosure_text(platform: MeetingPlatform) -> str:
    """The message posted in the meeting's chat on join. Kept short so it
    fits every platform's chat box, and deliberately not fully
    user-configurable — the org name is the only customizable part — so
    "turn off the announcement" isn't a reachable configuration."""
    name = settings.meeting_bot_display_name
    org = f" ({settings.meeting_bot_org_name})" if settings.meeting_bot_org_name else ""
    return (
        f"🤖 {name}{org} has joined this call to record audio and prepare "
        f"meeting minutes. This session is being recorded and transcribed. "
        f"/ Ushbu qo'ng'iroq protokol tayyorlash uchun yozib olinmoqda."
    )


@dataclass
class MeetingSession:
    session_id: str
    chat_id: int
    user_id: int
    meeting_url: str
    platform: MeetingPlatform
    status: str = "starting"  # starting|joining|announcing|recording|transcribing|done|failed|left
    disclosed: bool = False
    error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    audio_path: Path | None = None
    transcript_chars: int = 0
    stop_requested: bool = False
    task: "asyncio.Task | None" = None


# One active session per chat at a time — mirrors bot.py's _in_flight guard
# for /minutes and the main pipeline (no point joining two calls for the
# same chat concurrently, and it keeps the audio-sink naming simple).
_sessions: dict[int, MeetingSession] = {}


def get_active(chat_id: int) -> MeetingSession | None:
    return _sessions.get(chat_id)


def playwright_available() -> bool:
    return shutil.which("ffmpeg") is not None and _try_import_playwright() is not None


def _try_import_playwright():
    try:
        import playwright.async_api as pw  # noqa: F401
        return pw
    except ImportError:
        return None


class MeetingBotUnavailable(RuntimeError):
    pass


_chromium_ready = False


async def _ensure_chromium() -> tuple[bool, str | None]:
    """Downloads Playwright's Chromium binary on first use. Idempotent —
    the `playwright install` CLI already skips the download when the
    browser is present, so this is safe to call every session; the
    process-wide flag just avoids re-shelling out once we know it
    succeeded this run. Only caches SUCCESS — a transient network failure
    must not permanently lock out the next attempt."""
    global _chromium_ready
    if _chromium_ready:
        return True, None
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "playwright", "install", "chromium",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await proc.communicate()
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)[:500]
    if proc.returncode != 0:
        return False, (out or b"").decode(errors="replace")[-500:]
    _chromium_ready = True
    return True, None


def _load_storage_state():
    """Parses MEETING_STORAGE_STATE_JSON into the dict Playwright's
    new_context(storage_state=...) wants, or None when unset/malformed.
    Malformed never raises: an unusable saved session should degrade to an
    anonymous browser (which still works for Zoom/Teams) rather than
    killing every meeting session outright."""
    raw = (settings.meeting_storage_state_json or "").strip()
    if not raw:
        return None
    try:
        state = json.loads(raw)
    except ValueError:
        logger.warning("MEETING_STORAGE_STATE_JSON is not valid JSON — ignoring it (anonymous browser).")
        return None
    if not isinstance(state, dict) or not (state.get("cookies") or state.get("origins")):
        logger.warning("MEETING_STORAGE_STATE_JSON has no cookies/origins — ignoring it.")
        return None
    return state


def storage_state_json_configured() -> bool:
    """A pasted storage_state blob specifically (not auto-login)."""
    return _load_storage_state() is not None


def storage_state_configured() -> bool:
    """True when the browser will be signed in to Google by SOME means —
    either a pasted storage_state blob or auto-login credentials."""
    return storage_state_json_configured() or settings.meeting_google_auto_login_enabled


# Where a successful auto-login is cached. Google locks accounts that
# re-authenticate constantly, so the sign-in must happen once per container
# lifetime, not once per meeting. Railway's filesystem is wiped between
# deploys, which just means one fresh login after each deploy.
_GOOGLE_STATE_CACHE = Path("/tmp/meeting_google_state.json")
_google_login_failed_reason: str | None = None


def _read_cached_google_state():
    try:
        if _GOOGLE_STATE_CACHE.exists():
            return json.loads(_GOOGLE_STATE_CACHE.read_text())
    except Exception:  # noqa: BLE001
        logger.exception("Failed to read cached Google session — will re-login")
    return None


def invalidate_google_session() -> None:
    """Drops the cached sign-in so the next session logs in fresh. Called
    when Meet rejects us despite having a session — the usual cause is
    Google having expired it."""
    global _google_login_failed_reason
    _google_login_failed_reason = None
    try:
        _GOOGLE_STATE_CACHE.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to clear cached Google session")


async def _ensure_google_login(browser) -> tuple[dict | None, str | None]:
    """Signs the bot's browser in to Google with MEETING_GOOGLE_EMAIL/
    PASSWORD and returns its storage_state. Cached on success; a hard
    failure (bad password, 2FA, Google's automation block) is remembered
    for the process lifetime so every subsequent meeting doesn't retry a
    login that will fail the same way — and, more importantly, doesn't
    hammer Google with repeat attempts and get the account locked."""
    global _google_login_failed_reason
    if not settings.meeting_google_auto_login_enabled:
        return None, None
    if _google_login_failed_reason:
        return None, _google_login_failed_reason

    cached = _read_cached_google_state()
    if cached is not None:
        return cached, None

    context = None
    try:
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto(
            "https://accounts.google.com/ServiceLogin?service=mail",
            timeout=_JOIN_TIMEOUT_MS,
        )

        await page.fill('input[type="email"]', settings.meeting_google_email, timeout=15_000)
        await page.keyboard.press("Enter")
        try:
            await page.wait_for_selector('input[type="password"]', timeout=20_000)
        except Exception:  # noqa: BLE001
            reason = await _google_login_failure_reason(page, stage="email")
            _google_login_failed_reason = reason
            return None, reason

        await page.fill('input[type="password"]', settings.meeting_google_password, timeout=15_000)
        await page.keyboard.press("Enter")

        # A successful sign-in lands on myaccount/mail; anything else means
        # Google interrupted us (2FA, device confirmation, security block).
        try:
            await page.wait_for_url(
                re.compile(r"(myaccount\.google\.com|mail\.google\.com|google\.com/webhp)"),
                timeout=30_000,
            )
        except Exception:  # noqa: BLE001
            reason = await _google_login_failure_reason(page, stage="password")
            _google_login_failed_reason = reason
            return None, reason

        state = await context.storage_state()
        try:
            _GOOGLE_STATE_CACHE.write_text(json.dumps(state))
        except Exception:  # noqa: BLE001
            logger.exception("Failed to cache Google session (will re-login next time)")
        logger.info("Google auto-login succeeded for the meeting bot account.")
        return state, None
    except Exception as exc:  # noqa: BLE001
        logger.exception("Google auto-login raised")
        reason = f"Google'ga kirishda kutilmagan xatolik: {str(exc)[:200]}"
        _google_login_failed_reason = reason
        return None, reason
    finally:
        if context is not None:
            try:
                await context.close()
            except Exception:  # noqa: BLE001
                pass


async def _google_login_failure_reason(page, stage: str) -> str:
    """Turns whatever Google put on screen into an actionable message.
    These are the predicted failure modes, not hypotheticals — automated
    sign-in from a server IP is exactly what Google's defences target."""
    try:
        body = ((await page.inner_text("body")) or "").lower()
    except Exception:  # noqa: BLE001
        body = ""

    if "couldn't sign you in" in body or "browser or app may not be secure" in body:
        return (
            "Google avtomatik loginni bloklab qo'ydi (\"This browser or app may not be secure\"). "
            "Bu server IP'sidan kirishda odatiy hol va parolni o'zgartirish bilan hal bo'lmaydi.\n\n"
            "Yagona ishonchli yechim: MEETING_STORAGE_STATE_JSON ni qo'lda tayyorlash "
            "(README'dagi \"Google-сессия\" bo'limi). Yoki Zoom/Teams ishlating."
        )
    if "2-step verification" in body or "verify it's you" in body or "2-step" in body:
        return (
            "Akkauntda 2FA (ikki bosqichli tasdiqlash) yoqilgan — bot uni o'tolmaydi. "
            "Bot uchun 2FA'siz alohida akkaunt oching, yoki MEETING_STORAGE_STATE_JSON "
            "usulidan foydalaning."
        )
    if "wrong password" in body or "incorrect password" in body:
        return "Parol noto'g'ri — MEETING_GOOGLE_PASSWORD ni tekshiring."
    if "couldn't find your google account" in body or "enter a valid email" in body:
        return "Bunday Google akkaunt topilmadi — MEETING_GOOGLE_EMAIL ni tekshiring."
    if stage == "email":
        return "Google login sahifasida parol maydoniga o'tolmadim (Google jarayonni to'xtatdi)."
    return (
        "Google login yakunlanmadi — ehtimol qo'shimcha tasdiqlash so'raldi "
        "(telefon raqami, zaxira email yoki qurilmani tasdiqlash)."
    )


# Playwright's default UA contains "HeadlessChrome" even in headed mode on
# some builds, which is a trivial automation tell for anything checking.
_REALISTIC_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

_DISPLAY = ":99"
_display_ready = False


async def _ensure_display() -> str | None:
    """Starts a virtual X display (Xvfb) so a headed Chromium launch
    (settings.meeting_bot_headless=False, the default — meeting platforms
    are more likely to flag a real-headless browser as a bot) has
    somewhere to render. Returns the DISPLAY value to pass to Chromium, or
    None if Xvfb isn't available — caller falls back to real headless
    launch rather than hard-failing the whole session over this."""
    global _display_ready
    if _display_ready:
        return _DISPLAY
    if shutil.which("Xvfb") is None:
        return None
    try:
        await asyncio.create_subprocess_exec(
            "Xvfb", _DISPLAY, "-screen", "0", "1920x1080x24", "-nolisten", "tcp",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to start Xvfb")
        return None
    socket_path = Path(f"/tmp/.X11-unix/X{_DISPLAY.lstrip(':')}")
    for _ in range(30):
        if socket_path.exists():
            break
        await asyncio.sleep(0.1)
    else:
        logger.warning("Xvfb socket never appeared — falling back to headless Chromium")
        return None
    _display_ready = True
    return _DISPLAY


async def start(chat_id: int, user_id: int, meeting_url: str, bot) -> MeetingSession:
    """Kicks off a background session and returns immediately — joining,
    announcing, recording and transcribing all happen in the background
    task; progress/result is pushed to the chat via `bot`."""
    if not settings.meeting_bot_enabled:
        raise MeetingBotUnavailable(
            "Uchrashuvga qo'shilish funksiyasi o'chirilgan (MEETING_BOT_ENABLED=true qiling)."
        )
    if chat_id in _sessions and _sessions[chat_id].status not in ("done", "failed", "left"):
        raise MeetingBotUnavailable("Bu chat uchun allaqachon faol uchrashuv sessiyasi bor.")
    pw = _try_import_playwright()
    if pw is None:
        raise MeetingBotUnavailable(
            "Playwright python paketi o'rnatilmagan. Serverni requirements.txt "
            "yangilangan holatda qayta deploy qiling (`pip install playwright`)."
        )
    if shutil.which("ffmpeg") is None:
        raise MeetingBotUnavailable("ffmpeg topilmadi (audio yozib olish uchun kerak).")

    meeting_url = normalize_url(meeting_url)
    platform = detect_platform(meeting_url)
    if platform is MeetingPlatform.UNKNOWN:
        raise MeetingBotUnavailable(
            "Havola tanilmadi — faqat Google Meet, Zoom yoki Microsoft Teams qo'llab-quvvatlanadi."
        )

    session = MeetingSession(
        session_id=uuid.uuid4().hex[:12],
        chat_id=chat_id,
        user_id=user_id,
        meeting_url=meeting_url,
        platform=platform,
    )
    _sessions[chat_id] = session
    session.task = asyncio.create_task(_run_session(session, bot))
    return session


async def stop(chat_id: int) -> bool:
    """Signals a running session to leave and finalize. Returns False if
    there is nothing active for this chat."""
    session = _sessions.get(chat_id)
    if session is None or session.status in ("done", "failed", "left"):
        return False
    session.stop_requested = True
    return True


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
async def _run_session(session: MeetingSession, bot) -> None:
    chat_id = session.chat_id
    audio_dir = Path(settings.meeting_audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    raw_audio_path = audio_dir / f"{session.session_id}.wav"
    session.audio_path = raw_audio_path

    sink_name = f"meetbot_{session.session_id}"
    browser = None
    context = None
    ffmpeg_proc = None
    sink_ready = False
    try:
        session.status = "joining"
        pw = _try_import_playwright()
        ready, chromium_err = await _ensure_chromium()
        if not ready:
            session.status = "failed"
            session.error = f"Chromium o'rnata olmadim: {chromium_err or 'nomaʼlum xatolik'}"
            await _notify(bot, chat_id, f"⚠️ {session.error}")
            return
        sink_ready = await _setup_virtual_sink(sink_name)

        launch_headless = settings.meeting_bot_headless
        launch_env: dict[str, str] = {}
        if sink_ready:
            launch_env["PULSE_SINK"] = sink_name
        if not launch_headless:
            display = await _ensure_display()
            if display:
                launch_env["DISPLAY"] = display
            else:
                launch_headless = True

        async with pw.async_playwright() as p:
            browser = await p.chromium.launch(
                headless=launch_headless,
                args=[
                    "--autoplay-policy=no-user-gesture-required",
                    "--use-fake-ui-for-media-stream",
                    # Playwright's Chromium advertises itself as automated
                    # (navigator.webdriver, the "Chrome is being controlled
                    # by automated software" surface). Meeting platforms
                    # check for that, so a rejection can be about the
                    # browser looking automated rather than about the
                    # account — worth removing as a variable before
                    # concluding it's purely a permissions problem.
                    "--disable-blink-features=AutomationControlled",
                ],
                # Playwright's `env` REPLACES the child process's entire
                # environment rather than merging with it — passing just
                # {"DISPLAY": ...} would strip PATH/HOME/etc. and break the
                # launch outright, so start from a copy of our own env and
                # overlay only what we're actually adding.
                env={**os.environ, **launch_env} if launch_env else None,
            )
            # An explicitly pasted session wins; auto-login is the fallback.
            storage_state = _load_storage_state()
            if storage_state is None:
                storage_state, login_error = await _ensure_google_login(browser)
                # Meet cannot work without a signed-in browser, so a login
                # failure there is fatal and worth reporting properly.
                # Zoom/Teams take guests, so they carry on anonymously.
                if login_error and session.platform is MeetingPlatform.GOOGLE_MEET:
                    session.status = "failed"
                    session.error = login_error
                    await _notify(bot, chat_id, f"⚠️ {login_error}")
                    return

            context = await browser.new_context(
                permissions=["microphone", "camera"],
                storage_state=storage_state,
                user_agent=_REALISTIC_UA,
                viewport={"width": 1280, "height": 720},
                locale="en-US",
            )
            # The launch flag above doesn't clear navigator.webdriver on its
            # own; this does, before any page script can read it.
            await context.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
            )
            page = await context.new_page()

            joined, join_error = await _join(
                page, session.platform, session.meeting_url, settings.meeting_bot_display_name
            )
            if not joined:
                session.status = "failed"
                session.error = f"Uchrashuvga qo'shila olmadim — {join_error or 'sabab nomaʼlum'}"
                await _notify(bot, chat_id, f"⚠️ {session.error}")
                await _send_debug_screenshot(bot, chat_id, page, "Qo'shilishga urinilgan payt sahifa shunday ko'rinardi:")
                return

            # --- Mandatory disclosure gate --------------------------------
            session.status = "announcing"
            disclosed = await _announce(page, session.platform, disclosure_text(session.platform))
            session.disclosed = disclosed
            if not disclosed:
                session.status = "failed"
                session.error = (
                    "Chatga ochiq e'lon yubora olmadim — shu sabab ovoz yozib olishni "
                    "BOSHLAMADIM (yashirin yozib olish yo'q)."
                )
                await _send_debug_screenshot(bot, chat_id, page, "Chatga e'lon yozishga urinilgan payt sahifa shunday ko'rinardi:")
                await _leave(page, session.platform)
                await _notify(bot, chat_id, f"⚠️ {session.error}")
                return

            await _notify(
                bot, chat_id,
                "✅ Uchrashuvga qo'shildim va chatda ochiq e'lon qildim. Yozib olish boshlandi.",
            )

            # --- Audio capture (only reachable after disclosure) ---------
            session.status = "recording"
            ffmpeg_proc = await _start_recording(sink_name, raw_audio_path, sink_ready)

            deadline = asyncio.get_event_loop().time() + settings.meeting_max_duration_minutes * 60
            while not session.stop_requested and asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(5)
                if page.is_closed():
                    break

            await _leave(page, session.platform)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Meeting session %s failed", session.session_id)
        session.status = "failed"
        session.error = str(exc)[:300]
        await _notify(bot, chat_id, f"⚠️ Uchrashuv sessiyasida xatolik: {session.error}")
        return
    finally:
        if ffmpeg_proc is not None:
            await _stop_recording(ffmpeg_proc)
        await _teardown_virtual_sink(sink_name, sink_ready)
        try:
            if context is not None:
                await context.close()
            if browser is not None:
                await browser.close()
        except Exception:  # noqa: BLE001
            pass

    session.status = "transcribing"
    await _notify(bot, chat_id, "🎙 Yozib olish tugadi, transkript tayyorlanmoqda...")
    try:
        transcript = await transcribe_meeting_audio(raw_audio_path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Meeting transcription failed for session %s", session.session_id)
        session.status = "failed"
        session.error = f"Transkripsiya xatosi: {str(exc)[:200]}"
        await _notify(bot, chat_id, f"⚠️ {session.error}")
        return

    session.transcript_chars = len(transcript)
    session.status = "done"
    session.ended_at = datetime.now(timezone.utc)
    await _record_audit(session)

    if not transcript.strip():
        await _notify(bot, chat_id, "⚠️ Audio yozuvidan matn chiqmadi (jimjitlik yoki ovoz yozib olinmadi).")
        return

    await _deliver_minutes(bot, chat_id, transcript)


async def _notify(bot, chat_id: int, text: str) -> None:
    try:
        await bot.send_message(chat_id, text)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to notify chat=%s about meeting session progress", chat_id)


async def _send_debug_screenshot(bot, chat_id: int, page, caption: str) -> None:
    """Best-effort — sent on join/announce failure so a selector break can
    actually be diagnosed (what did the page look like?) instead of just
    guessed at from a generic error message. Never raises: a page that's
    already closed or a screenshot timeout must not mask the real error."""
    try:
        data = await page.screenshot(type="png", timeout=10_000)
        await bot.send_photo(chat_id, BufferedInputFile(data, filename="uchrashuv_debug.png"), caption=caption)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to capture/send debug screenshot for chat=%s", chat_id)


async def _deliver_minutes(bot, chat_id: int, transcript: str) -> None:
    data, error = await minutes_mod.extract_minutes(transcript)
    if data is None:
        await _notify(
            bot, chat_id,
            f"⚠️ Transkript tayyor, lekin protokolga aylantira olmadim: {error or 'nomaʼlum xatolik'}\n\n"
            f"Transkriptni o'zingiz /minutes ga yuborishingiz mumkin.",
        )
        return
    items = data["action_items"]
    decs = data["decisions"]
    keyboard = None
    if items or decs:
        batch_id = await minutes_mod.stash_batch(chat_id, data)
        keyboard = minutes_mod.minutes_keyboard(batch_id, len(items), len(decs))
    body = "🎥 Uchrashuv audiodan avtomatik protokol:\n\n" + minutes_mod.render_minutes(data)
    try:
        await bot.send_message(chat_id, body, reply_markup=keyboard)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to deliver auto-generated meeting minutes to chat=%s", chat_id)


# --------------------------------------------------------------------------
# Platform-specific join / announce / leave
#
# WHY THESE ARE FRAGILE ON PURPOSE (best-effort, multiple fallback
# selectors): none of these three products publish a stable DOM/automation
# contract for their web clients, and all three change markup often enough
# that hardcoding one selector breaks this within weeks. Every step here is
# wrapped so a selector miss fails that ONE step (join/announce returns
# False) rather than raising and losing the ability to leave cleanly.
# --------------------------------------------------------------------------
async def _join(
    page, platform: MeetingPlatform, meeting_url: str, display_name: str,
) -> tuple[bool, str | None]:
    """Returns (joined, reason_if_not). The reason is surfaced to the user
    verbatim — a generic "couldn't find the UI elements" message hid a
    plain navigation failure (schemeless URL) for a whole debugging round,
    so failures here say which step actually broke."""
    try:
        await page.goto(meeting_url, timeout=_JOIN_TIMEOUT_MS)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to navigate to meeting URL")
        return False, f"havolani ocholmadim: {str(exc)[:200]}"
    try:
        if platform is MeetingPlatform.GOOGLE_MEET:
            return await _join_google_meet(page, display_name)
        if platform is MeetingPlatform.ZOOM:
            return await _join_zoom(page, display_name)
        if platform is MeetingPlatform.TEAMS:
            return await _join_teams(page, display_name)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Join flow raised for platform=%s", platform.value)
        return False, f"qo'shilish jarayonida xatolik: {str(exc)[:200]}"
    return False, "qo'llab-quvvatlanmaydigan platforma"


async def _fill_name_field(page, display_name: str) -> None:
    for locator in (
        page.get_by_placeholder(re.compile("your name", re.I)),
        page.get_by_label(re.compile("your name", re.I)),
        page.locator('input[type="text"]').first,
    ):
        try:
            if await locator.count() > 0:
                await locator.fill(display_name, timeout=5000)
                return
        except Exception:  # noqa: BLE001
            continue


async def _click_first(page, patterns: list[str], timeout: int = 8000) -> bool:
    for pattern in patterns:
        try:
            btn = page.get_by_role("button", name=re.compile(pattern, re.I))
            if await btn.count() > 0:
                await btn.first.click(timeout=timeout)
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


async def _page_hint(page) -> str:
    """A short "what was actually on screen" string for failure messages —
    the page title plus the first bit of visible text. Google Meet's
    sign-in wall and "you can't join this call" states look identical to a
    selector break from the outside without this."""
    try:
        title = await page.title()
    except Exception:  # noqa: BLE001
        title = "?"
    body = ""
    try:
        body = ((await page.inner_text("body")) or "").strip().replace("\n", " / ")[:200]
    except Exception:  # noqa: BLE001
        pass
    return f"sahifa: \"{title}\"" + (f" — {body}" if body else "")


async def _meet_block_reason(page) -> str | None:
    """Distinguishes Google Meet's "we refuse you" walls from an ordinary
    selector miss. Meet blocks an anonymous (not-signed-in) browser before
    it ever offers the "Ask to join" knock, and the page it shows has no
    join button at all — indistinguishable from a broken selector unless
    the copy is actually read. Returns an actionable message, or None if
    this doesn't look like a block."""
    try:
        body = ((await page.inner_text("body")) or "").lower()
    except Exception:  # noqa: BLE001
        return None

    signed_in = storage_state_configured()
    if "you can't join this video call" in body or "you cannot join this video call" in body:
        if not signed_in:
            return (
                "Google Meet anonim (akkauntga kirmagan) brauzerni qo'ng'iroqqa umuman qo'ymaydi — "
                "\"Ask to join\" tugmasi ham ko'rsatilmaydi. Bu kod xatosi emas, Google siyosati, "
                "va kod bilan aylanib o'tib bo'lmaydi.\n\n"
                "HOZIR botda hech qanday Google akkaunt sozlanmagan. Railway → Variables'ga "
                "quyidagilardan BIRINI qo'shing:\n"
                "  • MEETING_GOOGLE_EMAIL + MEETING_GOOGLE_PASSWORD — bot o'zi kiradi "
                "(2FA'siz alohida akkaunt kerak; Google server IP'sidan bloklashi mumkin), yoki\n"
                "  • MEETING_STORAGE_STATE_JSON — qo'lda tayyorlangan sessiya (ishonchliroq, "
                "README'dagi \"Google-сессия\" bo'limi).\n\n"
                "Akkauntsiz sinash uchun: Zoom yoki Teams havolasini yuboring — ular mehmon "
                "sifatida kirishga ruxsat beradi."
            )
        # Most likely an expired session — drop the cached login so the
        # next attempt re-authenticates instead of reusing dead cookies.
        invalidate_google_session()
        return (
            "Google Meet bu akkauntni qo'ng'iroqqa qo'ymadi. Sabablari: sessiya eskirgan "
            "(keyingi urinishda qaytadan login qilaman — yana bir bor sinab ko'ring), yoki "
            "uchrashuv faqat tashkilot ichidagilar uchun ochiq, yoki uchrashuv hali boshlanmagan."
        )
    if "ask your host" in body or "denied your request" in body:
        return "Host botni qo'ng'iroqqa qabul qilmadi (so'rov rad etildi)."
    if "check your meeting code" in body or "invalid video call name" in body:
        return "Uchrashuv kodi noto'g'ri yoki havola eskirgan — havolani tekshirib qayta yuboring."
    return None


async def _join_google_meet(page, display_name: str) -> tuple[bool, str | None]:
    await _fill_name_field(page, display_name)
    if not await _click_first(page, [r"ask to join", r"join now"]):
        blocked = await _meet_block_reason(page)
        if blocked:
            return False, blocked
        return False, f"\"Ask to join\"/\"Join now\" tugmasi topilmadi ({await _page_hint(page)})"
    try:
        await page.get_by_role("button", name=re.compile(r"leave call|leave", re.I)).first.wait_for(
            timeout=_JOIN_TIMEOUT_MS
        )
        return True, None
    except Exception:  # noqa: BLE001
        return False, f"tugma bosildi, lekin qo'ng'iroqqa kirmadim — host qabul qilmadi yoki kutish xonasida qoldim ({await _page_hint(page)})"


async def _join_zoom(page, display_name: str) -> tuple[bool, str | None]:
    try:
        link = page.get_by_text(re.compile(r"join from.*browser", re.I))
        if await link.count() > 0:
            await link.first.click(timeout=8000)
    except Exception:  # noqa: BLE001
        pass
    await _fill_name_field(page, display_name)
    if not await _click_first(page, [r"^join$", r"join meeting"]):
        return False, f"\"Join\" tugmasi topilmadi ({await _page_hint(page)})"
    try:
        await page.get_by_text(re.compile(r"leave meeting|end meeting", re.I)).first.wait_for(
            timeout=_JOIN_TIMEOUT_MS
        )
        return True, None
    except Exception:  # noqa: BLE001
        # Common failure mode: stuck in the host's waiting room — not a
        # selector bug, just no admission within the timeout.
        return False, f"qo'ng'iroqqa kirmadim — host qabul qilmadi yoki kutish xonasida qoldim ({await _page_hint(page)})"


async def _join_teams(page, display_name: str) -> tuple[bool, str | None]:
    await _click_first(page, [r"continue on this browser", r"use the web app instead"])
    await _fill_name_field(page, display_name)
    if not await _click_first(page, [r"join now", r"join$"]):
        return False, f"\"Join now\" tugmasi topilmadi ({await _page_hint(page)})"
    try:
        await page.get_by_text(re.compile(r"leave", re.I)).first.wait_for(timeout=_JOIN_TIMEOUT_MS)
        return True, None
    except Exception:  # noqa: BLE001
        return False, f"qo'ng'iroqqa kirmadim — host qabul qilmadi yoki kutish xonasida qoldim ({await _page_hint(page)})"


async def _announce(page, platform: MeetingPlatform, text: str) -> bool:
    """Opens the in-call chat panel and posts `text`. Returns False (never
    raises past this point) on any failure — the caller treats False as
    "do not start recording"."""
    try:
        if platform is MeetingPlatform.GOOGLE_MEET:
            return await _announce_google_meet(page, text)
        if platform is MeetingPlatform.ZOOM:
            return await _announce_zoom(page, text)
        if platform is MeetingPlatform.TEAMS:
            return await _announce_teams(page, text)
    except Exception:  # noqa: BLE001
        logger.exception("Announce flow raised for platform=%s", platform.value)
    return False


async def _type_and_send(page, box, text: str) -> bool:
    try:
        await box.click(timeout=3000)
        await box.fill(text, timeout=3000)
        await box.press("Enter")
        return True
    except Exception:  # noqa: BLE001
        return False


async def _announce_google_meet(page, text: str) -> bool:
    if not await _click_first(page, [r"chat with everyone", r"^chat$"], timeout=_ANNOUNCE_TIMEOUT_MS):
        return False
    box = page.get_by_placeholder(re.compile("send a message", re.I))
    if await box.count() == 0:
        box = page.locator("textarea").last
    return await _type_and_send(page, box, text)


async def _announce_zoom(page, text: str) -> bool:
    if not await _click_first(page, [r"^chat$"], timeout=_ANNOUNCE_TIMEOUT_MS):
        return False
    box = page.get_by_placeholder(re.compile("type message", re.I))
    if await box.count() == 0:
        box = page.locator("textarea").last
    return await _type_and_send(page, box, text)


async def _announce_teams(page, text: str) -> bool:
    if not await _click_first(page, [r"show conversation", r"^chat$"], timeout=_ANNOUNCE_TIMEOUT_MS):
        return False
    box = page.get_by_placeholder(re.compile("type a (new )?message", re.I))
    if await box.count() == 0:
        box = page.locator("textarea").last
    return await _type_and_send(page, box, text)


async def _leave(page, platform: MeetingPlatform) -> None:
    try:
        await _click_first(page, [r"leave call", r"leave meeting", r"^leave$"], timeout=5000)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------
# Audio capture (PulseAudio null sink + ffmpeg) and transcription
# --------------------------------------------------------------------------
async def _run(*args: str) -> int:
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    return await proc.wait()


_pulseaudio_ready = False


async def _ensure_pulseaudio() -> bool:
    """Installing the `pulseaudio` package doesn't start a daemon — there's
    no desktop session to auto-launch one in a container. `pulseaudio
    --start` is the standard idempotent "start it if it isn't already
    running" invocation, safe to call every session."""
    global _pulseaudio_ready
    if _pulseaudio_ready:
        return True
    if shutil.which("pulseaudio") is None:
        return False
    try:
        rc = await _run("pulseaudio", "--start", "--exit-idle-time=-1")
    except Exception:  # noqa: BLE001
        logger.exception("Failed to start PulseAudio")
        return False
    if rc != 0:
        return False
    _pulseaudio_ready = True
    return True


async def _setup_virtual_sink(sink_name: str) -> bool:
    if shutil.which("pactl") is None:
        logger.warning("pactl not found — recording will capture silence unless PULSE_SINK routing exists.")
        return False
    if not await _ensure_pulseaudio():
        logger.warning("PulseAudio daemon unavailable — recording will capture silence.")
        return False
    try:
        rc = await _run("pactl", "load-module", "module-null-sink", f"sink_name={sink_name}")
        return rc == 0
    except Exception:  # noqa: BLE001
        logger.exception("Failed to create PulseAudio null sink %s", sink_name)
        return False


async def _teardown_virtual_sink(sink_name: str, was_created: bool) -> None:
    if not was_created or shutil.which("pactl") is None:
        return
    try:
        await _run("pactl", "unload-module", "module-null-sink")
    except Exception:  # noqa: BLE001
        logger.debug("Failed to unload null sink %s (non-fatal)", sink_name)


async def _start_recording(sink_name: str, out_path: Path, sink_ready: bool):
    source = f"{sink_name}.monitor" if sink_ready else "default"
    return await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-f", "pulse", "-i", source, "-ac", "1", "-ar", "16000", str(out_path),
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )


async def _stop_recording(proc) -> None:
    try:
        proc.terminate()
        await asyncio.wait_for(proc.wait(), timeout=10)
    except (ProcessLookupError, asyncio.TimeoutError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


async def transcribe_meeting_audio(wav_path: Path) -> str:
    """Splits the recording into `_CHUNK_SECONDS`-long chunks (so long
    meetings stay under Whisper's upload size limit) and transcribes each
    via the same Groq Whisper client voice messages already use."""
    if not wav_path.exists() or wav_path.stat().st_size == 0:
        return ""
    with tempfile.TemporaryDirectory() as tmp:
        pattern = str(Path(tmp) / "chunk_%03d.wav")
        rc = await _run(
            "ffmpeg", "-y", "-i", str(wav_path),
            "-f", "segment", "-segment_time", str(_CHUNK_SECONDS), "-c", "copy", pattern,
        )
        if rc != 0:
            raise RuntimeError("ffmpeg segmenting failed")
        chunks = sorted(Path(tmp).glob("chunk_*.wav"))
        pieces: list[str] = []
        for chunk in chunks:
            data = chunk.read_bytes()
            if not data:
                continue
            try:
                text = await file_processing.transcribe_audio(data, chunk.name)
            except Exception:  # noqa: BLE001
                logger.exception("Chunk transcription failed for %s", chunk.name)
                continue
            if text.strip():
                pieces.append(text.strip())
        return "\n".join(pieces)


# --------------------------------------------------------------------------
# Audit trail (best-effort; Postgres only — see db.py's fallback docstring.
# Without DATABASE_URL, sessions still run correctly, they're just not
# queryable history afterwards)
# --------------------------------------------------------------------------
async def _record_audit(session: MeetingSession) -> None:
    pool = None
    try:
        if await db.init_schema():
            pool = await db.get_pool()
    except Exception:  # noqa: BLE001
        pool = None
    if pool is None:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO meeting_sessions
                    (id, chat_id, user_id, platform, meeting_url, status, disclosed,
                     started_at, ended_at, error, transcript_chars)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (id) DO UPDATE SET
                    status = EXCLUDED.status, disclosed = EXCLUDED.disclosed,
                    ended_at = EXCLUDED.ended_at, error = EXCLUDED.error,
                    transcript_chars = EXCLUDED.transcript_chars
                """,
                session.session_id, session.chat_id, session.user_id, session.platform.value,
                session.meeting_url, session.status, session.disclosed, session.started_at,
                session.ended_at, session.error, session.transcript_chars,
            )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to record meeting session audit row")


async def list_recent(chat_id: int, limit: int = 10) -> list[dict]:
    try:
        if not await db.init_schema():
            return []
        pool = await db.get_pool()
        if pool is None:
            return []
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, platform, meeting_url, status, disclosed, started_at, ended_at, transcript_chars
                FROM meeting_sessions WHERE chat_id = $1 ORDER BY started_at DESC LIMIT $2
                """,
                chat_id, limit,
            )
        return [dict(r) for r in rows]
    except Exception:  # noqa: BLE001
        logger.exception("Failed to list recent meeting sessions")
        return []
