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
import logging
import re
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

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

        async with pw.async_playwright() as p:
            browser = await p.chromium.launch(
                headless=settings.meeting_bot_headless,
                args=[
                    "--autoplay-policy=no-user-gesture-required",
                    "--use-fake-ui-for-media-stream",
                ],
                env={"PULSE_SINK": sink_name} if sink_ready else None,
            )
            context = await browser.new_context(permissions=["microphone", "camera"])
            page = await context.new_page()

            joined = await _join(
                page, session.platform, session.meeting_url, settings.meeting_bot_display_name
            )
            if not joined:
                session.status = "failed"
                session.error = "Uchrashuvga qo'shila olmadim (interfeys elementlari topilmadi yoki host qabul qilmadi)."
                await _notify(bot, chat_id, f"⚠️ {session.error}")
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
async def _join(page, platform: MeetingPlatform, meeting_url: str, display_name: str) -> bool:
    try:
        await page.goto(meeting_url, timeout=_JOIN_TIMEOUT_MS)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to navigate to meeting URL")
        return False
    try:
        if platform is MeetingPlatform.GOOGLE_MEET:
            return await _join_google_meet(page, display_name)
        if platform is MeetingPlatform.ZOOM:
            return await _join_zoom(page, display_name)
        if platform is MeetingPlatform.TEAMS:
            return await _join_teams(page, display_name)
    except Exception:  # noqa: BLE001
        logger.exception("Join flow raised for platform=%s", platform.value)
    return False


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


async def _join_google_meet(page, display_name: str) -> bool:
    await _fill_name_field(page, display_name)
    if not await _click_first(page, [r"ask to join", r"join now"]):
        return False
    try:
        await page.get_by_role("button", name=re.compile(r"leave call|leave", re.I)).first.wait_for(
            timeout=_JOIN_TIMEOUT_MS
        )
        return True
    except Exception:  # noqa: BLE001
        return False


async def _join_zoom(page, display_name: str) -> bool:
    try:
        link = page.get_by_text(re.compile(r"join from.*browser", re.I))
        if await link.count() > 0:
            await link.first.click(timeout=8000)
    except Exception:  # noqa: BLE001
        pass
    await _fill_name_field(page, display_name)
    if not await _click_first(page, [r"^join$", r"join meeting"]):
        return False
    try:
        await page.get_by_text(re.compile(r"leave meeting|end meeting", re.I)).first.wait_for(
            timeout=_JOIN_TIMEOUT_MS
        )
        return True
    except Exception:  # noqa: BLE001
        # Common failure mode: stuck in the host's waiting room — not a
        # selector bug, just no admission within the timeout.
        return False


async def _join_teams(page, display_name: str) -> bool:
    await _click_first(page, [r"continue on this browser", r"use the web app instead"])
    await _fill_name_field(page, display_name)
    if not await _click_first(page, [r"join now", r"join$"]):
        return False
    try:
        await page.get_by_text(re.compile(r"leave", re.I)).first.wait_for(timeout=_JOIN_TIMEOUT_MS)
        return True
    except Exception:  # noqa: BLE001
        return False


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


async def _setup_virtual_sink(sink_name: str) -> bool:
    if shutil.which("pactl") is None:
        logger.warning("pactl not found — recording will capture silence unless PULSE_SINK routing exists.")
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
