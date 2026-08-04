"""meeting_attendee.py — platform detection, disclosure copy, config
gating, and (most important) the safety invariant that audio capture is
never started unless the in-meeting disclosure announcement is confirmed
sent. The DOM-scraping join/announce selectors themselves need a real
browser + a real meeting and aren't exercised here — these tests cover the
orchestration logic around them.
"""

import meeting_attendee as ma


def test_platform_detection():
    assert ma.detect_platform("https://meet.google.com/abc-defg-hij") == ma.MeetingPlatform.GOOGLE_MEET
    assert ma.detect_platform("https://us02web.zoom.us/j/123456789") == ma.MeetingPlatform.ZOOM
    assert ma.detect_platform("https://teams.microsoft.com/l/meetup-join/xyz") == ma.MeetingPlatform.TEAMS
    assert ma.detect_platform("https://example.com/not-a-meeting") == ma.MeetingPlatform.UNKNOWN
    assert ma.detect_platform("") == ma.MeetingPlatform.UNKNOWN


def test_disclosure_text_states_recording_clearly(monkeypatch):
    monkeypatch.setattr(ma.settings, "meeting_bot_org_name", "", raising=False)
    text = ma.disclosure_text(ma.MeetingPlatform.GOOGLE_MEET)
    assert "record" in text.lower()
    assert "transcri" in text.lower()


def test_disclosure_text_includes_org_name_when_set(monkeypatch):
    monkeypatch.setattr(ma.settings, "meeting_bot_org_name", "Acme Analytics", raising=False)
    text = ma.disclosure_text(ma.MeetingPlatform.ZOOM)
    assert "Acme Analytics" in text


async def test_start_rejects_when_feature_disabled(monkeypatch):
    monkeypatch.setattr(ma.settings, "meeting_bot_enabled", False, raising=False)
    try:
        await ma.start(111, 222, "https://meet.google.com/abc-defg-hij", bot=None)
        assert False, "expected MeetingBotUnavailable"
    except ma.MeetingBotUnavailable:
        pass


async def test_start_rejects_unsupported_url(monkeypatch):
    monkeypatch.setattr(ma.settings, "meeting_bot_enabled", True, raising=False)
    monkeypatch.setattr(ma, "_try_import_playwright", lambda: object())
    monkeypatch.setattr(ma.shutil, "which", lambda name: "/usr/bin/" + name)
    try:
        await ma.start(111, 222, "https://example.com/not-a-meeting", bot=None)
        assert False, "expected MeetingBotUnavailable"
    except ma.MeetingBotUnavailable:
        pass
    assert 111 not in ma._sessions


# --------------------------------------------------------------------------
# Fakes for the orchestration test: a minimal stand-in for Playwright's
# async API, just enough to drive _run_session() without a real browser.
# --------------------------------------------------------------------------
class _FakeBrowser:
    async def new_context(self, **kw):
        return _FakeContext()

    async def close(self):
        pass


class _FakeContext:
    async def new_page(self):
        return _FakePage()

    async def close(self):
        pass


class _FakePage:
    def is_closed(self):
        return False


class _FakeChromium:
    async def launch(self, **kw):
        return _FakeBrowser()


class _FakeAsyncPlaywright:
    def __init__(self):
        self.chromium = _FakeChromium()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakePlaywrightModule:
    def async_playwright(self):
        return _FakeAsyncPlaywright()


class _FakeBot:
    def __init__(self):
        self.sent: list[str] = []

    async def send_message(self, chat_id, text, **kw):
        self.sent.append(text)


def _patch_playwright_plumbing(monkeypatch):
    monkeypatch.setattr(ma, "_try_import_playwright", lambda: _FakePlaywrightModule())
    monkeypatch.setattr(ma, "_setup_virtual_sink", _async_return(False))
    monkeypatch.setattr(ma, "_teardown_virtual_sink", _async_noop)
    monkeypatch.setattr(ma, "_leave", _async_noop)


async def _async_noop(*a, **kw):
    return None


def _async_return(value):
    async def _factory(*a, **kw):
        return value
    return _factory


async def test_recording_never_starts_when_disclosure_fails(monkeypatch, tmp_path):
    """The core safety property: if the chat announcement can't be
    confirmed sent, _start_recording must never be called."""
    monkeypatch.setattr(ma.settings, "meeting_audio_dir", str(tmp_path), raising=False)
    _patch_playwright_plumbing(monkeypatch)
    monkeypatch.setattr(ma, "_join", _async_return(True))
    monkeypatch.setattr(ma, "_announce", _async_return(False))

    recording_started = {"called": False}

    async def _fake_start_recording(*a, **kw):
        recording_started["called"] = True
        raise AssertionError("audio capture must not start without confirmed disclosure")

    monkeypatch.setattr(ma, "_start_recording", _fake_start_recording)

    session = ma.MeetingSession(
        session_id="test1", chat_id=111, user_id=222,
        meeting_url="https://meet.google.com/abc-defg-hij",
        platform=ma.MeetingPlatform.GOOGLE_MEET,
    )
    bot = _FakeBot()
    await ma._run_session(session, bot)

    assert recording_started["called"] is False
    assert session.disclosed is False
    assert session.status == "failed"
    assert any("e'lon" in m or "recording" in m.lower() for m in bot.sent)


async def test_recording_starts_only_after_disclosure_succeeds(monkeypatch, tmp_path):
    monkeypatch.setattr(ma.settings, "meeting_audio_dir", str(tmp_path), raising=False)
    _patch_playwright_plumbing(monkeypatch)
    monkeypatch.setattr(ma, "_join", _async_return(True))
    monkeypatch.setattr(ma, "_announce", _async_return(True))

    recording_started = {"called": False}

    async def _fake_start_recording(*a, **kw):
        recording_started["called"] = True
        return None

    monkeypatch.setattr(ma, "_start_recording", _fake_start_recording)
    monkeypatch.setattr(ma, "_stop_recording", _async_noop)

    session = ma.MeetingSession(
        session_id="test2", chat_id=112, user_id=222,
        meeting_url="https://meet.google.com/abc-defg-hij",
        platform=ma.MeetingPlatform.GOOGLE_MEET,
    )
    session.stop_requested = True  # skip the recording wait loop
    bot = _FakeBot()
    await ma._run_session(session, bot)

    assert recording_started["called"] is True
    assert session.disclosed is True
    # No audio file was actually produced by the mock, so transcription
    # short-circuits on the empty file rather than failing the session.
    assert session.status in ("done",)


async def test_join_failure_never_reaches_announce(monkeypatch, tmp_path):
    monkeypatch.setattr(ma.settings, "meeting_audio_dir", str(tmp_path), raising=False)
    _patch_playwright_plumbing(monkeypatch)
    monkeypatch.setattr(ma, "_join", _async_return(False))

    announce_called = {"called": False}

    async def _fake_announce(*a, **kw):
        announce_called["called"] = True
        return True

    monkeypatch.setattr(ma, "_announce", _fake_announce)

    session = ma.MeetingSession(
        session_id="test3", chat_id=113, user_id=222,
        meeting_url="https://meet.google.com/abc-defg-hij",
        platform=ma.MeetingPlatform.GOOGLE_MEET,
    )
    bot = _FakeBot()
    await ma._run_session(session, bot)

    assert announce_called["called"] is False
    assert session.status == "failed"
