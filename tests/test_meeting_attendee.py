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


def test_normalize_url_adds_missing_scheme():
    """Regression: people paste links the way chat apps render them, with
    no scheme. Playwright's page.goto() rejects those outright, which
    surfaced as a confusing "UI elements not found" failure with a blank
    about:blank screenshot — the browser never navigated at all."""
    assert ma.normalize_url("meet.google.com/fov-kayt-yno") == "https://meet.google.com/fov-kayt-yno"
    assert ma.normalize_url("  meet.google.com/abc  ") == "https://meet.google.com/abc"
    # Already-schemed URLs pass through untouched.
    assert ma.normalize_url("https://meet.google.com/abc") == "https://meet.google.com/abc"
    assert ma.normalize_url("http://zoom.us/j/1") == "http://zoom.us/j/1"


async def test_start_stores_normalized_url(monkeypatch):
    """The schemeless URL was always recognised as a Meet link
    (detect_platform is scheme-agnostic) — the bug was that it reached
    page.goto() unnormalised. Assert start() fixes it up before the
    session (and therefore the browser) ever sees it."""
    monkeypatch.setattr(ma.settings, "meeting_bot_enabled", True, raising=False)
    monkeypatch.setattr(ma, "_try_import_playwright", lambda: object())
    monkeypatch.setattr(ma.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(ma, "_run_session", _async_noop)

    session = await ma.start(999, 222, "meet.google.com/fov-kayt-yno", bot=None)
    try:
        assert session.meeting_url == "https://meet.google.com/fov-kayt-yno"
        assert session.platform == ma.MeetingPlatform.GOOGLE_MEET
    finally:
        ma._sessions.pop(999, None)


def test_storage_state_parsing(monkeypatch):
    """A malformed saved session must degrade to an anonymous browser
    (still fine for Zoom/Teams) rather than killing every session."""
    monkeypatch.setattr(ma.settings, "meeting_storage_state_json", "", raising=False)
    assert ma._load_storage_state() is None
    assert ma.storage_state_configured() is False

    monkeypatch.setattr(ma.settings, "meeting_storage_state_json", "not json at all", raising=False)
    assert ma._load_storage_state() is None

    # Valid JSON but not a usable session shape.
    monkeypatch.setattr(ma.settings, "meeting_storage_state_json", '{"foo": 1}', raising=False)
    assert ma._load_storage_state() is None

    good = '{"cookies": [{"name": "SID", "value": "x"}], "origins": []}'
    monkeypatch.setattr(ma.settings, "meeting_storage_state_json", good, raising=False)
    assert ma._load_storage_state() == {"cookies": [{"name": "SID", "value": "x"}], "origins": []}
    assert ma.storage_state_configured() is True


class _BodyPage:
    def __init__(self, body: str, title: str = "Sign in"):
        self._body = body
        self._title = title

    async def inner_text(self, selector):
        return self._body

    async def title(self):
        return self._title


async def test_meet_block_reason_explains_anonymous_refusal(monkeypatch):
    """Meet's anonymous wall has no join button, so it's indistinguishable
    from a broken selector unless the page copy is actually read."""
    monkeypatch.setattr(ma.settings, "meeting_storage_state_json", "", raising=False)
    page = _BodyPage("You can't join this video call\nYour meeting is safe")
    reason = await ma._meet_block_reason(page)
    assert reason is not None
    assert "MEETING_STORAGE_STATE_JSON" in reason


async def test_meet_block_reason_differs_when_session_configured(monkeypatch):
    monkeypatch.setattr(
        ma.settings, "meeting_storage_state_json",
        '{"cookies": [{"name": "SID", "value": "x"}]}', raising=False,
    )
    page = _BodyPage("You can't join this video call")
    reason = await ma._meet_block_reason(page)
    assert reason is not None
    assert "eskirgan" in reason  # "expired session" guidance, not "set up a session"


async def test_google_login_failure_reasons_are_actionable(monkeypatch):
    """The predicted failure modes must each produce their own guidance —
    "login failed" alone would leave the user with no next step."""
    blocked = _BodyPage("Couldn't sign you in\nThis browser or app may not be secure")
    msg = await ma._google_login_failure_reason(blocked, stage="password")
    assert "MEETING_STORAGE_STATE_JSON" in msg

    two_fa = _BodyPage("2-Step Verification\nVerify it's you")
    assert "2FA" in await ma._google_login_failure_reason(two_fa, stage="password")

    bad_pw = _BodyPage("Wrong password. Try again")
    assert "MEETING_GOOGLE_PASSWORD" in await ma._google_login_failure_reason(bad_pw, stage="password")

    no_acct = _BodyPage("Couldn't find your Google Account")
    assert "MEETING_GOOGLE_EMAIL" in await ma._google_login_failure_reason(no_acct, stage="email")


async def test_unrecognised_login_page_reports_what_was_on_screen():
    """The catch-all branch is the one that fires when Google serves a
    sign-in front-end we don't recognise — it has to say what the page
    WAS, or there's nothing to act on but a bare selector timeout."""
    odd = _BodyPage("Choose an account to continue", title="Google Accounts")
    msg = await ma._google_login_failure_reason(odd, stage="email")
    assert "Google Accounts" in msg
    assert "Choose an account" in msg


async def test_fill_first_selector_reports_failure_not_raises():
    """A missing field must return False (so the caller can screenshot and
    explain) rather than bubbling a raw Playwright timeout to the user."""

    class _NoFieldPage:
        async def wait_for_selector(self, selector, **kw):
            raise TimeoutError("no such element")

    assert await ma._fill_first_selector(_NoFieldPage(), ('input[type="email"]',), "x") is False


async def test_google_login_skipped_when_not_configured(monkeypatch):
    monkeypatch.setattr(ma.settings, "meeting_google_email", "", raising=False)
    monkeypatch.setattr(ma.settings, "meeting_google_password", "", raising=False)
    state, err = await ma._ensure_google_login(browser=None)
    assert state is None and err is None


async def test_google_login_failure_is_remembered(monkeypatch, tmp_path):
    """A hard login failure must not be retried on every meeting — repeat
    attempts are exactly what gets a Google account locked."""
    monkeypatch.setattr(ma.settings, "meeting_google_email", "bot@example.com", raising=False)
    monkeypatch.setattr(ma.settings, "meeting_google_password", "hunter2", raising=False)
    monkeypatch.setattr(ma, "_GOOGLE_STATE_CACHE", tmp_path / "state.json")
    monkeypatch.setattr(ma, "_google_login_failed_reason", "2FA yoqilgan")

    attempts = {"n": 0}

    class _ExplodingBrowser:
        async def new_context(self, **kw):
            attempts["n"] += 1
            raise AssertionError("must not attempt login again after a hard failure")

    state, err = await ma._ensure_google_login(_ExplodingBrowser())
    assert state is None
    assert err == "2FA yoqilgan"
    assert attempts["n"] == 0


async def test_cached_google_session_is_reused(monkeypatch, tmp_path):
    cache = tmp_path / "state.json"
    cache.write_text('{"cookies": [{"name": "SID", "value": "cached"}]}')
    monkeypatch.setattr(ma.settings, "meeting_google_email", "bot@example.com", raising=False)
    monkeypatch.setattr(ma.settings, "meeting_google_password", "hunter2", raising=False)
    monkeypatch.setattr(ma, "_GOOGLE_STATE_CACHE", cache)
    monkeypatch.setattr(ma, "_google_login_failed_reason", None)

    class _ExplodingBrowser:
        async def new_context(self, **kw):
            raise AssertionError("must reuse the cached session instead of re-logging in")

    state, err = await ma._ensure_google_login(_ExplodingBrowser())
    assert err is None
    assert state == {"cookies": [{"name": "SID", "value": "cached"}]}


async def test_meet_block_reason_none_for_ordinary_page(monkeypatch):
    """A normal pre-join page must NOT be reported as a block — otherwise a
    genuine selector break gets misattributed to Google's policy."""
    page = _BodyPage("Ready to join? Nobody else is here")
    assert await ma._meet_block_reason(page) is None


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

    async def add_init_script(self, script):
        pass

    async def close(self):
        pass


class _FakePage:
    def is_closed(self):
        return False

    async def screenshot(self, **kw):
        return b"fake-png-bytes"


class _FakeChromium:
    def __init__(self):
        self.last_launch_kwargs: dict | None = None

    async def launch(self, **kw):
        self.last_launch_kwargs = kw
        return _FakeBrowser()


class _FakeAsyncPlaywright:
    def __init__(self, chromium):
        self.chromium = chromium

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakePlaywrightModule:
    def __init__(self):
        # One instance shared across async_playwright() calls so a test can
        # hold a reference and inspect what launch() was actually called with.
        self.chromium = _FakeChromium()

    def async_playwright(self):
        return _FakeAsyncPlaywright(self.chromium)


class _FakeBot:
    def __init__(self):
        self.sent: list[str] = []
        self.photos_sent = 0

    async def send_message(self, chat_id, text, **kw):
        self.sent.append(text)

    async def send_photo(self, chat_id, photo, **kw):
        self.photos_sent += 1


def _patch_playwright_plumbing(monkeypatch):
    module = _FakePlaywrightModule()
    monkeypatch.setattr(ma, "_try_import_playwright", lambda: module)
    # Real Chromium download needs network + takes ~a minute — not something
    # a unit test should depend on; the dedicated test below covers the
    # "install failed" branch on its own with an explicit mock.
    monkeypatch.setattr(ma, "_ensure_chromium", _async_return((True, None)))
    monkeypatch.setattr(ma, "_setup_virtual_sink", _async_return(False))
    monkeypatch.setattr(ma, "_teardown_virtual_sink", _async_noop)
    monkeypatch.setattr(ma, "_leave", _async_noop)
    return module


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
    monkeypatch.setattr(ma, "_join", _async_return((True, None)))
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
    monkeypatch.setattr(ma, "_join", _async_return((True, None)))
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


async def test_launch_env_merges_with_not_replaces_os_environ(monkeypatch, tmp_path):
    """Regression test: Playwright's launch(env=...) REPLACES the child
    process's whole environment rather than merging with it. Passing just
    {"PULSE_SINK": ..., "DISPLAY": ...} would silently strip PATH/HOME/etc.
    and break the browser launch outright — see _run_session's comment at
    the p.chromium.launch(...) call."""
    monkeypatch.setattr(ma.settings, "meeting_audio_dir", str(tmp_path), raising=False)
    monkeypatch.setenv("SOME_PREEXISTING_VAR", "should-survive")
    module = _patch_playwright_plumbing(monkeypatch)
    monkeypatch.setattr(ma, "_setup_virtual_sink", _async_return(True))  # force PULSE_SINK to be set
    monkeypatch.setattr(ma, "_ensure_display", _async_return(":99"))  # force DISPLAY to be set
    monkeypatch.setattr(ma, "_join", _async_return((False, "test")))  # fail fast, we only care about launch()

    session = ma.MeetingSession(
        session_id="test5", chat_id=115, user_id=222,
        meeting_url="https://meet.google.com/abc-defg-hij",
        platform=ma.MeetingPlatform.GOOGLE_MEET,
    )
    bot = _FakeBot()
    await ma._run_session(session, bot)

    launch_env = module.chromium.last_launch_kwargs["env"]
    assert launch_env["PULSE_SINK"] == "meetbot_test5"
    assert launch_env["DISPLAY"] == ":99"
    assert launch_env.get("SOME_PREEXISTING_VAR") == "should-survive"
    assert launch_env.get("PATH")  # the whole point: PATH must not have been wiped


async def test_chromium_install_failure_never_reaches_join(monkeypatch, tmp_path):
    monkeypatch.setattr(ma.settings, "meeting_audio_dir", str(tmp_path), raising=False)
    monkeypatch.setattr(ma, "_try_import_playwright", lambda: _FakePlaywrightModule())
    monkeypatch.setattr(ma, "_ensure_chromium", _async_return((False, "network unreachable")))

    join_called = {"called": False}

    async def _fake_join(*a, **kw):
        join_called["called"] = True
        return True

    monkeypatch.setattr(ma, "_join", _fake_join)

    session = ma.MeetingSession(
        session_id="test4", chat_id=114, user_id=222,
        meeting_url="https://meet.google.com/abc-defg-hij",
        platform=ma.MeetingPlatform.GOOGLE_MEET,
    )
    bot = _FakeBot()
    await ma._run_session(session, bot)

    assert join_called["called"] is False
    assert session.status == "failed"
    assert any("Chromium" in m for m in bot.sent)


async def test_join_failure_never_reaches_announce(monkeypatch, tmp_path):
    monkeypatch.setattr(ma.settings, "meeting_audio_dir", str(tmp_path), raising=False)
    _patch_playwright_plumbing(monkeypatch)
    monkeypatch.setattr(ma, "_join", _async_return((False, "test")))

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
    assert bot.photos_sent == 1  # debug screenshot sent so the failure is diagnosable
