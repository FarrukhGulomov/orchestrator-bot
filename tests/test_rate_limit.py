"""rate_limit.py — sliding-window burst protection, in-memory backend (no
Redis in this suite, same as everywhere else). See rate_limit.py's module
docstring for why this exists alongside telemetry.py's daily budget."""

import time

import rate_limit


async def test_allowed_under_the_limit(reload_env):
    reload_env({"RATE_LIMIT_MAX_PER_WINDOW": "3", "RATE_LIMIT_WINDOW_SECONDS": "20"})
    for _ in range(3):
        allowed, retry_after = await rate_limit.check(1)
        assert allowed is True
        assert retry_after == 0


async def test_blocked_once_the_limit_is_hit(reload_env):
    reload_env({"RATE_LIMIT_MAX_PER_WINDOW": "3", "RATE_LIMIT_WINDOW_SECONDS": "20"})
    for _ in range(3):
        allowed, _ = await rate_limit.check(1)
        assert allowed is True

    allowed, retry_after = await rate_limit.check(1)
    assert allowed is False
    assert retry_after > 0


async def test_retry_after_is_bounded_by_the_window(reload_env):
    reload_env({"RATE_LIMIT_MAX_PER_WINDOW": "1", "RATE_LIMIT_WINDOW_SECONDS": "20"})
    await rate_limit.check(1)
    _allowed, retry_after = await rate_limit.check(1)
    assert 0 < retry_after <= 20


async def test_limit_is_per_user():
    for _ in range(rate_limit.settings.rate_limit_max_per_window):
        assert (await rate_limit.check(1))[0] is True

    assert (await rate_limit.check(1))[0] is False
    # A different user has their own independent window.
    assert (await rate_limit.check(2))[0] is True


async def test_disabled_always_allows(reload_env):
    reload_env({"RATE_LIMIT_ENABLED": "false", "RATE_LIMIT_MAX_PER_WINDOW": "1"})
    for _ in range(10):
        allowed, retry_after = await rate_limit.check(1)
        assert (allowed, retry_after) == (True, 0)


async def test_falsy_user_id_always_allowed():
    # 0/None can't be a real Telegram user id — never gate on it rather
    # than accidentally sharing one bucket across "no user" callers.
    assert (await rate_limit.check(0))[0] is True
    assert (await rate_limit.check(None))[0] is True  # type: ignore[arg-type]


async def test_window_expiry_allows_again(reload_env, monkeypatch):
    reload_env({"RATE_LIMIT_MAX_PER_WINDOW": "1", "RATE_LIMIT_WINDOW_SECONDS": "5"})

    fake_now = [1_000_000.0]
    monkeypatch.setattr(time, "time", lambda: fake_now[0])

    assert (await rate_limit.check(1))[0] is True
    assert (await rate_limit.check(1))[0] is False

    fake_now[0] += 6  # past the 5s window
    assert (await rate_limit.check(1))[0] is True


async def test_a_broken_in_memory_backend_fails_open(reload_env, monkeypatch):
    reload_env({"RATE_LIMIT_MAX_PER_WINDOW": "1"})

    def _boom(user_id, now):
        raise RuntimeError("in-memory check exploded")

    monkeypatch.setattr(rate_limit, "_check_memory", _boom)
    # A broken limiter must never be why a real message goes unanswered.
    assert (await rate_limit.check(1)) == (True, 0)
