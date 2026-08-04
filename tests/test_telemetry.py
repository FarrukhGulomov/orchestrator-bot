"""telemetry.py: call recording, budget enforcement, spend/health reports.

All against the in-memory tier (no DATABASE_URL in this suite) — see
db.py's module docstring for why that tier exists and telemetry.py's
docstring for why it's the fallback here too.
"""

import pytest

import telemetry


class _Usage:
    def __init__(self, prompt=0, completion=0, total=None):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        if total is not None:
            self.total_tokens = total


class _AnthropicUsage:
    def __init__(self, inp, out):
        self.input_tokens = inp
        self.output_tokens = out


class _Response:
    def __init__(self, usage):
        self.usage = usage


def test_extract_usage_openai_shape():
    assert telemetry.extract_usage(_Response(_Usage(prompt=900, completion=210))) == (900, 210)


def test_extract_usage_anthropic_shape():
    assert telemetry.extract_usage(_Response(_AnthropicUsage(1200, 350))) == (1200, 350)


def test_extract_usage_total_only_attributes_to_output():
    # Some free OpenRouter models report only a total — attribute it to the
    # pricier side (output) so the estimate errs high, not low.
    assert telemetry.extract_usage(_Response(_Usage(total=700))) == (0, 700)


def test_extract_usage_missing_usage_is_zero():
    assert telemetry.extract_usage(object()) == (0, 0)


async def test_record_is_attributed_via_call_context():
    telemetry.set_call_context(user_id=42, chat_id=42, agent_key="ba")
    await telemetry.record("openai", "gpt-4.1", input_tokens=1200, output_tokens=350, latency_ms=1800)

    report = await telemetry.spend_report(days=1)
    assert report["calls"] == 1
    assert report["by_user"] == [("42", 1, report["cost"], 1550)]
    assert report["by_agent"][0][0] == "ba"


async def test_record_disabled_is_a_silent_noop(reload_env):
    _cfg, _llm, tel = reload_env({"TELEMETRY_ENABLED": "false"})
    tel.set_call_context(user_id=1, chat_id=1)
    await tel.record("openai", "gpt-4.1", input_tokens=100, output_tokens=50)
    report = await tel.spend_report(days=1)
    assert report["calls"] == 0


async def test_record_never_raises_on_bad_input():
    # cost estimation, dict construction etc. must not be able to break the
    # reply path — see the module docstring's "never raise" rule.
    await telemetry.record(provider="x", model=None, input_tokens="not-a-number", output_tokens=1)  # type: ignore[arg-type]


async def test_failed_call_is_still_recorded():
    await telemetry.record("openai", "gpt-4.1", ok=False, error="401 invalid_api_key", fallback_position=0)
    await telemetry.record("deepseek", "deepseek-chat", ok=True, input_tokens=500, output_tokens=100, fallback_position=1)

    report = await telemetry.spend_report(days=1)
    assert report["calls"] == 2
    assert report["failures"] == 1
    assert report["fallbacks"] == 1  # the deepseek call had fallback_position > 0


async def test_over_budget_false_when_no_cap_set():
    telemetry.set_call_context(user_id=7)
    await telemetry.record("openai", "gpt-4.1", input_tokens=999_999, output_tokens=999_999)
    over, used, limit = await telemetry.over_budget(7)
    assert (over, used, limit) == (False, 0, 0)


async def test_over_budget_trips_at_the_limit(reload_env):
    _cfg, _llm, tel = reload_env({"DAILY_USER_TOKEN_BUDGET": "1000"})
    tel.set_call_context(user_id=7, chat_id=7)

    await tel.record("openai", "gpt-4.1", input_tokens=600, output_tokens=100)
    over, used, limit = await tel.over_budget(7)
    assert (over, used, limit) == (False, 700, 1000)

    await tel.record("openai", "gpt-4.1", input_tokens=400, output_tokens=100)
    over, used, limit = await tel.over_budget(7)
    assert over is True
    assert used == 1200


async def test_over_budget_is_per_user(reload_env):
    _cfg, _llm, tel = reload_env({"DAILY_USER_TOKEN_BUDGET": "500"})
    tel.set_call_context(user_id=1)
    await tel.record("openai", "gpt-4.1", input_tokens=1000, output_tokens=0)

    over_user1, _, _ = await tel.over_budget(1)
    over_user2, used2, _ = await tel.over_budget(2)
    assert over_user1 is True
    assert (over_user2, used2) == (False, 0)


async def test_over_budget_fails_open_when_usage_query_breaks(reload_env, monkeypatch):
    _cfg, _llm, tel = reload_env({"DAILY_USER_TOKEN_BUDGET": "500"})

    async def _boom(user_id):
        raise RuntimeError("db exploded")

    monkeypatch.setattr(tel, "tokens_used_today", _boom)
    over, used, limit = await tel.over_budget(1)
    # Refusing to answer because bookkeeping broke would be worse than the
    # cap not applying for one request — see over_budget's docstring.
    assert over is False


async def test_provider_health_success_rate_and_median_latency():
    await telemetry.record("claude", "claude-sonnet-4-6", ok=True, latency_ms=100)
    await telemetry.record("claude", "claude-sonnet-4-6", ok=True, latency_ms=200)
    await telemetry.record("claude", "claude-sonnet-4-6", ok=False, latency_ms=50, error="529")

    health = await telemetry.provider_health(hours=24)
    claude_row = next(r for r in health if r[0] == "claude")
    _provider, calls, success_rate, median_latency = claude_row
    assert calls == 3
    assert round(success_rate, 2) == round(2 / 3 * 100, 2)
    assert median_latency == 100  # sorted [50, 100, 200] -> middle


async def test_spend_report_window_excludes_old_rows(monkeypatch):
    import datetime as dt

    telemetry.set_call_context(user_id=1)
    await telemetry.record("openai", "gpt-4.1", input_tokens=100, output_tokens=50)
    # Backdate the row we just wrote past the report window.
    telemetry._mem[-1]["created_at"] = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30)

    report = await telemetry.spend_report(days=7)
    assert report["calls"] == 0
