"""llm_clients._generate_with_failover / _generate_json_with_failover — the
core promise from this session's multi-provider work: a dead key on one
provider falls through to the next configured one instead of taking the
bot down. See llm_clients.py's module docstring.
"""

import pytest


async def test_first_configured_provider_wins_when_healthy(reload_env):
    _cfg, llm, _tel = reload_env({"OPENAI_API_KEY": "k1", "PROVIDER_PRIORITY": "openai,deepseek"})

    def fake(key, model, system, messages, temperature, max_tokens, usage_out=None):
        return f"answer-from-{key}"

    llm._call_main_sync = fake
    result = await llm.claude_generate("sys", [{"role": "user", "content": "hi"}])
    assert result == "answer-from-openai"


async def test_falls_through_to_next_configured_provider_on_failure(reload_env):
    _cfg, llm, _tel = reload_env({
        "OPENAI_API_KEY": "k1", "GEMINI_API_KEY": "k2", "DEEPSEEK_API_KEY": "k3",
        "PROVIDER_PRIORITY": "openai,gemini,deepseek",
    })
    tried = []

    def fake(key, model, system, messages, temperature, max_tokens, usage_out=None):
        tried.append(key)
        if key in ("openai", "gemini"):
            raise RuntimeError(f"{key} 401 invalid api key")
        return f"answer-from-{key}"

    llm._call_main_sync = fake
    result = await llm.claude_generate("sys", [{"role": "user", "content": "hi"}])
    assert result == "answer-from-deepseek"
    assert tried == ["openai", "gemini", "deepseek"], "must try providers in priority order, stopping at the first success"


async def test_unconfigured_providers_are_skipped_entirely(reload_env):
    # Only deepseek has a key — openai/gemini/grok/kimi/openrouter/claude
    # must never be attempted even though they're known providers.
    _cfg, llm, _tel = reload_env({"DEEPSEEK_API_KEY": "k3"})
    tried = []

    def fake(key, model, system, messages, temperature, max_tokens, usage_out=None):
        tried.append(key)
        return "ok"

    llm._call_main_sync = fake
    await llm.claude_generate("sys", [{"role": "user", "content": "hi"}])
    assert tried == ["deepseek"]


async def test_all_providers_failing_raises_the_last_error(reload_env):
    _cfg, llm, _tel = reload_env({"OPENAI_API_KEY": "k1", "DEEPSEEK_API_KEY": "k3"})

    def fake(key, model, system, messages, temperature, max_tokens, usage_out=None):
        raise RuntimeError(f"{key} is down")

    llm._call_main_sync = fake
    with pytest.raises(RuntimeError, match="deepseek is down"):
        await llm.claude_generate("sys", [{"role": "user", "content": "hi"}])


async def test_no_provider_configured_raises_immediately(reload_env):
    _cfg, llm, _tel = reload_env({})
    with pytest.raises(RuntimeError, match="Hech qanday AI provider"):
        await llm.claude_generate("sys", [{"role": "user", "content": "hi"}])


async def test_provider_priority_env_var_reorders_the_chain(reload_env):
    _cfg, llm, _tel = reload_env({
        "OPENAI_API_KEY": "k1", "DEEPSEEK_API_KEY": "k3",
        "PROVIDER_PRIORITY": "deepseek,openai",
    })
    tried = []

    def fake(key, model, system, messages, temperature, max_tokens, usage_out=None):
        tried.append(key)
        return f"answer-from-{key}"

    llm._call_main_sync = fake
    result = await llm.claude_generate("sys", [{"role": "user", "content": "hi"}])
    assert tried[0] == "deepseek"
    assert result == "answer-from-deepseek"


async def test_configured_provider_missing_from_priority_still_gets_used(reload_env):
    # A key added without touching PROVIDER_PRIORITY must not be silently
    # ignored — see _configured_chain's docstring.
    _cfg, llm, _tel = reload_env({
        "DEEPSEEK_API_KEY": "k3", "PROVIDER_PRIORITY": "openai",  # openai has no key
    })
    assert llm._configured_chain() == ["deepseek"]


async def test_json_failover_uses_json_dispatcher_and_fast_tier(reload_env):
    _cfg, llm, _tel = reload_env({"OPENAI_API_KEY": "k1", "DEEPSEEK_API_KEY": "k3"})
    seen_models = []

    def fake_json(key, model, system, messages, max_tokens, usage_out=None):
        seen_models.append(model)
        if key == "openai":
            raise RuntimeError("boom")
        return '{"ok": true}'

    llm._call_json_sync = fake_json
    result = await llm.claude_generate_json("sys", [{"role": "user", "content": "hi"}])
    assert result == '{"ok": true}'
    # No explicit model passed -> fast tier -> each provider's fast_model.
    assert seen_models == [_cfg.settings.openai_model, _cfg.settings.deepseek_model]


async def test_preferred_model_tried_first_then_falls_back(reload_env):
    _cfg, llm, _tel = reload_env({"ANTHROPIC_API_KEY": "k0", "DEEPSEEK_API_KEY": "k3"})
    tried = []

    def fake(key, model, system, messages, temperature, max_tokens, usage_out=None):
        tried.append(key)
        if key == "claude":
            raise RuntimeError("claude overloaded")
        return "ok-from-fallback"

    llm._call_main_sync = fake
    # An explicit Claude-shaped model id (no "/") should be tried via the
    # claude provider first, even though PROVIDER_PRIORITY defaults openai-first.
    result = await llm.claude_generate(
        "sys", [{"role": "user", "content": "hi"}], model="claude-sonnet-4-6",
    )
    assert tried[0] == "claude"
    assert result == "ok-from-fallback"


def test_configured_chain_filters_to_only_providers_with_keys(reload_env):
    _cfg, llm, _tel = reload_env({
        "OPENAI_API_KEY": "k1", "DEEPSEEK_API_KEY": "k3",
        "PROVIDER_PRIORITY": "openai,claude,gemini,grok,deepseek,kimi,openrouter",
    })
    assert llm._configured_chain() == ["openai", "deepseek"]
