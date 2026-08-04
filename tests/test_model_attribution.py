"""last_used_provider() / provider_label() — the fix for last session's bug
where a fallback answer (e.g. via DeepSeek after Claude failed) still
showed a Claude signature in the Telegram reply. See
llm_clients._attempt's docstring and bot.py's _answer_with_agent."""

import telemetry


async def test_last_used_provider_reflects_the_fallback_that_actually_answered(reload_env):
    _cfg, llm, _tel = reload_env({
        "ANTHROPIC_API_KEY": "k0", "DEEPSEEK_API_KEY": "k3",
        "PROVIDER_PRIORITY": "claude,deepseek",
    })

    def fake(key, model, system, messages, temperature, max_tokens, usage_out=None):
        if key == "claude":
            raise RuntimeError("claude 529 overloaded")
        return "answer from deepseek"

    llm._call_main_sync = fake
    result = await llm.claude_generate("sys", [{"role": "user", "content": "hi"}], model="claude-sonnet-4-6")

    assert result == "answer from deepseek"
    used_key, used_model = llm.last_used_provider()
    assert used_key == "deepseek", "must report the provider that actually answered, not the preferred one"
    assert used_model == _cfg.settings.deepseek_model
    assert llm.provider_label(used_key) == _cfg.settings.deepseek_model_label


async def test_last_used_provider_reflects_preferred_provider_when_it_succeeds(reload_env):
    _cfg, llm, _tel = reload_env({"ANTHROPIC_API_KEY": "k0"})

    def fake(key, model, system, messages, temperature, max_tokens, usage_out=None):
        return "ok"

    llm._call_main_sync = fake
    await llm.claude_generate("sys", [{"role": "user", "content": "hi"}], model="claude-sonnet-4-6")

    used_key, _used_model = llm.last_used_provider()
    assert used_key == "claude"


def test_provider_label_unknown_key_falls_back_to_the_key_itself(reload_env):
    _cfg, llm, _tel = reload_env({})
    assert llm.provider_label("some-future-provider") == "some-future-provider"
