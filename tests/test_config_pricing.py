"""config.estimate_cost_usd and the budget switch — see config.py's
MODEL_PRICES comment for why unknown models price at 0.0 rather than a guess."""

import config


def test_known_model_priced_correctly():
    # claude-sonnet-4-6: $3/$15 per 1M tokens (input, output).
    cost = config.estimate_cost_usd("claude-sonnet-4-6", 1_000_000, 1_000_000)
    assert cost == 18.0


def test_typical_reply_is_a_few_cents():
    cost = config.estimate_cost_usd("claude-sonnet-4-6", 3000, 800)
    assert 0.0 < cost < 0.05


def test_unknown_model_costs_zero_not_a_guess():
    assert config.estimate_cost_usd("some-brand-new-model-nobody-priced-yet", 5000, 5000) == 0.0


def test_free_router_model_costs_zero():
    assert config.estimate_cost_usd("openrouter/free", 999_999, 999_999) == 0.0
    assert config.estimate_cost_usd("meta-llama/llama-4-maverick:free", 999_999, 999_999) == 0.0


def test_empty_model_costs_zero():
    assert config.estimate_cost_usd("", 1000, 1000) == 0.0


def test_price_table_override_via_env(reload_env):
    cfg, _llm, _tel = reload_env({"MODEL_PRICES_JSON": '{"my-model": [1.0, 2.0]}'})
    assert cfg.estimate_cost_usd("my-model", 1_000_000, 1_000_000) == 3.0
    # Original table entries survive an override — it's a merge, not a replace.
    assert cfg.estimate_cost_usd("claude-sonnet-4-6", 1_000_000, 1_000_000) == 18.0


def test_malformed_price_override_falls_back_to_defaults(reload_env):
    cfg, _llm, _tel = reload_env({"MODEL_PRICES_JSON": "not valid json"})
    assert cfg.estimate_cost_usd("claude-sonnet-4-6", 1_000_000, 1_000_000) == 18.0


def test_budget_disabled_by_default(reload_env):
    cfg, _llm, _tel = reload_env({})
    assert cfg.settings.daily_user_token_budget == 0
    assert cfg.settings.budget_enforced is False


def test_budget_enforced_only_when_positive_and_telemetry_on(reload_env):
    cfg, _llm, _tel = reload_env({"DAILY_USER_TOKEN_BUDGET": "5000"})
    assert cfg.settings.budget_enforced is True

    cfg, _llm, _tel = reload_env({
        "DAILY_USER_TOKEN_BUDGET": "5000", "TELEMETRY_ENABLED": "false",
    })
    assert cfg.settings.budget_enforced is False, "a disabled telemetry pipe can't enforce a budget it can't measure"


def test_any_ai_key_set_covers_all_seven_providers(reload_env):
    for env_var in (
        "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY",
        "GEMINI_API_KEY", "XAI_API_KEY", "DEEPSEEK_API_KEY", "KIMI_API_KEY",
    ):
        cfg, _llm, _tel = reload_env({env_var: "test-key"})
        assert cfg.settings.any_ai_key_set is True, f"{env_var} alone should count as configured"


def test_no_keys_means_not_configured(reload_env):
    cfg, _llm, _tel = reload_env({})
    assert cfg.settings.any_ai_key_set is False
    assert cfg.settings.validate() != []
