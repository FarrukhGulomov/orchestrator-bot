"""Sanity checks on the agent roster, persona addressing, and the pure
routing/pricing logic that doesn't require a live LLM call.

These exist because the roster is the product's core asset (27 hand-written
personas — see the audit) and is edited by hand often; a duplicate name or
a persona-less agent is an easy, silent mistake with no other safety net.
"""

import agents
import request_types
import router
from config import settings


def test_every_agent_has_a_persona():
    missing = [key for key in agents.AGENTS if key not in agents.AGENT_PERSONAS]
    assert missing == [], f"agents with no (name, emoji) persona: {missing}"


def test_every_persona_points_to_a_real_agent():
    orphaned = [key for key in agents.AGENT_PERSONAS if key not in agents.AGENTS]
    assert orphaned == [], f"personas with no matching agent: {orphaned}"


def test_persona_names_are_unique():
    # A collision would misroute "Name, do X" to the wrong specialist — see
    # agents.py's AGENT_PERSONAS docstring on why names must be distinct.
    names = [name for name, _emoji in agents.AGENT_PERSONAS.values()]
    duplicates = {n for n in names if names.count(n) > 1}
    assert duplicates == set(), f"duplicate persona names: {duplicates}"


def test_agent_key_by_name_is_case_insensitive_and_reversible():
    for key, (name, _emoji) in agents.AGENT_PERSONAS.items():
        assert agents.agent_key_by_name(name) == key
        assert agents.agent_key_by_name(name.upper()) == key
        assert agents.agent_key_by_name(name.lower()) == key


def test_agent_key_by_name_unknown_name_returns_none():
    assert agents.agent_key_by_name("Not A Real Persona Name") is None


def test_get_agent_falls_back_to_default_for_unknown_key():
    assert agents.get_agent("totally-not-a-real-agent-key") is agents.AGENTS[agents.DEFAULT_AGENT_KEY]


def test_get_agent_returns_the_exact_agent_for_a_known_key():
    assert agents.get_agent("personal_assistant").key == "personal_assistant"


def test_agent_label_includes_name_emoji_and_role():
    label = agents.agent_label(agents.AGENTS["ba"])
    name, emoji = agents.persona("ba")
    assert name in label and emoji in label and agents.AGENTS["ba"].display_name in label


def test_personal_assistant_is_not_a_business_role():
    # The whole point of this persona (see the "Endi senga muhim topshiriq"
    # feature) is that it's reachable for non-work questions — assert the
    # router's classifier prompt actually documents it as an option.
    assert "personal_assistant" in router._AGENT_KEYS
    assert "personal_assistant" in router._ROUTER_SYSTEM


def test_request_type_unknown_key_falls_back_to_question():
    rt = request_types.get_request_type("not-a-real-type")
    assert rt.key == request_types.DEFAULT_REQUEST_TYPE_KEY == "question"


def test_question_type_never_creates_a_ticket():
    # See request_types.py's module docstring: without this, casual group
    # chat would flood the GitHub repo with noise.
    assert request_types.REQUEST_TYPES["question"].creates_ticket is False


def test_all_actionable_types_have_a_github_label_when_ticketed():
    for rt in request_types.REQUEST_TYPES.values():
        if rt.creates_ticket:
            assert rt.github_label, f"{rt.key} creates a ticket but has no github_label"


class _FakeAgent:
    key = "ba"


def test_model_for_hybrid_picks_free_model_for_low_and_claude_for_high(monkeypatch):
    monkeypatch.setattr(type(settings), "provider", property(lambda self: "hybrid"))
    agent = _FakeAgent()

    low_model, _label = router.model_for(agent, "low")
    high_model, _label2 = router.model_for(agent, "high")
    assert low_model == settings.or_fast_model
    assert high_model == settings.claude_model


def test_model_for_claude_only_uses_haiku_for_low_sonnet_for_high(monkeypatch):
    monkeypatch.setattr(type(settings), "provider", property(lambda self: "claude"))
    agent = _FakeAgent()

    low_model, _label = router.model_for(agent, "low")
    high_model, _label2 = router.model_for(agent, "high")
    assert low_model == settings.claude_fast_model
    assert high_model == settings.claude_model


def test_model_for_openrouter_only_uses_or_models_for_both_tiers(monkeypatch):
    monkeypatch.setattr(type(settings), "provider", property(lambda self: "openrouter"))
    agent = _FakeAgent()

    low_model, _label = router.model_for(agent, "low")
    high_model, _label2 = router.model_for(agent, "high")
    assert low_model == settings.or_fast_model
    assert high_model == settings.or_main_model
