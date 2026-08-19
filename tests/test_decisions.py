"""decisions.py — natural-language decision capture (looks_like_decision
pre-filter + extract_decision LLM step). The add/get/clear storage layer
already has three-tier fallback coverage elsewhere in this codebase's
design; these tests are scoped to the new detection/extraction logic."""

import decisions


def test_looks_like_decision_positive_cases():
    assert decisions.looks_like_decision("biz PostgreSQL ishlatishga qaror qildik") is True
    assert decisions.looks_like_decision("решили запустить релиз в понедельник") is True
    assert decisions.looks_like_decision("we decided to go with option B") is True
    assert decisions.looks_like_decision("договорились встретиться завтра") is True


def test_looks_like_decision_negative_cases():
    assert decisions.looks_like_decision("qanday fikrdasiz, PostgreSQL yaxshimi?") is False
    assert decisions.looks_like_decision("rahmat") is False
    assert decisions.looks_like_decision("") is False
    # A long paragraph isn't a one-line decision statement even if it
    # happens to contain a trigger phrase somewhere in it.
    long_text = "we decided to go with option B. " + "x" * 500
    assert decisions.looks_like_decision(long_text) is False


async def test_extract_decision_returns_none_for_non_decision(monkeypatch):
    async def _fake_llm(*a, **kw):
        return "NONE"

    monkeypatch.setattr(decisions, "claude_generate_fast", _fake_llm)
    assert await decisions.extract_decision("is postgres a good choice?") is None


async def test_extract_decision_returns_cleaned_entry(monkeypatch):
    async def _fake_llm(*a, **kw):
        return "PostgreSQL ishlatishga qaror qilindi"

    monkeypatch.setattr(decisions, "claude_generate_fast", _fake_llm)
    result = await decisions.extract_decision("biz PostgreSQL ishlatishga qaror qildik")
    assert result == "PostgreSQL ishlatishga qaror qilindi"


async def test_extract_decision_never_raises_on_llm_failure(monkeypatch):
    async def _fake_llm(*a, **kw):
        raise RuntimeError("provider down")

    monkeypatch.setattr(decisions, "claude_generate_fast", _fake_llm)
    assert await decisions.extract_decision("biz qaror qildik") is None
