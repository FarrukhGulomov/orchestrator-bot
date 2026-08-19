"""task_assistant.looks_like_task — the cheap pre-filter that decides
whether a plain message (no /addtask) even reaches the LLM classifier.
False negatives here are the actual bug class: a real reminder request
that never gets offered to the user at all."""

import task_assistant as ta


def test_reported_bug_now_detected():
    """The exact message that slipped through: colloquial "вспомнить"
    used to mean "remind me", not literally "recall" — see
    task_assistant._TRIGGER_WORDS' comment on why it's listed."""
    assert ta.looks_like_task("я хочу чтобы ти завтра мне вспомнить эту встрече") is True


def test_standard_trigger_words_across_scripts():
    assert ta.looks_like_task("ertaga eslatib qo'y") is True
    assert ta.looks_like_task("эртага эслатиб қўй") is True
    assert ta.looks_like_task("напомни мне завтра") is True
    assert ta.looks_like_task("не забудь про встречу") is True
    assert ta.looks_like_task("remind me tomorrow") is True
    assert ta.looks_like_task("don't forget the report") is True


def test_obligation_plus_time_signal():
    assert ta.looks_like_task("ertaga soat 15:00 da hisobot topshirishim kerak") is True
    # Obligation word alone, no time/date signal, isn't enough — too common
    # in ordinary messages ("yordam kerak" = "need help").
    assert ta.looks_like_task("yordam kerak") is False


def test_ordinary_messages_not_flagged():
    assert ta.looks_like_task("qanday kunlar edi bugun?") is False
    assert ta.looks_like_task("rahmat, tushunarli") is False
    assert ta.looks_like_task("") is False
