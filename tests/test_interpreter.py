from datetime import timedelta

import pytest

from conftest import START, final, ts
from drone_voice.clock import ManualClock
from drone_voice.contract import Confidence
from drone_voice.interpreter import Interpreter
from drone_voice.transcription import TranscriptionEvent


@pytest.fixture
def interpreter(vocabulary, clock, settings):
    return Interpreter(vocabulary, clock, settings.ttl_s)


def test_each_configured_expression_gives_its_intent(interpreter, vocabulary):
    for i, entry in enumerate(vocabulary.entries):
        msg = interpreter.process_event(final(f"u-{i}", entry.expression))
        assert msg.status == "recognized"
        assert msg.intent is entry.intent
        assert msg.reason is None
        assert msg.recognized_text == entry.expression


@pytest.mark.parametrize("raw", ["تقدّم", "تقدم", "  تقدّم  ", "تَقَدَّمْ!", "تقـدّم؟", "‏تقدم."])
def test_normalization_variants_match_and_original_is_preserved(interpreter, raw):
    msg = interpreter.process_event(final("u", raw))
    assert msg.intent.value == "MOVE_FORWARD"
    assert msg.recognized_text == raw          # byte-for-byte original
    assert msg.normalized_text == "تقدم"


@pytest.mark.parametrize("raw", ["", "   ", "\t\n", "؟", "ـــ"])
def test_empty_input_is_rejected(interpreter, raw):
    msg = interpreter.process_event(final("u", raw))
    assert (msg.status, msg.intent, msg.reason.value) == ("rejected", None, "EMPTY_INPUT")
    assert msg.recognized_text == raw


@pytest.mark.parametrize("raw", [None, 0, 3.5, True, [], {}, ["تقدّم"], {"text": "تقدّم"}])
def test_wrong_type_is_rejected(interpreter, raw):
    msg = interpreter.process_event(final("u", raw))
    assert (msg.status, msg.intent, msg.reason.value) == ("rejected", None, "INVALID_INPUT_TYPE")
    assert msg.recognized_text is None and msg.normalized_text is None


@pytest.mark.parametrize(
    "raw",
    [
        "ما تتقدّمش",        # negation
        "ما تقدّمش",
        "تقدّم لا",
        "قلتلك تقدّم",        # command inside a longer sentence
        "تقدّم شوية",
        "تقدّم و اطلع",       # two orders
        "تقدّم، اطلع",
        "تقدّم تقدّم",
        "امشي",               # incomplete
        "دور على روحك",       # unknown
        "أطلع",               # spelling variant not in the vocabulary
    ],
)
def test_unknown_negated_compound_or_multiple_is_rejected(interpreter, raw):
    msg = interpreter.process_event(final("u", raw))
    assert (msg.status, msg.intent, msg.reason.value) == ("rejected", None, "UNKNOWN_EXPRESSION")
    assert msg.recognized_text == raw


def test_partial_transcription_produces_nothing(interpreter):
    event = TranscriptionEvent(utterance_id="u", text="تقدّم", is_final=False, source="test")
    assert interpreter.process_event(event) is None


def test_command_id_stable_per_utterance_distinct_across_utterances(interpreter):
    a1 = interpreter.process_event(final("u-1", "تقدّم"))
    a2 = interpreter.process_event(final("u-1", "تقدّم"))
    b = interpreter.process_event(final("u-2", "تقدّم"))
    assert a1.command_id == a2.command_id
    assert b.command_id != a1.command_id


def test_interpret_final_without_utterance_id_is_a_new_utterance_each_time(interpreter):
    first = interpreter.interpret_final("وقف")
    second = interpreter.interpret_final("وقف")
    assert first.command_id != second.command_id
    assert first.intent is second.intent


def test_dates_are_generated_in_utc_with_configured_ttl(vocabulary):
    clock = ManualClock(START + timedelta(microseconds=123456))
    msg = Interpreter(vocabulary, clock, ttl_s=1.5).interpret_final("وقف").to_dict()
    assert msg["created_at"] == "2026-01-01T12:00:00.123Z"
    assert msg["expires_at"] == "2026-01-01T12:00:01.623Z"


def test_default_ttl_comes_from_settings(interpreter):
    msg = interpreter.interpret_final("وقف").to_dict()
    assert (msg["created_at"], msg["expires_at"]) == (ts(0), ts(2))


def test_confidence_is_carried_only_when_provided(interpreter):
    with_conf = interpreter.interpret_final("وقف", confidence=Confidence(-3.2, "model-x-logprob"))
    assert with_conf.to_dict()["confidence"] == {"value": -3.2, "provider": "model-x-logprob"}
    assert interpreter.interpret_final("وقف").to_dict()["confidence"] is None
    # a confidence never changes the decision
    low = interpreter.interpret_final("ما تتقدّمش", confidence=Confidence(0.99, "model-x"))
    assert low.intent is None
