import json

import pytest

from drone_voice.intents import Intent
from drone_voice.vocabulary import ConfigError, Vocabulary, load_vocabulary, vocabulary_from_dict

BASE = [
    ("تقدّم", "MOVE_FORWARD"),
    ("ارجع", "MOVE_BACKWARD"),
    ("امشي لليمين", "MOVE_RIGHT"),
    ("امشي لليسار", "MOVE_LEFT"),
    ("اطلع", "MOVE_UP"),
    ("اهبط", "MOVE_DOWN"),
    ("وقف", "STOP"),
]


def test_default_config_covers_every_intent(vocabulary):
    assert {e.intent for e in vocabulary.entries} == set(Intent)


@pytest.mark.parametrize("variant", ["وَقِّف", " وقف . ", "وقـف"])
def test_conflict_after_normalization_is_refused(variant):
    with pytest.raises(ConfigError, match="ambiguous"):
        Vocabulary(BASE + [(variant, "MOVE_UP")])


def test_redundant_variant_with_same_intent_is_allowed():
    vocab = Vocabulary(BASE + [("وَقِّف", "STOP")])
    assert vocab.lookup("وقف") is Intent.STOP


def test_missing_intent_is_refused():
    with pytest.raises(ConfigError, match="STOP"):
        Vocabulary(BASE[:-1])


@pytest.mark.parametrize("expression", ["", "   ", "؟", "ـ"])
def test_expression_empty_after_normalization_is_refused(expression):
    with pytest.raises(ConfigError, match="empty"):
        Vocabulary(BASE + [(expression, "STOP")])


def test_unknown_intent_is_refused():
    with pytest.raises(ConfigError, match="unknown intent"):
        Vocabulary(BASE + [("دور", "ROTATE_LEFT")])


def test_lookup_is_exact_never_substring():
    vocab = Vocabulary(BASE)
    assert vocab.lookup("تقدم") is Intent.MOVE_FORWARD
    for text in ["تقدم و اطلع", "قلتلك تقدم", "ما تتقدمش", "تقدم تقدم", "امشي", "لليمين"]:
        assert vocab.lookup(text) is None


@pytest.mark.parametrize(
    "data",
    [
        [],
        {"format_version": 2, "entries": [{"expression": "وقف", "intent": "STOP"}]},
        {"format_version": True, "entries": [{"expression": "وقف", "intent": "STOP"}]},
        {"format_version": 1, "entries": []},
        {"format_version": 1, "entries": [{"expression": "وقف"}]},
        {"format_version": 1, "entries": [{"expression": "وقف", "intent": "STOP", "regex": ".*"}]},
        {"format_version": 1, "entries": [], "fuzzy": True},
    ],
)
def test_invalid_file_structure_is_refused(data):
    with pytest.raises(ConfigError):
        vocabulary_from_dict(data)


def test_conflicting_file_is_refused(tmp_path):
    path = tmp_path / "vocab.json"
    entries = [{"expression": e, "intent": i} for e, i in BASE] + [{"expression": "تَقَدَّمْ", "intent": "STOP"}]
    path.write_text(json.dumps({"format_version": 1, "entries": entries}, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ConfigError, match="ambiguous"):
        load_vocabulary(path)
