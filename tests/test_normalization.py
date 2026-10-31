import pytest

from drone_voice.normalization import normalize_text


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("تقدّم", "تقدم"),                    # shadda
        ("تَقَدَّمْ", "تقدم"),                 # fatha, shadda, sukun
        ("تقـــدّم", "تقدم"),                 # tatweel
        ("  امشي   لليمين  ", "امشي لليمين"),  # spaces
        ("امشي\tلليمين\n", "امشي لليمين"),     # tab / newline
        ("وقف!", "وقف"),
        ("وقف ؟", "وقف"),
        ("وقف...", "وقف"),
        ("وقف ! ؟", "وقف"),
        ("‏وقف‏", "وقف"),           # RLM marks
        ("﻿وقف", "وقف"),                 # BOM
    ],
)
def test_allowed_variants(raw, expected):
    assert normalize_text(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("تقدم، ارجع", "تقدم، ارجع"),  # internal punctuation kept
        ("وقف! تقدم", "وقف! تقدم"),    # internal "!" kept, nothing after it dropped
        ("!وقف", "!وقف"),              # leading punctuation kept
        ("أطلع", "أطلع"),              # hamza forms are not unified
        ("وقف،", "وقف،"),              # Arabic comma is not sentence-final punctuation
    ],
)
def test_prudent_rules_keep_content(raw, expected):
    assert normalize_text(raw) == expected


def test_letters_not_unified():
    assert normalize_text("أطلع") != normalize_text("اطلع")


@pytest.mark.parametrize(
    "raw",
    ["تقدّم", " تَقَدَّمْ !! ", "ما تتقدّمش", "امشي  لليسار؟", "ـــ", "", "   ", "a  b . c ."],
)
def test_idempotent(raw):
    once = normalize_text(raw)
    assert normalize_text(once) == once


@pytest.mark.parametrize("value", [None, 1, 2.5, True, ["وقف"], {"t": "وقف"}, b"\xd9\x88"])
def test_non_string_raises_type_error(value):
    with pytest.raises(TypeError):
        normalize_text(value)
