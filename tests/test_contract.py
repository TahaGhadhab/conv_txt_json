from datetime import datetime

import pytest

from conftest import make_message, ts
from drone_voice.contract import ContractError, format_timestamp, validate_message
from drone_voice.jsonio import JsonFormatError, loads_strict, read_json_file, write_json_file


def code_of(message) -> str:
    with pytest.raises(ContractError) as info:
        validate_message(message)
    return info.value.code.value


def test_valid_recognized_and_rejected_messages():
    assert validate_message(make_message()).intent.value == "MOVE_FORWARD"
    rejected = make_message(intent=None, status="rejected", reason="UNKNOWN_EXPRESSION",
                            recognized_text="ما تتقدّمش", normalized_text="ما تتقدمش")
    assert validate_message(rejected).intent is None
    typed = make_message(intent=None, status="rejected", reason="INVALID_INPUT_TYPE",
                         recognized_text=None, normalized_text=None)
    assert validate_message(typed).reason.value == "INVALID_INPUT_TYPE"


def test_round_trip_is_identical():
    raw = make_message(confidence={"value": 0.5, "provider": "p"})
    assert validate_message(raw).to_dict() == raw


def test_arabic_survives_utf8_write_and_read(tmp_path):
    path = tmp_path / "m.json"
    write_json_file(path, make_message())
    data = path.read_bytes()
    assert "تقدّم".encode("utf-8") in data          # readable Arabic, not \\u escapes
    assert b"\\u" not in data
    assert read_json_file(path) == make_message()


def test_write_never_overwrites(tmp_path):
    path = tmp_path / "m.json"
    write_json_file(path, {"a": 1})
    with pytest.raises(FileExistsError):
        write_json_file(path, {"a": 2})
    assert read_json_file(path) == {"a": 1}


@pytest.mark.parametrize(
    "overrides, code",
    [
        ({"version": 2}, "UNSUPPORTED_VERSION"),
        ({"version": 0}, "UNSUPPORTED_VERSION"),
        ({"version": "1"}, "INVALID_STRUCTURE"),
        ({"version": True}, "INVALID_STRUCTURE"),
        ({"version": 1.0}, "INVALID_STRUCTURE"),
        ({"extra": 1}, "INVALID_STRUCTURE"),
        ({"command_id": ""}, "INVALID_STRUCTURE"),
        ({"command_id": 5}, "INVALID_STRUCTURE"),
        ({"command_id": "abc\n"}, "INVALID_STRUCTURE"),
        ({"utterance_id": "  "}, "INVALID_STRUCTURE"),
        ({"source": None}, "INVALID_STRUCTURE"),
        ({"status": "executed"}, "INVALID_STRUCTURE"),
        ({"recognized_text": 5}, "INVALID_STRUCTURE"),
        ({"intent": 3}, "INVALID_STRUCTURE"),
        ({"intent": "TAKE_OFF"}, "UNKNOWN_INTENT"),
        ({"intent": "move"}, "UNKNOWN_INTENT"),
        ({"intent": None}, "INCONSISTENT_MESSAGE"),
        ({"reason": "UNKNOWN_EXPRESSION"}, "INCONSISTENT_MESSAGE"),
        ({"normalized_text": "اطلع"}, "INCONSISTENT_MESSAGE"),
        ({"recognized_text": None}, "INCONSISTENT_MESSAGE"),
        ({"status": "rejected"}, "INCONSISTENT_MESSAGE"),
        ({"status": "rejected", "intent": None}, "INCONSISTENT_MESSAGE"),
        ({"status": "rejected", "intent": None, "reason": "BAD"}, "INVALID_STRUCTURE"),
        ({"status": "rejected", "intent": None, "reason": "EMPTY_INPUT"}, "INCONSISTENT_MESSAGE"),
        ({"created_at": "2026-01-01T12:00:00"}, "INVALID_TIMESTAMP"),
        ({"created_at": "2026-01-01T12:00:00+00:00"}, "INVALID_TIMESTAMP"),
        ({"created_at": "2026-13-01T12:00:00Z"}, "INVALID_TIMESTAMP"),
        ({"created_at": "٢٠٢٦-01-01T12:00:00Z"}, "INVALID_TIMESTAMP"),
        ({"created_at": 1767268800}, "INVALID_TIMESTAMP"),
        ({"expires_at": ts(0)}, "INVALID_TIMESTAMP"),
        ({"expires_at": ts(-1)}, "INVALID_TIMESTAMP"),
        ({"confidence": 0.9}, "INVALID_STRUCTURE"),
        ({"confidence": {"value": True, "provider": "p"}}, "INVALID_STRUCTURE"),
        ({"confidence": {"value": "0.9", "provider": "p"}}, "INVALID_STRUCTURE"),
        ({"confidence": {"value": 0.9}}, "INVALID_STRUCTURE"),
        ({"confidence": {"value": 0.9, "provider": "p", "threshold": 0.5}}, "INVALID_STRUCTURE"),
    ],
)
def test_invalid_messages(overrides, code):
    assert code_of(make_message(**overrides)) == code


@pytest.mark.parametrize("field", ["command_id", "created_at", "intent", "status", "reason", "confidence"])
def test_missing_field(field):
    message = make_message()
    del message[field]
    assert code_of(message) == "INVALID_STRUCTURE"


@pytest.mark.parametrize("value", [None, [], "msg", 1, [make_message()]])
def test_non_object(value):
    assert code_of(value) == "INVALID_STRUCTURE"


@pytest.mark.parametrize(
    "raw",
    [
        b"{",
        b'{"a": NaN}',
        b'{"a": Infinity}',
        b'{"a": 1e999}',
        b'{"a": 1, "a": 2}',
        b"\xef\xbb\xbf{}",
        b'{"a": "\xff"}',
    ],
)
def test_strict_json_reader(raw):
    with pytest.raises(JsonFormatError):
        loads_strict(raw)


def test_naive_datetime_cannot_be_formatted():
    with pytest.raises(ValueError):
        format_timestamp(datetime(2026, 1, 1))
