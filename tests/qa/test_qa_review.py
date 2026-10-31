"""Independent QA tests (QA review of 2026-10-31).

Written by the QA agent, independently of the developer's tests:
- expected values are literal (texts, timestamps, speeds), never recomputed with
  the implementation's own helpers (no normalize_text / derive_command_id in expectations);
- every "no movement" claim is checked by counting adapter calls, not by reading messages;
- time is controlled with ManualClock, no real sleeping.

Tests marked ``xfail(strict=True)`` are reproductions of open QA defects (QA-xx in
QA_REPORT.md). They assert the REQUIRED behavior and currently fail. Once the
defect is fixed they will XPASS, which fails the run: then remove the marker.
"""

from __future__ import annotations

import ast
import json
import math
import random
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from drone_voice.clock import ManualClock
from drone_voice.controller import AdapterResult, SimulatedController
from drone_voice.dedup import InMemoryDedupStore, JsonlDedupStore
from drone_voice.dispatcher import Dispatcher
from drone_voice.interpreter import Interpreter
from drone_voice.normalization import normalize_text
from drone_voice.pipeline import DEFAULT_SETTINGS_PATH, DEFAULT_VOCABULARY_PATH, build_pipeline
from drone_voice.settings import load_settings, settings_from_dict
from drone_voice.transcription import TranscriptionEvent
from drone_voice.vocabulary import ConfigError, Vocabulary, load_vocabulary, vocabulary_from_dict
from drone_voice.contract import Confidence

REPO = Path(__file__).resolve().parents[2]
T0 = datetime(2026, 3, 1, 9, 0, 0, tzinfo=timezone.utc)

# Hand-written expectations (from config/settings.json, read by a human, not by code).
EXPECTED = {
    "تقدّم": ("MOVE_FORWARD", "move", "FORWARD", 0.3, 1.0),
    "ارجع": ("MOVE_BACKWARD", "move", "BACKWARD", 0.3, 1.0),
    "امشي لليمين": ("MOVE_RIGHT", "move", "RIGHT", 0.3, 1.0),
    "امشي لليسار": ("MOVE_LEFT", "move", "LEFT", 0.3, 1.0),
    "اطلع": ("MOVE_UP", "move", "UP", 0.2, 1.0),
    "اهبط": ("MOVE_DOWN", "move", "DOWN", 0.2, 1.0),
    "وقف": ("STOP", "stop", None, None, None),
}


# -- helpers ------------------------------------------------------------------------------

@pytest.fixture
def clock():
    return ManualClock(T0)


@pytest.fixture
def sim(clock):
    return SimulatedController(clock)


@pytest.fixture
def pipe(clock, sim):
    return build_pipeline(clock, controller=sim, dedup_store=InMemoryDedupStore())


def ev(uid, text, final=True, source="qa"):
    return TranscriptionEvent(utterance_id=uid, text=text, is_final=final, source=source)


def calls(sim):
    return [(c["method"], c.get("direction") or c.get("cause")) for c in sim.calls]


def iso(seconds):
    dt = T0 + timedelta(seconds=seconds)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def msg(cid="cmd-qa-1", uid="qa-u1", text="تقدّم", norm="تقدم", intent="MOVE_FORWARD",
        status="recognized", reason=None, created="2026-03-01T09:00:00.000Z",
        expires="2026-03-01T09:00:02.000Z", confidence=None, **extra):
    m = {"version": 1, "command_id": cid, "utterance_id": uid, "source": "qa", "created_at": created,
         "expires_at": expires, "recognized_text": text, "normalized_text": norm, "intent": intent,
         "status": status, "reason": reason, "confidence": confidence}
    m.update(extra)
    return m


@pytest.fixture
def disp(clock, sim):
    return Dispatcher(sim, load_settings(DEFAULT_SETTINGS_PATH), clock,
                      vocabulary=load_vocabulary(DEFAULT_VOCABULARY_PATH))


# =========================================================================================
# A — Recognition and Arabic
# =========================================================================================

@pytest.mark.parametrize("text", list(EXPECTED))
def test_A_each_configured_expression_reaches_the_adapter_with_the_right_call(pipe, sim, text):
    intent, method, direction, speed, duration = EXPECTED[text]
    message, result = pipe.handle_event(ev("a1", text))
    assert message.intent.value == intent and message.status == "recognized"
    assert result.outcome.value == "adapter_accepted"
    assert len(sim.calls) == 1
    call = sim.calls[0]
    assert call["method"] == method
    if method == "move":
        assert (call["direction"], call["speed_m_s"], call["duration_s"]) == (direction, speed, duration)
    else:
        assert call["cause"] == "stop_command"


@pytest.mark.parametrize("variant, intent", [
    ("تَقَدَّمْ", "MOVE_FORWARD"),                  # full harakat
    ("تقدم", "MOVE_FORWARD"),                       # no shadda
    ("  تقدّم \t", "MOVE_FORWARD"),                 # outer whitespace
    ("تقـــدّم", "MOVE_FORWARD"),                   # tatweel
    ("تقدّم؟", "MOVE_FORWARD"),                     # trailing Arabic question mark (documented rule)
    ("تقدّم !!", "MOVE_FORWARD"),
    ("\u200fوقف\u200e", "STOP"),                    # RLM / LRM around the word
    ("\ufeffاهبط", "MOVE_DOWN"),                    # BOM
    ("امشي   لليمين", "MOVE_RIGHT"),               # inner spaces
    ("امشي\tلليسار", "MOVE_LEFT"),                  # tab
    ("امشي\nلليمين", "MOVE_RIGHT"),                 # newline is whitespace
])
def test_A_documented_normalization_variants(pipe, variant, intent):
    message, _ = pipe.handle_event(ev("v", variant))
    assert message.intent is not None and message.intent.value == intent
    assert message.recognized_text == variant  # original kept byte for byte


@pytest.mark.parametrize("text", [
    "تقدّم،",            # Arabic comma is not trailing punctuation in v1
    "تقد م",              # split word
    "ﺗﻘﺪﻡ",                # Arabic presentation forms (no NFKC in v1)
    "تقدّم\u200b",        # zero width space is not removed
    "إرجع",               # hamza alef: letters are not unified in v1
    "تقدن", "تقدّ", "تقدّمم", "ارجعو", "اطلعوا",
])
def test_A_close_but_different_strings_fail_closed(pipe, sim, text):
    message, result = pipe.handle_event(ev("c", text))
    assert message.status == "rejected" and message.reason.value == "UNKNOWN_EXPRESSION"
    assert result.outcome.value == "not_actionable"
    assert sim.calls == []


def test_A_normalization_is_idempotent_on_random_inputs():
    pool = list("تقدماهبطلعرجويسشنفكـ") + [chr(c) for c in range(0x064B, 0x0656)] + [
        "\u0670", " ", "\t", "\n", ".", "!", "?", "؟", "،", "\u200e", "\u200f", "\u061c", "\ufeff", "آ", "أ", "ا", "\u0654"]
    rng = random.Random(20261031)
    for _ in range(5000):
        s = "".join(rng.choice(pool) for _ in range(rng.randint(0, 12)))
        once = normalize_text(s)
        assert normalize_text(once) == once, repr(s)


def test_A_normalization_literal_expectations():
    assert normalize_text("  تَقَدَّمْ  ؟ . ") == "تقدم"
    assert normalize_text("امشي ـ لليمين") == "امشي لليمين"  # tatweel then whitespace collapse
    assert normalize_text("تقدم، اطلع!") == "تقدم، اطلع"      # inner comma kept


@pytest.mark.parametrize("a, b", [
    ("تقدّم", "تَقَدَّم"),     # diacritics only
    ("اطلع", "اطلع؟"),        # trailing punctuation only
    ("وقف", " وقف "),         # spaces only
    ("ارجع", "ار\u200cجع"),   # ZWNJ is NOT removed -> these differ, see the control below
])
def test_A_collision_after_normalization_is_refused(a, b):
    entries = [(e, i) for e, i in [("تقدّم", "MOVE_FORWARD"), ("ارجع", "MOVE_BACKWARD"),
                                   ("امشي لليمين", "MOVE_RIGHT"), ("امشي لليسار", "MOVE_LEFT"),
                                   ("اطلع", "MOVE_UP"), ("اهبط", "MOVE_DOWN"), ("وقف", "STOP")]]
    other = "STOP" if dict(entries).get(a) != "STOP" else "MOVE_UP"
    conflicting = entries + [(b, other)]
    if "\u200c" in b:  # control: genuinely different strings must NOT be reported as a collision
        Vocabulary(conflicting)
        return
    with pytest.raises(ConfigError, match="ambiguous"):
        Vocabulary(conflicting)


def test_A_conflicting_vocabulary_file_is_refused_by_the_cli(tmp_path):
    data = json.loads((REPO / "config" / "vocabulary.json").read_text(encoding="utf-8"))
    data["entries"].append({"expression": "وَقِّفْ", "intent": "MOVE_FORWARD"})  # == "وقف" after normalization
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    proc = subprocess.run([sys.executable, "-m", "drone_voice", "check-config", "--vocabulary", str(path)],
                          cwd=REPO, capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 2 and "ambiguous" in proc.stderr


@pytest.mark.parametrize("text, reason", [
    ("", "EMPTY_INPUT"), ("   ", "EMPTY_INPUT"), ("\t\n", "EMPTY_INPUT"), ("؟!", "EMPTY_INPUT"),
    ("ّ", "EMPTY_INPUT"),  # a lone shadda
    ("hello", "UNKNOWN_EXPRESSION"), ("123", "UNKNOWN_EXPRESSION"),
    (None, "INVALID_INPUT_TYPE"), (0, "INVALID_INPUT_TYPE"), (1.5, "INVALID_INPUT_TYPE"),
    (True, "INVALID_INPUT_TYPE"), ([], "INVALID_INPUT_TYPE"), (["تقدّم"], "INVALID_INPUT_TYPE"),
    ({}, "INVALID_INPUT_TYPE"), ({"text": "تقدّم"}, "INVALID_INPUT_TYPE"), (b"\xd9\x88\xd9\x82\xd9\x81", "INVALID_INPUT_TYPE"),
])
def test_A_empty_unknown_and_wrong_types_never_move(pipe, sim, text, reason):
    message, result = pipe.handle_event(ev("e", text))
    assert (message.status, message.intent, message.reason.value) == ("rejected", None, reason)
    assert result.transmitted is False and sim.calls == []


@pytest.mark.parametrize("text", [
    "ما تتقدّمش", "ماتتقدّمش", "لا تتقدّم", "ما تهبطش", "ما ترجعش", "ما توقفش",  # negations
    "قلتلك تقدّم", "تقدّم شوية", "تقدّم يا صاحبي", "قال لي اطلع",                  # embedded
    "تقدّم و اطلع", "تقدّم، اطلع", "تقدّم اطلع", "تقدّم\nوقف", "وقف ثم تقدّم",       # two orders
    "تقدّم تقدّم", "وقف وقف",                                                         # repetition
])
def test_A_negation_embedding_and_multiple_orders_never_move(pipe, sim, text):
    message, result = pipe.handle_event(ev("n", text))
    assert message.status == "rejected" and message.intent is None
    assert sim.calls == []


def test_A_confidence_never_turns_unknown_into_a_command_and_is_never_invented(pipe, sim):
    high = TranscriptionEvent("k1", "تقدّم شوية", True, "qa", Confidence(0.999, "qa-model:prob"))
    message, _ = pipe.handle_event(high)
    assert message.status == "rejected" and sim.calls == []
    assert message.to_dict()["confidence"] == {"value": 0.999, "provider": "qa-model:prob"}
    message, _ = pipe.handle_event(ev("k2", "تقدّم"))
    assert message.to_dict()["confidence"] is None


# =========================================================================================
# B — JSON contract
# =========================================================================================

def test_B_handwritten_valid_message_is_accepted(disp, sim):
    assert disp.dispatch(msg()).outcome.value == "adapter_accepted"
    assert calls(sim) == [("move", "FORWARD")]


@pytest.mark.parametrize("field", ["version", "command_id", "utterance_id", "source", "created_at",
                                   "expires_at", "recognized_text", "normalized_text", "intent",
                                   "status", "reason", "confidence"])
def test_B_every_field_is_required(disp, sim, field):
    m = msg()
    del m[field]
    assert disp.dispatch(m).outcome.value == "rejected"
    assert sim.calls == []


@pytest.mark.parametrize("mutation, code", [
    ({"extra": 1}, "INVALID_STRUCTURE"),
    ({"version": 2}, "UNSUPPORTED_VERSION"), ({"version": 0}, "UNSUPPORTED_VERSION"),
    ({"version": "1"}, "INVALID_STRUCTURE"), ({"version": 1.0}, "INVALID_STRUCTURE"),
    ({"version": True}, "INVALID_STRUCTURE"),
    ({"intent": "LAND"}, "UNKNOWN_INTENT"), ({"intent": "TAKEOFF"}, "UNKNOWN_INTENT"),
    ({"intent": "move_forward"}, "UNKNOWN_INTENT"), ({"intent": "__class__"}, "UNKNOWN_INTENT"),
    ({"intent": 1}, "INVALID_STRUCTURE"),
    ({"status": "executed"}, "INVALID_STRUCTURE"), ({"status": None}, "INVALID_STRUCTURE"),
    ({"reason": "UNKNOWN_EXPRESSION"}, "INCONSISTENT_MESSAGE"),           # recognized + reason
    ({"intent": None}, "INCONSISTENT_MESSAGE"),                           # recognized without intent
    ({"status": "rejected"}, "INCONSISTENT_MESSAGE"),                     # rejected + intent
    ({"status": "rejected", "intent": None}, "INCONSISTENT_MESSAGE"),     # rejected without reason
    ({"status": "rejected", "intent": None, "reason": "NOPE"}, "INVALID_STRUCTURE"),
    ({"normalized_text": "اطلع"}, "INCONSISTENT_MESSAGE"),
    ({"recognized_text": 5}, "INVALID_STRUCTURE"),
    ({"command_id": "cmd qa"}, "INVALID_STRUCTURE"), ({"command_id": "أمر-1"}, "INVALID_STRUCTURE"),
    ({"command_id": ""}, "INVALID_STRUCTURE"), ({"utterance_id": " "}, "INVALID_STRUCTURE"),
    ({"source": 3}, "INVALID_STRUCTURE"),
    ({"created_at": "2026-03-01T09:00:00+00:00"}, "INVALID_TIMESTAMP"),
    ({"created_at": "2026-03-01 09:00:00Z"}, "INVALID_TIMESTAMP"),
    ({"created_at": "2026-03-01T09:00:00"}, "INVALID_TIMESTAMP"),
    ({"created_at": "2026-02-30T09:00:00Z"}, "INVALID_TIMESTAMP"),
    ({"created_at": "2026-03-01T24:00:00Z"}, "INVALID_TIMESTAMP"),
    ({"created_at": "٢٠٢٦-03-01T09:00:00Z"}, "INVALID_TIMESTAMP"),   # Arabic-Indic digits
    ({"created_at": 1772355600}, "INVALID_TIMESTAMP"),
    ({"expires_at": "2026-03-01T09:00:00.000Z"}, "INVALID_TIMESTAMP"),   # == created_at
    ({"expires_at": "2026-03-01T08:59:59.000Z"}, "INVALID_TIMESTAMP"),   # before created_at
    ({"expires_at": "2026-03-01T09:00:10.001Z"}, "INVALID_TIMESTAMP"),   # window > max_ttl_s (10 s)
    ({"confidence": {"value": 0.5}}, "INVALID_STRUCTURE"),
    ({"confidence": {"value": "0.5", "provider": "m"}}, "INVALID_STRUCTURE"),
    ({"confidence": {"value": True, "provider": "m"}}, "INVALID_STRUCTURE"),
    ({"confidence": 0.5}, "INVALID_STRUCTURE"),
])
def test_B_invalid_messages_are_refused_without_any_call(disp, sim, mutation, code):
    result = disp.dispatch(msg(**mutation))
    assert (result.outcome.value, result.reason) == ("rejected", code)
    assert sim.calls == []


@pytest.mark.parametrize("raw", [
    b"", b"{", b"null", b"[]", b'"x"', b"\xff\xfe",
    "\ufeff".encode() + json.dumps(msg()).encode(),                                       # BOM
    json.dumps(msg()).encode()[:-1] + b', "version": 1}',                                 # duplicate key
    json.dumps(msg(confidence={"value": 1, "provider": "m"})).replace("1,", "NaN,", 1).encode(),
    json.dumps(msg(confidence={"value": 1, "provider": "m"})).replace('"value": 1', '"value": 1e999').encode(),
    json.dumps(msg()).encode() + b" {}",                                                  # trailing data
])
def test_B_malformed_json_is_refused_without_any_call(disp, sim, raw):
    result = disp.dispatch_json(raw)
    assert result.outcome.value == "rejected" and result.reason in {"MALFORMED_JSON", "INVALID_STRUCTURE"}
    assert sim.calls == []


@pytest.mark.parametrize("text, norm, intent", [
    ("ما تتقدّمش", "ما تتقدمش", "MOVE_FORWARD"),   # negation relabelled as a command
    ("اطلع", "اطلع", "MOVE_DOWN"),                  # opposite direction
    ("وقف", "وقف", "MOVE_UP"),                      # stop word relabelled as a movement
])
def test_B_forged_messages_bypassing_the_interpreter_are_refused(disp, sim, text, norm, intent):
    result = disp.dispatch(msg(text=text, norm=norm, intent=intent))
    assert (result.outcome.value, result.reason) == ("rejected", "TEXT_INTENT_MISMATCH")
    assert sim.calls == []


def test_B_QA03_dispatcher_built_without_vocabulary_must_not_accept_forged_intent(clock, sim):
    disp = Dispatcher(sim, load_settings(DEFAULT_SETTINGS_PATH), clock)  # public constructor, vocabulary omitted
    result = disp.dispatch(msg(text="ما تتقدّمش", norm="ما تتقدمش", intent="MOVE_FORWARD"))
    assert result.outcome.value == "rejected"
    assert sim.calls == []


def test_B_utf8_file_roundtrip_through_cli_and_dispatcher(tmp_path, clock, sim, disp):
    out = tmp_path / "m.json"
    proc = subprocess.run([sys.executable, "-m", "drone_voice", "demo", "امشي لليسار", "--now",
                           "2026-03-01T09:00:00Z", "--utterance-id", "rt-1", "--save-json", str(out)],
                          cwd=REPO, capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr
    raw = out.read_bytes()
    assert "امشي لليسار".encode("utf-8") in raw and b"\\u06" not in raw  # readable UTF-8, not escaped
    assert json.loads(raw.decode("utf-8"))["recognized_text"] == "امشي لليسار"
    assert disp.dispatch_json(raw).outcome.value == "adapter_accepted"
    assert calls(sim) == [("move", "LEFT")]


def test_B_recognized_is_not_executed_expired_message_keeps_status_but_no_call(disp, sim, clock):
    clock.advance(5)
    m = msg()
    result = disp.dispatch(m)
    assert m["status"] == "recognized"
    assert (result.outcome.value, result.reason, result.transmitted, result.simulated) == ("rejected", "EXPIRED", False, None)
    assert sim.calls == []


# =========================================================================================
# C — Time and repetitions (controlled clock)
# =========================================================================================

def test_C_partials_are_never_executed_even_when_they_match(pipe, sim):
    for text in ["تقدّم", "وقف", "اطلع"]:
        assert pipe.handle_event(ev("p", text, final=False)) == (None, None)
    assert sim.calls == []


def test_C_retransmission_and_revised_final_of_same_utterance_execute_once(pipe, sim, clock):
    pipe.handle_event(ev("r1", "تقدّم"))
    clock.advance(0.1)
    _, again = pipe.handle_event(ev("r1", "تقدّم"))
    _, revised = pipe.handle_event(ev("r1", "ارجع"))
    assert (again.reason, revised.reason) == ("DUPLICATE", "DUPLICATE")
    assert calls(sim) == [("move", "FORWARD")]


def test_C_distinct_utterances_with_identical_text_both_execute(pipe, sim, clock):
    pipe.handle_event(ev("d1", "اطلع"))
    clock.advance(0.3)
    pipe.handle_event(ev("d2", "اطلع"))
    assert calls(sim) == [("move", "UP"), ("move", "UP")]


def test_C_same_utterance_id_from_two_sources_are_distinct(pipe, sim):
    pipe.handle_event(ev("same", "اهبط", source="mic-A"))
    pipe.handle_event(ev("same", "اهبط", source="mic-B"))
    assert calls(sim) == [("move", "DOWN"), ("move", "DOWN")]


def test_C_exact_expiry_and_future_boundaries(disp, sim, clock):
    # Dev fix QA-07: distinct commands need distinct utterance_ids (one utterance = one effect).
    clock.set(T0 + timedelta(seconds=1.999))
    assert disp.dispatch(msg(cid="cmd-e1")).outcome.value == "adapter_accepted"
    clock.set(T0 + timedelta(seconds=2))
    assert disp.dispatch(msg(cid="cmd-e2", uid="qa-e2", text="وقف", norm="وقف", intent="STOP")).reason == "EXPIRED"
    clock.set(T0)
    ok = msg(cid="cmd-f1", uid="qa-f1", text="وقف", norm="وقف", intent="STOP",
             created="2026-03-01T09:00:00.500Z", expires="2026-03-01T09:00:02.500Z")
    late = msg(cid="cmd-f2", uid="qa-f2", text="وقف", norm="وقف", intent="STOP",
               created="2026-03-01T09:00:00.501Z", expires="2026-03-01T09:00:02.501Z")
    assert disp.dispatch(late).reason == "FUTURE_TIMESTAMP"
    assert disp.dispatch(ok).outcome.value == "adapter_accepted"
    assert calls(sim) == [("move", "FORWARD"), ("stop", "stop_command")]


def test_C_expiry_is_evaluated_after_waiting_for_the_lock(clock):
    """A message that expires while another call holds the dispatcher must be refused."""
    release, entered = threading.Event(), threading.Event()

    class Blocking(SimulatedController):
        def move(self, **kw):
            entered.set()
            assert release.wait(5)
            return super().move(**kw)

    sim = Blocking(clock)
    disp = Dispatcher(sim, load_settings(DEFAULT_SETTINGS_PATH), clock,
                      vocabulary=load_vocabulary(DEFAULT_VOCABULARY_PATH))
    results = {}
    t1 = threading.Thread(target=lambda: results.setdefault("first", disp.dispatch(msg(cid="cmd-l1"))))
    t1.start()
    assert entered.wait(5)
    waiting = msg(cid="cmd-l2", uid="qa-u2", text="اطلع", norm="اطلع", intent="MOVE_UP")
    t2 = threading.Thread(target=lambda: results.setdefault("second", disp.dispatch(waiting)))
    t2.start()
    clock.advance(3)  # message 2 expires while it waits behind message 1
    release.set()
    t1.join(5)
    t2.join(5)
    assert results["first"].outcome.value == "adapter_accepted"
    assert results["second"].reason == "EXPIRED"
    assert [c["method"] for c in sim.calls if c["method"] == "move"] == ["move"]


def test_C_out_of_order_move_after_stop_is_refused(disp, sim, clock):
    clock.advance(1)
    stop = msg(cid="cmd-s", uid="qa-s", text="وقف", norm="وقف", intent="STOP",
               created=iso(0.5), expires=iso(2.5))
    old_move = msg(cid="cmd-m", created=iso(0.2), expires=iso(2.2))
    assert disp.dispatch(stop).outcome.value == "adapter_accepted"
    assert disp.dispatch(old_move).reason == "OUT_OF_ORDER"
    assert calls(sim) == [("stop", "stop_command")]


def test_C_QA04_expired_utterance_must_not_execute_when_reinterpreted_later(clock, sim):
    vocab, settings = load_vocabulary(DEFAULT_VOCABULARY_PATH), load_settings(DEFAULT_SETTINGS_PATH)
    interp = Interpreter(vocab, clock, settings.ttl_s)
    disp = Dispatcher(sim, settings, clock, vocabulary=vocab)
    spoken = ev("late-1", "تقدّم")
    first = interp.process_event(spoken)          # spoken at T0
    clock.advance(5)                              # transport delay
    assert disp.dispatch(first.to_dict()).reason == "EXPIRED"
    again = interp.process_event(spoken)          # the model re-emits the same final utterance
    disp.dispatch(again.to_dict())
    assert sim.calls == []                        # the order given 5 s ago must not move the drone now


def test_C_QA04_move_superseded_by_stop_must_not_execute_when_reinterpreted(clock, sim):
    vocab, settings = load_vocabulary(DEFAULT_VOCABULARY_PATH), load_settings(DEFAULT_SETTINGS_PATH)
    interp = Interpreter(vocab, clock, settings.ttl_s)
    disp = Dispatcher(sim, settings, clock, vocabulary=vocab)
    move_event = ev("ooo-move", "تقدّم")
    move_1 = interp.process_event(move_event)                    # "forward" at T0
    clock.advance(0.5)
    disp.dispatch(interp.process_event(ev("ooo-stop", "وقف")).to_dict())  # "stop" at T0+0.5, delivered first
    clock.advance(0.2)
    assert disp.dispatch(move_1.to_dict()).reason == "OUT_OF_ORDER"
    clock.advance(0.3)
    disp.dispatch(interp.process_event(move_event).to_dict())  # retransmitted final of the OLD utterance
    assert calls(sim) == [("stop", "stop_command")]           # user's last word was STOP


def test_C_QA05_move_not_newer_than_accepted_stop_must_be_refused(disp, sim):
    disp.dispatch(msg(cid="cmd-stop", uid="s", text="وقف", norm="وقف", intent="STOP"))
    disp.dispatch(msg(cid="cmd-move", uid="m"))  # same created_at as the STOP, delivered after it
    assert calls(sim) == [("stop", "stop_command")]


def test_C_QA07_same_utterance_with_a_different_command_id_executes_once(disp, sim):
    disp.dispatch(msg(cid="cmd-relay-a"))
    disp.dispatch(msg(cid="cmd-relay-b"))  # same source + utterance_id, id rewritten by a relay
    assert calls(sim) == [("move", "FORWARD")]


def test_C_ids_after_errors(clock, sim):
    settings, vocab = load_settings(DEFAULT_SETTINGS_PATH), load_vocabulary(DEFAULT_VOCABULARY_PATH)
    # adapter failure consumes the id: no second effect on retransmission
    disp = Dispatcher(sim, settings, clock, vocabulary=vocab)
    sim.fail_next("move")
    assert disp.dispatch(msg()).outcome.value == "adapter_failed"
    assert disp.dispatch(msg()).reason == "DUPLICATE"
    assert [c["method"] for c in sim.calls] == ["move"]

    # state store failure does not consume the id: a retry executes exactly once
    class Flaky(InMemoryDedupStore):
        fail = True

        def add(self, cid, at):
            if Flaky.fail:
                raise OSError("disk full")
            super().add(cid, at)

    sim2 = SimulatedController(clock)
    disp2 = Dispatcher(sim2, settings, clock, dedup_store=Flaky(), vocabulary=vocab)
    assert disp2.dispatch(msg(cid="cmd-x")).reason == "STATE_STORE_ERROR"
    assert sim2.calls == []
    Flaky.fail = False
    assert disp2.dispatch(msg(cid="cmd-x")).outcome.value == "adapter_accepted"
    assert disp2.dispatch(msg(cid="cmd-x")).reason == "DUPLICATE"
    assert len(sim2.calls) == 1


def test_C_restart_with_persistent_store_keeps_dedup(tmp_path, clock):
    settings, vocab = load_settings(DEFAULT_SETTINGS_PATH), load_vocabulary(DEFAULT_VOCABULARY_PATH)
    state = tmp_path / "ids.jsonl"
    a = SimulatedController(clock)
    Dispatcher(a, settings, clock, dedup_store=JsonlDedupStore(state), vocabulary=vocab).dispatch(msg())
    b = SimulatedController(clock)  # "restart": new process objects, same file
    result = Dispatcher(b, settings, clock, dedup_store=JsonlDedupStore(state), vocabulary=vocab).dispatch(msg())
    assert result.reason == "DUPLICATE" and b.calls == []


def test_C_LIMIT_restart_with_in_memory_store_reexecutes_within_ttl(clock):
    """Documented limit (not a pass of the guarantee): in-memory dedup is lost on restart."""
    settings, vocab = load_settings(DEFAULT_SETTINGS_PATH), load_vocabulary(DEFAULT_VOCABULARY_PATH)
    a, b = SimulatedController(clock), SimulatedController(clock)
    Dispatcher(a, settings, clock, vocabulary=vocab).dispatch(msg())
    Dispatcher(b, settings, clock, vocabulary=vocab).dispatch(msg())
    assert len(a.calls) == len(b.calls) == 1  # second effect after restart: limit confirmed


def test_C_corrupt_state_file_fails_closed(tmp_path):
    state = tmp_path / "ids.jsonl"
    state.write_bytes(b'{"command_id": "cmd-1", "recorded_at": "x"}\n{"command_id": ')
    with pytest.raises(Exception):
        JsonlDedupStore(state)


# =========================================================================================
# D — Simulated execution
# =========================================================================================

def test_D_movement_stops_exactly_at_end_of_duration(pipe, sim, clock):
    pipe.handle_event(ev("dur", "ارجع"))
    clock.set(T0 + timedelta(seconds=0.999))
    assert pipe.dispatcher.poll() == []
    clock.set(T0 + timedelta(seconds=1))
    assert [r.kind for r in pipe.dispatcher.poll()] == ["auto_stop"]
    assert pipe.dispatcher.poll() == []
    assert calls(sim) == [("move", "BACKWARD"), ("stop", "duration_elapsed")]
    assert [c["at"] for c in sim.calls] == ["2026-03-01T09:00:00.000Z", "2026-03-01T09:00:01.000Z"]


def test_D_stop_interrupts_immediately_and_leaves_nothing_pending(pipe, sim, clock):
    pipe.handle_event(ev("m", "امشي لليمين"))
    clock.advance(0.2)
    _, result = pipe.handle_event(ev("s", "وقف"))
    assert result.at == "2026-03-01T09:00:00.200Z"  # did not wait for the 1 s movement
    assert pipe.dispatcher.active_movement is None
    clock.advance(5)
    pipe.dispatcher.poll()
    assert calls(sim) == [("move", "RIGHT"), ("stop", "stop_command")]


def test_D_new_command_during_movement_replaces_it(pipe, sim, clock):
    pipe.handle_event(ev("m1", "تقدّم"))
    clock.advance(0.4)
    pipe.handle_event(ev("m2", "اهبط"))
    clock.set(T0 + timedelta(seconds=1.0))  # end of the first movement: must not stop the second
    assert pipe.dispatcher.poll() == []
    clock.set(T0 + timedelta(seconds=1.4))
    pipe.dispatcher.poll()
    assert calls(sim) == [("move", "FORWARD"), ("move", "DOWN"), ("stop", "duration_elapsed")]


@pytest.mark.parametrize("speed, duration", [
    (0, 1.0), (0.3, 0), (-0.1, 1.0), (0.3, -1.0), (float("nan"), 1.0), (0.3, float("inf")),
    (0.51, 1.0), (0.3, 3.01), (True, 1.0), ("0.3", 1.0), (None, 1.0),
])
def test_D_motion_bounds_refused(speed, duration):
    data = json.loads((REPO / "config" / "settings.json").read_text(encoding="utf-8"))
    data["motion"]["intents"]["MOVE_UP"] = {"speed_m_s": speed, "duration_s": duration}
    with pytest.raises(ConfigError):
        settings_from_dict(data)


@pytest.mark.parametrize("key, value", [("max_speed_m_s", 1.01), ("max_duration_s", 5.01),
                                        ("max_speed_m_s", 0), ("max_duration_s", -1)])
def test_D_hard_caps_cannot_be_raised_by_config(key, value):
    data = json.loads((REPO / "config" / "settings.json").read_text(encoding="utf-8"))
    data["motion"][key] = value
    with pytest.raises(ConfigError):
        settings_from_dict(data)


def test_D_nonfinite_in_settings_file_is_refused(tmp_path):
    text = (REPO / "config" / "settings.json").read_text(encoding="utf-8").replace('"duration_s": 1.0', '"duration_s": NaN', 1)
    path = tmp_path / "s.json"
    path.write_text(text, encoding="utf-8")
    proc = subprocess.run([sys.executable, "-m", "drone_voice", "check-config", "--settings", str(path)],
                          cwd=REPO, capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 2


def test_D_adapter_failure_is_visible_not_retried_and_guarded(pipe, sim, clock):
    sim.fail_next("move")
    _, result = pipe.handle_event(ev("f", "اطلع"))
    assert (result.outcome.value, result.reason) == ("adapter_failed", "ADAPTER_ERROR")
    clock.advance(0.5)
    assert pipe.dispatcher.poll() == []
    clock.advance(0.5)
    pipe.dispatcher.poll()
    assert [c["method"] for c in sim.calls] == ["move", "stop"]  # never a second move
    assert sim.calls[0]["result"] == "error"


def test_D_adapter_returning_garbage_or_refusing_is_a_failure(clock):
    settings, vocab = load_settings(DEFAULT_SETTINGS_PATH), load_vocabulary(DEFAULT_VOCABULARY_PATH)

    class Odd:
        def __init__(self, answer):
            self.answer, self.n = answer, 0

        def move(self, **kw):
            self.n += 1
            return self.answer

        def stop(self, **kw):
            return AdapterResult(True, True, "ok")

    for answer in [None, True, {"accepted": True}, AdapterResult(False, True, "refused")]:
        odd = Odd(answer)
        result = Dispatcher(odd, settings, clock, vocabulary=vocab).dispatch(msg())
        assert result.outcome.value == "adapter_failed", answer
        assert odd.n == 1


def test_D_QA06_retried_stop_keeps_its_cause(pipe, sim, clock):
    pipe.handle_event(ev("m", "تقدّم"))
    clock.advance(0.1)
    sim.fail_next("stop")
    pipe.handle_event(ev("s", "وقف"))
    pipe.dispatcher.poll()  # retry
    assert calls(sim)[-1] == ("stop", "stop_command")


def test_D_package_imports_only_harmless_stdlib_modules():
    allowed = {"__future__", "argparse", "dataclasses", "datetime", "enum", "json", "math", "os", "pathlib",
               "re", "sys", "threading", "types", "typing", "unicodedata", "uuid"}
    found, dynamic = set(), []
    for path in (REPO / "drone_voice").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                found.add(node.module.split(".")[0])
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {
                    "eval", "exec", "compile", "getattr", "setattr", "__import__"}:
                dynamic.append(f"{path.name}:{node.lineno}")
    assert found <= allowed, found - allowed
    assert dynamic == []


# =========================================================================================
# E — End to end through the public CLI
# =========================================================================================

def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "drone_voice", *args], cwd=REPO,
                          capture_output=True, text=True, encoding="utf-8")


def batch(tmp_path, lines, name="in"):
    src = tmp_path / f"{name}.jsonl"
    src.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n", encoding="utf-8")
    out, state = tmp_path / f"{name}.out.jsonl", tmp_path / f"{name}.state.jsonl"
    proc = run_cli("batch", str(src), "--out", str(out), "--state", str(state), "--now", "2026-03-01T09:00:00Z")
    records = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()] if out.exists() else []
    return proc, records, state


def adapter_calls(records):
    tail = [r for r in records if "simulated_adapter_calls" in r]
    return [(c["method"], c.get("direction") or c.get("cause"), c["at"][11:23])
            for c in (tail[0]["simulated_adapter_calls"] if tail else [])]


def test_E_cli_scenarios_valid_unknown_negation_retransmission_stop(tmp_path):
    proc, records, _ = batch(tmp_path, [
        {"utterance_id": "e1", "text": "تقدّم", "is_final": True, "t": 0},
        {"utterance_id": "e1", "text": "تقدّم", "is_final": True, "t": 0.1},     # retransmission
        {"utterance_id": "e2", "text": "وقف", "is_final": True, "t": 0.4},        # stop during movement
        {"utterance_id": "e3", "text": "شنوة هذا", "is_final": True, "t": 0.5},   # unknown
        {"utterance_id": "e4", "text": "ما تتقدّمش", "is_final": True, "t": 0.6}, # negation
        {"utterance_id": "e5", "text": "اهبط", "is_final": False, "t": 0.7},      # partial
    ])
    assert proc.returncode == 0, proc.stderr
    assert adapter_calls(records) == [("move", "FORWARD", "09:00:00.000"), ("stop", "stop_command", "09:00:00.400")]
    reasons = [r["dispatch"]["reason"] for r in records if "dispatch" in r]
    assert reasons == [None, "DUPLICATE", None, "UNKNOWN_EXPRESSION", "UNKNOWN_EXPRESSION"]


def test_E_demo_valid_command_reaches_simulated_adapter_and_auto_stops():
    proc = run_cli("demo", "اهبط", "--now", "2026-03-01T09:00:00Z", "--utterance-id", "e-demo")
    assert proc.returncode == 0, proc.stderr
    calls_json = proc.stdout.split("== Simulated adapter calls ==\n", 1)[1].rsplit("SIMULATION ONLY", 1)[0]
    got = [(c["method"], c.get("direction") or c.get("cause"), c["at"]) for c in json.loads(calls_json)]
    assert got == [("move", "DOWN", "2026-03-01T09:00:00.000Z"), ("stop", "duration_elapsed", "2026-03-01T09:00:01.000Z")]


def test_E_decision_log_links_transcription_decision_result_and_can_drop_text(tmp_path):
    log = tmp_path / "decisions.jsonl"
    proc = run_cli("demo", "اطلع", "--now", "2026-03-01T09:00:00Z", "--utterance-id", "log-1",
                   "--decision-log", str(log), "--no-log-text")
    assert proc.returncode == 0, proc.stderr
    entries = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    interpreted = [e for e in entries if e["event"] == "interpreted"][0]
    dispatched = [e for e in entries if e["event"] == "dispatched"]
    assert interpreted["utterance_id"] == "log-1"
    assert {d["command_id"] for d in dispatched} == {interpreted["command_id"]}
    assert "اطلع" not in log.read_text(encoding="utf-8")  # no transcription text when asked
    assert all(e.get("text_redacted") for e in entries if e["event"] == "interpreted")


def test_E_QA01_batch_stops_movement_at_end_of_duration(tmp_path):
    proc, records, _ = batch(tmp_path, [
        {"utterance_id": "q1", "text": "امشي لليمين", "is_final": True, "t": 0},
        {"utterance_id": "q2", "text": "شنوة", "is_final": True, "t": 10},
    ])
    assert proc.returncode == 0
    assert adapter_calls(records) == [("move", "RIGHT", "09:00:00.000"), ("stop", "duration_elapsed", "09:00:01.000")]


def test_E_QA02_batch_refuses_invalid_file_before_any_action(tmp_path):
    proc, records, state = batch(tmp_path, [
        {"utterance_id": "ok-1", "text": "اطلع", "is_final": True, "t": 0},
        {"utterance_id": "x" * 300, "text": "وقف", "is_final": True, "t": 0.5},
    ])
    assert proc.returncode == 2 and "Traceback" not in proc.stderr
    assert adapter_calls(records) == []
    assert not state.exists() or state.read_text(encoding="utf-8") == ""


def test_E_QA02_pipeline_does_not_raise_on_event_fields_the_contract_refuses(pipe, sim):
    for uid, source in [("u" * 257, "qa"), ("ok", "s" * 65)]:
        try:
            event = TranscriptionEvent(uid, "وقف", True, source)
        except ValueError:
            continue  # refusing at the boundary is an acceptable fix
        pipe.handle_event(event)  # must not raise
    assert sim.calls == [] or calls(sim) == [("stop", "stop_command")]


# =========================================================================================
# Re-check of revision 2 (QA, 2026-10-31) — new findings
# =========================================================================================

@pytest.mark.xfail(strict=True, reason="QA-08: a message whose command_id is not its own derivation consumes another utterance's id; the later real STOP is DUPLICATE")
def test_R2_QA08_foreign_command_id_must_not_block_a_later_stop(clock, sim):
    vocab, settings = load_vocabulary(DEFAULT_VOCABULARY_PATH), load_settings(DEFAULT_SETTINGS_PATH)
    interp = Interpreter(vocab, clock, settings.ttl_s)
    disp = Dispatcher(sim, settings, clock, vocabulary=vocab)
    disp.dispatch(interp.process_event(ev("seg-1", "تقدّم", source="mic")).to_dict())
    other = interp.process_event(ev("seg-x", "ما نعرفش", source="mic")).to_dict()
    future_stop = interp.process_event(ev("seg-2", "وقف", source="mic"))  # what the next utterance will carry
    other["command_id"] = future_stop.command_id
    disp.dispatch(other)                      # relay/forger reuses the id of a future utterance
    clock.advance(0.3)
    disp.dispatch(future_stop.to_dict())      # the user's real STOP
    assert calls(sim) == [("move", "FORWARD"), ("stop", "stop_command")]


@pytest.mark.xfail(strict=True, reason="QA-09: an out-of-range spoken_at passes TranscriptionEvent and raises OverflowError in the interpreter")
def test_R2_QA09_out_of_range_spoken_at_is_refused_cleanly(pipe, sim):
    try:
        event = TranscriptionEvent("u", "تقدّم", True, "qa", spoken_at=datetime.max.replace(tzinfo=timezone.utc))
    except ValueError:
        return  # refusing at the boundary is an acceptable fix
    pipe.handle_event(event)  # must not raise
    assert sim.calls == []
