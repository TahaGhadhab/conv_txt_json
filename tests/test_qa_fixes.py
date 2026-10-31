"""Regression tests for the defects reported in QA_REPORT.md (QA-01 .. QA-07)."""

import io
import json
from datetime import timedelta

import pytest

from conftest import START, final, make_message, ts
from drone_voice.cli import main
from drone_voice.controller import SimulatedController
from drone_voice.dedup import JsonlDedupStore
from drone_voice.dispatcher import Dispatcher
from drone_voice.interpreter import Interpreter
from drone_voice.transcription import TranscriptionEvent


def methods(controller):
    return [(c["method"], c.get("direction") or c.get("cause")) for c in controller.calls]


def stop_message(utterance, created=0.0):
    return make_message(utterance=utterance, intent="STOP", recognized_text="وقف", normalized_text="وقف",
                        created_at=ts(created), expires_at=ts(created + 2))


# -- QA-04: a time/order refusal consumes the utterance; re-emission keeps its date --------

@pytest.mark.parametrize("setup, reason", [
    (lambda clock: clock.advance(2), "EXPIRED"),
    (lambda clock: clock.set(START - timedelta(seconds=1)), "FUTURE_TIMESTAMP"),
])
def test_time_refusal_consumes_the_utterance(dispatcher, controller, clock, setup, reason):
    setup(clock)
    assert dispatcher.dispatch(make_message()).reason == reason
    clock.set(START)
    assert dispatcher.dispatch(make_message()).reason == "DUPLICATE"
    assert controller.calls == []


def test_out_of_order_refusal_consumes_the_utterance(dispatcher, controller, clock):
    clock.advance(0.5)
    dispatcher.dispatch(stop_message("stop", created=0.5))
    assert dispatcher.dispatch(make_message()).reason == "OUT_OF_ORDER"
    fresher = make_message(created_at=ts(0.6), expires_at=ts(2.6))  # same utterance, re-dated
    assert dispatcher.dispatch(fresher).reason == "DUPLICATE"
    assert methods(controller) == [("stop", "stop_command")]


def test_invalid_window_refusal_consumes_the_utterance(dispatcher, controller, settings):
    assert dispatcher.dispatch(make_message(expires_at=ts(settings.max_ttl_s + 1))).reason == "INVALID_TIMESTAMP"
    assert dispatcher.dispatch(make_message()).reason == "DUPLICATE"
    assert controller.calls == []


def test_reemitted_final_keeps_its_first_timestamps(vocabulary, clock, settings):
    interpreter = Interpreter(vocabulary, clock, settings.ttl_s)
    first = interpreter.process_event(final("u", "تقدّم"))
    clock.advance(5)
    again = interpreter.process_event(final("u", "تقدّم"))
    assert (again.created_at, again.expires_at) == (first.created_at, first.expires_at) == \
        (START, START + timedelta(seconds=settings.ttl_s))
    other = interpreter.process_event(final("v", "تقدّم"))
    assert other.created_at == START + timedelta(seconds=5)


def test_spoken_at_dates_the_message(vocabulary, clock, settings):
    interpreter = Interpreter(vocabulary, clock, settings.ttl_s)
    clock.advance(10)
    event = TranscriptionEvent("u", "اطلع", True, "test", spoken_at=START + timedelta(seconds=9.5))
    assert interpreter.process_event(event).to_dict()["created_at"] == ts(9.5)


def test_naive_spoken_at_is_refused():
    with pytest.raises(ValueError):
        TranscriptionEvent("u", "اطلع", True, "test", spoken_at=START.replace(tzinfo=None))


def test_late_reemission_through_the_pipeline_never_moves(pipeline, controller, clock):
    clock.advance(5)  # the first copy was lost; the model re-emits 5 s later, with spoken_at
    event = TranscriptionEvent("late", "تقدّم", True, "test", spoken_at=START)
    _, result = pipeline.handle_event(event)
    assert result.reason == "EXPIRED"
    _, result = pipeline.handle_event(event)
    assert result.reason == "DUPLICATE"
    assert controller.calls == []


# -- QA-05: a MOVE must be strictly newer than the latest accepted STOP --------------------

def test_move_with_same_timestamp_as_accepted_stop_is_refused(dispatcher, controller):
    dispatcher.dispatch(stop_message("s"))
    assert dispatcher.dispatch(make_message(utterance="m")).reason == "OUT_OF_ORDER"
    assert dispatcher.dispatch(make_message(utterance="m2", created_at=ts(0.001), expires_at=ts(2))) \
        .outcome.value == "adapter_accepted"
    assert methods(controller) == [("stop", "stop_command"), ("move", "FORWARD")]


# -- QA-07: one utterance, one effect, whatever its command_id -----------------------------

def test_rewritten_command_id_of_same_utterance_is_a_duplicate(dispatcher, controller):
    dispatcher.dispatch(make_message(command_id="cmd-relay-a"))
    assert dispatcher.dispatch(make_message(command_id="cmd-relay-b")).reason == "DUPLICATE"
    assert dispatcher.dispatch(make_message()).reason == "DUPLICATE"  # canonical id too
    assert len(controller.calls) == 1


def test_utterance_key_survives_restart(tmp_path, settings, clock, vocabulary):
    path = tmp_path / "ids.jsonl"
    Dispatcher(SimulatedController(clock), settings, clock, dedup_store=JsonlDedupStore(path),
               vocabulary=vocabulary).dispatch(make_message(command_id="cmd-relay-a"))
    restarted = SimulatedController(clock)
    result = Dispatcher(restarted, settings, clock, dedup_store=JsonlDedupStore(path),
                        vocabulary=vocabulary).dispatch(make_message(command_id="cmd-relay-b"))
    assert result.reason == "DUPLICATE" and restarted.calls == []


# -- QA-03: no vocabulary -> fail closed unless explicitly unchecked -----------------------

def test_dispatcher_without_vocabulary_refuses_recognized_messages(controller, settings, clock):
    dispatcher = Dispatcher(controller, settings, clock)
    forged = make_message(recognized_text="ما تتقدّمش", normalized_text="ما تتقدمش")
    assert dispatcher.dispatch(forged).reason == "UNVERIFIED_INTENT"
    assert dispatcher.dispatch(make_message(utterance="u-2")).reason == "UNVERIFIED_INTENT"
    assert controller.calls == []


def test_unverified_refusal_does_not_consume_the_utterance(controller, settings, clock):
    Dispatcher(controller, settings, clock).dispatch(make_message())
    store_less = Dispatcher(controller, settings, clock, unchecked_text=True)
    assert store_less.dispatch(make_message()).outcome.value == "adapter_accepted"


def test_unchecked_text_must_be_explicit(controller, settings, clock, vocabulary):
    assert Dispatcher(controller, settings, clock, unchecked_text=True).dispatch(make_message()) \
        .outcome.value == "adapter_accepted"
    with pytest.raises(ValueError):
        Dispatcher(controller, settings, clock, vocabulary=vocabulary, unchecked_text=True)


# -- QA-06: a retried STOP keeps its cause; auto stops carry no recognized intent ----------

def test_retried_stop_keeps_cause_and_id(dispatcher, controller, clock):
    dispatcher.dispatch(make_message())
    controller.fail_next("stop")
    stop = dispatcher.dispatch(stop_message("s"))
    [retry] = dispatcher.poll()
    assert (retry.kind, retry.outcome.value, retry.command_id, retry.intent) == \
        ("stop_retry", "adapter_accepted", stop.command_id, None)
    assert methods(controller) == [("move", "FORWARD"), ("stop", "stop_command"), ("stop", "stop_command")]


def test_auto_stop_has_no_recognized_intent(dispatcher, clock):
    dispatcher.dispatch(make_message())
    clock.advance(1)
    [auto] = dispatcher.poll()
    assert (auto.kind, auto.intent) == ("auto_stop", None)


# -- QA-01: movement ends on time in batch, and the simulator bounds itself ----------------

def test_simulator_bounds_its_own_duration_without_a_stop(controller, clock, dispatcher):
    dispatcher.dispatch(make_message())
    assert controller.state == "moving"
    clock.advance(0.999)
    assert controller.state == "moving"
    clock.advance(0.001)
    assert controller.state == "holding"  # even though no stop has been sent yet
    assert [c["method"] for c in controller.calls] == ["move"]


def run_batch(tmp_path, lines):
    src = tmp_path / "in.jsonl"
    src.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n", encoding="utf-8")
    out, state = tmp_path / "out.jsonl", tmp_path / "state.jsonl"
    code = main(["batch", str(src), "--out", str(out), "--state", str(state), "--now", "2026-01-01T12:00:00Z"],
                out=io.StringIO())
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()] if out.exists() else []
    return code, records, state


def test_batch_stops_each_movement_at_its_end(tmp_path):
    code, records, _ = run_batch(tmp_path, [
        {"utterance_id": "a", "text": "امشي لليمين", "is_final": True, "t": 0},
        {"utterance_id": "b", "text": "اطلع", "is_final": True, "t": 0.5},   # replaces, ends at 1.5
        {"utterance_id": "c", "text": "شنوة", "is_final": True, "t": 10},
    ])
    assert code == 0
    calls = records[-1]["simulated_adapter_calls"]
    assert [(c["method"], c.get("direction") or c.get("cause"), c["at"]) for c in calls] == [
        ("move", "RIGHT", "2026-01-01T12:00:00.000Z"),
        ("move", "UP", "2026-01-01T12:00:00.500Z"),
        ("stop", "duration_elapsed", "2026-01-01T12:00:01.500Z"),
    ]


# -- QA-02: event limits match the contract; no traceback, nothing executed ----------------

@pytest.mark.parametrize("utterance_id, source, ok", [
    ("u" * 256, "s", True), ("u" * 257, "s", False), ("u", "s" * 64, True), ("u", "s" * 65, False),
])
def test_event_limits_match_the_contract(pipeline, utterance_id, source, ok):
    if not ok:
        with pytest.raises(ValueError):
            TranscriptionEvent(utterance_id, "وقف", True, source)
        return
    message, result = pipeline.handle_event(TranscriptionEvent(utterance_id, "وقف", True, source))
    assert message.intent.value == "STOP" and result.outcome.value == "adapter_accepted"


def test_batch_with_invalid_line_does_nothing(tmp_path):
    code, records, state = run_batch(tmp_path, [
        {"utterance_id": "ok", "text": "اطلع", "is_final": True, "t": 0},
        {"utterance_id": "x" * 300, "text": "وقف", "is_final": True, "t": 0.5},
    ])
    assert code == 2 and records == [] and not state.exists()


@pytest.mark.parametrize("extra", [["--utterance-id", "x" * 300], ["--now", "not-a-date"]])
def test_demo_with_invalid_arguments_exits_2(extra, capsys):
    assert main(["demo", "وقف", *extra], out=io.StringIO()) == 2
    assert "error:" in capsys.readouterr().err
