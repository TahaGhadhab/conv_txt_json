import json
import re
from pathlib import Path

import pytest

from conftest import make_message, ts
from drone_voice.controller import AdapterResult
from drone_voice.dispatcher import Dispatcher
from drone_voice.intents import Intent

DIRECTIONS = {
    "MOVE_FORWARD": ("تقدّم", "تقدم", "FORWARD"),
    "MOVE_BACKWARD": ("ارجع", "ارجع", "BACKWARD"),
    "MOVE_RIGHT": ("امشي لليمين", "امشي لليمين", "RIGHT"),
    "MOVE_LEFT": ("امشي لليسار", "امشي لليسار", "LEFT"),
    "MOVE_UP": ("اطلع", "اطلع", "UP"),
    "MOVE_DOWN": ("اهبط", "اهبط", "DOWN"),
}


def move_msg(intent="MOVE_FORWARD", utterance="u-1", created=0.0, ttl=2.0):
    text, norm, _ = DIRECTIONS[intent]
    return make_message(utterance=utterance, intent=intent, recognized_text=text, normalized_text=norm,
                        created_at=ts(created), expires_at=ts(created + ttl))


def stop_msg(utterance="u-stop", created=0.0, ttl=2.0):
    return make_message(utterance=utterance, intent="STOP", recognized_text="وقف", normalized_text="وقف",
                        created_at=ts(created), expires_at=ts(created + ttl))


def methods(controller):
    return [(c["method"], c.get("direction") or c.get("cause")) for c in controller.calls]


# -- mapping intent -> adapter method ---------------------------------------------------

@pytest.mark.parametrize("intent", list(DIRECTIONS))
def test_each_move_intent_calls_only_move_with_configured_bounds(dispatcher, controller, settings, intent):
    result = dispatcher.dispatch(move_msg(intent))
    profile = settings.motion[Intent(intent)]
    assert result.outcome.value == "adapter_accepted" and result.simulated is True
    assert controller.calls == [{
        "method": "move", "command_id": result.command_id, "direction": DIRECTIONS[intent][2],
        "speed_m_s": profile.speed_m_s, "duration_s": profile.duration_s, "at": ts(0), "result": "ok",
    }]
    assert 0 < profile.speed_m_s <= settings.max_speed_m_s
    assert 0 < profile.duration_s <= settings.max_duration_s


def test_stop_calls_only_stop(dispatcher, controller):
    result = dispatcher.dispatch(stop_msg())
    assert result.outcome.value == "adapter_accepted"
    assert methods(controller) == [("stop", "stop_command")]


# -- rejected input never moves --------------------------------------------------------

def test_not_actionable_message_makes_no_call(dispatcher, controller):
    msg = make_message(intent=None, status="rejected", reason="UNKNOWN_EXPRESSION",
                       recognized_text="ما تتقدّمش", normalized_text="ما تتقدمش")
    result = dispatcher.dispatch(msg)
    assert (result.outcome.value, result.transmitted) == ("not_actionable", False)
    assert controller.calls == []


@pytest.mark.parametrize(
    "raw",
    [b"{", b"[]", b'"\xd8\xaa\xd9\x82\xd8\xaf\xd9\x85"', b'{"version": 1}', b'{"version": 1, "version": 1}'],
)
def test_malformed_or_invalid_json_makes_no_call(dispatcher, controller, raw):
    assert dispatcher.dispatch_json(raw).outcome.value == "rejected"
    assert controller.calls == []


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"version": 2}, "UNSUPPORTED_VERSION"),
        ({"intent": "LAND"}, "UNKNOWN_INTENT"),
        ({"intent": "__import__('os').system('x')"}, "UNKNOWN_INTENT"),
        ({"intent": "stop"}, "UNKNOWN_INTENT"),
        ({"extra_field": True}, "INVALID_STRUCTURE"),
        ({"intent": "MOVE_DOWN"}, "TEXT_INTENT_MISMATCH"),  # forged: text says forward
        ({"recognized_text": "اطلع", "normalized_text": "اطلع"}, "TEXT_INTENT_MISMATCH"),
        ({"recognized_text": "قلتلك تقدّم", "normalized_text": "قلتلك تقدم"}, "TEXT_INTENT_MISMATCH"),
        ({"normalized_text": "اطلع"}, "INCONSISTENT_MESSAGE"),
    ],
)
def test_bypassing_the_interpreter_does_not_bypass_validation(dispatcher, controller, overrides, reason):
    result = dispatcher.dispatch(make_message(**overrides))
    assert (result.outcome.value, result.reason) == ("rejected", reason)
    assert controller.calls == []


# -- time window -----------------------------------------------------------------------

def test_expiry_boundary(dispatcher, controller, clock):
    clock.advance(1.999)
    assert dispatcher.dispatch(move_msg(utterance="a")).outcome.value == "adapter_accepted"
    clock.advance(0.001)  # now == expires_at of a message created at 0 with ttl 2
    result = dispatcher.dispatch(move_msg(utterance="b"))
    assert result.reason == "EXPIRED"
    assert len(controller.calls) == 1


def test_future_timestamp_boundary(dispatcher, controller, settings):
    skew = settings.max_future_skew_s
    assert dispatcher.dispatch(stop_msg(utterance="a", created=skew)).outcome.value == "adapter_accepted"
    result = dispatcher.dispatch(stop_msg(utterance="b", created=skew + 0.001))
    assert result.reason == "FUTURE_TIMESTAMP"
    assert len(controller.calls) == 1


def test_validity_window_longer_than_allowed_is_rejected(dispatcher, controller, settings):
    result = dispatcher.dispatch(move_msg(ttl=settings.max_ttl_s + 0.001))
    assert result.reason == "INVALID_TIMESTAMP"
    assert dispatcher.dispatch(move_msg(utterance="ok", ttl=settings.max_ttl_s)).outcome.value == "adapter_accepted"
    assert len(controller.calls) == 1


def test_out_of_order_move_rejected_but_stop_accepted(dispatcher, controller, clock):
    clock.advance(1.0)
    assert dispatcher.dispatch(move_msg(utterance="new", created=1.0)).outcome.value == "adapter_accepted"
    late_move = dispatcher.dispatch(move_msg("MOVE_UP", utterance="old", created=0.5))
    assert late_move.reason == "OUT_OF_ORDER"
    late_stop = dispatcher.dispatch(stop_msg(utterance="old-stop", created=0.5))
    assert late_stop.outcome.value == "adapter_accepted"
    assert methods(controller) == [("move", "FORWARD"), ("stop", "stop_command")]


def test_same_created_at_is_not_out_of_order(dispatcher, controller):
    dispatcher.dispatch(move_msg(utterance="a"))
    assert dispatcher.dispatch(move_msg("MOVE_UP", utterance="b")).outcome.value == "adapter_accepted"


# -- duplicates ------------------------------------------------------------------------

def test_retransmission_is_not_executed_twice(dispatcher, controller):
    msg = move_msg()
    assert dispatcher.dispatch(msg).outcome.value == "adapter_accepted"
    assert dispatcher.dispatch(dict(msg)).reason == "DUPLICATE"
    assert dispatcher.dispatch_json(json.dumps(msg, ensure_ascii=False)).reason == "DUPLICATE"
    assert len(controller.calls) == 1


def test_rejected_status_consumes_the_id(dispatcher, controller):
    rejected = make_message(intent=None, status="rejected", reason="UNKNOWN_EXPRESSION",
                            recognized_text="تقدّم شوية", normalized_text="تقدم شوية")
    dispatcher.dispatch(rejected)
    assert dispatcher.dispatch(move_msg()).reason == "DUPLICATE"  # same utterance, revised text
    assert controller.calls == []


def test_invalid_copy_does_not_consume_the_id(dispatcher, controller):
    msg = move_msg()
    assert dispatcher.dispatch({**msg, "intent": "LAND"}).reason == "UNKNOWN_INTENT"
    assert dispatcher.dispatch(msg).outcome.value == "adapter_accepted"


def test_store_failure_prevents_action(controller, settings, clock, vocabulary):
    class BrokenStore:
        def __contains__(self, item):
            return False

        def add(self, command_id, recorded_at):
            raise OSError("disk full")

    dispatcher = Dispatcher(controller, settings, clock, dedup_store=BrokenStore(), vocabulary=vocabulary)
    assert dispatcher.dispatch(move_msg()).reason == "STATE_STORE_ERROR"
    assert controller.calls == []


# -- duration, STOP, concurrency (controlled clock, no sleeping) -----------------------

def test_movement_stops_exactly_at_end_of_duration(dispatcher, controller, clock, settings):
    dispatcher.dispatch(move_msg())
    duration = settings.motion[dispatcher.active_movement.intent].duration_s
    clock.advance(duration - 0.001)
    assert dispatcher.poll() == []
    clock.advance(0.001)
    [result] = dispatcher.poll()
    assert (result.kind, result.outcome.value) == ("auto_stop", "adapter_accepted")
    assert methods(controller) == [("move", "FORWARD"), ("stop", "duration_elapsed")]
    assert dispatcher.active_movement is None
    clock.advance(10)
    assert dispatcher.poll() == []
    assert len(controller.calls) == 2


def test_stop_interrupts_immediately_without_waiting(dispatcher, controller, clock):
    dispatcher.dispatch(move_msg())
    clock.advance(0.2)
    result = dispatcher.dispatch(stop_msg(created=0.2))
    assert result.outcome.value == "adapter_accepted"
    assert controller.calls[-1]["at"] == ts(0.2)
    assert dispatcher.active_movement is None
    clock.advance(5)
    dispatcher.poll()
    assert methods(controller) == [("move", "FORWARD"), ("stop", "stop_command")]


def test_new_move_during_movement_replaces_it(dispatcher, controller, clock, settings):
    dispatcher.dispatch(move_msg())
    clock.advance(0.5)
    result = dispatcher.dispatch(move_msg("MOVE_UP", utterance="u-2", created=0.5))
    assert "replaces" in result.detail
    assert dispatcher.active_movement.command_id == result.command_id
    clock.advance(0.5)  # end of the first movement: nothing, it was replaced
    assert dispatcher.poll() == []
    clock.advance(settings.motion[dispatcher.active_movement.intent].duration_s - 0.5)
    assert len(dispatcher.poll()) == 1
    assert methods(controller) == [("move", "FORWARD"), ("move", "UP"), ("stop", "duration_elapsed")]


def test_dispatch_runs_pending_auto_stop_first(dispatcher, controller, clock):
    dispatcher.dispatch(move_msg())
    clock.advance(3)
    dispatcher.dispatch(stop_msg(created=3))
    assert methods(controller) == [("move", "FORWARD"), ("stop", "duration_elapsed"), ("stop", "stop_command")]


# -- adapter failures ------------------------------------------------------------------

def test_move_failure_is_visible_not_retried_and_still_guarded(dispatcher, controller, clock):
    controller.fail_next("move")
    msg = move_msg()
    result = dispatcher.dispatch(msg)
    assert (result.outcome.value, result.transmitted, result.reason) == ("adapter_failed", True, "ADAPTER_ERROR")
    assert dispatcher.dispatch(msg).reason == "DUPLICATE"   # resending does not re-execute
    clock.advance(1.0)
    dispatcher.poll()                                        # safety stop at planned end
    assert methods(controller) == [("move", "FORWARD"), ("stop", "duration_elapsed")]
    assert [c["result"] for c in controller.calls] == ["error", "ok"]


def test_stop_failure_is_visible_and_stop_is_retried(dispatcher, controller, clock):
    dispatcher.dispatch(move_msg())
    controller.fail_next("stop")
    result = dispatcher.dispatch(stop_msg(created=0))
    assert result.outcome.value == "adapter_failed"
    assert dispatcher.active_movement is not None
    [retry] = dispatcher.poll()
    assert retry.outcome.value == "adapter_accepted"
    assert [(m, c["result"]) for (m, _), c in zip(methods(controller), controller.calls)] == [
        ("move", "ok"), ("stop", "error"), ("stop", "ok")]


def test_auto_stop_failure_is_retried(dispatcher, controller, clock):
    dispatcher.dispatch(move_msg())
    controller.fail_next("stop")
    clock.advance(1.0)
    assert dispatcher.poll()[0].outcome.value == "adapter_failed"
    assert dispatcher.poll()[0].outcome.value == "adapter_accepted"
    assert dispatcher.poll() == []


def test_adapter_refusal_is_a_failure(settings, clock, vocabulary):
    class Refusing:
        def __init__(self):
            self.calls = 0

        def move(self, **kwargs):
            self.calls += 1
            return AdapterResult(accepted=False, simulated=True, detail="not in guided mode")

        def stop(self, **kwargs):
            return AdapterResult(True, True)

    refusing = Refusing()
    result = Dispatcher(refusing, settings, clock, vocabulary=vocabulary).dispatch(move_msg())
    assert result.outcome.value == "adapter_failed" and "not in guided mode" in result.detail
    assert refusing.calls == 1


# -- no dynamic execution --------------------------------------------------------------

def test_package_has_no_dynamic_code_execution():
    package = Path(__file__).resolve().parent.parent / "drone_voice"
    pattern = re.compile(
        r"(?<![\w.])(eval|exec|getattr|setattr|__import__|compile)\s*\("
        r"|\bimport\s+(importlib|subprocess|ctypes|socket)\b|\bos\.system\b"
    )
    offenders = [f"{p.name}:{n}" for p in package.glob("*.py")
                 for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if pattern.search(line)]
    assert offenders == []
