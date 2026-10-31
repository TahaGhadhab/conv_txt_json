"""Final transcription -> JSON -> dispatcher -> simulated adapter, through the public pipeline."""

from conftest import final
from drone_voice.transcription import TranscriptionEvent


def test_final_transcription_reaches_the_simulated_adapter(pipeline, controller, clock, settings):
    message, result = pipeline.handle_event(final("utt-1", "امشي لليسار"))
    assert message.to_dict()["intent"] == "MOVE_LEFT"
    assert (result.outcome.value, result.command_id) == ("adapter_accepted", message.command_id)
    profile = settings.motion[message.intent]
    assert [(c["method"], c["direction"], c["speed_m_s"], c["duration_s"], c["command_id"])
            for c in controller.calls] == [("move", "LEFT", profile.speed_m_s, profile.duration_s, message.command_id)]
    clock.advance(profile.duration_s)
    pipeline.dispatcher.poll()
    assert [c["method"] for c in controller.calls] == ["move", "stop"]


def test_partial_then_final_then_update_executes_once(pipeline, controller):
    partial = TranscriptionEvent(utterance_id="utt-1", text="تقدّ", is_final=False, source="test")
    assert pipeline.handle_event(partial) == (None, None)
    partial_full = TranscriptionEvent(utterance_id="utt-1", text="تقدّم", is_final=False, source="test")
    assert pipeline.handle_event(partial_full) == (None, None)
    assert controller.calls == []
    _, result = pipeline.handle_event(final("utt-1", "تقدّم"))
    assert result.outcome.value == "adapter_accepted"
    _, update = pipeline.handle_event(final("utt-1", "تقدّم."))  # a re-sent / revised final
    assert update.reason == "DUPLICATE"
    assert len(controller.calls) == 1


def test_two_distinct_utterances_with_same_text_are_both_executed(pipeline, controller):
    pipeline.handle_event(final("utt-1", "اطلع"))
    pipeline.handle_event(final("utt-2", "اطلع"))
    assert [(c["method"], c["direction"]) for c in controller.calls] == [("move", "UP"), ("move", "UP")]


def test_rejected_inputs_never_reach_the_adapter(pipeline, controller):
    for i, text in enumerate(["", "   ", None, 7, ["وقف"], "ما تتقدّمش", "قلتلك تقدّم", "تقدّم و اطلع", "طير"]):
        message, result = pipeline.handle_event(final(f"utt-{i}", text))
        assert message.status == "rejected" and message.intent is None
        assert (result.outcome.value, result.transmitted) == ("not_actionable", False)
    assert controller.calls == []


def test_stop_during_movement(pipeline, controller, clock):
    pipeline.handle_event(final("utt-1", "ارجع"))
    clock.advance(0.4)
    _, result = pipeline.handle_event(final("utt-2", "وقف"))
    assert "interrupted" in result.detail
    assert [(c["method"], c.get("direction") or c.get("cause")) for c in controller.calls] == [
        ("move", "BACKWARD"), ("stop", "stop_command")]


def test_decision_log_links_transcription_decision_and_result(clock, vocabulary, settings, tmp_path):
    from drone_voice.decision_log import DecisionLog
    from drone_voice.pipeline import build_pipeline

    for include_text in (True, False):
        path = tmp_path / f"log-{include_text}.jsonl"
        log = DecisionLog(path, include_text=include_text)
        p = build_pipeline(clock, vocabulary=vocabulary, settings=settings, log=log)
        p.handle_event(TranscriptionEvent(utterance_id="u", text="تقد", is_final=False, source="test"))
        message, _ = p.handle_event(final("u", "تقدّم"))
        events = [(e["event"], e.get("command_id")) for e in log.entries]
        assert events == [("partial_ignored", None), ("interpreted", message.command_id),
                          ("dispatched", message.command_id)]
        content = path.read_text(encoding="utf-8")
        assert ("تقدّم" in content) is include_text
        assert ("تقد" in content) is include_text
