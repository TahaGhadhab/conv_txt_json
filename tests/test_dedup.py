import pytest

from conftest import make_message
from drone_voice.controller import SimulatedController
from drone_voice.dedup import DedupStoreError, InMemoryDedupStore, JsonlDedupStore
from drone_voice.dispatcher import Dispatcher


def test_persistent_store_survives_restart(tmp_path, settings, clock, vocabulary):
    path = tmp_path / "state" / "ids.jsonl"
    first = SimulatedController(clock)
    assert Dispatcher(first, settings, clock, dedup_store=JsonlDedupStore(path), vocabulary=vocabulary) \
        .dispatch(make_message()).outcome.value == "adapter_accepted"
    restarted = SimulatedController(clock)
    result = Dispatcher(restarted, settings, clock, dedup_store=JsonlDedupStore(path), vocabulary=vocabulary) \
        .dispatch(make_message())
    assert result.reason == "DUPLICATE"
    assert restarted.calls == []


def test_in_memory_store_forgets_on_restart_documented_limit(settings, clock, vocabulary):
    """Documented limit: without a persistent store, a still-valid retransmission
    received after a restart is executed again. Only the TTL bounds this window."""
    for _ in range(2):
        controller = SimulatedController(clock)
        Dispatcher(controller, settings, clock, dedup_store=InMemoryDedupStore(), vocabulary=vocabulary) \
            .dispatch(make_message())
        assert len(controller.calls) == 1


@pytest.mark.parametrize("content", [b"{not json}\n", b'{"other": 1}\n', b'{"command_id": "a"}\n{"command_'])
def test_corrupt_store_fails_closed(tmp_path, content):
    path = tmp_path / "ids.jsonl"
    path.write_bytes(content)
    with pytest.raises(DedupStoreError):
        JsonlDedupStore(path)
