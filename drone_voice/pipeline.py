"""Wiring of the full chain. The message crosses a real JSON (UTF-8) boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .clock import Clock
from .contract import CommandMessage
from .controller import DroneController, SimulatedController
from .decision_log import DecisionLog
from .dedup import DedupStore
from .dispatcher import Dispatcher, DispatchResult
from .interpreter import Interpreter
from .jsonio import dumps
from .settings import Settings, load_settings
from .transcription import TranscriptionEvent
from .vocabulary import Vocabulary, load_vocabulary

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VOCABULARY_PATH = REPO_ROOT / "config" / "vocabulary.json"
DEFAULT_SETTINGS_PATH = REPO_ROOT / "config" / "settings.json"


@dataclass
class Pipeline:
    interpreter: Interpreter
    dispatcher: Dispatcher
    controller: DroneController

    def handle_event(self, event: TranscriptionEvent) -> tuple[CommandMessage | None, DispatchResult | None]:
        """Partial transcription -> (None, None): nothing is produced or dispatched."""
        message = self.interpreter.process_event(event)
        if message is None:
            return None, None
        wire = dumps(message.to_dict()).encode("utf-8")
        return message, self.dispatcher.dispatch_json(wire)


def build_pipeline(
    clock: Clock,
    *,
    vocabulary: Vocabulary | None = None,
    settings: Settings | None = None,
    controller: DroneController | None = None,
    dedup_store: DedupStore | None = None,
    log: DecisionLog | None = None,
) -> Pipeline:
    vocabulary = vocabulary or load_vocabulary(DEFAULT_VOCABULARY_PATH)
    settings = settings or load_settings(DEFAULT_SETTINGS_PATH)
    controller = controller or SimulatedController(clock)
    interpreter = Interpreter(vocabulary, clock, settings.ttl_s, log)
    dispatcher = Dispatcher(controller, settings, clock, dedup_store=dedup_store, vocabulary=vocabulary, log=log)
    return Pipeline(interpreter, dispatcher, controller)
