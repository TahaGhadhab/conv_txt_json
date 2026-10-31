from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from drone_voice.clock import ManualClock
from drone_voice.controller import SimulatedController
from drone_voice.dispatcher import Dispatcher
from drone_voice.interpreter import derive_command_id
from drone_voice.pipeline import DEFAULT_SETTINGS_PATH, DEFAULT_VOCABULARY_PATH, build_pipeline
from drone_voice.settings import load_settings
from drone_voice.transcription import TranscriptionEvent
from drone_voice.vocabulary import load_vocabulary

START = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def ts(seconds: float = 0.0) -> str:
    dt = START + timedelta(seconds=seconds)
    return f"{dt:%Y-%m-%dT%H:%M:%S}.{dt.microsecond // 1000:03d}Z"


def final(utterance_id: str, text: Any, source: str = "test") -> TranscriptionEvent:
    return TranscriptionEvent(utterance_id=utterance_id, text=text, is_final=True, source=source)


def make_message(**overrides: Any) -> dict[str, Any]:
    """A valid recognized MOVE_FORWARD message created at START (2 s validity)."""
    utterance = overrides.pop("utterance", "u-1")
    message = {
        "version": 1,
        "command_id": derive_command_id("test", utterance),
        "utterance_id": utterance,
        "source": "test",
        "created_at": ts(0),
        "expires_at": ts(2),
        "recognized_text": "تقدّم",
        "normalized_text": "تقدم",
        "intent": "MOVE_FORWARD",
        "status": "recognized",
        "reason": None,
        "confidence": None,
    }
    message.update(overrides)
    return message


@pytest.fixture
def vocabulary():
    return load_vocabulary(DEFAULT_VOCABULARY_PATH)


@pytest.fixture
def settings():
    return load_settings(DEFAULT_SETTINGS_PATH)


@pytest.fixture
def clock():
    return ManualClock(START)


@pytest.fixture
def controller(clock):
    return SimulatedController(clock)


@pytest.fixture
def dispatcher(controller, settings, clock, vocabulary):
    return Dispatcher(controller, settings, clock, vocabulary=vocabulary)


@pytest.fixture
def pipeline(clock, vocabulary, settings, controller):
    return build_pipeline(clock, vocabulary=vocabulary, settings=settings, controller=controller)
