"""Final transcription -> normalized text -> intent -> validated command message."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from .clock import Clock, truncate_ms
from .contract import (
    CONTRACT_VERSION,
    STATUS_RECOGNIZED,
    STATUS_REJECTED,
    CommandMessage,
    Confidence,
    InterpretationReason,
    derive_command_id,
    format_timestamp,
    validate_message,
)
from .decision_log import DecisionLog
from .normalization import normalize_text
from .transcription import MANUAL_SOURCE, TranscriptionEvent
from .vocabulary import Vocabulary

__all__ = ["Interpreter", "derive_command_id"]

# How many utterances keep their first timestamps (oldest forgotten first).
STAMP_MEMORY = 4096


class Interpreter:
    def __init__(self, vocabulary: Vocabulary, clock: Clock, ttl_s: float, log: DecisionLog | None = None):
        if not ttl_s > 0:
            raise ValueError("ttl_s must be positive")
        self._vocabulary = vocabulary
        self._clock = clock
        self._ttl = timedelta(seconds=ttl_s)
        self._log = log
        # command_id -> (created_at, expires_at) of the first interpretation of that utterance,
        # least recently used first. A re-emitted final keeps them, so it is never "rejuvenated" (QA-04).
        self._stamps: dict[str, tuple[datetime, datetime]] = {}

    def _stamp(self, command_id: str, event: TranscriptionEvent) -> tuple[datetime, datetime]:
        stamp = self._stamps.pop(command_id, None)
        if stamp is None:
            created_at = truncate_ms(event.spoken_at if event.spoken_at is not None else self._clock.now())
            stamp = (created_at, truncate_ms(created_at + self._ttl))
        self._stamps[command_id] = stamp
        if len(self._stamps) > STAMP_MEMORY:
            del self._stamps[next(iter(self._stamps))]
        return stamp

    def process_event(self, event: TranscriptionEvent) -> CommandMessage | None:
        """Return a message for a final transcription, ``None`` for a partial one."""
        if not event.is_final:
            if self._log:
                self._log.record("partial_ignored", format_timestamp(self._clock.now()),
                                 utterance_id=event.utterance_id, source=event.source,
                                 text=event.text if isinstance(event.text, str) else None)
            return None
        return self._interpret(event)

    def interpret_final(
        self,
        text: object,
        *,
        utterance_id: str | None = None,
        source: str = MANUAL_SOURCE,
        confidence: Confidence | None = None,
    ) -> CommandMessage:
        """Entry point for callers that know ``text`` is a FINAL transcription.

        Without ``utterance_id`` a fresh one is generated: each call is then a
        distinct utterance (never deduplicated against another call).
        """
        event = TranscriptionEvent(
            utterance_id=utterance_id or f"utt-{uuid.uuid4()}",
            text=text,
            is_final=True,
            source=source,
            confidence=confidence,
        )
        return self._interpret(event)

    def _interpret(self, event: TranscriptionEvent) -> CommandMessage:
        text = event.text
        recognized_text: str | None
        normalized_text: str | None
        intent = None
        reason: InterpretationReason | None = None
        if not isinstance(text, str):
            recognized_text = normalized_text = None
            reason = InterpretationReason.INVALID_INPUT_TYPE
        else:
            recognized_text = text
            normalized_text = normalize_text(text)
            if not normalized_text:
                reason = InterpretationReason.EMPTY_INPUT
            else:
                intent = self._vocabulary.lookup(normalized_text)
                if intent is None:
                    reason = InterpretationReason.UNKNOWN_EXPRESSION

        command_id = derive_command_id(event.source, event.utterance_id)
        created_at, expires_at = self._stamp(command_id, event)
        raw = {
            "version": CONTRACT_VERSION,
            "command_id": command_id,
            "utterance_id": event.utterance_id,
            "source": event.source,
            "created_at": format_timestamp(created_at),
            "expires_at": format_timestamp(expires_at),
            "recognized_text": recognized_text,
            "normalized_text": normalized_text,
            "intent": intent.value if intent else None,
            "status": STATUS_RECOGNIZED if intent else STATUS_REJECTED,
            "reason": reason.value if reason else None,
            "confidence": event.confidence.to_dict() if event.confidence else None,
        }
        message = validate_message(raw)  # the interpreter never emits an invalid message
        if self._log:
            self._log.record(
                "interpreted",
                format_timestamp(self._clock.now()),
                utterance_id=event.utterance_id,
                command_id=message.command_id,
                created_at=raw["created_at"],
                source=event.source,
                status=message.status,
                intent=raw["intent"],
                reason=raw["reason"],
                input_type=type(text).__name__,
                recognized_text=recognized_text,
                normalized_text=normalized_text,
            )
        return message
