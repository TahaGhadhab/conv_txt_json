"""Input side: transcription events coming from the speech model.

MODEL INTEGRATION POINT. The real model's output format was not available in
this repository. A model adapter must turn each model output into a
``TranscriptionEvent``:

- ``utterance_id``: identifier of one spoken utterance, stable across partial
  updates and retransmissions of that utterance, unique across utterances
  (including across restarts: prefix it with a session id if the model resets
  its counters).
- ``text``: the model's text, passed through unmodified.
- ``is_final``: True only for the final transcription. If the model has no such
  flag, do not guess: call ``Interpreter.interpret_final`` only once the caller
  knows the utterance is complete (e.g. end-of-speech from the recorder).
- ``confidence``: only if the model provides one; never invent it.
- ``spoken_at``: when the utterance ended (UTC), if the model or recorder knows
  it. It becomes the message ``created_at``, so a late or re-emitted final is
  dated from the speech, not from its processing.

Everything produced from fixtures in this repository uses ``SIMULATED_SOURCE``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .contract import MAX_SOURCE_LENGTH, MAX_UTTERANCE_ID_LENGTH, Confidence

SIMULATED_SOURCE = "simulated-fixture"
MANUAL_SOURCE = "manual-input"


@dataclass(frozen=True)
class TranscriptionEvent:
    utterance_id: str
    text: Any  # deliberately untyped: a non-string is rejected as INVALID_INPUT_TYPE
    is_final: bool
    source: str
    confidence: Confidence | None = None
    spoken_at: datetime | None = None

    def __post_init__(self) -> None:
        # Same limits as the message contract, so a valid event always yields a valid message.
        if (not isinstance(self.utterance_id, str) or not self.utterance_id.strip()
                or len(self.utterance_id) > MAX_UTTERANCE_ID_LENGTH):
            raise ValueError(f"utterance_id must be a non-blank string of at most {MAX_UTTERANCE_ID_LENGTH} characters")
        if type(self.is_final) is not bool:
            raise ValueError("is_final must be a bool")
        if not isinstance(self.source, str) or not self.source.strip() or len(self.source) > MAX_SOURCE_LENGTH:
            raise ValueError(f"source must be a non-blank string of at most {MAX_SOURCE_LENGTH} characters")
        if self.confidence is not None and not isinstance(self.confidence, Confidence):
            raise ValueError("confidence must be a Confidence or None")
        if self.spoken_at is not None and (
                not isinstance(self.spoken_at, datetime) or self.spoken_at.utcoffset() is None):
            raise ValueError("spoken_at must be a timezone-aware datetime or None")
