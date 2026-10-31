"""Command dispatcher: validates received messages and calls the controller adapter.

Policies (see docs/CONTRACT.md, "Dispatcher policies"):

- Every received message is validated again, even if it came from the interpreter.
- Order of checks: JSON -> contract -> duplicate -> text/intent verification
  -> validity window -> future date -> expiry -> chronological order
  -> record the utterance -> act.
- Duplicates are detected on ``command_id`` *and* on the utterance
  (``source``, ``utterance_id``): one utterance never produces two effects, even
  if a relay rewrote its ``command_id``.
- Text/intent verification needs the vocabulary. Without one, recognized
  messages are refused (UNVERIFIED_INTENT) unless ``unchecked_text=True`` was
  passed explicitly.
- A message is EXPIRED when ``now >= expires_at``; FUTURE_TIMESTAMP when
  ``created_at > now + max_future_skew_s``.
- A valid message consumes its utterance when it is acted upon, when it is
  ``status: rejected``, and also when it is refused for a time or order reason
  (window, future, expired, out of order): a retransmission of that utterance can
  never execute later. The utterance is recorded *before* the adapter call.
- Concurrency: no queue. A new MOVE while moving replaces the active movement
  (latest command wins). STOP is handled immediately and is never ordered
  behind a movement.
- A movement ends when ``now >= started + duration_s``: ``poll()`` (also run at
  the start of every ``dispatch``) sends ``stop(cause=duration_elapsed)``.
- MOVE calls are never retried. If a MOVE fails, the movement may still have
  started, so a stop is still scheduled at its planned end. A failed STOP is
  retried on every ``poll()`` while a movement is tracked (stop is idempotent),
  keeping its cause ``stop_command``.
- Chronological order: a recognized MOVE is OUT_OF_ORDER if it was created
  before the newest accepted recognized command, or not strictly after the
  newest accepted STOP. A STOP is accepted even out of order, since stopping is
  always the safe action.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import Enum
from types import MappingProxyType
from typing import Any

from .clock import Clock
from .contract import (
    CommandMessage,
    ContractError,
    ValidationCode,
    derive_command_id,
    format_timestamp,
    validate_message,
)
from .controller import AdapterResult, Direction, DroneController, StopCause
from .decision_log import DecisionLog
from .dedup import DedupStore, InMemoryDedupStore
from .intents import Intent
from .jsonio import JsonFormatError, loads_strict
from .settings import Settings
from .vocabulary import Vocabulary

# Closed mapping: the only way an intent reaches the adapter.
_MOVE_DIRECTIONS = MappingProxyType({
    Intent.MOVE_FORWARD: Direction.FORWARD,
    Intent.MOVE_BACKWARD: Direction.BACKWARD,
    Intent.MOVE_RIGHT: Direction.RIGHT,
    Intent.MOVE_LEFT: Direction.LEFT,
    Intent.MOVE_UP: Direction.UP,
    Intent.MOVE_DOWN: Direction.DOWN,
})


class Outcome(str, Enum):
    REJECTED = "rejected"                  # refused by validation: adapter not called
    NOT_ACTIONABLE = "not_actionable"      # valid message with status "rejected": adapter not called
    ADAPTER_ACCEPTED = "adapter_accepted"  # adapter accepted the call (see ``simulated``)
    ADAPTER_FAILED = "adapter_failed"      # adapter raised or refused: never reported as success


@dataclass(frozen=True)
class DispatchResult:
    kind: str  # "command", "auto_stop" (end of duration) or "stop_retry" (retry of a failed STOP)
    outcome: Outcome
    command_id: str | None
    intent: str | None  # recognized intent of the message; None for auto_stop / stop_retry
    reason: str | None
    transmitted: bool
    simulated: bool | None
    detail: str
    at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "outcome": self.outcome.value,
            "command_id": self.command_id,
            "intent": self.intent,
            "reason": self.reason,
            "transmitted": self.transmitted,
            "simulated": self.simulated,
            "detail": self.detail,
            "at": self.at,
        }


@dataclass(frozen=True)
class ActiveMovement:
    command_id: str
    intent: Intent
    ends_at: datetime  # when poll() must send the pending stop
    confirmed: bool  # False when the adapter failed: the movement may or may not have started
    stop_cause: StopCause = StopCause.DURATION_ELAPSED
    stop_command_id: str | None = None  # id of a failed STOP command being retried


class Dispatcher:
    def __init__(
        self,
        controller: DroneController,
        settings: Settings,
        clock: Clock,
        *,
        dedup_store: DedupStore | None = None,
        vocabulary: Vocabulary | None = None,
        unchecked_text: bool = False,
        log: DecisionLog | None = None,
    ):
        if vocabulary is not None and unchecked_text:
            raise ValueError("unchecked_text=True contradicts a provided vocabulary")
        self._controller = controller
        self._settings = settings
        self._clock = clock
        self._store = dedup_store if dedup_store is not None else InMemoryDedupStore()
        self._vocabulary = vocabulary
        self._unchecked_text = unchecked_text
        self._log = log
        self._lock = threading.Lock()
        self._active: ActiveMovement | None = None
        self._last_created: datetime | None = None
        self._last_stop_created: datetime | None = None
        self.history: list[DispatchResult] = []

    @property
    def active_movement(self) -> ActiveMovement | None:
        return self._active

    def dispatch_json(self, raw: str | bytes) -> DispatchResult:
        try:
            obj = loads_strict(raw)
        except JsonFormatError as exc:
            with self._lock:
                self._poll_locked()
                return self._reject(None, ValidationCode.MALFORMED_JSON, str(exc), self._clock.now())
        return self.dispatch(obj)

    def dispatch(self, obj: Any) -> DispatchResult:
        with self._lock:
            self._poll_locked()
            now = self._clock.now()
            try:
                msg = validate_message(obj)
            except ContractError as exc:
                return self._reject(_unverified_id(obj), exc.code, exc.detail, now)

            keys = _dedup_keys(msg)
            if any(key in self._store for key in keys):
                return self._reject(msg.command_id, ValidationCode.DUPLICATE,
                                    "command_id or utterance already processed", now)
            if msg.is_recognized:
                refusal = self._verify_intent(msg)
                if refusal is not None:
                    return self._reject(msg.command_id, *refusal, now)

            refusal = self._temporal_refusal(msg, now)
            if refusal is not None:
                code, detail = refusal
                store_error = self._consume(keys, now)
                if store_error is not None:
                    detail += f"; utterance could not be recorded ({store_error})"
                return self._reject(msg.command_id, code, detail, now)

            store_error = self._consume(keys, now)
            if store_error is not None:  # cannot guarantee deduplication -> do not act
                return self._reject(msg.command_id, ValidationCode.STATE_STORE_ERROR, store_error, now)

            if not msg.is_recognized:
                return self._finish("command", Outcome.NOT_ACTIONABLE, msg.command_id, None,
                                    msg.reason.value if msg.reason else None, False, None,
                                    "interpreter rejected the transcription; nothing sent", now)
            if self._last_created is None or msg.created_at > self._last_created:
                self._last_created = msg.created_at
            if msg.intent is Intent.STOP:
                if self._last_stop_created is None or msg.created_at > self._last_stop_created:
                    self._last_stop_created = msg.created_at
                return self._stop(msg, now)
            return self._move(msg, now)

    def poll(self) -> list[DispatchResult]:
        """Send the pending stop if the active movement is due. Call this regularly."""
        with self._lock:
            return self._poll_locked()

    # -- checks --------------------------------------------------------------------

    def _verify_intent(self, msg: CommandMessage) -> tuple[ValidationCode, str] | None:
        if self._vocabulary is None:
            if self._unchecked_text:
                return None
            return (ValidationCode.UNVERIFIED_INTENT,
                    "no vocabulary configured to verify the intent (pass unchecked_text=True to skip explicitly)")
        if self._vocabulary.lookup(msg.normalized_text or "") is not msg.intent:
            return (ValidationCode.TEXT_INTENT_MISMATCH,
                    "normalized_text does not map to this intent in the vocabulary")
        return None

    def _temporal_refusal(self, msg: CommandMessage, now: datetime) -> tuple[ValidationCode, str] | None:
        ttl = msg.expires_at - msg.created_at
        if ttl > timedelta(seconds=self._settings.max_ttl_s):
            return (ValidationCode.INVALID_TIMESTAMP,
                    f"validity window {ttl.total_seconds()} s exceeds {self._settings.max_ttl_s} s")
        if msg.created_at > now + timedelta(seconds=self._settings.max_future_skew_s):
            return ValidationCode.FUTURE_TIMESTAMP, "created_at is in the future beyond the allowed clock skew"
        if now >= msg.expires_at:
            return ValidationCode.EXPIRED, "message expired"
        if msg.is_recognized and msg.intent is not Intent.STOP:
            if self._last_stop_created is not None and msg.created_at <= self._last_stop_created:
                return ValidationCode.OUT_OF_ORDER, "not newer than the latest accepted STOP"
            if self._last_created is not None and msg.created_at < self._last_created:
                return ValidationCode.OUT_OF_ORDER, "older than the newest accepted command"
        return None

    def _consume(self, keys: tuple[str, ...], now: datetime) -> str | None:
        try:
            for key in keys:
                if key not in self._store:
                    self._store.add(key, format_timestamp(now))
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        return None

    # -- adapter calls -------------------------------------------------------------

    def _call(self, fn, **kwargs) -> tuple[AdapterResult | None, str | None]:
        try:
            result = fn(**kwargs)
        except Exception as exc:  # adapter errors are reported, never swallowed as success
            return None, f"{type(exc).__name__}: {exc}"
        if not isinstance(result, AdapterResult):
            return None, f"adapter returned {type(result).__name__}, expected AdapterResult"
        if not result.accepted:
            return result, f"adapter refused: {result.detail}"
        return result, None

    def _move(self, msg: CommandMessage, now: datetime) -> DispatchResult:
        assert msg.intent is not None
        direction = _MOVE_DIRECTIONS[msg.intent]
        profile = self._settings.motion[msg.intent]
        previous = self._active
        ends_at = now + timedelta(seconds=profile.duration_s)
        result, error = self._call(self._controller.move, command_id=msg.command_id, direction=direction,
                                   speed_m_s=profile.speed_m_s, duration_s=profile.duration_s)
        if error is not None:
            guard_end = max(ends_at, previous.ends_at) if previous else ends_at
            self._active = ActiveMovement(msg.command_id, msg.intent, guard_end, confirmed=False)
            return self._finish("command", Outcome.ADAPTER_FAILED, msg.command_id, msg.intent.value,
                                "ADAPTER_ERROR", True, result.simulated if result else None,
                                f"{error}; not retried; safety stop scheduled at {format_timestamp(guard_end)}", now)
        self._active = ActiveMovement(msg.command_id, msg.intent, ends_at, confirmed=True)
        detail = f"{result.detail}; ends at {format_timestamp(ends_at)}"
        if previous is not None:
            detail += f"; replaces {previous.command_id}"
        return self._finish("command", Outcome.ADAPTER_ACCEPTED, msg.command_id, msg.intent.value, None,
                            True, result.simulated, detail, now)

    def _stop(self, msg: CommandMessage, now: datetime) -> DispatchResult:
        previous = self._active
        result, error = self._call(self._controller.stop, command_id=msg.command_id, cause=StopCause.STOP_COMMAND)
        if error is not None:
            if previous is not None:  # retried at the next poll with the same cause; stop is idempotent
                self._active = replace(previous, ends_at=now, stop_cause=StopCause.STOP_COMMAND,
                                       stop_command_id=msg.command_id)
            return self._finish("command", Outcome.ADAPTER_FAILED, msg.command_id, Intent.STOP.value,
                                "ADAPTER_ERROR", True, result.simulated if result else None,
                                f"{error}; stop will be retried on next poll" if previous else error, now)
        self._active = None
        detail = result.detail + (f"; interrupted {previous.command_id}" if previous else "; no active movement")
        return self._finish("command", Outcome.ADAPTER_ACCEPTED, msg.command_id, Intent.STOP.value, None,
                            True, result.simulated, detail, now)

    def _poll_locked(self) -> list[DispatchResult]:
        active = self._active
        if active is None:
            return []
        now = self._clock.now()
        if now < active.ends_at:
            return []
        retry = active.stop_cause is StopCause.STOP_COMMAND
        kind = "stop_retry" if retry else "auto_stop"
        command_id = active.stop_command_id or active.command_id
        result, error = self._call(self._controller.stop, command_id=command_id, cause=active.stop_cause)
        if error is not None:
            return [self._finish(kind, Outcome.ADAPTER_FAILED, command_id, None, "ADAPTER_ERROR", True,
                                 result.simulated if result else None, f"{error}; will retry on next poll", now)]
        self._active = None
        what = "STOP retried" if retry else f"movement {active.intent.value} ended"
        return [self._finish(kind, Outcome.ADAPTER_ACCEPTED, command_id, None, None, True, result.simulated,
                             f"{result.detail}; {what} ({active.command_id})", now)]

    def _reject(self, command_id: str | None, code: ValidationCode, detail: str, now: datetime) -> DispatchResult:
        return self._finish("command", Outcome.REJECTED, command_id, None, code.value, False, None, detail, now)

    def _finish(self, kind: str, outcome: Outcome, command_id: str | None, intent: str | None,
                reason: str | None, transmitted: bool, simulated: bool | None, detail: str,
                now: datetime) -> DispatchResult:
        result = DispatchResult(kind, outcome, command_id, intent, reason, transmitted, simulated, detail,
                                format_timestamp(now))
        self.history.append(result)
        if self._log:
            self._log.record("dispatched", result.at, **{k: v for k, v in result.to_dict().items() if k != "at"})
        return result


def _dedup_keys(msg: CommandMessage) -> tuple[str, ...]:
    """The command_id, plus the utterance key when a relay changed the command_id (QA-07)."""
    utterance_key = derive_command_id(msg.source, msg.utterance_id)
    return (msg.command_id,) if utterance_key == msg.command_id else (msg.command_id, utterance_key)


def _unverified_id(obj: Any) -> str | None:
    """command_id of an invalid message, for diagnostics only (it was not validated)."""
    if isinstance(obj, dict):
        value = obj.get("command_id")
        if isinstance(value, str) and len(value) <= 128:
            return value
    return None
