"""Versioned command message contract (version 1), validated at runtime.

See docs/CONTRACT.md for the human-readable specification. ``validate_message``
is the single source of truth: the interpreter validates what it emits and the
dispatcher validates what it receives with the same function.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .intents import Intent
from .normalization import normalize_text

CONTRACT_VERSION = 1

STATUS_RECOGNIZED = "recognized"
STATUS_REJECTED = "rejected"

MESSAGE_FIELDS = (
    "version",
    "command_id",
    "utterance_id",
    "source",
    "created_at",
    "expires_at",
    "recognized_text",
    "normalized_text",
    "intent",
    "status",
    "reason",
    "confidence",
)

MAX_UTTERANCE_ID_LENGTH = 256
MAX_SOURCE_LENGTH = 64

COMMAND_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "urn:drone-voice-commands:command-id:v1")


def derive_command_id(source: str, utterance_id: str) -> str:
    """Same (source, utterance) -> same id, so retransmissions and repeated finals deduplicate."""
    return f"cmd-{uuid.uuid5(COMMAND_ID_NAMESPACE, source + chr(0x1F) + utterance_id)}"


# ASCII only and matched with fullmatch: no trailing newline, no non-ASCII digits.
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", re.ASCII)
_TS_RE = re.compile(r"([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})(?:\.([0-9]{1,6}))?Z", re.ASCII)


class InterpretationReason(str, Enum):
    """Why the interpreter produced ``status: "rejected"`` (field ``reason``)."""

    EMPTY_INPUT = "EMPTY_INPUT"
    INVALID_INPUT_TYPE = "INVALID_INPUT_TYPE"
    UNKNOWN_EXPRESSION = "UNKNOWN_EXPRESSION"


class ValidationCode(str, Enum):
    """Why a receiver (the dispatcher) refused a message. Never written into the message."""

    MALFORMED_JSON = "MALFORMED_JSON"
    INVALID_STRUCTURE = "INVALID_STRUCTURE"
    UNSUPPORTED_VERSION = "UNSUPPORTED_VERSION"
    UNKNOWN_INTENT = "UNKNOWN_INTENT"
    INCONSISTENT_MESSAGE = "INCONSISTENT_MESSAGE"
    INVALID_TIMESTAMP = "INVALID_TIMESTAMP"
    FUTURE_TIMESTAMP = "FUTURE_TIMESTAMP"
    EXPIRED = "EXPIRED"
    DUPLICATE = "DUPLICATE"
    OUT_OF_ORDER = "OUT_OF_ORDER"
    TEXT_INTENT_MISMATCH = "TEXT_INTENT_MISMATCH"
    UNVERIFIED_INTENT = "UNVERIFIED_INTENT"
    STATE_STORE_ERROR = "STATE_STORE_ERROR"


class ContractError(ValueError):
    def __init__(self, code: ValidationCode, detail: str):
        super().__init__(f"{code.value}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class Confidence:
    """A score reported by the speech model, kept as-is with its provider.

    It is not assumed to be a probability and is never used to accept or
    reject a command in this version.
    """

    value: float
    provider: str

    def __post_init__(self) -> None:
        if type(self.value) not in (int, float) or not math.isfinite(self.value):
            raise ValueError("confidence value must be a finite number")
        if not isinstance(self.provider, str) or not self.provider.strip() or len(self.provider) > 64:
            raise ValueError("confidence provider must be a non-blank string of at most 64 characters")

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "provider": self.provider}


@dataclass(frozen=True)
class CommandMessage:
    command_id: str
    utterance_id: str
    source: str
    created_at: datetime
    expires_at: datetime
    recognized_text: str | None
    normalized_text: str | None
    intent: Intent | None
    status: str
    reason: InterpretationReason | None
    confidence: Confidence | None
    version: int = CONTRACT_VERSION

    @property
    def is_recognized(self) -> bool:
        return self.status == STATUS_RECOGNIZED

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "command_id": self.command_id,
            "utterance_id": self.utterance_id,
            "source": self.source,
            "created_at": format_timestamp(self.created_at),
            "expires_at": format_timestamp(self.expires_at),
            "recognized_text": self.recognized_text,
            "normalized_text": self.normalized_text,
            "intent": self.intent.value if self.intent else None,
            "status": self.status,
            "reason": self.reason.value if self.reason else None,
            "confidence": self.confidence.to_dict() if self.confidence else None,
        }


def format_timestamp(dt: datetime) -> str:
    """UTC, millisecond precision, explicit ``Z`` suffix: ``2026-01-01T12:00:00.000Z``."""
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    dt = dt.astimezone(timezone.utc)
    return f"{dt:%Y-%m-%dT%H:%M:%S}.{dt.microsecond // 1000:03d}Z"


def parse_timestamp(value: Any, field: str = "timestamp") -> datetime:
    if not isinstance(value, str):
        raise ContractError(ValidationCode.INVALID_TIMESTAMP, f"{field} must be a string")
    match = _TS_RE.fullmatch(value)
    if not match:
        raise ContractError(
            ValidationCode.INVALID_TIMESTAMP,
            f"{field}={value!r} is not UTC ISO 8601 'YYYY-MM-DDTHH:MM:SS[.ffffff]Z'",
        )
    year, month, day, hour, minute, second, frac = match.groups()
    micro = int((frac or "0").ljust(6, "0"))
    try:
        return datetime(int(year), int(month), int(day), int(hour), int(minute), int(second), micro,
                        tzinfo=timezone.utc)
    except ValueError as exc:
        raise ContractError(ValidationCode.INVALID_TIMESTAMP, f"{field}={value!r}: {exc}") from None


def _structure(detail: str) -> ContractError:
    return ContractError(ValidationCode.INVALID_STRUCTURE, detail)


def _inconsistent(detail: str) -> ContractError:
    return ContractError(ValidationCode.INCONSISTENT_MESSAGE, detail)


def _identifier(obj: dict[str, Any], field: str) -> str:
    value = obj[field]
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise _structure(f"{field} must match {_ID_RE.pattern}")
    return value


def _label(obj: dict[str, Any], field: str, max_len: int) -> str:
    value = obj[field]
    if not isinstance(value, str) or not value.strip() or len(value) > max_len:
        raise _structure(f"{field} must be a non-blank string of at most {max_len} characters")
    return value


def _optional_str(obj: dict[str, Any], field: str) -> str | None:
    value = obj[field]
    if value is not None and not isinstance(value, str):
        raise _structure(f"{field} must be a string or null")
    return value


def _confidence(value: Any) -> Confidence | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"value", "provider"}:
        raise _structure("confidence must be null or an object with exactly 'value' and 'provider'")
    score = value["value"]
    if type(score) not in (int, float) or not math.isfinite(score):
        raise _structure("confidence.value must be a finite number")
    provider = value["provider"]
    if not isinstance(provider, str) or not provider.strip() or len(provider) > 64:
        raise _structure("confidence.provider must be a non-blank string of at most 64 characters")
    return Confidence(value=score, provider=provider)


def validate_message(obj: Any) -> CommandMessage:
    """Validate a decoded JSON message; return it typed or raise ``ContractError``.

    Policy for unknown fields: rejected (``INVALID_STRUCTURE``). Every field of
    ``MESSAGE_FIELDS`` is required, ``null`` where allowed.
    """
    if not isinstance(obj, dict):
        raise _structure("message must be a JSON object")
    if "version" not in obj:
        raise _structure("missing field 'version'")
    if type(obj["version"]) is not int:
        raise _structure("version must be an integer")
    if obj["version"] != CONTRACT_VERSION:
        raise ContractError(ValidationCode.UNSUPPORTED_VERSION, f"version {obj['version']} is not supported")
    missing = [f for f in MESSAGE_FIELDS if f not in obj]
    if missing:
        raise _structure(f"missing fields {missing}")
    extra = sorted(set(obj) - set(MESSAGE_FIELDS))
    if extra:
        raise _structure(f"unknown fields {extra} (unknown fields are rejected)")

    command_id = _identifier(obj, "command_id")
    utterance_id = _label(obj, "utterance_id", MAX_UTTERANCE_ID_LENGTH)
    source = _label(obj, "source", MAX_SOURCE_LENGTH)
    created_at = parse_timestamp(obj["created_at"], "created_at")
    expires_at = parse_timestamp(obj["expires_at"], "expires_at")
    if expires_at <= created_at:
        raise ContractError(ValidationCode.INVALID_TIMESTAMP, "expires_at must be after created_at")

    status = obj["status"]
    if status not in (STATUS_RECOGNIZED, STATUS_REJECTED):
        raise _structure(f"status must be '{STATUS_RECOGNIZED}' or '{STATUS_REJECTED}'")
    recognized_text = _optional_str(obj, "recognized_text")
    normalized_text = _optional_str(obj, "normalized_text")
    intent_raw = _optional_str(obj, "intent")
    reason_raw = _optional_str(obj, "reason")
    confidence = _confidence(obj["confidence"])

    intent: Intent | None = None
    reason: InterpretationReason | None = None
    if status == STATUS_RECOGNIZED:
        if intent_raw is None:
            raise _inconsistent("a recognized message must have an intent")
        try:
            intent = Intent(intent_raw)
        except ValueError:
            raise ContractError(ValidationCode.UNKNOWN_INTENT, f"unknown intent {intent_raw!r}") from None
        if reason_raw is not None:
            raise _inconsistent("a recognized message must have reason null")
        if recognized_text is None or normalized_text is None:
            raise _inconsistent("a recognized message must carry recognized_text and normalized_text")
        if not normalized_text:
            raise _inconsistent("a recognized message cannot have an empty normalized_text")
        if normalize_text(recognized_text) != normalized_text:
            raise _inconsistent("normalized_text is not the normalization of recognized_text")
    else:
        if intent_raw is not None:
            raise _inconsistent("a rejected message must have intent null")
        if reason_raw is None:
            raise _inconsistent("a rejected message must have a reason")
        try:
            reason = InterpretationReason(reason_raw)
        except ValueError:
            raise _structure(f"unknown reason {reason_raw!r}") from None
        if reason is InterpretationReason.INVALID_INPUT_TYPE:
            if recognized_text is not None or normalized_text is not None:
                raise _inconsistent("INVALID_INPUT_TYPE requires null recognized_text and normalized_text")
        else:
            if recognized_text is None or normalized_text is None:
                raise _inconsistent(f"{reason.value} requires recognized_text and normalized_text")
            if normalize_text(recognized_text) != normalized_text:
                raise _inconsistent("normalized_text is not the normalization of recognized_text")
            if reason is InterpretationReason.EMPTY_INPUT and normalized_text != "":
                raise _inconsistent("EMPTY_INPUT requires an empty normalized_text")
            if reason is InterpretationReason.UNKNOWN_EXPRESSION and normalized_text == "":
                raise _inconsistent("UNKNOWN_EXPRESSION requires a non-empty normalized_text")

    return CommandMessage(
        command_id=command_id,
        utterance_id=utterance_id,
        source=source,
        created_at=created_at,
        expires_at=expires_at,
        recognized_text=recognized_text,
        normalized_text=normalized_text,
        intent=intent,
        status=status,
        reason=reason,
        confidence=confidence,
    )
