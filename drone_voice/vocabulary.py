"""Configurable vocabulary: normalized full phrase -> intent (exact match only)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable

from .intents import Intent
from .jsonio import JsonFormatError, read_json_file
from .normalization import normalize_text

VOCABULARY_FORMAT_VERSION = 1
_TOP_LEVEL_KEYS = {"format_version", "status", "note", "entries"}
_ENTRY_KEYS = {"expression", "intent", "note"}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class VocabularyEntry:
    expression: str
    normalized: str
    intent: Intent


class Vocabulary:
    """Maps the *whole* normalized sentence to one intent.

    There is no substring search, no fuzzy matching and no segmentation: a
    sentence that is not exactly one configured expression has no intent.
    """

    def __init__(self, entries: Iterable[tuple[str, Intent | str]]):
        mapping: dict[str, Intent] = {}
        kept: list[VocabularyEntry] = []
        conflicts: list[str] = []
        for expression, intent_value in entries:
            if not isinstance(expression, str):
                raise ConfigError(f"expression must be a string, got {expression!r}")
            try:
                intent = Intent(intent_value)
            except ValueError:
                raise ConfigError(f"unknown intent {intent_value!r} for {expression!r}") from None
            normalized = normalize_text(expression)
            if not normalized:
                raise ConfigError(f"expression {expression!r} is empty after normalization")
            existing = mapping.get(normalized)
            if existing is not None and existing is not intent:
                conflicts.append(
                    f"{expression!r} normalizes to {normalized!r}, already mapped to "
                    f"{existing.value}, cannot also map to {intent.value}"
                )
                continue
            mapping[normalized] = intent
            kept.append(VocabularyEntry(expression, normalized, intent))
        if conflicts:
            raise ConfigError("ambiguous vocabulary: " + "; ".join(conflicts))
        missing = [i.value for i in Intent if i not in mapping.values()]
        if missing:
            raise ConfigError(f"no expression configured for intents: {', '.join(missing)}")
        self._mapping = MappingProxyType(mapping)
        self.entries: tuple[VocabularyEntry, ...] = tuple(kept)

    def lookup(self, normalized_text: str) -> Intent | None:
        return self._mapping.get(normalized_text)


def vocabulary_from_dict(data: Any) -> Vocabulary:
    if not isinstance(data, dict):
        raise ConfigError("vocabulary file must contain a JSON object")
    unknown = set(data) - _TOP_LEVEL_KEYS
    if unknown:
        raise ConfigError(f"unknown vocabulary keys: {sorted(unknown)}")
    if data.get("format_version") != VOCABULARY_FORMAT_VERSION or type(data.get("format_version")) is not int:
        raise ConfigError(f"vocabulary format_version must be {VOCABULARY_FORMAT_VERSION}")
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ConfigError("vocabulary 'entries' must be a non-empty list")
    pairs: list[tuple[str, str]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ConfigError(f"entry {index} must be an object")
        unknown = set(entry) - _ENTRY_KEYS
        if unknown or "expression" not in entry or "intent" not in entry:
            raise ConfigError(
                f"entry {index} must have 'expression' and 'intent' (optional 'note'), got {sorted(entry)}"
            )
        pairs.append((entry["expression"], entry["intent"]))
    return Vocabulary(pairs)


def load_vocabulary(path: str | Path) -> Vocabulary:
    try:
        data = read_json_file(path)
    except (OSError, JsonFormatError) as exc:
        raise ConfigError(f"cannot read vocabulary {path}: {exc}") from exc
    return vocabulary_from_dict(data)
