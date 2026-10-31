"""Structured decision log (JSON Lines). No audio, no secrets.

Transcription text can be dropped with ``include_text=False``; the
``utterance_id`` and ``command_id`` still link transcription, decision and result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .jsonio import dumps

TEXT_FIELDS = frozenset({"text", "recognized_text", "normalized_text"})


class DecisionLog:
    def __init__(self, path: str | Path | None = None, *, include_text: bool = True):
        self.path = Path(path) if path is not None else None
        self.include_text = include_text
        self.entries: list[dict[str, Any]] = []

    def record(self, event: str, at: str, **fields: Any) -> None:
        entry: dict[str, Any] = {"event": event, "at": at}
        for key, value in fields.items():
            if key in TEXT_FIELDS and not self.include_text:
                entry["text_redacted"] = True
                continue
            entry[key] = value
        self.entries.append(entry)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(dumps(entry, indent=None))
                fh.write("\n")
