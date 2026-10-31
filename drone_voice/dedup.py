"""Stores of command ids already consumed by the dispatcher.

Guarantee: at most one adapter call per ``command_id`` *per store*. The id is
recorded before the adapter is called, so a crash between the two loses the
command instead of repeating it. ``InMemoryDedupStore`` forgets everything on
restart; ``JsonlDedupStore`` survives restarts as long as the same file is used.
Neither gives "exactly once" towards a real drone.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from .jsonio import JsonFormatError, dumps, loads_strict


class DedupStoreError(RuntimeError):
    pass


class DedupStore(Protocol):
    def __contains__(self, command_id: object) -> bool: ...

    def add(self, command_id: str, recorded_at: str) -> None: ...


class InMemoryDedupStore:
    def __init__(self) -> None:
        self._ids: set[str] = set()

    def __contains__(self, command_id: object) -> bool:
        return command_id in self._ids

    def add(self, command_id: str, recorded_at: str) -> None:
        if command_id in self._ids:
            raise DedupStoreError(f"{command_id} already recorded")
        self._ids.add(command_id)


class JsonlDedupStore:
    """Append-only JSON Lines file, fsync'ed on every write.

    A corrupt or truncated file makes loading fail (fail closed): the operator
    must inspect it rather than risk re-executing commands.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._ids: set[str] = set()
        if self.path.exists():
            for line_no, line in enumerate(self.path.read_bytes().split(b"\n"), start=1):
                if not line.strip():
                    continue
                try:
                    record = loads_strict(line)
                except JsonFormatError as exc:
                    raise DedupStoreError(f"{self.path}:{line_no}: {exc}") from exc
                if not isinstance(record, dict) or not isinstance(record.get("command_id"), str):
                    raise DedupStoreError(f"{self.path}:{line_no}: missing command_id")
                self._ids.add(record["command_id"])

    def __contains__(self, command_id: object) -> bool:
        return command_id in self._ids

    def add(self, command_id: str, recorded_at: str) -> None:
        if command_id in self._ids:
            raise DedupStoreError(f"{command_id} already recorded")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = dumps({"command_id": command_id, "recorded_at": recorded_at}, indent=None) + "\n"
        with open(self.path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        self._ids.add(command_id)
