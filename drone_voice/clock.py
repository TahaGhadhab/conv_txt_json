"""Injectable UTC clocks. Tests and the CLI demo use ``ManualClock`` (no real waiting)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime:
        """Current time as a timezone-aware UTC datetime."""


def truncate_ms(dt: datetime) -> datetime:
    return dt.replace(microsecond=(dt.microsecond // 1000) * 1000)


def _require_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return dt.astimezone(timezone.utc)


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class ManualClock:
    def __init__(self, start: datetime):
        self._now = _require_aware(start)

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("ManualClock cannot go backwards; use set() explicitly")
        self._now += timedelta(seconds=seconds)

    def advance_to(self, dt: datetime) -> None:
        dt = _require_aware(dt)
        if dt > self._now:
            self._now = dt

    def set(self, dt: datetime) -> None:
        """Set any time, including the past (used to test clock jumps)."""
        self._now = _require_aware(dt)
