"""Closed set of internal intents."""

from __future__ import annotations

from enum import Enum


class Intent(str, Enum):
    MOVE_FORWARD = "MOVE_FORWARD"
    MOVE_BACKWARD = "MOVE_BACKWARD"
    MOVE_RIGHT = "MOVE_RIGHT"
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_UP = "MOVE_UP"
    MOVE_DOWN = "MOVE_DOWN"
    STOP = "STOP"


MOVE_INTENTS = frozenset(i for i in Intent if i is not Intent.STOP)
