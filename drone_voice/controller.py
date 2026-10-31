"""Drone controller interface and the simulated adapter.

DRONE INTEGRATION POINT. A real adapter implements ``DroneController``. It is not
provided: the autopilot, protocol and flight modes are unknown (see
docs/INTEGRATION.md). ``SimulatedController`` only records calls in memory; it
has no hardware, network or flight-physics access.

Frame of reference: the drone's body frame. FORWARD/BACKWARD follow the nose,
RIGHT/LEFT are lateral translations (never rotations), UP/DOWN change altitude
(DOWN is not a landing). ``stop`` ends the requested displacement and holds
position; it never cuts the motors.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Protocol

from .clock import Clock
from .contract import format_timestamp


class Direction(str, Enum):
    FORWARD = "FORWARD"
    BACKWARD = "BACKWARD"
    RIGHT = "RIGHT"
    LEFT = "LEFT"
    UP = "UP"
    DOWN = "DOWN"


class StopCause(str, Enum):
    STOP_COMMAND = "stop_command"
    DURATION_ELAPSED = "duration_elapsed"


@dataclass(frozen=True)
class AdapterResult:
    accepted: bool
    simulated: bool
    detail: str = ""


class AdapterError(RuntimeError):
    pass


class DroneController(Protocol):
    def move(self, *, command_id: str, direction: Direction, speed_m_s: float, duration_s: float) -> AdapterResult:
        """Start a body-frame translation; must itself not exceed ``duration_s``."""

    def stop(self, *, command_id: str, cause: StopCause) -> AdapterResult:
        """End any displacement and hold position. Must be idempotent. Never disarms."""


class SimulatedController:
    """In-memory recorder. Not a flight simulator: no physics, no position."""

    def __init__(self, clock: Clock | None = None):
        self._clock = clock
        self.calls: list[dict[str, Any]] = []
        self._moving = False
        self._moving_until: datetime | None = None
        self.current_command_id: str | None = None
        self._pending_failures = {"move": 0, "stop": 0}

    @property
    def state(self) -> str:
        """"moving" or "holding". With a clock, the simulator bounds each movement to its
        own ``duration_s`` even if no stop arrives (the adapter-side half of the double protection)."""
        if not self._moving:
            return "holding"
        if self._clock is not None and self._moving_until is not None and self._clock.now() >= self._moving_until:
            return "holding"
        return "moving"

    def fail_next(self, method: str, times: int = 1) -> None:
        """Make the next ``times`` calls of ``method`` raise ``AdapterError`` (tests)."""
        if method not in self._pending_failures:
            raise ValueError(f"unknown method {method!r}")
        self._pending_failures[method] += times

    def _record(self, call: dict[str, Any]) -> dict[str, Any]:
        if self._clock is not None:
            call["at"] = format_timestamp(self._clock.now())
        self.calls.append(call)
        if self._pending_failures[call["method"]] > 0:
            self._pending_failures[call["method"]] -= 1
            call["result"] = "error"
            raise AdapterError(f"simulated {call['method']} failure")
        call["result"] = "ok"
        return call

    def move(self, *, command_id: str, direction: Direction, speed_m_s: float, duration_s: float) -> AdapterResult:
        self._record({
            "method": "move",
            "command_id": command_id,
            "direction": Direction(direction).value,
            "speed_m_s": speed_m_s,
            "duration_s": duration_s,
        })
        self._moving = True
        self._moving_until = self._clock.now() + timedelta(seconds=duration_s) if self._clock else None
        self.current_command_id = command_id
        return AdapterResult(True, True, f"simulated move {direction.value} {speed_m_s} m/s for {duration_s} s")

    def stop(self, *, command_id: str, cause: StopCause) -> AdapterResult:
        self._record({"method": "stop", "command_id": command_id, "cause": StopCause(cause).value})
        self._moving = False
        self._moving_until = None
        self.current_command_id = None
        return AdapterResult(True, True, f"simulated stop ({cause.value}), holding position")
