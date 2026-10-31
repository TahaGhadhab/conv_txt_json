"""Validated runtime settings: message validity window and simulated motion profiles.

The configuration file cannot exceed the hard caps defined below. These values
exist for the local simulation only; they are NOT validated flight parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .intents import MOVE_INTENTS, Intent
from .jsonio import JsonFormatError, read_json_file
from .vocabulary import ConfigError

SETTINGS_FORMAT_VERSION = 1

HARD_MAX_SPEED_M_S = 1.0
HARD_MAX_DURATION_S = 5.0
HARD_MAX_TTL_S = 30.0
HARD_MAX_FUTURE_SKEW_S = 2.0
MIN_TTL_S = 0.1


@dataclass(frozen=True)
class MotionProfile:
    speed_m_s: float
    duration_s: float


@dataclass(frozen=True)
class Settings:
    ttl_s: float
    max_ttl_s: float
    max_future_skew_s: float
    max_speed_m_s: float
    max_duration_s: float
    motion: Mapping[Intent, MotionProfile]


def _number(value: Any, name: str, *, low: float, high: float, low_inclusive: bool) -> float:
    if type(value) not in (int, float):
        raise ConfigError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ConfigError(f"{name} must be finite")
    too_low = value < low if low_inclusive else value <= low
    if too_low or value > high:
        bound = "[" if low_inclusive else "]"
        raise ConfigError(f"{name}={value} outside {bound}{low}, {high}]")
    return float(value)


def _object(value: Any, name: str, allowed: set[str], required: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be an object")
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown or missing:
        raise ConfigError(f"{name}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    return value


def settings_from_dict(data: Any) -> Settings:
    root = _object(
        data,
        "settings",
        {"format_version", "simulation_only", "note", "message", "motion"},
        {"format_version", "simulation_only", "message", "motion"},
    )
    if type(root["format_version"]) is not int or root["format_version"] != SETTINGS_FORMAT_VERSION:
        raise ConfigError(f"settings format_version must be {SETTINGS_FORMAT_VERSION}")
    if root["simulation_only"] is not True:
        raise ConfigError("simulation_only must be true: these values are not flight parameters")

    msg = _object(root["message"], "message", {"ttl_s", "max_ttl_s", "max_future_skew_s"},
                  {"ttl_s", "max_ttl_s", "max_future_skew_s"})
    max_ttl = _number(msg["max_ttl_s"], "message.max_ttl_s", low=MIN_TTL_S, high=HARD_MAX_TTL_S, low_inclusive=True)
    ttl = _number(msg["ttl_s"], "message.ttl_s", low=MIN_TTL_S, high=max_ttl, low_inclusive=True)
    skew = _number(msg["max_future_skew_s"], "message.max_future_skew_s",
                   low=0.0, high=HARD_MAX_FUTURE_SKEW_S, low_inclusive=True)

    motion = _object(root["motion"], "motion", {"max_speed_m_s", "max_duration_s", "intents"},
                     {"max_speed_m_s", "max_duration_s", "intents"})
    max_speed = _number(motion["max_speed_m_s"], "motion.max_speed_m_s",
                        low=0.0, high=HARD_MAX_SPEED_M_S, low_inclusive=False)
    max_duration = _number(motion["max_duration_s"], "motion.max_duration_s",
                           low=0.0, high=HARD_MAX_DURATION_S, low_inclusive=False)
    move_names = {i.value for i in MOVE_INTENTS}
    intents = _object(motion["intents"], "motion.intents", move_names, move_names)
    profiles: dict[Intent, MotionProfile] = {}
    for name, profile in intents.items():
        p = _object(profile, f"motion.intents.{name}", {"speed_m_s", "duration_s"}, {"speed_m_s", "duration_s"})
        profiles[Intent(name)] = MotionProfile(
            speed_m_s=_number(p["speed_m_s"], f"{name}.speed_m_s", low=0.0, high=max_speed, low_inclusive=False),
            duration_s=_number(p["duration_s"], f"{name}.duration_s", low=0.0, high=max_duration, low_inclusive=False),
        )
    return Settings(
        ttl_s=ttl,
        max_ttl_s=max_ttl,
        max_future_skew_s=skew,
        max_speed_m_s=max_speed,
        max_duration_s=max_duration,
        motion=MappingProxyType(profiles),
    )


def load_settings(path: str | Path) -> Settings:
    try:
        data = read_json_file(path)
    except (OSError, JsonFormatError) as exc:
        raise ConfigError(f"cannot read settings {path}: {exc}") from exc
    return settings_from_dict(data)
