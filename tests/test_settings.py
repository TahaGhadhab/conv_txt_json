import copy
import json

import pytest

from drone_voice.pipeline import DEFAULT_SETTINGS_PATH
from drone_voice.settings import HARD_MAX_DURATION_S, HARD_MAX_SPEED_M_S, load_settings, settings_from_dict
from drone_voice.vocabulary import ConfigError


@pytest.fixture
def base():
    return json.loads(DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8"))


def with_value(base, path, value):
    data = copy.deepcopy(base)
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return data


def test_default_settings_are_within_hard_caps(settings):
    assert settings.max_speed_m_s <= HARD_MAX_SPEED_M_S
    assert settings.max_duration_s <= HARD_MAX_DURATION_S
    assert len(settings.motion) == 6


SPEED = ("motion", "intents", "MOVE_FORWARD", "speed_m_s")
DURATION = ("motion", "intents", "MOVE_UP", "duration_s")


@pytest.mark.parametrize(
    "path, value",
    [
        (SPEED, 0), (SPEED, -0.3), (SPEED, float("nan")), (SPEED, float("inf")), (SPEED, True),
        (SPEED, "0.3"), (SPEED, None), (SPEED, 0.51),
        (DURATION, 0), (DURATION, -1), (DURATION, float("inf")), (DURATION, 3.01),
        (("motion", "max_speed_m_s"), HARD_MAX_SPEED_M_S + 0.1),
        (("motion", "max_duration_s"), HARD_MAX_DURATION_S + 0.1),
        (("message", "ttl_s"), 0), (("message", "ttl_s"), 11), (("message", "max_ttl_s"), 31),
        (("message", "max_future_skew_s"), -0.1), (("message", "max_future_skew_s"), 3),
        (("simulation_only",), False), (("format_version",), 2),
        (("motion", "intents", "STOP"), {"speed_m_s": 0.1, "duration_s": 1}),
        (("motion", "intents", "MOVE_FORWARD", "acceleration"), 1),
    ],
)
def test_invalid_values_are_refused(base, path, value):
    with pytest.raises(ConfigError):
        settings_from_dict(with_value(base, path, value))


def test_missing_intent_profile_is_refused(base):
    del base["motion"]["intents"]["MOVE_LEFT"]
    with pytest.raises(ConfigError):
        settings_from_dict(base)


def test_non_finite_literal_in_file_is_refused(tmp_path):
    text = DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8").replace('"ttl_s": 2.0', '"ttl_s": NaN')
    path = tmp_path / "s.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(path)
