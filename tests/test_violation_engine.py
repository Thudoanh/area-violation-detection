import pytest

from src.violation.violation_engine import ViolationEngine
from src.violation.stationary import Motion
from src.violation.dwell_timer import Dwell
from src.violation.state_machine import State
from src.core.config import load_config
from tests.event_helpers import track, membership


@pytest.mark.parametrize("name,types,moving,duration,expected", [
    ("car", ("SIDEWALK",), False, 30, True),
    ("motorcycle", ("MONITORED",), False, 30, True),
    ("person", ("SIDEWALK",), False, 100, False),
    ("car", ("SIDEWALK", "ALLOWED"), False, 30, False),
    ("car", ("SIDEWALK", "IGNORE"), False, 30, False),
    ("car", (), False, 30, False),
    ("car", ("SIDEWALK",), True, 30, False),
    ("car", ("SIDEWALK",), False, 29.9, False),
])
def test_rule(name, types, moving, duration, expected):
    engine = ViolationEngine(load_config()["violation"])
    result = engine.evaluate(track(30, name=name), membership(types), Motion(not moving, 0),
        Dwell("0", 0, 0, duration), State.SUSPECTED_VIOLATION, run_id="r", video_id="v",
        camera_id="CAM_001", config_version="c", model_version="m")
    assert (result is not None) is expected
    if result:
        assert result.event_type == "SUSPECTED_AREA_OCCUPATION"
        assert result.status == "OPEN"
        assert result.violation_at == 30


def test_person_cannot_be_configured_as_target():
    config = load_config()["violation"]
    config["target_classes"].append("person")
    with pytest.raises(ValueError, match="vehicle"):
        ViolationEngine(config)
