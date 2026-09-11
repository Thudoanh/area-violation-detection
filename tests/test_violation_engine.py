import pytest

from src.violation.violation_engine import ViolationEngine
from src.violation.inside_frame_counter import InsideFrameStatus
from src.violation.state_machine import State
from src.core.config import load_config
from tests.event_helpers import track, membership


@pytest.mark.parametrize("name,types,inside_frames,expected", [
    ("car", ("SIDEWALK",), 30, True),
    ("motorcycle", ("MONITORED",), 30, True),
    ("person", ("SIDEWALK",), 100, False),
    ("car", ("SIDEWALK", "ALLOWED"), 30, False),
    ("car", ("SIDEWALK", "IGNORE"), 30, False),
    ("car", (), 30, False),
    ("car", ("SIDEWALK",), 29, False),
])
def test_rule(name, types, inside_frames, expected):
    config = load_config()["violation"]
    config["min_inside_frames"] = 30
    engine = ViolationEngine(config)
    result = engine.evaluate(track(30, name=name), membership(types),
        InsideFrameStatus("0", 0, inside_frames), State.SUSPECTED_VIOLATION,
        run_id="r", video_id="v",
        camera_id="CAM_001", config_version="c", model_version="m")
    assert (result is not None) is expected
    if result:
        assert result.event_type == "SUSPECTED_AREA_OCCUPATION"
        assert result.status == "OPEN"
        assert result.violation_at == 30
        assert result.inside_frame_count == 30


def test_person_cannot_be_configured_as_target():
    config = load_config()["violation"]
    config["target_classes"].append("person")
    with pytest.raises(ValueError, match="vehicle"):
        ViolationEngine(config)
