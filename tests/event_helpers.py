from src.core.config import load_config
from src.core.models import TrackedObject, ViolationEvent, Zone
from src.zones.zone_manager import ZoneManager


def track(timestamp=0.0, track_id=1, x=20, name="car"):
    return TrackedObject(track_id, (x, 10, x + 20, 50), 0.9, name, timestamp)


def manager(types=("SIDEWALK",)):
    return ZoneManager("CAM_001", [Zone(str(i), "CAM_001", kind,
        [(0, 0), (300, 0), (300, 100), (0, 100)]) for i, kind in enumerate(types)])


def membership(types=("SIDEWALK",)):
    return manager(types).get_membership(track(), "CAM_001")


def event(event_id="e1", track_id=1, timestamp=30.0):
    return ViolationEvent(event_id=event_id, run_id="r1", video_id="v1", camera_id="CAM_001",
        zone_id="0", zone_type="SIDEWALK", track_id=track_id, object_class="car",
        entered_at=0.0, stationary_since=0.0, violation_at=timestamp, left_at=None,
        dwell_time_sec=timestamp, inside_frame_count=30, confidence=0.9,
        snapshot_path="", status="OPEN",
        config_version="c1", model_version="m1")


def config(tmp_path):
    result = load_config()
    result["storage"]["sqlite_path"] = str(tmp_path / "events.db")
    result["storage"]["snapshot_dir"] = str(tmp_path / "snapshots")
    result["detector"]["weights"] = str(tmp_path / "unused.onnx")
    return result
