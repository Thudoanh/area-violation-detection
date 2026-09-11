from dataclasses import asdict

import pytest

from src.core.models import Detection, TrackedObject, Zone, ViolationEvent


def test_detection():
    detection = Detection((1, 2, 30, 40), 0.9, 0, "person")
    assert detection.class_name == "person"
    assert detection.bbox == (1, 2, 30, 40)


def test_tracked_object():
    track = TrackedObject(7, (1, 2, 30, 40), 0.9, "car", 1.5)
    assert track.track_id == 7
    assert track.timestamp == 1.5


@pytest.mark.parametrize("zone_type", ["SIDEWALK", "MONITORED", "ALLOWED", "IGNORE"])
def test_zone(zone_type):
    zone = Zone("SW01", "CAM_001", zone_type, [(0, 0), (100, 0), (100, 100)])
    assert zone.zone_type == zone_type
    assert len(zone.polygon) == 3


def test_event_canonical_schema():
    event = ViolationEvent(
        event_id="EV1", run_id="RUN1", video_id="VIDEO1", camera_id="CAM_001",
        zone_id="SW01", zone_type="SIDEWALK", track_id=7, object_class="car",
        entered_at=1.0, stationary_since=2.0, violation_at=32.0, left_at=None,
        dwell_time_sec=30.0, inside_frame_count=30, confidence=0.9,
        snapshot_path="data/events/EV1.jpg",
        status="OPEN", config_version="v1", model_version="baseline",
    )
    assert event.event_type == "SUSPECTED_AREA_OCCUPATION"
    assert event.left_at is None
    assert set(asdict(event)) == {
        "event_id", "event_type", "run_id", "video_id", "camera_id", "zone_id",
        "zone_type", "track_id", "object_class", "entered_at", "stationary_since",
        "violation_at", "left_at", "dwell_time_sec", "inside_frame_count",
        "confidence", "snapshot_path",
        "status", "config_version", "model_version",
    }
