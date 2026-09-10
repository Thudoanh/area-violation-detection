"""Canonical rule only; persistence and dedup remain separate."""

from uuid import uuid4

from src.core.models import ViolationEvent
from src.violation.settings import number
from src.violation.state_machine import State

VEHICLE_CLASSES = {"motorcycle", "bicycle", "car", "bus", "truck"}


class ViolationEngine:
    def __init__(self, config):
        targets = config.get("target_classes")
        if (not isinstance(targets, list) or not targets or
                any(not isinstance(v, str) or v not in VEHICLE_CLASSES for v in targets)):
            raise ValueError("violation.target_classes must contain only MVP vehicle classes")
        if config.get("event_type") != "SUSPECTED_AREA_OCCUPATION":
            raise ValueError("Unsupported violation.event_type")
        self.targets = set(targets)
        self.threshold = number(config.get("min_dwell_time_sec"), "min_dwell_time_sec", positive=True)

    def evaluate(self, track, membership, motion, dwell, state, *, run_id, video_id,
                 camera_id, config_version, model_version):
        flags = membership.by_type
        zone = membership.target_zone
        if (track.class_name not in self.targets or zone is None or
                flags["ALLOWED"] or flags["IGNORE"] or not motion.stationary or
                dwell.stationary_since is None or dwell.dwell_time_sec < self.threshold or
                state != State.SUSPECTED_VIOLATION):
            return None
        return ViolationEvent(
            event_id=uuid4().hex, run_id=run_id, video_id=video_id, camera_id=camera_id,
            zone_id=zone.zone_id, zone_type=zone.zone_type, track_id=track.track_id,
            object_class=track.class_name, entered_at=dwell.entered_at,
            stationary_since=dwell.stationary_since, violation_at=track.timestamp,
            left_at=None, dwell_time_sec=dwell.dwell_time_sec, confidence=track.confidence,
            snapshot_path="", status="OPEN", config_version=config_version,
            model_version=model_version,
        )
