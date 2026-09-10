"""Small adapter around supervision 0.25.0 ByteTrack; no ReID or history."""

import math

import numpy as np

from src.core.models import Detection, TrackedObject
from src.tracking.base import BaseTracker


def validate_config(config: dict, frame_rate: float) -> None:
    if not isinstance(config, dict) or config.get("backend") != "bytetrack":
        raise ValueError("Config must provide tracking.backend=bytetrack")
    for key, maximum in (("track_thresh", 0.9), ("match_thresh", 1.0)):
        value = config.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or not 0 < value <= maximum):
            raise ValueError(f"tracking.{key} must be a number in (0, {maximum}]")
    # This implementation starts new tracks at track_thresh + 0.1.
    buffer = config.get("track_buffer")
    if isinstance(buffer, bool) or not isinstance(buffer, int) or buffer < 1:
        raise ValueError("tracking.track_buffer must be a positive integer")
    if (isinstance(frame_rate, bool) or not isinstance(frame_rate, (int, float)) or
            not math.isfinite(frame_rate) or frame_rate <= 0):
        raise ValueError("Tracking requires a positive video frame rate")


class ByteTrackTracker(BaseTracker):
    """Create one instance per video/run; IDs are local to that instance.

    Supervision matches detections to tracks and preserves their metadata.
    Returned bbox/class/confidence describe the current matched detection,
    not a predicted box during a missing observation. Association is class
    agnostic; detector class changes can therefore change a track's label.
    """

    def __init__(self, config: dict, frame_rate: float):
        validate_config(config, frame_rate)
        try:
            import supervision as sv
        except ImportError as exc:
            raise RuntimeError("ByteTrack requires supervision; install requirements.txt") from exc
        self._detections_type = sv.Detections
        self._tracker = sv.ByteTrack(
            track_activation_threshold=float(config["track_thresh"]),
            lost_track_buffer=config["track_buffer"],
            minimum_matching_threshold=float(config["match_thresh"]),
            frame_rate=frame_rate,
        )

    def update(self, detections: list[Detection], frame,
               timestamp: float) -> list[TrackedObject]:
        """Advance ByteTrack even on empty frames; retain the supplied time.

        Frame is part of the shared interface; ByteTrack only needs detections.
        No independent clock, class filtering, or geometry is added here.
        """
        if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or
                not math.isfinite(timestamp) or timestamp < 0):
            raise ValueError("Tracking timestamp must be nonnegative video seconds")
        if detections:
            tracker_input = self._detections_type(
                xyxy=np.asarray([d.bbox for d in detections], dtype=np.float32),
                confidence=np.asarray([d.confidence for d in detections], dtype=np.float32),
                class_id=np.asarray([d.class_id for d in detections], dtype=int),
                data={"class_name": np.asarray([d.class_name for d in detections])},
            )
        else:
            tracker_input = self._detections_type.empty()
        try:
            tracked = self._tracker.update_with_detections(tracker_input)
        except Exception as exc:
            raise RuntimeError(f"ByteTrack update failed: {exc}") from exc
        if len(tracked) == 0:
            return []
        return [
            TrackedObject(
                track_id=int(tracked.tracker_id[i]),
                bbox=tuple(float(value) for value in tracked.xyxy[i]),
                confidence=float(tracked.confidence[i]),
                class_name=str(tracked.data["class_name"][i]),
                timestamp=timestamp,
            )
            for i in range(len(tracked))
        ]
