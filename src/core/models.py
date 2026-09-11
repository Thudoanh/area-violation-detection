"""Data containers only. Bboxes are pixel xyxy; times are video seconds."""

from dataclasses import dataclass, field
from typing import Literal, Optional


BBox = tuple[float, float, float, float]
ZoneType = Literal["SIDEWALK", "MONITORED", "ALLOWED", "IGNORE"]


@dataclass
class Detection:
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str


@dataclass
class TrackedObject:
    track_id: int
    bbox: BBox
    confidence: float
    class_name: str
    timestamp: float


@dataclass
class Zone:
    zone_id: str
    camera_id: str
    zone_type: ZoneType
    polygon: list[tuple[int, int]]


@dataclass
class ViolationEvent:
    """Canonical events schema in Solution Design section 13; no lifecycle logic."""

    event_id: str
    run_id: str
    video_id: str
    camera_id: str
    zone_id: str
    zone_type: ZoneType
    track_id: int
    object_class: str
    entered_at: float
    stationary_since: float
    violation_at: float
    left_at: Optional[float]
    dwell_time_sec: float
    inside_frame_count: int
    confidence: float
    snapshot_path: str
    status: Literal["OPEN", "CLOSED"]
    config_version: str
    model_version: str
    event_type: Literal["SUSPECTED_AREA_OCCUPATION"] = field(
        default="SUSPECTED_AREA_OCCUPATION", init=False
    )
