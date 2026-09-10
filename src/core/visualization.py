"""Minimal detection and tracking annotation."""

import cv2
import numpy as np

from src.core.models import Detection, TrackedObject
from src.zones.geometry import get_bottom_center


def draw_detections(frame, detections: list[Detection]):
    """Return an annotated copy, leaving the detector input unchanged."""
    annotated = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = (int(value) for value in detection.bbox)
        color = (0, 200, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(annotated, label, (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return annotated


def draw_tracks(frame, tracks: list[TrackedObject], memberships=None):
    """Return bbox + class + track ID + confidence on a copy of the frame."""
    annotated = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = (int(value) for value in track.bbox)
        color = (0, 200, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{track.class_name} ID:{track.track_id} {track.confidence:.2f}"
        if memberships is not None:
            membership = memberships[track.track_id]
            names = list(dict.fromkeys(z.zone_type for z in membership.matched_zones))
            label += " | " + ("+".join(names) if names else "OUTSIDE")
            if membership.effective_zone:
                label += f" -> {membership.effective_zone.zone_id}"
            anchor = tuple(int(v) for v in get_bottom_center(track.bbox))
            cv2.circle(annotated, anchor, 3, color, -1)
        cv2.putText(annotated, label, (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return annotated


def draw_zones(frame, zones):
    annotated = frame.copy()
    colors = {"SIDEWALK": (0, 200, 255), "MONITORED": (255, 180, 0),
              "ALLOWED": (0, 200, 0), "IGNORE": (180, 80, 180)}
    for zone in zones:
        points = np.asarray(zone.polygon, dtype=np.int32)
        color = colors[zone.zone_type]
        cv2.polylines(annotated, [points], True, color, 2)
        x, y = zone.polygon[0]
        cv2.putText(annotated, f"{zone.zone_id} {zone.zone_type}", (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return annotated


def draw_temporal(frame, tracks, statuses):
    """Add motion, stationary dwell and lifecycle, without changing source pixels."""
    annotated = frame.copy()
    for track in tracks:
        motion, dwell, state = statuses[track.track_id]
        label = f"{'STATIONARY' if motion.stationary else 'MOVING'} | dwell={dwell.dwell_time_sec:.1f}s | {state.value}"
        if state.value in ("SUSPECTED_VIOLATION", "ALERTED"):
            label += " | SUSPECTED_AREA_OCCUPATION"
        if track.class_name == "person":
            label = "CONTEXT ONLY"
        x, y = int(track.bbox[0]), int(track.bbox[1])
        cv2.putText(annotated, label, (x, min(annotated.shape[0] - 5, max(30, y + 15))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1, cv2.LINE_AA)
    return annotated
