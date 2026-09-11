"""Minimal detection and tracking annotation."""

import cv2
import numpy as np

from src.core.models import Detection, TrackedObject


def _draw_label(image, text, x, y, color):
    """Draw one readable label, replacing any previous label at this position."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    text_y = max(height + 4, y)
    cv2.rectangle(
        image, (x, text_y - height - 4),
        (min(image.shape[1] - 1, x + width + 6), text_y + baseline + 2),
        (20, 20, 20), -1,
    )
    cv2.putText(image, text, (x + 3, text_y), font, scale, color,
                thickness, cv2.LINE_AA)


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
    """Return bounding boxes labelled only with persistent track IDs."""
    annotated = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = (int(value) for value in track.bbox)
        color = (0, 200, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        _draw_label(annotated, f"ID:{track.track_id}", x1, y1 - 5, color)
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
    """Replace the ID label with a violation label after N-frame confirmation."""
    annotated = frame.copy()
    for track in tracks:
        _, state = statuses[track.track_id]
        if state.value not in ("SUSPECTED_VIOLATION", "ALERTED"):
            continue
        x1, y1, x2, y2 = (int(value) for value in track.bbox)
        color = (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        _draw_label(annotated, f"VIOLATION ID:{track.track_id}", x1, y1 - 5, color)
    return annotated
