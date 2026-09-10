import cv2
import numpy as np
import pytest

from src.zones.auto_roi import AutoROISettings, ClassicalAutoROIDetector


def striped_frame(transient=False):
    frame = np.zeros((240, 360, 3), dtype=np.uint8)
    for offset in range(0, 150, 25):
        cv2.line(frame, (90 + offset, 190), (145 + offset, 90), (255, 255, 255), 8)
    if transient:
        cv2.rectangle(frame, (0, 0), (359, 239), (255, 255, 255), -1)
    return frame


def test_auto_roi_finds_stable_diagonal_painted_region():
    detector = ClassicalAutoROIDetector(AutoROISettings(
        num_frames=5, vote_ratio=0.6, min_area_ratio=0.002,
    ))
    polygon = detector.detect_from_frames([striped_frame() for _ in range(5)])
    assert len(polygon) >= 3
    assert all(0 <= x < 360 and 0 <= y < 240 for x, y in polygon)
    assert cv2.contourArea(np.asarray(polygon, np.float32)) > 360 * 240 * 0.002


def test_temporal_vote_rejects_one_frame_marking():
    detector = ClassicalAutoROIDetector(AutoROISettings(
        num_frames=5, vote_ratio=0.6, min_area_ratio=0.002,
    ))
    blank = np.zeros((240, 360, 3), dtype=np.uint8)
    polygon = detector.detect_from_frames([striped_frame(), blank, blank, blank, blank])
    assert polygon == []


@pytest.mark.parametrize("kwargs", [
    {"num_frames": 0}, {"vote_ratio": 0}, {"vote_ratio": 1.1},
    {"min_area_ratio": 0}, {"simplify_epsilon_ratio": 2},
])
def test_auto_roi_settings_validation(kwargs):
    with pytest.raises(ValueError):
        AutoROISettings(**kwargs)
