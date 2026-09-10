import pytest

from src.violation.stationary import StationaryDetector
from tests.event_helpers import track


def test_full_window_required_and_small_motion():
    detector = StationaryDetector(3, 15)
    assert not detector.update(track(0)).stationary
    assert not detector.update(track(2.9, x=25)).stationary
    result = detector.update(track(3, x=30))
    assert result.stationary
    assert result.window_start == 0


def test_moving_and_independent_tracks():
    detector = StationaryDetector(3, 15)
    detector.update(track(0))
    detector.update(track(0, track_id=2))
    assert not detector.update(track(3, x=100)).stationary
    assert detector.update(track(3, track_id=2)).stationary


def test_window_pruning_irregular_timestamps_and_reset():
    detector = StationaryDetector(3, 15)
    for timestamp in (0, 1.1, 2.5, 3.2, 4.4, 5.7):
        detector.update(track(timestamp))
    assert detector.history[1][0][0] == 2.5
    detector.reset(1)
    assert not detector.update(track(6)).stationary


def test_invalid_window_and_duplicate_time():
    with pytest.raises(ValueError):
        StationaryDetector(0, 15)
    detector = StationaryDetector(3, 15)
    detector.update(track(0))
    with pytest.raises(ValueError, match="increase"):
        detector.update(track(0))
