from unittest.mock import Mock

import numpy as np
import pytest

from src.core.config import load_config
from src.core.models import Detection, TrackedObject
from src.tracking.base import BaseTracker
from src.tracking.bytetrack_tracker import ByteTrackTracker, validate_config


def detection(x=10, name="car", class_id=2, confidence=0.9):
    return Detection((x, 20, x + 40, 80), confidence, class_id, name)


def tracker():
    return ByteTrackTracker(load_config()["tracking"], frame_rate=30.0)


def test_interface():
    with pytest.raises(NotImplementedError):
        BaseTracker().update([], None, 0.0)


def test_config_parameters_reach_backend():
    config = load_config()["tracking"]
    validate_config(config, 25.0)
    adapter = ByteTrackTracker(config, 25.0)
    assert adapter._tracker.track_activation_threshold == 0.5
    assert adapter._tracker.minimum_matching_threshold == 0.8
    assert adapter._tracker.max_time_lost == 25


def test_input_conversion_and_output_metadata(monkeypatch):
    adapter = tracker()
    detections = [detection(), detection(200, "person", 0, 0.8)]

    def update(batch):
        assert batch.xyxy.shape == (2, 4)
        np.testing.assert_allclose(batch.xyxy, [d.bbox for d in detections])
        np.testing.assert_allclose(batch.confidence, [0.9, 0.8])
        np.testing.assert_array_equal(batch.class_id, [2, 0])
        batch.tracker_id = np.array([12, 45])
        return batch[[1, 0]]  # Metadata must follow backend output order.

    monkeypatch.setattr(adapter._tracker, "update_with_detections", update)
    result = adapter.update(detections, None, 12.345)
    assert all(isinstance(item, TrackedObject) for item in result)
    assert [item.track_id for item in result] == [45, 12]
    assert all(type(item.track_id) is int for item in result)
    assert [item.class_name for item in result] == ["person", "car"]
    assert result[0].confidence == pytest.approx(0.8)
    assert result[0].bbox == detections[1].bbox
    assert all(item.timestamp == 12.345 for item in result)


def test_empty_detections_advance_backend():
    adapter = tracker()
    assert adapter.update([], None, 0.0) == []
    assert adapter.update([], None, 0.04) == []
    assert adapter._tracker.frame_id == 2


def test_real_bytetrack_keeps_ids_for_two_objects():
    adapter = tracker()
    first = adapter.update([detection(), detection(200, "person", 0)], None, 0.0)
    second = adapter.update([detection(12), detection(202, "person", 0)], None, 0.137)
    assert len(first) == len(second) == 2
    assert len({t.track_id for t in first}) == 2
    assert [t.track_id for t in first] == [t.track_id for t in second]
    assert [t.class_name for t in second] == ["car", "person"]
    assert all(t.timestamp == 0.137 for t in second)


def test_new_object_gets_distinct_id_after_confirmation():
    adapter = tracker()
    first, = adapter.update([detection()], None, 0.0)
    adapter.update([detection(), detection(200)], None, 0.04)
    result = adapter.update([detection(), detection(200)], None, 0.08)
    assert len(result) == 2
    assert first.track_id in {t.track_id for t in result}
    assert len({t.track_id for t in result}) == 2


def test_short_gap_retains_id_without_emitting_predicted_track():
    adapter = tracker()
    first, = adapter.update([detection()], None, 0.0)
    assert adapter.update([], None, 0.04) == []
    recovered, = adapter.update([detection(12)], None, 0.08)
    assert recovered.track_id == first.track_id
    assert recovered.timestamp == 0.08


def test_low_score_association_keeps_existing_id():
    adapter = tracker()
    first, = adapter.update([detection()], None, 0.0)
    second, = adapter.update([detection(12, confidence=0.45)], None, 0.04)
    assert second.track_id == first.track_id
    assert second.confidence == pytest.approx(0.45)
    assert tracker().update([detection(confidence=0.45)], None, 0.0) == []


def test_expired_track_gets_new_id():
    adapter = tracker()
    first, = adapter.update([detection()], None, 0.0)
    for index in range(1, 34):
        assert adapter.update([], None, index / 30) == []
    adapter.update([detection()], None, 34 / 30)
    result, = adapter.update([detection()], None, 35 / 30)
    assert result.track_id != first.track_id


def test_new_instance_has_independent_run_ids():
    first, = tracker().update([detection()], None, 0.0)
    second, = tracker().update([detection()], None, 0.0)
    assert first.track_id == second.track_id == 1


def test_backend_error_is_clear(monkeypatch):
    adapter = tracker()
    monkeypatch.setattr(adapter._tracker, "update_with_detections",
                        Mock(side_effect=ValueError("bad input")))
    with pytest.raises(RuntimeError, match="ByteTrack update failed"):
        adapter.update([], None, 0.0)


@pytest.mark.parametrize("timestamp", [-1, float("nan"), float("inf"), None, True])
def test_invalid_timestamp(timestamp):
    adapter = tracker()
    with pytest.raises(ValueError, match="video seconds"):
        adapter.update([], None, timestamp)
    assert adapter._tracker.frame_id == 0


@pytest.mark.parametrize("key,value", [
    ("backend", "sort"), ("track_thresh", 0), ("track_thresh", 0.95),
    ("track_thresh", True), ("track_thresh", float("nan")),
    ("match_thresh", -0.1), ("match_thresh", 1.1),
    ("track_buffer", 0), ("track_buffer", 2.5), ("track_buffer", True),
])
def test_invalid_config(key, value):
    config = load_config()["tracking"]
    config[key] = value
    with pytest.raises(ValueError):
        ByteTrackTracker(config, 30.0)


@pytest.mark.parametrize("fps", [0, -1, float("nan"), True])
def test_invalid_frame_rate(fps):
    with pytest.raises(ValueError, match="frame rate"):
        ByteTrackTracker(load_config()["tracking"], fps)


def test_missing_config():
    with pytest.raises(ValueError, match="tracking.backend"):
        ByteTrackTracker(None, 30.0)
