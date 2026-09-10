from types import SimpleNamespace
from unittest.mock import Mock
import sys
import os

import numpy as np
import pytest

from src.core.config import load_config
from src.detection.yolo11_detector import (
    CLASS_NAMES, COCO_TO_PROJECT, YOLO11Detector, to_detections, validate_config,
)


def tensor(values):
    value = Mock()
    value.detach.return_value.cpu.return_value.numpy.return_value = np.asarray(values, dtype=np.float32)
    return value


def result(rows):
    data = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
    return SimpleNamespace(boxes=SimpleNamespace(
        xyxy=tensor(data[:, :4]), conf=tensor(data[:, 4]), cls=tensor(data[:, 5])))


@pytest.fixture
def backend(tmp_path, monkeypatch):
    config = load_config()["detector"]
    weights = tmp_path / "yolo11n.pt"
    weights.touch()
    config["weights"] = str(weights)
    model = Mock(task="detect", names={n: CLASS_NAMES[p] for n, p in COCO_TO_PROJECT.items()})
    model.predict.return_value = [result([[10, 10, 30, 50, .9, 2]])]
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=factory))
    return config, model, factory


@pytest.mark.parametrize("native,project", COCO_TO_PROJECT.items())
def test_mapping_original_pixels_and_python_types(native, project):
    output = result([[10, 20, 100, 60, .9, native]])
    detection, = to_detections(output, (100, 300, 3), CLASS_NAMES, .4)
    assert detection.bbox == (10., 20., 100., 60.)
    assert detection.class_id == project and detection.class_name == CLASS_NAMES[project]
    assert type(detection.class_id) is int and type(detection.confidence) is float
    assert all(type(v) is float for v in detection.bbox)
    output.boxes.xyxy.detach.return_value.cpu.assert_called_once()


def test_filters_clips_invalid_and_empty():
    output = result([[-10, -10, 500, 200, .9, 2], [1, 2, 4, 5, .9, 4],
                     [1, 2, 4, 5, .3, 2], [1, 2, 4, 5, .9, 0],
                     [4, 2, 1, 5, .9, 2], [np.nan, 2, 4, 5, .9, 2],
                     [1, 2, 4, 5, np.nan, 2], [1, 2, 4, 5, .9, 2.5]])
    detections = to_detections(output, (100, 300, 3), ["car"], .4)
    assert len(detections) == 1 and detections[0].bbox == (0., 0., 299., 99.)
    assert to_detections(result([]), (100, 300, 3), CLASS_NAMES, .4) == []
    assert to_detections(SimpleNamespace(boxes=None), (100, 300, 3), CLASS_NAMES, .4) == []


def test_load_predict_parameters_and_error(backend):
    config, model, factory = backend
    detector = YOLO11Detector(config)
    factory.assert_called_once_with(config["weights"], task="detect")
    frame = np.zeros((100, 300, 3), np.uint8)
    assert detector.detect(frame)[0].class_name == "car"
    kwargs = model.predict.call_args.kwargs
    assert kwargs["source"] is frame
    assert kwargs["conf"] == .4 and kwargs["iou"] == .45
    assert kwargs["device"] == "cpu" and kwargs["imgsz"] == 640
    assert set(kwargs["classes"]) == {0, 1, 2, 3, 5, 7}
    assert kwargs["agnostic_nms"] is False and kwargs["save"] is False
    model.predict.side_effect = RuntimeError("device failure")
    with pytest.raises(RuntimeError, match="YOLO11 inference failed"):
        detector.detect(frame)


@pytest.mark.parametrize("frame", [None, np.zeros((0, 0, 3), np.uint8),
                                  np.zeros((10, 10)), np.zeros((10, 10, 4), np.uint8)])
def test_invalid_frame(backend, frame):
    with pytest.raises(ValueError, match="BGR frame"):
        YOLO11Detector(backend[0]).detect(frame)


@pytest.mark.parametrize("key,value", [("backend", "yolox"), ("model", "yolo11s"),
    ("weights", "wrong.onnx"), ("device", ""), ("input_size", 0), ("input_size", True),
    ("confidence_threshold", float("nan")), ("nms_threshold", 2),
    ("detection_classes", ["car", "car"]), ("detection_classes", ["dog"])])
def test_config_errors(key, value):
    config = load_config()["detector"]
    config[key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_missing_weights_and_wrong_model(backend, tmp_path):
    config, model, _ = backend
    model.names = {0: "wrong"}
    with pytest.raises(ValueError, match="COCO"):
        YOLO11Detector(config)
    config["weights"] = str(tmp_path / "missing.pt")
    with pytest.raises(FileNotFoundError, match="YOLO11 weights not found"):
        YOLO11Detector(config)


def test_adapter_through_real_tracking_roi_rule_event(backend, tmp_path):
    import sqlite3
    from pathlib import Path
    from tests.event_helpers import config as event_config, manager
    from src.tracking.bytetrack_tracker import ByteTrackTracker
    from src.pipeline.pipeline import AreaMonitoringPipeline

    config = event_config(tmp_path)
    config["detector"] = backend[0]
    zones = manager()
    detector = YOLO11Detector(config["detector"])
    tracker = ByteTrackTracker(config["tracking"], frame_rate=1)
    pipeline = AreaMonitoringPipeline(config, "frames", zones)
    frame = np.zeros((100, 300, 3), np.uint8)
    try:
        for time in range(35):
            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame, float(time))
            memberships = {t.track_id: zones.get_membership(t, "CAM_001") for t in tracks}
            pipeline.process(tracks, memberships, float(time), frame)
        assert pipeline.stats["accepted"] == 1
    finally:
        pipeline.finish()
    with sqlite3.connect(config["storage"]["sqlite_path"]) as db:
        rows = db.execute("SELECT object_class,status,snapshot_path FROM events").fetchall()
    assert len(rows) == 1 and rows[0][:2] == ("car", "CLOSED")
    assert Path(rows[0][2]).is_file()


@pytest.mark.skipif(os.environ.get("RUN_YOLO11_SMOKE") != "1",
                    reason="Opt-in: requires local yolo11n.pt and installed Ultralytics")
def test_real_pretrained_frame_through_pipeline(tmp_path):
    import cv2
    import ultralytics
    from pathlib import Path
    from src.core.models import Zone
    from src.zones.zone_manager import ZoneManager
    from src.tracking.bytetrack_tracker import ByteTrackTracker
    from src.pipeline.pipeline import AreaMonitoringPipeline

    config = load_config()
    config["storage"] = {"sqlite_path": str(tmp_path / "events.db"),
                         "snapshot_dir": str(tmp_path / "snapshots")}
    # Existing package example image, no network or generated video.
    path = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    frame = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    assert frame is not None
    height, width = frame.shape[:2]
    zones = ZoneManager("CAM_001", [Zone("smoke", "CAM_001", "MONITORED",
        [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)])])
    detector = YOLO11Detector(config["detector"])
    tracker = ByteTrackTracker(config["tracking"], frame_rate=25)
    pipeline = AreaMonitoringPipeline(config, str(path), zones)
    try:
        for timestamp in (0., .04):
            detections = detector.detect(frame)
            assert any(d.class_name == "bus" for d in detections)
            assert all(type(d.confidence) is float and type(d.class_id) is int for d in detections)
            tracks = tracker.update(detections, frame, timestamp)
            assert tracks
            memberships = {t.track_id: zones.get_membership(t, "CAM_001") for t in tracks}
            assert all(m.target_zone is not None for m in memberships.values())
            assert pipeline.process(tracks, memberships, timestamp, frame).shape == frame.shape
        assert pipeline.stats["accepted"] == 0  # Two frames cannot meet dwell threshold.
    finally:
        pipeline.finish()
