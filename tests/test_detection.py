from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from src.core.config import load_config
from src.core.models import Detection
from src.detection.base import BaseDetector
from src.detection.yolox_detector import (
    CLASS_NAMES, COCO_TO_PROJECT, YOLOXDetector, decode_output, preprocess,
    to_detections, validate_config,
)


@pytest.fixture(autouse=True)
def legacy_config(monkeypatch):
    original = load_config
    def configured():
        config = original()
        config["detector"].update(backend="yolox", model="yolox_s", weights="weights/yolox_s.onnx")
        return config
    monkeypatch.setitem(globals(), "load_config", configured)


def row(native_id, confidence=0.9, box=(100, 100, 40, 60)):
    result = np.zeros(85, dtype=np.float32)
    result[:4] = box
    result[4] = 0.8
    result[5 + native_id] = confidence
    return result


def convert(rows, classes=None, scale=1):
    return to_detections(np.array(rows), scale, (480, 640, 3),
                         list(CLASS_NAMES) if classes is None else classes, 0.4, 0.45)


def test_interface():
    with pytest.raises(NotImplementedError):
        BaseDetector().detect(None)


@pytest.mark.parametrize("native_id,project_id", COCO_TO_PROJECT.items())
def test_class_mapping_and_coordinate_conversion(native_id, project_id):
    detection, = convert([row(native_id)], scale=2)
    assert isinstance(detection, Detection)
    assert detection.class_id == project_id
    assert detection.class_name == CLASS_NAMES[project_id]
    assert detection.confidence == pytest.approx(0.72)
    assert detection.bbox == pytest.approx((40, 35, 60, 65))


def test_class_filter_uses_best_coco_class_not_best_target():
    airplane = row(4)
    airplane[5 + 2] = 0.85  # Still not a car.
    assert convert([airplane, row(0), row(2)], classes=["car"]) == convert([row(2)])


def test_confidence_uses_objectness_and_class_probability():
    assert convert([row(2, confidence=0.49)]) == []  # 0.8 * 0.49 < 0.4


def test_nms_suppresses_same_class_but_keeps_other_classes():
    result = convert([row(2), row(2, confidence=0.8), row(0)])
    assert sorted(d.class_name for d in result) == ["car", "person"]
    assert all(d.confidence == pytest.approx(0.72) for d in result)


def test_clipping_and_invalid_boxes():
    result = convert([row(2, box=(0, 0, 40, 60)), row(2, box=(-500, -500, 10, 10))])
    assert result[0].bbox == (0, 0, 20, 30)
    assert len(result) == 1
    invalid = row(2)
    invalid[0] = np.nan
    assert convert([invalid, row(2, box=(100, 100, -1, 60))]) == []


def test_empty_result():
    assert convert(np.empty((0, 85))) == []


def test_preprocess_keeps_bgr_range_and_top_left_padding():
    frame = np.full((100, 200, 3), (10, 20, 30), dtype=np.uint8)
    tensor, scale = preprocess(frame)
    assert scale == 3.2
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    np.testing.assert_array_equal(tensor[0, :, 0, 0], [10, 20, 30])
    np.testing.assert_array_equal(tensor[0, :, 320, 0], [114, 114, 114])


@pytest.mark.parametrize("frame", [None, np.zeros((0, 0, 3), dtype=np.uint8),
                                   np.zeros((20, 20)), np.zeros((20, 20, 4), dtype=np.uint8)])
def test_invalid_frame(frame):
    with pytest.raises(ValueError, match="BGR frame"):
        preprocess(frame)


def test_decode_grid_strides_without_mutating_raw():
    raw = np.zeros((1, 8400, 85), dtype=np.float32)
    decoded = decode_output(raw)
    np.testing.assert_allclose(decoded[81, :4], [8, 8, 8, 8])
    np.testing.assert_allclose(decoded[6401, :4], [16, 0, 16, 16])
    np.testing.assert_allclose(decoded[8001, :4], [32, 0, 32, 32])
    assert not raw.any()


def test_invalid_output():
    with pytest.raises(ValueError, match="Expected raw YOLOX-S output"):
        decode_output(np.zeros((1, 10, 6)))


@pytest.mark.parametrize("key,value", [
    ("backend", "ultralytics"), ("model", "unknown"), ("device", "cuda"),
    ("confidence_threshold", -1), ("confidence_threshold", float("nan")),
    ("confidence_threshold", True), ("nms_threshold", 1.1),
    ("detection_classes", []), ("detection_classes", ["dog"]),
    ("detection_classes", ["car", "car"]), ("detection_classes", "car"),
    ("weights", ""),
])
def test_invalid_config(key, value):
    config = load_config()["detector"]
    config[key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_missing_config():
    with pytest.raises(ValueError, match="detector mapping"):
        validate_config(None)


def test_missing_weights(tmp_path):
    config = load_config()["detector"]
    config["weights"] = str(tmp_path / "missing.onnx")
    with pytest.raises(FileNotFoundError, match="YOLOX weights not found"):
        YOLOXDetector(config)


def test_detect_with_mocked_runtime(tmp_path, monkeypatch):
    # No real model is loaded by unit tests, even if weights exist locally.
    import sys
    onnxruntime = SimpleNamespace(InferenceSession=None)
    monkeypatch.setitem(sys.modules, "onnxruntime", onnxruntime)

    weights = tmp_path / "model.onnx"
    weights.touch()
    config = load_config()["detector"]
    config["weights"] = str(weights)
    session = Mock()
    session.get_inputs.return_value = [SimpleNamespace(
        name="images", shape=[1, 3, 640, 640], type="tensor(float)"
    )]
    raw = np.zeros((1, 8400, 85), dtype=np.float32)
    raw[0, 0, :4] = [12.5, 12.5, np.log(5), np.log(7.5)]
    raw[0, 0, 4] = 0.8
    raw[0, 0, 7] = 0.9  # COCO car index 2.
    session.run.return_value = [raw]
    factory = Mock(return_value=session)
    monkeypatch.setattr(onnxruntime, "InferenceSession", factory)
    detector = YOLOXDetector(config)
    result = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
    assert len(result) == 1
    assert result[0].class_name == "car"
    assert result[0].bbox == pytest.approx((80, 70, 120, 130))
    assert factory.call_args.kwargs["providers"] == ["CPUExecutionProvider"]
    session.run.side_effect = RuntimeError("backend error")
    with pytest.raises(RuntimeError, match="YOLOX inference failed"):
        detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
