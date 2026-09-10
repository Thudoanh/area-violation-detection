"""YOLOX-S official 0.1.1rc0 ONNX export, CPU inference only.

Input: float32 BGR 0..255, NCHW, 640x640, top-left letterbox (114).
Output: raw (1, 8400, 85), decoded here with strides 8/16/32.
This adapter is independently implemented; no training package is required.
"""

import math
from pathlib import Path

import cv2
import numpy as np

from src.core.models import Detection
from src.detection.base import BaseDetector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Stable project IDs, independent of config order and detector-native IDs.
CLASS_NAMES = ("person", "motorcycle", "car", "bicycle", "bus", "truck")
COCO_TO_PROJECT = {0: 0, 3: 1, 2: 2, 1: 3, 5: 4, 7: 5}
INPUT_SIZE = 640


def validate_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("Config must provide a detector mapping")
    if config.get("backend") != "yolox" or config.get("model") != "yolox_s":
        raise ValueError("Phase 2 supports detector.backend=yolox and model=yolox_s")
    if config.get("device", "cpu") != "cpu":
        raise ValueError("This YOLOX ONNX baseline supports device=cpu only")
    for key in ("confidence_threshold", "nms_threshold"):
        value = config.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or not 0 < value <= 1):
            raise ValueError(f"detector.{key} must be a number in (0, 1]")
    classes = config.get("detection_classes")
    if (not isinstance(classes, list) or not classes or
            any(not isinstance(name, str) or name not in CLASS_NAMES for name in classes)):
        raise ValueError(f"detector.detection_classes must be a nonempty list from {CLASS_NAMES}")
    if len(classes) != len(set(classes)):
        raise ValueError("detector.detection_classes must not contain duplicates")
    if not isinstance(config.get("weights"), str) or not config["weights"].strip():
        raise ValueError("detector.weights must be a local ONNX path")


def preprocess(frame: np.ndarray) -> tuple[np.ndarray, float]:
    if (not isinstance(frame, np.ndarray) or frame.ndim != 3 or
            frame.shape[2] != 3 or frame.size == 0 or frame.dtype != np.uint8):
        raise ValueError("Expected a nonempty uint8 OpenCV BGR frame with 3 channels")
    height, width = frame.shape[:2]
    scale = INPUT_SIZE / max(height, width)
    resized = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    canvas[:resized.shape[0], :resized.shape[1]] = resized
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32), scale


def decode_output(raw: np.ndarray) -> np.ndarray:
    """Decode raw grid offsets to center-x, center-y, width, height pixels."""
    if raw.shape != (1, 8400, 85):
        raise ValueError(f"Expected raw YOLOX-S output (1, 8400, 85), got {raw.shape}")
    decoded = raw[0].copy()
    offset = 0
    for stride in (8, 16, 32):
        side = INPUT_SIZE // stride
        count = side * side
        rows = decoded[offset:offset + count]
        cells = np.arange(count)
        rows[:, 0] = (rows[:, 0] + cells % side) * stride
        rows[:, 1] = (rows[:, 1] + cells // side) * stride
        with np.errstate(over="ignore", invalid="ignore"):
            rows[:, 2:4] = np.exp(rows[:, 2:4]) * stride
        offset += count
    return decoded


def to_detections(decoded: np.ndarray, scale: float, frame_shape: tuple,
                  classes: list[str], confidence_threshold: float,
                  nms_threshold: float) -> list[Detection]:
    """Convert decoded COCO rows, filter classes, and apply per-class box NMS.

    Confidence is objectness * best COCO class probability. Non-target winners
    are discarded, not relabeled as a weaker target class. NMS is per frame.
    """
    native_ids = decoded[:, 5:].argmax(axis=1)
    scores = decoded[:, 4] * decoded[np.arange(len(decoded)), 5 + native_ids]
    height, width = frame_shape[:2]
    candidates = []
    for row, native_id, score in zip(decoded, native_ids, scores):
        project_id = COCO_TO_PROJECT.get(int(native_id))
        if project_id is None or CLASS_NAMES[project_id] not in classes:
            continue
        if not np.isfinite(row).all() or not confidence_threshold <= score <= 1:
            continue
        cx, cy, box_width, box_height = row[:4] / scale
        x1, x2 = np.clip([cx - box_width / 2, cx + box_width / 2], 0, width - 1)
        y1, y2 = np.clip([cy - box_height / 2, cy + box_height / 2], 0, height - 1)
        if x2 <= x1 or y2 <= y1:
            continue
        candidates.append(Detection(
            tuple(float(v) for v in (x1, y1, x2, y2)), float(score),
            project_id, CLASS_NAMES[project_id],
        ))
    detections = []
    for name in classes:
        group = [d for d in candidates if d.class_name == name]
        if not group:
            continue
        boxes = [[d.bbox[0], d.bbox[1], d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]]
                 for d in group]
        keep = cv2.dnn.NMSBoxes(boxes, [d.confidence for d in group], 0.0, nms_threshold)
        detections.extend(group[int(i)] for i in np.asarray(keep).reshape(-1))
    return sorted(detections, key=lambda d: d.confidence, reverse=True)


class YOLOXDetector(BaseDetector):
    def __init__(self, config: dict):
        validate_config(config)
        self.classes = list(config["detection_classes"])
        self.confidence_threshold = float(config["confidence_threshold"])
        self.nms_threshold = float(config["nms_threshold"])
        weights = Path(config["weights"])
        if not weights.is_absolute():
            weights = PROJECT_ROOT / weights
        if not weights.is_file():
            raise FileNotFoundError(
                f"YOLOX weights not found: {weights}. Download official yolox_s.onnx "
                "(0.1.1rc0) using the README instructions."
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX Runtime is unavailable; install requirements.txt") from exc
        try:
            self._session = ort.InferenceSession(str(weights), providers=["CPUExecutionProvider"])
        except Exception as exc:
            raise RuntimeError(f"Cannot load YOLOX ONNX weights {weights}: {exc}") from exc
        inputs = self._session.get_inputs()
        if (len(inputs) != 1 or inputs[0].shape != [1, 3, 640, 640] or
                inputs[0].type != "tensor(float)"):
            raise ValueError("Expected official YOLOX-S float32 input [1, 3, 640, 640]")
        self._input_name = inputs[0].name

    def detect(self, frame: np.ndarray) -> list[Detection]:
        tensor, scale = preprocess(frame)
        try:
            raw = self._session.run(None, {self._input_name: tensor})[0]
        except Exception as exc:
            raise RuntimeError(f"YOLOX inference failed: {exc}") from exc
        return to_detections(decode_output(raw), scale, frame.shape, self.classes,
                             self.confidence_threshold, self.nms_threshold)
