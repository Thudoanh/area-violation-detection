"""Ultralytics YOLO11n adapter to the existing Detection contract."""

import math
from pathlib import Path

import numpy as np

from src.core.models import Detection
from src.detection.base import BaseDetector

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = ("person", "motorcycle", "car", "bicycle", "bus", "truck")
COCO_TO_PROJECT = {0: 0, 3: 1, 2: 2, 1: 3, 5: 4, 7: 5}


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Config must provide a detector mapping")
    if config.get("backend") != "ultralytics" or config.get("model") != "yolo11n":
        raise ValueError("Expected detector.backend=ultralytics and model=yolo11n")
    for key in ("confidence_threshold", "nms_threshold"):
        value = config.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 < value <= 1):
            raise ValueError(f"detector.{key} must be a number in (0, 1]")
    classes = config.get("detection_classes")
    if (not isinstance(classes, list) or not classes
            or any(not isinstance(c, str) or c not in CLASS_NAMES for c in classes)
            or len(classes) != len(set(classes))):
        raise ValueError("detector.detection_classes must contain unique supported class names")
    if not isinstance(config.get("weights"), str) or not config["weights"].strip():
        raise ValueError("detector.weights must be a local .pt path")
    if Path(config["weights"]).suffix.lower() != ".pt":
        raise ValueError("detector.weights must be a local .pt path")
    if not isinstance(config.get("device"), str) or not config["device"].strip():
        raise ValueError("detector.device must be a nonempty string")
    size = config.get("input_size", 640)
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0 or size % 32:
        raise ValueError("detector.input_size must be a positive multiple of 32")


def to_detections(result, frame_shape, classes, confidence_threshold):
    """Results.xyxy is already in original-image pixels; never rescale twice."""
    if result.boxes is None:
        return []
    boxes = result.boxes
    xyxy = boxes.xyxy.detach().cpu().numpy()
    scores = boxes.conf.detach().cpu().numpy()
    ids = boxes.cls.detach().cpu().numpy()
    if xyxy.shape != (len(scores), 4) or ids.shape != scores.shape or scores.ndim != 1:
        raise ValueError("Unexpected YOLO11 detection output shape")
    height, width = frame_shape[:2]
    detections = []
    for box, score, native_id in zip(xyxy, scores, ids):
        if (not np.isfinite(box).all() or not np.isfinite(native_id)
                or native_id != int(native_id) or not confidence_threshold <= score <= 1):
            continue
        project_id = COCO_TO_PROJECT.get(int(native_id))
        if project_id is None or CLASS_NAMES[project_id] not in classes:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        x1, x2 = np.clip([x1, x2], 0, width - 1)
        y1, y2 = np.clip([y1, y2], 0, height - 1)
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(Detection(tuple(float(v) for v in (x1, y1, x2, y2)),
                                    float(score), project_id, CLASS_NAMES[project_id]))
    return sorted(detections, key=lambda d: d.confidence, reverse=True)


class YOLO11Detector(BaseDetector):
    def __init__(self, config: dict):
        validate_config(config)
        self.config = dict(config)
        self.classes = list(config["detection_classes"])
        weights = Path(config["weights"])
        if not weights.is_absolute():
            weights = PROJECT_ROOT / weights
        if not weights.is_file():
            raise FileNotFoundError(f"YOLO11 weights not found: {weights}. Download yolo11n.pt as described in README.")
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(weights), task="detect")
        except Exception as exc:
            raise RuntimeError(f"Cannot load YOLO11 weights {weights}: {exc}") from exc
        if self._model.task != "detect" or any(
            self._model.names.get(native) != CLASS_NAMES[project]
            for native, project in COCO_TO_PROJECT.items()
        ):
            raise ValueError("Expected YOLO11 COCO detection weights with canonical class names")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if (not isinstance(frame, np.ndarray) or frame.ndim != 3
                or frame.shape[2] != 3 or frame.size == 0 or frame.dtype != np.uint8):
            raise ValueError("Expected a nonempty uint8 OpenCV BGR frame with 3 channels")
        try:
            results = self._model.predict(
                source=frame, conf=self.config["confidence_threshold"],
                iou=self.config["nms_threshold"], device=self.config["device"],
                imgsz=self.config.get("input_size", 640), rect=False,
                classes=[native for native, project in COCO_TO_PROJECT.items()
                         if CLASS_NAMES[project] in self.classes],
                agnostic_nms=False, verbose=False, save=False, stream=False,
            )
            if len(results) != 1:
                raise ValueError("Expected one YOLO11 result per frame")
            return to_detections(results[0], frame.shape, self.classes,
                                 self.config["confidence_threshold"])
        except Exception as exc:
            raise RuntimeError(f"YOLO11 inference failed: {exc}") from exc
