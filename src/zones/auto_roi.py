"""Classical ROI suggestion for stable painted or hatched areas in fixed video."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class AutoROISettings:
    num_frames: int = 15
    vote_ratio: float = 0.6
    min_area_ratio: float = 0.005
    simplify_epsilon_ratio: float = 0.015
    min_saturation: int = 0
    max_saturation: int = 120
    min_value: int = 150
    min_line_angle_deg: float = 15.0
    max_line_angle_deg: float = 75.0

    def __post_init__(self):
        if self.num_frames < 1:
            raise ValueError("num_frames must be positive")
        for name in ("vote_ratio", "min_area_ratio", "simplify_epsilon_ratio"):
            value = getattr(self, name)
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")


class ClassicalAutoROIDetector:
    """Suggest one polygon from diagonal, light painted markings.

    This is deliberately a suggestion backend. It does not infer the meaning of
    an arbitrary sidewalk or monitored area; the web UI always previews its
    polygon and lets the operator correct it before inference.
    """

    def __init__(self, settings: AutoROISettings | None = None):
        self.settings = settings or AutoROISettings()

    def _candidate_mask(self, frame: np.ndarray) -> np.ndarray:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a BGR video frame")
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bright = cv2.inRange(
            hsv,
            (0, self.settings.min_saturation, self.settings.min_value),
            (179, self.settings.max_saturation, 255),
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        min_dim = min(height, width)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=max(15, min_dim // 25),
            minLineLength=max(12, min_dim // 18),
            maxLineGap=max(6, min_dim // 80),
        )
        diagonal = np.zeros((height, width), dtype=np.uint8)
        if lines is not None:
            thickness = max(5, min_dim // 80)
            for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
                angle = min(angle, 180 - angle)
                if self.settings.min_line_angle_deg <= angle <= self.settings.max_line_angle_deg:
                    cv2.line(diagonal, (x1, y1), (x2, y2), 255, thickness)
        reach = max(7, min_dim // 35)
        diagonal = cv2.dilate(
            diagonal,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach, reach)),
        )
        return cv2.bitwise_and(bright, diagonal)

    def detect_from_frames(self, frames: list[np.ndarray]) -> list[tuple[int, int]]:
        if not frames:
            raise ValueError("At least one frame is required for automatic ROI")
        shape = frames[0].shape
        if any(frame.shape != shape for frame in frames):
            raise ValueError("Automatic ROI frames must have the same resolution")
        votes = np.zeros(shape[:2], dtype=np.uint16)
        for frame in frames:
            votes += (self._candidate_mask(frame) > 0).astype(np.uint16)
        required = max(1, int(np.ceil(len(frames) * self.settings.vote_ratio)))
        stable = np.where(votes >= required, 255, 0).astype(np.uint8)

        min_dim = min(shape[:2])
        bridge = max(9, min_dim // 18)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge, bridge))
        stable = cv2.morphologyEx(stable, cv2.MORPH_CLOSE, kernel, iterations=2)
        stable = cv2.dilate(stable, kernel, iterations=1)
        contours, _ = cv2.findContours(stable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        minimum_area = shape[0] * shape[1] * self.settings.min_area_ratio
        contours = [contour for contour in contours if cv2.contourArea(contour) >= minimum_area]
        if not contours:
            return []
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        epsilon = self.settings.simplify_epsilon_ratio * cv2.arcLength(hull, True)
        polygon = cv2.approxPolyDP(hull, epsilon, True).reshape(-1, 2)
        if len(polygon) < 3:
            return []
        return [(int(x), int(y)) for x, y in polygon]

    def detect_from_video(self, video_path) -> tuple[np.ndarray, list[tuple[int, int]]]:
        path = Path(video_path)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"Không thể mở video: {path}")
        try:
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if total > 0:
                indices = np.linspace(0, max(0, total - 1),
                                      min(self.settings.num_frames, total), dtype=int)
                frames = []
                for index in dict.fromkeys(int(value) for value in indices):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, frame = capture.read()
                    if ok:
                        frames.append(frame)
            else:
                frames = []
                while len(frames) < self.settings.num_frames:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frames.append(frame)
            if not frames:
                raise ValueError("Video không có khung hình đọc được")
            return frames[0], self.detect_from_frames(frames)
        finally:
            capture.release()
