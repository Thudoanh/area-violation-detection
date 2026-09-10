"""Detector contract shared by future backends."""

from src.core.models import Detection


class BaseDetector:
    def detect(self, frame) -> list[Detection]:
        raise NotImplementedError
