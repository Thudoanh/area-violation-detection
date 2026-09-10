"""Tracker contract from Solution Design, including video timestamp."""

from src.core.models import Detection, TrackedObject


class BaseTracker:
    def update(self, detections: list[Detection], frame,
               timestamp: float) -> list[TrackedObject]:
        raise NotImplementedError
