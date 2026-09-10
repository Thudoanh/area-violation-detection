"""Per-track bottom-center displacement over a sliding media-time window."""

from collections import deque
from dataclasses import dataclass
from math import dist
from typing import Optional

from src.zones.geometry import get_bottom_center
from src.violation.settings import number


@dataclass
class Motion:
    stationary: bool = False
    window_start: Optional[float] = None


class StationaryDetector:
    def __init__(self, window_sec, max_displacement_px):
        self.window_sec = number(window_sec, "stationary.window_sec", positive=True)
        self.max_displacement_px = number(max_displacement_px, "stationary.max_displacement_px")
        self.history = {}

    def reset(self, track_id):
        self.history.pop(track_id, None)

    def update(self, track) -> Motion:
        timestamp = number(track.timestamp, "timestamp")
        history = self.history.setdefault(track.track_id, deque())
        if history and timestamp <= history[-1][0]:
            raise ValueError("Track timestamps must increase")
        history.append((timestamp, get_bottom_center(track.bbox)))
        cutoff = timestamp - self.window_sec
        # Keep one observation at/before cutoff so irregular FPS can span the window.
        while len(history) > 1 and history[1][0] <= cutoff:
            history.popleft()
        if timestamp - history[0][0] < self.window_sec:
            return Motion()
        stationary = dist(history[0][1], history[-1][1]) <= self.max_displacement_px
        return Motion(stationary, history[0][0] if stationary else None)
