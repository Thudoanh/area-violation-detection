"""Stationary dwell in the effective target zone, separate from zone residence."""

from dataclasses import dataclass
from typing import Optional

from src.violation.settings import number


@dataclass
class Dwell:
    zone_id: Optional[str] = None
    entered_at: Optional[float] = None
    stationary_since: Optional[float] = None
    dwell_time_sec: float = 0.0
    zone_duration_sec: float = 0.0

    @property
    def stationary_duration_sec(self):
        return self.dwell_time_sec


class DwellTimer:
    def __init__(self):
        self.tracks = {}

    def reset(self, track_id):
        self.tracks.pop(track_id, None)

    def update(self, track_id, membership, motion, timestamp) -> Dwell:
        number(timestamp, "timestamp")
        zone = membership.target_zone
        if zone is None:
            self.reset(track_id)
            return Dwell()
        previous = self.tracks.get(track_id)
        if previous is None or previous.zone_id != zone.zone_id:
            previous = Dwell(zone.zone_id, timestamp)
        since = previous.stationary_since
        if not motion.stationary:
            since = None
        elif since is None:
            since = max(previous.entered_at, motion.window_start)
        result = Dwell(zone.zone_id, previous.entered_at, since,
                       timestamp - since if since is not None else 0.0,
                       timestamp - previous.entered_at)
        self.tracks[track_id] = result
        return result
