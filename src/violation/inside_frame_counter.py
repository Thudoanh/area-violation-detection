"""Count consecutive observed frames inside one effective target zone."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class InsideFrameStatus:
    zone_id: Optional[str] = None
    entered_at: Optional[float] = None
    consecutive_frames: int = 0


class InsideFrameCounter:
    """Maintain a run-local consecutive-inside count for every track ID."""

    def __init__(self):
        self.tracks = {}

    def reset(self, track_id):
        self.tracks.pop(track_id, None)

    def update(self, track_id, membership, timestamp) -> InsideFrameStatus:
        zone = membership.target_zone
        if zone is None:
            self.reset(track_id)
            return InsideFrameStatus()
        previous = self.tracks.get(track_id)
        if previous is None or previous.zone_id != zone.zone_id:
            result = InsideFrameStatus(zone.zone_id, timestamp, 1)
        else:
            result = InsideFrameStatus(
                zone.zone_id, previous.entered_at, previous.consecutive_frames + 1
            )
        self.tracks[track_id] = result
        return result
