"""Per-track episode lifecycle. Grace retains locks, never temporal evidence."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.violation.settings import number


class State(str, Enum):
    OUTSIDE = "OUTSIDE"
    ENTERING = "ENTERING"
    INSIDE_MOVING = "INSIDE_MOVING"
    STATIONARY = "STATIONARY"
    SUSPECTED_VIOLATION = "SUSPECTED_VIOLATION"
    ALERTED = "ALERTED"
    CLOSED = "CLOSED"


@dataclass
class Episode:
    track_id: int
    zone_id: str
    entered_at: float
    state: State = State.ENTERING
    absent_since: Optional[float] = None
    left_at: Optional[float] = None
    event_id: Optional[str] = None
    closed_at: Optional[float] = None


class StateMachine:
    def __init__(self, min_dwell_time_sec, exit_grace_sec):
        self.threshold = number(min_dwell_time_sec, "violation.min_dwell_time_sec", positive=True)
        self.grace = number(exit_grace_sec, "violation.exit_grace_sec")
        self.episodes = {}
        self.closed = []

    def close(self, track_id, timestamp, left_at=None):
        episode = self.episodes.pop(track_id)
        episode.state = State.CLOSED
        episode.closed_at = timestamp
        episode.left_at = left_at
        self.closed.append(episode)

    def update(self, track_id, zone_id, timestamp, stationary=False, dwell=0.0,
               observed=True) -> State:
        number(timestamp, "timestamp")
        episode = self.episodes.get(track_id)
        if episode and episode.absent_since is not None and timestamp - episode.absent_since >= self.grace:
            left_at = episode.left_at
            if left_at is None and observed and zone_id != episode.zone_id:
                left_at = timestamp
            self.close(track_id, timestamp, left_at)
            episode = None
            if zone_id is None:
                return State.CLOSED
        if episode and zone_id is not None and zone_id != episode.zone_id:
            self.close(track_id, timestamp, timestamp)
            episode = None
        if zone_id is None:
            if episode:
                if episode.absent_since is None:
                    episode.absent_since = timestamp
                if observed and episode.left_at is None:
                    episode.left_at = timestamp
                if timestamp - episode.absent_since >= self.grace:
                    self.close(track_id, timestamp, episode.left_at)
                    return State.CLOSED
                if episode.event_id is None:
                    episode.state = State.OUTSIDE
                return episode.state
            return State.OUTSIDE
        if episode is None:
            self.episodes[track_id] = Episode(track_id, zone_id, timestamp)
            return State.ENTERING
        episode.absent_since = episode.left_at = None
        if episode.event_id is not None:
            episode.state = State.ALERTED
        elif stationary and dwell >= self.threshold:
            episode.state = State.SUSPECTED_VIOLATION
        elif stationary:
            episode.state = State.STATIONARY
        else:
            episode.state = State.INSIDE_MOVING
        return episode.state

    def mark_alerted(self, track_id, event_id):
        episode = self.episodes[track_id]
        episode.event_id = event_id
        episode.state = State.ALERTED

    def finish(self, timestamp):
        for track_id, episode in list(self.episodes.items()):
            self.close(track_id, timestamp, episode.left_at)
