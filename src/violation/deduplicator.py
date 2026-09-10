"""Run-scoped track locks and recent spatial events; no ReID."""

from dataclasses import dataclass
from itertools import combinations
from math import dist
from typing import Optional

from src.core.models import ViolationEvent
from src.violation.settings import number


@dataclass
class Record:
    event: ViolationEvent
    center: tuple
    closed_at: Optional[float] = None


class Deduplicator:
    def __init__(self, cooldown_sec, spatial_distance_px):
        self.cooldown = number(cooldown_sec, "dedup.cooldown_sec")
        self.distance = number(spatial_distance_px, "dedup.spatial_distance_px")
        self.records = {}
        self.distinct = set()

    def observe(self, track_ids):
        # Simultaneously observed IDs represent distinct objects, even if close.
        self.distinct.update(frozenset(pair) for pair in combinations(track_ids, 2))

    def allow(self, candidate, center):
        for event_id, record in list(self.records.items()):
            if record.closed_at is not None and candidate.violation_at - record.closed_at >= self.cooldown:
                del self.records[event_id]
                continue
            old = record.event
            if (old.run_id, old.camera_id, old.zone_id) != (candidate.run_id, candidate.camera_id, candidate.zone_id):
                continue
            if record.closed_at is None and old.track_id == candidate.track_id:
                return False
            if old.object_class != candidate.object_class:
                continue
            if frozenset((old.track_id, candidate.track_id)) in self.distinct:
                continue
            if dist(record.center, center) < self.distance:
                return False
        return True

    def register(self, event, center):
        self.records[event.event_id] = Record(event, center)

    def close(self, event_id, timestamp):
        if event_id in self.records:
            self.records[event_id].closed_at = timestamp
