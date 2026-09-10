from src.violation.deduplicator import Deduplicator
from tests.event_helpers import event


def test_active_lock_has_no_time_expiry():
    dedup = Deduplicator(60, 50)
    dedup.register(event(), (0, 0))
    assert not dedup.allow(event("e2", timestamp=1000), (500, 500))
    assert not dedup.allow(event("e3", track_id=2, timestamp=1000), (1, 1))


def test_cooldown_from_close_and_spatial_distance():
    dedup = Deduplicator(60, 50)
    dedup.register(event(), (0, 0))
    dedup.close("e1", 100)
    assert not dedup.allow(event("e2", track_id=2, timestamp=159), (1, 1))
    assert dedup.allow(event("e3", track_id=2, timestamp=159), (50, 0))
    assert dedup.allow(event("e4", track_id=2, timestamp=160), (1, 1))


def test_distinct_objects_and_scope():
    dedup = Deduplicator(60, 50)
    dedup.observe({1, 2})
    dedup.register(event(), (0, 0))
    assert dedup.allow(event("e2", track_id=2), (1, 1))
    for field in ("run_id", "camera_id", "zone_id", "object_class"):
        candidate = event("e3", track_id=3)
        setattr(candidate, field, "other")
        assert dedup.allow(candidate, (1, 1))
