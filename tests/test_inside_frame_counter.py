from src.violation.inside_frame_counter import InsideFrameCounter
from tests.event_helpers import membership


def test_counts_consecutive_inside_frames_and_resets_outside():
    counter = InsideFrameCounter()
    zone = membership()
    assert counter.update(1, zone, 10).consecutive_frames == 1
    status = counter.update(1, zone, 11)
    assert status.consecutive_frames == 2
    assert status.entered_at == 10
    assert counter.update(1, membership(()), 12).consecutive_frames == 0
    assert counter.update(1, zone, 13).consecutive_frames == 1


def test_excluded_or_changed_zone_starts_a_new_count():
    counter = InsideFrameCounter()
    counter.update(1, membership(), 1)
    assert counter.update(1, membership(("SIDEWALK", "ALLOWED")), 2).consecutive_frames == 0
    other = membership(("MONITORED",))
    other.effective_zone.zone_id = "other"
    assert counter.update(1, other, 3).consecutive_frames == 1
    assert counter.update(1, other, 4).consecutive_frames == 2
