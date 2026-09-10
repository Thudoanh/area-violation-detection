from src.violation.dwell_timer import DwellTimer
from src.violation.stationary import Motion
from tests.event_helpers import membership


def test_zone_residence_is_not_stationary_dwell():
    timer = DwellTimer()
    zone = membership()
    timer.update(1, zone, Motion(), 10)
    result = timer.update(1, zone, Motion(), 20)
    assert result.entered_at == 10
    assert result.zone_duration_sec == 10
    assert result.dwell_time_sec == 0
    result = timer.update(1, zone, Motion(True, 18), 21)
    assert result.stationary_since == 18
    assert result.stationary_duration_sec == 3
    assert timer.update(1, zone, Motion(True, 19), 22).dwell_time_sec == 4
    assert timer.update(1, zone, Motion(), 23).stationary_since is None


def test_exclusions_exit_and_zone_change_reset():
    timer = DwellTimer()
    zone = membership()
    for excluded in (membership(()), membership(("SIDEWALK", "ALLOWED")), membership(("IGNORE",))):
        timer.update(1, zone, Motion(True, 0), 0)
        assert timer.update(1, excluded, Motion(True, 0), 10).dwell_time_sec == 0
        assert 1 not in timer.tracks
    timer.update(1, zone, Motion(), 10)
    new_zone = membership(("MONITORED",))
    new_zone.effective_zone.zone_id = "other"
    result = timer.update(1, new_zone, Motion(True, 0), 15)
    assert result.entered_at == result.stationary_since == 15
    assert result.dwell_time_sec == 0
