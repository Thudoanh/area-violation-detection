from src.violation.state_machine import StateMachine, State


def test_transitions_and_track_isolation():
    machine = StateMachine(30, 3)
    assert machine.update(1, None, 0) == State.OUTSIDE
    assert machine.update(1, "SW", 1) == State.ENTERING
    assert machine.update(1, "SW", 2, 2) == State.INSIDE_PENDING
    assert machine.update(1, "SW", 4, 3) == State.INSIDE_PENDING
    assert machine.update(1, "SW", 5, 4) == State.INSIDE_PENDING
    assert machine.update(2, "SW", 5) == State.ENTERING
    assert machine.update(1, "SW", 40, 30) == State.SUSPECTED_VIOLATION
    assert machine.episodes[2].state == State.ENTERING
    machine.mark_alerted(1, "event")
    assert machine.update(1, "SW", 41) == State.ALERTED


def test_grace_reentry_expiry_and_eof():
    machine = StateMachine(30, 3)
    machine.update(1, "SW", 0)
    machine.mark_alerted(1, "event")
    machine.update(1, None, 1, observed=False)
    assert machine.update(1, "SW", 2) == State.ALERTED
    machine.update(1, None, 3)
    assert machine.update(1, None, 6) == State.CLOSED
    assert machine.closed[0].left_at == 3
    assert machine.closed[0].closed_at == 6
    machine.update(1, "SW", 7)
    machine.finish(8)
    assert machine.closed[-1].left_at is None


def test_zone_switch_and_reentry_after_grace_start_new_episode():
    machine = StateMachine(30, 3)
    machine.update(1, "SW", 0)
    assert machine.update(1, "MON", 1) == State.ENTERING
    assert machine.closed[0].zone_id == "SW"
    machine.update(1, None, 2, observed=False)
    assert machine.update(1, "MON", 6) == State.ENTERING
    assert machine.closed[-1].left_at is None


def test_observed_exit_at_grace_expiry_after_missing_track():
    machine = StateMachine(30, 3)
    machine.update(1, "SW", 0)
    machine.update(1, None, 1, observed=False)
    assert machine.update(1, None, 4, observed=True) == State.CLOSED
    assert machine.closed[-1].left_at == 4


def test_one_frame_threshold_can_trigger_on_entry_frame():
    machine = StateMachine(1, 3)
    assert machine.update(1, "SW", 0, 1) == State.SUSPECTED_VIOLATION
