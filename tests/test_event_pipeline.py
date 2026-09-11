from pathlib import Path
import sqlite3

import numpy as np
import pytest

from src.pipeline.pipeline import AreaMonitoringPipeline
from src.violation.state_machine import State
from tests.event_helpers import config, manager, track


def step(pipeline, zones, timestamp, tracks):
    memberships = {t.track_id: zones.get_membership(t, "CAM_001") for t in tracks}
    return pipeline.process(tracks, memberships, timestamp, np.zeros((120, 500, 3), np.uint8))


def rows(pipeline):
    return [dict(row) for row in pipeline.repository.connection.execute("SELECT * FROM events")]


def test_one_event_extended_stop_close_and_evidence(tmp_path):
    zones = manager()
    pipeline = AreaMonitoringPipeline(config(tmp_path), "input.mp4", zones)
    try:
        for timestamp in range(35):
            step(pipeline, zones, timestamp, [track(timestamp)])
        assert pipeline.stats["accepted"] == pipeline.stats["snapshots"] == 1
        assert pipeline.stats["candidates"] == 1
        saved, = rows(pipeline)
        assert saved["violation_at"] == 29
        assert saved["inside_frame_count"] == 30
        assert saved["dwell_time_sec"] == 29
        assert saved["stationary_since"] == saved["entered_at"] == 0
        assert saved["status"] == "OPEN"
        assert Path(saved["snapshot_path"]).is_file()
        for timestamp in range(35, 39):
            step(pipeline, zones, timestamp, [track(timestamp, x=400)])
        saved, = rows(pipeline)
        assert saved["status"] == "CLOSED"
        assert saved["left_at"] == 35
    finally:
        pipeline.finish()


@pytest.mark.parametrize("name,types", [
    ("person", ("SIDEWALK",)), ("car", ("SIDEWALK", "ALLOWED")),
    ("car", ("SIDEWALK", "IGNORE")), ("car", ()),
])
def test_no_event_for_context_or_excluded_zone(tmp_path, name, types):
    zones = manager(types)
    pipeline = AreaMonitoringPipeline(config(tmp_path), "input.mp4", zones)
    try:
        for timestamp in range(40):
            step(pipeline, zones, timestamp, [track(timestamp, name=name)])
        assert rows(pipeline) == []
        assert pipeline.stats["candidates"] == 0
    finally:
        pipeline.finish()


def test_missing_track_resets_inside_frame_count_even_inside_grace(tmp_path):
    zones = manager()
    pipeline = AreaMonitoringPipeline(config(tmp_path), "input.mp4", zones)
    try:
        for timestamp in range(29):
            step(pipeline, zones, timestamp, [track(timestamp)])
        step(pipeline, zones, 29, [])
        step(pipeline, zones, 30, [track(30)])
        assert pipeline.inside_frames.tracks[1].consecutive_frames == 1
        for timestamp in range(31, 59):
            step(pipeline, zones, timestamp, [track(timestamp)])
        assert not rows(pipeline)
        step(pipeline, zones, 59, [track(59)])
        saved, = rows(pipeline)
        assert saved["inside_frame_count"] == 30
        assert saved["entered_at"] == 30
    finally:
        pipeline.finish()


def test_two_nearby_cars_get_separate_events(tmp_path):
    zones = manager()
    pipeline = AreaMonitoringPipeline(config(tmp_path), "input.mp4", zones)
    try:
        for timestamp in range(32):
            step(pipeline, zones, timestamp, [track(timestamp), track(timestamp, track_id=2, x=45)])
        assert len(rows(pipeline)) == 2
    finally:
        pipeline.finish()


def test_id_switch_spatial_suppression_then_cooldown_expiry(tmp_path):
    zones = manager()
    pipeline = AreaMonitoringPipeline(config(tmp_path), "input.mp4", zones)
    try:
        for timestamp in range(31):
            step(pipeline, zones, timestamp, [track(timestamp)])
        for timestamp in range(31, 94):
            step(pipeline, zones, timestamp, [track(timestamp, track_id=2)])
        assert len(rows(pipeline)) == 1
        assert pipeline.stats["suppressed"] > 0
        # Old ID first missing at 31, closed at 34; cooldown ends at 94.
        step(pipeline, zones, 94, [track(94, track_id=2)])
        assert len(rows(pipeline)) == 2
    finally:
        pipeline.finish()


@pytest.mark.parametrize("failing_stage", ["snapshot", "database"])
def test_failed_storage_does_not_lock_and_can_retry(tmp_path, monkeypatch, caplog, failing_stage):
    zones = manager()
    pipeline = AreaMonitoringPipeline(config(tmp_path), "input.mp4", zones)
    try:
        for timestamp in range(29):
            step(pipeline, zones, timestamp, [track(timestamp)])
        target, method = ((pipeline.evidence, "save") if failing_stage == "snapshot"
                          else (pipeline.repository, "create"))
        original = getattr(target, method)

        def fail(*args):
            raise OSError("storage failed")

        monkeypatch.setattr(target, method, fail)
        with pytest.raises(OSError, match="storage failed"):
            step(pipeline, zones, 29, [track(29)])
        assert pipeline.states.episodes[1].state == State.SUSPECTED_VIOLATION
        assert not pipeline.dedup.records
        assert not rows(pipeline)
        assert not list((tmp_path / "snapshots").glob("*.jpg"))
        assert "not ALERTED" in caplog.text
        monkeypatch.setattr(target, method, original)
        step(pipeline, zones, 31, [track(31)])
        assert len(rows(pipeline)) == 1
    finally:
        pipeline.finish()


def test_eof_closes_only_this_run_with_unknown_left_at(tmp_path):
    zones = manager()
    first = AreaMonitoringPipeline(config(tmp_path), "input.mp4", zones)
    for timestamp in range(31):
        step(first, zones, timestamp, [track(timestamp)])
    second = AreaMonitoringPipeline(config(tmp_path), "other.mp4", zones)
    second.finish()
    assert rows(first)[0]["status"] == "OPEN"
    first.finish()
    with sqlite3.connect(tmp_path / "events.db") as connection:
        assert connection.execute("SELECT status,left_at FROM events").fetchone() == ("CLOSED", None)


def test_motion_does_not_block_n_frame_validation(tmp_path):
    zones = manager()
    pipeline = AreaMonitoringPipeline(config(tmp_path), "input.mp4", zones)
    try:
        for timestamp in range(40):
            step(pipeline, zones, timestamp, [track(timestamp, x=timestamp * 8)])
        assert len(rows(pipeline)) == 1
    finally:
        pipeline.finish()


def test_failed_close_keeps_lock_until_sqlite_update_succeeds(tmp_path, monkeypatch, caplog):
    zones = manager()
    pipeline = AreaMonitoringPipeline(config(tmp_path), "input.mp4", zones)
    try:
        for timestamp in range(31):
            step(pipeline, zones, timestamp, [track(timestamp)])
        for timestamp in range(31, 34):
            step(pipeline, zones, timestamp, [])
        original = pipeline.repository.close_event

        def fail(*args):
            raise sqlite3.OperationalError("disk failure")

        monkeypatch.setattr(pipeline.repository, "close_event", fail)
        with pytest.raises(sqlite3.OperationalError):
            step(pipeline, zones, 34, [])
        assert rows(pipeline)[0]["status"] == "OPEN"
        assert next(iter(pipeline.dedup.records.values())).closed_at is None
        assert "Could not close event" in caplog.text
        monkeypatch.setattr(pipeline.repository, "close_event", original)
    finally:
        pipeline.finish()
