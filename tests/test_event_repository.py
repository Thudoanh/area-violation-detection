import sqlite3

import cv2
import numpy as np
import pytest

from src.storage.event_repository import EventRepository
from src.storage.evidence_writer import EvidenceWriter
from tests.event_helpers import event


def test_snapshot_and_repository_lifecycle(tmp_path):
    repository = EventRepository(tmp_path / "events.db")
    try:
        candidate = event()
        candidate.snapshot_path = EvidenceWriter(tmp_path / "ảnh").save(np.zeros((30, 40, 3), np.uint8), candidate)
        assert cv2.imdecode(np.fromfile(candidate.snapshot_path, dtype=np.uint8), cv2.IMREAD_COLOR).shape == (30, 40, 3)
        repository.create(candidate)
        assert repository.get("e1")["status"] == "OPEN"
        assert repository.get("e1")["left_at"] is None
        with pytest.raises(sqlite3.IntegrityError):
            repository.create(candidate)
        repository.close_event("e1", 40)
        assert repository.get("e1")["status"] == "CLOSED"
        assert repository.get("e1")["left_at"] == 40
        assert repository.get("absent") is None
    finally:
        repository.close()


def test_no_insert_without_evidence(tmp_path):
    repository = EventRepository(tmp_path / "events.db")
    try:
        with pytest.raises(ValueError, match="snapshot"):
            repository.create(event())
        assert repository.get("e1") is None
    finally:
        repository.close()


def test_list_by_run_is_isolated_and_sorted(tmp_path):
    repository = EventRepository(tmp_path / "events.db")
    writer = EvidenceWriter(tmp_path / "snapshots")
    try:
        late = event("late", timestamp=20)
        early = event("early", timestamp=10)
        other = event("other", timestamp=5)
        other.run_id = "r2"
        for candidate in (late, early, other):
            candidate.snapshot_path = writer.save(
                np.zeros((10, 10, 3), np.uint8), candidate
            )
            repository.create(candidate)
        assert [row["event_id"] for row in repository.list_by_run("r1")] == [
            "early", "late"
        ]
        assert [row["event_id"] for row in repository.list_by_run("r2")] == ["other"]
        assert repository.list_by_run("missing") == []
        with pytest.raises(ValueError, match="run_id"):
            repository.list_by_run("")
    finally:
        repository.close()
