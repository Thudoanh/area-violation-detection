"""Temporal/event orchestration for one video, after detection and tracking."""

from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
from uuid import uuid4

from src.core.visualization import draw_temporal
from src.storage.event_repository import EventRepository
from src.storage.evidence_writer import EvidenceWriter
from src.violation.inside_frame_counter import InsideFrameCounter, InsideFrameStatus
from src.violation.state_machine import StateMachine
from src.violation.violation_engine import ViolationEngine
from src.violation.deduplicator import Deduplicator
from src.violation.settings import number
from src.zones.geometry import get_bottom_center

ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def project_path(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Storage/config paths must be nonempty strings")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


class AreaMonitoringPipeline:
    def __init__(self, config, video_path, zone_manager, run_id=None):
        for key in ("violation", "dedup", "storage"):
            if not isinstance(config.get(key), dict):
                raise ValueError(f"Config must provide {key} mapping")
        violation, dedup = (config[k] for k in ("violation", "dedup"))
        self.inside_frames = InsideFrameCounter()
        self.states = StateMachine(violation.get("min_inside_frames"), violation.get("exit_grace_sec"))
        self.engine = ViolationEngine(violation)
        self.dedup = Deduplicator(dedup.get("cooldown_sec"), dedup.get("spatial_distance_px"))
        version = json.dumps({"config": config, "zones": [vars(z) for z in zone_manager.zones]}, sort_keys=True)
        model = config["detector"]["model"]
        weights = project_path(config["detector"]["weights"])
        if weights.is_file():
            digest = hashlib.sha256()
            with weights.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            model += ":sha256:" + digest.hexdigest()
        if run_id is not None and (not isinstance(run_id, str) or not run_id.isalnum()):
            raise ValueError("run_id must be alphanumeric")
        self.context = dict(run_id=run_id or uuid4().hex, video_id=str(Path(video_path).resolve()),
                            camera_id=zone_manager.camera_id,
                            config_version=hashlib.sha256(version.encode()).hexdigest(), model_version=model)
        self.evidence = EvidenceWriter(project_path(config["storage"].get("snapshot_dir")))
        self.repository = EventRepository(project_path(config["storage"].get("sqlite_path")))
        self.stats = Counter()
        self.qualified_track_ids = set()
        self.last_timestamp = None

    def _flush_closed(self):
        while self.states.closed:
            episode = self.states.closed[0]
            if episode.event_id is not None:
                try:
                    self.repository.close_event(episode.event_id, episode.left_at)
                except Exception:
                    logger.exception("Could not close event %s", episode.event_id)
                    raise
                self.dedup.close(episode.event_id, episode.closed_at)
            self.states.closed.pop(0)

    def process(self, tracks, memberships, timestamp, annotated):
        number(timestamp, "timestamp")
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("Pipeline frame timestamps must increase")
        if len({t.track_id for t in tracks}) != len(tracks):
            raise ValueError("Duplicate track IDs in one frame")
        if any(t.timestamp != timestamp for t in tracks):
            raise ValueError("TrackedObject timestamp must match the current video frame")
        self.last_timestamp = timestamp
        visible = {t.track_id for t in tracks}
        self.dedup.observe(visible)
        for missing in set(self.inside_frames.tracks) - visible:
            self.inside_frames.reset(missing)
        for missing in set(self.states.episodes) - visible:
            self.states.update(missing, None, timestamp, observed=False)
        statuses = {}
        candidates = []
        for track in tracks:
            membership = memberships[track.track_id]
            zone = membership.target_zone if track.class_name in self.engine.targets else None
            inside = InsideFrameStatus()
            if zone:
                inside = self.inside_frames.update(track.track_id, membership, timestamp)
                if inside.consecutive_frames >= self.engine.threshold:
                    self.qualified_track_ids.add(track.track_id)
            else:
                self.inside_frames.reset(track.track_id)
            state = self.states.update(track.track_id, zone.zone_id if zone else None, timestamp,
                                       inside.consecutive_frames)
            statuses[track.track_id] = (inside, state)
            candidate = self.engine.evaluate(track, membership, inside, state, **self.context)
            if candidate:
                candidates.append((candidate, get_bottom_center(track.bbox)))
        self._flush_closed()
        annotated = draw_temporal(annotated, tracks, statuses)
        for candidate, center in candidates:
            self.stats["candidates"] += 1
            if not self.dedup.allow(candidate, center):
                self.stats["suppressed"] += 1
                continue
            try:
                candidate.snapshot_path = self.evidence.save(annotated, candidate)
                self.repository.create(candidate)
            except Exception:
                logger.exception("Could not persist event %s; track is not ALERTED", candidate.event_id)
                # A failed insert must not leave an orphan snapshot or lock.
                if candidate.snapshot_path:
                    try:
                        Path(candidate.snapshot_path).unlink(missing_ok=True)
                    except OSError:
                        logger.exception("Could not remove orphan snapshot %s", candidate.snapshot_path)
                raise
            self.dedup.register(candidate, center)
            self.states.mark_alerted(candidate.track_id, candidate.event_id)
            self.stats["accepted"] += 1
            self.stats["snapshots"] += 1
        return annotated

    def finish(self):
        """Close this run's episodes at EOF/abort; unknown departure remains NULL."""
        try:
            self.states.finish(self.last_timestamp if self.last_timestamp is not None else 0.0)
            self._flush_closed()
        finally:
            self.repository.close()
