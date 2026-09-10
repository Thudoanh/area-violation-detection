"""Reusable end-to-end video runner for the CLI and the web demo."""

from collections import Counter
from dataclasses import dataclass, field
import math
from pathlib import Path
from time import perf_counter
from typing import Callable, Optional

import cv2

from src.core.frame_provider import VideoFrameProvider
from src.core.visualization import draw_tracks, draw_zones
from src.detection.yolo11_detector import CLASS_NAMES, YOLO11Detector
from src.pipeline.pipeline import AreaMonitoringPipeline
from src.tracking.bytetrack_tracker import ByteTrackTracker
from src.zones.models import ZONE_TYPES
from src.zones.zone_manager import ZoneManager

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class VideoRunResult:
    run_id: str
    video_path: Path
    output_path: Path
    database_path: Path
    events: list[dict] = field(default_factory=list)
    processed_frames: int = 0
    video_fps: float = 0.0
    duration_sec: float = 0.0
    processing_fps: float = 0.0
    detections: int = 0
    unique_tracks: int = 0
    average_active_tracks: float = 0.0
    stationary_tracks: int = 0
    candidates: int = 0
    accepted: int = 0
    suppressed: int = 0
    snapshots: int = 0
    class_detections: dict[str, int] = field(default_factory=dict)
    class_tracks: dict[str, int] = field(default_factory=dict)
    zone_tracks: dict[str, int] = field(default_factory=dict)


def _resolve_zone_manager(config, provider, supplied):
    camera = config.get("camera")
    if not isinstance(camera, dict) or not isinstance(camera.get("camera_id"), str):
        raise ValueError("Config must provide camera.camera_id")
    if supplied is None:
        zone_config = config.get("zones")
        if not isinstance(zone_config, dict) or not isinstance(zone_config.get("path"), str):
            raise ValueError("Config must provide zones.path")
        zone_path = Path(zone_config["path"])
        if not zone_path.is_absolute():
            zone_path = PROJECT_ROOT / zone_path
        supplied = ZoneManager.load_from_json(zone_path)
    if supplied.camera_id != camera["camera_id"]:
        raise ValueError("Zone file camera_id does not match camera.camera_id")
    supplied.validate_frame_size(provider.width, provider.height)
    return supplied, camera["camera_id"]


def process_video(
    video_path,
    config: dict,
    output_path: Path,
    *,
    zone_manager: Optional[ZoneManager] = None,
    run_id: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    frame_provider_factory=VideoFrameProvider,
    detector_factory=YOLO11Detector,
    tracker_factory=ByteTrackTracker,
    video_writer_factory=cv2.VideoWriter,
) -> VideoRunResult:
    """Process one local video and return run-scoped data for presentation."""
    output_path = Path(output_path)
    input_path = Path(video_path)
    try:
        if output_path.resolve() == input_path.resolve():
            raise ValueError("Output path must differ from input video")
    except OSError:
        pass
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("Output must have an .mp4 extension")

    with frame_provider_factory(video_path) as provider:
        if not math.isfinite(provider.fps) or provider.fps <= 0:
            raise ValueError("Tracking requires a positive video FPS")
        if provider.width < 2 or provider.height < 2:
            raise ValueError("Video resolution is invalid for export")
        zones, camera_id = _resolve_zone_manager(config, provider, zone_manager)
        camera_zones = zones.get_zones(camera_id)
        if not camera_zones:
            print("No zones configured; create ROI with scripts/select_roi.py. All tracks are OUTSIDE.")
        print(f"Video path: {provider.video_path}")
        print(f"Camera ID: {camera_id}")
        print(f"FPS: {provider.fps}")
        print(f"Resolution: {provider.width}x{provider.height}")
        print(f"Total frames (metadata): {provider.total_frames}")

        if progress_callback:
            progress_callback(0, provider.total_frames, "Đang tải mô hình nhận diện...")
        detector = detector_factory(config.get("detector"))
        tracker = tracker_factory(config.get("tracking"), frame_rate=provider.fps)
        event_pipeline = AreaMonitoringPipeline(config, video_path, zones, run_id=run_id)
        actual_run_id = event_pipeline.context["run_id"]
        writer = None
        count = 0
        counts = Counter()
        track_classes = {}
        active_track_count = 0
        zone_track_ids = {kind: set() for kind in ZONE_TYPES}
        start = perf_counter()
        try:
            for _, timestamp_sec, frame in provider:
                detections = detector.detect(frame)
                tracks = tracker.update(detections, frame, timestamp_sec)
                memberships = {
                    track.track_id: zones.get_membership(track, camera_id)
                    for track in tracks
                }
                for track_id, membership in memberships.items():
                    for kind, inside in membership.by_type.items():
                        if inside:
                            zone_track_ids[kind].add(track_id)
                annotated = draw_tracks(draw_zones(frame, camera_zones), tracks, memberships)
                annotated = event_pipeline.process(tracks, memberships, timestamp_sec, annotated)
                if writer is None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = video_writer_factory(
                        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                        provider.fps, (provider.width, provider.height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot open annotated video writer: {output_path}")
                writer.write(annotated)
                count += 1
                counts.update(d.class_name for d in detections)
                active_track_count += len(tracks)
                for track in tracks:
                    track_classes.setdefault(track.track_id, track.class_name)
                if progress_callback:
                    progress_callback(
                        count, provider.total_frames,
                        f"Đang xử lý khung hình {count}/{provider.total_frames or '?'}",
                    )
        finally:
            try:
                if writer is not None:
                    writer.release()
            finally:
                event_pipeline.finish()

        elapsed = perf_counter() - start
        if count == 0:
            raise ValueError("Video contains no decodable frames; no output was created")
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError(f"Annotated video was not written: {output_path}")

        repository = event_pipeline.repository.__class__(event_pipeline.repository.path)
        try:
            events = repository.list_by_run(actual_run_id)
        finally:
            repository.close()
        tracks_per_class = Counter(track_classes.values())
        processing_fps = count / max(elapsed, 1e-9)
        result = VideoRunResult(
            run_id=actual_run_id,
            video_path=input_path.resolve(), output_path=output_path.resolve(),
            database_path=Path(event_pipeline.repository.path).resolve(), events=events,
            processed_frames=count, video_fps=provider.fps,
            duration_sec=count / provider.fps, processing_fps=processing_fps,
            detections=sum(counts.values()), unique_tracks=len(track_classes),
            average_active_tracks=active_track_count / count,
            stationary_tracks=len(event_pipeline.stationary_ids),
            candidates=event_pipeline.stats["candidates"],
            accepted=event_pipeline.stats["accepted"],
            suppressed=event_pipeline.stats["suppressed"],
            snapshots=event_pipeline.stats["snapshots"],
            class_detections={name: counts[name] for name in CLASS_NAMES},
            class_tracks={name: tracks_per_class[name] for name in CLASS_NAMES},
            zone_tracks={kind: len(zone_track_ids[kind]) for kind in ZONE_TYPES},
        )

        print(f"Processed frames: {count}")
        print(f"Average detections/frame: {result.detections / count:.3f}")
        print("Class counts (detections across frames, not unique objects):")
        for name in CLASS_NAMES:
            print(f"{name}: {counts[name]}")
        print(f"Unique track IDs: {result.unique_tracks}")
        print(f"Average active tracks/frame: {result.average_active_tracks:.3f}")
        print("Track counts by first observed class (not guaranteed unique objects):")
        for name in CLASS_NAMES:
            print(f"{name} tracks: {tracks_per_class[name]}")
        print("Zone membership: unique IDs observed inside, including first observation; not violations.")
        for kind in ZONE_TYPES:
            print(f"Tracks entering {kind}: {result.zone_tracks[kind]}")
        print(f"Average processing FPS (excluding model load): {processing_fps:.2f}")
        print(f"Output video: {output_path}")
        print(f"Stationary tracks: {result.stationary_tracks}")
        print(f"Violation candidates: {result.candidates}")
        print(f"Accepted events: {result.accepted}")
        print(f"Duplicates suppressed: {result.suppressed}")
        print(f"Snapshots saved: {result.snapshots}")
        print(f"Database: {result.database_path}")
        return result
