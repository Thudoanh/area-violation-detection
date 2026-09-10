"""Run YOLO11n + ByteTrack and save an annotated local video without a GUI."""

import argparse
from pathlib import Path
import sys
import sqlite3

import cv2

# Support `python scripts/run_video.py` without installing a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import DEFAULT_CONFIG_PATH, load_config
from src.core.frame_provider import VideoFrameProvider
from src.detection.yolo11_detector import PROJECT_ROOT, YOLO11Detector
from src.tracking.bytetrack_tracker import ByteTrackTracker
from src.pipeline.video_runner import process_video as run_video_service


def process_video(video_path, config: dict, output_path: Path, **kwargs):
    """CLI-compatible adapter around the reusable pipeline runner."""
    return run_video_service(
        video_path,
        config,
        output_path,
        frame_provider_factory=VideoFrameProvider,
        detector_factory=YOLO11Detector,
        tracker_factory=ByteTrackTracker,
        video_writer_factory=cv2.VideoWriter,
        **kwargs,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Local video file")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, help="New annotated .mp4 path")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        output = args.output or PROJECT_ROOT / "data" / "events" / f"annotated_{Path(args.video).stem}.mp4"
        process_video(args.video, config, output)
    except (OSError, ValueError, RuntimeError, sqlite3.Error, cv2.error) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
