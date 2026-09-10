"""Draw a manual polygon on the first video frame and save camera zones."""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import DEFAULT_CONFIG_PATH, load_config
from src.core.frame_provider import VideoFrameProvider
from src.core.models import Zone
from src.core.visualization import draw_zones
from src.zones.models import ZONE_TYPES
from src.zones.zone_manager import ZoneManager, validate_zone


def save_polygon(output, camera_id, zone_id, zone_type, points):
    """Append a zone or replace the same ID in place, preserving other zones."""
    zone = Zone(zone_id, camera_id, zone_type, list(points))
    validate_zone(zone)
    output = Path(output)
    manager = (ZoneManager.load_from_json(output) if output.exists()
               else ZoneManager(camera_id, []))
    if manager.camera_id != camera_id:
        raise ValueError("Output JSON belongs to a different camera_id")
    for index, old in enumerate(manager.zones):
        if old.zone_id == zone_id:
            manager.zones[index] = zone
            break
    else:
        manager.zones.append(zone)
    manager.save_to_json(output)


def select_polygon(frame, output, camera_id, zone_id, zone_type):
    output = Path(output)
    existing = ZoneManager.load_from_json(output) if output.exists() else ZoneManager(camera_id, [])
    if existing.camera_id != camera_id:
        raise ValueError("Output JSON belongs to a different camera_id")
    existing.validate_frame_size(frame.shape[1], frame.shape[0])
    points = []
    window = "Select ROI"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            points.append((x, y))

    print("Left click: add point | r: reset | s: save and exit | q: quit without saving")
    print(f"Saving replaces zone {zone_id} if it exists; other zones keep their order.")
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    try:
        cv2.setMouseCallback(window, on_mouse)
        while True:
            preview = draw_zones(frame, existing.zones)
            if len(points) > 1:
                cv2.polylines(preview, [np.asarray(points, dtype=np.int32)], len(points) >= 3,
                              (0, 255, 255), 2)
            for point in points:
                cv2.circle(preview, point, 4, (0, 255, 255), -1)
            cv2.imshow(window, preview)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q") or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return False
            if key == ord("r"):
                points.clear()
            if key == ord("s"):
                try:
                    save_polygon(output, camera_id, zone_id, zone_type, points)
                except ValueError as exc:
                    print(f"Cannot save ROI: {exc}")
                    continue
                print(f"Saved {zone_id} to {output}")
                return True
    finally:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zone-type", choices=ZONE_TYPES, default="SIDEWALK")
    parser.add_argument("--zone-id", default="SW_01")
    parser.add_argument("--camera-id", help="Defaults to camera.camera_id in config")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    try:
        if args.output.resolve() == Path(args.video).resolve():
            raise ValueError("Output JSON must differ from input video")
        if args.output.suffix.lower() != ".json":
            raise ValueError("Output must have a .json extension")
        config = load_config(args.config)
        camera = config.get("camera", {})
        camera_id = args.camera_id or (camera.get("camera_id") if isinstance(camera, dict) else None)
        ZoneManager(camera_id, [])
        with VideoFrameProvider(args.video) as provider:
            try:
                _, _, frame = next(provider)
            except StopIteration as exc:
                raise ValueError("Video contains no decodable first frame") from exc
        select_polygon(frame, args.output, camera_id, args.zone_id, args.zone_type)
    except (OSError, ValueError, cv2.error) as exc:
        parser.exit(1, f"Error: {exc}\n")

if __name__ == "__main__":
    main()
