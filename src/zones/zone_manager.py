"""Load/save camera polygons and evaluate bottom-center membership."""

import json
from pathlib import Path
import tempfile

from shapely.geometry import Point, Polygon

from src.core.models import TrackedObject, Zone
from src.zones.geometry import get_bottom_center
from src.zones.models import ZONE_TYPES, ZoneMembership


def validate_zone(zone: Zone) -> None:
    for name in ("camera_id", "zone_id"):
        value = getattr(zone, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Zone {name} must be a nonempty string")
    if zone.zone_type not in ZONE_TYPES:
        raise ValueError(f"Invalid zone type: {zone.zone_type}")
    points = zone.polygon
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError("Polygon must contain at least 3 points")
    for point in points:
        if (not isinstance(point, (list, tuple)) or len(point) != 2 or
                any(type(v) is not int or not 0 <= v < 2**24 for v in point)):
            raise ValueError("Polygon coordinates must be nonnegative pixel integers below 2**24")
    if len(set(tuple(p) for p in points)) < 3:
        raise ValueError("Polygon requires at least 3 distinct points")
    shape = Polygon(points)
    if shape.is_empty or not shape.is_valid or shape.area == 0:
        raise ValueError("Polygon must be a valid nonzero-area shape")


class ZoneManager:
    def __init__(self, camera_id: str, zones: list[Zone]):
        if not isinstance(camera_id, str) or not camera_id.strip():
            raise ValueError("camera_id must be a nonempty string")
        seen = set()
        for zone in zones:
            validate_zone(zone)
            if zone.camera_id != camera_id:
                raise ValueError("Zone camera_id does not match the file camera_id")
            if zone.zone_id in seen:
                raise ValueError(f"Duplicate zone_id: {zone.zone_id}")
            seen.add(zone.zone_id)
        self.camera_id = camera_id
        self.zones = list(zones)
        self._polygons = {zone.zone_id: Polygon(zone.polygon) for zone in self.zones}

    @classmethod
    def load_from_json(cls, path):
        path = Path(path)
        try:
            with path.open(encoding="utf-8") as stream:
                data = json.load(stream)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Zone file not found: {path}; create ROI with select_roi.py") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid zone JSON {path}: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("zones"), list):
            raise ValueError("Zone JSON must contain camera_id and a zones list")
        zones = []
        for item in data["zones"]:
            if not isinstance(item, dict):
                raise ValueError("Each zone must be a JSON object")
            zone = Zone(item.get("zone_id"), data.get("camera_id"),
                        item.get("type"), item.get("polygon"))
            validate_zone(zone)
            zone.polygon = [tuple(p) for p in zone.polygon]
            zones.append(zone)
        return cls(data.get("camera_id"), zones)

    def save_to_json(self, path) -> None:
        # Validate the complete list before replacing any existing file.
        ZoneManager(self.camera_id, self.zones)
        data = {"camera_id": self.camera_id, "zones": [
            {"zone_id": z.zone_id, "type": z.zone_type, "polygon": z.polygon}
            for z in self.zones
        ]}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(data, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
            temporary.replace(path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def get_zones(self, camera_id: str) -> list[Zone]:
        return [z for z in self.zones if z.camera_id == camera_id]

    def validate_frame_size(self, width: int, height: int) -> None:
        for zone in self.zones:
            if any(x >= width or y >= height for x, y in zone.polygon):
                raise ValueError(f"Zone {zone.zone_id} lies outside video resolution {width}x{height}")

    def is_inside_zone(self, track: TrackedObject, zone: Zone) -> bool:
        x, y = get_bottom_center(track.bbox)
        return self._polygons[zone.zone_id].covers(Point(float(x), float(y)))

    def get_membership(self, track: TrackedObject, camera_id: str) -> ZoneMembership:
        matched = [z for z in self.get_zones(camera_id) if self.is_inside_zone(track, z)]
        # min preserves config order for equal priorities.
        priority = {"IGNORE": 0, "ALLOWED": 1, "SIDEWALK": 2, "MONITORED": 2}
        effective = min(matched, key=lambda z: priority[z.zone_type], default=None)
        return ZoneMembership(matched, effective)
