import json

import numpy as np
import pytest

from src.core.models import Zone, TrackedObject
from src.zones.geometry import get_bottom_center, point_in_polygon
from src.zones.zone_manager import ZoneManager
from scripts.select_roi import save_polygon, select_polygon

POLYGON = [(0, 0), (100, 0), (100, 100), (0, 100)]


def track(bbox=(20, 20, 40, 40)):
    return TrackedObject(1, bbox, 0.9, "person", 1.2)


def test_bottom_center():
    assert get_bottom_center((1, 2, 8, 20)) == (4.5, 20)


@pytest.mark.parametrize("point,inside", [((50, 50), True), ((101, 50), False),
                                          ((0, 50), True), ((100, 100), True)])
def test_geometry_boundary(point, inside):
    assert point_in_polygon(point, POLYGON) is inside


def test_concave_polygon():
    polygon = [(0, 0), (100, 0), (100, 20), (20, 20), (20, 100), (0, 100)]
    assert point_in_polygon((10, 70), polygon)
    assert not point_in_polygon((50, 50), polygon)


def test_load_save_roundtrip(tmp_path):
    path = tmp_path / "zones.json"
    original = ZoneManager("CAM", [Zone("SW", "CAM", "SIDEWALK", POLYGON)])
    original.save_to_json(path)
    loaded = ZoneManager.load_from_json(path)
    assert loaded.zones == original.zones
    assert json.loads(path.read_text())["zones"][0]["type"] == "SIDEWALK"


@pytest.mark.parametrize("change", [
    {"type": "BAD"}, {"zone_id": ""}, {"polygon": [[0, 0], [1, 1]]},
    {"polygon": [[0, 0], [1, 1], [2, 2]]},
    {"polygon": [[0, 0], [0, 0], [1, 2]]},
    {"polygon": [[0, 0], [1.2, 2], [5, 1]]},
    {"polygon": [[0, 0], [True, 2], [5, 1]]},
])
def test_invalid_zone(tmp_path, change):
    zone = {"zone_id": "SW", "type": "SIDEWALK", "polygon": POLYGON}
    zone.update(change)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"camera_id": "CAM", "zones": [zone]}))
    with pytest.raises(ValueError):
        ZoneManager.load_from_json(path)


@pytest.mark.parametrize("data", [{"zones": []}, {"camera_id": "", "zones": []},
                                  {"camera_id": "CAM", "zones": {}}, []])
def test_invalid_schema(tmp_path, data):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        ZoneManager.load_from_json(path)


def test_missing_and_malformed_file(tmp_path):
    path = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="Zone file not found"):
        ZoneManager.load_from_json(path)
    path.write_text("{bad")
    with pytest.raises(ValueError, match="Invalid zone JSON"):
        ZoneManager.load_from_json(path)


def test_membership_bottom_center_not_bbox_overlap():
    zone = Zone("SW", "CAM", "SIDEWALK", POLYGON)
    manager = ZoneManager("CAM", [zone])
    assert manager.is_inside_zone(track(), zone)
    assert not manager.is_inside_zone(track((20, 80, 40, 120)), zone)
    assert manager.get_membership(track(), "OTHER").matched_zones == []
    assert manager.get_zones("OTHER") == []


def test_all_types_precedence_and_config_order():
    zones = [Zone(str(i), "CAM", kind, POLYGON) for i, kind in enumerate(
        ["MONITORED", "SIDEWALK", "ALLOWED", "IGNORE"])]
    for count, expected in [(4, "IGNORE"), (3, "ALLOWED"), (2, "MONITORED")]:
        result = ZoneManager("CAM", zones[:count]).get_membership(track(), "CAM")
        assert len(result.matched_zones) == count
        assert result.effective_zone.zone_type == expected
        assert result.by_type["MONITORED"]
        assert (result.target_zone is None) == (count > 2)


def test_duplicate_ids_and_camera_mismatch():
    zone = Zone("SW", "CAM", "SIDEWALK", POLYGON)
    with pytest.raises(ValueError, match="Duplicate"):
        ZoneManager("CAM", [zone, zone])
    with pytest.raises(ValueError, match="camera_id"):
        ZoneManager("OTHER", [zone])


def test_resolution_validation():
    manager = ZoneManager("CAM", [Zone("SW", "CAM", "SIDEWALK", POLYGON)])
    manager.validate_frame_size(101, 101)
    with pytest.raises(ValueError, match="outside video resolution"):
        manager.validate_frame_size(100, 100)


def test_selector_save_preserves_other_zones_and_order(tmp_path):
    path = tmp_path / "cam.json"
    save_polygon(path, "CAM", "SW", "SIDEWALK", POLYGON)
    save_polygon(path, "CAM", "AL", "ALLOWED", POLYGON)
    save_polygon(path, "CAM", "SW", "MONITORED", POLYGON)
    manager = ZoneManager.load_from_json(path)
    assert [(z.zone_id, z.zone_type) for z in manager.zones] == [("SW", "MONITORED"), ("AL", "ALLOWED")]
    before = path.read_bytes()
    with pytest.raises(ValueError):
        save_polygon(path, "OTHER", "SW", "SIDEWALK", POLYGON)
    with pytest.raises(ValueError):
        save_polygon(path, "CAM", "SW", "SIDEWALK", [])
    assert path.read_bytes() == before


@pytest.mark.parametrize("quit_only", [True, False])
def test_selector_mouse_reset_save_quit_without_gui(tmp_path, monkeypatch, quit_only):
    from scripts import select_roi
    cv2 = select_roi.cv2
    callback = {}
    monkeypatch.setattr(cv2, "namedWindow", lambda *a: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda w, f: callback.update(mouse=f))
    monkeypatch.setattr(cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *a: 1)
    closed = []
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: closed.append(True))
    keys = iter(["s", "r", "q" if quit_only else "s"])

    def wait(_):
        key = next(keys)
        if key == "r":
            callback["mouse"](cv2.EVENT_LBUTTONDOWN, 50, 50, 0, None)
        elif key == "s" and callback.get("attempted"):
            for x, y in [(10, 10), (90, 10), (90, 90)]:
                callback["mouse"](cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
        callback["attempted"] = True
        return ord(key)

    monkeypatch.setattr(cv2, "waitKey", wait)
    path = tmp_path / "cam.json"
    saved = select_polygon(np.zeros((100, 100, 3), dtype=np.uint8), path, "CAM", "SW", "SIDEWALK")
    assert saved is not quit_only
    assert path.exists() is not quit_only
    assert closed == [True]
    if saved:
        assert ZoneManager.load_from_json(path).zones[0].polygon == [(10, 10), (90, 10), (90, 90)]
