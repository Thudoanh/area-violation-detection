"""Shapely pixel geometry for manual ROI; boundary points count as inside."""

from shapely.geometry import Point, Polygon


def get_bottom_center(bbox):
    x1, _, x2, y2 = bbox
    return (x1 + x2) / 2, y2


def point_in_polygon(point, polygon) -> bool:
    if not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
        raise ValueError("Polygon must contain at least 3 xy points")
    shape = Polygon(polygon)
    if shape.is_empty or not shape.is_valid or shape.area == 0:
        raise ValueError("Polygon must be a valid nonzero-area shape")
    # covers(), unlike contains(), includes points on the polygon boundary.
    return shape.covers(Point(float(point[0]), float(point[1])))
