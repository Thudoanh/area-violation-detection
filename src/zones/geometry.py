"""Pixel geometry for manual ROI; boundary points count as inside."""

import cv2
import numpy as np


def get_bottom_center(bbox):
    x1, _, x2, y2 = bbox
    return (x1 + x2) / 2, y2


def point_in_polygon(point, polygon) -> bool:
    contour = np.asarray(polygon, dtype=np.float32)
    if contour.ndim != 2 or contour.shape[1] != 2 or len(contour) < 3:
        raise ValueError("Polygon must contain at least 3 xy points")
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0
