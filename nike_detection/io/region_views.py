"""In-memory region views of a decoded full TIFF (no extra decode)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from nike_detection.config.schema import BoundingBox, RegionSpec
from nike_detection.io.image_loader import Scan

logger = logging.getLogger(__name__)


@dataclass
class RegionView:
    name: str
    region_type: str
    bgr: np.ndarray
    gray: np.ndarray
    origin_xy: Tuple[int, int]
    bbox: Tuple[int, int, int, int]


def sanitize_xyxy(
    top_x: float, top_y: float, bottom_x: float, bottom_y: float
) -> Tuple[int, int, int, int]:
    x1 = int(min(abs(top_x), abs(bottom_x)))
    y1 = int(min(abs(top_y), abs(bottom_y)))
    x2 = int(max(abs(top_x), abs(bottom_x)))
    y2 = int(max(abs(top_y), abs(bottom_y)))
    return x1, y1, x2, y2


def clip_to_image(
    x1: int, y1: int, x2: int, y2: int, width: int, height: int, name: str
) -> Optional[Tuple[int, int, int, int]]:
    cx1 = max(0, x1)
    cy1 = max(0, y1)
    cx2 = min(width, x2)
    cy2 = min(height, y2)
    if cx2 <= cx1 or cy2 <= cy1:
        logger.warning("Region '%s' has no overlap with image %dx%d", name, width, height)
        return None
    if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
        logger.warning(
            "Region '%s' clipped from (%s,%s)-(%s,%s) to (%s,%s)-(%s,%s)",
            name, x1, y1, x2, y2, cx1, cy1, cx2, cy2,
        )
    return cx1, cy1, cx2, cy2


def slice_view(scan: Scan, spec: RegionSpec) -> Optional[RegionView]:
    height, width = scan.shape[:2]
    x1, y1, x2, y2 = spec.bounding_box_pixels.as_int_xyxy()
    clipped = clip_to_image(x1, y1, x2, y2, width, height, spec.name)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    # Materialized per region rather than for the sheet: the crops together
    # are a fraction of a full scan, and on a memory-mapped one this is the
    # only point where pixels are actually read.
    bgr = scan.crop_bgr(x1, y1, x2, y2)
    return RegionView(
        name=spec.name,
        region_type=spec.type,
        bgr=bgr,
        gray=cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
        origin_xy=(x1, y1),
        bbox=(x1, y1, x2, y2),
    )


def slice_views(scan: Scan, specs: List[RegionSpec]) -> List[RegionView]:
    views: List[RegionView] = []
    for spec in specs:
        view = slice_view(scan, spec)
        if view is not None:
            views.append(view)
    return views
