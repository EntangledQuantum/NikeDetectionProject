"""Shared new-pattern and legacy island geometry. Built once per region."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from nike_detection.geometry.island_line_extractor import IslandLineExtractor
from nike_detection.geometry.line_detector import LineDetector
from nike_detection.geometry.material_profile import estimate_background_level
from nike_detection.geometry.vertical_band_detector import (
    VerticalBandDetector,
    paint_vertical_line_regions,
)

logger = logging.getLogger(__name__)


class GeometryError(RuntimeError):
    """Raised when required island geometry cannot be produced."""


def detect_bands(
    image: np.ndarray,
    clear: bool = False,
    debug: bool = False,
    **kwargs: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], VerticalBandDetector]:
    detector = VerticalBandDetector(clear=clear, **kwargs)
    bands = detector.detect(image, debug)
    vlines = list(detector.last_vlines or [])
    if not bands:
        raise GeometryError("VerticalBandDetector returned no bands")
    return bands, vlines, detector


def extract_band_lines(
    gray: np.ndarray,
    bands: List[Dict[str, Any]],
    clear: bool = False,
    debug: bool = False,
    extractor: Optional[IslandLineExtractor] = None,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Run IslandLineExtractor once per band. Fails loudly on empty usable bands."""
    extractor = extractor or IslandLineExtractor(clear=clear)
    extractions: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    skipped: List[int] = []
    for band in bands:
        x0, x1 = band["x0"], band["x1"]
        if x1 - x0 < 20:
            skipped.append(band.get("index", -1))
            continue
        result = extractor.extract(gray[:, x0:x1 + 1], debug)
        if result is None:
            raise GeometryError(
                f"IslandLineExtractor returned None for band {band.get('index')} "
                f"x=[{x0},{x1}] (width={x1 - x0})"
            )
        extractions.append((band, result))
    if not extractions:
        raise GeometryError(
            f"No usable bands for line extraction (skipped={skipped})"
        )
    return extractions


def paint_structure(
    gray: np.ndarray,
    extractions: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    vlines: List[Dict[str, Any]],
    thickness: int,
    fill: int,
    extension: int = 20,
) -> np.ndarray:
    """Paint horizontal trajectories and vertical envelopes on a gray copy."""
    painted = gray.copy()
    for band, result in extractions:
        x0, x1 = band["x0"], band["x1"]
        xs = np.arange(-extension, x1 - x0 + 1 + extension, 8)
        for line in result["lines"]:
            ys = IslandLineExtractor.line_y(result, line, xs)
            pts = np.stack([xs + x0, np.round(ys)], axis=1)
            pts = pts.reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(painted, [pts], False, int(fill), int(thickness))
    if vlines:
        painted = paint_vertical_line_regions(painted, vlines)
    return painted


def detect_legacy_lines(
    image: np.ndarray,
    sensitivity: str,
    exclusion_zones: Optional[List[Dict[str, Any]]] = None,
    debug: bool = False,
    image_path: Optional[str] = None,
    ideal: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], np.ndarray, LineDetector]:
    detector = LineDetector(sensitivity)
    if ideal is not None:
        detector.IDEAL_IMAGE_WIDTH = int(ideal.width)
        detector.IDEAL_IMAGE_HEIGHT = int(ideal.height)
        detector.Y_DELTA_MIN = int(ideal.y_delta_min)
        detector.Y_DELTA_MAX = int(ideal.y_delta_max)
        detector.SLOPE_MIN = float(ideal.slope_min)
        detector.SLOPE_MAX = float(ideal.slope_max)
    if exclusion_zones:
        detector.exclusion_zones = exclusion_zones
    elif image_path:
        detector.load_exclusion_zones(image_path)
    matched, *_rest = detector.detect_lines(image, debug)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    return matched, binary, detector
