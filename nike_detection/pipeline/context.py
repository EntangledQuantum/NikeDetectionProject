"""Per-region ImageContext with lazy shared layers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from nike_detection.config.schema import RunSettings
from nike_detection.geometry.material_profile import estimate_background_level
from nike_detection.io.image_loader import apply_clahe
from nike_detection.pipeline.preprocess_island import (
    GeometryError,
    detect_bands,
    detect_legacy_lines,
    extract_band_lines,
    paint_structure,
)
from nike_detection.pipeline.preprocess_stripe import (
    StripeBounds,
    StripeEdges,
    extract_stripe_edges,
    find_stripe_bounds,
)
from nike_detection.pipeline.types import ImageType

logger = logging.getLogger(__name__)

STRIPE_EDGE_LAYERS = frozenset({"stripe_edges"})
STRIPE_LAB_LAYERS = frozenset({"lab", "stripe_bounds"})
STRIPE_CLAHE_LAYERS = frozenset({"clahe"})
ISLAND_NEW_BANDS = frozenset({"bands", "vlines"})
ISLAND_NEW_EXTRACT = frozenset({"extractor_per_band"})
ISLAND_NEW_PAINT = frozenset({"lines_removed_gray"})
ISLAND_LEGACY_LINES = frozenset({"matched_lines", "binary_127"})


def load_exclusion_zones(image_path: Optional[str]) -> List[Dict[str, Any]]:
    if not image_path:
        return []
    json_path = Path(image_path).with_suffix(".json")
    if not json_path.exists():
        return []
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        logger.warning("Could not load exclusion zones from %s: %s", json_path, exc)
        return []
    zones = []
    for zone in data.get("exclusion_zones", []):
        bbox = zone.get("bounding_box_pixels", {})
        x1 = int(min(abs(float(bbox.get("top_x", 0))), abs(float(bbox.get("bottom_x", 0)))))
        y1 = int(min(abs(float(bbox.get("top_y", 0))), abs(float(bbox.get("bottom_y", 0)))))
        x2 = int(max(abs(float(bbox.get("top_x", 0))), abs(float(bbox.get("bottom_x", 0)))))
        y2 = int(max(abs(float(bbox.get("top_y", 0))), abs(float(bbox.get("bottom_y", 0)))))
        zones.append({
            "top_x": x1, "top_y": y1, "bottom_x": x2, "bottom_y": y2,
            "name": zone.get("name", "unnamed"),
        })
    return zones


def point_in_exclusion(zones: List[Dict[str, Any]], x: int, y: int) -> bool:
    for zone in zones:
        x1, x2 = min(zone["top_x"], zone["bottom_x"]), max(zone["top_x"], zone["bottom_x"])
        y1, y2 = min(zone["top_y"], zone["bottom_y"]), max(zone["top_y"], zone["bottom_y"])
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False


def region_overlaps_exclusion(
    zones: List[Dict[str, Any]], x: int, y: int, width: int, height: int
) -> bool:
    for zone in zones:
        x1, x2 = min(zone["top_x"], zone["bottom_x"]), max(zone["top_x"], zone["bottom_x"])
        y1, y2 = min(zone["top_y"], zone["bottom_y"]), max(zone["top_y"], zone["bottom_y"])
        if x < x2 and x + width > x1 and y < y2 and y + height > y1:
            return True
    return False


class ImageContext:
    """Decoded image plus lazily built shared geometry."""

    def __init__(
        self,
        bgr: np.ndarray,
        gray: np.ndarray,
        path: str,
        name: str,
        image_type: ImageType,
        settings: RunSettings,
        origin_xy: Tuple[int, int] = (0, 0),
        parent_image: Optional[str] = None,
        exclusion_zones: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.bgr = bgr
        self.gray = gray
        self.path = path
        self.name = name
        self.image_type = image_type
        self.settings = settings
        self.origin_xy = origin_xy
        self.parent_image = parent_image
        self.exclusion_zones = (
            exclusion_zones if exclusion_zones is not None
            else load_exclusion_zones(path)
        )
        self.debug: Dict[str, Any] = {}
        self._cache: Dict[str, Any] = {}

    @property
    def height(self) -> int:
        return int(self.gray.shape[0])

    @property
    def width(self) -> int:
        return int(self.gray.shape[1])

    def background_level(self) -> float:
        if "background_level" not in self._cache:
            self._cache["background_level"] = estimate_background_level(self.gray)
        return self._cache["background_level"]

    def lab(self) -> np.ndarray:
        if "lab" not in self._cache:
            self._cache["lab"] = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2LAB)
        return self._cache["lab"]

    def hsv(self) -> np.ndarray:
        if "hsv" not in self._cache:
            self._cache["hsv"] = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2HSV)
        return self._cache["hsv"]

    def clahe(self, clip_limit: float = 2.0) -> np.ndarray:
        key = f"clahe_{clip_limit}"
        if key not in self._cache:
            self._cache[key] = apply_clahe(self.gray, clip_limit=clip_limit)
        return self._cache[key]

    def stripe_bounds(self) -> Optional[StripeBounds]:
        if "stripe_bounds" not in self._cache:
            self._cache["stripe_bounds"] = find_stripe_bounds(self.lab())
        return self._cache["stripe_bounds"]

    def stripe_edges(self) -> Optional[StripeEdges]:
        if "stripe_edges" not in self._cache:
            median_window = int(
                self.settings.detector_params("stripe_misalignment").get("median_window", 31)
            )
            self._cache["stripe_edges"] = extract_stripe_edges(self.gray, median_window)
        return self._cache["stripe_edges"]

    def bands(self) -> List[Dict[str, Any]]:
        self._ensure_bands()
        return self._cache["bands"]

    def vlines(self) -> List[Dict[str, Any]]:
        self._ensure_bands()
        return self._cache["vlines"]

    def extractor_per_band(self) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        if "extractor_per_band" not in self._cache:
            self._cache["extractor_per_band"] = extract_band_lines(
                self.gray,
                self.bands(),
                clear=self.settings.clear,
                debug=self.settings.debug,
            )
        return self._cache["extractor_per_band"]

    def lines_removed_gray(self) -> np.ndarray:
        if "lines_removed_gray" not in self._cache:
            overspray_params = self.settings.detector_params("overspray_island")
            thickness = int(overspray_params.get("line_thickness", 15)) * 3
            fill = (
                int(round(self.background_level()))
                if self.settings.clear else 255
            )
            painted = paint_structure(
                self.gray,
                self.extractor_per_band(),
                self.vlines(),
                thickness=thickness,
                fill=fill,
                extension=self.settings.config.material.line_paint_extension,
            )
            if self.settings.clear:
                painted = cv2.medianBlur(painted, 5)
            self._cache["lines_removed_gray"] = painted
        return self._cache["lines_removed_gray"]

    def matched_lines(self) -> List[Dict[str, Any]]:
        self._ensure_legacy_lines()
        return self._cache["matched_lines"]

    def binary_127(self) -> np.ndarray:
        self._ensure_legacy_lines()
        return self._cache["binary_127"]

    def _ensure_bands(self) -> None:
        if "bands" in self._cache:
            return
        vb = self.settings.config.vertical_band
        bands, vlines, detector = detect_bands(
            self.bgr,
            clear=self.settings.clear,
            debug=self.settings.debug,
            binary_threshold=vb.binary_threshold,
            content_binary_threshold=vb.content_binary_threshold,
            min_coverage=vb.min_coverage,
            inner_margin=vb.inner_margin,
        )
        self._cache["bands"] = bands
        self._cache["vlines"] = vlines
        self._cache["band_detector"] = detector
        self.debug["band_detector"] = detector

    def _ensure_legacy_lines(self) -> None:
        if "matched_lines" in self._cache:
            return
        matched, binary, detector = detect_legacy_lines(
            self.bgr,
            sensitivity=self.settings.sensitivity,
            exclusion_zones=self.exclusion_zones,
            debug=self.settings.debug,
            image_path=self.path,
            ideal=self.settings.config.ideal_reference,
        )
        self._cache["matched_lines"] = matched
        self._cache["binary_127"] = binary
        self._cache["line_detector"] = detector
        self.debug["line_detector"] = detector

    def ensure_layers(self, layers: Set[str]) -> None:
        """Build only the geometry requested by the selected detectors."""
        if layers & STRIPE_LAB_LAYERS:
            self.lab()
            self.stripe_bounds()
        if layers & STRIPE_EDGE_LAYERS:
            self.stripe_edges()
        if layers & STRIPE_CLAHE_LAYERS:
            self.clahe()
        if self.settings.clear:
            self.background_level()
        if layers & ISLAND_NEW_BANDS or layers & ISLAND_NEW_EXTRACT or layers & ISLAND_NEW_PAINT:
            self.bands()
        if layers & ISLAND_NEW_EXTRACT or layers & ISLAND_NEW_PAINT:
            self.extractor_per_band()
        if layers & ISLAND_NEW_PAINT:
            self.lines_removed_gray()
        if layers & ISLAND_LEGACY_LINES:
            self.matched_lines()
