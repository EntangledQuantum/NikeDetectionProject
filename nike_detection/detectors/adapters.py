"""Detector adapters: config params in, ImageContext in, Defects out, no I/O."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from nike_detection.config.schema import RunSettings
from nike_detection.detectors.island_legacy.debris import DebrisIslandDetector
from nike_detection.detectors.island_legacy.line_defect import LineDefectDetector
from nike_detection.detectors.island_legacy.overspray import OversprayIslandDetector
from nike_detection.detectors.island_new.debris import NewPatternDebrisIslandDetector
from nike_detection.detectors.island_new.line_defect import NewPatternLineDefectDetector
from nike_detection.detectors.island_new.overspray import NewPatternOversprayIslandDetector
from nike_detection.detectors.stripe.debris import DebrisStripeDetector
from nike_detection.detectors.stripe.misalignment import StripeMisalignmentDetector
from nike_detection.detectors.stripe.overspray import OversprayDetector
from nike_detection.detectors.stripe.roughness import StripeEdgeRoughnessDetector
from nike_detection.detectors.stripe.surface_treatment import SurfaceTreatmentDetector
from nike_detection.detectors.stripe.void import VoidDetector
from nike_detection.pipeline.context import (
    ImageContext,
    point_in_exclusion,
    region_overlaps_exclusion,
)
from nike_detection.pipeline.types import Defect, defects_from_legacy

logger = logging.getLogger(__name__)


def _apply(obj: Any, params: Dict[str, Any], aliases: Optional[Dict[str, str]] = None) -> None:
    aliases = aliases or {}
    for key, value in params.items():
        attr = aliases.get(key, key)
        setattr(obj, attr, value)


def _filter_exclusions(ctx: ImageContext, defects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not ctx.exclusion_zones:
        return defects
    kept = []
    for item in defects:
        bbox = item.get("bbox")
        if bbox is not None and len(bbox) == 4:
            x, y, w, h = [int(v) for v in bbox]
            # Support both xywh and xyxy.
            if w > 0 and h > 0 and (x + w) <= ctx.width * 2:
                if region_overlaps_exclusion(ctx.exclusion_zones, x, y, abs(w), abs(h)):
                    continue
        loc = item.get("location")
        if loc is not None and len(loc) >= 2:
            if point_in_exclusion(ctx.exclusion_zones, int(loc[0]), int(loc[1])):
                continue
        kept.append(item)
    return kept


class _BaseAdapter:
    key: str = ""
    layers: frozenset[str] = frozenset()

    def __init__(self, settings: RunSettings) -> None:
        self.settings = settings
        self.debug = settings.debug
        self._vis = None
        self._impl = None

    def required_layers(self) -> frozenset[str]:
        return self.layers

    def render(self, ctx: ImageContext, defects: List[Defect]):
        if self._vis is not None:
            return self._vis
        return ctx.bgr.copy()

    def _pack(self, ctx: ImageContext, raw: List[Dict[str, Any]]) -> List[Defect]:
        raw = _filter_exclusions(ctx, raw)
        return defects_from_legacy(self.key, raw, ctx.origin_xy)


class VoidAdapter(_BaseAdapter):
    key = "void"
    layers = frozenset({"lab", "stripe_bounds"})

    def __init__(self, settings: RunSettings) -> None:
        super().__init__(settings)
        params = settings.detector_params("void")
        self._impl = VoidDetector(
            sensitivity=settings.sensitivity,
            debug=settings.debug,
            max_area_frac=params.get("max_area_frac", 0.10),
            max_dim_ratio_to_stripe_w=params.get("max_dim_ratio_to_stripe_w", 1.0),
            inner_pad_frac=params.get("inner_pad_frac", 0.05),
            inner_pad_min=int(params.get("inner_pad_min", 20)),
        )
        _apply(self._impl, params, {
            "mad_k": "_mad_k",
            "min_area_floor": "_min_area_floor",
            "min_area_frac": "_min_area_frac",
            "score_floor": "_score_floor",
            "weak_fraction": "_weak_fraction",
        })

    def detect(self, ctx: ImageContext) -> List[Defect]:
        vis, raw = self._impl.detect(ctx.bgr, ctx.path)
        self._vis = vis
        return self._pack(ctx, raw)


class DebrisStripeAdapter(_BaseAdapter):
    key = "debris_stripe"
    layers = frozenset({"lab", "stripe_bounds"})

    def __init__(self, settings: RunSettings) -> None:
        super().__init__(settings)
        params = settings.detector_params("debris_stripe")
        self._impl = DebrisStripeDetector(
            sensitivity=settings.sensitivity,
            debug=settings.debug,
            inner_pad_frac=params.get("inner_pad_frac", 0.035),
            inner_pad_min=int(params.get("inner_pad_min", 8)),
            max_area_frac=params.get("max_area_frac", 0.20),
        )
        _apply(self._impl, params, {
            "score_k": "_score_k",
            "strong_floor": "_strong_floor",
            "weak_floor": "_weak_floor",
            "min_area_floor": "_min_area_floor",
            "min_area_frac": "_min_area_frac",
        })

    def detect(self, ctx: ImageContext) -> List[Defect]:
        vis, raw = self._impl.detect(ctx.bgr, ctx.path)
        self._vis = vis
        return self._pack(ctx, raw)


class MisalignmentAdapter(_BaseAdapter):
    key = "stripe_misalignment"
    layers = frozenset({"stripe_edges"})

    def __init__(self, settings: RunSettings) -> None:
        super().__init__(settings)
        params = settings.detector_params("stripe_misalignment")
        self._impl = StripeMisalignmentDetector(
            sensitivity=settings.sensitivity, debug=settings.debug
        )
        _apply(self._impl, params)

    def detect(self, ctx: ImageContext) -> List[Defect]:
        edges = ctx.stripe_edges()
        impl = self._impl
        if edges is None:
            self._vis = ctx.bgr.copy()
            return []
        left, right = edges.median_left, edges.median_right
        impl._last_profiles = (left, right)
        stitches = impl._detect_stitches(left, right)
        rolls = impl._detect_roll(left, right, [s["y"] for s in stitches], len(left))
        raw: List[Dict[str, Any]] = []
        for s in stitches:
            y = s["y"]
            x = int(np.nanmedian(left[max(0, y - 200):y + 200]))
            raw.append({
                "type": "stripe_misalignment",
                "kind": "stitch",
                "y": int(y),
                "x": x,
                "x_delta": abs(round(s["step"], 1)),
                "step_px": round(s["step"], 1),
                "edges": s["edges"],
                "location": (x, int(y)),
                "threshold": impl.step_threshold,
            })
        for r in rolls:
            yc = (r["y0"] + r["y1"]) // 2
            x = int(np.nanmedian(left[max(0, yc - 200):yc + 200]))
            raw.append({
                "type": "roll_error",
                "y0": r["y0"], "y1": r["y1"],
                "drift_px": round(r["drift_px"], 1),
                "slope_px_per_1k_rows": round(r["slope_px_per_1k_rows"], 2),
                "location": (x, yc),
                "threshold": impl.roll_threshold,
            })
        self._vis = impl.create_visualization(ctx.bgr, raw, left, right)
        return self._pack(ctx, raw)


class RoughnessAdapter(_BaseAdapter):
    key = "edge_roughness"
    layers = frozenset({"stripe_edges"})

    def __init__(self, settings: RunSettings) -> None:
        super().__init__(settings)
        params = settings.detector_params("edge_roughness")
        self._impl = StripeEdgeRoughnessDetector(
            sensitivity=settings.sensitivity, debug=settings.debug
        )
        _apply(self._impl, params)

    def detect(self, ctx: ImageContext) -> List[Defect]:
        edges = ctx.stripe_edges()
        if edges is not None:
            self._impl._precomputed_edges = (
                edges.subpixel_left, edges.subpixel_right, edges.mid,
                edges.gx_left, edges.gx_right,
            )
        vis, raw = self._impl.detect(ctx.bgr, ctx.path)
        self._vis = vis
        return self._pack(ctx, raw)


class OversprayAdapter(_BaseAdapter):
    key = "overspray"
    layers = frozenset({"clahe"})

    def __init__(self, settings: RunSettings) -> None:
        super().__init__(settings)
        params = settings.detector_params("overspray")
        self._impl = OversprayDetector(
            sensitivity=settings.sensitivity, debug=settings.debug
        )
        _apply(self._impl, params)

    def detect(self, ctx: ImageContext) -> List[Defect]:
        vis, raw = self._impl.detect(ctx.bgr)
        self._vis = vis
        return self._pack(ctx, raw)


class SurfaceTreatmentAdapter(_BaseAdapter):
    key = "surface_treatment"
    layers = frozenset({"clahe"})

    def __init__(self, settings: RunSettings) -> None:
        super().__init__(settings)
        params = settings.detector_params("surface_treatment")
        self._impl = SurfaceTreatmentDetector(
            contrast_threshold=params.get("contrast_threshold", 50),
            void_size_threshold=params.get("void_size_threshold", 150),
            coalescence_threshold=params.get("coalescence_threshold", 300),
            kernel_size=params.get("kernel_size", 10),
        )

    def detect(self, ctx: ImageContext) -> List[Defect]:
        vis, raw = self._impl.detect(ctx.clahe(2.0) if False else ctx.bgr)
        self._vis = vis
        return self._pack(ctx, raw)


class LegacyDebrisAdapter(_BaseAdapter):
    key = "debris_island"
    layers = frozenset({"matched_lines", "binary_127"})

    def __init__(self, settings: RunSettings) -> None:
        super().__init__(settings)
        params = settings.detector_params("debris_island")
        self._impl = DebrisIslandDetector(
            sensitivity=settings.sensitivity, debug=settings.debug
        )
        _apply(self._impl, params)
        if hasattr(self._impl, "line_detector"):
            _apply(self._impl.line_detector, settings.detector_params("line_detector"))

    def detect(self, ctx: ImageContext) -> List[Defect]:
        self._impl._precomputed_lines = ctx.matched_lines()
        self._impl.line_detector.exclusion_zones = ctx.exclusion_zones
        vis, raw = self._impl.detect(ctx.bgr, ctx.path)
        self._vis = vis
        return self._pack(ctx, raw)


class LegacyOversprayIslandAdapter(_BaseAdapter):
    key = "overspray_island"
    layers = frozenset({"matched_lines", "binary_127"})

    def __init__(self, settings: RunSettings) -> None:
        super().__init__(settings)
        params = settings.detector_params("overspray_island")
        self._impl = OversprayIslandDetector(
            sensitivity=settings.sensitivity, debug=settings.debug
        )
        _apply(self._impl, params)
        if hasattr(self._impl, "line_detector"):
            _apply(self._impl.line_detector, settings.detector_params("line_detector"))

    def detect(self, ctx: ImageContext) -> List[Defect]:
        self._impl._precomputed_lines = ctx.matched_lines()
        self._impl.line_detector.exclusion_zones = ctx.exclusion_zones
        vis, raw = self._impl.detect(ctx.bgr, ctx.path)
        self._vis = vis
        return self._pack(ctx, raw)


class LegacyLineDefectAdapter(_BaseAdapter):
    key = "line_defect"
    layers = frozenset({"matched_lines", "binary_127"})

    def __init__(self, settings: RunSettings) -> None:
        super().__init__(settings)
        params = settings.detector_params("line_defect_legacy")
        self._impl = LineDefectDetector(
            sensitivity=settings.sensitivity, debug=settings.debug
        )
        self._impl.debug = settings.debug
        _apply(self._impl, params)

    def detect(self, ctx: ImageContext) -> List[Defect]:
        self._impl._precomputed_lines = ctx.matched_lines()
        self._impl.line_detector.exclusion_zones = ctx.exclusion_zones
        vis, raw = self._impl.detect(ctx.bgr, ctx.path)
        self._vis = vis
        return self._pack(ctx, raw)


class NewDebrisAdapter(_BaseAdapter):
    key = "debris_island"
    layers = frozenset({"bands", "vlines", "extractor_per_band", "lines_removed_gray"})

    def __init__(self, settings: RunSettings) -> None:
        super().__init__(settings)
        params = settings.detector_params("debris_island")
        self._impl = NewPatternDebrisIslandDetector(
            sensitivity=settings.sensitivity, debug=settings.debug, clear=settings.clear
        )
        _apply(self._impl.base, params)

    def detect(self, ctx: ImageContext) -> List[Defect]:
        painted = ctx.lines_removed_gray()
        if ctx.settings.clear:
            offset = ctx.settings.config.material.clear_debris_offsets.get(
                ctx.settings.sensitivity, 45
            )
            self._impl.base.background_threshold = int(
                np.clip(ctx.background_level() - offset, 5, 250)
            )
            painted = cv2.medianBlur(painted, 3)
        contours, _ = self._impl.base.detect_debris(painted)
        total_lines = sum(len(result["lines"]) for _, result in ctx.extractor_per_band())
        raw = self._impl._build_defects(ctx.bands(), total_lines, contours)
        self._vis = self._impl.base.create_debris_visualization(ctx.bgr, contours, [])
        self._impl._debug_lines_removed_image = painted
        return self._pack(ctx, raw)


class NewOversprayIslandAdapter(_BaseAdapter):
    key = "overspray_island"
    layers = frozenset({"bands", "vlines", "extractor_per_band", "lines_removed_gray"})

    def __init__(self, settings: RunSettings) -> None:
        super().__init__(settings)
        params = settings.detector_params("overspray_island")
        self._impl = NewPatternOversprayIslandDetector(
            sensitivity=settings.sensitivity, debug=settings.debug, clear=settings.clear
        )
        _apply(self._impl.base, params)

    def detect(self, ctx: ImageContext) -> List[Defect]:
        painted = ctx.lines_removed_gray()
        if ctx.settings.clear:
            offset = ctx.settings.config.material.clear_overspray_offsets.get(
                ctx.settings.sensitivity, 45
            )
            color_threshold = int(np.clip(ctx.background_level() - offset, 5, 250))
            self._impl.base.background_threshold = 180 - color_threshold
        regions, _ = self._impl.base.detect_colored_regions(painted)
        regions = self._impl.base.group_nearby_regions(regions)
        total_lines = sum(len(result["lines"]) for _, result in ctx.extractor_per_band())
        raw = self._impl._build_defects(ctx.bands(), total_lines, regions)
        self._vis = self._impl.base.create_overspray_visualization(ctx.bgr, regions, [])
        self._impl._debug_lines_removed_image = painted
        return self._pack(ctx, raw)


class NewLineDefectAdapter(_BaseAdapter):
    key = "line_defect"
    layers = frozenset({"bands", "vlines", "extractor_per_band"})

    def __init__(self, settings: RunSettings) -> None:
        super().__init__(settings)
        params = settings.detector_params("line_defect_new")
        self._impl = NewPatternLineDefectDetector(
            sensitivity=settings.sensitivity, debug=settings.debug, clear=settings.clear
        )
        _apply(self._impl, params)

    def detect(self, ctx: ImageContext) -> List[Defect]:
        impl = self._impl
        extractions = ctx.extractor_per_band()
        all_missing: List[Dict[str, Any]] = []
        all_misaligned: List[Dict[str, Any]] = []
        band_summaries: List[Dict[str, Any]] = []
        for band, result in extractions:
            missing, misaligned = impl._evaluate_band(band, result)
            all_missing.extend(missing)
            all_misaligned.extend(misaligned)
            band_summaries.append({
                "index": band["index"],
                "x0": band["x0"], "x1": band["x1"],
                "vline_xs": band["vline_xs"],
                "line_count": len(result["lines"]),
                "inserted_line_count": sum(1 for line in result["lines"] if line["inserted"]),
                "slope": result["slope"],
                "spacing": result["spacing"],
                "missing_defects": len(missing),
                "missing_pixels": int(sum(d["missing_pixels"] for d in missing)),
                "misaligned_defects": len(misaligned),
            })
        spacing = float(np.median([b["spacing"] for b in band_summaries])) if band_summaries else 96.0
        density_regions = impl._find_density_regions(ctx.gray.shape, all_missing, spacing)
        self._vis = impl._create_visualization(
            ctx.bgr, all_missing, all_misaligned, density_regions, spacing
        )
        band_detector = ctx.debug.get("band_detector")
        if band_detector is not None:
            impl.band_detector = band_detector
        else:
            impl.band_detector.last_vlines = ctx.vlines()
        impl._debug_lines_image = impl._create_lines_debug(ctx.bgr, extractions)
        raw = impl._build_defects(band_summaries, all_missing, all_misaligned, density_regions)
        return self._pack(ctx, raw)
