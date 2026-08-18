"""Visualization writers. Detectors score; this module draws and saves."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from nike_detection.config.schema import RegionSpec
from nike_detection.geometry.full_region_detector import overlay_canvas
from nike_detection.io.image_saver import save_image
from nike_detection.pipeline.context import ImageContext
from nike_detection.pipeline.types import Defect, ImageResult

logger = logging.getLogger(__name__)

VOID_TIFF_KEYS = {"void"}

# Glyph colours match the per-detector overlays (BGR).
DEFECT_BGR: Dict[str, Tuple[int, int, int]] = {
    "missing_line": (0, 0, 255),
    "misaligned_line": (0, 255, 255),
    "stitch_error": (255, 80, 0),
    "high_density_region": (0, 128, 255),
    "stripe_misalignment": (255, 255, 0),
    "roll_error": (255, 0, 180),
    "edge_roughness": (180, 0, 255),
    "void": (0, 165, 255),
}
SKIP_OVERLAY_KINDS = {
    "bands_detected",
    "edge_roughness_summary",
    "lines_detected",
}
ISLAND_BOX_BGR = (80, 180, 80)
STRIPE_BOX_BGR = (180, 120, 40)
_LEGEND = (
    ("missing nozzles", DEFECT_BGR["missing_line"]),
    ("hazy / misaligned", DEFECT_BGR["misaligned_line"]),
    ("stitch / calibration", DEFECT_BGR["stitch_error"]),
    ("edge roughness", DEFECT_BGR["edge_roughness"]),
    ("stripe misalignment", DEFECT_BGR["stripe_misalignment"]),
    ("void", DEFECT_BGR["void"]),
)


def downscale(image: np.ndarray, max_edge: int = 8000) -> np.ndarray:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return image
    scale = max_edge / float(longest)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def save_visualization(
    ctx: ImageContext,
    detector: Any,
    defects: List[Defect],
    output_dir: str,
    detector_key: str,
    write: bool = True,
    downscale_vis: bool = False,
) -> Optional[str]:
    if not write:
        return None
    vis = detector.render(ctx, defects)
    if vis is None:
        return None
    if downscale_vis:
        vis = downscale(vis)
    if detector_key in VOID_TIFF_KEYS:
        path = f"{output_dir}/{detector_key}_visualization.tiff"
        ok = cv2.imwrite(path, vis)
        return path if ok else None
    return save_image(output_dir, detector_key, vis, "visualization") or None


def save_debug(detector: Any, output_dir: str, base_name: str, enabled: bool) -> None:
    if not enabled:
        return
    impl = getattr(detector, "_impl", None)
    if impl is not None and hasattr(impl, "save_debug_images"):
        try:
            impl.save_debug_images(output_dir, base_name)
        except Exception:
            logger.exception("Failed to save debug images for %s", base_name)


def _pt(x: float, y: float, origin: Tuple[int, int], scale: float) -> Tuple[int, int]:
    return (int(round((x + origin[0]) * scale)), int(round((y + origin[1]) * scale)))


def _clamp_rect(
    p0: Tuple[int, int],
    p1: Tuple[int, int],
    shape: Tuple[int, ...],
    min_span: int = 3,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    x0, y0 = min(p0[0], p1[0]), min(p0[1], p1[1])
    x1, y1 = max(p0[0], p1[0]), max(p0[1], p1[1])
    if x1 - x0 < min_span:
        pad = (min_span - (x1 - x0)) // 2
        x0 -= pad
        x1 = x0 + min_span
    if y1 - y0 < min_span:
        pad = (min_span - (y1 - y0)) // 2
        y0 -= pad
        y1 = y0 + min_span
    h, w = shape[:2]
    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w - 1, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h - 1, y1))
    return (x0, y0), (x1, y1)


def _spec_xyxy(spec: Optional[RegionSpec]) -> Optional[Tuple[int, int, int, int]]:
    if spec is None:
        return None
    return spec.bounding_box_pixels.as_int_xyxy()


def _draw_legend(image: np.ndarray, stamp: str) -> None:
    bar_h = 36
    cv2.rectangle(image, (0, 0), (image.shape[1], bar_h), (20, 20, 20), -1)
    x = 8
    cv2.putText(
        image, stamp, (x, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA,
    )
    x += 8 + cv2.getTextSize(stamp, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
    for label, color in _LEGEND:
        cv2.rectangle(image, (x, 10), (x + 14, 24), color, -1)
        x += 18
        cv2.putText(
            image, label, (x, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA,
        )
        x += 8 + cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
        if x > image.shape[1] - 80:
            break


def _draw_one_defect(
    image: np.ndarray,
    item: Dict[str, Any],
    origin: Tuple[int, int],
    scale: float,
    spec: Optional[RegionSpec],
) -> None:
    kind = str(item.get("type") or "")
    if kind in SKIP_OVERLAY_KINDS:
        return
    color = DEFECT_BGR.get(kind)
    if color is None:
        return
    box = _spec_xyxy(spec)
    thickness = max(2, int(round(3 * scale)) if scale >= 0.2 else 2)

    if kind in {"missing_line", "misaligned_line", "stitch_error"}:
        y = float(item.get("y", (item.get("location") or [0, 0])[1]))
        x0 = float(item.get("start_x", origin[0]))
        x1 = float(item.get("end_x", x0 + 1))
        half = 8.0 if kind == "stitch_error" else 5.0
        p0, p1 = _clamp_rect(
            _pt(x0, y - half, origin, scale),
            _pt(x1, y + half, origin, scale),
            image.shape,
        )
        cv2.rectangle(image, p0, p1, color, -1)
        return

    if kind == "high_density_region":
        bbox = item.get("bbox") or []
        if len(bbox) == 4:
            x, y, w, h = [float(v) for v in bbox]
            p0, p1 = _clamp_rect(
                _pt(x, y, origin, scale),
                _pt(x + w, y + h, origin, scale),
                image.shape,
                min_span=6,
            )
            cv2.rectangle(image, p0, p1, color, thickness)
        return

    if kind == "void":
        bbox = item.get("bbox") or []
        if len(bbox) == 4:
            x, y, w, h = [float(v) for v in bbox]
            p0, p1 = _clamp_rect(
                _pt(x, y, origin, scale),
                _pt(x + w, y + h, origin, scale),
                image.shape,
                min_span=4,
            )
            cv2.rectangle(image, p0, p1, color, max(2, thickness))
        return

    if kind == "stripe_misalignment":
        y = float(item.get("y", (item.get("location") or [0, 0])[1]))
        if box is not None:
            p0 = _pt(0, y, origin, scale)
            p1 = _pt(box[2] - box[0], y, origin, scale)
        else:
            x = float(item.get("x", (item.get("location") or [0, 0])[0]))
            p0 = _pt(x - 40, y, origin, scale)
            p1 = _pt(x + 40, y, origin, scale)
        cv2.line(image, (p0[0], p0[1]), (p1[0], p1[1]), color, thickness)
        loc = _pt(
            float(item.get("x", 0)), y, origin, scale,
        )
        cv2.circle(image, loc, max(4, thickness + 2), color, -1)
        return

    if kind == "roll_error":
        y0 = float(item.get("y0", 0))
        y1 = float(item.get("y1", y0 + 1))
        if box is not None:
            p0, p1 = _clamp_rect(
                _pt(0, y0, origin, scale),
                _pt(box[2] - box[0], y1, origin, scale),
                image.shape,
                min_span=4,
            )
        else:
            loc = item.get("location") or [0, 0]
            p0, p1 = _clamp_rect(
                _pt(float(loc[0]) - 20, y0, origin, scale),
                _pt(float(loc[0]) + 20, y1, origin, scale),
                image.shape,
                min_span=4,
            )
        cv2.rectangle(image, p0, p1, color, thickness)
        return

    if kind == "edge_roughness":
        y0 = float(item.get("y0", 0))
        y1 = float(item.get("y1", y0 + 1))
        x = float(item.get("x", (item.get("location") or [0, 0])[0]))
        p0 = _pt(x, y0, origin, scale)
        p1 = _pt(x, y1, origin, scale)
        cv2.line(image, p0, p1, color, max(2, thickness))
        return


def draw_full_defect_overlay(
    scan: Any,
    results: Sequence[ImageResult],
    specs: Optional[Sequence[RegionSpec]] = None,
    max_edge: int = 4000,
    stamp: Optional[str] = None,
) -> np.ndarray:
    """Whole-scan image with every enabled detector's findings drawn on it."""
    out, scale = overlay_canvas(scan, max_edge)
    by_name = {spec.name: spec for spec in (specs or [])}
    for spec in specs or []:
        x0, y0, x1, y1 = spec.bounding_box_pixels.as_int_xyxy()
        color = ISLAND_BOX_BGR if spec.type == "island" else STRIPE_BOX_BGR
        p0 = (int(x0 * scale), int(y0 * scale))
        p1 = (int(x1 * scale), int(y1 * scale))
        cv2.rectangle(out, p0, p1, color, 1)
        cv2.putText(
            out, spec.name, (p0[0] + 4, max(48, p0[1] + 22)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA,
        )
    for result in results:
        origin = tuple(result.origin_xy or (0, 0))
        spec = by_name.get(result.image_name)
        for detection in result.detections.values():
            for item in detection.defects or []:
                if isinstance(item, dict):
                    _draw_one_defect(out, item, origin, scale, spec)
    label = stamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _draw_legend(out, f"Defects  {label}")
    return out
