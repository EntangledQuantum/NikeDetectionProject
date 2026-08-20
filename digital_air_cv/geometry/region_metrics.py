"""Score predicted region boxes against ground-truth print corners."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

INWARD_FAIL_PX = 15
OUTWARD_CAP_PX = 80
IOU_MISSING_INK = 0.98
IOU_SLANT = 0.95
COVERAGE_MIN = 0.995


def aabb_from_xyxy(box: Sequence[float]) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = [float(v) for v in box]
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def aabb_from_corners(corners: Dict[str, Sequence[float]]) -> Tuple[float, float, float, float]:
    xs = [float(pt[0]) for pt in corners.values()]
    ys = [float(pt[1]) for pt in corners.values()]
    return min(xs), min(ys), max(xs), max(ys)


def aabb_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = aabb_from_xyxy(a)
    bx0, by0, bx1, by1 = aabb_from_xyxy(b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def inward_offsets(
    pred: Sequence[float], gt: Sequence[float]
) -> Dict[str, float]:
    """Positive = predicted edge sits inside the GT print (clips ink)."""
    px0, py0, px1, py1 = aabb_from_xyxy(pred)
    gx0, gy0, gx1, gy1 = aabb_from_xyxy(gt)
    return {
        "left": float(px0 - gx0),
        "top": float(py0 - gy0),
        "right": float(gx1 - px1),
        "bottom": float(gy1 - py1),
    }


def max_inward(offsets: Dict[str, float]) -> float:
    return float(max(offsets.values()))


def max_outward(offsets: Dict[str, float]) -> float:
    return float(max(-v for v in offsets.values()))


def corner_errors(
    pred: Dict[str, Sequence[float]],
    gt: Dict[str, Sequence[float]],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, gpt in gt.items():
        ppt = pred.get(name)
        if ppt is None:
            out[name] = float("inf")
            continue
        dx = float(ppt[0]) - float(gpt[0])
        dy = float(ppt[1]) - float(gpt[1])
        out[name] = float(np.hypot(dx, dy))
    return out


def print_coverage(pred: Sequence[float], gt: Sequence[float]) -> float:
    """Fraction of the GT AABB still inside the predicted AABB."""
    px0, py0, px1, py1 = aabb_from_xyxy(pred)
    gx0, gy0, gx1, gy1 = aabb_from_xyxy(gt)
    ix0, iy0 = max(px0, gx0), max(py0, gy0)
    ix1, iy1 = min(px1, gx1), min(py1, gy1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    gt_area = max(0.0, gx1 - gx0) * max(0.0, gy1 - gy0)
    return float(inter / gt_area) if gt_area > 0 else 0.0


def corners_from_xyxy(x0: float, y0: float, x1: float, y1: float) -> Dict[str, List[float]]:
    return {
        "tl": [float(x0), float(y0)],
        "tr": [float(x1), float(y0)],
        "bl": [float(x0), float(y1)],
        "br": [float(x1), float(y1)],
    }


def score_region(
    pred_xyxy: Sequence[float],
    gt_xyxy: Sequence[float],
    pred_corners: Optional[Dict[str, Sequence[float]]] = None,
    gt_corners: Optional[Dict[str, Sequence[float]]] = None,
    *,
    iou_min: float = IOU_MISSING_INK,
    inward_fail: float = INWARD_FAIL_PX,
    outward_cap: float = OUTWARD_CAP_PX,
    coverage_min: float = COVERAGE_MIN,
) -> Dict[str, Any]:
    offsets = inward_offsets(pred_xyxy, gt_xyxy)
    errors = (
        corner_errors(pred_corners or {}, gt_corners or {})
        if gt_corners
        else {}
    )
    inward = max_inward(offsets)
    outward = max_outward(offsets)
    iou = aabb_iou(pred_xyxy, gt_xyxy)
    coverage = print_coverage(pred_xyxy, gt_xyxy)
    reasons: List[str] = []
    if inward > inward_fail:
        reasons.append(f"inward {inward:.1f}px > {inward_fail}")
    if outward > outward_cap:
        reasons.append(f"outward {outward:.1f}px > {outward_cap}")
    if iou < iou_min:
        reasons.append(f"iou {iou:.4f} < {iou_min}")
    if coverage < coverage_min:
        reasons.append(f"coverage {coverage:.4f} < {coverage_min}")
    return {
        "pred": [float(v) for v in aabb_from_xyxy(pred_xyxy)],
        "gt": [float(v) for v in aabb_from_xyxy(gt_xyxy)],
        "offsets": offsets,
        "max_inward": inward,
        "max_outward": outward,
        "iou": iou,
        "coverage": coverage,
        "corner_errors": errors,
        "pass": not reasons,
        "reasons": reasons,
    }


def boxes_from_debug(debug: Dict[str, Any]) -> Dict[str, Dict[str, Tuple[int, int, int, int]]]:
    """Unpadded print boxes from detect_full_regions debug payload."""
    out: Dict[str, Dict[str, Tuple[int, int, int, int]]] = {}
    for color in debug.get("colors") or []:
        name = str(color.get("name") or "Region")
        y0, y1 = color["y"]
        ix0, ix1 = color["island_x"]
        sx0, sx1 = color["stripe_x"]
        out[name] = {
            "island": (int(ix0), int(y0), int(ix1), int(y1)),
            "stripe": (int(sx0), int(y0), int(sx1), int(y1)),
        }
    return out


def corners_from_debug(debug: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    out: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for color in debug.get("colors") or []:
        name = str(color.get("name") or "Region")
        stored = color.get("corners") or {}
        if stored:
            out[name] = stored
            continue
        y0, y1 = color["y"]
        ix0, ix1 = color["island_x"]
        sx0, sx1 = color["stripe_x"]
        out[name] = {
            "island": corners_from_xyxy(ix0, y0, ix1, y1),
            "stripe": corners_from_xyxy(sx0, y0, sx1, y1),
        }
    return out
