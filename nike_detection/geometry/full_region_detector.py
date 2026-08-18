"""
Locate the island and stripe regions on a single-colour *full* scan.

Nominal layout at 2400 DPI (``island_front=true``)::

    |<------- island 5100 ------->|<- gap 580 ->|<- stripe 1050 ->|
    | V  dashed jet lines  V | V  ...  V |     |#################|
    ^                        ^                 ^                 ^
    outer vertical      inner verticals        solid bar edges

Both regions are ~33000 px tall and share a y range.

No search seeds. The whole scan is searched; the nominal dimensions above
(``config.region_reference``) only disambiguate candidates, validate the
result, and predict an edge whose print is missing.

Three signals, in order of reliability:

1. **The solid bar.** Its columns are ~100% ink over the full region
   height. Nothing in the island comes close, so it is found first and
   anchors the rest of the layout.
2. **The four vertical lines.** They are continuous over the whole region
   height, so their column ink density (~1.0) sits an order of magnitude
   above the dashed horizontal print (~0.15) and far above overspray haze
   (<0.02). A vertical that failed to print over part of its length still
   stands out, which is what makes bad corners survivable.
3. **The nominal layout.** Any edge that cannot be measured confidently is
   predicted from one that can (stripe left - gap = island right, island
   right - island width = island left, ...).

Every boundary chosen coarsely is then snapped to the real ink edge at full
resolution.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from nike_detection.config.schema import (
    AppConfig,
    BoundingBox,
    RegionReference,
    RegionSpec,
)

logger = logging.getLogger(__name__)

COLOR_TOKENS = (
    ("cyan", "Cyan"),
    ("magenta", "Magenta"),
    ("yellow", "Yellow"),
    ("black", "Key"),
    ("key", "Key"),
)

ISLAND_BGR = (0, 220, 0)
STRIPE_BGR = (255, 90, 0)
VLINE_BGR = (255, 0, 255)


class RegionDetectionError(ValueError):
    """Raised when a full scan carries no usable print structure."""


def color_prefix_from_path(path: str) -> str:
    """Output-name prefix: the ink colour when the filename carries one."""
    stem = os.path.splitext(os.path.basename(path))[0]
    lowered = stem.lower()
    for token, label in COLOR_TOKENS:
        if token in lowered:
            return label
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", stem).strip("_")
    return cleaned or "region"


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------

def detect_full_regions(
    gray: np.ndarray,
    config: AppConfig,
    image_path: str = "",
) -> Tuple[List[RegionSpec], Dict[str, Any]]:
    """Return [island, stripe] specs plus a debug/measurement dict."""
    height, width = gray.shape[:2]
    ref = config.region_reference
    island_front = bool(config.geometry.island_front)
    buffers = config.geometry.buffer or {}
    buffer_h = int(buffers.get("horizontal", 50))
    buffer_v = int(buffers.get("vertical", 50))

    warnings: List[str] = []
    sources: Dict[str, str] = {}

    scale = _coarse_scale(width)
    small = _downsample(gray, scale)
    bg, contrast = _ink_levels(small)
    ink = _density(small, bg, contrast)

    # Averaged over tens of thousands of rows, so it needs no smoothing --
    # and smoothing would flatten the very peaks we are looking for: a
    # vertical line is only a couple of columns wide at this scale.
    col = ink.mean(axis=0)
    peak = float(np.percentile(col, 99.5))
    if peak < 0.05:
        raise RegionDetectionError(
            f"No print structure found in {os.path.basename(image_path) or 'image'}"
        )
    stripe_span = _find_solid_bar(col, scale, ref, peak, warnings)
    vertical_profile = _vertical_structure_profile(ink, scale, peak)
    vlines = _find_vertical_lines(
        vertical_profile, scale, ref, stripe_span, island_front, warnings
    )
    content_span = _find_island_content(col, scale, ref, peak, stripe_span, island_front)

    island_span, stripe_span = _solve_x_layout(
        vlines=vlines,
        content_span=content_span,
        stripe_span=stripe_span,
        ref=ref,
        island_front=island_front,
        width=width,
        sources=sources,
        warnings=warnings,
    )

    y_span = _solve_y_extent(
        ink=ink,
        scale=scale,
        island_span=island_span,
        stripe_span=stripe_span,
        ref=ref,
        height=height,
        sources=sources,
        warnings=warnings,
    )

    island_span, stripe_span, y_span = _refine_full_resolution(
        gray=gray,
        bg=bg,
        contrast=contrast,
        island_span=island_span,
        stripe_span=stripe_span,
        y_span=y_span,
        scale=scale,
        sources=sources,
    )

    y0, y1 = y_span
    island_box = _pad_box(island_span, (y0, y1), buffer_h, buffer_v, width, height)
    stripe_box = _pad_box(stripe_span, (y0, y1), buffer_h, buffer_v, width, height)
    island_box, stripe_box = _split_overlap(island_box, stripe_box, island_front)

    prefix = color_prefix_from_path(image_path)
    specs = [
        _spec(f"{prefix}Island", "island", island_box),
        _spec(f"{prefix}Stripe", "stripe", stripe_box),
    ]

    measured = {
        "island_width": island_span[1] - island_span[0] + 1,
        "stripe_width": stripe_span[1] - stripe_span[0] + 1,
        "island_stripe_gap": (
            stripe_span[0] - island_span[1] - 1 if island_front
            else island_span[0] - stripe_span[1] - 1
        ),
        "height": y1 - y0 + 1,
    }
    _check_against_reference(measured, ref, warnings)

    debug = {
        "prefix": prefix,
        "scale": scale,
        "background_gray": round(bg, 1),
        "ink_contrast": round(contrast, 1),
        "vertical_lines": vlines,
        "island_x": list(island_span),
        "stripe_x": list(stripe_span),
        "y": [y0, y1],
        "measured": measured,
        "reference": {
            "island_width": ref.island_width,
            "island_stripe_gap": ref.island_stripe_gap,
            "stripe_width": ref.stripe_width,
            "height": ref.height,
        },
        "sources": sources,
        "warnings": warnings,
    }
    logger.info(
        "%s regions: island x=%s-%s stripe x=%s-%s y=%s-%s "
        "(island %spx, gap %spx, stripe %spx, height %spx)%s",
        prefix, island_span[0], island_span[1], stripe_span[0], stripe_span[1],
        y0, y1, measured["island_width"], measured["island_stripe_gap"],
        measured["stripe_width"], measured["height"],
        f" WARNINGS: {'; '.join(warnings)}" if warnings else "",
    )
    return specs, debug


# ----------------------------------------------------------------------
# Coarse structure search
# ----------------------------------------------------------------------

def _find_solid_bar(
    col: np.ndarray,
    scale: int,
    ref: RegionReference,
    peak: float,
    warnings: List[str],
) -> Optional[Tuple[int, int]]:
    """The stripe: the one wide run of near-saturated columns.

    Merged island verticals can never qualify: even if two of them touch,
    the low-ink gutter between them drags the run's *mean* density far
    below a solid bar's.
    """
    mask = col >= max(0.30, 0.55 * peak)
    merge_gap = max(1, int(round(0.05 * ref.stripe_width / scale)))
    runs = _bool_runs(mask, min_len=1, merge_gap=merge_gap)

    best: Optional[Tuple[int, int]] = None
    best_score = 0.0
    for start, end in runs:
        x0, x1 = start * scale, (end + 1) * scale - 1
        run_width = x1 - x0 + 1
        if run_width < 0.5 * ref.stripe_width or run_width > 2.5 * ref.stripe_width:
            continue
        density = float(col[start:end + 1].mean())
        if density < 0.6 * peak:
            continue
        fit = min(run_width, ref.stripe_width) / float(max(run_width, ref.stripe_width))
        score = density * (0.4 + 0.6 * fit)
        if score > best_score:
            best_score = score
            best = (x0, x1)
    if best is None:
        warnings.append("solid stripe bar not found; stripe predicted from the island")
    return best


def _vertical_structure_profile(ink: np.ndarray, scale: int, peak: float) -> np.ndarray:
    """Per-column share of the height covered by *vertical* structure.

    The island's boundary lines are printed faintly and speckled, and they
    drift sideways over 33000 rows, so their raw ink density is no higher
    than the dashed horizontal print. Binarizing and then opening along y
    with a kernel far taller than a horizontal line -- but far shorter than
    a region -- deletes the horizontal print outright, which turns a weak
    contrast into a near-binary one.
    """
    threshold = max(0.08, 0.12 * peak)
    mask = (ink >= threshold).astype(np.uint8)
    bridge = max(3, int(round(48.0 / scale)))     # close speckle gaps
    keep = max(bridge + 2, int(round(320.0 / scale)))  # drop horizontal lines
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((bridge, 1), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((keep, 1), np.uint8))
    return mask.mean(axis=0, dtype=np.float64)


def _find_vertical_lines(
    vertical_profile: np.ndarray,
    scale: int,
    ref: RegionReference,
    stripe_span: Optional[Tuple[int, int]],
    island_front: bool,
    warnings: List[str],
) -> List[Dict[str, Any]]:
    """Narrow columns of vertical structure = island boundary lines."""
    max_width = max(3 * scale, int(0.30 * ref.stripe_width))
    merge_gap = max(1, int(round(40.0 / scale)))

    lines: List[Dict[str, Any]] = []
    for level in (0.30, 0.15, 0.08):
        lines = []
        for start, end in _bool_runs(
            vertical_profile >= level, min_len=1, merge_gap=merge_gap
        ):
            x0, x1 = start * scale, (end + 1) * scale - 1
            if (x1 - x0 + 1) > max_width:
                continue
            if stripe_span is not None:
                if island_front and x0 >= stripe_span[0] - scale:
                    continue
                if not island_front and x1 <= stripe_span[1] + scale:
                    continue
            lines.append({
                "x0": int(x0),
                "x1": int(x1),
                "center": int((x0 + x1) // 2),
                "coverage": round(float(vertical_profile[start:end + 1].max()), 3),
            })
        if len(lines) >= 2:
            break
    if not lines:
        warnings.append("no island vertical lines found")
    lines.sort(key=lambda v: v["center"])
    return lines


def _find_island_content(
    col: np.ndarray,
    scale: int,
    ref: RegionReference,
    peak: float,
    stripe_span: Optional[Tuple[int, int]],
    island_front: bool,
) -> Optional[Tuple[int, int]]:
    """Fallback span: every column carrying real print, bar excluded.

    The threshold sits well above overspray haze but below the dashed
    horizontal lines, and runs are merged across the island's internal
    gutter but never across the much wider island/stripe gap.
    """
    mask = col >= max(0.02, 0.10 * peak)
    if stripe_span is not None:
        lo = max(0, stripe_span[0] // scale - 1)
        hi = min(mask.size, stripe_span[1] // scale + 2)
        mask[lo:hi] = False
    merge_gap = max(1, int(round(0.5 * ref.island_stripe_gap / scale)))
    runs = _bool_runs(mask, min_len=max(1, int(round(200.0 / scale))), merge_gap=merge_gap)
    if not runs:
        return None
    if stripe_span is not None:
        runs = [
            r for r in runs
            if (r[1] * scale < stripe_span[0]) == island_front
        ]
        if not runs:
            return None
    start, end = max(runs, key=lambda r: r[1] - r[0])
    return int(start * scale), int((end + 1) * scale - 1)


# ----------------------------------------------------------------------
# Layout solving
# ----------------------------------------------------------------------

def _solve_x_layout(
    vlines: Sequence[Dict[str, Any]],
    content_span: Optional[Tuple[int, int]],
    stripe_span: Optional[Tuple[int, int]],
    ref: RegionReference,
    island_front: bool,
    width: int,
    sources: Dict[str, str],
    warnings: List[str],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Island span from the verticals, completed by the nominal layout.

    ``inner`` is the island edge facing the gap, ``outer`` the far one.
    """
    tol_width = max(150.0, ref.tolerance * ref.island_width)
    tol_gap = max(120.0, ref.tolerance * ref.island_stripe_gap)

    predicted_inner: Optional[float] = None
    if stripe_span is not None:
        predicted_inner = (
            stripe_span[0] - ref.island_stripe_gap if island_front
            else stripe_span[1] + ref.island_stripe_gap
        )

    inner_of = (lambda v: v["x1"]) if island_front else (lambda v: v["x0"])
    outer_of = (lambda v: v["x0"]) if island_front else (lambda v: v["x1"])

    inner: Optional[float] = None
    outer: Optional[float] = None

    pair = _best_vertical_pair(vlines, ref, predicted_inner, tol_width, island_front)
    if pair is not None:
        outer_v, inner_v = pair
        outer, inner = outer_of(outer_v), inner_of(inner_v)
        sources["island_inner"] = "vertical-line"
        sources["island_outer"] = "vertical-line"
    else:
        if vlines and predicted_inner is not None:
            nearest = min(vlines, key=lambda v: abs(inner_of(v) - predicted_inner))
            if abs(inner_of(nearest) - predicted_inner) <= tol_gap:
                inner = inner_of(nearest)
                sources["island_inner"] = "vertical-line"
        if inner is None and predicted_inner is not None:
            inner = predicted_inner
            sources["island_inner"] = "predicted-from-stripe"
            warnings.append("island inner edge predicted from the stripe")
        if inner is None and content_span is not None:
            inner = content_span[1] if island_front else content_span[0]
            sources["island_inner"] = "print-content"
        if inner is None:
            raise RegionDetectionError("Could not locate the island or the stripe")

        if vlines:
            want = inner - ref.island_width if island_front else inner + ref.island_width
            nearest = min(vlines, key=lambda v: abs(outer_of(v) - want))
            if abs(outer_of(nearest) - want) <= tol_width:
                outer = outer_of(nearest)
                sources["island_outer"] = "vertical-line"
        if outer is None:
            outer = inner - ref.island_width if island_front else inner + ref.island_width
            sources["island_outer"] = "predicted-from-reference"
            warnings.append("island outer edge predicted from the reference width")

    island_span = (
        (int(round(outer)), int(round(inner))) if island_front
        else (int(round(inner)), int(round(outer)))
    )
    island_span = _extend_to_content(
        island_span, content_span, ref, island_front, sources, warnings
    )

    if stripe_span is None:
        gap = ref.island_stripe_gap
        if island_front:
            sx0 = island_span[1] + gap
            stripe_span = (sx0, sx0 + ref.stripe_width - 1)
        else:
            sx1 = island_span[0] - gap
            stripe_span = (sx1 - ref.stripe_width + 1, sx1)
        sources["stripe"] = "predicted-from-island"
    else:
        sources["stripe"] = "solid-bar"

    island_span = _clamp_span(island_span, width)
    stripe_span = _clamp_span(stripe_span, width)
    return island_span, stripe_span


def _best_vertical_pair(
    vlines: Sequence[Dict[str, Any]],
    ref: RegionReference,
    predicted_inner: Optional[float],
    tol_width: float,
    island_front: bool,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Outermost vertical pair whose span matches the nominal island width."""
    if len(vlines) < 2:
        return None
    best = None
    best_score = float("inf")
    for i, left in enumerate(vlines):
        for right in vlines[i + 1:]:
            span = right["x1"] - left["x0"] + 1
            width_error = abs(span - ref.island_width)
            if width_error > tol_width:
                continue
            score = width_error
            if predicted_inner is not None:
                inner = right["x1"] if island_front else left["x0"]
                score += abs(inner - predicted_inner)
            if score < best_score:
                best_score = score
                best = (left, right) if island_front else (right, left)
    return best


def _extend_to_content(
    island_span: Tuple[int, int],
    content_span: Optional[Tuple[int, int]],
    ref: RegionReference,
    island_front: bool,
    sources: Dict[str, str],
    warnings: List[str],
) -> Tuple[int, int]:
    """Grow the box if real print sits outside a *predicted* edge.

    An edge measured from a vertical line is the boundary by definition, so
    it is never overridden -- the head labels printed in the island/stripe
    gap must not be able to drag the island box across it.
    """
    if content_span is None:
        return island_span
    x0, x1 = island_span
    max_width = int((1.0 + ref.tolerance) * ref.island_width)
    slack = max(30, int(0.01 * ref.island_width))
    left_key = "island_outer" if island_front else "island_inner"
    right_key = "island_inner" if island_front else "island_outer"

    if (sources.get(left_key) != "vertical-line"
            and content_span[0] < x0 - slack
            and (x1 - content_span[0] + 1) <= max_width):
        x0 = content_span[0]
        sources[left_key] = "print-content"
        warnings.append("island left edge taken from print (no vertical line there)")
    if (sources.get(right_key) != "vertical-line"
            and content_span[1] > x1 + slack
            and (content_span[1] - x0 + 1) <= max_width):
        x1 = content_span[1]
        sources[right_key] = "print-content"
        warnings.append("island right edge taken from print (no vertical line there)")
    return x0, x1


def _solve_y_extent(
    ink: np.ndarray,
    scale: int,
    island_span: Tuple[int, int],
    stripe_span: Tuple[int, int],
    ref: RegionReference,
    height: int,
    sources: Dict[str, str],
    warnings: List[str],
) -> Tuple[int, int]:
    """One y range for both regions: they are printed by the same pass."""
    close_gap = max(1, int(round(400.0 / scale)))
    island_y = _row_cluster(ink, island_span, scale, close_gap, rel=0.25)
    stripe_y = _row_cluster(ink, stripe_span, scale, close_gap, rel=0.50)

    tol = ref.tolerance * ref.height
    island_ok = island_y is not None and abs((island_y[1] - island_y[0] + 1) - ref.height) <= tol
    stripe_ok = stripe_y is not None and abs((stripe_y[1] - stripe_y[0] + 1) - ref.height) <= tol

    if island_ok and stripe_ok:
        aligned = (
            abs(island_y[0] - stripe_y[0]) <= 0.02 * ref.height
            and abs(island_y[1] - stripe_y[1]) <= 0.02 * ref.height
        )
        if aligned:
            sources["y"] = "island+stripe"
            return min(island_y[0], stripe_y[0]), max(island_y[1], stripe_y[1])
        warnings.append("island and stripe disagree on the y range; using the stripe")
        sources["y"] = "stripe"
        return stripe_y
    if stripe_ok:
        sources["y"] = "stripe"
        return stripe_y
    if island_ok:
        sources["y"] = "island"
        return island_y

    candidates = [c for c in (island_y, stripe_y) if c is not None]
    if not candidates:
        warnings.append("no y extent found; using the full image height")
        sources["y"] = "image"
        return 0, height - 1
    best = min(candidates, key=lambda c: abs((c[1] - c[0] + 1) - ref.height))
    warnings.append(
        f"y extent {best[1] - best[0] + 1}px differs from the reference {ref.height}px"
    )
    sources["y"] = "island" if best is island_y else "stripe"
    return best


def _row_cluster(
    ink: np.ndarray,
    x_span: Tuple[int, int],
    scale: int,
    close_gap: int,
    rel: float,
) -> Optional[Tuple[int, int]]:
    x0 = max(0, x_span[0] // scale)
    x1 = min(ink.shape[1], max(x0 + 1, x_span[1] // scale + 1))
    row = ink[:, x0:x1].mean(axis=1)
    ceiling = float(np.percentile(row, 99.5))
    if ceiling <= 0.02:
        return None
    runs = _bool_runs(row >= max(0.03, rel * ceiling), min_len=1, merge_gap=close_gap)
    if not runs:
        return None
    start, end = max(runs, key=lambda r: r[1] - r[0])
    return int(start * scale), int((end + 1) * scale - 1)


def _check_against_reference(
    measured: Dict[str, int], ref: RegionReference, warnings: List[str]
) -> None:
    expected = {
        "island_width": ref.island_width,
        "stripe_width": ref.stripe_width,
        "island_stripe_gap": ref.island_stripe_gap,
        "height": ref.height,
    }
    for key, want in expected.items():
        got = measured[key]
        if abs(got - want) > max(60, ref.tolerance * want):
            warnings.append(f"{key} is {got}px, reference {want}px")


# ----------------------------------------------------------------------
# Full-resolution refinement
# ----------------------------------------------------------------------

def _refine_full_resolution(
    gray: np.ndarray,
    bg: float,
    contrast: float,
    island_span: Tuple[int, int],
    stripe_span: Tuple[int, int],
    y_span: Tuple[int, int],
    scale: int,
    sources: Dict[str, str],
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Snap every coarse boundary onto the real ink edge."""
    height, width = gray.shape[:2]
    x_window = max(250, 8 * scale)
    y_window = max(400, 8 * scale)
    y0, y1 = y_span
    core_y0 = int(np.clip(y0, 0, height - 2))
    core_y1 = int(np.clip(y1 + 1, core_y0 + 1, height))

    island = (
        _snap_x(gray, bg, contrast, island_span[0], core_y0, core_y1, x_window, "first"),
        _snap_x(gray, bg, contrast, island_span[1], core_y0, core_y1, x_window, "last"),
    )
    stripe = (
        _snap_x(gray, bg, contrast, stripe_span[0], core_y0, core_y1, x_window, "first"),
        _snap_x(gray, bg, contrast, stripe_span[1], core_y0, core_y1, x_window, "last"),
    )
    island = (island[0] if island[0] is not None else island_span[0],
              island[1] if island[1] is not None else island_span[1])
    stripe = (stripe[0] if stripe[0] is not None else stripe_span[0],
              stripe[1] if stripe[1] is not None else stripe_span[1])

    # The two regions share one box, so the shared y range is the union of
    # what each of them actually printed.
    tops = [
        _snap_y(gray, bg, contrast, y0, island[0], island[1] + 1, y_window, "first"),
        _snap_y(gray, bg, contrast, y0, stripe[0], stripe[1] + 1, y_window, "first"),
    ]
    bottoms = [
        _snap_y(gray, bg, contrast, y1, island[0], island[1] + 1, y_window, "last"),
        _snap_y(gray, bg, contrast, y1, stripe[0], stripe[1] + 1, y_window, "last"),
    ]
    tops = [t for t in tops if t is not None] or [y0]
    bottoms = [b for b in bottoms if b is not None] or [y1]
    refined_y = (min(tops), max(bottoms))

    sources["refinement"] = f"full-resolution snap (+/-{x_window}px in x, +/-{y_window}px in y)"
    if refined_y[1] <= refined_y[0]:
        refined_y = (y0, y1)
    if island[1] <= island[0]:
        island = island_span
    if stripe[1] <= stripe[0]:
        stripe = stripe_span
    return island, stripe, refined_y


def _snap_x(
    gray: np.ndarray,
    bg: float,
    contrast: float,
    guess: int,
    y0: int,
    y1: int,
    window: int,
    edge: str,
) -> Optional[int]:
    height, width = gray.shape[:2]
    a = int(np.clip(guess - window, 0, width - 1))
    b = int(np.clip(guess + window + 1, a + 1, width))
    strip = gray[y0:y1:4, a:b]
    density = _density(strip, bg, contrast).mean(axis=0)
    index = _edge_index(density, edge, min_run=2)
    return None if index is None else a + index


def _snap_y(
    gray: np.ndarray,
    bg: float,
    contrast: float,
    guess: int,
    x0: int,
    x1: int,
    window: int,
    edge: str,
) -> Optional[int]:
    height, width = gray.shape[:2]
    x0 = int(np.clip(x0, 0, width - 1))
    x1 = int(np.clip(x1, x0 + 1, width))
    a = int(np.clip(guess - window, 0, height - 1))
    b = int(np.clip(guess + window + 1, a + 1, height))
    strip = gray[a:b, x0:x1:4]
    density = _density(strip, bg, contrast).mean(axis=1)
    index = _edge_index(density, edge, min_run=2)
    return None if index is None else a + index


def _edge_index(density: np.ndarray, edge: str, min_run: int) -> Optional[int]:
    """First/last index of real structure inside a refinement window.

    The threshold sits 40% of the way from the window's paper level to its
    strongest ink. A refinement window always straddles an edge, so its low
    quantile is paper: that keeps a partly-missing vertical line above the
    dashed print it borders, and keeps stray specks below both.
    """
    if density.size == 0:
        return None
    strongest = float(density.max())
    if strongest <= 0.02:
        return None
    paper = float(np.percentile(density, 10))
    if paper > 0.5 * strongest:
        paper = 0.0
    threshold = max(paper + 0.4 * (strongest - paper), 0.02)
    runs = _bool_runs(density >= threshold, min_len=min_run, merge_gap=0)
    if not runs:
        return None
    return runs[0][0] if edge == "first" else runs[-1][1]


# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------

def draw_region_overlay(
    bgr: np.ndarray,
    specs: Sequence[RegionSpec],
    debug: Optional[Dict[str, Any]] = None,
    max_edge: int = 4000,
) -> np.ndarray:
    """Whole-scan overlay: island green, stripe blue, verticals magenta."""
    height, width = bgr.shape[:2]
    scale = 1.0
    vis = bgr
    if max(height, width) > max_edge:
        scale = max_edge / float(max(height, width))
        vis = cv2.resize(
            bgr, (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    out = vis.copy()
    if debug:
        for vline in debug.get("vertical_lines") or []:
            x0 = int(vline["x0"] * scale)
            x1 = max(x0 + 1, int(vline["x1"] * scale))
            cv2.rectangle(out, (x0, 0), (x1, out.shape[0] - 1), VLINE_BGR, -1)
    for spec in specs:
        x0, y0, x1, y1 = spec.bounding_box_pixels.as_int_xyxy()
        color = ISLAND_BGR if spec.type == "island" else STRIPE_BGR
        p0 = (int(x0 * scale), int(y0 * scale))
        p1 = (int(x1 * scale), int(y1 * scale))
        cv2.rectangle(out, p0, p1, color, 3)
        cv2.putText(
            out, spec.name, (p0[0] + 6, p0[1] + 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA,
        )
    return out


def draw_corner_montage(
    bgr: np.ndarray,
    specs: Sequence[RegionSpec],
    crop: int = 900,
) -> Optional[np.ndarray]:
    """Full-resolution crops of each region's top-left and bottom-right.

    This is the only view that shows whether a corner is actually on the
    print, which a whole-scan overlay is far too small to reveal.
    """
    panels: List[np.ndarray] = []
    for spec in specs:
        x0, y0, x1, y1 = spec.bounding_box_pixels.as_int_xyxy()
        color = ISLAND_BGR if spec.type == "island" else STRIPE_BGR
        panels.append(_corner_panel(bgr, x0, y0, crop, color, f"{spec.name} top-left"))
        panels.append(_corner_panel(bgr, x1, y1, crop, color, f"{spec.name} bottom-right"))
    if not panels:
        return None
    rows = [np.hstack(panels[i:i + 2]) for i in range(0, len(panels), 2)]
    widest = max(row.shape[1] for row in rows)
    padded = [
        np.pad(row, ((0, 0), (0, widest - row.shape[1]), (0, 0)), constant_values=255)
        for row in rows
    ]
    return np.vstack(padded)


def _corner_panel(
    bgr: np.ndarray,
    x: int,
    y: int,
    crop: int,
    color: Tuple[int, int, int],
    label: str,
) -> np.ndarray:
    height, width = bgr.shape[:2]
    half = crop // 2
    x0 = int(np.clip(x - half, 0, max(0, width - 1)))
    y0 = int(np.clip(y - half, 0, max(0, height - 1)))
    x1 = int(np.clip(x0 + crop, 1, width))
    y1 = int(np.clip(y0 + crop, 1, height))
    panel = np.full((crop, crop, 3), 245, dtype=np.uint8)
    piece = bgr[y0:y1, x0:x1]
    if piece.ndim == 2:
        piece = cv2.cvtColor(piece, cv2.COLOR_GRAY2BGR)
    panel[: piece.shape[0], : piece.shape[1]] = piece
    cx = int(np.clip(x - x0, 0, crop - 1))
    cy = int(np.clip(y - y0, 0, crop - 1))
    cv2.line(panel, (cx, 0), (cx, crop - 1), color, 2)
    cv2.line(panel, (0, cy), (crop - 1, cy), color, 2)
    cv2.putText(
        panel, f"{label} ({x},{y})", (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
    )
    return cv2.copyMakeBorder(panel, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(60, 60, 60))


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------

def _spec(name: str, kind: str, xyxy: Tuple[int, int, int, int]) -> RegionSpec:
    x0, y0, x1, y1 = xyxy
    return RegionSpec(
        name=name,
        type=kind,
        bounding_box_pixels=BoundingBox(
            top_x=float(x0), top_y=float(y0),
            bottom_x=float(x1), bottom_y=float(y1),
        ),
    )


def _coarse_scale(width: int) -> int:
    scale = 1
    while width // (scale * 2) >= 700 and scale < 16:
        scale *= 2
    return scale


def _downsample(gray: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return gray
    height, width = gray.shape[:2]
    return cv2.resize(
        gray, (max(1, width // scale), max(1, height // scale)),
        interpolation=cv2.INTER_AREA,
    )


def _ink_levels(small: np.ndarray) -> Tuple[float, float]:
    """(paper gray, paper-to-ink range) measured on the scan itself."""
    values = small.astype(np.float32)
    background = float(np.percentile(values, 95))
    darkest = float(np.percentile(values, 2))
    return background, max(12.0, background - darkest)


def _density(image: np.ndarray, background: float, contrast: float) -> np.ndarray:
    values = image.astype(np.float32)
    return np.clip((background - values) / contrast, 0.0, 1.0)


def _bool_runs(mask: np.ndarray, min_len: int = 1, merge_gap: int = 0) -> List[Tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > merge_gap + 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    return [
        (int(s), int(e)) for s, e in zip(starts, ends) if (e - s + 1) >= min_len
    ]


def _clamp_span(span: Tuple[int, int], width: int) -> Tuple[int, int]:
    x0 = int(np.clip(span[0], 0, width - 2))
    x1 = int(np.clip(span[1], x0 + 1, width - 1))
    return x0, x1


def _pad_box(
    x_span: Tuple[int, int],
    y_span: Tuple[int, int],
    buffer_h: int,
    buffer_v: int,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x0 = int(np.clip(x_span[0] - buffer_h, 0, width - 1))
    x1 = int(np.clip(x_span[1] + buffer_h + 1, x0 + 1, width))
    y0 = int(np.clip(y_span[0] - buffer_v, 0, height - 1))
    y1 = int(np.clip(y_span[1] + buffer_v + 1, y0 + 1, height))
    return x0, y0, x1, y1


def _split_overlap(
    island: Tuple[int, int, int, int],
    stripe: Tuple[int, int, int, int],
    island_front: bool,
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    ix0, iy0, ix1, iy1 = island
    sx0, sy0, sx1, sy1 = stripe
    if island_front and ix1 > sx0:
        middle = (ix1 + sx0) // 2
        ix1, sx0 = middle, middle
    elif not island_front and sx1 > ix0:
        middle = (sx1 + ix0) // 2
        sx1, ix0 = middle, middle
    return (ix0, iy0, max(ix1, ix0 + 1), iy1), (sx0, sy0, max(sx1, sx0 + 1), sy1)
