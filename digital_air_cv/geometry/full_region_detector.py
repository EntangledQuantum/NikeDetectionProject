"""
Locate every colour's island and stripe on a *full* press scan.

One colour block at 2400 DPI (``island_front=true``)::

    |<------- island 5100 ------->|<- gap 580 ->|<- stripe 1050 ->|
    | V  dashed jet lines  V | V ... V |        |#################|
    ^                        ^                  ^                 ^
    outer vertical      inner verticals         solid bar edges

A multi-colour scan repeats that block once per colour, left to right in
``geometry.colors`` order, separated by ``color_gap`` (~250 px). Successive
colours may sit up to ``color_y_tolerance`` (~200 px) higher or lower, so
every colour gets its own y range.

No search seeds and no positional priors: the scan is searched from (0, 0).
``config.region_reference`` supplies nominal sizes only, used to
disambiguate candidates, validate the result, and predict an edge whose
print is missing.

Three signals, in order of reliability:

1. **The solid bars.** A bar's columns are ~100% ink over the full region
   height; nothing in an island comes close. One bar is found per colour and
   anchors that colour's block.
2. **The four vertical lines.** They run the whole region height, so after
   the horizontal print is morphologically removed they are the only thing
   left in an island. A vertical that failed to print over part of its
   length still stands out, which is what makes bad corners survivable.
3. **The nominal layout.** Any edge that cannot be measured is predicted
   from one that can, and a colour whose bar is missing entirely is
   recovered from its verticals or from the lattice of its neighbours.

Ink is measured as the per-pixel *minimum* of B, G and R rather than
luminance: yellow is nearly invisible in grayscale (~226 against 250 paper)
but its blue channel is as dark as any other ink, so the minimum channel
puts all four colours on the same footing.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from digital_air_cv.config.schema import (
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

# Vertical structure below this share of a region's height is not a boundary
# line. Kept low so a faint colour still registers; candidates are validated
# geometrically afterwards.
VLINE_FLOOR = 0.08

# Working-set target for any pass that has to walk the whole sheet.
_CHUNK_BYTES = 256 * 1024 * 1024


class RegionDetectionError(ValueError):
    """Raised when a full scan carries no usable print structure."""


@dataclass
class _Block:
    """One colour: its island, its stripe, and the y range they share."""

    island: Tuple[int, int]
    stripe: Tuple[int, int]
    y: Tuple[int, int] = (0, 0)
    sources: Dict[str, str] = field(default_factory=dict)
    vlines: List[Dict[str, Any]] = field(default_factory=list)
    corners: Dict[str, Any] = field(default_factory=dict)
    edge_stations: Dict[str, Any] = field(default_factory=dict)


def color_token_from_path(path: str) -> Optional[str]:
    lowered = os.path.splitext(os.path.basename(path))[0].lower()
    for token, label in COLOR_TOKENS:
        if token in lowered:
            return label
    return None


def color_prefix_from_path(path: str) -> str:
    """Output-name prefix: the ink colour when the filename carries one."""
    token = color_token_from_path(path)
    if token:
        return token
    stem = os.path.splitext(os.path.basename(path))[0]
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", stem).strip("_")
    return cleaned or "region"


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------

def detect_full_regions(
    scan: Any,
    config: AppConfig,
    image_path: str = "",
) -> Tuple[List[RegionSpec], Dict[str, Any]]:
    """Return island+stripe specs for every colour, plus measurements.

    ``scan`` is a ``Scan`` or a plain array. Nothing here reads the whole
    image at once, so a memory-mapped multi-gigabyte sheet is fine.
    """
    image = _pixels(scan)
    height, width = image.shape[:2]
    ref = config.region_reference
    island_front = bool(config.geometry.island_front)
    buffers = config.geometry.buffer or {}
    buffer_h = int(buffers.get("horizontal", 50))
    buffer_v = int(buffers.get("vertical", 50))
    warnings: List[str] = []

    scale = _coarse_scale(ref, width, height)
    small = _coarse_ink_map(image, scale)
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

    bars = _find_solid_bars(col, scale, ref, peak)
    vertical_profile = _vertical_structure_profile(ink, scale, peak, ref)
    vlines = _find_vertical_lines(vertical_profile, scale, ref, bars)

    ltr = _blocks_from_bars(bars, vlines, ref, island_front, warnings)
    rtl = _blocks_from_verticals_rtl(bars, vlines, ref, island_front, warnings)
    blocks = _merge_directional_blocks(ltr, rtl, ref, island_front, warnings)
    blocks += _blocks_from_orphan_verticals(vlines, blocks, ref, island_front, warnings)
    blocks.sort(key=lambda b: b.island[0])
    blocks = _dedupe_overlapping_blocks(blocks, ref)
    blocks = _complete_lattice(blocks, ref, island_front, warnings)
    if not blocks:
        raise RegionDetectionError(
            f"No island or stripe found in {os.path.basename(image_path) or 'image'}"
        )

    # A colour recovered from the lattice printed nothing, so there is no
    # ink to measure a y extent from or to snap its edges onto; it borrows
    # both from the colours around it once those are settled.
    ghosts = [i for i, b in enumerate(blocks) if b.sources.get("y") == "lattice"]
    for index, block in enumerate(blocks):
        block.island = _clamp_span(block.island, width)
        block.stripe = _clamp_span(block.stripe, width)
        if index in ghosts:
            continue
        block.y = _solve_y_extent(ink, scale, block, ref, height, warnings)
        _refine_block(
            image, bg, contrast, block, ref,
            bounds=_neighbour_bounds(blocks, index, ref, width),
            island_front=island_front,
            geometry=config.geometry,
            warnings=warnings,
        )
    for index in ghosts:
        blocks[index].y = _borrow_y(blocks, index, ghosts, ref, height)

    names = _name_blocks(blocks, config.geometry.colors, image_path, warnings)
    _check_layout(blocks, names, ref, island_front, warnings)

    specs: List[RegionSpec] = []
    colors: List[Dict[str, Any]] = []
    for name, block in zip(names, blocks):
        island_box = _pad_box(block.island, block.y, buffer_h, buffer_v, width, height)
        stripe_box = _pad_box(block.stripe, block.y, buffer_h, buffer_v, width, height)
        island_box, stripe_box = _split_overlap(island_box, stripe_box, island_front)
        specs.append(_spec(
            f"{name}Island", "island", island_box,
            (block.corners or {}).get("island"),
        ))
        specs.append(_spec(
            f"{name}Stripe", "stripe", stripe_box,
            (block.corners or {}).get("stripe"),
        ))
        colors.append({
            "name": name,
            "island_x": list(block.island),
            "stripe_x": list(block.stripe),
            "y": list(block.y),
            "measured": _block_measurements(block, island_front),
            "sources": block.sources,
            "vertical_lines": block.vlines,
            "corners": block.corners,
            "edge_stations": block.edge_stations,
        })

    prefix = color_prefix_from_path(image_path)
    debug = {
        "prefix": prefix,
        "scale": scale,
        "paper_level": round(bg, 1),
        "ink_contrast": round(contrast, 1),
        "colors": colors,
        "reference": {
            "island_width": ref.island_width,
            "island_stripe_gap": ref.island_stripe_gap,
            "stripe_width": ref.stripe_width,
            "height": ref.height,
            "color_gap": ref.color_gap,
        },
        "warnings": warnings,
    }
    logger.info(
        "%s: %s colour block(s) %s%s",
        prefix, len(blocks),
        [f"{c['name']} island {c['island_x']} stripe {c['stripe_x']} y {c['y']}"
         for c in colors],
        f" WARNINGS: {'; '.join(warnings)}" if warnings else "",
    )
    return specs, debug


# ----------------------------------------------------------------------
# Coarse structure search
# ----------------------------------------------------------------------

def _find_solid_bars(
    col: np.ndarray, scale: int, ref: RegionReference, peak: float
) -> List[Tuple[int, int]]:
    """Every stripe: a wide run of near-saturated columns.

    Two island verticals that happen to touch can never qualify -- the
    low-ink gutter between them drags the run's *mean* density far below a
    solid bar's.
    """
    mask = col >= max(0.30, 0.55 * peak)
    merge_gap = max(1, int(round(0.05 * ref.stripe_width / scale)))
    bars: List[Tuple[int, int]] = []
    for start, end in _bool_runs(mask, min_len=1, merge_gap=merge_gap):
        x0, x1 = start * scale, (end + 1) * scale - 1
        run_width = x1 - x0 + 1
        if run_width < 0.5 * ref.stripe_width or run_width > 2.5 * ref.stripe_width:
            continue
        if float(col[start:end + 1].mean()) < 0.6 * peak:
            continue
        bars.append((int(x0), int(x1)))
    return bars


def _vertical_structure_profile(
    ink: np.ndarray, scale: int, peak: float, ref: RegionReference
) -> np.ndarray:
    """Per-column share of the height covered by *vertical* structure.

    Island boundary lines are printed faintly and speckled, and they drift
    sideways over 33000 rows, so their raw ink density is no higher than the
    dashed horizontal print. Binarizing and then opening along y with a
    kernel far taller than a horizontal line -- but far shorter than a
    region -- deletes the horizontal print outright, which turns a weak
    contrast into a near-binary one.
    """
    threshold = max(0.08, 0.12 * peak)
    mask = (ink >= threshold).astype(np.uint8)
    # Both kernels are region-relative so the detector is not tied to
    # 2400 DPI: ~1/700 of a region closes the speckle gaps in a dotted
    # line without reaching across two horizontal lines, and ~1/100 of a
    # region is far thicker than any horizontal line yet far shorter than
    # a vertical one.
    bridge = max(3, int(round(ref.height / 700.0 / scale)))
    keep = max(bridge + 2, int(round(ref.height / 100.0 / scale)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((bridge, 1), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((keep, 1), np.uint8))
    return mask.mean(axis=0, dtype=np.float64)


def _find_vertical_lines(
    vertical_profile: np.ndarray,
    scale: int,
    ref: RegionReference,
    bars: Sequence[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    """Narrow columns of vertical structure = island boundary lines.

    Candidates are collected at a permissive floor so a faint colour is not
    dropped next to a strong one, but each line's extent is then taken at
    half of its *own* peak, which keeps the envelope tight.
    """
    max_width = max(3 * scale, int(0.30 * ref.stripe_width))
    merge_gap = max(1, int(round(40.0 / scale)))

    lines: List[Dict[str, Any]] = []
    for start, end in _bool_runs(
        vertical_profile >= VLINE_FLOOR, min_len=1, merge_gap=merge_gap
    ):
        segment = vertical_profile[start:end + 1]
        local_peak = float(segment.max())
        core = np.flatnonzero(segment >= 0.5 * local_peak)
        x0 = (start + int(core[0])) * scale
        x1 = (start + int(core[-1]) + 1) * scale - 1
        if (x1 - x0 + 1) > max_width:
            continue
        if any(x0 <= bar[1] and x1 >= bar[0] for bar in bars):
            continue
        lines.append({
            "x0": int(x0),
            "x1": int(x1),
            "center": int((x0 + x1) // 2),
            "coverage": round(local_peak, 3),
        })
    lines.sort(key=lambda v: v["center"])
    return lines


# ----------------------------------------------------------------------
# Block assembly
# ----------------------------------------------------------------------

def _blocks_from_bars(
    bars: Sequence[Tuple[int, int]],
    vlines: Sequence[Dict[str, Any]],
    ref: RegionReference,
    island_front: bool,
    warnings: List[str],
) -> List[_Block]:
    """One block per solid bar, its island read off the vertical lines."""
    tol_width = max(150.0, ref.tolerance * ref.island_width)
    tol_gap = max(120.0, ref.tolerance * ref.island_stripe_gap)

    blocks: List[_Block] = []
    for bar in bars:
        predicted_inner = (
            bar[0] - ref.island_stripe_gap if island_front
            else bar[1] + ref.island_stripe_gap
        )
        reach = ref.island_width + tol_width
        lo = predicted_inner - reach if island_front else predicted_inner - tol_gap
        hi = predicted_inner + tol_gap if island_front else predicted_inner + reach
        local = [v for v in vlines if lo <= v["center"] <= hi]

        sources: Dict[str, str] = {}
        pair = _best_vertical_pair(
            local, ref, predicted_inner, tol_width, tol_gap, island_front
        )
        if pair is not None:
            outer_v, inner_v = pair
            outer = outer_v["x0"] if island_front else outer_v["x1"]
            inner = inner_v["x1"] if island_front else inner_v["x0"]
            sources["island_inner"] = "vertical-line"
            sources["island_outer"] = "vertical-line"
            used = [outer_v, inner_v]
        else:
            inner = None
            outer = None
            used = []
            if local:
                inner_of = (lambda v: v["x1"]) if island_front else (lambda v: v["x0"])
                nearest = min(local, key=lambda v: abs(inner_of(v) - predicted_inner))
                if abs(inner_of(nearest) - predicted_inner) <= tol_gap:
                    inner = inner_of(nearest)
                    sources["island_inner"] = "vertical-line"
                    used.append(nearest)
            if inner is None:
                inner = predicted_inner
                sources["island_inner"] = "predicted-from-stripe"
                warnings.append("an island inner edge was predicted from its stripe")
            want = inner - ref.island_width if island_front else inner + ref.island_width
            if local:
                outer_of = (lambda v: v["x0"]) if island_front else (lambda v: v["x1"])
                nearest = min(local, key=lambda v: abs(outer_of(v) - want))
                if abs(outer_of(nearest) - want) <= tol_width:
                    outer = outer_of(nearest)
                    sources["island_outer"] = "vertical-line"
                    used.append(nearest)
            if outer is None:
                outer = want
                sources["island_outer"] = "predicted-from-reference"
                warnings.append("an island outer edge was predicted from the reference width")

        island = (
            (int(round(outer)), int(round(inner))) if island_front
            else (int(round(inner)), int(round(outer)))
        )
        sources["stripe"] = "solid-bar"
        blocks.append(_Block(island=island, stripe=bar, sources=sources, vlines=used))
    return blocks


def _best_vertical_pair(
    vlines: Sequence[Dict[str, Any]],
    ref: RegionReference,
    predicted_inner: Optional[float],
    tol_width: float,
    tol_gap: float,
    island_front: bool,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """The vertical pair spanning one island.

    Both constraints matter on a multi-colour scan: an inner line of one
    colour and an outer line of the next are roughly an island apart, so
    matching the width alone would straddle two colours. Anchoring the inner
    edge on the bar rules that out.
    """
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
            inner = right["x1"] if island_front else left["x0"]
            if predicted_inner is not None:
                inner_error = abs(inner - predicted_inner)
                if inner_error > tol_gap:
                    continue
            else:
                inner_error = 0.0
            strength = min(left["coverage"], right["coverage"])
            score = width_error + inner_error - 200.0 * strength
            if score < best_score:
                best_score = score
                best = (left, right) if island_front else (right, left)
    return best


def _blocks_from_verticals_rtl(
    bars: Sequence[Tuple[int, int]],
    vlines: Sequence[Dict[str, Any]],
    ref: RegionReference,
    island_front: bool,
    warnings: List[str],
) -> List[_Block]:
    """Right-to-left: island verticals first, then attach or predict the stripe."""
    tol_width = max(150.0, ref.tolerance * ref.island_width)
    extra: List[_Block] = []
    free = list(vlines)
    claimed_bars = set()
    while len(free) >= 2:
        pair = _best_vertical_pair(free, ref, None, tol_width, 0.0, island_front)
        if pair is None:
            break
        outer_v, inner_v = pair
        outer = outer_v["x0"] if island_front else outer_v["x1"]
        inner = inner_v["x1"] if island_front else inner_v["x0"]
        island = (
            (int(outer), int(inner)) if island_front else (int(inner), int(outer))
        )
        if island_front:
            sx0 = island[1] + ref.island_stripe_gap
            predicted = (sx0, sx0 + ref.stripe_width - 1)
        else:
            sx1 = island[0] - ref.island_stripe_gap
            predicted = (sx1 - ref.stripe_width + 1, sx1)
        matched = None
        for index, bar in enumerate(bars):
            if index in claimed_bars:
                continue
            overlap = min(bar[1], predicted[1]) - max(bar[0], predicted[0])
            if overlap > 0.4 * ref.stripe_width:
                matched = index
                break
        if matched is not None:
            stripe = bars[matched]
            claimed_bars.add(matched)
            stripe_src = "solid-bar"
        else:
            stripe = predicted
            stripe_src = "predicted-from-island"
        extra.append(_Block(
            island=island,
            stripe=stripe,
            sources={
                "island_inner": "vertical-line",
                "island_outer": "vertical-line",
                "stripe": stripe_src,
                "direction": "rtl",
            },
            vlines=[outer_v, inner_v],
        ))
        lo, hi = min(island[0], stripe[0]), max(island[1], stripe[1])
        free = [v for v in free if not lo <= v["center"] <= hi]
    return extra


def _dedupe_overlapping_blocks(blocks: Sequence[_Block], ref: RegionReference) -> List[_Block]:
    """Keep one block when dual-band verticals or a stitch duplicate a colour."""
    kept: List[_Block] = []
    for block in blocks:
        lo = min(block.island[0], block.stripe[0])
        hi = max(block.island[1], block.stripe[1])
        replaced = False
        for index, other in enumerate(kept):
            olo = min(other.island[0], other.stripe[0])
            ohi = max(other.island[1], other.stripe[1])
            overlap = min(hi, ohi) - max(lo, olo)
            if overlap > 0.4 * ref.island_width:
                if _measured(block, "stripe") and not _measured(other, "stripe"):
                    kept[index] = block
                replaced = True
                break
        if not replaced:
            kept.append(block)
    return kept


def _merge_directional_blocks(
    ltr: Sequence[_Block],
    rtl: Sequence[_Block],
    ref: RegionReference,
    island_front: bool,
    warnings: List[str],
) -> List[_Block]:
    """Prefer a measured edge; among measured edges take the outer one."""
    used = set()
    merged: List[_Block] = []
    for left in ltr:
        match = None
        width_a = max(1, left.island[1] - left.island[0])
        for index, right in enumerate(rtl):
            if index in used:
                continue
            overlap = min(left.island[1], right.island[1]) - max(left.island[0], right.island[0])
            width_b = max(1, right.island[1] - right.island[0])
            if overlap > 0.5 * min(width_a, width_b):
                match = index
                break
        if match is None:
            left.sources.setdefault("direction", "ltr")
            merged.append(left)
            continue
        used.add(match)
        merged.append(_cover_merge_blocks(left, rtl[match], island_front))
    for index, right in enumerate(rtl):
        if index not in used:
            merged.append(right)
    _ = (ref, warnings)
    return merged


def _measured(block: _Block, key: str) -> bool:
    return not str(block.sources.get(key, "")).startswith("predicted")


def _cover_merge_blocks(left: _Block, right: _Block, island_front: bool) -> _Block:
    def pick_left(a: int, a_ok: bool, b: int, b_ok: bool) -> int:
        measured = [v for v, ok in ((a, a_ok), (b, b_ok)) if ok]
        return min(measured) if measured else a

    def pick_right(a: int, a_ok: bool, b: int, b_ok: bool) -> int:
        measured = [v for v, ok in ((a, a_ok), (b, b_ok)) if ok]
        return max(measured) if measured else a

    outer_ok_l = _measured(left, "island_outer")
    outer_ok_r = _measured(right, "island_outer")
    inner_ok_l = _measured(left, "island_inner")
    inner_ok_r = _measured(right, "island_inner")
    stripe_ok_l = _measured(left, "stripe")
    stripe_ok_r = _measured(right, "stripe")
    if island_front:
        i0 = pick_left(left.island[0], outer_ok_l, right.island[0], outer_ok_r)
        i1 = pick_right(left.island[1], inner_ok_l, right.island[1], inner_ok_r)
    else:
        i0 = pick_left(left.island[0], inner_ok_l, right.island[0], inner_ok_r)
        i1 = pick_right(left.island[1], outer_ok_l, right.island[1], outer_ok_r)
    s0 = pick_left(left.stripe[0], stripe_ok_l, right.stripe[0], stripe_ok_r)
    s1 = pick_right(left.stripe[1], stripe_ok_l, right.stripe[1], stripe_ok_r)
    sources = dict(left.sources)
    sources.update({k: v for k, v in right.sources.items() if not str(v).startswith("predicted")})
    sources["direction"] = "ltr+rtl"
    vlines = left.vlines or right.vlines
    return _Block(island=(i0, i1), stripe=(s0, s1), sources=sources, vlines=list(vlines))


def _blocks_from_orphan_verticals(
    vlines: Sequence[Dict[str, Any]],
    blocks: Sequence[_Block],
    ref: RegionReference,
    island_front: bool,
    warnings: List[str],
) -> List[_Block]:
    """Recover a colour whose solid bar never printed."""
    tol_width = max(150.0, ref.tolerance * ref.island_width)
    claimed = [(min(b.island[0], b.stripe[0]), max(b.island[1], b.stripe[1]))
               for b in blocks]
    free = [
        v for v in vlines
        if not any(lo <= v["center"] <= hi for lo, hi in claimed)
    ]
    extra: List[_Block] = []
    while len(free) >= 2:
        pair = _best_vertical_pair(free, ref, None, tol_width, 0.0, island_front)
        if pair is None:
            break
        outer_v, inner_v = pair
        outer = outer_v["x0"] if island_front else outer_v["x1"]
        inner = inner_v["x1"] if island_front else inner_v["x0"]
        island = (
            (int(outer), int(inner)) if island_front else (int(inner), int(outer))
        )
        if island_front:
            sx0 = island[1] + ref.island_stripe_gap
            stripe = (sx0, sx0 + ref.stripe_width - 1)
        else:
            sx1 = island[0] - ref.island_stripe_gap
            stripe = (sx1 - ref.stripe_width + 1, sx1)
        extra.append(_Block(
            island=island,
            stripe=stripe,
            sources={
                "island_inner": "vertical-line",
                "island_outer": "vertical-line",
                "stripe": "predicted-from-island",
            },
            vlines=[outer_v, inner_v],
        ))
        warnings.append("a colour has no solid bar; its stripe was predicted")
        lo, hi = min(island[0], stripe[0]), max(island[1], stripe[1])
        free = [v for v in free if not lo <= v["center"] <= hi]
    return extra


def _complete_lattice(
    blocks: List[_Block],
    ref: RegionReference,
    island_front: bool,
    warnings: List[str],
) -> List[_Block]:
    """Insert a colour that left no trace at all between two that did."""
    if len(blocks) < 2:
        return blocks
    starts = [b.island[0] for b in blocks]
    steps = np.diff(starts)
    pitch = float(np.median(steps))
    nominal = float(ref.color_pitch)
    if not 0.75 * nominal <= pitch <= 1.25 * nominal:
        pitch = nominal

    filled: List[_Block] = [blocks[0]]
    for previous, block in zip(blocks, blocks[1:]):
        missing = int(round((block.island[0] - previous.island[0]) / pitch)) - 1
        for step in range(1, missing + 1):
            shift = int(round(step * pitch))
            island = (previous.island[0] + shift, previous.island[1] + shift)
            stripe = (previous.stripe[0] + shift, previous.stripe[1] + shift)
            filled.append(_Block(
                island=island,
                stripe=stripe,
                y=previous.y,
                sources={
                    "island_inner": "predicted-from-lattice",
                    "island_outer": "predicted-from-lattice",
                    "stripe": "predicted-from-lattice",
                    "y": "lattice",
                },
            ))
            warnings.append("a colour printed nothing; its boxes came from the layout pitch")
        filled.append(block)
    return filled


def _name_blocks(
    blocks: Sequence[_Block],
    colors: Sequence[str],
    image_path: str,
    warnings: List[str],
) -> List[str]:
    """Left-to-right colour order, or the filename's colour for a single block."""
    if len(blocks) == 1:
        return [color_token_from_path(image_path) or "Region"]
    if len(blocks) != len(colors):
        warnings.append(
            f"found {len(blocks)} colour block(s) but the config lists {len(colors)}"
        )
    names: List[str] = []
    for index in range(len(blocks)):
        names.append(colors[index] if index < len(colors) else f"Color{index + 1}")
    return names


# ----------------------------------------------------------------------
# Vertical extent
# ----------------------------------------------------------------------

def _solve_y_extent(
    ink: np.ndarray,
    scale: int,
    block: _Block,
    ref: RegionReference,
    height: int,
    warnings: List[str],
) -> Tuple[int, int]:
    """One y range per colour: its island and stripe are printed together."""
    close_gap = max(1, int(round(ref.height / 80.0 / scale)))
    island_y = _row_cluster(ink, block.island, scale, close_gap, rel=0.25)
    stripe_y = _row_cluster(ink, block.stripe, scale, close_gap, rel=0.50)

    tol = ref.tolerance * ref.height
    island_ok = island_y is not None and abs((island_y[1] - island_y[0] + 1) - ref.height) <= tol
    stripe_ok = stripe_y is not None and abs((stripe_y[1] - stripe_y[0] + 1) - ref.height) <= tol

    if island_ok and stripe_ok:
        outer_top = min(island_y[0], stripe_y[0])
        outer_bot = max(island_y[1], stripe_y[1])
        grown = outer_bot - outer_top + 1
        if abs(grown - ref.height) <= tol:
            block.sources["y"] = "island+stripe"
            return outer_top, outer_bot
        warnings.append("an island y would inflate the box; using the stripe")
        block.sources["y"] = "stripe"
        return stripe_y
    if stripe_ok:
        block.sources["y"] = "stripe"
        return stripe_y
    if island_ok:
        block.sources["y"] = "island"
        return island_y

    candidates = [c for c in (island_y, stripe_y) if c is not None]
    if not candidates:
        warnings.append("a colour has no y extent; using the full image height")
        block.sources["y"] = "image"
        return 0, height - 1
    best = min(candidates, key=lambda c: abs((c[1] - c[0] + 1) - ref.height))
    warnings.append(
        f"a y extent of {best[1] - best[0] + 1}px differs from the reference {ref.height}px"
    )
    block.sources["y"] = "island" if best is island_y else "stripe"
    return best


def _borrow_y(
    blocks: Sequence[_Block],
    index: int,
    ghosts: Sequence[int],
    ref: RegionReference,
    height: int,
) -> Tuple[int, int]:
    """The y range of the nearest colours that did print."""
    measured = [b.y for i, b in enumerate(blocks) if i not in ghosts and b.y[1] > b.y[0]]
    if not measured:
        return 0, min(height - 1, ref.height - 1)
    tops = [y[0] for y in measured]
    bottoms = [y[1] for y in measured]
    return int(np.median(tops)), int(np.median(bottoms))


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


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

def _block_measurements(block: _Block, island_front: bool) -> Dict[str, int]:
    return {
        "island_width": block.island[1] - block.island[0] + 1,
        "stripe_width": block.stripe[1] - block.stripe[0] + 1,
        "island_stripe_gap": (
            block.stripe[0] - block.island[1] - 1 if island_front
            else block.island[0] - block.stripe[1] - 1
        ),
        "height": block.y[1] - block.y[0] + 1,
    }


def _check_layout(
    blocks: Sequence[_Block],
    names: Sequence[str],
    ref: RegionReference,
    island_front: bool,
    warnings: List[str],
) -> None:
    expected = {
        "island_width": ref.island_width,
        "stripe_width": ref.stripe_width,
        "island_stripe_gap": ref.island_stripe_gap,
        "height": ref.height,
    }
    for name, block in zip(names, blocks):
        measured = _block_measurements(block, island_front)
        for key, want in expected.items():
            got = measured[key]
            if abs(got - want) > max(60, ref.tolerance * want):
                warnings.append(f"{name} {key} is {got}px, reference {want}px")
    for (name, block), (next_name, next_block) in zip(
        zip(names, blocks), zip(names[1:], blocks[1:])
    ):
        if island_front:
            gap = next_block.island[0] - block.stripe[1] - 1
        else:
            gap = next_block.stripe[0] - block.island[1] - 1
        if abs(gap - ref.color_gap) > max(80, ref.tolerance * ref.color_pitch):
            warnings.append(
                f"gap between {name} and {next_name} is {gap}px, reference {ref.color_gap}px"
            )
        drop = abs(next_block.y[0] - block.y[0])
        if drop > ref.color_y_tolerance * 1.5:
            warnings.append(
                f"{next_name} sits {drop}px below {name}, more than the expected "
                f"+/-{ref.color_y_tolerance}px"
            )


# ----------------------------------------------------------------------
# Full-resolution refinement
# ----------------------------------------------------------------------

def _neighbour_bounds(
    blocks: Sequence[_Block], index: int, ref: RegionReference, width: int
) -> Tuple[int, int]:
    """The x territory one colour may be refined within.

    Only ``color_gap`` (~250 px) separates two colours, so an unbounded
    refinement window around a stripe's right edge could reach the next
    colour's first vertical line and snap onto it. Each block is fenced in
    at the midpoint of the gap to its neighbours.
    """
    left, right = 0, width - 1
    own = (min(blocks[index].island[0], blocks[index].stripe[0]),
           max(blocks[index].island[1], blocks[index].stripe[1]))
    if index > 0:
        previous = max(blocks[index - 1].island[1], blocks[index - 1].stripe[1])
        left = max(left, (previous + own[0]) // 2)
    if index + 1 < len(blocks):
        following = min(blocks[index + 1].island[0], blocks[index + 1].stripe[0])
        right = min(right, (own[1] + following) // 2)
    return left, right


def _refine_block(
    image: np.ndarray,
    bg: float,
    contrast: float,
    block: _Block,
    ref: RegionReference,
    bounds: Tuple[int, int],
    island_front: bool = True,
    geometry: Any = None,
    warnings: Optional[List[str]] = None,
) -> None:
    """Snap boundaries from edge stations, then cover with layout constraints."""
    warnings = warnings if warnings is not None else []
    height, width = image.shape[:2]
    x_window = max(24, ref.island_stripe_gap // 2, 160, int(0.04 * ref.island_width))
    y_window = max(24, int(ref.height / 80.0))
    y0, y1 = block.y
    band = max(32, int(ref.height / 250.0))

    ys = np.unique(np.clip(
        np.round(np.linspace(y0, y1, 12)).astype(int), 0, height - 1,
    ))
    stitch_ys: List[int] = []
    n_heads = int(getattr(geometry, "num_heads", 0) or 0) if geometry is not None else 0
    if n_heads > 1:
        for k in range(1, n_heads):
            stitch_ys.append(int(np.clip(y0 + k * (y1 - y0) / n_heads, y0, y1)))
        ys = np.unique(np.concatenate([ys, np.asarray(stitch_ys, dtype=int)]))

    def x_stations(guess: int, edge: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for y in ys:
            local = _local_edges_at_y(
                image, bg, contrast, int(y), band, block, ref, bounds, island_front,
            )
            island_left, island_right, stripe_left, stripe_right = local
            if island_front:
                mapping = {
                    ("first", True): island_left,
                    ("last", True): island_right,
                    ("first", False): stripe_left,
                    ("last", False): stripe_right,
                }
                on_island = guess <= (block.island[1] + block.stripe[0]) // 2
            else:
                mapping = {
                    ("first", True): stripe_left,
                    ("last", True): stripe_right,
                    ("first", False): island_left,
                    ("last", False): island_right,
                }
                on_island = guess >= (block.stripe[1] + block.island[0]) // 2
            x = mapping[(edge, on_island)]
            if x is not None and abs(int(x) - int(guess)) > max(200, x_window):
                x = None
            out.append({"y": int(y), "x": None if x is None else int(x)})
        return out

    left_i = x_stations(block.island[0], "first")
    right_i = x_stations(block.island[1], "last")
    left_s = x_stations(block.stripe[0], "first")
    right_s = x_stations(block.stripe[1], "last")

    i0 = _cover_x_edge(left_i, "first", block.island[0])
    i1 = _cover_x_edge(right_i, "last", block.island[1])
    s0 = _cover_x_edge(left_s, "first", block.stripe[0])
    s1 = _cover_x_edge(right_s, "last", block.stripe[1])
    counts = {
        "i0": sum(1 for s in left_i if s.get("x") is not None),
        "i1": sum(1 for s in right_i if s.get("x") is not None),
        "s0": sum(1 for s in left_s if s.get("x") is not None),
        "s1": sum(1 for s in right_s if s.get("x") is not None),
    }
    i0, i1, s0, s1 = _constraint_solve_x(
        (i0, i1), (s0, s1), block.sources, ref, island_front, warnings, counts,
    )

    vband = max(4, ref.island_width // 250)
    reliable_x = [
        (s0, s1 + 1),
        (i0, i0 + vband),
        (max(i0, i1 - vband), i1 + 1),
        (s0, min(s1 + 1, s0 + vband)),
        (max(s0, s1 - vband), s1 + 1),
    ]
    island_x = (i0, i1 + 1)

    def snap_y(guess: int, xa: int, xb: int, edge: str, nearest: bool = True) -> Optional[int]:
        return _snap_y(
            image, bg, contrast, guess, xa, xb, y_window, edge,
            nearest=nearest, min_run=1 if not nearest else 2,
        )

    reliable_tops = [snap_y(y0, xa, xb, "first") for xa, xb in reliable_x]
    reliable_bots = [snap_y(y1, xa, xb, "last") for xa, xb in reliable_x]
    island_top = snap_y(y0, island_x[0], island_x[1], "first")
    island_bot = snap_y(y1, island_x[0], island_x[1], "last")

    x_points = np.unique(np.concatenate([
        np.round(np.linspace(i0, i1, 8)).astype(int),
        np.round(np.linspace(s0, s1, 4)).astype(int),
        np.array([i0, i1, s0, s1], dtype=int),
    ]))
    xband = max(8, ref.island_width // 80)

    def y_across(guess: int, edge: str) -> List[Optional[int]]:
        votes: List[Optional[int]] = []
        for x in x_points:
            xa = int(x) - xband
            xb = int(x) + xband
            votes.append(snap_y(guess, xa, xb, edge))
        return votes

    reliable_tops = list(reliable_tops) + y_across(y0, "first")
    reliable_bots = list(reliable_bots) + y_across(y1, "last")
    corner_xs = []
    for stations, reverse in ((left_i, False), (right_i, False), (left_i, True), (right_i, True)):
        seq = reversed(list(stations)) if reverse else stations
        for item in seq:
            if item.get("x") is not None:
                corner_xs.append(int(item["x"]))
                break
    for x in (i0, i1, s0, s1, *corner_xs):
        reliable_tops.append(snap_y(y0, int(x) - xband, int(x) + xband, "first", nearest=False))
        reliable_bots.append(snap_y(y1, int(x) - xband, int(x) + xband, "last", nearest=False))
    top = _cover_y_edge(reliable_tops, island_top, "first", y0, ref)
    bot = _cover_y_edge(reliable_bots, island_bot, "last", y1, ref)

    want = ref.height - 1
    if bot - top + 1 < ref.height * (1.0 - ref.tolerance):
        stripe_top = reliable_tops[0]
        stripe_bot = reliable_bots[0]
        if stripe_top is not None and abs((stripe_top + want) - bot) >= abs(top + want - bot):
            bot = min(height - 1, stripe_top + want)
            top = stripe_top
            block.sources["y"] = "predicted-from-height"
            warnings.append("y was short of the reference height; extended from the stripe top")
        elif stripe_bot is not None:
            top = max(0, stripe_bot - want)
            bot = stripe_bot
            block.sources["y"] = "predicted-from-height"
            warnings.append("y was short of the reference height; extended from the stripe bottom")
        else:
            bot = min(height - 1, top + want)
            block.sources["y"] = "predicted-from-height"

    if i1 > i0:
        block.island = (i0, i1)
    if s1 > s0:
        block.stripe = (s0, s1)
    if bot > top:
        block.y = (top, bot)

    def x_at(stations: Sequence[Dict[str, Any]], y: int, fallback: int) -> int:
        valid = [(int(s["y"]), int(s["x"])) for s in stations if s.get("x") is not None]
        if not valid:
            return fallback
        return min(valid, key=lambda item: abs(item[0] - y))[1]

    block.corners = {
        "island": {
            "tl": [x_at(left_i, top, i0), top],
            "tr": [x_at(right_i, top, i1), top],
            "bl": [x_at(left_i, bot, i0), bot],
            "br": [x_at(right_i, bot, i1), bot],
        },
        "stripe": {
            "tl": [x_at(left_s, top, s0), top],
            "tr": [x_at(right_s, top, s1), top],
            "bl": [x_at(left_s, bot, s0), bot],
            "br": [x_at(right_s, bot, s1), bot],
        },
    }
    block.island = _span_from_stations(left_i, right_i, block.island)
    block.stripe = _span_from_stations(left_s, right_s, block.stripe)
    block.edge_stations = {
        "island_left": left_i,
        "island_right": right_i,
        "stripe_left": left_s,
        "stripe_right": right_s,
        "stitch_y": stitch_ys,
    }
    _flag_edge_defects(block, warnings)
    block.sources["refinement"] = (
        f"station snap + constraint cover (+/-{x_window}px in x, +/-{y_window}px in y)"
    )


def _local_edges_at_y(
    image: np.ndarray,
    bg: float,
    contrast: float,
    y: int,
    band: int,
    block: _Block,
    ref: RegionReference,
    bounds: Tuple[int, int],
    island_front: bool,
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """Island and stripe x-edges in a narrow y-band, anchored on the local bar.

    A mid-height stitch shifts the whole block, so a window around the coarse
    guess can sit on the wrong vertical. Measuring the solid bar in this band
    and walking back by the nominal gap recovers the true inner/outer pair.
    """
    height, width = image.shape[:2]
    ya = int(np.clip(y - band, 0, height - 2))
    yb = int(np.clip(y + band, ya + 1, height))
    xa = int(np.clip(min(bounds[0], block.island[0], block.stripe[0]) - 160, 0, width - 2))
    xb = int(np.clip(max(bounds[1], block.island[1], block.stripe[1]) + 160, xa + 1, width))
    strip = _min_channel(image[ya:yb:2, xa:xb])
    dens = _density(strip, bg, contrast).mean(axis=0)
    if dens.size == 0 or float(dens.max()) <= 0.02:
        return None, None, None, None

    peak = float(np.percentile(dens, 99.5))
    bar_runs = _bool_runs(dens >= max(0.30, 0.55 * peak), min_len=1, merge_gap=2)
    stripe_w = ref.stripe_width
    bars = []
    for start, end in bar_runs:
        run_w = end - start + 1
        if 0.45 * stripe_w <= run_w <= 2.6 * stripe_w and float(dens[start:end + 1].mean()) >= 0.5 * peak:
            bars.append((xa + start, xa + end))
    stripe_guess = (block.stripe[0] + block.stripe[1]) // 2
    bar = None
    if bars:
        bar = min(bars, key=lambda b: abs((b[0] + b[1]) // 2 - stripe_guess))
    ink_runs = _bool_runs(dens >= max(0.08, 0.2 * peak), min_len=2, merge_gap=3)

    def last_before(limit: int) -> Optional[int]:
        candidates = [xa + r[1] for r in ink_runs if xa + r[1] < limit]
        return max(candidates) if candidates else None

    def first_after(limit: int) -> Optional[int]:
        candidates = [xa + r[0] for r in ink_runs if xa + r[0] > limit]
        return min(candidates) if candidates else None

    s0 = s1 = i0 = i1 = None
    if bar is not None:
        s0, s1 = int(bar[0]), int(bar[1])
        if island_front:
            i1 = last_before(s0 - max(8, ref.island_stripe_gap // 4))
            if i1 is None:
                i1 = s0 - ref.island_stripe_gap
            i0 = first_after(i1 - ref.island_width - max(40, int(0.08 * ref.island_width)))
            if i0 is None or abs((i1 - i0 + 1) - ref.island_width) > max(80, 0.2 * ref.island_width):
                # Prefer a run near the outer vertical rather than the first speck.
                target = i1 - ref.island_width + 1
                outer = min(
                    (xa + r[0] for r in ink_runs),
                    key=lambda x: abs(x - target),
                    default=target,
                )
                i0 = int(outer)
        else:
            i0 = first_after(s1 + max(8, ref.island_stripe_gap // 4))
            if i0 is None:
                i0 = s1 + ref.island_stripe_gap
            target = i0 + ref.island_width - 1
            i1 = min(
                (xa + r[1] for r in ink_runs),
                key=lambda x: abs(x - target),
                default=target,
            )
    return (
        None if i0 is None else int(i0),
        None if i1 is None else int(i1),
        None if s0 is None else int(s0),
        None if s1 is None else int(s1),
    )


def _drop_sparse_outliers(values: Sequence[int], min_count: int = 2, radius: int = 25) -> List[int]:
    kept = [
        x for x in values
        if sum(1 for y in values if abs(y - x) <= radius) >= min_count
    ]
    return kept or list(values)


def _cover_x_edge(stations: Sequence[Dict[str, Any]], edge: str, guess: int) -> int:
    xs = [int(s["x"]) for s in stations if s.get("x") is not None]
    if not xs:
        return int(guess)
    clustered = _drop_sparse_outliers(xs)
    return int(min(clustered) if edge == "first" else max(clustered))


def _cover_y_edge(
    reliable: Sequence[Optional[int]],
    island_vote: Optional[int],
    edge: str,
    guess: int,
    ref: RegionReference,
) -> int:
    values = [int(v) for v in reliable if v is not None]
    spread = max(8, ref.height // 200)
    if not values:
        return int(island_vote if island_vote is not None else guess)
    clustered = _drop_sparse_outliers(values, radius=spread)
    middle = float(np.median(clustered))
    outer = min(values) if edge == "first" else max(values)
    if abs(outer - middle) > spread:
        near_outer = [v for v in values if abs(v - outer) <= spread]
        if len(near_outer) < 2:
            outer = min(clustered) if edge == "first" else max(clustered)
    _ = island_vote
    return int(outer)


def _constraint_solve_x(
    island: Tuple[int, int],
    stripe: Tuple[int, int],
    sources: Dict[str, str],
    ref: RegionReference,
    island_front: bool,
    warnings: List[str],
    counts: Optional[Dict[str, int]] = None,
) -> Tuple[int, int, int, int]:
    """Fill an edge that failed to measure; do not override a well-sampled one.

    Real islands are a bit narrower than the nominal 5100 px, so applying the
    reference width to a measured outer vertical would expand the box.
    """
    i0, i1 = island
    s0, s1 = stripe
    iw, sw, gap = ref.island_width, ref.stripe_width, ref.island_stripe_gap
    counts = counts or {}

    def weak(key: str) -> bool:
        return int(counts.get(key, 0)) < 3

    def fill_left(measured: int, predicted: int, key: str, name: str) -> int:
        if not weak(key):
            return int(measured)
        inward = measured - predicted
        if inward > 15:
            warnings.append(f"{name} had too few stations; using layout prediction")
            sources[name] = "predicted-cover"
            return int(predicted)
        return int(measured)

    def fill_right(measured: int, predicted: int, key: str, name: str) -> int:
        if not weak(key):
            return int(measured)
        inward = predicted - measured
        if inward > 15:
            warnings.append(f"{name} had too few stations; using layout prediction")
            sources[name] = "predicted-cover"
            return int(predicted)
        return int(measured)

    if island_front:
        ltr_i1 = s0 - gap
        ltr_i0 = ltr_i1 - iw + 1
        rtl_s0 = i1 + gap
        rtl_s1 = rtl_s0 + sw - 1
        i0 = fill_left(i0, ltr_i0, "i0", "island_outer")
        i1 = fill_right(i1, ltr_i1, "i1", "island_inner")
        s0 = fill_left(s0, rtl_s0, "s0", "stripe_left")
        s1 = fill_right(s1, rtl_s1, "s1", "stripe_right")
    else:
        ltr_i0 = s1 + gap
        ltr_i1 = ltr_i0 + iw - 1
        rtl_s1 = i0 - gap
        rtl_s0 = rtl_s1 - sw + 1
        i0 = fill_left(i0, ltr_i0, "i0", "island_inner")
        i1 = fill_right(i1, ltr_i1, "i1", "island_outer")
        s0 = fill_left(s0, rtl_s0, "s0", "stripe_left")
        s1 = fill_right(s1, rtl_s1, "s1", "stripe_right")
    return i0, i1, s0, s1


def _span_from_stations(
    left: Sequence[Dict[str, Any]],
    right: Sequence[Dict[str, Any]],
    fallback: Tuple[int, int],
) -> Tuple[int, int]:
    xs_l = _drop_sparse_outliers([int(s["x"]) for s in left if s.get("x") is not None])
    xs_r = _drop_sparse_outliers([int(s["x"]) for s in right if s.get("x") is not None])
    x0 = min(xs_l) if xs_l else fallback[0]
    x1 = max(xs_r) if xs_r else fallback[1]
    return min(x0, fallback[0]), max(x1, fallback[1])


def _flag_edge_defects(block: _Block, warnings: List[str]) -> None:
    for name, stations in (block.edge_stations or {}).items():
        if name == "stitch_y" or not isinstance(stations, list):
            continue
        xs = [int(s["x"]) for s in stations if isinstance(s, dict) and s.get("x") is not None]
        missing = sum(1 for s in stations if isinstance(s, dict) and s.get("x") is None)
        if missing:
            block.sources[f"{name}_missing"] = str(missing)
            warnings.append(f"{missing} station(s) on {name} had no ink")
        if len(xs) >= 2:
            spread = max(xs) - min(xs)
            if spread > 40:
                block.sources["slant_px"] = str(max(int(block.sources.get("slant_px") or 0), spread))
                block.sources["stitch_offset_px"] = block.sources["slant_px"]
                warnings.append(f"{name} varies by {spread}px (slant or stitch)")


def _consensus(
    candidates: Sequence[Optional[int]],
    spread: int,
    take,
    fallback: int,
) -> int:
    """Agree on one edge from several strips, ignoring the odd one out.

    An ink drip or a speck below a region will drag a single strip hundreds
    of pixels past the print, so anything far from the majority is dropped;
    the survivors are then combined with ``min``/``max``, which keeps a
    genuine difference between what the island and the stripe printed.
    """
    values = [c for c in candidates if c is not None]
    if not values:
        return fallback
    middle = float(np.median(values))
    agreed = [v for v in values if abs(v - middle) <= spread]
    return int(take(agreed or values))


def _snap_x(
    image: np.ndarray,
    bg: float,
    contrast: float,
    guess: int,
    y0: int,
    y1: int,
    window: int,
    edge: str,
    bounds: Tuple[int, int],
) -> Optional[int]:
    width = image.shape[1]
    low = max(0, bounds[0])
    high = min(width - 1, bounds[1])
    if high <= low:
        low, high = 0, width - 1
    a = int(np.clip(guess - window, low, high))
    b = int(np.clip(guess + window + 1, a + 1, high + 1))
    strip = _min_channel(image[y0:y1:4, a:b])
    density = _density(strip, bg, contrast).mean(axis=0)
    index = _edge_index(density, edge, min_run=2, nearest=int(guess - a))
    return None if index is None else a + index


def _snap_y(
    image: np.ndarray,
    bg: float,
    contrast: float,
    guess: int,
    x0: int,
    x1: int,
    window: int,
    edge: str,
    nearest: bool = True,
    min_run: int = 2,
) -> Optional[int]:
    height, width = image.shape[:2]
    x0 = int(np.clip(x0, 0, width - 1))
    x1 = int(np.clip(x1, x0 + 1, width))
    a = int(np.clip(guess - window, 0, height - 1))
    b = int(np.clip(guess + window + 1, a + 1, height))
    strip = _min_channel(image[a:b, x0:x1:4])
    density = _density(strip, bg, contrast).mean(axis=1)
    index = _edge_index(
        density, edge, min_run=min_run,
        nearest=int(guess - a) if nearest else None,
    )
    return None if index is None else a + index


def _edge_index(
    density: np.ndarray, edge: str, min_run: int, nearest: Optional[int] = None,
) -> Optional[int]:
    """First/last index of real structure inside a refinement window.

    The threshold sits 40% of the way from the window's paper level to its
    strongest ink. When ``nearest`` is set, the run closest to the coarse
    guess is used so a wide window that also contains a neighbour (the
    stripe, an overspray blob) does not steal the edge.
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
    if nearest is None:
        return runs[0][0] if edge == "first" else runs[-1][1]
    best = min(runs, key=lambda r: abs(0.5 * (r[0] + r[1]) - nearest))
    return best[0] if edge == "first" else best[1]


# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------

def overlay_canvas(scan: Any, max_edge: int = 4000) -> Tuple[np.ndarray, float]:
    """Downsampled BGR view of a (possibly huge) scan, plus the scale used."""
    image = _pixels(scan)
    swap_rb = _is_rgb(scan)
    height, width = image.shape[:2]
    scale = 1.0
    if max(height, width) > max_edge:
        scale = max_edge / float(max(height, width))
        out = _downsample_bgr(image, scale)
    else:
        out = _ensure_bgr(np.asarray(image)).copy()
    if swap_rb:
        out = out[:, :, ::-1].copy()
    return out, scale


def draw_region_overlay(
    scan: Any,
    specs: Sequence[RegionSpec],
    debug: Optional[Dict[str, Any]] = None,
    max_edge: int = 4000,
) -> np.ndarray:
    """Whole-scan overlay: island green, stripe blue, verticals magenta."""
    out, scale = overlay_canvas(scan, max_edge)
    for color in (debug or {}).get("colors") or []:
        for vline in color.get("vertical_lines") or []:
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
    scan: Any,
    specs: Sequence[RegionSpec],
    crop: int = 900,
) -> Optional[np.ndarray]:
    """Full-resolution crops of each region's four corners."""
    image = _pixels(scan)
    swap_rb = _is_rgb(scan)
    rows: List[np.ndarray] = []
    for spec in specs:
        x0, y0, x1, y1 = spec.bounding_box_pixels.as_int_xyxy()
        color = ISLAND_BGR if spec.type == "island" else STRIPE_BGR
        corners = spec.corners or {
            "tl": [x0, y0], "tr": [x1, y0], "bl": [x0, y1], "br": [x1, y1],
        }
        panels = [
            _corner_panel(
                image, int(corners["tl"][0]), int(corners["tl"][1]),
                crop, color, f"{spec.name} top-left", swap_rb,
            ),
            _corner_panel(
                image, int(corners["tr"][0]), int(corners["tr"][1]),
                crop, color, f"{spec.name} top-right", swap_rb,
            ),
            _corner_panel(
                image, int(corners["bl"][0]), int(corners["bl"][1]),
                crop, color, f"{spec.name} bottom-left", swap_rb,
            ),
            _corner_panel(
                image, int(corners["br"][0]), int(corners["br"][1]),
                crop, color, f"{spec.name} bottom-right", swap_rb,
            ),
        ]
        rows.append(np.vstack([
            np.hstack(panels[:2]),
            np.hstack(panels[2:]),
        ]))
    if not rows:
        return None
    return np.vstack(rows)


def _corner_panel(
    image: np.ndarray,
    x: int,
    y: int,
    crop: int,
    color: Tuple[int, int, int],
    label: str,
    swap_rb: bool = False,
) -> np.ndarray:
    height, width = image.shape[:2]
    half = crop // 2
    x0 = int(np.clip(x - half, 0, max(0, width - 1)))
    y0 = int(np.clip(y - half, 0, max(0, height - 1)))
    x1 = int(np.clip(x0 + crop, 1, width))
    y1 = int(np.clip(y0 + crop, 1, height))
    panel = np.full((crop, crop, 3), 245, dtype=np.uint8)
    piece = _ensure_bgr(np.asarray(image[y0:y1, x0:x1]))
    if swap_rb:
        piece = piece[:, :, ::-1]
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

def _spec(
    name: str,
    kind: str,
    xyxy: Tuple[int, int, int, int],
    corners: Optional[Dict[str, Any]] = None,
) -> RegionSpec:
    x0, y0, x1, y1 = xyxy
    return RegionSpec(
        name=name,
        type=kind,
        bounding_box_pixels=BoundingBox(
            top_x=float(x0), top_y=float(y0),
            bottom_x=float(x1), bottom_y=float(y1),
        ),
        corners=corners,
    )


def _coarse_scale(ref: RegionReference, width: int, height: int) -> int:
    """Driven by feature size, not image size: ~130 coarse px per stripe."""
    scale = max(1, int(round(ref.stripe_width / 130.0)))
    while scale > 1 and (width // scale < 200 or height // scale < 200):
        scale //= 2
    return scale


def _coarse_ink_map(image: np.ndarray, scale: int) -> np.ndarray:
    """Downsample to the darkest channel, a band of rows at a time.

    Row bands keep peak memory near ``_CHUNK_BYTES`` whatever the sheet
    size, which is what lets a 6.7 GB memory-mapped scan be profiled at all.
    Each band is a whole multiple of ``scale`` rows so that coarse row *r*
    always means full-resolution row *r * scale*.
    """
    height, width = image.shape[:2]
    out_width = max(1, width // scale)
    rows = max(scale, (_CHUNK_BYTES // max(1, width * 3) // scale) * scale)
    if scale <= 1 and rows >= height:
        return _min_channel(np.asarray(image))

    bands: List[np.ndarray] = []
    for top in range(0, height, rows):
        bottom = min(height, top + rows)
        if bottom - top < scale and bands:
            break  # a stub shorter than one coarse row adds nothing
        band = _min_channel(np.asarray(image[top:bottom]))
        out_height = max(1, (bottom - top) // scale)
        bands.append(cv2.resize(
            band, (out_width, out_height), interpolation=cv2.INTER_AREA,
        ))
    return np.vstack(bands)


def _downsample_bgr(image: np.ndarray, scale: float) -> np.ndarray:
    """Chunked BGR downsample for the whole-sheet overlay."""
    height, width = image.shape[:2]
    out_width = max(1, int(width * scale))
    rows = max(1, _CHUNK_BYTES // max(1, width * 3))
    bands: List[np.ndarray] = []
    written = 0
    for top in range(0, height, rows):
        bottom = min(height, top + rows)
        out_height = int(round(bottom * scale)) - written
        if out_height <= 0:
            continue
        written += out_height
        bands.append(cv2.resize(
            _ensure_bgr(np.asarray(image[top:bottom])),
            (out_width, out_height), interpolation=cv2.INTER_AREA,
        ))
    return np.vstack(bands)


def _pixels(scan: Any) -> np.ndarray:
    """Accept either a ``Scan`` or a bare array."""
    return scan if isinstance(scan, np.ndarray) else scan.data


def _is_rgb(scan: Any) -> bool:
    return not isinstance(scan, np.ndarray) and bool(getattr(scan, "rgb", False))


def _min_channel(image: np.ndarray) -> np.ndarray:
    """Ink strength for any ink colour.

    Yellow is nearly invisible in luminance but its blue channel is as dark
    as any other ink, so the darkest channel is what makes cyan, magenta,
    yellow and black comparable on one scan.
    """
    if image.ndim == 2:
        return image
    return image.min(axis=2)


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _ink_levels(single_channel: np.ndarray) -> Tuple[float, float]:
    """(paper level, paper-to-ink range) measured on the scan itself."""
    values = single_channel.astype(np.float32)
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
