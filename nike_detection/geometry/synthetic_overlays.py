"""Apply synthetic corner defects onto a scan without writing a new TIFF.

Patches are stored as small replacement tiles and spliced into any slice
the region detector reads. Ground-truth print corners do not change when
ink is erased or overspray is added.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from nike_detection.config.schema import RegionReference
from nike_detection.geometry.region_metrics import corners_from_xyxy
from nike_detection.io.image_loader import Scan

WHITEPAPER_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..",
        "data", "20260617_P1_WhitePaper_KCM-updated-folder",
    )
)
SYNTHETIC_DIR = os.path.join(WHITEPAPER_DIR, "synthetic")

CORNERS = ("tl", "tr", "bl", "br")
LINE_SPACING_FULL = 100

GEO_REF = RegionReference(
    island_width=640,
    island_stripe_gap=72,
    stripe_width=128,
    height=4096,
    color_gap=32,
    color_y_tolerance=24,
    tolerance=0.12,
)
GEO_PAD = 96
GEO_LINE_SPACING = 16
GEO_VLINE = 4


def load_gt(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def gt_path_for(stem: str, synthetic_dir: str = SYNTHETIC_DIR) -> str:
    return os.path.join(synthetic_dir, "ground_truth", f"{stem}.json")


def resolve_source_tiff(gt: Dict[str, Any], whitepaper_dir: str = WHITEPAPER_DIR) -> str:
    return os.path.normpath(os.path.join(whitepaper_dir, gt["source_tiff"]))


def paper_color(gt: Dict[str, Any], rgb: bool) -> Tuple[int, int, int]:
    rgb_c = [int(v) for v in gt.get("paper_rgb") or (255, 255, 255)]
    return (rgb_c[0], rgb_c[1], rgb_c[2]) if rgb else (rgb_c[2], rgb_c[1], rgb_c[0])


def ink_color(gt: Dict[str, Any], rgb: bool) -> Tuple[int, int, int]:
    rgb_c = [int(v) for v in gt.get("ink_rgb") or (20, 20, 20)]
    return (rgb_c[0], rgb_c[1], rgb_c[2]) if rgb else (rgb_c[2], rgb_c[1], rgb_c[0])


def region_box(gt: Dict[str, Any], kind: str, color_index: int = 0) -> Tuple[int, int, int, int]:
    block = gt["colors"][color_index][kind]
    x0, x1 = block["x"]
    y0, y1 = block["y"]
    return int(x0), int(y0), int(x1), int(y1)


def region_corners(gt: Dict[str, Any], kind: str, color_index: int = 0) -> Dict[str, List[int]]:
    block = gt["colors"][color_index][kind]
    return {k: [int(p[0]), int(p[1])] for k, p in block["corners"].items()}


def open_memmap_scan(path: str) -> Scan:
    """Memory-map an uncompressed TIFF regardless of file size."""
    import tifffile

    data = tifffile.memmap(path, mode="r")
    rgb = data.ndim == 3
    return Scan(data=data, rgb=rgb, memmap=True, path=path)


# ----------------------------------------------------------------------
# Overlay array: splice patch tiles into detector slices
# ----------------------------------------------------------------------

class OverlayArray:
    """Read-only view of a scan with a handful of replacement tiles."""

    def __init__(self, base: np.ndarray, patches: Sequence["_Patch"]):
        self._base = base
        self._patches = list(patches)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._base.shape)

    @property
    def ndim(self) -> int:
        return int(self._base.ndim)

    @property
    def dtype(self):
        return self._base.dtype

    def __array__(self, dtype=None):
        raise TypeError("OverlayArray cannot be materialised whole; slice it")

    def __getitem__(self, key):
        result = np.asarray(self._base[key])
        y_sl, x_sl = _yx_slices(key, self.shape)
        if y_sl is None:
            return result
        y0, y1, ystep = y_sl
        x0, x1, xstep = x_sl
        copied = False
        for patch in self._patches:
            iy0 = max(y0, patch.y0)
            iy1 = min(y1, patch.y1)
            ix0 = max(x0, patch.x0)
            ix1 = min(x1, patch.x1)
            if iy0 >= iy1 or ix0 >= ix1:
                continue
            ky = 0 if iy0 <= y0 else (iy0 - y0 + ystep - 1) // ystep
            kx = 0 if ix0 <= x0 else (ix0 - x0 + xstep - 1) // xstep
            sy = y0 + ky * ystep
            sx = x0 + kx * xstep
            if sy >= iy1 or sx >= ix1:
                continue
            if not copied:
                result = np.array(result, copy=True)
                copied = True
            src = patch.data[
                sy - patch.y0: iy1 - patch.y0: ystep,
                sx - patch.x0: ix1 - patch.x0: xstep,
            ]
            result[ky:ky + src.shape[0], kx:kx + src.shape[1]] = src
        return result


def _yx_slices(key, shape) -> Tuple[Optional[Tuple[int, int, int]], Optional[Tuple[int, int, int]]]:
    if not isinstance(key, tuple):
        key = (key,)
    def _span(item, length):
        if isinstance(item, slice):
            start, stop, step = item.indices(length)
            return start, stop, step
        idx = int(item)
        if idx < 0:
            idx += length
        return idx, idx + 1, 1
    y = _span(key[0], shape[0])
    x = _span(key[1], shape[1]) if len(key) > 1 else (0, shape[1], 1)
    return y, x


@dataclass
class _Patch:
    y0: int
    x0: int
    y1: int
    x1: int
    data: np.ndarray


class OverlayScan:
    """A ``Scan`` whose pixels are the original plus in-memory patches."""

    def __init__(self, base: Scan, patches: Sequence[_Patch]):
        self.path = getattr(base, "path", "")
        self.rgb = bool(getattr(base, "rgb", False))
        self.memmap = True
        self._array = OverlayArray(_pixels(base), patches)

    @property
    def data(self) -> OverlayArray:
        return self._array

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._array.shape

    def crop_bgr(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        piece = np.ascontiguousarray(self.data[y0:y1, x0:x1])
        if piece.ndim == 2:
            return cv2.cvtColor(piece, cv2.COLOR_GRAY2BGR)
        if piece.shape[2] == 4:
            piece = piece[:, :, :3]
        if self.rgb:
            piece = piece[:, :, ::-1]
        return np.ascontiguousarray(piece)


def _pixels(scan: Any) -> np.ndarray:
    return scan if isinstance(scan, np.ndarray) else scan.data


def apply_recipe(
    scan: Scan,
    recipe: Dict[str, Any],
    gt: Dict[str, Any],
) -> OverlayScan:
    base = _pixels(scan)
    rgb = bool(getattr(scan, "rgb", False))
    paper = paper_color(gt, rgb)
    ink = ink_color(gt, rgb)
    patches = [
        _render_op(base, op, gt, paper, ink)
        for op in recipe.get("patches") or []
    ]
    patches = [p for p in patches if p is not None]
    return OverlayScan(scan, patches)


def apply_ops_inplace(
    image: np.ndarray,
    ops: Sequence[Dict[str, Any]],
    gt: Dict[str, Any],
    *,
    rgb: bool = False,
) -> np.ndarray:
    """Mutate a (possibly cropped) BGR/RGB array. Used for geometric canvases."""
    paper = paper_color(gt, rgb)
    ink = ink_color(gt, rgb)
    out = np.array(image, copy=True)
    for op in ops:
        patch = _render_op(out, op, gt, paper, ink)
        if patch is None:
            continue
        out[patch.y0:patch.y1, patch.x0:patch.x1] = patch.data
    return out


def _render_op(
    image: np.ndarray,
    op: Dict[str, Any],
    gt: Dict[str, Any],
    paper: Tuple[int, int, int],
    ink: Tuple[int, int, int],
) -> Optional[_Patch]:
    kind = str(op.get("op") or "")
    height, width = image.shape[:2]
    if kind == "mask_rect":
        x0, y0, x1, y1 = _rect_from_op(op, gt, height, width)
        return _fill_patch(image, x0, y0, x1, y1, paper)
    if kind == "mask_triangle":
        x0, y0, x1, y1 = _rect_from_op(op, gt, height, width)
        return _triangle_patch(image, x0, y0, x1, y1, paper, str(op.get("corner") or "tl"))
    if kind == "mask_blob":
        x0, y0, x1, y1 = _rect_from_op(op, gt, height, width)
        return _blob_patch(image, x0, y0, x1, y1, paper, int(op.get("seed") or 0))
    if kind == "mask_band_h":
        return _band_h_patch(image, op, gt, paper)
    if kind == "mask_band_v":
        return _band_v_patch(image, op, gt, paper)
    if kind == "feeble_vertical":
        return _feeble_vertical_patch(image, op, gt, paper)
    if kind == "thin_vertical":
        return _thin_vertical_patch(image, op, gt)
    if kind == "sparse_vertical":
        return _sparse_vertical_patch(image, op, gt, paper, int(op.get("seed") or 1))
    if kind == "overspray_dots":
        return _overspray_dots_patch(image, op, gt, ink, int(op.get("seed") or 2))
    if kind == "overspray_blob":
        return _overspray_blob_patch(image, op, gt, ink, int(op.get("seed") or 3))
    if kind == "overspray_edge":
        return _overspray_edge_patch(image, op, gt, ink, int(op.get("seed") or 4))
    raise ValueError(f"Unknown overlay op {kind!r}")


def _clip_box(x0, y0, x1, y1, width, height) -> Tuple[int, int, int, int]:
    x0 = int(np.clip(x0, 0, width))
    x1 = int(np.clip(x1, 0, width))
    y0 = int(np.clip(y0, 0, height))
    y1 = int(np.clip(y1, 0, height))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def _fill_patch(image, x0, y0, x1, y1, color) -> Optional[_Patch]:
    if x1 <= x0 or y1 <= y0:
        return None
    tile = np.array(image[y0:y1, x0:x1], copy=True)
    tile[:] = color
    return _Patch(y0, x0, y1, x1, tile)


def _copy_patch(image, x0, y0, x1, y1) -> Optional[_Patch]:
    if x1 <= x0 or y1 <= y0:
        return None
    tile = np.array(image[y0:y1, x0:x1], copy=True)
    return _Patch(y0, x0, y1, x1, tile)


def _rect_from_op(op, gt, height, width) -> Tuple[int, int, int, int]:
    if "xyxy" in op:
        x0, y0, x1, y1 = [int(v) for v in op["xyxy"]]
        return _clip_box(x0, y0, x1, y1, width, height)
    kind = str(op.get("region") or "island")
    corner = str(op.get("corner") or "tl")
    w = int(op.get("w") or 200)
    h = int(op.get("h") or 200)
    x0, y0, x1, y1 = region_box(gt, kind)
    if corner == "tl":
        return _clip_box(x0, y0, x0 + w, y0 + h, width, height)
    if corner == "tr":
        return _clip_box(x1 - w + 1, y0, x1 + 1, y0 + h, width, height)
    if corner == "bl":
        return _clip_box(x0, y1 - h + 1, x0 + w, y1 + 1, width, height)
    return _clip_box(x1 - w + 1, y1 - h + 1, x1 + 1, y1 + 1, width, height)


def _triangle_patch(image, x0, y0, x1, y1, color, corner) -> Optional[_Patch]:
    patch = _copy_patch(image, x0, y0, x1, y1)
    if patch is None:
        return None
    h, w = patch.data.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    if corner == "tl":
        pts = [(0, 0), (w - 1, 0), (0, h - 1)]
    elif corner == "tr":
        pts = [(0, 0), (w - 1, 0), (w - 1, h - 1)]
    elif corner == "bl":
        pts = [(0, 0), (0, h - 1), (w - 1, h - 1)]
    else:
        pts = [(w - 1, 0), (w - 1, h - 1), (0, h - 1)]
    cv2.fillConvexPoly(mask, np.array(pts, np.int32), 255)
    patch.data[mask > 0] = color
    return patch


def _blob_patch(image, x0, y0, x1, y1, color, seed) -> Optional[_Patch]:
    patch = _copy_patch(image, x0, y0, x1, y1)
    if patch is None:
        return None
    rng = np.random.default_rng(seed)
    h, w = patch.data.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for _ in range(6):
        cx = int(rng.integers(0, max(1, w)))
        cy = int(rng.integers(0, max(1, h)))
        ax = int(rng.integers(max(8, w // 6), max(9, w // 2)))
        ay = int(rng.integers(max(8, h // 6), max(9, h // 2)))
        cv2.ellipse(mask, (cx, cy), (ax, ay), float(rng.integers(0, 180)), 0, 360, 255, -1)
    patch.data[mask > 0] = color
    return patch


def _band_h_patch(image, op, gt, paper) -> Optional[_Patch]:
    height, width = image.shape[:2]
    kind = str(op.get("region") or "island")
    x0, y0, x1, y1 = region_box(gt, kind)
    n_lines = int(op.get("n_lines") or 5)
    spacing = int(op.get("spacing") or LINE_SPACING_FULL)
    which = str(op.get("which") or "top")
    thick = max(1, n_lines * spacing)
    if which == "top":
        return _fill_patch(image, x0, y0, x1 + 1, y0 + thick, paper)
    return _fill_patch(image, x0, y1 - thick + 1, x1 + 1, y1 + 1, paper)


def _band_v_patch(image, op, gt, paper) -> Optional[_Patch]:
    height, width = image.shape[:2]
    kind = str(op.get("region") or "island")
    x0, y0, x1, y1 = region_box(gt, kind)
    which = str(op.get("which") or "outer")
    frac = float(op.get("y_frac") or 0.3)
    band = int(op.get("w") or 40)
    y_span = y1 - y0 + 1
    y_cut = y0 + int(round(frac * y_span))
    island_front = True
    if which == "outer":
        vx0, vx1 = x0, x0 + band
        yy0, yy1 = y0, y_cut
    elif which == "outer_bottom":
        vx0, vx1 = x0, x0 + band
        yy0, yy1 = y1 - int(round(frac * y_span)), y1 + 1
    elif which == "inner":
        vx0, vx1 = x1 - band + 1, x1 + 1
        yy0, yy1 = y0, y_cut
    else:
        vx0, vx1 = x1 - band + 1, x1 + 1
        yy0, yy1 = y1 - int(round(frac * y_span)), y1 + 1
    _ = island_front
    return _fill_patch(image, vx0, yy0, vx1, yy1, paper)


def _vertical_window(op, gt, image_shape) -> Tuple[int, int, int, int]:
    height, width = image_shape[:2]
    kind = str(op.get("region") or "island")
    x0, y0, x1, y1 = region_box(gt, kind)
    which = str(op.get("which") or "outer")
    band = int(op.get("w") or 50)
    y_frac = op.get("y_frac") or [0.0, 0.2]
    ya = y0 + int(float(y_frac[0]) * (y1 - y0 + 1))
    yb = y0 + int(float(y_frac[1]) * (y1 - y0 + 1))
    if which == "outer":
        xa, xb = x0, x0 + band
    else:
        xa, xb = x1 - band + 1, x1 + 1
    return _clip_box(xa, ya, xb, yb, width, height)


def _feeble_vertical_patch(image, op, gt, paper) -> Optional[_Patch]:
    x0, y0, x1, y1 = _vertical_window(op, gt, image.shape)
    patch = _copy_patch(image, x0, y0, x1, y1)
    if patch is None:
        return None
    alpha = float(op.get("alpha") or 0.35)
    paper_arr = np.array(paper, dtype=np.float32)
    blended = patch.data.astype(np.float32) * alpha + paper_arr * (1.0 - alpha)
    patch.data = np.clip(blended, 0, 255).astype(np.uint8)
    return patch


def _thin_vertical_patch(image, op, gt) -> Optional[_Patch]:
    x0, y0, x1, y1 = _vertical_window(op, gt, image.shape)
    patch = _copy_patch(image, x0, y0, x1, y1)
    if patch is None:
        return None
    k = int(op.get("erode") or 2)
    kernel = np.ones((1, max(1, k * 2 + 1)), np.uint8)
    patch.data = cv2.erode(patch.data, kernel)
    return patch


def _sparse_vertical_patch(image, op, gt, paper, seed) -> Optional[_Patch]:
    x0, y0, x1, y1 = _vertical_window(op, gt, image.shape)
    patch = _copy_patch(image, x0, y0, x1, y1)
    if patch is None:
        return None
    rng = np.random.default_rng(seed)
    drop = float(op.get("drop") or 0.65)
    mask = rng.random(patch.data.shape[:2]) < drop
    patch.data[mask] = paper
    return patch


def _outside_rect(op, gt, image_shape, pad: int = 0) -> Tuple[int, int, int, int]:
    height, width = image_shape[:2]
    kind = str(op.get("region") or "island")
    corner = str(op.get("corner") or "tl")
    w = int(op.get("w") or 180)
    h = int(op.get("h") or 180)
    x0, y0, x1, y1 = region_box(gt, kind)
    if corner == "tl":
        box = (x0 - w, y0 - h, x0, y0)
    elif corner == "tr":
        box = (x1 + 1, y0 - h, x1 + 1 + w, y0)
    elif corner == "bl":
        box = (x0 - w, y1 + 1, x0, y1 + 1 + h)
    else:
        box = (x1 + 1, y1 + 1, x1 + 1 + w, y1 + 1 + h)
    return _clip_box(box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad, width, height)


def _overspray_dots_patch(image, op, gt, ink, seed) -> Optional[_Patch]:
    x0, y0, x1, y1 = _outside_rect(op, gt, image.shape)
    patch = _copy_patch(image, x0, y0, x1, y1)
    if patch is None:
        return None
    rng = np.random.default_rng(seed)
    n = int(op.get("n") or 40)
    h, w = patch.data.shape[:2]
    for _ in range(n):
        cx = int(rng.integers(0, max(1, w)))
        cy = int(rng.integers(0, max(1, h)))
        r = int(rng.integers(1, int(op.get("r") or 6) + 1))
        cv2.circle(patch.data, (cx, cy), r, ink, -1)
    return patch


def _overspray_blob_patch(image, op, gt, ink, seed) -> Optional[_Patch]:
    height, width = image.shape[:2]
    kind = str(op.get("region") or "island")
    x0, y0, x1, y1 = region_box(gt, kind)
    past = int(op.get("past") or 200)
    bw = int(op.get("w") or 80)
    bh = int(op.get("h") or 60)
    which = str(op.get("which") or "below")
    mx = (x0 + x1) // 2
    if which == "below":
        bx0, by0 = mx - bw // 2, y1 + past
    elif which == "above":
        bx0, by0 = mx - bw // 2, y0 - past - bh
    elif which == "left":
        bx0, by0 = x0 - past - bw, (y0 + y1) // 2
    else:
        bx0, by0 = x1 + past, (y0 + y1) // 2
    bx0, by0, bx1, by1 = _clip_box(bx0, by0, bx0 + bw, by0 + bh, width, height)
    patch = _copy_patch(image, bx0, by0, bx1, by1)
    if patch is None:
        return None
    h, w = patch.data.shape[:2]
    cv2.ellipse(patch.data, (w // 2, h // 2), (max(4, w // 2 - 2), max(4, h // 2 - 2)), 0, 0, 360, ink, -1)
    _ = seed
    return patch


def _overspray_edge_patch(image, op, gt, ink, seed) -> Optional[_Patch]:
    height, width = image.shape[:2]
    kind = str(op.get("region") or "island")
    x0, y0, x1, y1 = region_box(gt, kind)
    which = str(op.get("which") or "left")
    thick = int(op.get("thick") or 18)
    if which == "left":
        box = (x0 - thick - 8, y0, x0, y1 + 1)
    elif which == "right":
        box = (x1 + 1, y0, x1 + 1 + thick + 8, y1 + 1)
    elif which == "top":
        box = (x0, y0 - thick - 8, x1 + 1, y0)
    else:
        box = (x0, y1 + 1, x1 + 1, y1 + 1 + thick + 8)
    xa, ya, xb, yb = _clip_box(*box, width, height)
    patch = _copy_patch(image, xa, ya, xb, yb)
    if patch is None:
        return None
    rng = np.random.default_rng(seed)
    h, w = patch.data.shape[:2]
    for _ in range(int(op.get("n") or 80)):
        cx = int(rng.integers(0, max(1, w)))
        cy = int(rng.integers(0, max(1, h)))
        cv2.circle(patch.data, (cx, cy), int(rng.integers(1, 5)), ink, -1)
    return patch


# ----------------------------------------------------------------------
# Geometric 1-colour sheet
# ----------------------------------------------------------------------

@dataclass
class GeometricLayout:
    island_width: int = GEO_REF.island_width
    stripe_width: int = GEO_REF.stripe_width
    gap: int = GEO_REF.island_stripe_gap
    height: int = GEO_REF.height
    pad: int = GEO_PAD
    line_spacing: int = GEO_LINE_SPACING
    vline_width: int = GEO_VLINE
    rotation_deg: float = 0.0
    shear_x: float = 0.0
    stitch_offset: int = 0
    num_heads: int = 3
    paper: Tuple[int, int, int] = (255, 255, 255)
    ink: Tuple[int, int, int] = (20, 20, 20)
    ops: List[Dict[str, Any]] = field(default_factory=list)


def render_geometric_sheet(
    layout: Optional[GeometricLayout] = None,
) -> Tuple[np.ndarray, Dict[str, Any], RegionReference]:
    """BGR canvas of one colour block with exact print-corner GT."""
    lay = layout or GeometricLayout()
    x_island0 = lay.pad + (abs(int(lay.stitch_offset)) if lay.stitch_offset < 0 else 0)
    x_island1 = x_island0 + lay.island_width - 1
    x_stripe0 = x_island1 + lay.gap + 1
    x_stripe1 = x_stripe0 + lay.stripe_width - 1
    y0 = lay.pad
    y1 = y0 + lay.height - 1
    extra_right = abs(int(lay.stitch_offset)) if lay.stitch_offset > 0 else 0
    width = x_stripe1 + lay.pad + extra_right + 1
    height = y1 + lay.pad + 1
    canvas = np.full((height, width, 3), lay.paper, dtype=np.uint8)
    ink = lay.ink

    canvas[y0:y1 + 1, x_stripe0:x_stripe1 + 1] = ink
    canvas[y0:y1 + 1, x_island0:x_island0 + lay.vline_width] = ink
    canvas[y0:y1 + 1, x_island1 - lay.vline_width + 1:x_island1 + 1] = ink
    # Inner dual-band verticals, inset ~1/4 and ~3/4 of the island.
    inner_a = x_island0 + lay.island_width // 4
    inner_b = x_island0 + (3 * lay.island_width) // 4
    canvas[y0:y1 + 1, inner_a:inner_a + max(2, lay.vline_width // 2)] = ink
    canvas[y0:y1 + 1, inner_b:inner_b + max(2, lay.vline_width // 2)] = ink
    dash = max(2, lay.line_spacing // 4)
    for y in range(y0, y1 + 1, lay.line_spacing):
        canvas[y:min(y1 + 1, y + dash), x_island0:x_island1 + 1] = ink

    if lay.stitch_offset:
        head_h = lay.height // max(1, lay.num_heads)
        for k in range(1, lay.num_heads):
            ya = y0 + k * head_h
            yb = y0 + lay.height if k + 1 == lay.num_heads else y0 + (k + 1) * head_h
            shift = int(lay.stitch_offset)
            strip = canvas[ya:yb].copy()
            canvas[ya:yb] = lay.paper
            if shift > 0:
                canvas[ya:yb, shift:] = strip[:, :-shift]
            elif shift < 0:
                canvas[ya:yb, :shift] = strip[:, -shift:]

    island_corners = corners_from_xyxy(x_island0, y0, x_island1, y1)
    stripe_corners = corners_from_xyxy(x_stripe0, y0, x_stripe1, y1)
    if lay.stitch_offset:
        # Lower heads move; covering AABB must include the shifted edges.
        shift = int(lay.stitch_offset)
        for name in ("bl", "br"):
            island_corners[name][0] += shift
            stripe_corners[name][0] += shift
        if lay.num_heads > 2:
            # Mid-height dogleg lives on the first stitch row.
            mid_y = y0 + (lay.height // max(1, lay.num_heads))
            island_corners["ml"] = [float(x_island0 + shift), float(mid_y)]
            island_corners["mr"] = [float(x_island1 + shift), float(mid_y)]
            stripe_corners["ml"] = [float(x_stripe0 + shift), float(mid_y)]
            stripe_corners["mr"] = [float(x_stripe1 + shift), float(mid_y)]

    matrix = None
    if lay.rotation_deg or lay.shear_x:
        cx, cy = width / 2.0, height / 2.0
        rot = cv2.getRotationMatrix2D((cx, cy), lay.rotation_deg, 1.0)
        rot[0, 1] += lay.shear_x
        matrix = rot
        canvas = cv2.warpAffine(
            canvas, rot, (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=lay.paper,
        )
        island_corners = _transform_corners(island_corners, rot)
        stripe_corners = _transform_corners(stripe_corners, rot)

    gt = _gt_from_corners(
        "Geo", island_corners, stripe_corners, canvas.shape,
        paper_bgr=lay.paper, ink_bgr=lay.ink,
    )
    if lay.ops:
        canvas = apply_ops_inplace(canvas, lay.ops, gt, rgb=False)

    ref = replace(
        GEO_REF,
        island_width=lay.island_width,
        island_stripe_gap=lay.gap,
        stripe_width=lay.stripe_width,
        height=lay.height,
    )
    _ = matrix
    return canvas, gt, ref


def _transform_corners(corners: Dict[str, List[float]], matrix: np.ndarray) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for name, pt in corners.items():
        x, y = float(pt[0]), float(pt[1])
        nx = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
        ny = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
        out[name] = [float(nx), float(ny)]
    return out


def _gt_from_corners(
    name: str,
    island: Dict[str, List[float]],
    stripe: Dict[str, List[float]],
    shape: Tuple[int, ...],
    paper_bgr: Tuple[int, int, int],
    ink_bgr: Tuple[int, int, int],
) -> Dict[str, Any]:
    def span(corners):
        xs = [c[0] for k, c in corners.items() if k in CORNERS]
        ys = [c[1] for k, c in corners.items() if k in CORNERS]
        return [int(round(min(xs))), int(round(max(xs)))], [int(round(min(ys))), int(round(max(ys)))]

    ix, iy = span(island)
    sx, sy = span(stripe)
    y = [min(iy[0], sy[0]), max(iy[1], sy[1])]
    paper_rgb = (paper_bgr[2], paper_bgr[1], paper_bgr[0])
    ink_rgb = (ink_bgr[2], ink_bgr[1], ink_bgr[0])
    return {
        "image_stem": "geometric",
        "shape": list(shape),
        "channel_order": "bgr",
        "paper_rgb": list(paper_rgb),
        "ink_rgb": list(ink_rgb),
        "colors": [{
            "name": name,
            "island": {"x": ix, "y": y, "corners": island},
            "stripe": {"x": sx, "y": y, "corners": stripe},
        }],
    }


def covering_xyxy(corners: Dict[str, Sequence[float]]) -> Tuple[int, int, int, int]:
    xs = [float(pt[0]) for pt in corners.values()]
    ys = [float(pt[1]) for pt in corners.values()]
    return int(np.floor(min(xs))), int(np.floor(min(ys))), int(np.ceil(max(xs))), int(np.ceil(max(ys)))
