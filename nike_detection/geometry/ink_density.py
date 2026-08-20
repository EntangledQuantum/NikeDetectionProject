"""
Self-calibrating ink density for a print region.

Every island detector used to threshold **grayscale**, which silently makes
the algorithm color dependent: on white paper the luminance contrast between
paper and a saturated line is roughly

    Key ~180   Magenta ~127   Cyan ~85   Yellow ~21

so one universal grayscale threshold is simultaneously far too strict for
Yellow and loose for Key. Yellow is the pathological case: its Otsu value
lands *above* the usual clamp, so almost no yellow pixel is ever counted as
ink.

Each ink absorbs a different channel, though, and in *its own* channel every
color separates cleanly (Yellow ~155 in blue, Cyan ~219 in red, Magenta ~188
in green). This module measures that direction from the image itself and
returns a normalized density map:

    alpha = 0.0   paper
    alpha = 1.0   a healthy, fully saturated line core

Because the paper white, the absorbance direction and the saturated-core
level are all *measured per region*, downstream thresholds expressed in alpha
are genuinely color independent -- there is no per-ink constant anywhere.

The map is stored as uint8 (``SCALE`` counts per unit alpha) so it costs the
same memory as the binary mask it replaces, while carrying 256 density levels
instead of 1 bit. That extra resolution is what makes "ink is present but
pale and spread out" (haze) separable from "no ink at all" (missing nozzle).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# uint8 counts per unit alpha. 180 keeps ~40% headroom above a healthy core so
# the densest pixels still differ from a merely healthy one; saturating here
# would flatten exactly the contrast the haze test reads.
SCALE = 180.0

# Rows per chunk when converting a full region, so a 5163x44228 island never
# needs a full-size float temporary.
_CHUNK_ROWS = 4096


@dataclass
class InkField:
    """Normalized ink density for one region.

    Attributes:
        alpha: uint8 [H, W] density map, ``SCALE`` counts per unit alpha.
        paper: Measured paper white per channel (BGR), float32[3].
        axis: Unit absorbance direction of this region's ink, float32[3].
        core: Absorbance projection of a saturated line core (pre-normalization).
        noise: Robust paper noise level in alpha units (1 sigma).
        contrast: Paper-to-core separation in raw 8-bit counts along ``axis``;
            a quality signal for logs, not used for decisions.
    """

    alpha: np.ndarray
    paper: np.ndarray
    axis: np.ndarray
    core: float
    noise: float
    contrast: float

    @property
    def scale(self) -> float:
        return SCALE

    def to_float(self, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Float32 alpha for a sub-region (x0, y0, x1, y1), or the whole map."""
        if region is None:
            view = self.alpha
        else:
            x0, y0, x1, y1 = region
            view = self.alpha[y0:y1, x0:x1]
        return view.astype(np.float32) / SCALE


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def measure_ink_field(
    image: np.ndarray,
    subsample: int = 4,
    paper_percentile: float = 92.0,
    ink_percentile: float = 99.0,
    core_percentile: float = 90.0,
) -> InkField:
    """Measure this region's paper, ink axis and core level, then map to alpha.

    Args:
        image: Region image, BGR or grayscale.
        subsample: Row/column step used for the statistics pass. The map
            itself is always built at full resolution.
        paper_percentile: Per-channel intensity percentile taken as paper
            white. Ink covers well under half the area, so a high percentile
            is a robust unlit-paper estimate that also tracks the gray
            background of the clear material.
        ink_percentile: Percentile of the absorbance used to locate saturated
            line cores when fitting the ink axis.
        core_percentile: Percentile *within* those core pixels taken as full
            density. A median would sit on the shoulders of the line profile
            rather than its ridge, putting real cores above alpha 1 and
            clipping them all to the same value.

    Returns:
        An :class:`InkField`. When the region holds no measurable ink the map
        is all zeros and ``core`` is 0.0.
    """
    bgr = _as_bgr(image)
    height, width = bgr.shape[:2]

    step = max(1, int(subsample))
    sub = bgr[::step, ::step].reshape(-1, 3).astype(np.float32)

    paper = np.percentile(sub, paper_percentile, axis=0).astype(np.float32)
    paper = np.maximum(paper, 1.0)

    absorb = np.clip(1.0 - sub / paper, 0.0, 1.0)

    # Max-over-channels is already color agnostic (it picks blue for yellow,
    # red for cyan) and is used only to *select* core pixels. The axis is
    # then fitted from those pixels so chromatic noise and off-color debris
    # project weakly instead of counting as full ink.
    strength = absorb.max(axis=1)
    peak = float(np.percentile(strength, ink_percentile))
    if peak <= 0.02:
        return InkField(
            alpha=np.zeros((height, width), dtype=np.uint8),
            paper=paper,
            axis=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            core=0.0,
            noise=0.0,
            contrast=0.0,
        )

    core_pixels = strength >= 0.5 * peak
    axis = absorb[core_pixels].mean(axis=0)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-6:
        axis = np.array([1.0, 1.0, 1.0], dtype=np.float32) / np.sqrt(3.0)
    else:
        axis = (axis / norm).astype(np.float32)

    projection = absorb @ axis
    core = float(np.percentile(projection[core_pixels], core_percentile))
    if core <= 1e-3:
        core = float(np.percentile(projection, ink_percentile))
    if core <= 1e-3:
        core = 1.0

    # Paper noise in alpha units: MAD of the projection over paper pixels.
    paper_pixels = projection[strength < 0.15 * peak]
    if paper_pixels.size > 32:
        mad = float(np.median(np.abs(paper_pixels - np.median(paper_pixels))))
        noise = 1.4826 * mad / core
    else:
        noise = 0.0

    alpha = _project_to_alpha(bgr, paper, axis, core)
    contrast = float(core * np.dot(axis, paper))

    return InkField(
        alpha=alpha,
        paper=paper,
        axis=axis,
        core=core,
        noise=float(noise),
        contrast=contrast,
    )


def _project_to_alpha(
    bgr: np.ndarray,
    paper: np.ndarray,
    axis: np.ndarray,
    core: float,
) -> np.ndarray:
    """Map BGR to uint8 alpha via one three-channel lookup table.

    ``alpha = sum_c axis_c * clip(1 - I_c / paper_c, 0, 1) / core`` is separable
    across channels, so it collapses into a single 256-entry BGR table. The
    per-channel contributions are then combined with saturating 8-bit adds,
    which both clamps at full density and keeps the whole conversion inside
    OpenCV's SIMD paths -- no float image is ever materialized.
    """
    values = np.arange(256, dtype=np.float32)
    table = np.empty((1, 256, 3), dtype=np.uint8)
    for channel in range(3):
        absorb = np.clip(1.0 - values / paper[channel], 0.0, 1.0)
        counts = (SCALE / core) * float(axis[channel]) * absorb
        table[0, :, channel] = np.clip(np.rint(counts), 0, 255).astype(np.uint8)

    height, width = bgr.shape[:2]
    alpha = np.empty((height, width), dtype=np.uint8)
    for y0 in range(0, height, _CHUNK_ROWS):
        y1 = min(height, y0 + _CHUNK_ROWS)
        blue, green, red = cv2.split(cv2.LUT(bgr[y0:y1], table))
        cv2.add(cv2.add(blue, green), red, dst=alpha[y0:y1])
    return alpha


def as_pseudo_gray(field: InkField) -> np.ndarray:
    """Density inverted into a gray-like image: paper bright, ink dark.

    Lets geometry code written against grayscale (band and vertical-line
    detection) run on normalized density instead, so a Yellow region -- which
    has only ~21 levels of real luminance contrast -- presents the same
    separation as Key. Threshold ``t`` on this image means ``alpha >=
    (255 - t) / SCALE``.
    """
    return cv2.bitwise_not(field.alpha)
