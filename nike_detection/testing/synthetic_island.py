"""
Synthetic island-band generator with ground truth.

Real scans tell you whether a result *looks* right; they cannot tell you what
was missed. This module renders islands with known defects so the line-defect
detector can be scored on precision and recall, and -- more importantly --
so a threshold can be checked against every ink at once.

The renderer is deliberately physical rather than cosmetic, because the two
failure modes of the previous detector both came from effects a naive
generator would omit:

* **Stipple.** Print lines are dotted, so ink mass fluctuates strongly along
  x. Any test that keys off raw thickness fires constantly on healthy print.
  ``stipple`` reproduces that modulation, so a false-positive-prone rule will
  show up here rather than in production.
* **Ink colour.** Each ink absorbs a different channel, so a threshold tuned
  in grayscale silently changes meaning between Key and Yellow. Inks are
  rendered through per-channel absorbance measured from real scans, which is
  what makes a cross-colour sweep meaningful.

Haze is modelled the way it physically occurs -- the same ink spread over more
area -- by lowering the ridge amplitude while widening its profile, instead of
merely dimming pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Per-channel absorbance (BGR) of a saturated line, measured from real 2400-DPI
# white-paper scans. Yellow barely dents luminance but absorbs blue almost
# completely, which is exactly why grayscale thresholding fails on it.
INK_ABSORBANCE: Dict[str, Tuple[float, float, float]] = {
    "key": (0.86, 0.86, 0.85),
    "cyan": (0.08, 0.35, 0.95),
    "magenta": (0.30, 0.92, 0.16),
    "yellow": (0.95, 0.10, 0.04),
}

PAPER_BGR: Tuple[float, float, float] = (231.0, 229.0, 224.0)


@dataclass
class Sample:
    """A rendered island plus the defects that were injected into it."""

    image: np.ndarray
    spacing: float
    slope: float
    color: str
    bands: List[Tuple[int, int]]
    n_lines: int = 0
    missing: List[dict] = field(default_factory=list)
    haze: List[dict] = field(default_factory=list)
    stitch: List[dict] = field(default_factory=list)


class IslandBuilder:
    """Compose a synthetic island band by band, defect by defect.

    Example:
        >>> builder = IslandBuilder(color="cyan", n_lines=30, seed=1)
        >>> builder.add_missing(line=5, x0=300, x1=500)
        >>> builder.add_haze(line=9, x0=200, x1=700, strength=0.6)
        >>> sample = builder.build()
    """

    def __init__(
        self,
        color: str = "cyan",
        n_lines: int = 30,
        spacing: float = 96.0,
        width: int = 1600,
        slope: float = 0.020,
        thickness: float = 4.2,
        stipple: float = 0.28,
        noise: float = 2.2,
        blur: float = 1.4,
        margin: float = 60.0,
        seed: int = 0,
    ) -> None:
        """Configure the print being simulated.

        Args:
            color: Key of :data:`INK_ABSORBANCE`.
            n_lines: Print lines rendered in the band.
            spacing: Vertical distance between line centres, px.
            width: Band width, px.
            slope: Line slope (dy/dx), matching the real slanted print.
            thickness: Gaussian sigma of a healthy line's vertical profile.
            stipple: Relative amplitude of the dotted-print modulation. This
                is the texture that used to trigger false haze.
            noise: Per-channel scanner noise sigma, 8-bit counts.
            blur: Scanner point-spread sigma, px.
            margin: Blank margin above the first and below the last line.
            seed: RNG seed, so a failing case is reproducible.
        """
        if color not in INK_ABSORBANCE:
            raise ValueError(f"unknown ink {color!r}; expected {sorted(INK_ABSORBANCE)}")
        self.color = color
        self.n_lines = int(n_lines)
        self.spacing = float(spacing)
        self.width = int(width)
        self.slope = float(slope)
        self.thickness = float(thickness)
        self.stipple = float(stipple)
        self.noise = float(noise)
        self.blur = float(blur)
        self.margin = float(margin)
        self.rng = np.random.default_rng(seed)

        self._missing: List[dict] = []
        self._haze: List[dict] = []
        self._stitch: List[dict] = []

    # ------------------------------------------------------------------
    # Defect injection
    # ------------------------------------------------------------------

    def add_missing(self, line: int, x0: int, x1: int) -> "IslandBuilder":
        """A run of nozzles that did not fire: no ink at all."""
        self._missing.append({"line": int(line), "x0": int(x0), "x1": int(x1)})
        return self

    def add_haze(self, line: int, x0: int, x1: int, strength: float = 0.5) -> "IslandBuilder":
        """Ink that landed but smeared.

        Args:
            strength: 0 = healthy, 1 = maximally diffuse. The ridge amplitude
                falls and its width grows together, conserving most of the ink
                so this stays distinguishable from a missing nozzle.
        """
        self._haze.append({
            "line": int(line), "x0": int(x0), "x1": int(x1),
            "strength": float(np.clip(strength, 0.0, 1.0)),
        })
        return self

    def add_stitch(self, line: int, amplitude: float = 3.0) -> "IslandBuilder":
        """A head-join calibration error: short-range zig-zag of the trajectory.

        Args:
            amplitude: RMS wander in px, before scanner blur.
        """
        self._stitch.append({"line": int(line), "amplitude": float(amplitude)})
        return self

    def add_whole_line_missing(self, line: int) -> "IslandBuilder":
        return self.add_missing(line, 0, self.width - 1)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def build(self, with_verticals: bool = True) -> Sample:
        """Render the configured island to a BGR image plus ground truth."""
        height = int(round(2 * self.margin + (self.n_lines - 1) * self.spacing
                           + self.slope * self.width)) + 1
        alpha = np.zeros((height, self.width), dtype=np.float32)
        xs = np.arange(self.width, dtype=np.float32)

        missing_by_line = self._group(self._missing)
        haze_by_line = self._group(self._haze)
        stitch_by_line = {s["line"]: s for s in self._stitch}

        for index in range(self.n_lines):
            base = self.margin + index * self.spacing + self.slope * xs
            base = base + self._wander(xs, scale=0.35 * self.thickness, wavelength=9.0)

            stitch = stitch_by_line.get(index)
            if stitch is not None:
                base = base + self._wander(
                    xs, scale=stitch["amplitude"],
                    wavelength=1.4 * self.spacing)

            amp = np.full(self.width, 1.0, dtype=np.float32)
            amp *= self._stipple_profile(xs)
            sigma = np.full(self.width, self.thickness, dtype=np.float32)

            for spec in haze_by_line.get(index, []):
                lo, hi = self._clip_span(spec["x0"], spec["x1"])
                if hi <= lo:
                    continue
                ramp = self._edge_ramp(hi - lo)
                strength = spec["strength"] * ramp
                # Same ink, wider footprint: peak falls roughly as the inverse
                # of the spread, which is the signature the detector keys on.
                spread = 1.0 + 2.6 * strength
                sigma[lo:hi] *= spread
                amp[lo:hi] *= (1.0 - 0.25 * strength) / spread

            for spec in missing_by_line.get(index, []):
                lo, hi = self._clip_span(spec["x0"], spec["x1"])
                if hi > lo:
                    amp[lo:hi] = 0.0

            self._draw_ridge(alpha, base, amp, sigma)

        bands = [(0, self.width - 1)]
        if with_verticals:
            alpha = self._add_verticals(alpha)

        return Sample(
            image=self._to_bgr(alpha), spacing=self.spacing, slope=self.slope,
            color=self.color, bands=bands, n_lines=self.n_lines,
            missing=[dict(m) for m in self._missing],
            haze=[dict(h) for h in self._haze],
            stitch=[dict(s) for s in self._stitch],
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _group(specs: List[dict]) -> Dict[int, List[dict]]:
        grouped: Dict[int, List[dict]] = {}
        for spec in specs:
            grouped.setdefault(spec["line"], []).append(spec)
        return grouped

    def _clip_span(self, x0: int, x1: int) -> Tuple[int, int]:
        return max(0, int(x0)), min(self.width, int(x1) + 1)

    @staticmethod
    def _edge_ramp(length: int, fraction: float = 0.15) -> np.ndarray:
        """Taper a defect in and out; real smears do not start abruptly."""
        ramp = np.ones(length, dtype=np.float32)
        edge = max(1, int(length * fraction))
        if 2 * edge < length:
            ramp[:edge] = np.linspace(0.0, 1.0, edge, dtype=np.float32)
            ramp[-edge:] = np.linspace(1.0, 0.0, edge, dtype=np.float32)
        return ramp

    def _wander(self, xs: np.ndarray, scale: float, wavelength: float) -> np.ndarray:
        """Band-limited random walk: smooth noise at a chosen wavelength."""
        if scale <= 0:
            return np.zeros_like(xs)
        raw = self.rng.standard_normal(xs.size).astype(np.float32)
        k = max(3, int(round(wavelength)) | 1)
        smooth = cv2.GaussianBlur(raw.reshape(1, -1), (k, 1), 0).reshape(-1)
        std = float(smooth.std())
        if std <= 1e-6:
            return np.zeros_like(xs)
        return (smooth / std) * scale

    def _stipple_profile(self, xs: np.ndarray) -> np.ndarray:
        """Dotted-print modulation: healthy but strongly varying ink mass."""
        if self.stipple <= 0:
            return np.ones_like(xs)
        raw = self.rng.standard_normal(xs.size).astype(np.float32)
        smooth = cv2.GaussianBlur(raw.reshape(1, -1), (5, 1), 0).reshape(-1)
        std = float(smooth.std()) or 1.0
        return np.clip(1.0 + self.stipple * smooth / std, 0.25, 1.75)

    @staticmethod
    def _draw_ridge(
        alpha: np.ndarray, center: np.ndarray, amp: np.ndarray, sigma: np.ndarray
    ) -> None:
        """Accumulate one Gaussian ridge, touching only the rows it covers."""
        height, width = alpha.shape
        reach = float(np.max(sigma)) * 3.0
        y0 = max(0, int(np.floor(center.min() - reach)))
        y1 = min(height, int(np.ceil(center.max() + reach)) + 1)
        if y1 <= y0:
            return
        rows = np.arange(y0, y1, dtype=np.float32)[:, None]
        delta = rows - center[None, :]
        alpha[y0:y1] += amp[None, :] * np.exp(-0.5 * (delta / sigma[None, :]) ** 2)

    def _add_verticals(self, alpha: np.ndarray) -> np.ndarray:
        """Boundary lines that frame a real band, so band detection has work."""
        height, width = alpha.shape
        for x in (2, width - 3):
            lo, hi = max(0, x - 2), min(width, x + 3)
            alpha[:, lo:hi] = np.maximum(alpha[:, lo:hi], 0.95)
        return alpha

    def _to_bgr(self, alpha: np.ndarray) -> np.ndarray:
        """Ink density -> scanned pixels, through this ink's absorbance."""
        absorb = np.asarray(INK_ABSORBANCE[self.color], dtype=np.float32)
        paper = np.asarray(PAPER_BGR, dtype=np.float32)
        alpha = np.clip(alpha, 0.0, 1.0)

        image = paper[None, None, :] * (1.0 - alpha[:, :, None] * absorb[None, None, :])
        if self.blur > 0:
            image = cv2.GaussianBlur(image, (0, 0), self.blur)
        if self.noise > 0:
            image += self.rng.normal(0.0, self.noise, image.shape).astype(np.float32)
        return np.clip(image, 0, 255).astype(np.uint8)


def healthy(color: str = "cyan", seed: int = 0, **kwargs) -> Sample:
    """A clean island: anything reported on this is a false positive."""
    return IslandBuilder(color=color, seed=seed, **kwargs).build()
