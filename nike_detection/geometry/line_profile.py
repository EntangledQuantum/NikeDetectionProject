"""
Per-column line profiling on the normalized ink-density map.

This replaces the binary kernel/threshold statistics the island detectors used
to run. Working on :mod:`nike_detection.geometry.ink_density` alpha instead of
a binary mask buys two things:

* **Color independence.** Alpha is normalized against the region's own paper
  and saturated-core level, so a Yellow band and a Key band produce the same
  numbers for the same physical defect.
* **A haze/missing distinction that actually exists in the data.** A binary
  mask can only say "ink" or "no ink", so a pale smear and a healthy line are
  identical to it. Alpha keeps the *amount*, which is the whole signal.

Every line is reduced to three per-column series, all in alpha units:

    mass(x)   total ink in the corridor      -> how much ink landed
    peak(x)   densest pixel in the corridor  -> how concentrated it is
    center(x) ink-weighted row               -> where it landed

From those, two dimensionless ratios drive every decision downstream:

    coverage = mass / baseline_mass    ~1 healthy, ~0 missing
    density  = peak / baseline_peak    ~1 healthy, low = pale / spread (haze)

Both baselines are measured from the band itself with a bounded per-line
correction, so slow drift in illumination or ink laydown cancels out and no
absolute, color-specific constant is ever needed.

Speed comes from doing the geometry once per band and the statistics in
line groups: the sheared map is sliced by contiguous row ranges, so peak
memory stays proportional to a group of lines rather than the whole region.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from nike_detection.geometry.ink_density import SCALE, InkField

# Lines profiled per slice of the sheared map. Bounds peak memory on a
# full-height island without leaving numpy for a per-line Python loop.
_LINE_GROUP = 48


@dataclass
class BandProfile:
    """Geometry and per-column ink statistics for one print band."""

    x0: int
    x1: int
    slope: float
    spacing: float
    shift: np.ndarray             # int32 [W] sheared->true row offset per column
    rows: np.ndarray              # float32 [L] sheared row of each line
    inserted: np.ndarray          # bool [L] synthesized for a fully missing line
    center: np.ndarray            # float32 [L, W] corridor center (sheared, relative to row)
    mass: np.ndarray              # float32 [L, W] ink mass per column
    peak: np.ndarray              # float32 [L, W] densest alpha per column
    coverage: np.ndarray          # float32 [L, W] mass / baseline
    density: np.ndarray           # float32 [L, W] peak / baseline
    residual: np.ndarray          # float32 [L, W] short-range trajectory wander (px)
    tilt: np.ndarray              # float32 [L] straight-fit residual slope
    base_mass: float = 0.0        # band-level healthy mass (alpha-pixels)
    base_peak: float = 0.0        # band-level healthy core density (alpha)
    noise: float = 0.0            # paper noise in alpha units
    corridor: int = 0             # corridor half-height in px

    @property
    def n_lines(self) -> int:
        return int(self.rows.shape[0])

    @property
    def width(self) -> int:
        return int(self.shift.shape[0])

    def true_y(self, line_index: int, xs: np.ndarray) -> np.ndarray:
        """True (unsheared) row of a line at the given columns."""
        idx = np.clip(np.rint(xs).astype(np.int32), 0, self.width - 1)
        return self.rows[line_index] + self.center[line_index, idx] + self.shift[idx]


def _box(values: np.ndarray, size: int) -> np.ndarray:
    """Odd-width moving average along the last axis (edge replicated)."""
    k = max(1, int(size) | 1)
    if k <= 1:
        return values.astype(np.float32, copy=False)
    src = np.ascontiguousarray(values, dtype=np.float32)
    if src.ndim == 1:
        src = src[None, :]
        return cv2.blur(src, (k, 1), borderType=cv2.BORDER_REPLICATE)[0]
    return cv2.blur(src, (k, 1), borderType=cv2.BORDER_REPLICATE)


def column_runs(mask: np.ndarray, min_len: int = 1) -> List[Tuple[int, int]]:
    """Contiguous True runs of a 1-D mask as (start, end_inclusive)."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    return [
        (int(s), int(e))
        for s, e in zip(starts, ends)
        if e - s + 1 >= min_len
    ]


class BandProfiler:
    """Turn a band crop into line geometry plus normalized per-column series.

    The only tunables are structural (how tall a corridor is, how the slope
    search is bounded). Defect thresholds live in the detector, expressed in
    the dimensionless units this class produces.
    """

    def __init__(
        self,
        slope_min: float = -0.005,
        slope_max: float = 0.035,
        coarse_step: float = 0.002,
        fine_step: float = 0.0002,
        window_fraction: float = 0.45,
        corridor_fraction: float = 0.34,
        peak_rel_threshold: float = 0.22,
        missing_line_gap_factor: float = 1.6,
    ) -> None:
        self.slope_min = slope_min
        self.slope_max = slope_max
        self.coarse_step = coarse_step
        self.fine_step = fine_step
        self.window_fraction = window_fraction
        self.corridor_fraction = corridor_fraction
        self.peak_rel_threshold = peak_rel_threshold
        self.missing_line_gap_factor = missing_line_gap_factor

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def profile(
        self,
        field: InkField,
        x0: int,
        x1: int,
        debug: bool = False,
    ) -> Optional[BandProfile]:
        """Profile the band spanning columns [x0, x1] of an ink field.

        Returns None when the band holds no usable print.
        """
        alpha = field.alpha[:, x0:x1 + 1]
        if alpha.shape[1] < 16:
            return None
        if int(alpha.max()) < int(0.2 * SCALE):
            return None

        slope = self._find_slope(alpha)
        shift = np.round(slope * np.arange(alpha.shape[1])).astype(np.int32)
        sheared = self._shear(alpha, shift)

        rows, spacing = self._find_rows(sheared)
        if rows is None or len(rows) < 2:
            return None
        rows, inserted = self._insert_missing(rows, spacing)

        profile = self._measure(sheared, rows, inserted, spacing, field.noise)
        profile.x0, profile.x1 = int(x0), int(x1)
        profile.slope = float(slope)
        profile.shift = shift

        if debug:
            print(
                f"BandProfiler[{x0}:{x1}] slope={slope:.5f} spacing={spacing:.1f}px "
                f"lines={profile.n_lines} (+{int(inserted.sum())} inserted) "
                f"base_mass={profile.base_mass:.2f} base_peak={profile.base_peak:.2f}"
            )
        return profile

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _find_slope(self, alpha: np.ndarray, block: int = 32) -> float:
        """Shear search maximizing sharpness of the row profile.

        Runs on alpha rather than a binary mask, so a faint Yellow band scores
        exactly like a saturated Key band.
        """
        height, width = alpha.shape
        n_blocks = (width + block - 1) // block
        profiles = np.zeros((n_blocks, height), dtype=np.float32)
        centers = np.zeros(n_blocks, dtype=np.float64)
        for b in range(n_blocks):
            bx0, bx1 = b * block, min(width, (b + 1) * block)
            profiles[b] = alpha[:, bx0:bx1].sum(axis=1, dtype=np.float32)
            centers[b] = (bx0 + bx1 - 1) / 2.0

        def score(candidate: float) -> float:
            total = np.zeros(height, dtype=np.float32)
            shifts = np.round(candidate * centers).astype(int)
            for b in range(n_blocks):
                sh = shifts[b]
                if sh >= 0:
                    if sh < height:
                        total[:height - sh] += profiles[b][sh:]
                else:
                    total[-sh:] += profiles[b][:height + sh]
            return float(np.dot(total, total))

        coarse = np.arange(self.slope_min, self.slope_max + 1e-9, self.coarse_step)
        best = max(coarse, key=score)
        fine = np.arange(best - self.coarse_step, best + self.coarse_step + 1e-9,
                         self.fine_step)
        best = max(fine, key=score)
        return float(best)

    @staticmethod
    def _shear(alpha: np.ndarray, shift: np.ndarray) -> np.ndarray:
        height = alpha.shape[0]
        sheared = np.zeros_like(alpha)
        for sh in np.unique(shift):
            cols = np.flatnonzero(shift == sh)
            if sh >= 0:
                if sh < height:
                    sheared[:height - sh, cols] = alpha[sh:, cols]
            else:
                sheared[-sh:, cols] = alpha[:height + sh, cols]
        return sheared

    def _find_rows(self, sheared: np.ndarray) -> Tuple[Optional[List[float]], float]:
        """Line row centers and spacing from the sheared alpha row profile."""
        profile = sheared.mean(axis=1, dtype=np.float32)
        smooth = _box(profile, 5)
        reference = float(np.percentile(smooth, 99))
        if reference <= 1e-3:
            return None, 0.0

        active = smooth > self.peak_rel_threshold * reference
        runs = column_runs(active)
        merged: List[Tuple[int, int]] = []
        for s, e in runs:
            if merged and s - merged[-1][1] - 1 <= 3:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        if len(merged) < 2:
            return None, 0.0

        centers = []
        for s, e in merged:
            seg = smooth[s:e + 1].astype(np.float64)
            idx = np.arange(s, e + 1)
            total = seg.sum()
            centers.append(float((idx * seg).sum() / total) if total > 0 else float(s))

        deltas = np.diff(centers)
        spacing = float(np.median(deltas))
        if spacing > 0 and deltas.size >= 4:
            inliers = deltas[(deltas > 0.55 * spacing) & (deltas < 1.55 * spacing)]
            if inliers.size >= 2:
                spacing = float(np.median(inliers))
        if spacing <= 2:
            return None, 0.0
        return centers, spacing

    def _insert_missing(
        self, centers: List[float], spacing: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Synthesize rows where a whole line is absent so it is still scored."""
        rows: List[float] = [centers[0]]
        flags: List[bool] = [False]
        for current in centers[1:]:
            gap = current - rows[-1]
            if spacing > 0 and gap > self.missing_line_gap_factor * spacing:
                n_missing = int(round(gap / spacing)) - 1
                previous = rows[-1]
                for k in range(1, n_missing + 1):
                    rows.append(previous + gap * k / (n_missing + 1))
                    flags.append(True)
            rows.append(current)
            flags.append(False)
        return np.asarray(rows, dtype=np.float32), np.asarray(flags, dtype=bool)

    # ------------------------------------------------------------------
    # Per-column statistics
    # ------------------------------------------------------------------

    def _measure(
        self,
        sheared: np.ndarray,
        rows: np.ndarray,
        inserted: np.ndarray,
        spacing: float,
        noise: float,
    ) -> BandProfile:
        height, width = sheared.shape
        n_lines = rows.shape[0]
        half = max(4, int(round(spacing * self.window_fraction)))
        corridor = max(3, int(round(spacing * self.corridor_fraction)))
        smooth_k = max(3, int(round(0.08 * spacing)) | 1)

        mass = np.zeros((n_lines, width), dtype=np.float32)
        peak = np.zeros((n_lines, width), dtype=np.float32)
        center = np.zeros((n_lines, width), dtype=np.float32)
        residual = np.zeros((n_lines, width), dtype=np.float32)
        tilt = np.zeros(n_lines, dtype=np.float32)

        # The trajectory residual keeps only wander shorter than a few line
        # spacings, so paper roll and head-to-head drift cannot masquerade as
        # the short-range zig-zag a stitch produces. It is kept per column
        # rather than reduced here, because whether a given column's wander is
        # trustworthy depends on whether that column has healthy ink -- a
        # judgement the detector makes, not the profiler.
        wave_k = max(9, int(round(4.0 * spacing)) | 1)

        xs = np.arange(width, dtype=np.float32)
        for start in range(0, n_lines, _LINE_GROUP):
            stop = min(n_lines, start + _LINE_GROUP)
            gy0 = max(0, int(np.floor(rows[start])) - half - 1)
            gy1 = min(height, int(np.ceil(rows[stop - 1])) + half + 2)
            if gy1 <= gy0:
                continue
            block = sheared[gy0:gy1].astype(np.float32)
            block /= SCALE

            for line in range(start, stop):
                row = float(rows[line])
                y0 = max(gy0, int(round(row)) - half)
                y1 = min(gy1, int(round(row)) + half + 1)
                if y1 - y0 < 3:
                    continue
                window = block[y0 - gy0:y1 - gy0]
                offsets = (np.arange(y0, y1, dtype=np.float32) - row)

                # Pass 1: coarse centroid over the full window locates the ink
                # even when the straight fit is a poor model (stitch zig-zag).
                totals = window.sum(axis=0)
                safe = np.maximum(totals, 1e-6)
                coarse = (offsets[:, None] * window).sum(axis=0) / safe
                coarse[totals <= 1e-6] = np.nan
                local = self._fill(coarse)
                local = _box(local, smooth_k)
                np.clip(local, offsets[0] + 1.0, offsets[-1] - 1.0, out=local)

                # Pass 2: tight corridor around that path, so printed ink that
                # zig-zags at a head stitch stays inside the mask instead of
                # reading as a missing nozzle.
                inside = np.abs(offsets[:, None] - local[None, :]) <= corridor
                masked = window * inside
                line_mass = masked.sum(axis=0)
                line_peak = masked.max(axis=0)
                safe_mass = np.maximum(line_mass, 1e-6)
                refined = (offsets[:, None] * masked).sum(axis=0) / safe_mass
                refined[line_mass <= 1e-6] = np.nan

                mass[line] = line_mass
                peak[line] = line_peak
                center[line] = local

                trace = self._fill(refined)
                slope, intercept = self._fit(xs, trace, line_mass)
                drift = trace - (intercept + slope * xs)
                residual[line] = drift - _box(drift, wave_k)
                tilt[line] = float(slope)

        base_mass, base_peak = self._baselines(mass, peak, inserted)
        coverage = mass / max(base_mass, 1e-6)
        density = peak / max(base_peak, 1e-6)

        return BandProfile(
            x0=0, x1=width - 1, slope=0.0, spacing=float(spacing),
            shift=np.zeros(width, dtype=np.int32),
            rows=rows, inserted=inserted, center=center,
            mass=mass, peak=peak, coverage=coverage, density=density,
            residual=residual, tilt=tilt,
            base_mass=float(base_mass), base_peak=float(base_peak),
            noise=float(noise), corridor=int(corridor),
        )

    @staticmethod
    def _fill(values: np.ndarray) -> np.ndarray:
        """Linear-interpolate NaNs so a trajectory survives ink-free columns."""
        out = np.array(values, dtype=np.float32, copy=True)
        valid = np.isfinite(out)
        if valid.all():
            return out
        if not valid.any():
            return np.zeros_like(out)
        idx = np.arange(out.size, dtype=np.float32)
        out[~valid] = np.interp(idx[~valid], idx[valid], out[valid]).astype(np.float32)
        return out

    @staticmethod
    def _fit(
        xs: np.ndarray, ys: np.ndarray, weights: np.ndarray
    ) -> Tuple[float, float]:
        """Ink-weighted least-squares line fit (one pass, no iteration)."""
        w = np.maximum(weights, 0.0).astype(np.float64)
        total = w.sum()
        if total <= 1e-6:
            return 0.0, float(np.median(ys))
        mx = float((w * xs).sum() / total)
        my = float((w * ys).sum() / total)
        dx = xs - mx
        var = float((w * dx * dx).sum())
        if var <= 1e-6:
            return 0.0, my
        slope = float((w * dx * (ys - my)).sum() / var)
        return slope, my - slope * mx

    @staticmethod
    def _baselines(
        mass: np.ndarray, peak: np.ndarray, inserted: np.ndarray
    ) -> Tuple[float, float]:
        """Band-level healthy mass / core density.

        Uses the upper quartile of every real line, then the median across
        lines: high enough to sit inside healthy print even when a line is
        largely defective, robust enough to ignore a few outlier lines.
        """
        real = ~inserted
        if not real.any():
            return 1.0, 1.0
        line_mass = np.percentile(mass[real], 75, axis=1)
        line_peak = np.percentile(peak[real], 75, axis=1)
        base_mass = float(np.median(line_mass))
        base_peak = float(np.median(line_peak))
        return max(base_mass, 1e-3), max(base_peak, 1e-3)


