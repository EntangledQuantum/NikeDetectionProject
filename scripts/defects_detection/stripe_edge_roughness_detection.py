"""
Stripe Edge Roughness Detection

Traces the left and right edges of a solid-color vertical stripe and scores
high-frequency jaggedness (saw-tooth / scalloped print fringe).

Calibration errors are NOT roughness:
  - Stitch: an abrupt lateral step at a head boundary.
  - Roll: a gradual lateral drift across a head.

Those low-frequency components are removed before scoring: the raw edge is
split at detected stitches, a robust linear fit removes roll on each head
segment, and a wide median high-pass keeps only the short-scale residual.
Debris spikes are rejected by using MAD + P95 (not RMS) as the decision
metrics.

Sensitivity presets (high / medium / low) change how large and persistent
the residual must be before a span is flagged. Both edges are always
quantified, even when neither is flagged.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.image_saver import save_image


def _median_filter_1d(values: np.ndarray, window: int) -> np.ndarray:
    """NaN-aware odd-window median filter for a 1-D profile."""
    n = len(values)
    win = max(3, int(window) | 1)
    valid = ~np.isnan(values)
    if valid.sum() <= win:
        return values.copy()
    filled = values.copy()
    xs = np.flatnonzero(valid)
    filled[~valid] = np.interp(np.flatnonzero(~valid), xs, values[valid])
    pad = win // 2
    padded = np.pad(filled, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, win)
    smoothed = np.median(windows, axis=1).astype(np.float64)
    out = np.full(n, np.nan, dtype=np.float64)
    out[valid] = smoothed[valid]
    return out


def _rolling_step_profile(edge: np.ndarray, window: int, guard: int,
                          stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """Step size at sampled rows: median(below) - median(above)."""
    n = len(edge)
    ys = np.arange(window + guard, n - window - guard, stride)
    steps = np.full(len(ys), np.nan)
    for i, y in enumerate(ys):
        above = edge[y - guard - window:y - guard]
        below = edge[y + guard:y + guard + window]
        above = above[~np.isnan(above)]
        below = below[~np.isnan(below)]
        if above.size < window // 4 or below.size < window // 4:
            continue
        steps[i] = np.median(below) - np.median(above)
    return ys, steps


def _runs_of(mask: np.ndarray, min_len: int = 1) -> List[Tuple[int, int]]:
    """Contiguous True runs as (start, end_inclusive)."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s + 1 >= min_len]


def _profile_metrics(residual: np.ndarray) -> Dict[str, float]:
    """Robust residual statistics. Empty input returns zeros."""
    v = residual[~np.isnan(residual)]
    if v.size < 8:
        return {
            "rms_px": 0.0, "mad_px": 0.0, "sigma_px": 0.0,
            "p95_px": 0.0, "peak_to_peak_px": 0.0, "n": 0,
        }
    rms = float(np.sqrt(np.mean(v * v)))
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    p95 = float(np.percentile(np.abs(v), 95))
    ptp = float(np.percentile(v, 95) - np.percentile(v, 5))
    return {
        "rms_px": rms,
        "mad_px": mad,
        "sigma_px": mad * 1.4826,
        "p95_px": p95,
        "peak_to_peak_px": ptp,
        "n": int(v.size),
    }


class StripeEdgeRoughnessDetector:
    """Detect and quantify left/right stripe-edge roughness."""

    def __init__(self, sensitivity: str = "medium", debug: bool = False):
        """Configure residual thresholds and debug output.

        Args:
            sensitivity: 'low' | 'medium' | 'high'.
            debug: If True, keep a residual-profile plot for saving.
        """
        self.sensitivity = sensitivity
        self.debug = debug

        # Decision uses MAD (typical tooth size, outlier-robust) AND P95
        # (amplitude of the larger teeth). RMS is reported but not used to
        # flag, because a few debris hijacks inflate it.
        if sensitivity == "high":
            self.mad_threshold = 1.30
            self.p95_threshold = 4.2
            self.min_span_frac = 0.08
        elif sensitivity == "low":
            self.mad_threshold = 2.50
            self.p95_threshold = 7.2
            self.min_span_frac = 0.18
        else:
            self.mad_threshold = 1.70
            self.p95_threshold = 5.0
            self.min_span_frac = 0.12

        # Stitch split (px). Kept slightly more sensitive than the dedicated
        # misalignment detector so residual spikes at head joints are removed
        # even when those joints are not reported as stitch defects.
        self.stitch_step_threshold = 4.0
        self.stitch_guard_rows = 40
        self.highpass_window = 81
        self.local_window = 256
        self.local_stride = 64

        self._debug_plot = None
        self._last_profiles = None

        print(f"StripeEdgeRoughness ({sensitivity}): "
              f"MAD>={self.mad_threshold:.2f}px and P95>={self.p95_threshold:.1f}px")

    # ------------------------------------------------------------------
    # Edge tracing
    # ------------------------------------------------------------------

    def _extract_raw_edges(self, gray: np.ndarray):
        """Sub-pixel left/right edge X vs row, interior-anchored.

        Same walk as stripe misalignment (cannot be hijacked by paper-side
        marks) but without the 31-row median that would erase roughness.
        """
        col_mean = gray.mean(axis=0)
        lo, hi = float(col_mean.min()), float(col_mean.max())
        if hi - lo < 20:
            return None
        mid = (lo + hi) / 2.0

        dark_cols = np.flatnonzero(col_mean < mid)
        if dark_cols.size < 10:
            return None
        gx_left, gx_right = int(dark_cols.min()), int(dark_cols.max())
        stripe_w = gx_right - gx_left
        inset = max(30, stripe_w // 6)
        anchor_l = min(gx_left + inset, gray.shape[1] - 1)
        anchor_r = max(gx_right - inset, 0)

        mask = gray < mid
        h, w = gray.shape

        corridor = mask[:, :anchor_l + 1][:, ::-1]
        run = np.logical_and.accumulate(corridor, axis=1).sum(axis=1)
        left_i = (anchor_l - run + 1).astype(np.int32)
        left_i[run == 0] = -1

        corridor = mask[:, anchor_r:]
        run = np.logical_and.accumulate(corridor, axis=1).sum(axis=1)
        right_i = (anchor_r + run - 1).astype(np.int32)
        right_i[run == 0] = -1

        left = self._subpixel_refine(gray, left_i, mid, side="left")
        right = self._subpixel_refine(gray, right_i, mid, side="right")
        return left, right, mid, gx_left, gx_right

    @staticmethod
    def _subpixel_refine(gray: np.ndarray, integer_edge: np.ndarray,
                         mid: float, side: str) -> np.ndarray:
        """Linear-interpolate the mid-grey crossing around the integer edge."""
        h, w = gray.shape
        ys = np.arange(h)
        valid = integer_edge >= 0
        x = np.clip(integer_edge, 1, w - 2)
        if side == "left":
            g0 = gray[ys, x - 1].astype(np.float64)
            g1 = gray[ys, x].astype(np.float64)
            denom = g1 - g0
            t = (mid - g0) / np.where(np.abs(denom) < 1e-3, np.nan, denom)
            t = np.clip(t, -1.5, 1.5)
            refined = (x - 1) + t
        else:
            g0 = gray[ys, x].astype(np.float64)
            g1 = gray[ys, x + 1].astype(np.float64)
            denom = g1 - g0
            t = (mid - g0) / np.where(np.abs(denom) < 1e-3, np.nan, denom)
            t = np.clip(t, -1.5, 1.5)
            refined = x + t
        out = np.full(h, np.nan, dtype=np.float64)
        ok = valid & ~np.isnan(refined)
        out[ok] = refined[ok]
        # Fall back to the integer edge when the local gradient is flat.
        fallback = valid & np.isnan(refined)
        out[fallback] = integer_edge[fallback].astype(np.float64)
        return out

    # ------------------------------------------------------------------
    # Stitch / roll removal
    # ------------------------------------------------------------------

    def _detect_stitch_ys(self, left: np.ndarray, right: np.ndarray) -> List[int]:
        """Head-boundary rows from a slow edge profile.

        Only **both-edge** steps are used. A one-sided jump is typical of a
        ragged fringe and must not split the residual, or real roughness
        would be mistaken for a stitch and punched out of the score.
        """
        n = len(left)
        window = min(300, max(30, n // 20))
        guard = min(15, max(2, window // 10))
        stride = 5
        min_sep = 2 * window
        slow_l = _median_filter_1d(left, 31)
        slow_r = _median_filter_1d(right, 31)

        raw = []
        for edge_name, edge in (('left', slow_l), ('right', slow_r)):
            ys, steps = _rolling_step_profile(edge, window, guard, stride)
            if len(ys) == 0:
                continue
            mag = np.abs(np.nan_to_num(steps, nan=0.0))
            order = np.argsort(mag)[::-1]
            taken = []
            for idx in order:
                if mag[idx] < self.stitch_step_threshold:
                    break
                y = int(ys[idx])
                if any(abs(y - t) < min_sep for t in taken):
                    continue
                taken.append(y)
                raw.append({'y': y, 'edge': edge_name})

        if not raw:
            return []
        raw.sort(key=lambda c: c['y'])
        merged = []
        for cand in raw:
            if merged and abs(cand['y'] - merged[-1]['y']) < min_sep:
                merged[-1]['edges'].add(cand['edge'])
                merged[-1]['y'] = int(round((merged[-1]['y'] + cand['y']) / 2.0))
            else:
                merged.append({'y': cand['y'], 'edges': {cand['edge']}})
        return [m['y'] for m in merged if len(m['edges']) >= 2]

    def _piecewise_linear_trend(self, edge: np.ndarray,
                                stitch_ys: List[int]) -> np.ndarray:
        """Per-head robust line fit (removes roll; stitches are segment bounds)."""
        n = len(edge)
        trend = np.full(n, np.nan, dtype=np.float64)
        bounds = [0] + list(stitch_ys) + [n]
        guard = self.stitch_guard_rows
        for y0, y1 in zip(bounds[:-1], bounds[1:]):
            s0 = y0 + (guard if y0 > 0 else 0)
            s1 = y1 - (guard if y1 < n else 0)
            if s1 - s0 < 20:
                s0, s1 = y0, y1
            ys = np.arange(s0, s1, dtype=np.float64)
            vals = edge[s0:s1]
            valid = ~np.isnan(vals)
            if valid.sum() < 20:
                continue
            xs, vv = ys[valid], vals[valid].astype(np.float64)
            slope, intercept = np.polyfit(xs, vv, 1)
            res = vv - (slope * xs + intercept)
            sigma = float(np.median(np.abs(res - np.median(res)))) * 1.4826
            if sigma > 0.1:
                keep = np.abs(res) <= 2.8 * sigma
                if keep.sum() > 15:
                    slope, intercept = np.polyfit(xs[keep], vv[keep], 1)
            # Extrapolate across the stitch-guard gap so residual is defined.
            all_ys = np.arange(y0, y1, dtype=np.float64)
            trend[y0:y1] = slope * all_ys + intercept
        return trend

    def _highpass_residual(self, edge: np.ndarray, trend: np.ndarray) -> np.ndarray:
        """Edge minus roll/stitch trend, then a wide median high-pass."""
        residual = edge - trend
        n = len(residual)
        valid = ~np.isnan(residual)
        if valid.sum() < self.highpass_window:
            return residual
        filled = residual.copy()
        xs = np.flatnonzero(valid)
        filled[~valid] = np.interp(np.flatnonzero(~valid), xs, residual[valid])
        slow = _median_filter_1d(filled, self.highpass_window)
        hp = filled - slow
        hp[~valid] = np.nan
        # Mask rows next to NaNs / image ends lightly already handled.
        hp[:2] = np.nan
        hp[n - 2:] = np.nan
        return hp

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _local_flags(self, residual: np.ndarray, stitch_ys: List[int],
                     height: int) -> List[Tuple[int, int, Dict[str, float]]]:
        """Sliding-window MAD/P95 flags, merged, stitch neighborhoods skipped."""
        n = len(residual)
        win = min(self.local_window, max(64, n // 4))
        stride = min(self.local_stride, max(16, win // 4))
        blocked = np.zeros(n, dtype=bool)
        for y in stitch_ys:
            blocked[max(0, y - self.stitch_guard_rows):
                    min(n, y + self.stitch_guard_rows + 1)] = True

        hits = np.zeros(n, dtype=bool)
        window_metrics: List[Tuple[int, int, Dict[str, float]]] = []
        for y0 in range(0, max(1, n - win + 1), stride):
            y1 = min(n, y0 + win)
            sl = residual[y0:y1]
            usable = ~np.isnan(sl) & ~blocked[y0:y1]
            if usable.sum() < win * 0.4:
                continue
            metrics = _profile_metrics(sl[usable])
            # Debris / void hijacks: one tall spike raises P95 (and RMS)
            # far above MAD. True saw-tooth roughness keeps P95 within a
            # few multiples of MAD.
            if metrics["p95_px"] > 5.5 * max(metrics["mad_px"], 0.4):
                continue
            if (metrics["mad_px"] >= self.mad_threshold
                    and metrics["p95_px"] >= self.p95_threshold):
                hits[y0:y1] = usable
                window_metrics.append((y0, y1 - 1, metrics))

        # Span floor is in rows, tied to the analysis window — not the full
        # stripe height — so a locally rough stretch on a tall scan is kept.
        min_span = max(48, int(round(self.min_span_frac * 10 * win)))
        regions = []
        for s, e in _runs_of(hits, min_len=min_span):
            metrics = _profile_metrics(residual[s:e + 1])
            if (metrics["mad_px"] >= self.mad_threshold * 0.85
                    and metrics["p95_px"] >= self.p95_threshold * 0.85):
                regions.append((s, e, metrics))
        return regions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image, image_path: Optional[str] = None):
        """Run edge-roughness detection.

        Args:
            image: Stripe image (BGR or grayscale).
            image_path: Unused; accepted for pipeline compatibility.

        Returns:
            (visualization_bgr, defects). Defect types:
              edge_roughness_summary — always, both-edge quantification
              edge_roughness — one per flagged (edge, y-span)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        extracted = self._extract_raw_edges(gray)
        if extracted is None:
            print("StripeEdgeRoughness: no stripe/paper contrast found")
            vis = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return vis, []

        left, right, mid, gx_left, gx_right = extracted
        height = len(left)
        stitch_ys = self._detect_stitch_ys(left, right)

        defects: List[Dict] = []
        residuals = {}
        overall = {}
        flagged = {}

        for name, prof in (("left", left), ("right", right)):
            trend = self._piecewise_linear_trend(prof, stitch_ys)
            residual = self._highpass_residual(prof, trend)
            residuals[name] = residual
            metrics = _profile_metrics(residual)
            overall[name] = metrics
            regions = self._local_flags(residual, stitch_ys, height)
            flagged[name] = regions

            # Whole-edge flag if the global score is over threshold even when
            # no local window survived the min-span merge (short crops).
            if (not regions
                    and metrics["mad_px"] >= self.mad_threshold
                    and metrics["p95_px"] >= self.p95_threshold
                    and metrics["n"] >= 64):
                regions = [(0, height - 1, metrics)]
                flagged[name] = regions

            for y0, y1, m in regions:
                yc = (y0 + y1) // 2
                x = float(np.nanmedian(prof[max(0, yc - 50):yc + 51]))
                if np.isnan(x):
                    x = float(gx_left if name == "left" else gx_right)
                defects.append({
                    "type": "edge_roughness",
                    "edge": name,
                    "y0": int(y0),
                    "y1": int(y1),
                    "span_px": int(y1 - y0 + 1),
                    "x": round(x, 1),
                    "location": (int(round(x)), yc),
                    "roughness_sigma_px": round(m["sigma_px"], 2),
                    "mad_px": round(m["mad_px"], 2),
                    "p95_px": round(m["p95_px"], 2),
                    "rms_px": round(m["rms_px"], 2),
                    "peak_to_peak_px": round(m["peak_to_peak_px"], 2),
                    "mad_threshold": self.mad_threshold,
                    "p95_threshold": self.p95_threshold,
                })

        summary = {
            "type": "edge_roughness_summary",
            "sensitivity": self.sensitivity,
            "stitch_splits": [int(y) for y in stitch_ys],
            "left": {k: (round(v, 2) if isinstance(v, float) else v)
                     for k, v in overall["left"].items()},
            "right": {k: (round(v, 2) if isinstance(v, float) else v)
                      for k, v in overall["right"].items()},
            "left_flagged": bool(flagged["left"]),
            "right_flagged": bool(flagged["right"]),
            "mad_threshold": self.mad_threshold,
            "p95_threshold": self.p95_threshold,
        }
        # Put the summary first so JSON readers see quantification immediately.
        defects = [summary] + defects

        print(
            "StripeEdgeRoughness: "
            f"left sig={overall['left']['sigma_px']:.2f}px "
            f"(MAD {overall['left']['mad_px']:.2f}, P95 {overall['left']['p95_px']:.2f}"
            f"{', ROUGH' if flagged['left'] else ''}) | "
            f"right sig={overall['right']['sigma_px']:.2f}px "
            f"(MAD {overall['right']['mad_px']:.2f}, P95 {overall['right']['p95_px']:.2f}"
            f"{', ROUGH' if flagged['right'] else ''}) | "
            f"{sum(1 for d in defects if d['type']=='edge_roughness')} region(s), "
            f"{len(stitch_ys)} stitch split(s)"
        )

        visualization = self._create_visualization(
            image, left, right, residuals, flagged, overall, stitch_ys)
        self._last_profiles = (left, right, residuals)
        if self.debug:
            self._debug_plot = self._plot_residuals(residuals, height, flagged)
        return visualization, defects

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _create_visualization(self, original, left, right, residuals,
                              flagged, overall, stitch_ys):
        if len(original.shape) == 2:
            vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            vis = original.copy()
        h, w = vis.shape[:2]
        step = max(1, h // 5000)

        def _draw_profile(prof, color, thickness):
            ys = np.arange(0, h, step)
            xs = prof[ys]
            ok = ~np.isnan(xs)
            pts = np.stack([xs[ok], ys[ok]], axis=1).astype(np.int32)
            if len(pts) > 1:
                cv2.polylines(vis, [pts.reshape(-1, 1, 2)], False, color, thickness)

        # Baseline traces (smooth = green, will be overpainted where rough)
        _draw_profile(left, (40, 180, 40), 2)
        _draw_profile(right, (40, 180, 40), 2)

        for name, prof in (("left", left), ("right", right)):
            for y0, y1, _m in flagged[name]:
                ys = np.arange(y0, y1 + 1, step)
                xs = prof[ys]
                ok = ~np.isnan(xs)
                pts = np.stack([xs[ok], ys[ok]], axis=1).astype(np.int32)
                if len(pts) > 1:
                    cv2.polylines(vis, [pts.reshape(-1, 1, 2)], False, (0, 0, 220), 3)

        for y in stitch_ys:
            cv2.line(vis, (0, y), (w - 1, y), (180, 180, 180), 1)

        def _label(name, x_hint):
            m = overall[name]
            tag = "ROUGH" if flagged[name] else "smooth"
            color = (0, 0, 220) if flagged[name] else (40, 160, 40)
            text = (f"{name.upper()} {tag}  sig={m['sigma_px']:.2f}px  "
                    f"MAD={m['mad_px']:.2f}  P95={m['p95_px']:.2f}  "
                    f"ptp={m['peak_to_peak_px']:.1f}")
            x = 8 if name == "left" else max(8, w - 720)
            y = 28 if name == "left" else 56
            cv2.putText(vis, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        color, 2, cv2.LINE_AA)

        _label("left", 0)
        _label("right", w)
        cv2.putText(
            vis,
            f"Edge roughness [{self.sensitivity}]  "
            f"MAD>={self.mad_threshold:.2f}px  P95>={self.p95_threshold:.1f}px",
            (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1,
            cv2.LINE_AA,
        )
        return vis

    @staticmethod
    def _plot_residuals(residuals, height, flagged):
        """Compact residual plot for debug (left=red, right=blue)."""
        plot_h, plot_w = 480, 1100
        img = np.full((plot_h, plot_w, 3), 255, dtype=np.uint8)
        cv2.line(img, (50, plot_h // 2), (plot_w - 10, plot_h // 2), (200, 200, 200), 1)
        all_v = np.concatenate([
            residuals[k][~np.isnan(residuals[k])] for k in residuals
            if np.any(~np.isnan(residuals[k]))
        ]) if residuals else np.array([1.0])
        span = max(4.0, float(np.percentile(np.abs(all_v), 98)) * 2)
        for name, color in (("left", (0, 0, 200)), ("right", (200, 0, 0))):
            hp = residuals[name]
            ys = np.linspace(0, height - 1, plot_w - 60).astype(int)
            xs = hp[ys]
            ok = ~np.isnan(xs)
            px = 50 + np.flatnonzero(ok)
            py = (plot_h // 2 - (xs[ok] / span * (plot_h - 40))).astype(int)
            py = np.clip(py, 0, plot_h - 1)
            pts = np.stack([px, py], axis=1).astype(np.int32)
            if len(pts) > 1:
                cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, color, 1)
        cv2.putText(img, "residual px (left=red, right=blue)", (50, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)
        return img

    def save_debug_images(self, output_dir: str, base_name: str):
        """Save the residual plot when debug mode is enabled."""
        if self.debug and self._debug_plot is not None:
            return save_image(output_dir, base_name, self._debug_plot,
                              "edge_roughness_residual")
        return None
