"""
Stripe Misalignment Detection Algorithm

Detects lateral misalignment of the printed vertical stripe caused by
printer-head calibration errors between successive heads:

  - Stitch error: an abrupt horizontal step of the stripe edge at a head
    boundary (one head prints shifted relative to the next).
  - Roll error: a gradual lateral drift of the stripe edge across a head
    segment (the head is rotated / the web rolls).

Algorithm (edge-profile based, 1 px resolution):
  1. Binarize stripe vs paper at the mid-level between the stripe interior
     and the paper, measured from the column intensity profile.
  2. Per row, record the stripe's left and right edge x positions; median
     filter the profiles over rows to remove debris/ragged-edge outliers.
  3. Stitch detection: at every row, compare the median edge position in a
     window ABOVE against the window BELOW (guard gap around the row). A
     step larger than the sensitivity threshold, at a local extremum, is a
     stitch misalignment. Detected independently on the left and right
     edges, then merged (a true stitch shifts both edges together).
  4. Roll detection: between consecutive stitches (or over the whole image),
     robust-fit the stripe center position vs y; if the total drift across
     the segment exceeds the threshold, report a roll error.

The number of print heads does not matter: steps are found wherever they
occur, so 3-head and 4-head patterns are handled identically.

This detector works at 1 px resolution (the old kernel-scanning version was
quantized to 5 px steps with a 20 px threshold and could not see the real
stitch errors of a few pixels).
"""

import cv2
import numpy as np

from nike_detection.io.image_saver import save_image


def _rolling_step_profile(edge, window, guard, stride):
    """Step size at sampled rows: median(below window) - median(above window).

    Args:
        edge: 1D float array of edge x positions (NaN where unknown).
        window: Rows in each median window.
        guard: Rows skipped on both sides of the evaluated row (keeps the
            transition itself out of both windows).
        stride: Evaluation stride in rows.

    Returns:
        (ys, steps): sampled row indices and their step values (NaN where
        either window has too little data).
    """
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


class StripeMisalignmentDetector:
    """Detect stitch steps and roll drift of a vertical stripe's edges."""

    def __init__(self, kernel_size=50, kernel_width=None, kernel_height=None,
                 step_size=None, line_detection_threshold=0.15,
                 defect_threshold=None, sensitivity='medium', debug=False):
        """Configure thresholds and debug mode.

        Args:
            kernel_size: Unused; accepted for backwards compatibility.
            kernel_width: Unused; accepted for backwards compatibility.
            kernel_height: Unused; accepted for backwards compatibility.
            step_size: Unused; accepted for backwards compatibility.
            line_detection_threshold: Unused; accepted for compatibility.
            defect_threshold: Optional override (px) for the stitch step
                threshold. None uses the sensitivity preset.
            sensitivity: One of {'low', 'medium', 'high'}.
            debug: If True, keep debug artifacts (edge mask, profiles).
        """
        self.sensitivity = sensitivity
        self.debug = debug

        print(f"Sensitivity: {sensitivity}")

        if sensitivity == 'high':
            self.step_threshold = 3.0     # px: stitch step
            self.roll_threshold = 5.0     # px: total drift across a segment
        elif sensitivity == 'low':
            self.step_threshold = 10.0
            self.roll_threshold = 15.0
        else:  # medium
            self.step_threshold = 5.0
            self.roll_threshold = 8.0

        if defect_threshold is not None:
            self.step_threshold = float(defect_threshold)

        # Analysis geometry (rows); scaled down for short images at runtime.
        self.median_window = 31       # edge-profile median filter
        self.step_window = 300        # median window on each side of a row
        self.step_guard = 15          # guard gap around the evaluated row
        self.step_stride = 5          # step-profile sampling stride
        self.segment_margin = 150     # rows ignored at segment ends for roll

        self._debug_edge_image = None
        self._last_profiles = None

    # ------------------------------------------------------------------
    # Edge profiles
    # ------------------------------------------------------------------

    def _extract_edge_profiles(self, gray):
        """Per-row left/right stripe edge x positions (median filtered).

        The edge is defined as the boundary of the contiguous ink run that
        is connected to the stripe interior. Tracing is anchored at columns
        safely inside the stripe and walks outward while the mask stays
        dark, so pen marks / debris on the paper next to the stripe cannot
        hijack the edge (a plain "first dark pixel per row" scan does).

        Returns:
            (left, right, mid_level) with NaN where a row has no stripe, or
            None when no stripe-vs-paper contrast exists.
        """
        col_mean = gray.mean(axis=0)
        lo, hi = float(col_mean.min()), float(col_mean.max())
        if hi - lo < 20:
            return None
        mid = (lo + hi) / 2.0

        # Global stripe bounds from the column profile
        dark_cols = np.flatnonzero(col_mean < mid)
        if dark_cols.size < 10:
            return None
        gx_left, gx_right = int(dark_cols.min()), int(dark_cols.max())
        stripe_w = gx_right - gx_left
        inset = max(30, stripe_w // 6)
        anchor_l = min(gx_left + inset, gray.shape[1] - 1)
        anchor_r = max(gx_right - inset, 0)

        mask = gray < mid

        # Left edge: from anchor_l walk left while dark (vectorized as a
        # reversed cumulative AND over the corridor columns).
        corridor = mask[:, :anchor_l + 1][:, ::-1]
        run = np.logical_and.accumulate(corridor, axis=1).sum(axis=1)
        left = (anchor_l - run + 1).astype(np.float32)
        left[run == 0] = np.nan

        # Right edge: from anchor_r walk right while dark.
        corridor = mask[:, anchor_r:]
        run = np.logical_and.accumulate(corridor, axis=1).sum(axis=1)
        right = (anchor_r + run - 1).astype(np.float32)
        right[run == 0] = np.nan

        # Median filter over rows: debris/ragged-edge outliers vanish
        win = self.median_window
        for prof in (left, right):
            valid = ~np.isnan(prof)
            if valid.sum() <= win:
                continue
            # Bridge NaN rows so the sliding window stays well defined,
            # then keep smoothed values only where the row was valid.
            filled = prof.copy()
            filled[~valid] = np.interp(np.flatnonzero(~valid),
                                       np.flatnonzero(valid),
                                       prof[valid])
            pad = win // 2
            padded = np.pad(filled, pad, mode='edge')
            windows = np.lib.stride_tricks.sliding_window_view(padded, win)
            smoothed = np.median(windows, axis=1).astype(np.float32)
            prof[valid] = smoothed[valid]

        return left, right, mid

    # ------------------------------------------------------------------
    # Stitch + roll detection
    # ------------------------------------------------------------------

    def _detect_stitches(self, left, right):
        """Find abrupt edge steps; merge left/right edge detections.

        A step seen on both edges is a confident stitch (the whole head
        print shifted). A step on a single edge can also be caused by a
        ragged print fringe, so it must exceed 1.6x the threshold to be
        reported.
        """
        n = len(left)
        window = min(self.step_window, max(30, n // 20))
        guard = min(self.step_guard, max(2, window // 10))
        # Real stitches are one head apart; anything closer than this is
        # the same (possibly smeared) transition.
        min_separation = 2 * window

        candidates = []
        for edge_name, edge in (('left', left), ('right', right)):
            ys, steps = _rolling_step_profile(edge, window, guard,
                                              self.step_stride)
            if len(ys) == 0:
                continue
            mag = np.abs(np.nan_to_num(steps, nan=0.0))
            order = np.argsort(mag)[::-1]
            taken = []
            for idx in order:
                if mag[idx] < self.step_threshold:
                    break
                y = int(ys[idx])
                if any(abs(y - t) < min_separation for t in taken):
                    continue  # non-max suppression
                taken.append(y)
                candidates.append({
                    'y': y,
                    'step': float(steps[idx]),
                    'edge': edge_name,
                })

        # Merge left/right detections of the same stitch (close rows)
        candidates.sort(key=lambda c: c['y'])
        merged = []
        for cand in candidates:
            if merged and abs(cand['y'] - merged[-1]['y']) < min_separation:
                prev = merged[-1]
                prev['edges'].append(cand['edge'])
                # Same-direction steps on both edges = confident stitch
                prev['step'] = float(np.mean([prev['step'], cand['step']]))
                prev['y'] = int(round((prev['y'] + cand['y']) / 2))
            else:
                merged.append({
                    'y': cand['y'],
                    'step': cand['step'],
                    'edges': [cand['edge']],
                })

        # Single-edge steps need stronger evidence than both-edge steps
        return [m for m in merged
                if len(set(m['edges'])) > 1
                or abs(m['step']) >= 1.6 * self.step_threshold]

    def _detect_roll(self, left, right, stitch_ys, height):
        """Per-segment robust drift of the stripe center between stitches."""
        center = (left + right) / 2.0
        boundaries = [0] + sorted(stitch_ys) + [height]
        margin = self.segment_margin

        rolls = []
        for y0, y1 in zip(boundaries[:-1], boundaries[1:]):
            s0, s1 = y0 + margin, y1 - margin
            if s1 - s0 < max(500, margin * 3):
                continue
            seg = center[s0:s1]
            ys = np.arange(s0, s1, dtype=np.float64)
            valid = ~np.isnan(seg)
            if valid.sum() < (s1 - s0) * 0.3:
                continue

            xs, vals = ys[valid], seg[valid].astype(np.float64)
            slope, intercept = np.polyfit(xs, vals, 1)
            # One sigma-clip pass against debris-driven residuals
            residuals = vals - (slope * xs + intercept)
            sigma = residuals.std()
            if sigma > 0.1:
                keep = np.abs(residuals) <= 2.5 * sigma
                if keep.sum() > 100:
                    slope, intercept = np.polyfit(xs[keep], vals[keep], 1)

            drift = float(slope * (s1 - s0))
            if abs(drift) >= self.roll_threshold:
                rolls.append({
                    'y0': int(y0), 'y1': int(y1),
                    'drift_px': drift,
                    'slope_px_per_1k_rows': float(slope * 1000.0),
                })
        return rolls

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image):
        """Run stitch/roll misalignment detection on a stripe image.

        Args:
            image: Input stripe image (BGR or grayscale).

        Returns:
            tuple: (visualization_bgr, defects). Defect types:
              'stripe_misalignment' (kind='stitch'): abrupt edge step
                  {y, x, x_delta, step_px, edges, threshold}
              'roll_error': gradual drift across a segment
                  {y0, y1, drift_px, slope_px_per_1k_rows, threshold}
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        profiles = self._extract_edge_profiles(gray)
        if profiles is None:
            print("StripeMisalignment: no stripe/paper contrast found")
            return (image.copy() if image.ndim == 3
                    else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)), []
        left, right, mid = profiles
        self._last_profiles = (left, right)

        stitches = self._detect_stitches(left, right)
        rolls = self._detect_roll(left, right,
                                  [s['y'] for s in stitches], len(left))

        defects = []
        for s in stitches:
            y = s['y']
            x = int(np.nanmedian(left[max(0, y - 200):y + 200]))
            defects.append({
                'type': 'stripe_misalignment',
                'kind': 'stitch',
                'y': int(y),
                'x': x,
                'x_delta': abs(round(s['step'], 1)),
                'step_px': round(s['step'], 1),
                'edges': s['edges'],
                'location': (x, int(y)),
                'threshold': self.step_threshold,
            })
        for r in rolls:
            yc = (r['y0'] + r['y1']) // 2
            x = int(np.nanmedian(left[max(0, yc - 200):yc + 200]))
            defects.append({
                'type': 'roll_error',
                'y0': r['y0'], 'y1': r['y1'],
                'drift_px': round(r['drift_px'], 1),
                'slope_px_per_1k_rows': round(r['slope_px_per_1k_rows'], 2),
                'location': (x, yc),
                'threshold': self.roll_threshold,
            })

        if self.debug:
            print(f"StripeMisalignment: {len(stitches)} stitch step(s), "
                  f"{len(rolls)} roll segment(s)")
            for s in stitches:
                print(f"  stitch at y={s['y']}: {s['step']:+.1f}px "
                      f"({'+'.join(s['edges'])})")
            for r in rolls:
                print(f"  roll y=[{r['y0']}, {r['y1']}]: "
                      f"{r['drift_px']:+.1f}px total drift")

        visualization = self.create_visualization(image, defects, left, right)
        return visualization, defects

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def create_visualization(self, original, defects, left, right):
        """Draw edge trajectories, stitch markers, and roll annotations."""
        if len(original.shape) == 2:
            vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            vis = original.copy()
        height, width = vis.shape[:2]

        # Edge trajectories (green, thin) so misalignment is visible at a glance
        step = max(1, height // 4000)
        for prof in (left, right):
            ys = np.arange(0, height, step)
            xs = prof[ys]
            valid = ~np.isnan(xs)
            pts = np.stack([xs[valid], ys[valid]], axis=1).astype(np.int32)
            if len(pts) > 1:
                cv2.polylines(vis, [pts.reshape(-1, 1, 2)], False, (0, 200, 0), 2)

        for d in defects:
            if d['type'] == 'stripe_misalignment':
                y = d['y']
                cv2.line(vis, (0, y), (width - 1, y), (0, 0, 255), 4)
                cv2.putText(vis, f"STITCH {d['step_px']:+.1f}px",
                            (10, max(40, y - 15)), cv2.FONT_HERSHEY_SIMPLEX,
                            1.4, (0, 0, 255), 3)
            else:  # roll_error
                y0, y1 = d['y0'], d['y1']
                x = min(width - 220, d['location'][0] + 60)
                cv2.arrowedLine(vis, (x, y0 + 100), (x, y1 - 100),
                                (0, 128, 255), 3, tipLength=0.02)
                cv2.putText(vis, f"ROLL {d['drift_px']:+.1f}px",
                            (x + 15, (y0 + y1) // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 128, 255), 3)

        n_stitch = sum(1 for d in defects if d['type'] == 'stripe_misalignment')
        n_roll = len(defects) - n_stitch
        cv2.putText(vis, f"Stitch steps: {n_stitch} | Roll errors: {n_roll} "
                    f"(step thr {self.step_threshold}px, "
                    f"roll thr {self.roll_threshold}px)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        if self.debug:
            self._debug_edge_image = self._plot_profiles(left, right, height)
        return vis

    @staticmethod
    def _plot_profiles(left, right, height):
        """Render the edge profiles as a simple plot image for debugging."""
        plot_h, plot_w = 600, 1200
        img = np.full((plot_h, plot_w, 3), 255, dtype=np.uint8)
        for prof, color in ((left, (200, 0, 0)), (right, (0, 0, 200))):
            valid = ~np.isnan(prof)
            if valid.sum() < 2:
                continue
            vmin, vmax = np.nanmin(prof), np.nanmax(prof)
            span = max(4.0, vmax - vmin)
            ys = np.linspace(0, height - 1, plot_w).astype(int)
            xs = prof[ys]
            ok = ~np.isnan(xs)
            px = np.flatnonzero(ok)
            py = ((xs[ok] - vmin) / span * (plot_h - 40) + 20).astype(int)
            pts = np.stack([px, plot_h - 1 - py], axis=1).astype(np.int32)
            if len(pts) > 1:
                cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, color, 1)
        return img

    def save_debug_images(self, output_dir, base_name):
        """Save the edge-profile plot when debug mode is enabled."""
        if self.debug and self._debug_edge_image is not None:
            return save_image(output_dir, base_name, self._debug_edge_image,
                              'edge_profiles')
        return None
