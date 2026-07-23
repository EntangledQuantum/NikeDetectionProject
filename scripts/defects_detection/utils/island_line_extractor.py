"""
Island Line Extractor for New-Pattern Island Bands

Robustly extracts EVERY slanted horizontal print line inside one band crop
(the region between a pair of vertical boundary lines) and produces
column-resolution ink statistics per line. This is the precursor for the
missing-nozzle / misalignment evaluation.

Why not kernel scanning: the legacy approach walks small kernels along a
guessed trajectory and thresholds the ink ratio inside each kernel. On the
stippled print (dotted lines, thickness ~10 px) that produces mass false
positives whenever the trajectory is a few pixels off or the stipple is
locally sparse. This extractor instead:

  1. Binarizes with an Otsu-derived ink threshold (clamped to a sane range).
  2. Finds the global line slope by a shear search: for candidate slopes the
     image columns are shifted so lines become horizontal, and the slope that
     maximizes the sharpness (sum of squares) of the row ink profile wins.
     Two-stage search (coarse then fine), no reliance on calibration.
  3. Shears the binary image by the winning slope (exact per-column shifts).
     Every print line now occupies a narrow horizontal row band.
  4. Detects line rows as peaks (runs) of the sheared row profile; the median
     peak distance gives the line spacing. Rows where a whole line is missing
     (gap ~2x spacing between peaks) are inserted so fully-dead lines are
     still evaluated.
  5. Per line, slices a window (+-0.4 spacing) around its row, fits the
     residual slope/intercept from per-column ink centroids (sigma-clipped
     least squares - each line gets its own correction, so per-line
     calibration drift is handled), then computes per-column statistics
     inside a tight corridor around the fitted center:
       - ink presence / ink pixel count
       - number of separate ink runs (2+ = the line splits in two)
       - ink extent minus ink count (large = hollow/split line)
       - centroid deviation from the fit
  6. Maps everything back to original (unsheared) image coordinates via the
     stored per-column shift.

The returned structure is geometry + raw per-column evidence; deciding what
constitutes a defect (gap length thresholds, split persistence, etc.) is the
caller's job.
"""

import cv2
import numpy as np


class IslandLineExtractor:
    """Extract all print lines of a band crop with per-column ink statistics."""

    def __init__(self,
                 ink_threshold=None,
                 min_ink_threshold=120,
                 max_ink_threshold=200,
                 block_width=32,
                 slope_min=-0.005,
                 slope_max=0.035,
                 coarse_slope_step=0.002,
                 fine_slope_step=0.0002,
                 peak_rel_threshold=0.25,
                 window_half_fraction=0.40,
                 corridor_half_fraction=0.30,
                 missing_line_gap_factor=1.6,
                 clip_sigma=2.5,
                 clip_iterations=3,
                 clear=False,
                 clear_min_ink_offset=15,
                 clear_max_ink_offset=85):
        """Configure line extraction.

        Args:
            ink_threshold: Fixed grayscale ink threshold. None = Otsu, clamped
                to [min_ink_threshold, max_ink_threshold].
            min_ink_threshold: Lower clamp for the automatic threshold.
            max_ink_threshold: Upper clamp for the automatic threshold.
            block_width: Column block width used for the fast shear-profile
                slope search (slope * block_width stays < 1 px).
            slope_min: Lower bound of the slope search (y per x, image coords).
            slope_max: Upper bound of the slope search.
            coarse_slope_step: Step of the coarse slope sweep.
            fine_slope_step: Step of the refinement sweep around the best
                coarse slope.
            peak_rel_threshold: Row-profile runs above this fraction of the
                99th-percentile profile value are treated as line rows.
            window_half_fraction: Per-line analysis window half-height as a
                fraction of the measured line spacing.
            corridor_half_fraction: Tight corridor half-height (fraction of
                spacing) around the fitted center used for the final
                per-column statistics.
            missing_line_gap_factor: Peak gaps larger than this multiple of
                the spacing get synthetic (inserted) lines so fully missing
                lines are still evaluated.
            clip_sigma: Sigma-clipping threshold for the residual line fit.
            clip_iterations: Sigma-clipping refit passes.
            clear: If True, the scan uses the clear material (gray background,
                fainter ink, lower SNR). The crop is despeckled with a median
                blur and the automatic ink threshold is clamped relative to
                the measured background level instead of the fixed
                white-paper range.
            clear_min_ink_offset: Clear mode: the ink threshold stays at
                least this far below the background level (keeps background
                noise out of the binary mask).
            clear_max_ink_offset: Clear mode: the ink threshold never drops
                further than this below the background level (keeps faint
                ink detectable).
        """
        self.ink_threshold = ink_threshold
        self.min_ink_threshold = min_ink_threshold
        self.max_ink_threshold = max_ink_threshold
        self.block_width = block_width
        self.slope_min = slope_min
        self.slope_max = slope_max
        self.coarse_slope_step = coarse_slope_step
        self.fine_slope_step = fine_slope_step
        self.peak_rel_threshold = peak_rel_threshold
        self.window_half_fraction = window_half_fraction
        self.corridor_half_fraction = corridor_half_fraction
        self.missing_line_gap_factor = missing_line_gap_factor
        self.clip_sigma = clip_sigma
        self.clip_iterations = clip_iterations
        self.clear = clear
        self.clear_min_ink_offset = clear_min_ink_offset
        self.clear_max_ink_offset = clear_max_ink_offset

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, gray_band, debug=False):
        """Extract all print lines from a grayscale band crop.

        Args:
            gray_band: Grayscale band crop (uint8), boundary vertical lines
                already excluded by the caller's crop bounds.
            debug: If True, print stage summaries.

        Returns:
            Dict with:
              'slope':      global line slope (y per x)
              'spacing':    measured line spacing (px)
              'threshold':  ink threshold used
              'shift':      per-column shear shift (int array, len W);
                            true_y(x) = sheared_y + shift[x]
              'lines':      list of per-line dicts sorted top-to-bottom:
                 'row'            sheared row center (float)
                 'res_slope'      residual slope on top of the global slope
                 'res_intercept'  residual intercept at x=0 (sheared coords)
                 'inserted'       True if synthesized for a missing peak
                 'thickness'      median ink thickness (px)
                 'ink_cols'       bool[W]  ink present in corridor
                 'ink_count'      int16[W] ink pixels per column
                 'n_runs'         int16[W] separate ink runs per column
                 'hollow'         int16[W] extent minus ink count
                 'centroid_dev'   float32[W] centroid minus fit (NaN = no ink)
            Returns None when the band contains no usable ink.
        """
        height, width = gray_band.shape

        if self.clear:
            # Low SNR: kill salt-and-pepper speckle so isolated noise pixels
            # neither bridge real missing-nozzle gaps nor fake split lines.
            gray_band = cv2.medianBlur(gray_band, 3)

        threshold = self._resolve_threshold(gray_band)
        binary = (gray_band < threshold).astype(np.uint8)
        if int(binary.sum()) < width:  # essentially empty band
            return None

        slope = self._find_global_slope(binary, debug)
        shift = np.round(slope * np.arange(width)).astype(np.int32)
        sheared = self._shear(binary, shift)

        centers, spacing, profile = self._find_line_rows(sheared, debug)
        if not centers:
            return None

        centers = self._insert_missing_rows(centers, spacing, debug)

        lines = []
        for center, inserted in centers:
            line = self._analyze_line(sheared, center, spacing)
            line['inserted'] = inserted
            lines.append(line)

        if debug:
            n_ins = sum(1 for l in lines if l['inserted'])
            print(f"IslandLineExtractor: slope={slope:.5f} spacing={spacing:.1f}px "
                  f"threshold={threshold} lines={len(lines)} (+{n_ins} inserted)")

        return {
            'slope': float(slope),
            'spacing': float(spacing),
            'threshold': int(threshold),
            'shift': shift,
            'lines': lines,
        }

    @staticmethod
    def line_y(result, line, x):
        """True (unsheared) y of a line at column x (scalar or array)."""
        x = np.asarray(x)
        y_sheared = line['row'] + line['res_intercept'] + line['res_slope'] * x
        return y_sheared + result['shift'][np.clip(x, 0, len(result['shift']) - 1)]

    # ------------------------------------------------------------------
    # Stage 1: threshold + global slope
    # ------------------------------------------------------------------

    def _resolve_threshold(self, gray_band):
        """Otsu ink threshold (subsampled), clamped to a sane ink range.

        On white paper the clamp is the fixed [min_ink_threshold,
        max_ink_threshold] range. On the clear material the background is
        mid-gray, so the clamp is anchored to the measured background level
        instead: the threshold must sit below the background (or the whole
        background becomes ink) but not so far down that faint ink is lost.
        """
        if self.ink_threshold is not None:
            return int(self.ink_threshold)
        sub = gray_band[::4, ::4]
        otsu, _ = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if self.clear:
            background = float(np.median(sub))
            low = max(5.0, background - self.clear_max_ink_offset)
            high = max(low + 1.0, background - self.clear_min_ink_offset)
            return int(np.clip(otsu, low, high))
        return int(np.clip(otsu, self.min_ink_threshold, self.max_ink_threshold))

    def _find_global_slope(self, binary, debug=False):
        """Two-stage shear search maximizing row-profile sharpness."""
        height, width = binary.shape
        block = self.block_width
        n_blocks = (width + block - 1) // block

        # Row ink profile per column block (computed once)
        profiles = np.zeros((n_blocks, height), dtype=np.float64)
        centers = np.zeros(n_blocks)
        for b in range(n_blocks):
            x0, x1 = b * block, min(width, (b + 1) * block)
            profiles[b] = binary[:, x0:x1].sum(axis=1)
            centers[b] = (x0 + x1 - 1) / 2.0

        def score(s):
            total = np.zeros(height, dtype=np.float64)
            shifts = np.round(s * centers).astype(int)
            for b in range(n_blocks):
                sh = shifts[b]
                if sh >= 0:
                    if sh < height:
                        total[:height - sh] += profiles[b][sh:]
                else:
                    total[-sh:] += profiles[b][:height + sh]
            return float((total * total).sum())

        coarse = np.arange(self.slope_min, self.slope_max + 1e-9,
                           self.coarse_slope_step)
        best = max(coarse, key=score)
        fine = np.arange(best - self.coarse_slope_step,
                         best + self.coarse_slope_step + 1e-9,
                         self.fine_slope_step)
        best = max(fine, key=score)

        if debug:
            print(f"IslandLineExtractor: global slope search -> {best:.5f}")
        return float(best)

    @staticmethod
    def _shear(binary, shift):
        """Shear the binary image so lines become horizontal.

        sheared[y, x] = binary[y + shift[x], x]; rows shifted past the bottom
        are zero-filled. Grouped by unique shift value for speed.
        """
        height, width = binary.shape
        sheared = np.zeros_like(binary)
        for sh in np.unique(shift):
            cols = np.flatnonzero(shift == sh)
            if sh >= 0:
                if sh < height:
                    sheared[:height - sh, cols] = binary[sh:, cols]
            else:
                sheared[-sh:, cols] = binary[:height + sh, cols]
        return sheared

    # ------------------------------------------------------------------
    # Stage 2: line rows + spacing
    # ------------------------------------------------------------------

    def _find_line_rows(self, sheared, debug=False):
        """Detect line row centers and spacing from the sheared row profile."""
        profile = sheared.sum(axis=1).astype(np.float64)
        kernel = np.ones(5) / 5.0
        smooth = np.convolve(profile, kernel, mode='same')

        ref = np.percentile(smooth, 99)
        if ref <= 0:
            return [], 0.0, profile
        level = max(2.0, self.peak_rel_threshold * ref)

        active = smooth > level
        runs = []
        start = None
        for i, val in enumerate(active):
            if val and start is None:
                start = i
            elif not val and start is not None:
                runs.append((start, i - 1))
                start = None
        if start is not None:
            runs.append((start, len(active) - 1))

        # Merge runs separated by tiny dips, then compute weighted centers
        merged = []
        for s, e in runs:
            if merged and s - merged[-1][1] - 1 <= 3:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))

        centers = []
        for s, e in merged:
            seg = smooth[s:e + 1]
            idx = np.arange(s, e + 1)
            centers.append(float((idx * seg).sum() / seg.sum()))

        if len(centers) < 2:
            return [(c, False) for c in centers], 0.0, profile

        deltas = np.diff(centers)
        spacing = float(np.median(deltas))

        if debug:
            print(f"IslandLineExtractor: {len(centers)} line rows, "
                  f"spacing={spacing:.1f}px (profile ref={ref:.0f})")

        return [(c, False) for c in centers], spacing, profile

    def _insert_missing_rows(self, centers, spacing, debug=False):
        """Insert synthetic rows where a whole line appears to be missing."""
        if spacing <= 0 or len(centers) < 2:
            return centers

        out = [centers[0]]
        for (c, ins) in centers[1:]:
            prev_c = out[-1][0]
            gap = c - prev_c
            if gap > self.missing_line_gap_factor * spacing:
                n_missing = int(round(gap / spacing)) - 1
                for k in range(1, n_missing + 1):
                    out.append((prev_c + gap * k / (n_missing + 1), True))
                if debug:
                    print(f"IslandLineExtractor: inserted {n_missing} missing "
                          f"line(s) between rows {prev_c:.0f} and {c:.0f}")
            out.append((c, ins))
        return out

    # ------------------------------------------------------------------
    # Stage 3: per-line analysis
    # ------------------------------------------------------------------

    def _analyze_line(self, sheared, center, spacing):
        """Fit one line's residual trajectory and gather per-column stats."""
        height, width = sheared.shape
        half = max(4, int(round(spacing * self.window_half_fraction))) \
            if spacing > 0 else 8

        y0 = max(0, int(round(center)) - half)
        y1 = min(height, int(round(center)) + half + 1)
        window = sheared[y0:y1].astype(bool)
        rows_rel = (np.arange(y0, y1) - center).astype(np.float32)

        col_count = window.sum(axis=0)
        has_ink = col_count > 0
        xs = np.flatnonzero(has_ink)

        # Per-column ink centroid relative to the row center
        centroid = np.full(width, np.nan, dtype=np.float32)
        if xs.size:
            centroid[xs] = (rows_rel @ window[:, xs]) / col_count[xs]

        res_slope, res_intercept = 0.0, 0.0
        if xs.size >= max(20, width // 100):
            res_slope, res_intercept = self._robust_fit(
                xs.astype(np.float64), centroid[xs].astype(np.float64))
            # A residual should be a small correction; reject wild fits
            if abs(res_slope) * width > spacing:
                res_slope, res_intercept = 0.0, float(np.nanmedian(centroid))

        # Tight corridor around the fitted center for the final statistics
        corridor = max(3, int(round(spacing * self.corridor_half_fraction))) \
            if spacing > 0 else 6
        fit_rel = (res_intercept + res_slope * np.arange(width)).astype(np.float32)
        dist = np.abs(rows_rel[:, None] - fit_rel[None, :])
        in_corridor = window & (dist <= corridor)

        ink_count = in_corridor.sum(axis=0).astype(np.int16)
        ink_cols = ink_count > 0

        # Number of separate ink runs per column (2+ = split line)
        starts = in_corridor[0].astype(np.int16) + \
            (in_corridor[1:] & ~in_corridor[:-1]).sum(axis=0).astype(np.int16)

        # Vertical extent minus ink pixels (hollow = split / double line)
        n_rows = in_corridor.shape[0]
        first = np.where(ink_cols, in_corridor.argmax(axis=0), 0)
        last = np.where(ink_cols,
                        n_rows - 1 - in_corridor[::-1].argmax(axis=0), 0)
        extent = np.where(ink_cols, last - first + 1, 0)
        hollow = (extent - ink_count).astype(np.int16)

        # Centroid deviation from the fit (recomputed inside the corridor)
        centroid_dev = np.full(width, np.nan, dtype=np.float32)
        xs2 = np.flatnonzero(ink_cols)
        if xs2.size:
            cen = (rows_rel @ in_corridor[:, xs2]) / ink_count[xs2]
            centroid_dev[xs2] = cen - fit_rel[xs2]

        thickness = float(np.median(ink_count[xs2])) if xs2.size else 0.0

        return {
            'row': float(center),
            'res_slope': float(res_slope),
            'res_intercept': float(res_intercept),
            'thickness': thickness,
            'ink_cols': ink_cols,
            'ink_count': ink_count,
            'n_runs': starts,
            'hollow': hollow,
            'centroid_dev': centroid_dev,
        }

    def _robust_fit(self, xs, ys):
        """Least-squares line fit with iterative sigma-clipping."""
        keep = np.ones(xs.size, dtype=bool)
        slope, intercept = 0.0, float(np.median(ys))
        for _ in range(self.clip_iterations + 1):
            if keep.sum() < 2:
                break
            slope, intercept = np.polyfit(xs[keep], ys[keep], 1)
            residuals = ys - (slope * xs + intercept)
            sigma = residuals[keep].std()
            if sigma < 0.3:
                break
            new_keep = np.abs(residuals) <= self.clip_sigma * sigma
            if new_keep.sum() < max(2, int(0.5 * xs.size)):
                break
            if np.array_equal(new_keep, keep):
                break
            keep = new_keep
        return float(slope), float(intercept)
