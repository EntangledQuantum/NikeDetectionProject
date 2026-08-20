"""
Band Line Refiner for New-Pattern Island Images

Refines the coarse horizontal-line matches produced by ``BandLineDetector``
into full-width line trajectories for a single print band.

Why this exists: the coarse detector reports one left endpoint and one right
endpoint per line, found by scanning a few kernel columns inward from each
band edge. When ink is missing at a band edge (missing nozzles at the start
or end of a line) the coarse endpoint sits wherever ink was first seen, so a
defect scan between the coarse endpoints silently skips the missing edge
segment. This refiner:

  1. Samples the actual ink along each line across the full band width
     (center-of-mass of ink inside a vertical search window that tracks the
     line, staying below half the line spacing so a neighbor line is never
     grabbed).
  2. Fits a straight line per trajectory with iterative sigma-clipping
     (robust to debris/overspray outliers). Each line gets its OWN slope and
     intercept - constant slope/spacing across lines is NOT assumed, so
     calibration/printing drift does not break the fit.
  3. Extrapolates every fit to the band's inner bounds ``x0..x1`` (just
     inside the vertical boundary lines). Missing start/end ink no longer
     shortens the scanned span; the missing-nozzle scan will flag it.
  4. For lines with too little ink to fit (heavily damaged or fully missing),
     interpolates the trajectory from the nearest well-fitted neighbor lines
     above and below, so the defect scan still walks the expected path and
     reports the gap.

All coordinates are band-local (relative to the crop given to
``BandLineDetector``); callers offset by the band's ``x0`` as usual.
"""

import numpy as np


class BandLineRefiner:
    """Refine coarse line matches into full-band-width fitted trajectories."""

    def __init__(self,
                 step_fraction=0.01,
                 min_step=8,
                 search_half_height_fraction=0.35,
                 min_fit_samples=8,
                 min_fit_span_fraction=0.25,
                 clip_sigma=2.5,
                 clip_iterations=3,
                 dedupe_fraction=0.5):
        """Configure trajectory sampling and fitting.

        Args:
            step_fraction: Horizontal sampling step as a fraction of the band
                width.
            min_step: Minimum sampling step in pixels.
            search_half_height_fraction: Half-height of the vertical ink
                search window as a fraction of the local line spacing. Must
                stay below 0.5 so the window can never reach a neighbor line.
            min_fit_samples: Minimum number of ink samples required to fit a
                line; below this the trajectory is interpolated from
                neighbors instead.
            min_fit_span_fraction: Samples must span at least this fraction
                of the band width for the fit to be trusted (a cluster of
                samples on one side extrapolates poorly).
            clip_sigma: Sigma-clipping threshold for outlier rejection.
            clip_iterations: Number of sigma-clipping refit passes.
            dedupe_fraction: Two refined lines whose mid-band y positions are
                closer than this fraction of the median spacing are treated
                as duplicates of the same physical line.
        """
        self.step_fraction = step_fraction
        self.min_step = min_step
        self.search_half_height_fraction = search_half_height_fraction
        self.min_fit_samples = min_fit_samples
        self.min_fit_span_fraction = min_fit_span_fraction
        self.clip_sigma = clip_sigma
        self.clip_iterations = clip_iterations
        self.dedupe_fraction = dedupe_fraction

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refine(self, binary_crop, matched_lines, debug=False):
        """Refine coarse matches into full-width trajectories.

        Args:
            binary_crop: Binary band crop (ink = white, from
                ``THRESH_BINARY_INV``), band-local coordinates.
            matched_lines: Matched line dicts from ``BandLineDetector``
                (each with 'left'/'right' endpoint dicts).
            debug: If True, print per-line refinement info.

        Returns:
            List of refined line dicts sorted top-to-bottom:
              {
                'left':  {'x': x0, 'y': y_at_x0},   # extrapolated start point
                'right': {'x': x1, 'y': y_at_x1},   # extrapolated end point
                'slope': float, 'intercept': float,
                'n_samples': int,
                'sample_x0': int, 'sample_x1': int,  # actual ink extent seen
                'source': 'fitted' | 'interpolated',
              }
            The 'left'/'right' layout is compatible with
            ``LineDefectDetector.scan_line_for_defects``.
        """
        height, width = binary_crop.shape
        if width < 4:
            return []

        if not matched_lines:
            # Coarse detection failed entirely (e.g. heavily damaged band).
            # Seed trajectories from the band's row ink profile instead.
            matched_lines = self._seed_from_row_profile(binary_crop, debug)
            if not matched_lines:
                return []

        spacing = self._estimate_spacing(matched_lines, height)

        # Sample and fit each coarse trajectory independently
        fitted = []
        for i, match in enumerate(matched_lines):
            entry = self._refine_single(binary_crop, match, spacing, width)
            entry['coarse_index'] = i
            fitted.append(entry)

        # Fill unfittable lines from their fitted neighbors
        self._interpolate_unfitted(fitted, spacing, debug)

        # Drop entries that could not be resolved at all
        refined = [f for f in fitted if f.get('slope') is not None]

        # Deduplicate trajectories that converged onto the same physical line
        refined = self._dedupe(refined, spacing, width)

        # Materialize extrapolated endpoints across the full band width
        x0, x1 = 0, width - 1
        results = []
        for entry in refined:
            slope, intercept = entry['slope'], entry['intercept']
            y_left = intercept
            y_right = slope * (x1 - x0) + intercept
            if not (-spacing <= y_left <= height - 1 + spacing):
                continue
            results.append({
                'left': {'x': int(x0), 'y': int(round(y_left)), 'type': 'refined'},
                'right': {'x': int(x1), 'y': int(round(y_right)), 'type': 'refined'},
                'slope': float(slope),
                'intercept': float(intercept),
                'n_samples': int(entry['n_samples']),
                'sample_x0': int(entry['sample_x0']),
                'sample_x1': int(entry['sample_x1']),
                'source': entry['source'],
            })

        results.sort(key=lambda r: r['left']['y'])

        if debug:
            n_fit = len([r for r in results if r['source'] == 'fitted'])
            n_interp = len(results) - n_fit
            print(f"BandLineRefiner: {len(results)} lines "
                  f"({n_fit} fitted, {n_interp} interpolated), "
                  f"spacing~{spacing:.1f}px")

        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _seed_from_row_profile(self, binary_crop, debug=False):
        """Build pseudo coarse matches from the band's row ink profile.

        Used when the kernel-based coarse detector found nothing. Each run of
        inked rows becomes one flat left/right match that the sampling + fit
        stage then refines into a true trajectory.
        """
        height, width = binary_crop.shape
        row_frac = (binary_crop > 0).sum(axis=1).astype(np.float64) / max(1, width)
        peak = float(row_frac.max())
        if peak <= 0:
            return []

        active = row_frac > max(0.02, 0.25 * peak)
        runs = []
        start = None
        for i, val in enumerate(active):
            if val and start is None:
                start = i
            elif not val and start is not None:
                runs.append((start, i - 1))
                start = None
        if start is not None:
            runs.append((start, height - 1))

        matches = []
        for s, e in runs:
            y = (s + e) // 2
            matches.append({
                'left': {'x': 0, 'y': y, 'type': 'seed'},
                'right': {'x': width - 1, 'y': y, 'type': 'seed'},
                'valid_slope': False,
            })

        if debug and matches:
            print(f"BandLineRefiner: seeded {len(matches)} trajectories "
                  f"from row profile (no coarse matches)")
        return matches

    def _estimate_spacing(self, matched_lines, height):
        """Median y-spacing between consecutive coarse lines (left side)."""
        ys = sorted(m['left']['y'] for m in matched_lines)
        deltas = [b - a for a, b in zip(ys, ys[1:]) if b - a > 2]
        if deltas:
            return float(np.median(deltas))
        # Single line: fall back to ideal-geometry scaling
        return max(10.0, 100.0 * height / 44228.0)

    def _refine_single(self, binary_crop, match, spacing, width):
        """Sample ink along one coarse trajectory and fit a line to it."""
        left, right = match['left'], match['right']
        x_l, y_l = left['x'], left['y']
        x_r, y_r = right['x'], right['y']

        # Coarse guide slope for the search window path. For ghost/invalid
        # matches, a flat guide through the known-real endpoint (or midpoint)
        # is safer than a wild cross-endpoint slope.
        guide_slope = 0.0
        if x_r > x_l:
            candidate = (y_r - y_l) / float(x_r - x_l)
            # Only trust the coarse slope when it is physically plausible
            # (never steeper than half a spacing over the band width)
            if abs(candidate) * width <= 0.5 * spacing or match.get('valid_slope', False):
                guide_slope = candidate

        if left.get('type') == 'ghost' and right.get('type') != 'ghost':
            anchor_x, anchor_y = x_r, y_r
        else:
            anchor_x, anchor_y = x_l, y_l

        half_h = max(3, int(round(spacing * self.search_half_height_fraction)))
        step = max(self.min_step, int(round(width * self.step_fraction)))
        height = binary_crop.shape[0]

        xs, ys = [], []
        for x in range(0, width, step):
            y_expected = anchor_y + guide_slope * (x - anchor_x)
            yc = int(round(y_expected))
            win_y0 = max(0, yc - half_h)
            win_y1 = min(height, yc + half_h + 1)
            if win_y1 <= win_y0:
                continue
            column = binary_crop[win_y0:win_y1, x:min(width, x + step)]
            ink_rows, _ = np.nonzero(column)
            if ink_rows.size == 0:
                continue
            xs.append(x + min(step, width - x) / 2.0)
            ys.append(win_y0 + float(ink_rows.mean()))

        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)

        entry = {
            'slope': None,
            'intercept': None,
            'n_samples': int(xs.size),
            'sample_x0': int(xs.min()) if xs.size else -1,
            'sample_x1': int(xs.max()) if xs.size else -1,
            'source': 'fitted',
            'anchor_y': float(anchor_y),
        }

        span_ok = (xs.size >= 2 and
                   (xs.max() - xs.min()) >= self.min_fit_span_fraction * width)
        if xs.size < self.min_fit_samples or not span_ok:
            return entry  # left unfitted; neighbor interpolation will fill it

        slope, intercept = self._robust_fit(xs, ys)
        entry['slope'] = float(slope)
        entry['intercept'] = float(intercept)
        return entry

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
            if sigma < 0.5:
                break
            new_keep = np.abs(residuals) <= self.clip_sigma * sigma
            if new_keep.sum() < max(2, int(0.5 * xs.size)):
                break
            if np.array_equal(new_keep, keep):
                break
            keep = new_keep
        return float(slope), float(intercept)

    def _interpolate_unfitted(self, fitted, spacing, debug=False):
        """Fill unfittable trajectories from the nearest fitted neighbors.

        Neighbors are the closest fitted lines above and below (by coarse
        index order, which follows the top-to-bottom scan). The interpolated
        slope is the neighbor average; the intercept is placed at the
        proportional position between the neighbor intercepts. With only one
        fitted neighbor, its slope is reused and the intercept is offset by
        the local spacing.
        """
        fitted_idx = [i for i, f in enumerate(fitted) if f['slope'] is not None]
        if not fitted_idx:
            return

        for i, entry in enumerate(fitted):
            if entry['slope'] is not None:
                continue

            below = [j for j in fitted_idx if j < i]
            above = [j for j in fitted_idx if j > i]

            if below and above:
                j0, j1 = below[-1], above[0]
                f0, f1 = fitted[j0], fitted[j1]
                t = (i - j0) / float(j1 - j0)
                entry['slope'] = f0['slope'] + t * (f1['slope'] - f0['slope'])
                entry['intercept'] = (f0['intercept'] +
                                      t * (f1['intercept'] - f0['intercept']))
            elif below:
                j0 = below[-1]
                f0 = fitted[j0]
                entry['slope'] = f0['slope']
                entry['intercept'] = f0['intercept'] + spacing * (i - j0)
            else:
                j1 = above[0]
                f1 = fitted[j1]
                entry['slope'] = f1['slope']
                entry['intercept'] = f1['intercept'] - spacing * (j1 - i)

            entry['source'] = 'interpolated'
            if debug:
                print(f"BandLineRefiner: line {i} interpolated "
                      f"(only {entry['n_samples']} ink samples)")

    def _dedupe(self, refined, spacing, width):
        """Merge refined lines that resolved to the same physical trajectory."""
        if len(refined) < 2:
            return refined

        mid_x = width / 2.0
        refined = sorted(refined,
                         key=lambda f: f['slope'] * mid_x + f['intercept'])
        threshold = self.dedupe_fraction * spacing

        result = [refined[0]]
        for entry in refined[1:]:
            y_mid = entry['slope'] * mid_x + entry['intercept']
            prev = result[-1]
            prev_y_mid = prev['slope'] * mid_x + prev['intercept']
            if abs(y_mid - prev_y_mid) < threshold:
                # Keep whichever has more supporting ink samples
                if entry['n_samples'] > prev['n_samples']:
                    result[-1] = entry
            else:
                result.append(entry)
        return result
