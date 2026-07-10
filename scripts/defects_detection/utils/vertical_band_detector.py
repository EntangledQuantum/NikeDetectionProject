"""
Vertical Band Detector for New-Pattern Island Images

The new island pattern places two horizontal-print bands side by side,
separated by a white gap. Each band is flanked by a pair of vertical
boundary lines:

    [V prints V]   gap   [V prints V]

This utility auto-detects the vertical boundary lines and the print bands
from a column ink-density profile (no hard-coded coordinates). It returns
one entry per band with:

  - crop bounds ``x0``/``x1`` (the inner print region, vertical lines excluded)
  - the full band extent ``outer_x0``/``outer_x1``
  - the detected vertical-line center x positions ``vline_xs`` (full-image x)

The vertical lines themselves are structural only and are never inspected for
defects; ``paint_vertical_lines_white`` is provided so any vertical line that
survives inside a crop can be painted out before debris/overspray thresholding.
"""

import cv2
import numpy as np


def _runs_from_mask(mask, min_gap=0, min_len=1):
    """Return contiguous True runs in a 1D boolean mask as (start, end_inclusive).

    Args:
        mask: 1D boolean numpy array.
        min_gap: Runs separated by a False gap shorter than this are merged.
        min_len: Runs shorter than this (after merging) are discarded.

    Returns:
        List of (start, end_inclusive) tuples.
    """
    runs = []
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))

    if not runs:
        return runs

    # Merge runs separated by small gaps
    merged = [runs[0]]
    for s, e in runs[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e - 1 <= min_gap:
            merged[-1] = (prev_s, e)
        else:
            merged.append((s, e))

    # Discard runs that are too short
    return [(s, e) for (s, e) in merged if (e - s + 1) >= min_len]


class VerticalBandDetector:
    """Detect print bands and their vertical boundary lines in an island image."""

    def __init__(self,
                 vline_threshold=0.45,
                 content_threshold=0.006,
                 smooth_fraction=0.004,
                 min_band_fraction=0.03,
                 min_gap_fraction=0.015,
                 inner_margin=3):
        """Configure band detection thresholds.

        Args:
            vline_threshold: Column ink fraction above which a column is treated
                as part of a continuous vertical boundary line.
            content_threshold: Column ink fraction above which a column is
                treated as belonging to a print band (dashed lines are sparse).
            smooth_fraction: Box-smoothing window as a fraction of image width.
            min_band_fraction: Minimum band width as a fraction of image width.
            min_gap_fraction: Gaps between content runs narrower than this
                fraction of width are merged into a single band.
            inner_margin: Extra pixels trimmed inside each vertical line when
                computing the print crop bounds.
        """
        self.vline_threshold = vline_threshold
        self.content_threshold = content_threshold
        self.smooth_fraction = smooth_fraction
        self.min_band_fraction = min_band_fraction
        self.min_gap_fraction = min_gap_fraction
        self.inner_margin = inner_margin

    def _column_ink_fraction(self, image):
        """Compute a smoothed per-column ink fraction (dark pixels / height)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        height, width = gray.shape
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        col_frac = (binary > 0).sum(axis=0).astype(np.float64) / max(1, height)

        # Smooth to suppress single-column noise while keeping structure
        window = max(1, int(width * self.smooth_fraction))
        if window > 1:
            kernel = np.ones(window, dtype=np.float64) / window
            col_frac = np.convolve(col_frac, kernel, mode='same')

        return col_frac

    def _detect_vlines_in_run(self, col_frac, run_start, run_end):
        """Find vertical-line centers/thickness within a band run.

        Returns a list of dicts with 'center' and 'thickness' in full-image x.
        """
        segment = col_frac[run_start:run_end + 1]
        vline_mask = segment > self.vline_threshold
        vline_runs = _runs_from_mask(vline_mask, min_gap=1, min_len=1)

        vlines = []
        for s, e in vline_runs:
            center = run_start + (s + e) // 2
            thickness = e - s + 1
            vlines.append({'center': int(center), 'thickness': int(thickness)})
        return vlines

    def detect(self, image, debug=False):
        """Detect print bands with their vertical boundary lines.

        Args:
            image: Input island image (BGR or grayscale).
            debug: If True, print a short summary of what was found.

        Returns:
            List of band dicts sorted left-to-right. Each dict has keys:
              'index', 'x0', 'x1' (inner print crop bounds),
              'outer_x0', 'outer_x1' (full band extent), and
              'vline_xs' (list of vertical-line center x positions).
        """
        if len(image.shape) == 3:
            width = image.shape[1]
        else:
            width = image.shape[1]

        col_frac = self._column_ink_fraction(image)

        min_gap = max(1, int(width * self.min_gap_fraction))
        min_len = max(1, int(width * self.min_band_fraction))

        content_mask = col_frac > self.content_threshold
        band_runs = _runs_from_mask(content_mask, min_gap=min_gap, min_len=min_len)

        bands = []

        if not band_runs:
            # Fallback: could not isolate bands; degrade to a single full-width band
            if debug:
                print("VerticalBandDetector: no bands found, using full image as one band")
            bands.append(self._make_band(0, 0, width - 1, []))
            return bands

        for outer_x0, outer_x1 in band_runs:
            vlines = self._detect_vlines_in_run(col_frac, outer_x0, outer_x1)
            bands.append(self._make_band_from_run(outer_x0, outer_x1, vlines))

        # Assign left-to-right indices
        bands.sort(key=lambda b: b['outer_x0'])
        for i, band in enumerate(bands):
            band['index'] = i

        if debug:
            print(f"VerticalBandDetector: {len(bands)} band(s) detected "
                  f"(width={width})")
            for band in bands:
                print(f"  Band {band['index']}: outer=[{band['outer_x0']}, "
                      f"{band['outer_x1']}], crop=[{band['x0']}, {band['x1']}], "
                      f"vlines={band['vline_xs']}")

        return bands

    def _make_band_from_run(self, outer_x0, outer_x1, vlines):
        """Build a band dict, trimming the crop inside the boundary vertical lines."""
        vline_xs = [v['center'] for v in vlines]

        if len(vlines) >= 2:
            left_v = min(vlines, key=lambda v: v['center'])
            right_v = max(vlines, key=lambda v: v['center'])
            inner_x0 = left_v['center'] + left_v['thickness'] // 2 + self.inner_margin
            inner_x1 = right_v['center'] - right_v['thickness'] // 2 - self.inner_margin
        elif len(vlines) == 1:
            # Only one boundary line found: trim just that side, keep the rest
            only_v = vlines[0]
            run_mid = (outer_x0 + outer_x1) // 2
            if only_v['center'] <= run_mid:
                inner_x0 = only_v['center'] + only_v['thickness'] // 2 + self.inner_margin
                inner_x1 = outer_x1
            else:
                inner_x0 = outer_x0
                inner_x1 = only_v['center'] - only_v['thickness'] // 2 - self.inner_margin
        else:
            inner_x0 = outer_x0
            inner_x1 = outer_x1

        # Guard against degenerate crops
        if inner_x1 <= inner_x0:
            inner_x0, inner_x1 = outer_x0, outer_x1

        return self._make_band(inner_x0, outer_x0, outer_x1, vline_xs, inner_x1=inner_x1)

    def _make_band(self, inner_x0, outer_x0, outer_x1, vline_xs, inner_x1=None):
        """Construct a normalized band dict."""
        if inner_x1 is None:
            inner_x1 = outer_x1
        return {
            'index': 0,
            'x0': int(inner_x0),
            'x1': int(inner_x1),
            'outer_x0': int(outer_x0),
            'outer_x1': int(outer_x1),
            'vline_xs': [int(x) for x in vline_xs],
        }


def paint_vertical_lines_white(gray_image, vline_xs, thickness):
    """Paint vertical boundary lines white so they are never counted as defects.

    Args:
        gray_image: Grayscale image to modify (a copy is returned).
        vline_xs: Iterable of vertical-line center x positions (image coords).
        thickness: Full width in pixels to paint around each center.

    Returns:
        Grayscale image copy with the vertical lines painted white (255).
    """
    painted = gray_image.copy()
    height = painted.shape[0]
    half = max(1, thickness // 2)
    for x in vline_xs:
        x1 = max(0, int(x) - half)
        x2 = min(painted.shape[1], int(x) + half + 1)
        if x2 > x1:
            painted[0:height, x1:x2] = 255
    return painted
