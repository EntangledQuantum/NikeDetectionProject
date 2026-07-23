"""
Vertical Band Detector for New-Pattern Island Images

The new island pattern places two horizontal-print bands side by side,
separated by a white gap. Each band is flanked by a pair of vertical
boundary lines (4 vertical lines total):

    [V prints V]   gap   [V prints V]

This utility robustly auto-detects the vertical boundary lines and derives
the two print bands from them (no hard-coded coordinates). The vertical
lines can be noisy (speckled columns of dots), have imperfect start/end
points, and drift slightly across the image height, so detection works on a
morphologically extracted "vertical structure" mask rather than a raw
column-density profile:

  1. Binarize with a generous ink threshold (catches faint speckles).
  2. Vertical CLOSE with a kernel well below the horizontal line spacing:
     bridges the small gaps in a dotted vertical line without merging
     adjacent horizontal print lines.
  3. Vertical OPEN with a kernel well above the horizontal line thickness:
     removes every horizontal-line crossing, keeping only tall vertical
     structures.
  4. Horizontal dilation: tolerates skew/drift of the vertical line across
     columns over the full image height.
  5. Column coverage profile of the result -> candidate vertical lines
     (center, x-extent envelope, y-extent, coverage).
  6. Candidate selection: if more than 4 candidates survive, the best
     4-combination is chosen by coverage and pattern geometry
     (two similar wide bands separated by a narrower central gap).

Fallbacks: if exactly 4 lines cannot be selected, bands degrade to
ink-content runs (the previous behavior), and finally to a single
full-width band, so downstream detection always has something to work on.

The vertical lines themselves are structural only and are never inspected
for defects. ``paint_vertical_line_regions`` / ``paint_vertical_lines_white``
are provided so the vertical lines (and their speckle halo) can be painted
out before debris/overspray thresholding.
"""

import itertools

import cv2
import numpy as np

from utils.material_profile import estimate_background_level


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
    """Detect the 4 vertical boundary lines and the 2 print bands of a
    new-pattern island image."""

    # Reference geometry shared with LineDetector (ideal full-resolution island)
    IDEAL_IMAGE_HEIGHT = 44228.0
    IDEAL_LINE_SPACING = 100.0  # ~ (Y_DELTA_MIN + Y_DELTA_MAX) / 2 at ideal size

    def __init__(self,
                 binary_threshold=170,
                 content_binary_threshold=127,
                 min_coverage=0.25,
                 relative_coverage=0.45,
                 close_spacing_factor=0.45,
                 open_spacing_factor=1.6,
                 drift_dilate_fraction=0.004,
                 merge_gap_fraction=0.003,
                 max_thickness_fraction=0.05,
                 inner_margin=5,
                 content_threshold=0.006,
                 smooth_fraction=0.004,
                 min_band_fraction=0.03,
                 min_gap_fraction=0.015,
                 expected_spacing=None,
                 clear=False):
        """Configure vertical line / band detection.

        Args:
            binary_threshold: Grayscale value below which a pixel is treated
                as ink when building the vertical-structure mask. Generous
                (higher than the usual 127) so faint speckles of a noisy
                vertical line still contribute.
            content_binary_threshold: Ink threshold for the general content
                profile used by the fallback band segmentation.
            min_coverage: Minimum fraction of image height a candidate
                vertical line must cover (after morphology). Tolerates
                imperfect starting/ending points.
            relative_coverage: A candidate must also reach this fraction of
                the strongest candidate's coverage.
            close_spacing_factor: Vertical CLOSE kernel length as a fraction
                of the expected horizontal-line spacing. Must stay < 1 so
                adjacent horizontal lines are never merged vertically.
            open_spacing_factor: Vertical OPEN kernel length as a multiple of
                the expected horizontal-line spacing. Must be comfortably
                above the horizontal line thickness so all horizontal
                crossings are removed.
            drift_dilate_fraction: Horizontal dilation width (as a fraction
                of image width) applied before the coverage profile, so a
                vertical line that drifts across columns still produces a
                strong single peak.
            merge_gap_fraction: Candidate column runs separated by less than
                this fraction of the width are merged into one line.
            max_thickness_fraction: Candidates wider than this fraction of
                the image width are rejected (dark blobs, not lines).
            inner_margin: Extra pixels trimmed inside each vertical line when
                computing the print crop bounds.
            content_threshold: Column ink fraction above which a column is
                treated as belonging to a print band (fallback path).
            smooth_fraction: Box-smoothing window for the content profile as
                a fraction of image width (fallback path).
            min_band_fraction: Minimum band width as a fraction of the image
                width (fallback path).
            min_gap_fraction: Content-run gaps narrower than this fraction of
                the width are merged into a single band (fallback path).
            expected_spacing: Optional explicit horizontal-line spacing in
                pixels. When None it is scaled from the ideal geometry by the
                image height (same convention as ``LineDetector``).
            clear: If True, the scan uses the clear material (gray background,
                fainter ink, lower SNR). All ink thresholds are then derived
                from the measured background level of each image instead of
                the fixed white-paper values above.
        """
        self.binary_threshold = binary_threshold
        self.content_binary_threshold = content_binary_threshold
        self.min_coverage = min_coverage
        self.relative_coverage = relative_coverage
        self.close_spacing_factor = close_spacing_factor
        self.open_spacing_factor = open_spacing_factor
        self.drift_dilate_fraction = drift_dilate_fraction
        self.merge_gap_fraction = merge_gap_fraction
        self.max_thickness_fraction = max_thickness_fraction
        self.inner_margin = inner_margin
        self.content_threshold = content_threshold
        self.smooth_fraction = smooth_fraction
        self.min_band_fraction = min_band_fraction
        self.min_gap_fraction = min_gap_fraction
        self.expected_spacing = expected_spacing
        self.clear = clear

        # Populated on every detect() call
        self.last_vlines = []       # selected boundary lines (ideally 4)
        self.last_candidates = []   # all vertical-line candidates found
        self._background_level = None  # measured in clear mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image, debug=False):
        """Detect the print bands and their vertical boundary lines.

        Args:
            image: Input island image (BGR or grayscale).
            debug: If True, print a short summary of what was found.

        Returns:
            List of band dicts sorted left-to-right. Each dict has keys:
              'index', 'x0', 'x1' (inner print crop bounds),
              'outer_x0', 'outer_x1' (full band extent),
              'vline_xs' (vertical-line center x positions), and
              'vlines' (full vertical-line dicts, see ``_extract_candidates``).

        Side effects:
            ``self.last_vlines`` holds the selected boundary lines and
            ``self.last_candidates`` all raw candidates, both in full-image
            coordinates, for callers that need to mask the vertical lines.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if self.clear:
            # Clear material: gray background, fainter ink. Re-anchor the
            # fixed white-paper thresholds to the measured background so
            # the background itself is never binarized as ink.
            bg = estimate_background_level(gray)
            self._background_level = bg
            self.binary_threshold = int(np.clip(bg - 25, 5, 250))
            self.content_binary_threshold = int(np.clip(bg - 20, 5, 250))
            if debug:
                print(f"VerticalBandDetector: clear mode, background={bg:.0f}, "
                      f"ink threshold={self.binary_threshold}")

        height, width = gray.shape
        spacing = self._expected_line_spacing(gray)

        # ---- Step 1: robust vertical-line candidates -------------------
        # The boundary lines can be printed faintly (or scanned light), so if
        # the configured ink threshold yields fewer than 4 candidates, retry
        # with progressively more generous thresholds. Spurious extra
        # candidates are handled by the 4-of-N geometric selection below.
        candidates = []
        for threshold in self._threshold_ladder():
            candidates = self._detect_vline_candidates(gray, spacing, debug,
                                                       binary_threshold=threshold)
            if len(candidates) >= 4:
                break
            if debug and threshold != self._threshold_ladder()[-1]:
                print(f"VerticalBandDetector: only {len(candidates)} candidate(s) "
                      f"at threshold {threshold}, retrying more generously")
        vlines = self._select_boundary_lines(candidates, debug)

        self.last_candidates = candidates
        self.last_vlines = vlines

        # ---- Step 2: segregate the two print regions -------------------
        bands = None
        if len(vlines) == 4:
            bands = self._bands_from_vlines(vlines, width)
            if bands is None and debug:
                print("VerticalBandDetector: 4 vertical lines found but band "
                      "geometry was degenerate; falling back to content runs")

        if bands is None:
            bands = self._bands_from_content(gray, candidates, debug)

        # Assign left-to-right indices
        bands.sort(key=lambda b: b['outer_x0'])
        for i, band in enumerate(bands):
            band['index'] = i

        if debug:
            print(f"VerticalBandDetector: {len(vlines)}/4 boundary lines, "
                  f"{len(bands)} band(s) (image {width}x{height}, "
                  f"expected spacing {spacing:.1f}px)")
            for v in vlines:
                print(f"  VLine center={v['center']} x=[{v['x0']}, {v['x1']}] "
                      f"y=[{v['y0']}, {v['y1']}] coverage={v['coverage']:.2f}")
            for band in bands:
                print(f"  Band {band['index']}: outer=[{band['outer_x0']}, "
                      f"{band['outer_x1']}], crop=[{band['x0']}, {band['x1']}], "
                      f"vlines={band['vline_xs']}")

        return bands

    # ------------------------------------------------------------------
    # Vertical line candidates
    # ------------------------------------------------------------------

    def _expected_line_spacing(self, gray):
        """Horizontal-line spacing in pixels, measured from the image itself.

        Uses the median distance between ink-row runs of the row profile so
        the value adapts to the actual print (no reliance on perfect
        calibration). Falls back to ideal-geometry scaling when the profile
        is unusable.
        """
        if self.expected_spacing:
            return float(self.expected_spacing)

        height, width = gray.shape
        fallback = max(8.0, self.IDEAL_LINE_SPACING * height / self.IDEAL_IMAGE_HEIGHT)

        _, ink = cv2.threshold(gray, self.content_binary_threshold, 255,
                               cv2.THRESH_BINARY_INV)
        row_frac = (ink > 0).sum(axis=1).astype(np.float64) / max(1, width)

        # Rows belonging to a horizontal line have noticeably more ink than
        # the white space between lines (dashed lines are still ~solid rows).
        active = row_frac > max(0.02, 0.25 * float(row_frac.max()))
        runs = _runs_from_mask(active, min_gap=1, min_len=1)
        if len(runs) < 4:
            return fallback

        centers = [(s + e) / 2.0 for s, e in runs]
        deltas = np.diff(centers)
        deltas = deltas[deltas > 2]
        if deltas.size < 3:
            return fallback

        spacing = float(np.median(deltas))
        if not (4.0 <= spacing <= height / 2.0):
            return fallback
        return spacing

    def _threshold_ladder(self):
        """Ink thresholds to try, from configured to most generous."""
        if self.clear and self._background_level is not None:
            # Never step past the background level: on the clear material a
            # too-generous threshold turns the whole background into ink.
            bg = self._background_level
            ladder = sorted({int(np.clip(bg - d, 5, 250)) for d in (25, 18, 12)})
            return ladder
        ladder = [self.binary_threshold]
        for extra in (200, 230):
            if extra > ladder[-1]:
                ladder.append(extra)
        return ladder

    def _detect_vline_candidates(self, gray, spacing, debug=False,
                                 binary_threshold=None):
        """Extract vertical-line candidates via morphological filtering.

        Returns a list of candidate dicts sorted by center x:
          {'center', 'x0', 'x1', 'thickness', 'y0', 'y1', 'coverage'}
        All coordinates are full-image.
        """
        height, width = gray.shape
        if binary_threshold is None:
            binary_threshold = self.binary_threshold

        _, ink = cv2.threshold(gray, binary_threshold, 255,
                               cv2.THRESH_BINARY_INV)

        drift = max(3, int(round(width * self.drift_dilate_fraction)))
        if drift % 2 == 0:
            drift += 1

        # A noisy vertical line is a column of speckles spread a few pixels
        # sideways; bridge that spread first so the vertical morphology sees
        # a connected column. (This also solidifies the dashed horizontal
        # lines, but the vertical OPEN below removes them regardless.)
        k_hclose = cv2.getStructuringElement(cv2.MORPH_RECT, (drift, 1))
        merged = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k_hclose)

        # Bridge the vertical gaps of a dotted/noisy vertical line. The
        # kernel stays below the horizontal line spacing so separate
        # horizontal lines are never merged into a false vertical structure.
        close_len = max(3, int(round(spacing * self.close_spacing_factor)))
        # Remove every horizontal-line crossing (their vertical extent is the
        # line thickness, far below one spacing).
        open_len = max(close_len + 2, int(round(spacing * self.open_spacing_factor)))

        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, close_len))
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, open_len))
        vmask = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, k_close)
        vmask = cv2.morphologyEx(vmask, cv2.MORPH_OPEN, k_open)

        # Tolerate skew/drift of the vertical line across columns
        k_drift = cv2.getStructuringElement(cv2.MORPH_RECT, (drift, 1))
        vmask_drift = cv2.dilate(vmask, k_drift)

        coverage = (vmask_drift > 0).sum(axis=0).astype(np.float64) / max(1, height)
        peak = float(coverage.max()) if coverage.size else 0.0

        if debug:
            print(f"VerticalBandDetector: close={close_len}px open={open_len}px "
                  f"drift={drift}px peak_coverage={peak:.2f}")

        if peak < self.min_coverage:
            return []

        threshold = max(self.min_coverage, self.relative_coverage * peak)
        merge_gap = max(2, int(round(width * self.merge_gap_fraction)))
        runs = _runs_from_mask(coverage >= threshold, min_gap=merge_gap, min_len=1)

        max_thickness = max(4, int(round(width * self.max_thickness_fraction)))
        half_drift = drift // 2

        candidates = []
        for run_x0, run_x1 in runs:
            # Shrink the dilated envelope back to the true ink extent
            seg = vmask[:, run_x0:run_x1 + 1]
            col_any = (seg > 0).any(axis=0)
            if not col_any.any():
                # Only dilated pixels here; approximate by removing dilation
                x0 = min(run_x1, run_x0 + half_drift)
                x1 = max(run_x0, run_x1 - half_drift)
                y0, y1 = 0, height - 1
                col_weights = None
            else:
                cols = np.flatnonzero(col_any)
                x0 = run_x0 + int(cols[0])
                x1 = run_x0 + int(cols[-1])
                rows = np.flatnonzero((seg > 0).any(axis=1))
                y0, y1 = int(rows[0]), int(rows[-1])
                col_weights = (seg > 0).sum(axis=0).astype(np.float64)

            thickness = x1 - x0 + 1
            if thickness > max_thickness:
                if debug:
                    print(f"  Rejected candidate x=[{x0}, {x1}]: too wide "
                          f"({thickness}px > {max_thickness}px)")
                continue

            if col_weights is not None and col_weights.sum() > 0:
                xs = np.arange(run_x0, run_x1 + 1, dtype=np.float64)
                center = int(round(float((xs * col_weights).sum() / col_weights.sum())))
            else:
                center = (x0 + x1) // 2

            candidates.append({
                'center': int(center),
                'x0': int(x0),
                'x1': int(x1),
                'thickness': int(thickness),
                'y0': int(y0),
                'y1': int(y1),
                'coverage': float(coverage[run_x0:run_x1 + 1].max()),
            })

        candidates.sort(key=lambda c: c['center'])
        return candidates

    def _select_boundary_lines(self, candidates, debug=False):
        """Select the 4 boundary lines from the candidate list.

        With more than 4 candidates, every 4-combination is scored on
        coverage plus the expected geometry: two similar wide bands (v1-v2
        and v3-v4) separated by a narrower central gap (v2-v3).
        """
        if len(candidates) <= 4:
            return list(candidates)

        # Keep the strongest candidates to bound the combinatorial search
        strongest = sorted(candidates, key=lambda c: c['coverage'], reverse=True)[:10]
        strongest.sort(key=lambda c: c['center'])

        best_combo = None
        best_score = -1.0
        for combo in itertools.combinations(strongest, 4):
            c1, c2, c3, c4 = combo  # already sorted by center
            d1 = c2['center'] - c1['center']
            d2 = c3['center'] - c2['center']
            d3 = c4['center'] - c3['center']
            if d1 <= 0 or d2 <= 0 or d3 <= 0:
                continue

            band_symmetry = 1.0 - abs(d1 - d3) / float(max(d1, d3))
            central_gap_ok = d2 < 0.6 * min(d1, d3)
            coverage_score = sum(c['coverage'] for c in combo) / 4.0

            score = coverage_score * (0.4 + 0.6 * band_symmetry)
            score *= 1.0 if central_gap_ok else 0.25

            if score > best_score:
                best_score = score
                best_combo = combo

        if best_combo is None:
            # Degenerate geometry everywhere; fall back to the 4 strongest
            best_combo = tuple(sorted(strongest[:4], key=lambda c: c['center']))

        if debug:
            centers = [c['center'] for c in best_combo]
            print(f"VerticalBandDetector: selected 4/{len(candidates)} "
                  f"candidates at x={centers} (score={best_score:.3f})")

        return list(best_combo)

    # ------------------------------------------------------------------
    # Band construction
    # ------------------------------------------------------------------

    def _bands_from_vlines(self, vlines, width):
        """Build the two bands from 4 boundary lines; None if degenerate."""
        v1, v2, v3, v4 = sorted(vlines, key=lambda v: v['center'])
        margin = self.inner_margin

        bands = []
        for left_v, right_v in ((v1, v2), (v3, v4)):
            inner_x0 = left_v['x1'] + 1 + margin
            inner_x1 = right_v['x0'] - 1 - margin
            if inner_x1 - inner_x0 < max(10, int(width * 0.01)):
                return None
            bands.append({
                'index': 0,
                'x0': int(inner_x0),
                'x1': int(inner_x1),
                'outer_x0': int(left_v['x0']),
                'outer_x1': int(right_v['x1']),
                'vline_xs': [int(left_v['center']), int(right_v['center'])],
                'vlines': [left_v, right_v],
            })
        return bands

    def _bands_from_content(self, gray, candidates, debug=False):
        """Fallback: derive bands from the ink-content profile.

        Used when the 4 boundary lines could not be selected reliably. Any
        vertical-line candidates that fall inside a band run are still used
        to trim the crop bounds.
        """
        height, width = gray.shape

        _, binary = cv2.threshold(gray, self.content_binary_threshold, 255,
                                  cv2.THRESH_BINARY_INV)
        col_frac = (binary > 0).sum(axis=0).astype(np.float64) / max(1, height)

        window = max(1, int(width * self.smooth_fraction))
        if window > 1:
            kernel = np.ones(window, dtype=np.float64) / window
            col_frac = np.convolve(col_frac, kernel, mode='same')

        min_gap = max(1, int(width * self.min_gap_fraction))
        min_len = max(1, int(width * self.min_band_fraction))
        band_runs = _runs_from_mask(col_frac > self.content_threshold,
                                    min_gap=min_gap, min_len=min_len)

        if not band_runs:
            if debug:
                print("VerticalBandDetector: no content runs found, "
                      "using full image as one band")
            return [{
                'index': 0,
                'x0': 0,
                'x1': int(width - 1),
                'outer_x0': 0,
                'outer_x1': int(width - 1),
                'vline_xs': [],
                'vlines': [],
            }]

        bands = []
        for outer_x0, outer_x1 in band_runs:
            run_vlines = [c for c in candidates
                          if outer_x0 <= c['center'] <= outer_x1]
            bands.append(self._band_from_run(outer_x0, outer_x1, run_vlines))
        return bands

    def _band_from_run(self, outer_x0, outer_x1, vlines):
        """Build a band dict from a content run, trimming inside its vlines."""
        margin = self.inner_margin
        vlines = sorted(vlines, key=lambda v: v['center'])

        if len(vlines) >= 2:
            left_v, right_v = vlines[0], vlines[-1]
            inner_x0 = left_v['x1'] + 1 + margin
            inner_x1 = right_v['x0'] - 1 - margin
        elif len(vlines) == 1:
            only_v = vlines[0]
            run_mid = (outer_x0 + outer_x1) // 2
            if only_v['center'] <= run_mid:
                inner_x0 = only_v['x1'] + 1 + margin
                inner_x1 = outer_x1
            else:
                inner_x0 = outer_x0
                inner_x1 = only_v['x0'] - 1 - margin
        else:
            inner_x0, inner_x1 = outer_x0, outer_x1

        if inner_x1 <= inner_x0:
            inner_x0, inner_x1 = outer_x0, outer_x1

        return {
            'index': 0,
            'x0': int(inner_x0),
            'x1': int(inner_x1),
            'outer_x0': int(outer_x0),
            'outer_x1': int(outer_x1),
            'vline_xs': [int(v['center']) for v in vlines],
            'vlines': vlines,
        }


# ----------------------------------------------------------------------
# Masking helpers
# ----------------------------------------------------------------------

def paint_vertical_line_regions(gray_image, vlines, pad=None):
    """Paint detected vertical lines (and their speckle halo) white.

    Uses the measured x-extent envelope of each line rather than a fixed
    thickness, so noisy/drifting lines are fully removed before debris or
    overspray thresholding.

    Args:
        gray_image: Grayscale image to modify (a copy is returned).
        vlines: Iterable of vertical-line dicts with 'x0'/'x1' (as produced
            by ``VerticalBandDetector``).
        pad: Extra pixels painted on each side of the measured envelope.
            Defaults to twice the line thickness (minimum 10) to cover the
            speckle halo around a noisy line.

    Returns:
        Grayscale image copy with the vertical line regions painted white.
    """
    painted = gray_image.copy()
    height, width = painted.shape[:2]
    for vline in vlines:
        local_pad = pad
        if local_pad is None:
            local_pad = max(10, 2 * int(vline.get('thickness', 5)))
        x1 = max(0, int(vline['x0']) - local_pad)
        x2 = min(width, int(vline['x1']) + local_pad + 1)
        if x2 > x1:
            painted[0:height, x1:x2] = 255
    return painted


def paint_vertical_lines_white(gray_image, vline_xs, thickness):
    """Paint vertical boundary lines white so they are never counted as defects.

    Backwards-compatible helper working from center positions and a fixed
    thickness. Prefer ``paint_vertical_line_regions`` when full vertical-line
    dicts are available.

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
