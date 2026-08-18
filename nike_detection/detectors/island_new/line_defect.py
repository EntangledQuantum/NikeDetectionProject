"""
New-Pattern (Dual-Band) Line Defect Detection

Detects print defects on the changed island pattern, where each image
contains two horizontal-print bands separated by a gap, each flanked by
vertical boundary lines (4 vertical lines total):

    [V prints V]   gap   [V prints V]

Defect types produced:
  - ``missing_line``    (red)    a gap in a print line = missing nozzles.
                                 At 2400 DPI a single missing-jet gap is
                                 ~90 px wide; the threshold scales with the
                                 measured line spacing. ``missing_pixels``
                                 counts ink-free columns in the gap.
  - ``misaligned_line`` (yellow) jets fired but the print is hazy/smudged
                                 (spread or split droplets), not a true gap.
  - ``stitch_error``    (blue)   jagged zig-zag on the 1-2 lines at a
                                 vertical-head stitch. 3 heads → up to 2
                                 zones; 4 heads → up to 3. Not scored as
                                 missing just because the straight fit fails.
  - ``high_density_region``      arbitrary-shape regions where the
                                 missing-nozzle density is much higher than
                                 the rest of the image.

Pipeline:
  1. ``VerticalBandDetector`` finds the 4 vertical boundary lines and the two
     print bands (morphology-based, tolerant of noisy lines).
  2. ``IslandLineExtractor`` extracts EVERY horizontal line per band: global
     slope via shear search, per-line slope/intercept, and a local centroid
     trajectory so the ink corridor survives stitch/roll. Fully missing
     lines are inserted from the spacing so they are still evaluated.
  3. Missing nozzles are ink-free runs along that local corridor, longer
     than a spacing-scaled ~90 px gap after a 1-D close that only bridges
     healthy stipple.
  4. Stitch errors are the strongest jagged-line clusters (high-frequency
     deviation from the straight fit), capped at ``num_heads - 1`` zones
     with at most two lines each.
  5. Hazy/smudged segments (split or vertically spread ink) are yellow.
  6. Missing pixels feed a density map for high-density regions.

A "detected lines" debug image (vertical lines magenta, horizontal line
trajectories green/blue, inserted lines red) is always produced and saved via
``save_debug_images`` so the line-extraction precursor can be verified.
"""

import cv2
import numpy as np

from nike_detection.detectors.base_legacy import BaseDetector
from nike_detection.geometry.vertical_band_detector import VerticalBandDetector
from nike_detection.geometry.island_line_extractor import IslandLineExtractor
from nike_detection.io.image_saver import save_image


def _runs_of(mask, min_len=1):
    """Contiguous True runs of a 1D bool mask as (start, end_inclusive)."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s + 1 >= min_len]


def _close_1d(mask, length):
    """1D binary closing (fill False gaps up to `length`) via OpenCV."""
    if length < 1:
        return mask.copy()
    arr = mask.astype(np.uint8).reshape(1, -1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(length), 1))
    closed = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)
    return closed.reshape(-1).astype(bool)


class NewPatternLineDefectDetector(BaseDetector):
    """Detect missing-nozzle / misalignment defects on a new-pattern island."""

    def __init__(self, sensitivity='medium', debug=False, clear=False):
        """Initialize the detector.

        Args:
            sensitivity: One of {'low', 'medium', 'high'}.
            debug: If True, print stage information (the lines debug image is
                always produced regardless).
            clear: If True, adapt to the clear scan material (gray
                background, fainter ink, lower SNR): all binarization
                thresholds are derived from the measured background level and
                the input is despeckled. Defect types, spacing-relative
                decision thresholds, and reporting are unchanged.
        """
        super().__init__()
        self.sensitivity = sensitivity
        self.debug = debug
        self.clear = clear

        self.band_detector = VerticalBandDetector(clear=clear)
        self.extractor = IslandLineExtractor(clear=clear)

        # Gap threshold is a fraction of the scaled ~90 px missing-jet width.
        if sensitivity == 'high':
            self.min_gap_fraction = 0.70
            self.split_min_fraction = 0.40
            self.dev_min_fraction = 0.40
        elif sensitivity == 'low':
            self.min_gap_fraction = 1.10
            self.split_min_fraction = 0.90
            self.dev_min_fraction = 1.00
        else:  # medium
            self.min_gap_fraction = 0.85
            self.split_min_fraction = 0.55
            self.dev_min_fraction = 0.60

        self.split_min_hollow = 3
        self.dev_abs_floor = 3.0
        self.dev_thickness_factor = 0.6
        self.hazy_thickness_factor = 1.7
        self.hazy_weak_factor = 0.55
        self.edge_ignore_fraction = 0.02
        self.expected_gap_px = 90.0
        self.ideal_spacing_px = 100.0
        self.num_heads = 3
        self.head_height = 0
        self.stitch_rms_fraction = 0.05
        self.stitch_score_ratio = 2.2
        self.stitch_max_lines_per_zone = 2

        # Density-map parameters
        self.density_downsample = 16
        self.density_rel_threshold = 0.40
        self.density_min_defects = 8

        self._debug_lines_image = None

        print(f"NewPattern Line Defect Detector ({sensitivity})"
              f"{' [clear material]' if clear else ''}:")
        print(f"  min gap: {self.min_gap_fraction:.2f} x expected ~{self.expected_gap_px:.0f}px jet")
        print(f"  split/hazy persistence: {self.split_min_fraction:.2f} x spacing")
        print(f"  stitch: up to {max(1, int(self.num_heads) - 1)} zone(s), "
              f"{self.stitch_max_lines_per_zone} lines each")

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def detect(self, image, image_path=None):
        """Run line-defect detection.

        Args:
            image: Input island image (BGR or grayscale).
            image_path: Accepted for interface compatibility (unused).

        Returns:
            tuple: (visualization_bgr, defects)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        bands = self.band_detector.detect(gray, self.debug)

        all_missing = []
        all_misaligned = []
        all_stitch = []
        band_summaries = []
        extractions = []  # (band, result) for debug drawing

        for band in bands:
            x0, x1 = band['x0'], band['x1']
            if x1 - x0 < 20:
                continue

            result = self.extractor.extract(gray[:, x0:x1 + 1], self.debug)
            if result is None:
                continue
            extractions.append((band, result))

            missing, misaligned, stitch = self._evaluate_band(band, result)
            all_missing.extend(missing)
            all_misaligned.extend(misaligned)
            all_stitch.extend(stitch)

            band_summaries.append({
                'index': band['index'],
                'x0': x0, 'x1': x1,
                'vline_xs': band['vline_xs'],
                'line_count': len(result['lines']),
                'inserted_line_count': sum(1 for l in result['lines'] if l['inserted']),
                'slope': result['slope'],
                'spacing': result['spacing'],
                'missing_defects': len(missing),
                'missing_pixels': int(sum(d['missing_pixels'] for d in missing)),
                'misaligned_defects': len(misaligned),
                'stitch_defects': len(stitch),
            })

        spacing = float(np.median([b['spacing'] for b in band_summaries])) \
            if band_summaries else 96.0

        density_regions = self._find_density_regions(gray.shape, all_missing,
                                                     spacing)

        visualization = self._create_visualization(
            image, all_missing, all_misaligned, density_regions, spacing,
            stitch=all_stitch)
        self._debug_lines_image = self._create_lines_debug(
            image, extractions)

        defects = self._build_defects(band_summaries, all_missing,
                                      all_misaligned, density_regions,
                                      stitch=all_stitch)

        if self.debug:
            print(f"NewPatternLineDefect: {len(all_missing)} missing, "
                  f"{len(all_misaligned)} hazy, {len(all_stitch)} stitch, "
                  f"{len(density_regions)} high-density regions")

        return visualization, defects

    # ------------------------------------------------------------------
    # Per-band defect evaluation
    # ------------------------------------------------------------------

    def _expected_gap(self, spacing):
        """Missing-jet width at this image's scale (~90 px at 2400 DPI)."""
        ideal = max(1.0, float(self.ideal_spacing_px or 100.0))
        return float(self.expected_gap_px) * (float(spacing) / ideal)

    def _evaluate_band(self, band, result):
        """Derive missing / hazy / stitch defects from a band's line statistics."""
        x0 = band['x0']
        spacing = result['spacing'] or 96.0
        lines = result['lines']
        expected = self._expected_gap(spacing)
        min_gap = max(8, int(round(self.min_gap_fraction * expected)))
        split_min = max(8, int(round(self.split_min_fraction * spacing)))
        hazy_min = max(8, int(round(self.split_min_fraction * spacing)))

        close_len = self._measure_stipple_close_len(lines, min_gap, expected)
        stitch_idx = self._select_stitch_lines(lines, spacing)

        missing, misaligned, stitch = [], [], []
        for line_idx, line in enumerate(lines):
            width = line['ink_cols'].shape[0]
            xs = np.arange(width)
            y_true = IslandLineExtractor.line_y(result, line, xs)
            edge = max(2, int(round(self.edge_ignore_fraction * width)))

            ink_closed = _close_1d(line['ink_cols'], close_len)
            is_stitch = line_idx in stitch_idx
            ink_frac = float(line['ink_cols'].mean())
            # First/last rows in a crop are often clipped lines, not missing jets.
            clipped_edge = (
                line_idx in (0, len(lines) - 1) and ink_frac < 0.70
            )
            if not is_stitch and not clipped_edge:
                for s, e in _runs_of(~ink_closed, min_len=min_gap):
                    s_c = max(s, edge)
                    e_c = min(e, width - 1 - edge)
                    if e_c - s_c + 1 < min_gap:
                        continue
                    raw_missing = int((~line['ink_cols'][s_c:e_c + 1]).sum())
                    mid = (s_c + e_c) // 2
                    yc = int(y_true[mid])
                    missing.append({
                        'type': 'missing_line',
                        'band': band['index'],
                        'line_index': line_idx,
                        'start_x': int(s_c + x0),
                        'end_x': int(e_c + x0),
                        'y': yc,
                        'location': (int(mid + x0), yc),
                        'size': int(e_c - s_c + 1),
                        'missing_pixels': raw_missing,
                        'whole_line': bool(line['inserted']),
                    })

            if line['inserted']:
                continue

            if line_idx in stitch_idx:
                stitch.append(self._stitch_record(
                    band, line_idx, line, y_true, x0, width, spacing))
                continue

            # ---- Hazy / smudged (yellow): ink is present but split or spread
            split_mask = (line['n_runs'] >= 2) & \
                         (line['hollow'] >= self.split_min_hollow)
            split_mask = _close_1d(split_mask, max(5, close_len))
            split_runs = _runs_of(split_mask, min_len=split_min)

            med_thick = max(1.0, float(line['thickness'] or 1.0))
            extent = line.get('extent', line['ink_count'])
            thick_smear = line['ink_cols'] & (
                line['ink_count'] >= self.hazy_thickness_factor * med_thick)
            # Weak/faint ink is only meaningful when a healthy line is several
            # pixels thick (2400 DPI). On downscaled snippets thickness is ~2
            # and single-pixel stipple would flood yellow.
            if med_thick >= 4.0:
                weak = line['ink_cols'] & (
                    line['ink_count'] <= self.hazy_weak_factor * med_thick)
            else:
                weak = np.zeros_like(line['ink_cols'])
            spread = line['ink_cols'] & (
                extent >= max(int(med_thick) + 3, int(np.ceil(2.2 * med_thick))))
            hazy_mask = _close_1d(thick_smear | weak | spread, max(5, close_len))
            hazy_runs = _runs_of(hazy_mask, min_len=hazy_min)

            for kind, runs in (('split', split_runs), ('hazy', hazy_runs)):
                for s, e in runs:
                    yc = int(y_true[(s + e) // 2])
                    misaligned.append({
                        'type': 'misaligned_line',
                        'kind': kind,
                        'band': band['index'],
                        'line_index': line_idx,
                        'start_x': int(s + x0),
                        'end_x': int(e + x0),
                        'y': yc,
                        'location': (int((s + e) // 2 + x0), yc),
                        'size': int(e - s + 1),
                    })

        misaligned = self._merge_overlaps(misaligned)
        return missing, misaligned, stitch

    def _stitch_record(self, band, line_idx, line, y_true, x0, width, spacing):
        """One jagged line at a vertical-head stitch."""
        yc = int(np.median(y_true))
        return {
            'type': 'stitch_error',
            'band': band['index'],
            'line_index': line_idx,
            'start_x': int(x0),
            'end_x': int(x0 + width - 1),
            'y': yc,
            'location': (int(x0 + width // 2), yc),
            'size': int(width),
            'jagged_rms': float(line.get('jagged_rms', 0.0)),
            'jagged_hf': float(line.get('jagged_hf', 0.0)),
            'spacing': float(spacing),
        }

    def _select_stitch_lines(self, lines, spacing):
        """Pick up to (num_heads-1) clusters of 1-2 jagged lines."""
        if not lines:
            return set()
        scores = np.array([
            max(float(line.get('jagged_rms', 0.0)), float(line.get('jagged_hf', 0.0)))
            if not line.get('inserted') else 0.0
            for line in lines
        ], dtype=np.float64)
        positive = scores[scores > 0]
        if positive.size < 4:
            return set()
        med = float(np.median(positive))
        mad = float(np.median(np.abs(positive - med))) + 1e-6
        abs_thr = max(0.6, float(self.stitch_rms_fraction) * spacing)
        rel_thr = med + max(2.0 * 1.4826 * mad, self.stitch_score_ratio * med)
        thr = max(abs_thr, rel_thr)

        flags = scores >= thr
        if not flags.any():
            return set()

        clusters = []
        i = 0
        n = len(lines)
        max_lines = max(1, int(self.stitch_max_lines_per_zone))
        while i < n:
            if flags[i]:
                j = i
                while j + 1 < n and (
                    flags[j + 1]
                    or (j - i + 1 < max_lines and scores[j + 1] >= 0.65 * thr)
                ):
                    j += 1
                if (j - i + 1) <= max_lines + 1:
                    clusters.append((i, j, float(scores[i:j + 1].max())))
                i = j + 1
            else:
                i += 1

        n_expect = max(1, int(self.num_heads) - 1)
        clusters.sort(key=lambda item: -item[2])
        chosen = set()
        for start, end, _score in clusters[:n_expect]:
            order = sorted(range(start, end + 1), key=lambda k: -scores[k])[:max_lines]
            chosen.update(order)
            # A stitch zone is typically two adjacent lines; the second may be
            # hazy/low-ink rather than the peak-jagged one.
            if len(order) < max_lines:
                seed = order[0]
                neighbors = [nb for nb in (seed - 1, seed + 1)
                             if 0 <= nb < n and nb not in chosen
                             and not lines[nb].get('inserted')
                             and float(lines[nb]['ink_cols'].mean()) < 0.80]
                neighbors.sort(key=lambda nb: -scores[nb])
                for nb in neighbors[:max_lines - len(order)]:
                    chosen.add(nb)
        return chosen

    @staticmethod
    def _measure_stipple_close_len(lines, min_gap, expected=None):
        """Measured 1-D closing length that bridges the healthy stipple.

        Collects the ink-free run lengths INSIDE each line's printed extent
        and returns ~3x their 90th percentile, capped well below the expected
        missing-jet width so real ~90 px gaps are never closed.
        """
        texture_cap = 0.45 * float(expected if expected else min_gap)
        gaps = []
        for line in lines:
            if line['inserted']:
                continue
            ink = line['ink_cols']
            idx = np.flatnonzero(ink)
            if idx.size < 20:
                continue
            deltas = np.diff(idx) - 1
            gaps.extend(deltas[(deltas > 0) & (deltas < texture_cap)].tolist())

        if not gaps:
            return max(4, min(min_gap // 4, max(4, int(texture_cap))))
        p90 = float(np.percentile(gaps, 90))
        return int(np.clip(3 * p90, 4, max(4, texture_cap)))

    @staticmethod
    def _merge_overlaps(defects):
        """Merge misaligned segments that overlap on the same line."""
        by_line = {}
        for d in defects:
            by_line.setdefault((d['band'], d['line_index']), []).append(d)

        merged = []
        for group in by_line.values():
            group.sort(key=lambda d: d['start_x'])
            current = group[0]
            for d in group[1:]:
                if d['start_x'] <= current['end_x'] + 5:
                    current['end_x'] = max(current['end_x'], d['end_x'])
                    current['size'] = current['end_x'] - current['start_x'] + 1
                    if current['kind'] != d['kind']:
                        current['kind'] = 'hazy+split'
                else:
                    merged.append(current)
                    current = d
            merged.append(current)
        return merged

    # ------------------------------------------------------------------
    # High-density regions
    # ------------------------------------------------------------------

    def _find_density_regions(self, shape, missing_defects, spacing=96.0):
        """Find arbitrary-shape regions with unusually dense missing ink.

        Missing pixels are splatted into a downsampled accumulator, smoothed
        with a wide Gaussian (~2 measured line spacings), and thresholded
        relative to the peak density. Blob contours are reported in
        full-image coords.
        """
        if len(missing_defects) < self.density_min_defects:
            return []

        height, width = shape[:2]
        ds = self.density_downsample
        map_h = (height + ds - 1) // ds
        map_w = (width + ds - 1) // ds
        density = np.zeros((map_h, map_w), dtype=np.float32)

        spacing_guess = np.median([d['size'] for d in missing_defects]) or 60
        for d in missing_defects:
            row = min(map_h - 1, d['y'] // ds)
            c0 = d['start_x'] // ds
            c1 = min(map_w - 1, d['end_x'] // ds)
            per_cell = d['missing_pixels'] / max(1, c1 - c0 + 1)
            density[row, c0:c1 + 1] += per_cell

        sigma = max(1.5, 2.0 * spacing / ds)
        ksize = int(sigma * 6) | 1
        blurred = cv2.GaussianBlur(density, (ksize, ksize), sigma)

        peak = float(blurred.max())
        if peak <= 0:
            return []
        threshold = self.density_rel_threshold * peak

        mask = (blurred >= threshold).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < 9:  # tiny blobs in map cells
                continue
            bbox = (int(x * ds), int(y * ds), int(w * ds), int(h * ds))
            inside = [d for d in missing_defects
                      if bbox[0] <= d['location'][0] < bbox[0] + bbox[2]
                      and bbox[1] <= d['y'] < bbox[1] + bbox[3]]
            if len(inside) < self.density_min_defects:
                continue
            regions.append({
                'type': 'high_density_region',
                'bbox': bbox,
                'contour': (contour.astype(np.int32) * ds),
                'defect_count': len(inside),
                'missing_pixels': int(sum(d['missing_pixels'] for d in inside)),
                'peak_density': float(blurred[y:y + h, x:x + w].max()),
            })

        regions.sort(key=lambda r: r['missing_pixels'], reverse=True)
        return regions

    # ------------------------------------------------------------------
    # Output assembly
    # ------------------------------------------------------------------

    def _build_defects(self, band_summaries, missing, misaligned, regions,
                       stitch=None):
        """Assemble the structured defect output list."""
        stitch = stitch or []
        total_missing_px = int(sum(d['missing_pixels'] for d in missing))
        defects = [{
            'type': 'bands_detected',
            'band_count': len(band_summaries),
            'bands': band_summaries,
            'total_lines': int(sum(b['line_count'] for b in band_summaries)),
            'total_missing_defects': len(missing),
            'total_missing_pixels': total_missing_px,
            'estimated_missing_nozzles': total_missing_px,
            'total_misaligned_defects': len(misaligned),
            'total_stitch_defects': len(stitch),
        }]
        defects.extend(missing)
        defects.extend(misaligned)
        defects.extend(stitch)
        for r in regions:
            entry = dict(r)
            entry.pop('contour', None)  # keep JSON small
            defects.append(entry)
        return defects

    def _create_visualization(self, image, missing, misaligned, regions,
                              spacing=96.0, stitch=None):
        """Red = missing nozzles, yellow = hazy, blue = stitch, orange = density."""
        stitch = stitch or []
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()

        half = max(3, int(round(0.15 * spacing)))
        stitch_half = max(half + 1, int(round(0.22 * spacing)))

        overlay = vis.copy()
        for d in missing:
            cv2.rectangle(overlay, (d['start_x'], d['y'] - half),
                          (d['end_x'], d['y'] + half), (0, 0, 255), -1)
        for d in misaligned:
            cv2.rectangle(overlay, (d['start_x'], d['y'] - half),
                          (d['end_x'], d['y'] + half), (0, 255, 255), -1)
        for d in stitch:
            cv2.rectangle(overlay, (d['start_x'], d['y'] - stitch_half),
                          (d['end_x'], d['y'] + stitch_half), (255, 80, 0), -1)
        cv2.addWeighted(vis, 0.65, overlay, 0.35, 0, dst=vis)

        for r in regions:
            cv2.drawContours(vis, [r['contour']], -1, (0, 128, 255), 8)
            x, y, w, h = r['bbox']
            cv2.putText(vis, f"HIGH DENSITY: {r['missing_pixels']}px missing",
                        (x, max(30, y - 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 128, 255), 3)

        total_px = sum(d['missing_pixels'] for d in missing)
        text = (f"Missing: {len(missing)} gaps / {total_px} px "
                f"| Hazy: {len(misaligned)} "
                f"| Stitch: {len(stitch)} "
                f"| Density regions: {len(regions)}")
        cv2.putText(vis, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 3)
        return vis

    def _create_lines_debug(self, image, extractions):
        """Debug image: vertical lines magenta, line trajectories colored."""
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()

        for v in self.band_detector.last_vlines:
            cv2.rectangle(vis, (v['x0'] - 2, v['y0']), (v['x1'] + 2, v['y1']),
                          (255, 0, 255), -1)

        palette = [(0, 200, 0), (255, 128, 0)]
        for band, result in extractions:
            x0, x1 = band['x0'], band['x1']
            xs = np.arange(0, x1 - x0 + 1, 8)
            for i, line in enumerate(result['lines']):
                ys = IslandLineExtractor.line_y(result, line, xs).astype(np.int32)
                color = (0, 0, 255) if line['inserted'] else palette[i % 2]
                pts = np.stack([xs + x0, ys], axis=1).reshape(-1, 1, 2).astype(np.int32)
                cv2.polylines(vis, [pts], False, color, 3)

        cv2.putText(vis, "Magenta: verticals | Green/Orange: local trajectories | "
                    "Red: inserted (fully missing) lines", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)
        return vis

    def save_debug_images(self, output_dir, base_name):
        """Always save the detected-lines debug image; more when debug=True."""
        debug_paths = []
        if self._debug_lines_image is not None:
            saved = save_image(output_dir, base_name, self._debug_lines_image,
                               'newpattern_detected_lines')
            if saved:
                debug_paths.append(saved)
        return debug_paths if debug_paths else None
