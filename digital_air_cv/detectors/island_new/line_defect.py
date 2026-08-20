"""
New-Pattern (Dual-Band) Line Defect Detection

Scores the three island line defects on the *normalized ink density* map from
:mod:`digital_air_cv.geometry.ink_density` rather than on a thresholded
grayscale image. Everything the detector looks at is therefore dimensionless
and self-calibrated against the region it came from, which is what makes one
parameter set work for Key, Cyan, Magenta and Yellow alike.

Two ratios carry all the evidence (see
:mod:`digital_air_cv.geometry.line_profile`):

    coverage(x) = ink mass / healthy mass       ~1 healthy, ~0 no ink
    density(x)  = core density / healthy core   ~1 healthy, low = pale + spread

and they separate the defects cleanly, because a missing nozzle and a smear
differ in *kind*, not merely in degree:

    ==================  ==========  =========  ============================
    condition           coverage    density    verdict
    ==================  ==========  =========  ============================
    healthy print       ~1.0        ~1.0       -
    nozzle not firing   ~0.0        ~0.0       missing_line   (red)
    ink fired, smeared  ~0.5-1.0    low        misaligned_line (yellow)
    thin but crisp      lower       ~1.0       - (not a defect)
    ==================  ==========  =========  ============================

That last row is the one the previous implementation could not express. It
keyed haze off per-line ink *thickness* in pixels, so on Cyan and Magenta --
where stipple makes thickness fluctuate -- normal print constantly tripped the
test, while on Yellow almost nothing was detected at all because the grayscale
ink threshold never matched yellow's 21-level luminance contrast. Coverage and
density are immune to both effects.

Defect types produced:
  - ``missing_line``    (red)    coverage collapses over a run longer than a
                                 missing-jet gap (~90 px at 2400 DPI, scaled
                                 by the measured line spacing).
  - ``misaligned_line`` (yellow) ink present but pale/diffuse; carries a 0-1
                                 ``severity`` so a faint haze and a heavy
                                 smear are distinguishable.
  - ``stitch_error``    (blue)   short-range zig-zag on the 1-2 lines where
                                 two print heads join. Capped at
                                 ``num_heads - 1`` zones by construction and
                                 reported with a continuous ``severity``.
  - ``high_density_region``      arbitrary-shape clusters of missing ink.

Only five thresholds drive the three detectors, replacing the thirteen the
previous version needed; each is a ratio, so none of them has to be retuned
for a different ink, exposure or scan resolution.
"""

import cv2
import numpy as np

from digital_air_cv.detectors.base_legacy import BaseDetector
from digital_air_cv.geometry.ink_density import as_pseudo_gray, measure_ink_field
from digital_air_cv.geometry.line_profile import BandProfiler, column_runs
from digital_air_cv.geometry.vertical_band_detector import VerticalBandDetector
from digital_air_cv.io.image_saver import save_image

# Severity bands shared by haze and stitch reporting.
_SEVERITY_EDGES = ((0.35, "mild"), (0.65, "moderate"))


def _severity_label(value):
    for edge, name in _SEVERITY_EDGES:
        if value < edge:
            return name
    return "severe"


def _close(mask, length):
    """1-D binary closing so healthy stipple does not fragment a run."""
    if length < 1:
        return mask.copy()
    arr = mask.astype(np.uint8).reshape(1, -1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(length), 1))
    return cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel).reshape(-1).astype(bool)


class NewPatternLineDefectDetector(BaseDetector):
    """Detect missing-nozzle / haze / stitch defects on a new-pattern island."""

    def __init__(self, sensitivity='medium', debug=False, clear=False):
        """Initialize the detector.

        Args:
            sensitivity: One of {'low', 'medium', 'high'}. Shifts the two
                decision levels and how long a defect must persist; it does
                not change what is measured.
            debug: If True, print stage information.
            clear: Clear scan material (gray background, fainter ink). Only
                band geometry needs telling: the ink field already calibrates
                against whatever the measured background is, so the defect
                thresholds are the same ones used on white paper.
        """
        super().__init__()
        self.sensitivity = sensitivity
        self.debug = debug
        self.clear = clear

        self.band_detector = VerticalBandDetector(clear=clear)
        self.profiler = BandProfiler()

        # ---- the five decision thresholds -------------------------------
        if sensitivity == 'high':
            self.missing_level = 0.28
            self.haze_level = 0.68
            self.min_gap_fraction = 0.70
            self.haze_min_fraction = 0.45
            self.stitch_wave = 0.024
        elif sensitivity == 'low':
            self.missing_level = 0.12
            self.haze_level = 0.44
            self.min_gap_fraction = 1.10
            self.haze_min_fraction = 0.90
            self.stitch_wave = 0.040
        else:  # medium
            self.missing_level = 0.20
            self.haze_level = 0.55
            self.min_gap_fraction = 0.85
            self.haze_min_fraction = 0.60
            self.stitch_wave = 0.030

        # ---- geometry / reporting (not defect sensitivity) --------------
        self.expected_gap_px = 90.0
        self.ideal_spacing_px = 100.0
        self.num_heads = 3
        self.head_height = 0
        self.edge_ignore_fraction = 0.02
        self.density_downsample = 16
        self.density_rel_threshold = 0.40
        self.density_min_defects = 8

        self._debug_lines_image = None
        self._last_field = None

        if debug:
            print(f"NewPattern Line Defect Detector ({sensitivity})"
                  f"{' [clear material]' if clear else ''}: "
                  f"missing<{self.missing_level:.2f} haze<{self.haze_level:.2f} "
                  f"stitch>{self.stitch_wave:.3f}")

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def detect(self, image, image_path=None):
        """Run line-defect detection.

        Args:
            image: Island image (BGR preferred; grayscale still works but
                loses the per-ink channel separation).
            image_path: Accepted for interface compatibility (unused).

        Returns:
            tuple: (visualization_bgr, defects)
        """
        field = measure_ink_field(image)
        # Band geometry runs on density too, not grayscale, so the vertical
        # boundary lines of a Yellow island are as visible as a Key one.
        bands = self.band_detector.detect(as_pseudo_gray(field), self.debug)

        profiles = []
        for band in bands:
            result = self.profiler.profile(field, band['x0'], band['x1'], self.debug)
            if result is not None:
                profiles.append((band, result))

        return self.evaluate(image, profiles, field)

    def evaluate(self, image, profiles, field):
        """Score pre-built band profiles and render the overlay.

        Split out from :meth:`detect` so the pipeline can reuse geometry that
        was already built once for the region.
        """
        self._last_field = field
        missing, hazy, stitch, summaries = [], [], [], []

        for band, profile in profiles:
            band_missing, band_hazy, band_stitch = self._evaluate_band(band, profile)
            missing.extend(band_missing)
            hazy.extend(band_hazy)
            stitch.extend(band_stitch)
            summaries.append({
                'index': band['index'],
                'x0': band['x0'], 'x1': band['x1'],
                'vline_xs': band.get('vline_xs', []),
                'line_count': profile.n_lines,
                'inserted_line_count': int(profile.inserted.sum()),
                'slope': profile.slope,
                'spacing': profile.spacing,
                'base_mass': round(profile.base_mass, 3),
                'base_peak': round(profile.base_peak, 3),
                'missing_defects': len(band_missing),
                'missing_pixels': int(sum(d['missing_pixels'] for d in band_missing)),
                'misaligned_defects': len(band_hazy),
                'stitch_defects': len(band_stitch),
            })

        spacing = float(np.median([s['spacing'] for s in summaries])) if summaries else 96.0
        shape = image.shape[:2]
        regions = self._find_density_regions(shape, missing, spacing)

        visualization = self._create_visualization(
            image, missing, hazy, regions, spacing, stitch)
        self._debug_lines_image = self._create_lines_debug(image, profiles)

        defects = self._build_defects(summaries, missing, hazy, regions, stitch, field)

        if self.debug:
            print(f"NewPatternLineDefect: {len(missing)} missing, {len(hazy)} hazy, "
                  f"{len(stitch)} stitch, {len(regions)} density regions")
        return visualization, defects

    # ------------------------------------------------------------------
    # Per-band evaluation
    # ------------------------------------------------------------------

    def _evaluate_band(self, band, profile):
        """Derive missing / hazy / stitch defects from one band's profile."""
        spacing = profile.spacing or 96.0
        x0 = band['x0']
        width = profile.width

        expected_gap = self.expected_gap_px * spacing / max(1.0, self.ideal_spacing_px)
        min_gap = max(6, int(round(self.min_gap_fraction * expected_gap)))
        min_haze = max(6, int(round(self.haze_min_fraction * spacing)))
        # Bridge only the healthy stipple, never a real jet gap.
        bridge = max(3, int(round(0.25 * expected_gap)))
        edge = max(2, int(round(self.edge_ignore_fraction * width)))

        stitch_lines = self._select_stitch(profile)
        xs = np.arange(width)

        missing, hazy, stitch = [], [], []
        for line in range(profile.n_lines):
            coverage = profile.coverage[line]
            density = profile.density[line]
            true_y = profile.true_y(line, xs)

            if line in stitch_lines:
                severity, waviness = stitch_lines[line]
                stitch.append(self._stitch_record(
                    band, line, profile, true_y, severity, waviness))

            # A stitch line is still scored for ink defects. The corridor is
            # built around each line's measured trajectory rather than a
            # straight fit, so printed zig-zag stays inside the mask and a
            # join no longer hides a real gap or smear sitting on it.

            # ---- missing nozzles: ink mass collapses --------------------
            gone = _close(coverage < self.missing_level, bridge)
            for start, end in column_runs(gone, min_len=min_gap):
                start, end = max(start, edge), min(end, width - 1 - edge)
                if end - start + 1 < min_gap:
                    continue
                raw = int((coverage[start:end + 1] < self.missing_level).sum())
                mid = (start + end) // 2
                missing.append({
                    'type': 'missing_line',
                    'band': band['index'],
                    'line_index': line,
                    'start_x': int(start + x0),
                    'end_x': int(end + x0),
                    'y': int(true_y[mid]),
                    'location': (int(mid + x0), int(true_y[mid])),
                    'size': int(end - start + 1),
                    'missing_pixels': raw,
                    'whole_line': bool(profile.inserted[line]),
                })

            if profile.inserted[line]:
                continue

            # ---- haze: ink landed, but pale and spread ------------------
            smeared = _close(
                (density < self.haze_level) & (coverage >= self.missing_level),
                bridge)
            for start, end in column_runs(smeared, min_len=min_haze):
                segment = density[start:end + 1]
                severity = float(np.clip(1.0 - segment.mean() / self.haze_level, 0.0, 1.0))
                mid = (start + end) // 2
                hazy.append({
                    'type': 'misaligned_line',
                    'kind': 'hazy',
                    'band': band['index'],
                    'line_index': line,
                    'start_x': int(start + x0),
                    'end_x': int(end + x0),
                    'y': int(true_y[mid]),
                    'location': (int(mid + x0), int(true_y[mid])),
                    'size': int(end - start + 1),
                    'severity': round(severity, 3),
                    'grade': _severity_label(severity),
                    'mean_density': round(float(segment.mean()), 3),
                })

        return missing, hazy, stitch

    def _stitch_record(self, band, line, profile, true_y, severity, waviness):
        width = profile.width
        return {
            'type': 'stitch_error',
            'band': band['index'],
            'line_index': line,
            'start_x': int(band['x0']),
            'end_x': int(band['x0'] + width - 1),
            'y': int(np.median(true_y)),
            'location': (int(band['x0'] + width // 2), int(np.median(true_y))),
            'size': int(width),
            'waviness': round(float(waviness), 4),
            'amplitude_px': round(float(waviness * profile.spacing), 2),
            'severity': round(float(severity), 3),
            'grade': _severity_label(severity),
            'spacing': round(float(profile.spacing), 2),
        }

    def _waviness(self, profile):
        """Short-range trajectory wander per line, in units of line spacing.

        Only columns with healthy ink contribute. Haze spreads a line's ink
        sideways and drags its centroid around without the printed core
        actually moving, so measuring wander across smeared columns reports a
        zig-zag that is not there -- which is precisely how a hazy band gets
        mistaken for a head join. Restricting to healthy ink asks the right
        question: did the *well-printed* part of this line wander?
        """
        healthy = ((profile.coverage >= self.missing_level)
                   & (profile.density >= self.haze_level))
        counts = healthy.sum(axis=1)
        residual = profile.residual * healthy
        energy = np.einsum("ij,ij->i", residual, residual)

        wave = np.zeros(profile.n_lines, dtype=np.float32)
        usable = counts >= 16
        wave[usable] = np.sqrt(energy[usable] / counts[usable]) / profile.spacing
        return wave

    def _select_stitch(self, profile):
        """Pick the lines where two heads join.

        A stitch is physically constrained: with ``num_heads`` heads there are
        at most ``num_heads - 1`` joins, spaced evenly down the region. Rather
        than trusting that geometry blindly (head heights drift, and a crop may
        contain only part of a head) the waviness outliers are found first and
        the expected positions only break ties. The count is capped, so a
        uniformly wavy band can never report eight stitch zones.

        Returns:
            dict mapping line index -> severity in 0..1.
        """
        wave = self._waviness(profile)
        real = np.flatnonzero(~profile.inserted)
        if real.size < 2:
            return {}

        # With enough lines the band supplies its own baseline, and a join
        # stands out from merely textured print by a wide margin -- so the
        # band-relative test leads and the absolute floor only guards against
        # a band that is uniformly, implausibly smooth. A short crop cannot
        # estimate that spread at all, so it falls back to the floor rather
        # than trusting a MAD taken over three samples.
        if real.size >= 6:
            values = wave[real]
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median))) + 1e-9
            threshold = max(0.5 * self.stitch_wave,
                            median + 3.0 * 1.4826 * mad,
                            1.6 * median)
        else:
            threshold = self.stitch_wave

        candidates = [int(i) for i in real if wave[i] >= threshold]
        if not candidates:
            return {}

        # Collapse neighbouring lines into one zone; a join disturbs one or two
        # adjacent lines, not a broad span.
        zones = []
        current = [candidates[0]]
        for index in candidates[1:]:
            if index - current[-1] <= 2:
                current.append(index)
            else:
                zones.append(current)
                current = [index]
        zones.append(current)

        rows = profile.rows
        span = float(rows[-1] - rows[0]) or 1.0
        expected = [k / self.num_heads for k in range(1, max(2, self.num_heads))]

        ranked = []
        for zone in zones:
            best = max(zone, key=lambda i: wave[i])
            position = float(rows[best] - rows[0]) / span
            nearest = min(abs(position - e) for e in expected)
            # Soft preference only: a real join anywhere still outranks noise.
            bonus = 1.0 + 0.5 * float(np.exp(-(nearest / 0.12) ** 2))
            ranked.append((wave[best] * bonus, best))

        ranked.sort(reverse=True)
        keep = max(1, int(self.num_heads) - 1)

        selected = {}
        for _score, line in ranked[:keep]:
            excess = wave[line] / max(threshold, 1e-6) - 1.0
            selected[line] = (float(np.clip(excess / 1.5, 0.0, 1.0)),
                              float(wave[line]))
        return selected

    # ------------------------------------------------------------------
    # High-density regions
    # ------------------------------------------------------------------

    def _find_density_regions(self, shape, missing, spacing=96.0):
        """Arbitrary-shape clusters where missing ink is unusually dense."""
        if len(missing) < self.density_min_defects:
            return []

        height, width = shape[:2]
        ds = self.density_downsample
        map_h, map_w = (height + ds - 1) // ds, (width + ds - 1) // ds
        density = np.zeros((map_h, map_w), dtype=np.float32)

        for defect in missing:
            row = min(map_h - 1, defect['y'] // ds)
            c0 = defect['start_x'] // ds
            c1 = min(map_w - 1, defect['end_x'] // ds)
            density[row, c0:c1 + 1] += defect['missing_pixels'] / max(1, c1 - c0 + 1)

        sigma = max(1.5, 2.0 * spacing / ds)
        ksize = int(sigma * 6) | 1
        blurred = cv2.GaussianBlur(density, (ksize, ksize), sigma)
        peak = float(blurred.max())
        if peak <= 0:
            return []

        mask = (blurred >= self.density_rel_threshold * peak).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < 9:
                continue
            bbox = (int(x * ds), int(y * ds), int(w * ds), int(h * ds))
            inside = [
                d for d in missing
                if bbox[0] <= d['location'][0] < bbox[0] + bbox[2]
                and bbox[1] <= d['y'] < bbox[1] + bbox[3]
            ]
            if len(inside) < self.density_min_defects:
                continue
            regions.append({
                'type': 'high_density_region',
                'bbox': bbox,
                'contour': contour.astype(np.int32) * ds,
                'defect_count': len(inside),
                'missing_pixels': int(sum(d['missing_pixels'] for d in inside)),
                'peak_density': float(blurred[y:y + h, x:x + w].max()),
            })
        regions.sort(key=lambda r: r['missing_pixels'], reverse=True)
        return regions

    # ------------------------------------------------------------------
    # Output assembly
    # ------------------------------------------------------------------

    def _build_defects(self, summaries, missing, hazy, regions, stitch, field=None):
        total_missing_px = int(sum(d['missing_pixels'] for d in missing))
        header = {
            'type': 'bands_detected',
            'band_count': len(summaries),
            'bands': summaries,
            'total_lines': int(sum(s['line_count'] for s in summaries)),
            'total_missing_defects': len(missing),
            'total_missing_pixels': total_missing_px,
            'estimated_missing_nozzles': total_missing_px,
            'total_misaligned_defects': len(hazy),
            'total_stitch_defects': len(stitch),
        }
        if field is not None:
            header['ink'] = {
                'axis_bgr': [round(float(v), 3) for v in field.axis],
                'paper_bgr': [round(float(v), 1) for v in field.paper],
                'contrast': round(float(field.contrast), 1),
                'noise': round(float(field.noise), 4),
            }
        if hazy:
            header['haze_severity_mean'] = round(
                float(np.mean([d['severity'] for d in hazy])), 3)
        if stitch:
            header['stitch_severity_max'] = round(
                float(max(d['severity'] for d in stitch)), 3)

        defects = [header]
        defects.extend(missing)
        defects.extend(hazy)
        defects.extend(stitch)
        for region in regions:
            entry = dict(region)
            entry.pop('contour', None)  # keep the JSON small
            defects.append(entry)
        return defects

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _create_visualization(self, image, missing, hazy, regions,
                              spacing=96.0, stitch=None):
        """Red = missing nozzles, yellow = haze, blue = stitch, orange = cluster."""
        stitch = stitch or []
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()

        half = max(3, int(round(0.15 * spacing)))
        stitch_half = max(half + 2, int(round(0.30 * spacing)))
        # Keep the legend legible on a small crop without burying it.
        scale = float(np.clip(image.shape[1] / 1600.0, 0.35, 2.0))
        weight = max(1, int(round(2 * scale)))

        overlay = vis.copy()
        for defect in missing:
            cv2.rectangle(overlay, (defect['start_x'], defect['y'] - half),
                          (defect['end_x'], defect['y'] + half), (0, 0, 255), -1)
        for defect in hazy:
            # Heavier smears render more opaque than faint haze.
            shade = int(round(120 + 135 * defect.get('severity', 0.5)))
            cv2.rectangle(overlay, (defect['start_x'], defect['y'] - half),
                          (defect['end_x'], defect['y'] + half), (0, shade, shade), -1)
        cv2.addWeighted(vis, 0.6, overlay, 0.4, 0, dst=vis)

        # Stitch spans a whole line, so it is outlined rather than filled --
        # a filled bar would hide the missing/haze marks sitting inside it.
        for defect in stitch:
            cv2.rectangle(vis, (defect['start_x'], defect['y'] - stitch_half),
                          (defect['end_x'], defect['y'] + stitch_half),
                          (255, 80, 0), max(2, weight))
            cv2.putText(vis, f"stitch {defect['grade']} "
                        f"{defect['amplitude_px']:.1f}px",
                        (defect['start_x'] + 8, defect['y'] - stitch_half - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (255, 80, 0), weight)

        for region in regions:
            cv2.drawContours(vis, [region['contour']], -1, (0, 128, 255), 8)
            x, y, _w, _h = region['bbox']
            cv2.putText(vis, f"HIGH DENSITY: {region['missing_pixels']}px missing",
                        (x, max(30, y - 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2 * scale, (0, 128, 255), weight + 1)

        total_px = sum(d['missing_pixels'] for d in missing)
        text = (f"Missing: {len(missing)} gaps / {total_px} px "
                f"| Hazy: {len(hazy)} | Stitch: {len(stitch)} "
                f"| Density regions: {len(regions)}")
        cv2.putText(vis, text, (10, int(28 * scale) + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9 * scale, (0, 0, 255), weight + 1)
        return vis

    def _create_lines_debug(self, image, profiles):
        """Debug image: verticals magenta, trajectories green/orange, inserted red."""
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()

        for vline in self.band_detector.last_vlines:
            cv2.rectangle(vis, (vline['x0'] - 2, vline['y0']),
                          (vline['x1'] + 2, vline['y1']), (255, 0, 255), -1)

        palette = [(0, 200, 0), (255, 128, 0)]
        for band, profile in profiles:
            xs = np.arange(0, profile.width, 8)
            for line in range(profile.n_lines):
                ys = profile.true_y(line, xs).astype(np.int32)
                color = (0, 0, 255) if profile.inserted[line] else palette[line % 2]
                pts = np.stack([xs + band['x0'], ys], axis=1)
                cv2.polylines(vis, [pts.reshape(-1, 1, 2).astype(np.int32)],
                              False, color, 3)

        cv2.putText(vis, "Magenta: verticals | Green/Orange: trajectories | "
                    "Red: inserted (fully missing) lines", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)
        return vis

    def save_debug_images(self, output_dir, base_name):
        """Always save the detected-lines debug image."""
        if self._debug_lines_image is None:
            return None
        saved = save_image(output_dir, base_name, self._debug_lines_image,
                           'newpattern_detected_lines')
        return [saved] if saved else None
