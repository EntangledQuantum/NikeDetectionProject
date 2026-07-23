"""
New-Pattern (Dual-Band) Line Defect Detection

Detects missing (ink gap / missing nozzle) and jagged (zig-zag) horizontal
print-line segments on the changed island pattern, where each image contains
two horizontal-print bands separated by a gap, each flanked by vertical
boundary lines (4 vertical lines total):

    [V prints V]   gap   [V prints V]

Pipeline (reuses existing island primitives, no edits to existing files):
  1. Robustly detect the 4 vertical boundary lines and segregate the two
     print regions with ``VerticalBandDetector`` (morphology-based, tolerant
     of noisy lines and imperfect line start/end points).
  2. Per band, find coarse horizontal-line endpoints with ``BandLineDetector``
     (kernel scaling referenced to the full image width).
  3. Refine each line with ``BandLineRefiner``: sample the actual ink along
     the trajectory, robust-fit a per-line slope/intercept (constant slope /
     spacing across lines is NOT assumed), and extrapolate the start and end
     points to the band's inner bounds. Lines whose start/end ink is missing
     (missing nozzles) therefore still get a full-width trajectory, and lines
     with too little ink are interpolated from their fitted neighbors.
  4. Walk every refined trajectory across the full band width with the
     reused ``LineDefectDetector.scan_line_for_defects`` to flag missing /
     jagged segments - including gaps at the very start or end of a line.
  5. Offset each band's defects back to full-image coordinates and build one
     composited visualization.

The vertical lines are structural only; no defect detection runs on them,
and the central gap between the two bands is never scanned for missing
nozzles.
"""

import cv2
import numpy as np

from detector_base import BaseDetector
from line_defect_detection import LineDefectDetector
from utils.band_line_detector import BandLineDetector
from utils.band_line_refiner import BandLineRefiner
from utils.vertical_band_detector import VerticalBandDetector
from utils.image_saver import save_image


class NewPatternLineDefectDetector(BaseDetector):
    """Detect line defects across the two bands of a new-pattern island image."""

    def __init__(self, sensitivity='medium', debug=False):
        """Initialize the dual-band line defect detector.

        Args:
            sensitivity: One of {'low', 'medium', 'high'}.
            debug: If True, store and optionally save intermediate debug images.
        """
        super().__init__()
        self.sensitivity = sensitivity
        self.debug = debug

        # Reused per-line walk logic and combined visualization
        self.base = LineDefectDetector(sensitivity=sensitivity, debug=debug)
        self.base.debug = debug  # LineDefectDetector forces debug on in __init__
        self.band_detector = VerticalBandDetector()
        self.refiner = BandLineRefiner()

        self._debug_bands_image = None
        self._debug_trajectories_image = None

    def detect(self, image, image_path=None):
        """Run dual-band line defect detection.

        Args:
            image: Input island image (BGR or grayscale).
            image_path: Accepted for interface compatibility (unused).

        Returns:
            tuple: (visualization_bgr, defects)
        """
        reference_width = image.shape[1]

        if len(image.shape) == 3:
            gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_full = image.copy()

        bands = self.band_detector.detect(image, self.debug)

        all_defects = []
        band_line_reports = {}
        total_lines = 0

        for band in bands:
            x0, x1 = band['x0'], band['x1']
            if x1 - x0 < 5:
                continue

            crop = image[:, x0:x1 + 1]
            gray_crop = gray_full[:, x0:x1 + 1]

            # Coarse horizontal-line endpoints (kernels scaled to full width)
            line_detector = BandLineDetector(self.sensitivity,
                                             reference_width=reference_width)
            matched_lines, _, _, _, _ = line_detector.detect_lines(crop, self.debug)

            # Binary line map for the band (dark ink -> white)
            _, binary_crop = cv2.threshold(gray_crop, 127, 255,
                                           cv2.THRESH_BINARY_INV)

            # Refine: robust per-line fit + extrapolation to the band bounds
            refined_lines = self.refiner.refine(binary_crop, matched_lines,
                                                self.debug)
            total_lines += len(refined_lines)
            band_line_reports[band['index']] = refined_lines

            for refined in refined_lines:
                band_defects, _ = self.base.scan_line_for_defects(binary_crop,
                                                                  refined)
                for defect in band_defects:
                    all_defects.append(self._offset_defect(defect, x0))

        visualization = self.base.create_combined_visualization(image, all_defects)

        if self.debug:
            self._debug_bands_image = self._create_bands_visualization(image, bands)
            self._debug_trajectories_image = self._create_trajectories_visualization(
                image, bands, band_line_reports)
            missing = len([d for d in all_defects if d['type'] == 'missing_line'])
            jagged = len([d for d in all_defects if d['type'] == 'jagged_line'])
            print(f"NewPatternLineDefect: {len(bands)} band(s), "
                  f"{total_lines} horizontal lines, "
                  f"{missing} missing, {jagged} jagged")

        defects = self._build_defects(bands, band_line_reports, total_lines,
                                      all_defects)
        return visualization, defects

    def _offset_defect(self, defect, x0):
        """Shift a defect's x coordinates into full-image space."""
        shifted = dict(defect)
        if defect['type'] == 'missing_line':
            shifted['start_x'] = defect['start_x'] + x0
            shifted['end_x'] = defect['end_x'] + x0
            lx, ly = defect['location']
            shifted['location'] = (lx + x0, ly)
        elif defect['type'] == 'jagged_line':
            shifted['x'] = defect['x'] + x0
            lx, ly = defect['location']
            shifted['location'] = (lx + x0, ly)
        return shifted

    def _build_defects(self, bands, band_line_reports, total_lines, all_defects):
        """Assemble the structured defect output list."""
        missing = [d for d in all_defects if d['type'] == 'missing_line']
        jagged = [d for d in all_defects if d['type'] == 'jagged_line']

        band_entries = []
        for band in bands:
            lines = band_line_reports.get(band['index'], [])
            band_entries.append({
                'index': band['index'],
                'x0': band['x0'],
                'x1': band['x1'],
                'vline_xs': band['vline_xs'],
                'line_count': len(lines),
                'lines': [{
                    'start': (line['left']['x'] + band['x0'], line['left']['y']),
                    'end': (line['right']['x'] + band['x0'], line['right']['y']),
                    'slope': line['slope'],
                    'source': line['source'],
                    'n_samples': line['n_samples'],
                } for line in lines],
            })

        defects = [{
            'type': 'bands_detected',
            'band_count': len(bands),
            'bands': band_entries,
            'total_lines': total_lines,
        }]
        defects.extend(missing)
        defects.extend(jagged)
        return defects

    def _create_bands_visualization(self, image, bands):
        """Draw detected band crops and vertical boundary lines for debugging."""
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()

        height = vis.shape[0]
        for band in bands:
            cv2.rectangle(vis, (band['x0'], 0), (band['x1'], height - 1),
                          (0, 255, 0), 3)
            cv2.putText(vis, f"Band {band['index']}", (band['x0'] + 5, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            for vx in band['vline_xs']:
                cv2.line(vis, (vx, 0), (vx, height - 1), (255, 0, 255), 2)

        return vis

    def _create_trajectories_visualization(self, image, bands, band_line_reports):
        """Draw the refined full-width line trajectories for debugging."""
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()

        for band in bands:
            x0 = band['x0']
            for line in band_line_reports.get(band['index'], []):
                color = (0, 255, 0) if line['source'] == 'fitted' else (255, 165, 0)
                cv2.line(vis,
                         (line['left']['x'] + x0, line['left']['y']),
                         (line['right']['x'] + x0, line['right']['y']),
                         color, 2)
                cv2.circle(vis, (line['left']['x'] + x0, line['left']['y']),
                           4, (255, 0, 0), -1)
                cv2.circle(vis, (line['right']['x'] + x0, line['right']['y']),
                           4, (0, 0, 255), -1)

        cv2.putText(vis, "Green: fitted | Orange: interpolated", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return vis

    def save_debug_images(self, output_dir, base_name):
        """Persist debug images (if available) to the output directory."""
        if not self.debug:
            return None

        debug_paths = []
        images_to_save = {
            '_debug_bands_image': 'newpattern_line_defect_bands',
            '_debug_trajectories_image': 'newpattern_line_defect_trajectories',
        }
        for attr, suffix in images_to_save.items():
            image = getattr(self, attr, None)
            if image is not None:
                saved_path = save_image(output_dir, base_name, image, suffix)
                if saved_path:
                    debug_paths.append(saved_path)

        return debug_paths if debug_paths else None
