"""
New-Pattern (Dual-Band) Line Defect Detection

Detects missing (ink gap) and jagged (zig-zag) horizontal print-line segments
on the changed island pattern, where each image contains two horizontal-print
bands separated by a gap, each flanked by vertical boundary lines:

    [V prints V]   gap   [V prints V]

Approach (reuses existing island primitives, no edits to existing files):
  1. Split the image into its print bands with ``VerticalBandDetector``.
  2. For each band crop, find horizontal lines with ``BandLineDetector``.
  3. Reuse ``LineDefectDetector.scan_line_for_defects`` to walk each line and
     flag missing/jagged segments. Invalid-slope matches are skipped and ghost
     endpoints are repaired using the band's median slope for a stable path.
  4. Offset each band's defects back to full-image coordinates and build one
     composited visualization.
"""

import cv2
import numpy as np

from detector_base import BaseDetector
from line_defect_detection import LineDefectDetector
from utils.band_line_detector import BandLineDetector
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

        self._debug_bands_image = None

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
        total_matched_lines = 0

        for band in bands:
            x0, x1 = band['x0'], band['x1']
            if x1 - x0 < 5:
                continue

            crop = image[:, x0:x1]
            gray_crop = gray_full[:, x0:x1]

            line_detector = BandLineDetector(self.sensitivity, reference_width=reference_width)
            matched_lines, _, _, _, _ = line_detector.detect_lines(crop, self.debug)
            total_matched_lines += len(matched_lines)

            # Binary line map for the band (dark ink -> white)
            _, binary_crop = cv2.threshold(gray_crop, 127, 255, cv2.THRESH_BINARY_INV)

            # Band median slope for repairing ghost endpoints
            valid_slopes = [m['slope'] for m in matched_lines if m.get('valid_slope', True)]
            median_slope = float(np.median(valid_slopes)) if valid_slopes else None

            for match in matched_lines:
                repaired = self._repair_match(match, median_slope)
                if repaired is None:
                    continue

                band_defects, _ = self.base.scan_line_for_defects(binary_crop, repaired)
                for defect in band_defects:
                    all_defects.append(self._offset_defect(defect, x0))

        visualization = self.base.create_combined_visualization(image, all_defects)

        if self.debug:
            self._debug_bands_image = self._create_bands_visualization(image, bands)
            missing = len([d for d in all_defects if d['type'] == 'missing_line'])
            jagged = len([d for d in all_defects if d['type'] == 'jagged_line'])
            print(f"NewPatternLineDefect: {len(bands)} band(s), "
                  f"{total_matched_lines} horizontal lines, "
                  f"{missing} missing, {jagged} jagged")

        defects = self._build_defects(bands, total_matched_lines, all_defects)
        return visualization, defects

    def _repair_match(self, match, median_slope):
        """Skip invalid-slope matches; repair ghost endpoints via median slope.

        Args:
            match: Matched line dict from ``BandLineDetector``.
            median_slope: Band median slope, or None if unavailable.

        Returns:
            A ``{'left', 'right'}`` dict suitable for ``scan_line_for_defects``,
            or None if the match should be skipped.
        """
        if not match.get('valid_slope', True):
            return None

        left = dict(match['left'])
        right = dict(match['right'])

        if median_slope is not None and right['x'] != left['x']:
            span = right['x'] - left['x']
            if left.get('type') == 'ghost' and right.get('type') != 'ghost':
                left['y'] = int(right['y'] - median_slope * span)
            elif right.get('type') == 'ghost' and left.get('type') != 'ghost':
                right['y'] = int(left['y'] + median_slope * span)

        return {'left': left, 'right': right}

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

    def _build_defects(self, bands, total_matched_lines, all_defects):
        """Assemble the structured defect output list."""
        missing = [d for d in all_defects if d['type'] == 'missing_line']
        jagged = [d for d in all_defects if d['type'] == 'jagged_line']

        defects = [{
            'type': 'bands_detected',
            'band_count': len(bands),
            'bands': [{'index': b['index'], 'x0': b['x0'], 'x1': b['x1'],
                       'vline_xs': b['vline_xs']} for b in bands],
            'total_lines': total_matched_lines,
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
            cv2.rectangle(vis, (band['x0'], 0), (band['x1'], height - 1), (0, 255, 0), 3)
            cv2.putText(vis, f"Band {band['index']}", (band['x0'] + 5, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            for vx in band['vline_xs']:
                cv2.line(vis, (vx, 0), (vx, height - 1), (255, 0, 255), 2)

        return vis

    def save_debug_images(self, output_dir, base_name):
        """Persist debug images (if available) to the output directory."""
        if not self.debug:
            return None

        debug_paths = []
        image = getattr(self, '_debug_bands_image', None)
        if image is not None:
            saved_path = save_image(output_dir, base_name, image, 'newpattern_line_defect_bands')
            if saved_path:
                debug_paths.append(saved_path)

        return debug_paths if debug_paths else None
