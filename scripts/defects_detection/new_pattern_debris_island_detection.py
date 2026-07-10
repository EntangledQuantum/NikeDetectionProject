"""
New-Pattern (Dual-Band) Debris Island Detection

Detects dark debris on the changed island pattern, where each image contains
two horizontal-print bands separated by a gap, each flanked by vertical
boundary lines:

    [V prints V]   gap   [V prints V]

Approach (reuses existing island primitives, no edits to existing files):
  1. Split the image into its print bands with ``VerticalBandDetector``.
  2. For each band crop, find horizontal lines with ``BandLineDetector``.
  3. Reuse ``DebrisIslandDetector`` helpers to paint out horizontal lines and
     threshold the residual dark material; additionally paint out any vertical
     boundary lines so they are never counted as debris.
  4. Offset each band's contours back to full-image coordinates and build one
     composited visualization.
"""

import cv2
import numpy as np

from detector_base import BaseDetector
from debris_island_detection import DebrisIslandDetector
from utils.band_line_detector import BandLineDetector
from utils.vertical_band_detector import VerticalBandDetector, paint_vertical_lines_white
from utils.image_saver import save_image


class NewPatternDebrisIslandDetector(BaseDetector):
    """Detect debris across the two bands of a new-pattern island image."""

    def __init__(self, sensitivity='medium', debug=False):
        """Initialize the dual-band debris detector.

        Args:
            sensitivity: One of {'low', 'medium', 'high'}.
            debug: If True, store and optionally save intermediate debug images.
        """
        super().__init__()
        self.sensitivity = sensitivity
        self.debug = debug

        # Reused per-image helpers (line removal, debris thresholding, drawing)
        self.base = DebrisIslandDetector(sensitivity=sensitivity, debug=debug)
        self.band_detector = VerticalBandDetector()

        # Defensive thickness for painting out any vertical line inside a crop
        self.vline_paint_thickness = max(10, self.base.line_thickness)

        self._debug_bands_image = None
        self._debug_lines_removed_image = None

    def detect(self, image, image_path=None):
        """Run dual-band debris detection.

        Args:
            image: Input island image (BGR or grayscale).
            image_path: Accepted for interface compatibility (unused; band and
                vertical-line geometry is auto-detected).

        Returns:
            tuple: (visualization_bgr, defects)
        """
        reference_width = image.shape[1]

        if len(image.shape) == 3:
            gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_full = image.copy()

        bands = self.band_detector.detect(image, self.debug)

        all_contours = []
        total_matched_lines = 0
        lines_removed_full = gray_full.copy() if self.debug else None

        for band in bands:
            x0, x1 = band['x0'], band['x1']
            if x1 - x0 < 5:
                continue

            crop = image[:, x0:x1]
            gray_crop = gray_full[:, x0:x1]

            # Horizontal lines within this band, scaled to the full image width
            line_detector = BandLineDetector(self.sensitivity, reference_width=reference_width)
            matched_lines, _, _, _, _ = line_detector.detect_lines(crop, self.debug)
            total_matched_lines += len(matched_lines)

            # Remove horizontal print lines (reused helper)
            lines_removed = self.base.remove_lines_from_image(gray_crop, matched_lines)

            # Remove any vertical boundary line that falls inside the crop
            vlines_local = [vx - x0 for vx in band['vline_xs'] if x0 <= vx < x1]
            if vlines_local:
                lines_removed = paint_vertical_lines_white(
                    lines_removed, vlines_local, self.vline_paint_thickness)

            # Detect debris on the cleaned band (reused helper)
            debris_contours, _ = self.base.detect_debris(lines_removed)

            # Offset contours back to full-image coordinates
            for contour in debris_contours:
                shifted = contour.copy()
                shifted[:, :, 0] += x0
                all_contours.append(shifted)

            if self.debug and lines_removed_full is not None:
                lines_removed_full[:, x0:x1] = lines_removed

        visualization = self.base.create_debris_visualization(image, all_contours, [])

        if self.debug:
            self._debug_bands_image = self._create_bands_visualization(image, bands)
            self._debug_lines_removed_image = lines_removed_full
            print(f"NewPatternDebrisIsland: {len(bands)} band(s), "
                  f"{total_matched_lines} horizontal lines, "
                  f"{len(all_contours)} debris regions")

        defects = self._build_defects(bands, total_matched_lines, all_contours)
        return visualization, defects

    def _build_defects(self, bands, total_matched_lines, all_contours):
        """Assemble the structured defect output list."""
        defects = [{
            'type': 'bands_detected',
            'band_count': len(bands),
            'bands': [{'index': b['index'], 'x0': b['x0'], 'x1': b['x1'],
                       'vline_xs': b['vline_xs']} for b in bands],
            'total_lines': total_matched_lines,
        }]

        if all_contours:
            debris_info = []
            for contour in all_contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                debris_info.append({
                    'bbox': (int(x), int(y), int(w), int(h)),
                    'area': float(area),
                    'center': (int(x + w / 2), int(y + h / 2)),
                })
            defects.append({
                'type': 'debris_detected',
                'debris_count': len(all_contours),
                'debris_regions': debris_info,
            })

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
        images_to_save = {
            '_debug_bands_image': 'newpattern_debris_bands',
            '_debug_lines_removed_image': 'newpattern_debris_lines_removed',
        }
        for attr, suffix in images_to_save.items():
            image = getattr(self, attr, None)
            if image is not None:
                saved_path = save_image(output_dir, base_name, image, suffix)
                if saved_path:
                    debug_paths.append(saved_path)

        return debug_paths if debug_paths else None
