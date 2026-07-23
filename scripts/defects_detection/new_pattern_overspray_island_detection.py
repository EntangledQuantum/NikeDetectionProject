"""
New-Pattern (Dual-Band) Overspray Island Detection

Detects colored overspray on the changed island pattern, where each image
contains two horizontal-print bands separated by a gap, each flanked by
vertical boundary lines (4 vertical lines total):

    [V prints V]   gap   [V prints V]

Overspray can happen ANYWHERE in the image - inside the bands, in the
central gap, or outside the boundary lines - so detection runs on the full
image after masking out all printed structure:

  1. Robustly detect the 4 vertical boundary lines and the two print bands
     with ``VerticalBandDetector``.
  2. Per band, find and refine the horizontal print lines
     (``BandLineDetector`` + ``BandLineRefiner``: robust per-line fit,
     extrapolated to the band bounds, tolerant of missing start/end ink).
  3. Paint every refined horizontal line white on the FULL grayscale image
     (reusing ``OversprayIslandDetector`` thickness conventions), slightly
     extended past the band bounds so line ends never leak.
  4. Paint the vertical boundary lines white using their measured x-extent
     envelope plus a speckle-halo pad (``paint_vertical_line_regions``), so
     noisy vertical lines are never counted as overspray.
  5. Run the reused ``OversprayIslandDetector`` color thresholding +
     morphology + proximity grouping over the full cleaned image.
"""

import cv2
import numpy as np

from detector_base import BaseDetector
from overspray_island_detection import OversprayIslandDetector
from utils.band_line_detector import BandLineDetector
from utils.band_line_refiner import BandLineRefiner
from utils.vertical_band_detector import VerticalBandDetector, paint_vertical_line_regions
from utils.image_saver import save_image


class NewPatternOversprayIslandDetector(BaseDetector):
    """Detect overspray anywhere on a new-pattern island image."""

    # Horizontal lines are painted slightly past the band bounds so residual
    # ink at the line ends (next to the vertical lines) never leaks through.
    LINE_PAINT_EXTENSION = 20

    def __init__(self, sensitivity='medium', debug=False):
        """Initialize the dual-band overspray detector.

        Args:
            sensitivity: One of {'low', 'medium', 'high'}.
            debug: If True, store and optionally save intermediate debug images.
        """
        super().__init__()
        self.sensitivity = sensitivity
        self.debug = debug

        # Reused per-image helpers (colored regions, grouping, drawing)
        self.base = OversprayIslandDetector(sensitivity=sensitivity, debug=debug)
        self.band_detector = VerticalBandDetector()
        self.refiner = BandLineRefiner()

        self._debug_bands_image = None
        self._debug_lines_removed_image = None

    def detect(self, image, image_path=None):
        """Run full-image overspray detection with printed structure masked out.

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

        # ---- Mask horizontal print lines (per band, refined) ------------
        lines_removed = gray_full.copy()
        total_lines = 0

        for band in bands:
            x0, x1 = band['x0'], band['x1']
            if x1 - x0 < 5:
                continue

            crop = image[:, x0:x1 + 1]
            gray_crop = gray_full[:, x0:x1 + 1]

            line_detector = BandLineDetector(self.sensitivity,
                                             reference_width=reference_width)
            matched_lines, _, _, _, _ = line_detector.detect_lines(crop, self.debug)

            _, binary_crop = cv2.threshold(gray_crop, 127, 255,
                                           cv2.THRESH_BINARY_INV)
            refined_lines = self.refiner.refine(binary_crop, matched_lines,
                                                self.debug)
            total_lines += len(refined_lines)

            ext = self.LINE_PAINT_EXTENSION
            for line in refined_lines:
                lx = line['left']['x'] + x0
                rx = line['right']['x'] + x0
                slope = line['slope']
                p_left = (lx - ext, int(round(line['left']['y'] - slope * ext)))
                p_right = (rx + ext, int(round(line['right']['y'] + slope * ext)))
                cv2.line(lines_removed, p_left, p_right, 255,
                         self.base.line_thickness * 3)

        # ---- Mask the vertical boundary lines (measured envelopes) ------
        vlines = self._collect_vlines(bands)
        if vlines:
            lines_removed = paint_vertical_line_regions(lines_removed, vlines)

        # ---- Overspray on the full cleaned image (reused helpers) -------
        overspray_regions, _ = self.base.detect_colored_regions(lines_removed)
        overspray_regions = self.base.group_nearby_regions(overspray_regions)

        visualization = self.base.create_overspray_visualization(
            image, overspray_regions, [])

        if self.debug:
            self._debug_bands_image = self._create_bands_visualization(image, bands)
            self._debug_lines_removed_image = lines_removed
            print(f"NewPatternOversprayIsland: {len(bands)} band(s), "
                  f"{total_lines} horizontal lines, "
                  f"{len(overspray_regions)} overspray regions")

        defects = self._build_defects(bands, total_lines, overspray_regions)
        return visualization, defects

    def _collect_vlines(self, bands):
        """Union of selected boundary lines and per-band vertical lines."""
        vlines = {id(v): v for v in self.band_detector.last_vlines}
        for band in bands:
            for v in band.get('vlines', []):
                vlines[id(v)] = v
        return list(vlines.values())

    def _build_defects(self, bands, total_matched_lines, all_regions):
        """Assemble the structured defect output list."""
        defects = [{
            'type': 'bands_detected',
            'band_count': len(bands),
            'bands': [{'index': b['index'], 'x0': b['x0'], 'x1': b['x1'],
                       'vline_xs': b['vline_xs']} for b in bands],
            'total_lines': total_matched_lines,
        }]

        if all_regions:
            overspray_info = []
            for region in all_regions:
                overspray_info.append({
                    'bbox': region['bbox'],
                    'area': float(region['area']),
                    'center': region['center'],
                    'density': float(region.get('density', 0)),
                    'merged_count': region.get('merged_count', 1),
                    'original_area': float(region.get('original_area', region['area'])),
                })
            defects.append({
                'type': 'overspray_detected',
                'overspray_count': len(all_regions),
                'overspray_regions': overspray_info,
                'min_area_threshold': self.base.overspray_min_area,
                'max_grouping_distance': self.base.overspray_max_distance,
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
            cv2.rectangle(vis, (band['x0'], 0), (band['x1'], height - 1),
                          (0, 255, 0), 3)
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
            '_debug_bands_image': 'newpattern_overspray_bands',
            '_debug_lines_removed_image': 'newpattern_overspray_lines_removed',
        }
        for attr, suffix in images_to_save.items():
            image = getattr(self, attr, None)
            if image is not None:
                saved_path = save_image(output_dir, base_name, image, suffix)
                if saved_path:
                    debug_paths.append(saved_path)

        return debug_paths if debug_paths else None
