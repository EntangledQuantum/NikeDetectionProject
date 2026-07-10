"""
New-Pattern (Dual-Band) Overspray Island Detection

Detects colored overspray on the changed island pattern, where each image
contains two horizontal-print bands separated by a gap, each flanked by
vertical boundary lines:

    [V prints V]   gap   [V prints V]

Approach (reuses existing island primitives, no edits to existing files):
  1. Split the image into its print bands with ``VerticalBandDetector``.
  2. For each band crop, find horizontal lines with ``BandLineDetector``.
  3. Reuse ``OversprayIslandDetector`` helpers to paint out horizontal lines,
     threshold colored regions, and group nearby regions; additionally paint
     out vertical boundary lines. Grouping stays within a band so overspray is
     never merged across the central gap.
  4. Offset each band's regions back to full-image coordinates and build one
     composited visualization.
"""

import cv2
import numpy as np

from detector_base import BaseDetector
from overspray_island_detection import OversprayIslandDetector
from utils.band_line_detector import BandLineDetector
from utils.vertical_band_detector import VerticalBandDetector, paint_vertical_lines_white
from utils.image_saver import save_image


class NewPatternOversprayIslandDetector(BaseDetector):
    """Detect overspray across the two bands of a new-pattern island image."""

    def __init__(self, sensitivity='medium', debug=False):
        """Initialize the dual-band overspray detector.

        Args:
            sensitivity: One of {'low', 'medium', 'high'}.
            debug: If True, store and optionally save intermediate debug images.
        """
        super().__init__()
        self.sensitivity = sensitivity
        self.debug = debug

        # Reused per-image helpers (line removal, colored regions, grouping)
        self.base = OversprayIslandDetector(sensitivity=sensitivity, debug=debug)
        self.band_detector = VerticalBandDetector()

        self.vline_paint_thickness = max(10, self.base.line_thickness)

        self._debug_bands_image = None
        self._debug_lines_removed_image = None

    def detect(self, image, image_path=None):
        """Run dual-band overspray detection.

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

        all_regions = []
        total_matched_lines = 0
        lines_removed_full = gray_full.copy() if self.debug else None

        for band in bands:
            x0, x1 = band['x0'], band['x1']
            if x1 - x0 < 5:
                continue

            crop = image[:, x0:x1]
            gray_crop = gray_full[:, x0:x1]

            line_detector = BandLineDetector(self.sensitivity, reference_width=reference_width)
            matched_lines, _, _, _, _ = line_detector.detect_lines(crop, self.debug)
            total_matched_lines += len(matched_lines)

            lines_removed = self.base.remove_lines_from_image(gray_crop, matched_lines)

            vlines_local = [vx - x0 for vx in band['vline_xs'] if x0 <= vx < x1]
            if vlines_local:
                lines_removed = paint_vertical_lines_white(
                    lines_removed, vlines_local, self.vline_paint_thickness)

            # Colored regions within the band, then group within the band only
            overspray_regions, _ = self.base.detect_colored_regions(lines_removed)
            overspray_regions = self.base.group_nearby_regions(overspray_regions)

            for region in overspray_regions:
                all_regions.append(self._offset_region(region, x0))

            if self.debug and lines_removed_full is not None:
                lines_removed_full[:, x0:x1] = lines_removed

        visualization = self.base.create_overspray_visualization(image, all_regions, [])

        if self.debug:
            self._debug_bands_image = self._create_bands_visualization(image, bands)
            self._debug_lines_removed_image = lines_removed_full
            print(f"NewPatternOversprayIsland: {len(bands)} band(s), "
                  f"{total_matched_lines} horizontal lines, "
                  f"{len(all_regions)} overspray regions")

        defects = self._build_defects(bands, total_matched_lines, all_regions)
        return visualization, defects

    def _offset_region(self, region, x0):
        """Shift a region dict (contour, center, bbox) into full-image coords."""
        shifted = dict(region)
        contour = region['contour'].copy()
        contour[:, :, 0] += x0
        shifted['contour'] = contour

        cx, cy = region['center']
        shifted['center'] = (cx + x0, cy)

        bx, by, bw, bh = region['bbox']
        shifted['bbox'] = (bx + x0, by, bw, bh)

        return shifted

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
