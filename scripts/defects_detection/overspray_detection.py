"""
Overspray Detection Algorithm for Stripe Images
Detects overspray defects by masking vertical line then finding scattered pixels

Author: Assistant
Date: 2024
"""

import cv2
import numpy as np
from scipy import ndimage
import os
from utils.vertical_line_detector import VerticalLineDetector


class OversprayDetector:
    """Detect overspray defects by grid scanning and scatter analysis.

    The detector preprocesses the image to a binary representation where spray
    pixels are white, then slides a kernel across a grid to measure spatial
    scatter of white pixels. Regions exceeding a scatter threshold are flagged
    as overspray, and adjacent kernels can be merged into larger regions.
    """
    
    def __init__(self, kernel_size=30, step_size=None, scatter_threshold=0.3,
                 min_pixels_threshold=0.05, sensitivity='medium', debug=False):
        """Configure grid kernel size, step, and thresholds.

        Args:
            kernel_size: Square kernel size in pixels for grid scanning.
            step_size: Grid step in pixels; defaults to `kernel_size` (no overlap).
            scatter_threshold: Threshold (0..1) for scatter metric to flag overspray.
            min_pixels_threshold: Minimum white pixel ratio within kernel to analyze.
            sensitivity: One of {'low', 'medium', 'high'}; adjusts defaults.
            debug: If True, stores kernel states for visualization.
        """
        self.kernel_size = kernel_size
        self.step_size = step_size if step_size else kernel_size
        self.scatter_threshold = scatter_threshold
        self.min_pixels_threshold = min_pixels_threshold
        self.debug = debug # force debug mode for now
        self.sensitivity = sensitivity
        self.exclusion_zones = []  # Will be populated from vertical line detector
        
        print(f"Sensitivity: {sensitivity}")
        
        # Initialize vertical line detector for stripe images
        self.vertical_line_detector = VerticalLineDetector(sensitivity)
        
        # Adjust parameters based on sensitivity
        if sensitivity == 'high':
            self.kernel_size = 20
            self.step_size = 15  # More overlap for high sensitivity
            self.scatter_threshold = 0.2  # Lower threshold - more sensitive
            self.min_pixels_threshold = 0.03  # Detect smaller amounts of scatter
            self.line_mask_thickness = 5
        elif sensitivity == 'low':
            self.kernel_size = 50
            self.step_size = 50  # No overlap
            self.scatter_threshold = 0.5  # Higher threshold - less sensitive
            self.min_pixels_threshold = 0.1  # Need more pixels to consider
            self.line_mask_thickness = 5
        else:  # medium
            self.kernel_size = 20
            self.step_size = 50  # Small overlap
            self.scatter_threshold = 0.5
            self.min_pixels_threshold = 50
            self.line_mask_thickness = 5  # Thicker mask for medium
        
        # Store debug images
        self._debug_line_polygon = None
        self._debug_line_masked_image = None
        self._debug_polygon_with_coords = None
    
    def preprocess_image(self, image):
        """Preprocess input image into a binary mask for overspray analysis.

        Steps: grayscale, CLAHE contrast enhancement, adaptive threshold, and
        optional inversion so overspray pixels are white.

        Args:
            image: Input image (BGR or grayscale).

        Returns:
            Binary uint8 image with overspray pixels as white (255).
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Binary threshold
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, 21, -5)
        
        # Invert if necessary (we want spray pixels to be white)
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
        
        return binary
    
    def calculate_scatter_metric(self, kernel_region):
        """Compute a scatter score for white pixels within a kernel region.

        Combines spread (std of coordinates normalized by kernel size) and
        distance variance between sampled points, with weighting.

        Args:
            kernel_region: Binary kernel crop (uint8) from the preprocessed image.

        Returns:
            float in [0, 1] representing scatter; higher implies overspray-like.
        """
        if kernel_region.size == 0:
            return 0
        
        # Get white pixel positions
        white_pixels = np.where(kernel_region > 0)
        
        if len(white_pixels[0]) == 0:
            return 0
        
        # Calculate metrics
        total_pixels = kernel_region.size
        white_pixel_count = len(white_pixels[0])
        white_ratio = white_pixel_count / total_pixels
        
        # If too few pixels, not overspray
        if white_ratio < self.min_pixels_threshold:
            return 0
        
        # If too many pixels, likely solid region not overspray
        if white_ratio > 0.7:
            return 0
        
        # Calculate spatial distribution - how spread out are the pixels
        if white_pixel_count > 1:
            # Calculate standard deviation of positions
            y_std = np.std(white_pixels[0])
            x_std = np.std(white_pixels[1])
            
            # Normalize by kernel size
            y_spread = y_std / (kernel_region.shape[0] / 2)
            x_spread = x_std / (kernel_region.shape[1] / 2)
            
            # Combined spread metric
            spread = (y_spread + x_spread) / 2
            
            # Also consider the distribution pattern
            # Calculate distances between consecutive pixels
            points = np.column_stack(white_pixels)
            if len(points) > 2:
                # Calculate pairwise distances
                distances = []
                for i in range(min(len(points), 20)):  # Sample up to 20 points
                    for j in range(i + 1, min(len(points), 20)):
                        dist = np.sqrt((points[i][0] - points[j][0])**2 + 
                                     (points[i][1] - points[j][1])**2)
                        distances.append(dist)
                
                if distances:
                    # High variance in distances indicates scattered pattern
                    dist_variance = np.var(distances) / (self.kernel_size**2)
                    
                    # Combine spread and variance metrics
                    scatter_metric = (spread * 0.6 + dist_variance * 0.4)
                else:
                    scatter_metric = spread
            else:
                scatter_metric = spread
            
            # Adjust by pixel ratio - moderate amounts of pixels are more likely overspray
            if 0.05 < white_ratio < 0.3:
                scatter_metric *= 1.2  # Boost score for typical overspray density
            
            return min(scatter_metric, 1.0)  # Cap at 1.0
        
        return 0
    
    def scan_grid(self, binary_image):
        """Slide kernel over a grid to identify overspray kernels.

        Args:
            binary_image: Preprocessed binary image with overspray pixels white.

        Returns:
            tuple: (kernel_states, defects)
                - kernel_states: List of per-kernel diagnostics (when debug)
                - defects: List of kernel-level detections or merged regions
        """
        height, width = binary_image.shape
        kernel_states = []
        defects = []
        
        # Scan in grid pattern
        y = self.kernel_size // 2
        while y < height - self.kernel_size // 2:
            x = self.kernel_size // 2
            while x < width - self.kernel_size // 2:
                # Extract kernel region
                y1 = max(0, y - self.kernel_size // 2)
                y2 = min(height, y + self.kernel_size // 2)
                x1 = max(0, x - self.kernel_size // 2)
                x2 = min(width, x + self.kernel_size // 2)
                
                kernel_region = binary_image[y1:y2, x1:x2]
                
                # Calculate scatter metric
                scatter_metric = self.calculate_scatter_metric(kernel_region)
                
                # Determine if this is overspray
                is_overspray = scatter_metric > self.scatter_threshold
                
                # Record kernel state for debug
                if self.debug:
                    kernel_states.append({
                        'x': x,
                        'y': y,
                        'has_overspray': is_overspray,
                        'scatter_metric': scatter_metric,
                        'bbox': (x1, y1, x2, y2)
                    })
                
                # Record defect if found
                if is_overspray:
                    defects.append({
                        'type': 'overspray',
                        'x': x,
                        'y': y,
                        'location': (x, y),
                        'scatter_metric': float(scatter_metric),
                        'kernel_size': self.kernel_size,
                        'bbox': (x1, y1, x2, y2)
                    })
                    
                    # if self.debug:
                    #     print(f"Overspray detected at X={x}, Y={y}: scatter={scatter_metric:.3f}")
                
                # Move to next position
                x += self.step_size
            
            # Move to next row
            y += self.step_size
        
        # Merge nearby defects to create regions
        if defects and not self.debug:
            defects = self.merge_nearby_defects(defects)
        
        return kernel_states, defects
    
    def merge_nearby_defects(self, defects):
        """Merge spatially adjacent kernel-level detections into regions.

        Args:
            defects: List of kernel-level overspray detections with bboxes.

        Returns:
            List of merged region dicts with bbox, area, center, and metrics.
        """
        if not defects:
            return defects
        
        # Create a mask for defect regions
        # Find image bounds
        max_x = max(d['x'] for d in defects) + self.kernel_size
        max_y = max(d['y'] for d in defects) + self.kernel_size
        
        # Create mask
        defect_mask = np.zeros((max_y, max_x), dtype=np.uint8)
        
        # Fill defect regions
        for defect in defects:
            x1, y1, x2, y2 = defect['bbox']
            defect_mask[y1:y2, x1:x2] = 255
        
        # Apply morphological closing to merge nearby regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (self.kernel_size, self.kernel_size))
        merged_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours of merged regions
        contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        # Create new defect list from merged regions
        merged_defects = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w // 2
            cy = y + h // 2
            
            # Calculate average scatter metric for this region
            region_metrics = [d['scatter_metric'] for d in defects 
                            if x <= d['x'] <= x + w and y <= d['y'] <= y + h]
            avg_metric = np.mean(region_metrics) if region_metrics else 0
            
            merged_defects.append({
                'type': 'overspray_region',
                'x': cx,
                'y': cy,
                'location': (cx, cy),
                'scatter_metric': float(avg_metric),
                'bbox': (x, y, x + w, y + h),
                'area': cv2.contourArea(contour),
                'kernel_count': len(region_metrics)
            })
        
        return merged_defects
    
    def mask_vertical_line(self, gray_image, polygons_list):
        """Paint out all vertical line polygons to avoid detecting them as overspray
        
        Args:
            gray_image: Grayscale image
            polygons_list: List of polygon dicts from vertical line detector (one per print head)
            
        Returns:
            Grayscale image with vertical line segments masked out (painted white)
        """
        if not polygons_list:
            return gray_image
        
        line_masked = gray_image.copy()
        
        # Mask out each polygon (print head segment)
        for poly_data in polygons_list:
            polygon = poly_data['polygon']
            
            # Fill the polygon with white to mask out this segment
            cv2.fillPoly(line_masked, [polygon], 255)
            
            # Also add extra thickness around the polygon edges
            cv2.polylines(line_masked, [polygon], True, 255, self.line_mask_thickness * 2)
        
        return line_masked
    
    def detect_colored_regions(self, line_masked_image):
        """Detect colored (non-white) regions as overspray candidates
        Similar to overspray_island_detection.py approach
        
        Args:
            line_masked_image: Grayscale image with vertical line masked out
            
        Returns:
            tuple: (overspray_regions, colored_mask)
        """
        # Find anything that's not white or near-white (colored areas)
        # After line is painted white, only overspray should remain dark/colored
        
        # Background threshold - similar to island detector
        if self.sensitivity == 'high':
            background_threshold = 140
            min_area = 100
        elif self.sensitivity == 'low':
            background_threshold = 100
            min_area = 1000
        else:  # medium
            background_threshold = 120
            min_area = 500
        
        # Create mask for colored areas
        _, colored_mask = cv2.threshold(line_masked_image, background_threshold, 255, cv2.THRESH_BINARY_INV)
        
        if self.debug:
            print(f"Colored mask: {np.sum(colored_mask > 0)} pixels below threshold {background_threshold}")
        
        # Morphological operations to connect nearby regions
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        colored_mask = cv2.dilate(colored_mask, kernel_dilate, iterations=2)
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        colored_mask = cv2.morphologyEx(colored_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Find contours
        contours, _ = cv2.findContours(colored_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by minimum area and exclusion zones
        overspray_regions = []
        excluded_count = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                # Check if this region is in an exclusion zone
                if self._is_region_in_exclusion_zone(contour):
                    excluded_count += 1
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    overspray_regions.append({
                        'contour': contour,
                        'area': area,
                        'center': (cx, cy),
                        'bbox': (x, y, w, h),
                        'density': area / (w * h) if w * h > 0 else 0
                    })
        
        if self.debug:
            print(f"Found {len(contours)} total colored regions (after line masking), {len(overspray_regions)} above {min_area}px² (potential overspray)")
            if excluded_count > 0:
                print(f"Excluded {excluded_count} regions in exclusion zones")
        
        return overspray_regions, colored_mask
    
    def detect(self, image, image_path=None):
        """Run overspray detection with vertical line masking.

        Args:
            image: Input image (BGR or grayscale).
            image_path: Optional path for loading exclusion zones

        Returns:
            tuple: (visualization_bgr, defects)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detect vertical line segments (one polygon per print head)
        polygons_list, left_points, right_points, left_kernels, right_kernels = \
            self.vertical_line_detector.detect_vertical_line(image, self.debug, image_path)
        
        if self.debug:
            print(f"\nVertical line: {len(polygons_list)} print head segments")
            for i, poly_data in enumerate(polygons_list):
                print(f"  Segment {i+1} WIDTH: {poly_data['width']}px")
        
        # Mask out all vertical line polygons
        line_masked = self.mask_vertical_line(gray, polygons_list)
        
        # Copy exclusion zones from vertical line detector
        self.exclusion_zones = self.vertical_line_detector.exclusion_zones
        
        # Detect colored regions (overspray)
        overspray_regions, colored_mask = self.detect_colored_regions(line_masked)
        
        # Create visualization
        visualization = self.create_overspray_visualization(image, overspray_regions, polygons_list)
        
        # Store debug images
        if self.debug:
            self._debug_line_polygon = self.vertical_line_detector.create_visualization(
                image, polygons_list, left_points, right_points, left_kernels, right_kernels
            )
            self._debug_line_masked_image = line_masked
            self._debug_polygon_with_coords = self.create_polygon_coords_visualization(
                image, polygons_list
            )
        
        # Prepare defects list
        defects = []
        for region in overspray_regions:
            defects.append({
                'type': 'overspray',
                'location': region['center'],
                'area': float(region['area']),
                'bbox': region['bbox'],
                'density': float(region['density'])
            })
        
        return visualization, defects
    
    def create_polygon_coords_visualization(self, original, polygons_list):
        """Create visualization showing only polygons with coordinate labels
        
        Args:
            original: Original image
            polygons_list: List of polygon dicts
            
        Returns:
            BGR image with polygons and coordinate labels
        """
        if len(original.shape) == 2:
            vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            vis = original.copy()
        
        height, width = vis.shape[:2]
        overlay = vis.copy()
        
        # Calculate text scale based on image size
        text_scale = max(0.4, min(1.2, (height / 44167) * 0.8))
        text_thickness = max(1, int(text_scale * 2))
        
        # Colors for different segments
        colors = [
            (0, 255, 0),    # Green
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 128, 255),  # Orange
            (255, 128, 0),  # Cyan
        ]
        
        for i, poly_data in enumerate(polygons_list):
            polygon = poly_data['polygon']
            color = colors[i % len(colors)]
            
            # Draw filled polygon (semi-transparent)
            poly_overlay = overlay.copy()
            cv2.fillPoly(poly_overlay, [polygon], color)
            overlay = cv2.addWeighted(overlay, 0.85, poly_overlay, 0.15, 0)
            
            # Draw polygon outline (thick)
            cv2.polylines(overlay, [polygon], True, color, max(2, int(text_scale * 3)))
            
            # Get the four corner points
            top_left, top_right, bottom_right, bottom_left = polygon
            
            # Draw corner points
            for pt in [top_left, top_right, bottom_right, bottom_left]:
                cv2.circle(overlay, tuple(pt), max(3, int(text_scale * 5)), color, -1)
            
            # Calculate polygon center for ALL text
            center_y = (poly_data['top_y'] + poly_data['bottom_y']) // 2
            center_x = int(poly_data.get('center_x', (poly_data.get('left_x', 0) + poly_data.get('right_x', 0)) / 2))
            
            # Build text lines
            shape = poly_data.get('shape_type', 'rectangle').upper()
            text_lines = [
                f"Print Head {i+1} ({shape})",
                f"WIDTH: {poly_data['width']}px",
                f"X-SHIFT: {poly_data.get('x_shift', 0):.1f}px",
                "",
                f"TL: ({top_left[0]},{top_left[1]})",
                f"TR: ({top_right[0]},{top_right[1]})",
                f"BR: ({bottom_right[0]},{bottom_right[1]})",
                f"BL: ({bottom_left[0]},{bottom_left[1]})"
            ]
            
            # Calculate starting Y position to center all text vertically
            line_height = int(25 * text_scale)
            total_text_height = len(text_lines) * line_height
            start_y = center_y - total_text_height // 2
            
            # Draw all text lines centered in the polygon
            for j, text_line in enumerate(text_lines):
                if not text_line:  # Skip blank lines
                    continue
                    
                current_y = start_y + int(j * line_height)
                (text_w, text_h), _ = cv2.getTextSize(text_line, cv2.FONT_HERSHEY_SIMPLEX, text_scale * 0.5, text_thickness)
                
                # Center text horizontally
                text_x = center_x - text_w // 2
                
                # Draw text background (black)
                cv2.rectangle(overlay,
                            (text_x - 3, current_y - text_h - 2),
                            (text_x + text_w + 3, current_y + 4),
                            (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(overlay, text_line, (text_x, current_y),
                           cv2.FONT_HERSHEY_SIMPLEX, text_scale * 0.5, (255, 255, 255), text_thickness)
        
        # Add overall legend
        legend_scale = text_scale * 0.7
        legend_thickness = max(1, int(legend_scale * 2))
        cv2.putText(overlay, f"Total Print Head Segments: {len(polygons_list)}", (10, int(30 * text_scale)),
                   cv2.FONT_HERSHEY_SIMPLEX, legend_scale, (255, 255, 255), legend_thickness)
        
        return overlay
    
    def _is_region_in_exclusion_zone(self, contour):
        """Check if an overspray region overlaps with any exclusion zone
        
        Args:
            contour: OpenCV contour of the overspray region
            
        Returns:
            bool: True if overlaps with exclusion zone, False otherwise
        """
        if not self.exclusion_zones:
            return False
        
        # Get bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        for zone in self.exclusion_zones:
            # Convert zone coordinates to proper order
            zone_x1 = min(zone['top_x'], zone['bottom_x'])
            zone_y1 = min(zone['top_y'], zone['bottom_y'])
            zone_x2 = max(zone['top_x'], zone['bottom_x'])
            zone_y2 = max(zone['top_y'], zone['bottom_y'])
            
            # Check if overspray bounding box overlaps with exclusion zone
            if (x < zone_x2 and x + w > zone_x1 and
                y < zone_y2 and y + h > zone_y1):
                if self.debug:
                    print(f"    Overspray at ({x}, {y}) excluded by zone '{zone['name']}'")
                return True
        
        return False
    
    def create_overspray_visualization(self, original, overspray_regions, polygons_list):
        """Create visualization showing overspray regions and print head polygons
        
        Args:
            original: Original image
            overspray_regions: List of overspray region dicts
            polygons_list: List of polygon dicts (one per print head segment)
            
        Returns:
            BGR visualization image
        """
        if len(original.shape) == 2:
            vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            vis = original.copy()
        
        overlay = vis.copy()
        
        # Draw the vertical line polygons for reference (green outlines)
        if polygons_list:
            for poly_data in polygons_list:
                polygon = poly_data['polygon']
                cv2.polylines(overlay, [polygon], True, (0, 255, 0), 2)
        
        # Draw overspray regions in red
        colors = [
            (0, 0, 255),    # Red
            (255, 0, 0),    # Blue  
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
        ]
        
        for i, region in enumerate(overspray_regions):
            color = colors[i % len(colors)]
            
            # Draw contour outline
            cv2.drawContours(overlay, [region['contour']], -1, color, 3)
            
            # Fill region with transparent color
            region_overlay = overlay.copy()
            cv2.fillPoly(region_overlay, [region['contour']], color)
            overlay = cv2.addWeighted(overlay, 0.7, region_overlay, 0.3, 0)
        
        return overlay
    
    def save_debug_images(self, output_dir, base_name):
        """Save debug images if debug mode is enabled"""
        debug_paths = []
        
        if self.debug:
            if self._debug_polygon_with_coords is not None:
                path = os.path.join(output_dir, f"{base_name}_polygon_coordinates.jpg")
                cv2.imwrite(path, self._debug_polygon_with_coords, [cv2.IMWRITE_JPEG_QUALITY, 95])
                debug_paths.append(path)
                print(f"    Saved polygon with coordinates: {path}")
            
            if self._debug_line_polygon is not None:
                path = os.path.join(output_dir, f"{base_name}_vertical_line_polygon.jpg")
                cv2.imwrite(path, self._debug_line_polygon, [cv2.IMWRITE_JPEG_QUALITY, 95])
                debug_paths.append(path)
                print(f"    Saved vertical line polygon debug: {path}")
            
            if self._debug_line_masked_image is not None:
                path = os.path.join(output_dir, f"{base_name}_line_masked.jpg")
                cv2.imwrite(path, self._debug_line_masked_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                debug_paths.append(path)
                print(f"    Saved line-masked image debug: {path}")
        
        return debug_paths if debug_paths else None
    
    def create_visualization(self, original, defects, kernel_states=None):
        """Create a visualization image highlighting overspray findings.

        Args:
            original: Original image (BGR or grayscale).
            defects: List of defect dicts (kernel-level or merged regions).
            kernel_states: Optional kernel state list for debug overlays.

        Returns:
            BGR image with overlays indicating overspray.
        """
        if len(original.shape) == 2:
            vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            vis = original.copy()
        
        # Create overlay
        overlay = vis.copy()
        
        # Draw debug kernels if enabled
        if self.debug and kernel_states:
            for state in kernel_states:
                x = state['x']
                y = state['y']
                x1, y1, x2, y2 = state['bbox']
                
                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(vis.shape[1], x2)
                y2 = min(vis.shape[0], y2)
                
                # Color based on scatter metric
                if state['has_overspray']:
                    # Red for overspray
                    color = (0, 0, 255)
                elif state['scatter_metric'] > self.scatter_threshold * 0.7:
                    # Yellow for near-threshold
                    color = (0, 255, 255)
                else:
                    # Green for clean areas
                    color = (0, 255, 0)
                
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
                
                # Draw center point
                cv2.circle(overlay, (x, y), 2, (255, 0, 0), -1)
            
            # In debug mode, blend lightly to see kernels clearly
            result = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
            return result
        
        # Non-debug mode - highlight overspray regions
        for defect in defects:
            if 'bbox' in defect:
                x1, y1, x2, y2 = defect['bbox']
                
                # Draw filled red rectangle overlay for overspray region
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
        
        # Blend with original to create overlay effect
        result = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
        
        return result 