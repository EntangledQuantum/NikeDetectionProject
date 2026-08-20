"""
Overspray Detection Algorithm for Island Images
Uses line detection to remove lines, then clusters nearby debris to detect overspray
Overspray is detected as groups of small debris particles close to each other
"""

import cv2
import numpy as np
import os
from digital_air_cv.detectors.base_legacy import BaseDetector
from digital_air_cv.io.image_saver import save_image
from digital_air_cv.geometry.line_detector import LineDetector


class OversprayIslandDetector(BaseDetector):
    """Detect overspray in island images after removing slanted lines.

    Pipeline:
      1) Detect and remove slanted lines to avoid false positives
      2) Detect colored (non-white) regions under a background threshold
      3) Optionally group nearby regions into larger overspray shapes
      4) Produce final visualization and structured defect output
    """
    
    def __init__(self, sensitivity='medium', debug=False):
        """Configure overspray detection parameters and debugging behavior.

        Args:
            sensitivity: One of {'low', 'medium', 'high'} controlling thresholds
                and grouping distances.
            debug: If True, collects intermediate masks and helper images.
        """
        super().__init__()  # Initialize BaseDetector
        self.debug = debug
        self.sensitivity = sensitivity
        
        print(f"Overspray Detection Sensitivity: {sensitivity}")
        
        # Initialize line detector with same sensitivity
        self.line_detector = LineDetector(sensitivity)
        
        # Set overspray detection parameters based on sensitivity
        if sensitivity == 'high':
            self.line_thickness = 5
            self.background_threshold = 5  # Very sensitive to color differences
            self.overspray_min_area = 100  # Smaller minimum area (more sensitive)
            self.overspray_max_distance = 300  # Closer grouping distance
        elif sensitivity == 'low':
            self.line_thickness = 5
            self.background_threshold = 20  # Less sensitive to color differences
            self.overspray_min_area = 1000  # Larger minimum area (less sensitive)
            self.overspray_max_distance = 800  # Larger grouping distance
        else:  # medium (default)
            self.line_thickness = 15  # Thicker to ensure lines are fully covered
            self.background_threshold = 50  # Sensitivity to color detection
            self.overspray_min_area = 5000  # MINIMUM area to be considered overspray (user control)
            self.overspray_max_distance = 1000  # Maximum distance to group nearby regions
        
        # Store debug images
        self._debug_kernel_image = None
        self._debug_lines_removed_image = None
        self._debug_debris_mask = None
        self._debug_clustering_image = None
        self._debug_line_points_image = None
        
        # Print configuration
        print(f"Overspray Island Detector Configuration:")
        print(f"  Line thickness for removal: {self.line_thickness}")
        print(f"  Background threshold: {self.background_threshold} (difference from detected background)")
        print(f"  MINIMUM overspray area: {self.overspray_min_area}px² (user controlled)")
        print(f"  Maximum grouping distance: {self.overspray_max_distance}px")
        print(f"  Multi-method detection: background_diff + dark_areas + non_white")
        print(f"  Aggressive morphological operations for region connection")
    
    def remove_lines_from_image(self, gray_image, matched_lines):
        """Paint out valid-slope lines in grayscale image to isolate overspray.

        Args:
            gray_image: Grayscale image to modify in-place copy.
            matched_lines: Matched line dicts with endpoints and slope validity.

        Returns:
            Grayscale image with lines painted white using configured thickness.
        """
        # Create a copy to work on
        lines_removed = gray_image.copy()
        
        # Paint each line white with the specified thickness (only valid slopes)
        for match in matched_lines:
            # Only remove lines with valid slopes
            if match.get('valid_slope', True):
                left_pt = match['left']
                right_pt = match['right']
                
                # Draw white line to remove it
                cv2.line(lines_removed, (left_pt['x'], left_pt['y']), 
                        (right_pt['x'], right_pt['y']), 
                        255, self.line_thickness * 3)  # Extra thick to ensure complete removal
        
        return lines_removed
    
    def _is_debris_in_exclusion_zone(self, cx, cy, contour):
        """Return True if a contour overlaps any configured exclusion zone.

        Args:
            cx: Unused x-center placeholder (kept for compatibility).
            cy: Unused y-center placeholder (kept for compatibility).
            contour: OpenCV contour representing a candidate region.

        Returns:
            bool: True if the contour's bbox intersects an exclusion zone.
        """
        # Access exclusion zones from line detector
        if not hasattr(self.line_detector, 'exclusion_zones') or not self.line_detector.exclusion_zones:
            return False
        
        # Get bounding box of the contour for more accurate checking
        x, y, w, h = cv2.boundingRect(contour)
        
        for zone in self.line_detector.exclusion_zones:
            # Convert zone coordinates to proper order
            zone_x1 = min(zone['top_x'], zone['bottom_x'])
            zone_y1 = min(zone['top_y'], zone['bottom_y'])
            zone_x2 = max(zone['top_x'], zone['bottom_x'])
            zone_y2 = max(zone['top_y'], zone['bottom_y'])
            
            # Check if debris bounding box overlaps with exclusion zone
            if (x < zone_x2 and x + w > zone_x1 and
                y < zone_y2 and y + h > zone_y1):
                # if self.debug:
                #     print(f"    Debris at ({cx}, {cy}) excluded by zone '{zone['name']}'")
                return True
        
        return False
    
    def detect_colored_regions(self, lines_removed_image):
        """Detect non-white regions after line removal as overspray candidates.

        Uses a background-based threshold to create a binary mask of colored
        pixels, followed by aggressive morphological operations to connect
        nearby areas.

        Args:
            lines_removed_image: Grayscale image with lines painted white.

        Returns:
            tuple: (overspray_regions, colored_mask)
                - overspray_regions: List of region dicts with contour, bbox,
                  area, center, and density
                - colored_mask: Binary mask of colored regions after morphology
        """
        if self.debug:
            print(f"Original image range: {lines_removed_image.min()} to {lines_removed_image.max()}")
        
        # Simple approach: Find anything that's not white or near-white
        # After lines are painted white, only colored areas should remain dark
        
        # Define what we consider "white/background"
        # Typical background is 180-255, colored areas are darker
        color_threshold = 180 - self.background_threshold  # Lower = more sensitive to colors
        
        if self.debug:
            print(f"Color threshold: {color_threshold} (anything below is considered colored)")
            print(f"Lines should be painted white (255), background should be light (>180)")
        
        # Create mask for colored areas - simple and direct
        # Find all pixels darker than the threshold (these are colored areas)
        _, colored_mask = cv2.threshold(lines_removed_image, color_threshold, 255, cv2.THRESH_BINARY_INV)
        colored_mask = colored_mask.astype(np.uint8)
        
        # Also create a stricter mask for very colored areas
        _, strong_colored = cv2.threshold(lines_removed_image, 120, 255, cv2.THRESH_BINARY_INV)
        strong_colored = strong_colored.astype(np.uint8)
        
        # Debug masks for analysis
        not_white_mask = colored_mask.copy()  # For debug compatibility
        colored_areas = strong_colored.copy()  # For debug compatibility
        
        if self.debug:
            print(f"Colored mask (<{color_threshold}): {np.sum(colored_mask > 0)} pixels")
            print(f"Strong colored areas (<120): {np.sum(strong_colored > 0)} pixels") 
            print(f"Using colored mask with {np.sum(colored_mask > 0)} pixels for detection")
            
            # Store individual masks for debug saving
            self._debug_mask1 = not_white_mask
            self._debug_mask2 = colored_areas
            self._debug_mask3 = colored_mask
        
        # More aggressive morphological operations to connect colored regions
        # Start with dilation to expand colored regions
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        colored_mask = cv2.dilate(colored_mask, kernel_dilate, iterations=2)
        
        # Close gaps between nearby colored pixels
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        colored_mask = cv2.morphologyEx(colored_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Fill holes inside colored regions
        kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        colored_mask = cv2.morphologyEx(colored_mask, cv2.MORPH_CLOSE, kernel_fill)
        
        # Final erosion to restore approximate original size
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        colored_mask = cv2.erode(colored_mask, kernel_erode, iterations=1)
        
        if self.debug:
            print(f"Final colored mask after morphology: {np.sum(colored_mask > 0)} white pixels")
        
        # Find connected components (regions of colored areas)
        contours, _ = cv2.findContours(colored_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter regions by minimum area and exclusion zones
        overspray_regions = []
        excluded_regions = []
        small_regions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Track small regions separately
            if area < self.overspray_min_area:
                small_regions.append(contour)
                continue
            
            # Only consider regions above minimum area
            if area >= self.overspray_min_area:
                # Check if region is in exclusion zone
                is_excluded = self._is_debris_in_exclusion_zone(0, 0, contour)  # Use existing method
                
                if is_excluded:
                    excluded_regions.append(contour)
                else:
                    # Calculate region properties
                    x, y, w, h = cv2.boundingRect(contour)
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        region_data = {
                            'contour': contour,
                            'area': area,
                            'center': (cx, cy),
                            'bbox': (x, y, w, h),
                            'density': area / (w * h) if w * h > 0 else 0
                        }
                        overspray_regions.append(region_data)
        
        if self.debug:
            print(f"Found {len(contours)} total colored regions")
            print(f"Small regions (< {self.overspray_min_area}px²): {len(small_regions)}")
            print(f"Kept {len(overspray_regions)} regions above minimum area ({self.overspray_min_area}px²)")
            if excluded_regions:
                print(f"Excluded {len(excluded_regions)} regions in exclusion zones")
        
        return overspray_regions, colored_mask
    
    def group_nearby_regions(self, overspray_regions):
        """Group overspray regions based on centroid proximity.

        Args:
            overspray_regions: List of individual region dicts from detection.

        Returns:
            List of grouped region dicts. Multi-region groups include a convex
            hull contour, merged area, center, density, and merged_count.
        """
        if len(overspray_regions) <= 1:
            return overspray_regions
        
        # Simple distance-based grouping
        grouped_regions = []
        used_indices = set()
        
        for i, region1 in enumerate(overspray_regions):
            if i in used_indices:
                continue
                
            # Start a new group with this region
            group = [region1]
            used_indices.add(i)
            
            # Find nearby regions to group with
            center1 = region1['center']
            
            for j, region2 in enumerate(overspray_regions):
                if j == i or j in used_indices:
                    continue
                    
                center2 = region2['center']
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                if distance <= self.overspray_max_distance:
                    group.append(region2)
                    used_indices.add(j)
            
            # Create merged region if we have multiple regions to group
            if len(group) > 1:
                # Merge contours
                all_contours = [region['contour'] for region in group]
                all_points = []
                total_area = 0
                
                for contour in all_contours:
                    all_points.extend(contour.reshape(-1, 2))
                    total_area += cv2.contourArea(contour)
                
                # Create convex hull around all regions
                all_points = np.array(all_points)
                hull = cv2.convexHull(all_points)
                hull_area = cv2.contourArea(hull)
                
                # Calculate center of merged region
                M = cv2.moments(hull)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = center1  # Fallback to first region's center
                
                grouped_region = {
                    'contour': hull,
                    'area': hull_area,
                    'center': (cx, cy),
                    'bbox': cv2.boundingRect(hull),
                    'density': total_area / hull_area if hull_area > 0 else 0,
                    'merged_count': len(group),
                    'original_area': total_area
                }
                grouped_regions.append(grouped_region)
            else:
                # Single region, keep as is
                grouped_regions.append(group[0])
        
        if self.debug:
            print(f"Region grouping results:")
            print(f"  Original regions: {len(overspray_regions)}")
            print(f"  Grouped regions: {len(grouped_regions)}")
            for i, region in enumerate(grouped_regions):
                merged_count = region.get('merged_count', 1)
                print(f"    Region {i+1}: {region['area']:.0f}px², density: {region['density']:.3f}, merged: {merged_count}")
        
        return grouped_regions
    
    def create_region_visualization(self, image, overspray_regions, matched_lines):
        """Visualize colored regions and reference lines for debugging.

        Args:
            image: Original image (BGR or grayscale).
            overspray_regions: List of region dicts to draw.
            matched_lines: Matched lines; drawn for reference only.

        Returns:
            BGR visualization image with contours, fills, labels, and legends.
        """
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
        
        # Create overlay
        overlay = vis.copy()
        
        # Draw detected lines in green (for reference)
        for match in matched_lines:
            if match.get('valid_slope', True):
                left_pt = match['left']
                right_pt = match['right']
                cv2.line(overlay, (left_pt['x'], left_pt['y']), 
                        (right_pt['x'], right_pt['y']), 
                        (0, 255, 0), max(2, self.line_thickness // 2))
        
        # Draw overspray regions with different colors
        colors = [
            (0, 0, 255),    # Red
            (255, 0, 0),    # Blue  
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 165, 0),  # Orange
            (0, 128, 255),  # Light Blue
            (128, 0, 255),  # Purple
            (0, 255, 128),  # Light Green
        ]
        
        for i, region in enumerate(overspray_regions):
            color = colors[i % len(colors)]
            
            # Draw contour outline
            cv2.drawContours(overlay, [region['contour']], -1, color, 3)
            
            # Fill region with transparent color
            region_overlay = overlay.copy()
            cv2.fillPoly(region_overlay, [region['contour']], color)
            overlay = cv2.addWeighted(overlay, 0.8, region_overlay, 0.2, 0)
            
            # Add region label
            center = region['center']
            merged_count = region.get('merged_count', 1)
            label = f"R{i+1}: {region['area']:.0f}px²"
            if merged_count > 1:
                label += f" (M{merged_count})"
                
            cv2.putText(overlay, label, center, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw exclusion zones if they exist
        if hasattr(self.line_detector, 'exclusion_zones') and self.line_detector.exclusion_zones:
            for i, zone in enumerate(self.line_detector.exclusion_zones):
                # Convert zone coordinates to proper order
                x1 = min(zone['top_x'], zone['bottom_x'])
                y1 = min(zone['top_y'], zone['bottom_y'])
                x2 = max(zone['top_x'], zone['bottom_x'])
                y2 = max(zone['top_y'], zone['bottom_y'])
                
                # Draw exclusion zone as magenta rectangle
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 255), 3)
                
                # Add zone label
                label = f"Exclusion {i+1}: {zone.get('name', 'unnamed')}"
                cv2.putText(overlay, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Add legend
        y_offset = 30
        cv2.putText(overlay, f"Overspray regions: {len(overspray_regions)}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30
        cv2.putText(overlay, f"Min area: {self.overspray_min_area}px²", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(overlay, f"Max grouping distance: {self.overspray_max_distance}px", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add exclusion zone info
        if hasattr(self.line_detector, 'exclusion_zones') and self.line_detector.exclusion_zones:
            y_offset += 30
            cv2.putText(overlay, f"Exclusion zones: {len(self.line_detector.exclusion_zones)}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        return overlay
    
    def create_clustering_visualization(self, image, debris_particles, overspray_regions, noise_particles, matched_lines):
        """Visualize clustering results: particles, noise, and grouped regions.

        Args:
            image: Original image (BGR or grayscale).
            debris_particles: List of particle dicts in clusters.
            overspray_regions: Grouped region dicts with hull contours.
            noise_particles: Particles rejected as noise.
            matched_lines: Matched slanted lines for reference drawing.

        Returns:
            BGR image with overlays for clusters, noise, lines, and labels.
        """
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
        
        # Create overlay
        overlay = vis.copy()
        
        # Draw detected lines in green (for reference)
        for match in matched_lines:
            if match.get('valid_slope', True):
                left_pt = match['left']
                right_pt = match['right']
                cv2.line(overlay, (left_pt['x'], left_pt['y']), 
                        (right_pt['x'], right_pt['y']), 
                        (0, 255, 0), max(2, self.line_thickness // 2))
        
        # Draw noise particles in gray
        for particle in noise_particles:
            cv2.drawContours(overlay, [particle['contour']], -1, (128, 128, 128), -1)
            cv2.circle(overlay, particle['center'], 3, (64, 64, 64), -1)
        
        # Draw overspray regions with different colors
        colors = [
            (0, 0, 255),    # Red
            (255, 0, 0),    # Blue  
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 165, 0),  # Orange
            (0, 128, 255),  # Light Blue
            (128, 0, 255),  # Purple
            (0, 255, 128),  # Light Green
        ]
        
        for i, region in enumerate(overspray_regions):
            color = colors[i % len(colors)]
            
            # Draw convex hull
            cv2.drawContours(overlay, [region['hull']], -1, color, 3)
            
            # Fill hull with transparent color
            hull_overlay = overlay.copy()
            cv2.fillPoly(hull_overlay, [region['hull']], color)
            overlay = cv2.addWeighted(overlay, 0.8, hull_overlay, 0.2, 0)
            
            # Draw individual particles in cluster
            for particle in region['particles']:
                cv2.drawContours(overlay, [particle['contour']], -1, color, -1)
                cv2.circle(overlay, particle['center'], 2, (255, 255, 255), -1)
            
            # Add region label
            # Properly handle hull coordinates - hull is shape (n, 1, 2)
            hull_points = region['hull'].reshape(-1, 2)  # Reshape to (n, 2)
            hull_center = np.mean(hull_points, axis=0)
            center_point = (int(hull_center[0]), int(hull_center[1]))
            label = f"O{i+1}: {region['particle_count']}p"
            cv2.putText(overlay, label, center_point, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw exclusion zones if they exist
        if hasattr(self.line_detector, 'exclusion_zones') and self.line_detector.exclusion_zones:
            for i, zone in enumerate(self.line_detector.exclusion_zones):
                # Convert zone coordinates to proper order
                x1 = min(zone['top_x'], zone['bottom_x'])
                y1 = min(zone['top_y'], zone['bottom_y'])
                x2 = max(zone['top_x'], zone['bottom_x'])
                y2 = max(zone['top_y'], zone['bottom_y'])
                
                # Draw exclusion zone as magenta rectangle
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 255), 3)
                
                # Add zone label
                label = f"Exclusion {i+1}: {zone.get('name', 'unnamed')}"
                cv2.putText(overlay, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Add legend
        y_offset = 30
        cv2.putText(overlay, f"Overspray regions: {len(overspray_regions)}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30
        cv2.putText(overlay, f"Noise particles: {len(noise_particles)}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        y_offset += 30
        cv2.putText(overlay, f"Total debris: {len(debris_particles)}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add exclusion zone info
        if hasattr(self.line_detector, 'exclusion_zones') and self.line_detector.exclusion_zones:
            y_offset += 30
            cv2.putText(overlay, f"Exclusion zones: {len(self.line_detector.exclusion_zones)}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        return overlay
    
    def create_overspray_visualization(self, image, overspray_regions, matched_lines):
        """Create the final overspray visualization without reference lines.

        Args:
            image: Original image (BGR or grayscale).
            overspray_regions: Final list of overspray regions to display.
            matched_lines: Matched lines (omitted from final visualization).

        Returns:
            BGR image highlighting overspray areas with labels.
        """
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
        
        # Create overlay
        overlay = vis.copy()
        
        # Don't draw lines in final visualization - focus only on overspray
        # (Lines are already removed/painted white in processing)
        
        # Highlight overspray regions in red
        for i, region in enumerate(overspray_regions):
            # Draw thick red outline
            cv2.drawContours(overlay, [region['contour']], -1, (0, 0, 255), 4)
            
            # Fill with semi-transparent red
            region_overlay = overlay.copy()
            cv2.fillPoly(region_overlay, [region['contour']], (0, 0, 255))
            overlay = cv2.addWeighted(overlay, 0.7, region_overlay, 0.3, 0)
            
            # Add overspray label
            center_point = region['center']
            
            # Calculate label positions with bounds checking
            label_point = (max(10, center_point[0] - 40), max(20, center_point[1]))
            detail_point = (max(10, center_point[0] - 30), min(center_point[1] + 20, overlay.shape[0] - 10))
            
            label = f"OVERSPRAY {i+1}"
            cv2.putText(overlay, label, label_point, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add details
            merged_count = region.get('merged_count', 1)
            details = f"{region['area']:.0f}px²"
            if merged_count > 1:
                details += f" (M{merged_count})"
            cv2.putText(overlay, details, detail_point, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add summary
        text = f"Overspray regions: {len(overspray_regions)}"
        cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return overlay
    
    def detect(self, image, image_path=None):
        """Run overspray detection on an island image and return results.

        Steps:
          1) Detect slanted lines and paint them out
          2) Detect colored regions below background threshold
          3) Group regions by proximity and create final visualization

        Args:
            image: Input image (BGR or grayscale).
            image_path: Optional path to load per-image exclusion zones.

        Returns:
            tuple: (visualization_bgr, defects)
        """
        # Use line detector to find lines
        precomputed = getattr(self, "_precomputed_lines", None)
        if precomputed is not None:
            matched_lines = precomputed
            left_lines = right_lines = left_kernels = right_kernels = []
        else:
            matched_lines, left_lines, right_lines, left_kernels, right_kernels = self.line_detector.detect_lines(image, self.debug, image_path)
        
        # Remove lines from image
        if len(image.shape) == 3:
            gray_for_processing = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_for_processing = image.copy()
        
        lines_removed = self.remove_lines_from_image(gray_for_processing, matched_lines)
        
        # Detect colored regions (overspray areas)
        overspray_regions, colored_mask = self.detect_colored_regions(lines_removed)
        
        # Group nearby regions together
        overspray_regions = self.group_nearby_regions(overspray_regions)
        
        # Create final visualization
        visualization = self.create_overspray_visualization(image, overspray_regions, matched_lines)
        
        # Store debug images
        if self.debug:
            print(f"OVERSPRAY DETECTION RESULTS:")
            print(f"  Lines detected: {len(matched_lines)}")
            print(f"  Colored regions found: {len(overspray_regions)}")
            if hasattr(self.line_detector, 'exclusion_zones') and self.line_detector.exclusion_zones:
                print(f"  Exclusion zones: {len(self.line_detector.exclusion_zones)} zones loaded")
            
            self._debug_lines_removed_image = lines_removed
            self._debug_debris_mask = colored_mask
            self._debug_clustering_image = self.create_region_visualization(image, overspray_regions, matched_lines)
            
            # Store additional debug masks for analysis
            if hasattr(self, '_debug_mask1'):
                self._debug_mask1_stored = self._debug_mask1
            if hasattr(self, '_debug_mask2'):
                self._debug_mask2_stored = self._debug_mask2
            if hasattr(self, '_debug_mask3'):
                self._debug_mask3_stored = self._debug_mask3
            
            # Create line points debug image if lines were detected
            if left_lines or right_lines:
                self._debug_line_points_image = self.create_line_points_visualization(image, matched_lines, left_lines, right_lines)
        
        # Prepare defects/results
        defects = []
        
        # Add line detection results
        if matched_lines:
            line_stats = self.line_detector.get_line_statistics(matched_lines)
            valid_lines = [l for l in matched_lines if l.get('valid_slope', True)]
            invalid_lines = [l for l in matched_lines if not l.get('valid_slope', True)]
            
            defects.append({
                'type': 'lines_detected',
                'line_count': line_stats['line_count'],
                'average_y_delta': line_stats['average_y_delta'],
                'std_y_delta': line_stats['std_y_delta'],
                'valid_lines': len(valid_lines),
                'invalid_lines': len(invalid_lines)
            })
        
        # Add overspray detection results
        if overspray_regions:
            overspray_info = []
            for region in overspray_regions:
                bbox = region['bbox']
                center = region['center']
                overspray_info.append({
                    'bbox': bbox,
                    'area': float(region['area']),
                    'center': center,
                    'density': float(region['density']),
                    'merged_count': region.get('merged_count', 1),
                    'original_area': float(region.get('original_area', region['area']))
                })
            
            defects.append({
                'type': 'overspray_detected',
                'overspray_count': len(overspray_regions),
                'overspray_regions': overspray_info,
                'min_area_threshold': self.overspray_min_area,
                'max_grouping_distance': self.overspray_max_distance
            })
        
        return visualization, defects
    
    def create_line_points_visualization(self, image, matched_lines, left_lines, right_lines):
        """Show all detected line points for debugging detection density.

        Args:
            image: Original image (BGR or grayscale).
            matched_lines: Matched pairs list (not directly drawn).
            left_lines: Raw detections from the left scan.
            right_lines: Raw detections from the right scan.

        Returns:
            BGR image with colored dots and a compact legend.
        """
        if len(image.shape) == 2:
            debug_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            debug_vis = image.copy()
        
        # Count different types
        left_real = [l for l in left_lines if l.get('type', 'real') == 'real']
        left_ghost = [l for l in left_lines if l.get('type', 'real') == 'ghost']
        right_real = [l for l in right_lines if l.get('type', 'real') == 'real']
        right_ghost = [l for l in right_lines if l.get('type', 'real') == 'ghost']
        
        # Draw dots for left-side detections
        for left_pt in left_lines:
            dot_color = (128, 128, 255) if left_pt.get('type', 'real') == 'ghost' else (0, 0, 255)
            cv2.circle(debug_vis, (left_pt['x'], left_pt['y']), 15, dot_color, -1)
        
        # Draw dots for right-side detections  
        for right_pt in right_lines:
            dot_color = (128, 255, 128) if right_pt.get('type', 'real') == 'ghost' else (0, 255, 0)
            cv2.circle(debug_vis, (right_pt['x'], right_pt['y']), 15, dot_color, -1)
        
        # Add legend
        cv2.putText(debug_vis, f"Left: {len(left_real)}r+{len(left_ghost)}g | Right: {len(right_real)}r+{len(right_ghost)}g | Matched: {len(matched_lines)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return debug_vis
    
    def save_debug_images(self, output_dir, base_name):
        """Persist collected debug images to disk when debug is enabled.

        Args:
            output_dir: Directory where images will be saved.
            base_name: Base name to prefix output files.

        Returns:
            list[str] | None: Saved file paths or None if nothing saved.
        """
        debug_paths = []
        
        if self.debug:
            images_to_save = {
                '_debug_lines_removed_image': 'overspray_lines_removed',
                '_debug_debris_mask': 'overspray_colored_mask',
                '_debug_clustering_image': 'overspray_regions',
                '_debug_mask1_stored': 'overspray_mask1_not_white',
                '_debug_mask2_stored': 'overspray_mask2_colored_areas',
                '_debug_mask3_stored': 'overspray_mask3_final',
                '_debug_line_points_image': 'overspray_line_points'
            }

            for attr, suffix in images_to_save.items():
                image = getattr(self, attr, None)
                if image is not None:
                    saved_path = save_image(output_dir, base_name, image, suffix)
                    if saved_path:
                        debug_paths.append(saved_path)

        return debug_paths if debug_paths else None