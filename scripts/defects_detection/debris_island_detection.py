"""
Debris Detection Algorithm for Island Images
Detects horizontal slanted lines and prepares for debris detection
Specifically designed for island-type images

Author: Assistant
Date: 2024
"""

import cv2
import numpy as np
import os
from detector_base import BaseDetector
from utils.line_detector import LineDetector


class DebrisIslandDetector(BaseDetector):
    """Detects debris in island images by first removing detected lines"""
    
    def __init__(self, sensitivity='medium', debug=False):
        """
        Args:
            sensitivity: Detection sensitivity level ('low', 'medium', 'high')
            debug: Whether to enable debug visualization
        """
        super().__init__()  # Initialize BaseDetector with exclusion zone support
        self.debug = debug
        self.sensitivity = sensitivity
        
        print(f"Sensitivity: {sensitivity}")
        
        # Initialize line detector with same sensitivity
        self.line_detector = LineDetector(sensitivity)
        
        # Set debris detection parameters based on sensitivity
        if sensitivity == 'high':
            self.line_thickness = 5
            self.background_threshold = 140  # More aggressive - treat more as background
            self.debris_min_area = 5  # Detect smaller debris
        elif sensitivity == 'low':
            self.line_thickness = 5
            self.background_threshold = 100  # Less aggressive - darker threshold for background
            self.debris_min_area = 50  # Only detect larger debris
        else:  # medium (default)
            self.line_thickness = 20
            self.background_threshold = 120  # Medium aggressiveness
            self.debris_min_area = 10  # Medium size requirement
        
        # Store debug images
        self._debug_kernel_image = None
        self._debug_lines_removed_image = None
        self._debug_debris_mask = None
        self._debug_enhanced_debris_mask = None
        self._debug_line_points_image = None
        
        # Print configuration
        print(f"Debris Island Detector Configuration:")
        print(f"  Line thickness for removal: {self.line_thickness}")
        print(f"  Background threshold: {self.background_threshold} (anything above = background)")
        print(f"  Min debris area: {self.debris_min_area}px²")
    

    
    def create_debug_kernel_visualization(self, image, left_kernels, right_kernels, matched_lines):
        """Create a debug visualization showing both kernels AND lines"""
        if len(image.shape) == 2:
            debug_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            debug_vis = image.copy()
        
        overlay = debug_vis.copy()
        
        # First draw the lines with different colors based on validity and type
        for i, match in enumerate(matched_lines):
            left_pt = match['left']
            right_pt = match['right']
            valid_slope = match.get('valid_slope', True)
            left_type = match.get('left_type', 'real')
            right_type = match.get('right_type', 'real')
            
            # Choose color based on validity and line types
            if not valid_slope:
                line_color = (0, 0, 255)  # Red for invalid slope
                line_thickness = self.line_thickness
            elif left_type == 'ghost' or right_type == 'ghost':
                line_color = (255, 165, 0)  # Orange for ghost lines
                line_thickness = max(2, self.line_thickness // 2)  # Thinner for ghost
            else:
                line_color = (0, 255, 0)  # Green for valid real lines
                line_thickness = self.line_thickness
            
            # Draw the line from left detection to right detection
            cv2.line(overlay, (left_pt['x'], left_pt['y']), 
                    (right_pt['x'], right_pt['y']), 
                    line_color, line_thickness)
            
            # Draw dots at detection points with different colors for ghost/real
            left_dot_color = (128, 128, 255) if left_type == 'ghost' else (255, 0, 0)  # Light blue for ghost, blue for real
            right_dot_color = (255, 128, 128) if right_type == 'ghost' else (0, 0, 255)  # Light red for ghost, red for real
            
            cv2.circle(overlay, (left_pt['x'], left_pt['y']), 5, left_dot_color, -1)
            cv2.circle(overlay, (right_pt['x'], right_pt['y']), 5, right_dot_color, -1)
            
            if i < 5:  # Show info for first 5 lines
                # Add text showing the delta and slope
                mid_x = (left_pt['x'] + right_pt['x']) // 2
                mid_y = (left_pt['y'] + right_pt['y']) // 2
                slope_text = f"dY={match['y_delta']:.0f}, S={match['slope']:.4f}"
                cv2.putText(overlay, slope_text, 
                           (mid_x - 50, mid_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Draw left kernels
        for state in left_kernels:
            x1, y1, x2, y2 = state['bbox']
            color = (0, 255, 0) if state['has_line'] else (128, 128, 128)  # Green if line, gray if not
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Add text showing pixel ratio
            if state['has_line']:
                ratio_text = f"{state['pixel_ratio']:.2f}"
                cv2.putText(overlay, ratio_text, (x1+2, y1+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw right kernels
        for state in right_kernels:
            x1, y1, x2, y2 = state['bbox']
            color = (0, 255, 255) if state['has_line'] else (128, 128, 128)  # Yellow if line, gray if not
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Add text showing pixel ratio
            if state['has_line']:
                ratio_text = f"{state['pixel_ratio']:.2f}"
                cv2.putText(overlay, ratio_text, (x1+2, y1+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Add legend
        cv2.putText(overlay, "Green: Valid real lines", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(overlay, "Orange: Ghost lines", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        cv2.putText(overlay, "Red: Invalid slopes", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(overlay, "Yellow: Right kernels with lines", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(overlay, "Gray: Kernels without lines", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        cv2.putText(overlay, f"Y-delta range: {self.line_detector.Y_DELTA_MIN}-{self.line_detector.Y_DELTA_MAX}", (10, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(overlay, f"Slope range: {self.line_detector.SLOPE_MIN:.4f}-{self.line_detector.SLOPE_MAX:.4f}", (10, 210), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
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
            
            # Add exclusion zone legend
            cv2.putText(overlay, f"Magenta: {len(self.line_detector.exclusion_zones)} Exclusion zones", (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Blend with original
        debug_vis = cv2.addWeighted(debug_vis, 0.6, overlay, 0.4, 0)
        
        return debug_vis
    
    def create_line_points_visualization(self, image, matched_lines, left_lines, right_lines):
        """Create a simple visualization showing ALL detected line points with dots"""
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
            dot_color = (128, 128, 255) if left_pt.get('type', 'real') == 'ghost' else (0, 0, 255)  # Light blue for ghost, red for real
            cv2.circle(debug_vis, (left_pt['x'], left_pt['y']), 15, dot_color, -1)
        
        # Draw dots for right-side detections  
        for right_pt in right_lines:
            dot_color = (128, 255, 128) if right_pt.get('type', 'real') == 'ghost' else (0, 255, 0)  # Light green for ghost, green for real
            cv2.circle(debug_vis, (right_pt['x'], right_pt['y']), 15, dot_color, -1)
        
        # Add legend
        cv2.putText(debug_vis, "Red: Real left detections", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(debug_vis, "Light Blue: Ghost left detections", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 255), 2)
        cv2.putText(debug_vis, "Green: Real right detections", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_vis, "Light Green: Ghost right detections", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 255, 128), 2)
        cv2.putText(debug_vis, f"Left: {len(left_real)}r+{len(left_ghost)}g | Right: {len(right_real)}r+{len(right_ghost)}g | Matched: {len(matched_lines)}", (10, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return debug_vis
    
    def remove_lines_from_image(self, gray_image, matched_lines):
        """Remove detected lines by painting them white (only valid slopes)"""
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
                        255, self.line_thickness * 2)  # Make it thicker to ensure complete removal
        
        return lines_removed
    
    def _is_debris_in_exclusion_zone(self, contour):
        """Check if a debris contour overlaps with any exclusion zone"""
        # Access exclusion zones from line detector
        if not hasattr(self.line_detector, 'exclusion_zones') or not self.line_detector.exclusion_zones:
            return False
        
        # Get bounding box of the contour for checking
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
                if self.debug:
                    print(f"    Debris at ({x}, {y}) excluded by zone '{zone['name']}'")
                return True
        
        return False
    
    def detect_debris(self, lines_removed_image):
        """Detect debris (dark regions) with aggressive contrast enhancement"""
        # Convert background colors to grayscale values for reference:
        # rgb(193, 179, 157) -> grayscale ≈ 180
        # rgb(142, 131, 115) -> grayscale ≈ 133
        # We'll treat anything above the configured threshold as background (to be ignored/white)
        # and anything below the threshold as potential debris (to be highlighted/black)
        
        if self.debug:
            print(f"Original image range: {lines_removed_image.min()} to {lines_removed_image.max()}")
            print(f"Using background threshold: {self.background_threshold} (anything above = background, below = debris)")
        
        # Apply aggressive contrast enhancement
        # Step 1: Use configured background threshold - anything above this is background
        background_threshold = self.background_threshold
        
        # Step 2: Create binary mask - anything darker than background becomes debris
        # Use inverse threshold so debris (dark) becomes white (255) and background becomes black (0)
        _, debris_mask = cv2.threshold(lines_removed_image, background_threshold, 255, cv2.THRESH_BINARY_INV)
        
        if self.debug:
            print(f"Debris mask after aggressive thresholding: {np.sum(debris_mask > 0)} white pixels (potential debris)")
        
        # Step 3: Clean up the mask with morphological operations (single filter as requested)
        # Use closing to fill small gaps and connect nearby debris
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        debris_mask = cv2.morphologyEx(debris_mask, cv2.MORPH_CLOSE, kernel)
        
        if self.debug:
            print(f"Debris mask after morphological cleaning: {np.sum(debris_mask > 0)} white pixels")
        
        # Find contours of debris regions
        contours, _ = cv2.findContours(debris_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and exclusion zones
        debris_contours = []
        excluded_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.debris_min_area:
                # Check if debris is in exclusion zone
                if self._is_debris_in_exclusion_zone(contour):
                    excluded_contours.append(contour)
                else:
                    debris_contours.append(contour)
        
        if self.debug:
            print(f"Found {len(contours)} total contours, {len(debris_contours)} above minimum area ({self.debris_min_area}px²)")
            if excluded_contours:
                print(f"Excluded {len(excluded_contours)} debris contours in exclusion zones")
        
        return debris_contours, debris_mask
    
    def create_debris_visualization(self, image, debris_contours, matched_lines):
        """Create visualization highlighting debris regions"""
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
        
        # Create overlay for highlighting
        overlay = vis.copy()
        
        # First draw the detected lines (for reference) - only valid slopes
        for match in matched_lines:
            if match.get('valid_slope', True):  # Only show valid slope lines
                left_pt = match['left']
                right_pt = match['right']
                cv2.line(overlay, (left_pt['x'], left_pt['y']), 
                        (right_pt['x'], right_pt['y']), 
                        (0, 255, 0), self.line_thickness)
        
        # Create a separate overlay for debris
        debris_overlay = vis.copy()
        
        # Fill debris regions with red highlight
        cv2.drawContours(debris_overlay, debris_contours, -1, (0, 0, 255), -1)
        
        # Blend the debris overlay with less opacity for transparency
        result = cv2.addWeighted(overlay, 0.7, debris_overlay, 0.3, 0)
        
        # Draw bounding boxes around debris
        for contour in debris_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Add small label with area
            area = cv2.contourArea(contour)
            cv2.putText(result, f"{int(area)}", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Draw exclusion zones if they exist
        if hasattr(self.line_detector, 'exclusion_zones') and self.line_detector.exclusion_zones:
            for i, zone in enumerate(self.line_detector.exclusion_zones):
                # Convert zone coordinates to proper order
                x1 = min(zone['top_x'], zone['bottom_x'])
                y1 = min(zone['top_y'], zone['bottom_y'])
                x2 = max(zone['top_x'], zone['bottom_x'])
                y2 = max(zone['top_y'], zone['bottom_y'])
                
                # Draw exclusion zone as magenta rectangle
                cv2.rectangle(result, (x1, y1), (x2, y2), (255, 0, 255), 3)
                
                # Add zone label
                label = f"Exclusion {i+1}: {zone.get('name', 'unnamed')}"
                cv2.putText(result, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Add text summary with shadow effect
        text = f"Lines: {len(matched_lines)} | Debris: {len(debris_contours)} regions"
        if hasattr(self.line_detector, 'exclusion_zones') and self.line_detector.exclusion_zones:
            text += f" | Exclusions: {len(self.line_detector.exclusion_zones)}"
        # Shadow
        cv2.putText(result, text, (11, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        # Main text
        cv2.putText(result, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return result
    
    def detect(self, image, image_path=None):
        """Main detection method"""
        # Use line detector to find lines
        matched_lines, left_lines, right_lines, left_kernels, right_kernels = self.line_detector.detect_lines(image, self.debug, image_path)
        
        # Remove lines from image and detect debris
        if matched_lines:
            # Get grayscale version
            if len(image.shape) == 3:
                gray_for_debris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_for_debris = image.copy()
            
            # Remove lines from image
            lines_removed = self.remove_lines_from_image(gray_for_debris, matched_lines)
            
            # Detect debris in the lines-removed image
            debris_contours, dark_regions = self.detect_debris(lines_removed)
            
            # Create debris visualization (this is our main result)
            visualization = self.create_debris_visualization(image, debris_contours, matched_lines)
            
            if self.debug:
                print(f"Detected {len(debris_contours)} debris regions after removing {len(matched_lines)} lines")
                if hasattr(self.line_detector, 'exclusion_zones') and self.line_detector.exclusion_zones:
                    print(f"Exclusion zones loaded: {len(self.line_detector.exclusion_zones)} zones")
                # Store debug images
                self._debug_lines_removed_image = lines_removed
                self._debug_enhanced_debris_mask = dark_regions
        else:
            # No lines detected, just show original
            if len(image.shape) == 2:
                visualization = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                visualization = image.copy()
            cv2.putText(visualization, "No lines detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            debris_contours = []
        
        # Create debug visualizations if debug mode is enabled
        if self.debug:
            self._debug_kernel_image = self.create_debug_kernel_visualization(image, left_kernels, right_kernels, matched_lines)
            # Create line points debug image - show ALL detections (left_lines and right_lines)
            if left_lines or right_lines:
                self._debug_line_points_image = self.create_line_points_visualization(image, matched_lines, left_lines, right_lines)
        
        # Prepare defects/results
        defects = []
        
        # Add line detection results
        if matched_lines:
            # Get line statistics from line detector
            line_stats = self.line_detector.get_line_statistics(matched_lines)
            
            # Count valid vs invalid lines
            valid_lines = [l for l in matched_lines if l.get('valid_slope', True)]
            invalid_lines = [l for l in matched_lines if not l.get('valid_slope', True)]
            
            # Count real vs ghost lines
            left_real = len([l for l in left_lines if l.get('type', 'real') == 'real'])
            left_ghost = len([l for l in left_lines if l.get('type', 'real') == 'ghost'])
            right_real = len([l for l in right_lines if l.get('type', 'real') == 'real'])
            right_ghost = len([l for l in right_lines if l.get('type', 'real') == 'ghost'])
            
            defects.append({
                'type': 'lines_detected',
                'line_count': line_stats['line_count'],
                'average_y_delta': line_stats['average_y_delta'],
                'std_y_delta': line_stats['std_y_delta'],
                'left_detections': len(left_lines),
                'right_detections': len(right_lines),
                'matched_lines': len(matched_lines),
                'valid_lines': len(valid_lines),
                'invalid_lines': len(invalid_lines),
                'left_real': left_real,
                'left_ghost': left_ghost,
                'right_real': right_real,
                'right_ghost': right_ghost
            })
            
            if self.debug:
                print(f"FINAL LINE STATISTICS:")
                print(f"  Total matched lines: {len(matched_lines)}")
                print(f"  Valid slopes: {len(valid_lines)}, Invalid slopes: {len(invalid_lines)}")
                print(f"  Left side: {left_real} real + {left_ghost} ghost = {len(left_lines)} total")
                print(f"  Right side: {right_real} real + {right_ghost} ghost = {len(right_lines)} total")
                print(f"  Line statistics: avg delta_y={line_stats['average_y_delta']:.1f}, std={line_stats['std_y_delta']:.1f}")
        
        # Add debris detection results
        if 'debris_contours' in locals() and debris_contours:
            debris_info = []
            for contour in debris_contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                debris_info.append({
                    'bbox': (int(x), int(y), int(w), int(h)),
                    'area': float(area),
                    'center': (int(x + w/2), int(y + h/2))
                })
            
            defects.append({
                'type': 'debris_detected',
                'debris_count': len(debris_contours),
                'debris_regions': debris_info
            })
        
        return visualization, defects
    
    def save_debug_images(self, output_dir, base_name):
        """Save debug images if debug mode is enabled"""
        debug_paths = []
        
        if self.debug:
            # Save kernel debug image
            if hasattr(self, '_debug_kernel_image') and self._debug_kernel_image is not None:
                debug_path = os.path.join(output_dir, f"{base_name}_kernel_debug.jpg")
                cv2.imwrite(debug_path, self._debug_kernel_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                debug_paths.append(debug_path)
            
            # Save lines removed image
            if hasattr(self, '_debug_lines_removed_image') and self._debug_lines_removed_image is not None:
                lines_removed_path = os.path.join(output_dir, f"{base_name}_lines_removed.jpg")
                cv2.imwrite(lines_removed_path, self._debug_lines_removed_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                debug_paths.append(lines_removed_path)
            
            # Save enhanced debris mask
            if hasattr(self, '_debug_enhanced_debris_mask') and self._debug_enhanced_debris_mask is not None:
                enhanced_debris_path = os.path.join(output_dir, f"{base_name}_enhanced_debris.jpg")
                cv2.imwrite(enhanced_debris_path, self._debug_enhanced_debris_mask, [cv2.IMWRITE_JPEG_QUALITY, 95])
                debug_paths.append(enhanced_debris_path)
            
            # Save line points debug image
            if hasattr(self, '_debug_line_points_image') and self._debug_line_points_image is not None:
                line_points_path = os.path.join(output_dir, f"{base_name}_line_points.jpg")
                cv2.imwrite(line_points_path, self._debug_line_points_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                debug_paths.append(line_points_path)
        
        return debug_paths if debug_paths else None