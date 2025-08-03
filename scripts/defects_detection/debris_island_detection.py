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
        self.debug = debug
        self.sensitivity = sensitivity
        
        print(f"Sensitivity: {sensitivity}")
        
        # Initialize line detector with same sensitivity
        self.line_detector = LineDetector(sensitivity)
        
        # Set debris detection parameters based on sensitivity
        if sensitivity == 'high':
            self.line_thickness = 5
            self.debris_threshold = 80  # Lower threshold - more sensitive to dark regions
            self.debris_min_area = 10  # Detect smaller debris
        elif sensitivity == 'low':
            self.line_thickness = 5
            self.debris_threshold = 120  # Higher threshold - less sensitive
            self.debris_min_area = 50  # Only detect larger debris
        else:  # medium (default)
            self.line_thickness = 20
            self.debris_threshold = 1.0  # Medium threshold
            self.debris_min_area = 10  # Medium size requirement
        
        # Store debug images
        self._debug_kernel_image = None
        self._debug_lines_removed_image = None
        self._debug_debris_mask = None
        self._debug_line_points_image = None
        
        # Print configuration
        print(f"Debris Island Detector Configuration:")
        print(f"  Line thickness for removal: {self.line_thickness}")
        print(f"  Debris threshold: {self.debris_threshold}")
        print(f"  Min debris area: {self.debris_min_area}px²")
    

    
    def create_debug_kernel_visualization(self, image, left_kernels, right_kernels, matched_lines):
        """Create a debug visualization showing both kernels AND lines"""
        if len(image.shape) == 2:
            debug_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            debug_vis = image.copy()
        
        overlay = debug_vis.copy()
        
        # First draw the lines
        for i, match in enumerate(matched_lines):
            left_pt = match['left']
            right_pt = match['right']
            
            # Draw the line from left detection to right detection
            cv2.line(overlay, (left_pt['x'], left_pt['y']), 
                    (right_pt['x'], right_pt['y']), 
                    (0, 255, 0), self.line_thickness)
            
            # Draw dots at detection points
            cv2.circle(overlay, (left_pt['x'], left_pt['y']), 5, (255, 0, 0), -1)  # Blue dot on left
            cv2.circle(overlay, (right_pt['x'], right_pt['y']), 5, (0, 0, 255), -1)  # Red dot on right
            
            if i < 5:  # Show info for first 5 lines
                # Add text showing the delta
                mid_x = (left_pt['x'] + right_pt['x']) // 2
                mid_y = (left_pt['y'] + right_pt['y']) // 2
                cv2.putText(overlay, f"dY={match['y_delta']:.0f}", 
                           (mid_x - 30, mid_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
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
        cv2.putText(overlay, "Green: Left kernels with lines", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(overlay, "Yellow: Right kernels with lines", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(overlay, "Gray: Kernels without lines", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        cv2.putText(overlay, f"Min detections required: {self.line_detector.min_detection_count}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(overlay, f"Min line distance: {self.line_detector.min_distance}px", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Blend with original
        debug_vis = cv2.addWeighted(debug_vis, 0.6, overlay, 0.4, 0)
        
        return debug_vis
    
    def create_line_points_visualization(self, image, matched_lines, left_lines, right_lines):
        """Create a simple visualization showing ALL detected line points with dots"""
        if len(image.shape) == 2:
            debug_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            debug_vis = image.copy()
        
        # Draw dots for ALL left-side detections (red)
        for left_pt in left_lines:
            cv2.circle(debug_vis, (left_pt['x'], left_pt['y']), 15, (0, 0, 255), -1)  # Red dot
        
        # Draw dots for ALL right-side detections (green)  
        for right_pt in right_lines:
            cv2.circle(debug_vis, (right_pt['x'], right_pt['y']), 15, (0, 255, 0), -1)  # Green dot
        
        # Add legend
        cv2.putText(debug_vis, "Red: ALL left detections", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(debug_vis, "Green: ALL right detections", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(debug_vis, f"Left: {len(left_lines)} | Right: {len(right_lines)} | Matched: {len(matched_lines)}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return debug_vis
    
    def remove_lines_from_image(self, gray_image, matched_lines):
        """Remove detected lines by painting them white"""
        # Create a copy to work on
        lines_removed = gray_image.copy()
        
        # Paint each line white with the specified thickness
        for match in matched_lines:
            left_pt = match['left']
            right_pt = match['right']
            
            # Draw white line to remove it
            cv2.line(lines_removed, (left_pt['x'], left_pt['y']), 
                    (right_pt['x'], right_pt['y']), 
                    255, self.line_thickness * 2)  # Make it thicker to ensure complete removal
        
        return lines_removed
    
    def detect_debris(self, lines_removed_image):
        """Detect debris (dark regions) in the image after lines are removed"""
        # Apply threshold to find dark regions
        _, dark_regions = cv2.threshold(lines_removed_image, self.debris_threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Apply morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dark_regions = cv2.morphologyEx(dark_regions, cv2.MORPH_CLOSE, kernel)
        dark_regions = cv2.morphologyEx(dark_regions, cv2.MORPH_OPEN, kernel)
        
        # Find contours of dark regions
        contours, _ = cv2.findContours(dark_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        debris_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.debris_min_area:
                debris_contours.append(contour)
        
        return debris_contours, dark_regions
    
    def create_debris_visualization(self, image, debris_contours, matched_lines):
        """Create visualization highlighting debris regions"""
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
        
        # Create overlay for highlighting
        overlay = vis.copy()
        
        # First draw the detected lines in green (for reference)
        for match in matched_lines:
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
        
        # Add text summary with shadow effect
        text = f"Lines: {len(matched_lines)} | Debris: {len(debris_contours)} regions"
        # Shadow
        cv2.putText(result, text, (11, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        # Main text
        cv2.putText(result, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return result
    
    def detect(self, image):
        """Main detection method"""
        # Use line detector to find lines
        matched_lines, left_lines, right_lines, left_kernels, right_kernels = self.line_detector.detect_lines(image, self.debug)
        
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
                # Store debug images
                self._debug_lines_removed_image = lines_removed
                self._debug_debris_mask = dark_regions
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
            
            defects.append({
                'type': 'lines_detected',
                'line_count': line_stats['line_count'],
                'average_y_delta': line_stats['average_y_delta'],
                'std_y_delta': line_stats['std_y_delta'],
                'left_detections': len(left_lines),
                'right_detections': len(right_lines),
                'matched_lines': len(matched_lines)
            })
            
            if self.debug:
                print(f"Line statistics: avg delta_y={line_stats['average_y_delta']:.1f}, std={line_stats['std_y_delta']:.1f}")
        
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
            
            # Save debris mask
            if hasattr(self, '_debug_debris_mask') and self._debug_debris_mask is not None:
                debris_mask_path = os.path.join(output_dir, f"{base_name}_debris_mask.jpg")
                cv2.imwrite(debris_mask_path, self._debug_debris_mask, [cv2.IMWRITE_JPEG_QUALITY, 95])
                debug_paths.append(debris_mask_path)
            
            # Save line points debug image
            if hasattr(self, '_debug_line_points_image') and self._debug_line_points_image is not None:
                line_points_path = os.path.join(output_dir, f"{base_name}_line_points.jpg")
                cv2.imwrite(line_points_path, self._debug_line_points_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                debug_paths.append(line_points_path)
        
        return debug_paths if debug_paths else None