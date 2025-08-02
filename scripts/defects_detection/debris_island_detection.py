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


class DebrisIslandDetector(BaseDetector):
    """Detects lines in island images by scanning from both sides"""
    
    def __init__(self, sensitivity='medium', debug=False):
        """
        Args:
            sensitivity: Detection sensitivity level ('low', 'medium', 'high')
            debug: Whether to enable debug visualization
        """
        self.debug = debug
        
        print(f"Sensitivity: {sensitivity}")
        
        # Set all parameters based on sensitivity
        if sensitivity == 'high':
            self.kernel_width = 20
            self.kernel_height = 20
            self.num_vertical_scans = 7  # More scans for high sensitivity
            self.line_detection_threshold = 0.10
            self.line_thickness = 5
            self.min_detection_count = 2
            self.min_distance = 30  # More sensitive - allow closer lines
            self.debris_threshold = 80  # Lower threshold - more sensitive to dark regions
            self.debris_min_area = 10  # Detect smaller debris
        elif sensitivity == 'low':
            self.kernel_width = 40
            self.kernel_height = 40
            self.num_vertical_scans = 3  # Fewer scans for low sensitivity
            self.line_detection_threshold = 0.20
            self.line_thickness = 5
            self.min_detection_count = 3  # Require more detections
            self.min_distance = 80  # Less sensitive - require more separation
            self.debris_threshold = 120  # Higher threshold - less sensitive
            self.debris_min_area = 50  # Only detect larger debris
        else:  # medium (default)
            self.kernel_width = 10
            self.kernel_height = 50
            self.num_vertical_scans = 50
            self.line_detection_threshold = 0.05
            self.line_thickness = 20
            self.min_detection_count = 10
            self.min_distance = 50
            self.debris_threshold = 1.0  # Medium threshold
            self.debris_min_area = 10  # Medium size requirement
        
        # Store debug images
        self._debug_kernel_image = None
        self._debug_lines_removed_image = None
        self._debug_debris_mask = None
        
        # Print configuration
        print(f"Debris Island Detector Configuration:")
        print(f"  Kernel size: {self.kernel_width}x{self.kernel_height}")
        print(f"  Vertical scans per side: {self.num_vertical_scans}")
        print(f"  Line detection threshold: {self.line_detection_threshold}")
        print(f"  Min detection count: {self.min_detection_count}")
        print(f"  Min line distance: {self.min_distance}px")
        print(f"  Debris threshold: {self.debris_threshold}")
        print(f"  Min debris area: {self.debris_min_area}px²")
    
    def scan_vertical_column(self, binary_image, x_position, scan_from_top=True):
        """Scan a vertical column to find all horizontal lines"""
        height, width = binary_image.shape
        detected_lines = []
        kernel_states = []
        
        # Ensure x position is valid
        if x_position < self.kernel_width // 2 or x_position >= width - self.kernel_width // 2:
            return detected_lines, kernel_states
        
        # Start scanning from top or continue from where we left
        if scan_from_top:
            y = self.kernel_height // 2
        else:
            y = self.kernel_height // 2
        
        while y < height - self.kernel_height // 2:
            # Extract kernel region
            y1 = max(0, y - self.kernel_height // 2)
            y2 = min(height, y + self.kernel_height // 2)
            x1 = max(0, x_position - self.kernel_width // 2)
            x2 = min(width, x_position + self.kernel_width // 2)
            
            kernel_region = binary_image[y1:y2, x1:x2]
            
            # Check if there's a line in kernel
            white_pixels = np.sum(kernel_region > 0)
            total_pixels = (y2 - y1) * (x2 - x1)
            
            if total_pixels > 0:
                pixel_ratio = white_pixels / total_pixels
                has_line = pixel_ratio > self.line_detection_threshold
                
                if has_line:
                    # Calculate center of detected pixels
                    y_indices, x_indices = np.where(kernel_region > 0)
                    if len(y_indices) > 0:
                        local_y_center = np.mean(y_indices)
                        line_y = y1 + int(local_y_center)
                        detected_lines.append({
                            'x': x_position,
                            'y': line_y,
                            'strength': pixel_ratio
                        })
                
                # Record kernel state for debug
                if self.debug:
                    kernel_states.append({
                        'x': x_position,
                        'y': y,
                        'has_line': has_line,
                        'bbox': (x1, y1, x2, y2),
                        'pixel_ratio': pixel_ratio
                    })
            
            # Move to next position (no overlap)
            y += self.kernel_height
        
        return detected_lines, kernel_states
    
    def scan_from_left(self, binary_image):
        """Scan multiple vertical columns from the left side
        
        For each detected line, we want the LEFTMOST kernel position,
        which represents where the line starts on the left edge.
        """
        height, width = binary_image.shape
        all_kernel_states = []
        lines_by_y = {}  # Group lines by approximate Y position
        
        # Start from left edge
        x_start = self.kernel_width // 2
        
        for i in range(self.num_vertical_scans):
            x_position = x_start + i * self.kernel_width  # No overlap
            
            if x_position >= width - self.kernel_width // 2:
                break
            
            lines, states = self.scan_vertical_column(binary_image, x_position)
            
            if self.debug:
                print(f"Left scan {i} at x={x_position}: Found {len(lines)} lines")
            
            # Group lines by approximate Y position
            for line in lines:
                y_key = int(line['y'] / self.kernel_height) * self.kernel_height  # Group by kernel height
                if y_key not in lines_by_y:
                    lines_by_y[y_key] = []
                lines_by_y[y_key].append(line)
            
            all_kernel_states.extend(states)
        
        # Filter lines that have at least min_detection_count detections and min_distance separation
        filtered_lines = []
        last_y = None
        
        for y_key in sorted(lines_by_y.keys()):
            line_group = lines_by_y[y_key]
            if len(line_group) >= self.min_detection_count:
                # Use the leftmost detection (smallest x) as the line position
                leftmost_line = min(line_group, key=lambda l: l['x'])
                
                # Check minimum distance from previous line
                if last_y is None or abs(leftmost_line['y'] - last_y) >= self.min_distance:
                    filtered_lines.append(leftmost_line)
                    last_y = leftmost_line['y']
                    if self.debug:
                        x_positions = [l['x'] for l in line_group]
                        print(f"Left line at Y≈{y_key}: {len(line_group)} detections at x={x_positions}, using leftmost at x={leftmost_line['x']}, y={leftmost_line['y']}")
                elif self.debug:
                    print(f"Left line at Y≈{y_key}: Skipped - too close to previous line (distance={abs(leftmost_line['y'] - last_y)} < {self.min_distance})")
        
        return filtered_lines, all_kernel_states
    
    def scan_from_right(self, binary_image):
        """Scan multiple vertical columns from the right side
        
        For each detected line, we want the RIGHTMOST kernel position,
        which represents where the line ends on the right edge.
        """
        height, width = binary_image.shape
        all_kernel_states = []
        lines_by_y = {}  # Group lines by approximate Y position
        
        # Start from right edge
        x_start = width - self.kernel_width // 2 - 1
        
        for i in range(self.num_vertical_scans):
            x_position = x_start - i * self.kernel_width  # Move left, no overlap
            
            if x_position < self.kernel_width // 2:
                break
            
            lines, states = self.scan_vertical_column(binary_image, x_position)
            
            if self.debug:
                print(f"Right scan {i} at x={x_position}: Found {len(lines)} lines")
            
            # Group lines by approximate Y position
            for line in lines:
                y_key = int(line['y'] / self.kernel_height) * self.kernel_height  # Group by kernel height
                if y_key not in lines_by_y:
                    lines_by_y[y_key] = []
                lines_by_y[y_key].append(line)
            
            all_kernel_states.extend(states)
        
        # Filter lines that have at least min_detection_count detections and min_distance separation
        filtered_lines = []
        last_y = None
        
        for y_key in sorted(lines_by_y.keys()):
            line_group = lines_by_y[y_key]
            if len(line_group) >= self.min_detection_count:
                # Use the rightmost detection (largest x) as the line position
                rightmost_line = max(line_group, key=lambda l: l['x'])
                
                # Check minimum distance from previous line
                if last_y is None or abs(rightmost_line['y'] - last_y) >= self.min_distance:
                    filtered_lines.append(rightmost_line)
                    last_y = rightmost_line['y']
                    if self.debug:
                        x_positions = [l['x'] for l in line_group]
                        print(f"Right line at Y≈{y_key}: {len(line_group)} detections at x={x_positions}, using rightmost at x={rightmost_line['x']}, y={rightmost_line['y']}")
                elif self.debug:
                    print(f"Right line at Y≈{y_key}: Skipped - too close to previous line (distance={abs(rightmost_line['y'] - last_y)} < {self.min_distance})")
        
        return filtered_lines, all_kernel_states
    
    def match_lines(self, left_lines, right_lines):
        """Match lines by index - 1st left to 1st right, 2nd to 2nd, etc."""
        matched_lines = []
        
        # Simple index-based matching
        num_matches = min(len(left_lines), len(right_lines))
        
        for i in range(num_matches):
            left_line = left_lines[i]
            right_line = right_lines[i]
            
            # Calculate slope for this specific line
            dx = right_line['x'] - left_line['x']
            dy = right_line['y'] - left_line['y']
            slope = dy / dx if dx != 0 else 0
            
            matched_lines.append({
                'left': left_line,
                'right': right_line,
                'slope': slope,
                'y_delta': dy
            })
            
            if self.debug:
                print(f"Matched line {i}: Left({left_line['x']}, {left_line['y']}) -> Right({right_line['x']}, {right_line['y']}), delta_y={dy}")
        
        if self.debug:
            print(f"Matched {len(matched_lines)} lines (left: {len(left_lines)}, right: {len(right_lines)})")
        
        return matched_lines
    
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
        cv2.putText(overlay, f"Min detections required: {self.min_detection_count}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(overlay, f"Min line distance: {self.min_distance}px", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Blend with original
        debug_vis = cv2.addWeighted(debug_vis, 0.6, overlay, 0.4, 0)
        
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
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold to get binary image (lines should be dark/black)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Scan from left side
        left_lines, left_kernels = self.scan_from_left(binary)
        
        # Scan from right side
        right_lines, right_kernels = self.scan_from_right(binary)
        
        # Match lines from left to right
        matched_lines = self.match_lines(left_lines, right_lines)
        
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
        
        # Create debug kernel visualization if debug mode is enabled
        if self.debug:
            self._debug_kernel_image = self.create_debug_kernel_visualization(image, left_kernels, right_kernels, matched_lines)
        
        # Prepare defects/results
        defects = []
        
        # Add line detection results
        if matched_lines:
            # Calculate statistics about the lines
            y_deltas = [m['y_delta'] for m in matched_lines]
            avg_delta = np.mean(y_deltas) if y_deltas else 0
            std_delta = np.std(y_deltas) if y_deltas else 0
            
            defects.append({
                'type': 'lines_detected',
                'line_count': len(matched_lines),
                'average_y_delta': float(avg_delta),
                'std_y_delta': float(std_delta),
                'left_detections': len(left_lines),
                'right_detections': len(right_lines),
                'matched_lines': len(matched_lines)
            })
            
            if self.debug:
                print(f"Line statistics: avg delta_y={avg_delta:.1f}, std={std_delta:.1f}")
        
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
        
        return debug_paths if debug_paths else None