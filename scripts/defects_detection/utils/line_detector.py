"""
Line Detection Utility for Island Images
Detects horizontal slanted lines by scanning from both sides
Can be used independently by other scripts

Author: Assistant
Date: 2024
"""

import cv2
import numpy as np


class LineDetector:
    """Detects lines in island images by scanning from both sides"""
    
    def __init__(self, sensitivity='medium'):
        """
        Args:
            sensitivity: Detection sensitivity level ('low', 'medium', 'high')
        """
        self.sensitivity = sensitivity
        
        # Set all parameters based on sensitivity
        if sensitivity == 'high':
            self.kernel_width = 20
            self.kernel_height = 20
            self.num_vertical_scans = 7  # More scans for high sensitivity
            self.line_detection_threshold = 0.10
            self.min_detection_count = 2
            self.min_distance = 30  # More sensitive - allow closer lines
        elif sensitivity == 'low':
            self.kernel_width = 40
            self.kernel_height = 40
            self.num_vertical_scans = 3  # Fewer scans for low sensitivity
            self.line_detection_threshold = 0.20
            self.min_detection_count = 3  # Require more detections
            self.min_distance = 80  # Less sensitive - require more separation
        else:  # medium (default)
            self.kernel_width = 10
            self.kernel_height = 50
            self.num_vertical_scans = 50
            self.line_detection_threshold = 0.05
            self.min_detection_count = 10
            self.min_distance = 50
        
        # Prefixed Constants 
    
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
    
    def scan_from_left(self, binary_image, debug=False):
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
            
            if debug:
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
                    if debug:
                        x_positions = [l['x'] for l in line_group]
                        print(f"Left line at Y≈{y_key}: {len(line_group)} detections at x={x_positions}, using leftmost at x={leftmost_line['x']}, y={leftmost_line['y']}")
                elif debug:
                    print(f"Left line at Y≈{y_key}: Skipped - too close to previous line (distance={abs(leftmost_line['y'] - last_y)} < {self.min_distance})")
        
        return filtered_lines, all_kernel_states
    
    def scan_from_right(self, binary_image, debug=False):
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
            
            if debug:
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
                    if debug:
                        x_positions = [l['x'] for l in line_group]
                        print(f"Right line at Y≈{y_key}: {len(line_group)} detections at x={x_positions}, using rightmost at x={rightmost_line['x']}, y={rightmost_line['y']}")
                elif debug:
                    print(f"Right line at Y≈{y_key}: Skipped - too close to previous line (distance={abs(rightmost_line['y'] - last_y)} < {self.min_distance})")
        
        return filtered_lines, all_kernel_states
    
    def match_lines(self, left_lines, right_lines, debug=False):
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
            
            if debug:
                print(f"Matched line {i}: Left({left_line['x']}, {left_line['y']}) -> Right({right_line['x']}, {right_line['y']}), delta_y={dy}")
        
        if debug:
            print(f"Matched {len(matched_lines)} lines (left: {len(left_lines)}, right: {len(right_lines)})")
        
        return matched_lines
    
    def detect_lines(self, image, debug=False):
        """
        Detect lines in the image
        
        Args:
            image: Input image (can be grayscale or color)
            debug: Whether to print debug information
            
        Returns:
            tuple: (matched_lines, left_lines, right_lines, left_kernel_states, right_kernel_states)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold to get binary image (lines should be dark/black)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Scan from left side
        left_lines, left_kernel_states = self.scan_from_left(binary, debug)
        
        # Scan from right side
        right_lines, right_kernel_states = self.scan_from_right(binary, debug)
        
        # Match lines from left to right
        matched_lines = self.match_lines(left_lines, right_lines, debug)
        
        return matched_lines, left_lines, right_lines, left_kernel_states, right_kernel_states
    
    def get_line_statistics(self, matched_lines):
        """Calculate statistics about the detected lines"""
        if not matched_lines:
            return None
        
        y_deltas = [m['y_delta'] for m in matched_lines]
        avg_delta = np.mean(y_deltas) if y_deltas else 0
        std_delta = np.std(y_deltas) if y_deltas else 0
        
        return {
            'line_count': len(matched_lines),
            'average_y_delta': float(avg_delta),
            'std_y_delta': float(std_delta),
            'y_deltas': y_deltas
        }