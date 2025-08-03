"""
Line Detection Utility for Island Images
Detects horizontal slanted lines by scanning from both sides
Can be used independently by other scripts

Author: Assistant
Date: 2024
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path


class LineDetector:
    """Detects lines in island images by scanning from both sides"""
    
    def __init__(self, sensitivity='medium'):
        """
        Args:
            sensitivity: Detection sensitivity level ('low', 'medium', 'high')
        """
        self.sensitivity = sensitivity
        self.exclusion_zones = []  # Will be populated when detecting lines
        
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
        
    def load_exclusion_zones(self, image_path):
        """Load exclusion zones from JSON file with same name as image"""
        try:
            # Get JSON file path (same name as image, different extension)
            image_path = Path(image_path)
            json_path = image_path.with_suffix('.json')
            
            if not json_path.exists():
                self.exclusion_zones = []
                return
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Extract exclusion zones from JSON
            self.exclusion_zones = []
            if 'exclusion_zones' in data:
                for zone in data['exclusion_zones']:
                    bbox = zone.get('bounding_box_pixels', {})
                    
                    # Convert coordinates like tiff_extractor.py does:
                    # Convert to positive values if negative and ensure proper ordering
                    raw_top_x = float(bbox.get('top_x', 0))
                    raw_top_y = float(bbox.get('top_y', 0))
                    raw_bottom_x = float(bbox.get('bottom_x', 0))
                    raw_bottom_y = float(bbox.get('bottom_y', 0))
                    
                    x1 = int(min(abs(raw_top_x), abs(raw_bottom_x)))
                    y1 = int(min(abs(raw_top_y), abs(raw_bottom_y)))
                    x2 = int(max(abs(raw_top_x), abs(raw_bottom_x)))
                    y2 = int(max(abs(raw_top_y), abs(raw_bottom_y)))
                    
                    self.exclusion_zones.append({
                        'top_x': x1,
                        'top_y': y1,
                        'bottom_x': x2,
                        'bottom_y': y2,
                        'name': zone.get('name', 'unnamed')
                    })
            
            if self.exclusion_zones:
                print(f"Loaded {len(self.exclusion_zones)} exclusion zones from {json_path}")
                for i, zone in enumerate(self.exclusion_zones):
                    print(f"  Zone {i+1} '{zone['name']}': ({zone['top_x']}, {zone['top_y']}) to ({zone['bottom_x']}, {zone['bottom_y']})")
            
        except Exception as e:
            print(f"Warning: Could not load exclusion zones from {json_path}: {e}")
            self.exclusion_zones = []
    
    def is_in_exclusion_zone(self, x, y, width, height):
        """Check if a kernel region overlaps with any exclusion zone"""
        for zone in self.exclusion_zones:
            # Convert zone coordinates to proper order
            zone_x1 = min(zone['top_x'], zone['bottom_x'])
            zone_y1 = min(zone['top_y'], zone['bottom_y'])
            zone_x2 = max(zone['top_x'], zone['bottom_x'])
            zone_y2 = max(zone['top_y'], zone['bottom_y'])
            
            # Kernel boundaries
            kernel_x1 = x - width // 2
            kernel_y1 = y - height // 2
            kernel_x2 = x + width // 2
            kernel_y2 = y + height // 2
            
            # Check for overlap
            if (kernel_x1 < zone_x2 and kernel_x2 > zone_x1 and
                kernel_y1 < zone_y2 and kernel_y2 > zone_y1):
                return True, zone
        
        return False, None
    
    def get_exclusion_zones_for_side(self, image_width, side='left'):
        """Get exclusion zones that affect the specified side (left or right)"""
        relevant_zones = []
        
        for zone in self.exclusion_zones:
            zone_x1 = min(zone['top_x'], zone['bottom_x'])
            zone_x2 = max(zone['top_x'], zone['bottom_x'])
            zone_center_x = (zone_x1 + zone_x2) / 2
            
            # Determine if zone is on left or right half of image
            if side == 'left' and zone_center_x < image_width / 2:
                relevant_zones.append(zone)
            elif side == 'right' and zone_center_x >= image_width / 2:
                relevant_zones.append(zone)
        
        return relevant_zones
    
    def adjust_x_position_for_exclusions(self, x_position, y_position, image_width, side='left', debug=False):
        """Adjust x position to avoid exclusion zones"""
        # Check if current position is in an exclusion zone
        in_zone, zone = self.is_in_exclusion_zone(x_position, y_position, self.kernel_width, self.kernel_height)
        
        if not in_zone:
            return x_position  # No adjustment needed
        
        if debug:
            print(f"    Kernel at ({x_position}, {y_position}) overlaps with exclusion zone '{zone['name']}'")
        
        # Adjust position based on side
        if side == 'left':
            # For left side, move right to avoid the zone
            zone_right = max(zone['top_x'], zone['bottom_x'])
            new_x = zone_right + self.kernel_width // 2 + 5  # Small buffer
            # Make sure we don't go too far right
            if new_x < image_width - self.kernel_width // 2:
                if debug:
                    print(f"    Shifted LEFT scanner from x={x_position} to x={new_x} (moved right to avoid zone)")
                return new_x
        else:  # right side
            # For right side, move left to avoid the zone
            zone_left = min(zone['top_x'], zone['bottom_x'])
            new_x = zone_left - self.kernel_width // 2 - 5  # Small buffer
            # Make sure we don't go too far left
            if new_x > self.kernel_width // 2:
                if debug:
                    print(f"    Shifted RIGHT scanner from x={x_position} to x={new_x} (moved left to avoid zone)")
                return new_x
        
        if debug:
            print(f"    Could not adjust position for zone '{zone['name']}' - keeping original x={x_position}")
        return x_position  # Return original if adjustment is not possible
    
    def scan_vertical_column(self, binary_image, x_position, scan_from_top=True, debug=False):
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
            # Adjust x position if it's in an exclusion zone
            adjusted_x = x_position
            if hasattr(self, 'exclusion_zones') and self.exclusion_zones:
                # Determine which side we're scanning from based on x_position
                side = 'left' if x_position < width / 2 else 'right'
                adjusted_x = self.adjust_x_position_for_exclusions(x_position, y, width, side, debug)
            
            # Extract kernel region with adjusted x position
            y1 = max(0, y - self.kernel_height // 2)
            y2 = min(height, y + self.kernel_height // 2)
            x1 = max(0, adjusted_x - self.kernel_width // 2)
            x2 = min(width, adjusted_x + self.kernel_width // 2)
            
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
                            'x': adjusted_x,  # Use adjusted x position
                            'y': line_y,
                            'strength': pixel_ratio
                        })
                
                # Record kernel state for debug (use adjusted position)
                kernel_states.append({
                    'x': adjusted_x,
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
            
            lines, states = self.scan_vertical_column(binary_image, x_position, True, debug)
            
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
            
            lines, states = self.scan_vertical_column(binary_image, x_position, True, debug)
            
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
    
    def detect_lines(self, image, debug=False, image_path=None):
        """
        Detect lines in the image
        
        Args:
            image: Input image (can be grayscale or color)
            debug: Whether to print debug information
            image_path: Path to the image file (for loading exclusion zones)
            
        Returns:
            tuple: (matched_lines, left_lines, right_lines, left_kernel_states, right_kernel_states)
        """
        # Load exclusion zones if image path is provided
        if image_path:
            self.load_exclusion_zones(image_path)
        
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