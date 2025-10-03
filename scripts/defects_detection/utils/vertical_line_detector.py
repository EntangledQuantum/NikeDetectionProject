"""
Vertical Line Detection Utility for Stripe Images
Detects vertical stripe lines by scanning from top and bottom
Returns a four-sided polygon (parallelogram or rectangle) representing the line

Author: Assistant
Date: 2024
"""

import cv2
import numpy as np
import json
import os
import math
from pathlib import Path


class VerticalLineDetector:
    """Detects vertical line in stripe images by scanning from top and bottom"""
    
    def __init__(self, sensitivity='medium'):
        """
        Args:
            sensitivity: Detection sensitivity level ('low', 'medium', 'high')
        """
        self.sensitivity = sensitivity
        self.exclusion_zones = []
        
        # Fixed constants for line detection validation (similar to horizontal line detector)
        # For vertical lines, X_DELTA is the horizontal distance between top and bottom detections
        self.X_DELTA_MIN = 85
        self.X_DELTA_MAX = 120
        
        # Ideal image dimensions (based on typical stripe image)
        self.IDEAL_IMAGE_WIDTH = 1230  # Typical stripe width
        self.IDEAL_IMAGE_HEIGHT = 44167  # Typical stripe height
        
        # Set base parameters - kernel should be small and square for detection
        if sensitivity == 'high':
            self.base_kernel_width = 5
            self.base_kernel_height = 60
            self.line_detection_threshold = 0.10  # 10% of pixels must be BLACK to detect line
            self.num_kernels_per_side = 30  # Number of kernels to scan from each edge
            self.x_shift_threshold = 50  # X deviation to trigger new print head polygon
        elif sensitivity == 'low':
            self.base_kernel_width = 10
            self.base_kernel_height = 100
            self.line_detection_threshold = 0.30  # 30% of pixels must be BLACK
            self.num_kernels_per_side = 20
            self.x_shift_threshold = 100
        else:  # medium (default)
            self.base_kernel_width = 5
            self.base_kernel_height = 10
            self.line_detection_threshold = 0.80  # 20% of pixels must be BLACK to detect line
            self.num_kernels_per_side = 70  # Scan 30 kernels from left, 30 from right per row
            self.x_shift_threshold = 100  # If X shifts more than 50px, new print head
        
        # These will be set dynamically based on actual image size
        self.kernel_width = self.base_kernel_width
        self.kernel_height = self.base_kernel_height
        
        if sensitivity == 'medium':
            print(f"TUNABLE PARAMETERS:")
            print(f"  num_kernels_per_side: {self.num_kernels_per_side} (how many kernels from each edge per row)")
            print(f"  line_detection_threshold: {self.line_detection_threshold} (fraction of BLACK pixels needed to detect line)")
            print(f"  x_shift_threshold: {self.x_shift_threshold}px (X deviation to start new polygon/print head)")
    
    def calculate_scaled_kernel_dimensions(self, image_width, image_height, debug=False):
        """Calculate scaled kernel dimensions based on image size"""
        width_scale = image_width / self.IDEAL_IMAGE_WIDTH
        height_scale = image_height / self.IDEAL_IMAGE_HEIGHT
        
        # Scale kernel dimensions
        scaled_kernel_width = max(1, int(self.base_kernel_width * width_scale))
        scaled_kernel_height = max(1, int(self.base_kernel_height * height_scale))
        
        if debug:
            print(f"KERNEL SCALING:")
            print(f"  Image: {image_width}x{image_height} | Ideal: {self.IDEAL_IMAGE_WIDTH}x{self.IDEAL_IMAGE_HEIGHT}")
            print(f"  Scales: width={width_scale:.3f}, height={height_scale:.3f}")
            print(f"  Kernel: {scaled_kernel_width}x{scaled_kernel_height} (from base {self.base_kernel_width}x{self.base_kernel_height})")
        
        return scaled_kernel_width, scaled_kernel_height
    
    def update_kernel_dimensions_for_image(self, image_width, image_height, debug=False):
        """Update kernel dimensions for the current image size"""
        self.kernel_width, self.kernel_height = self.calculate_scaled_kernel_dimensions(
            image_width, image_height, debug
        )
    
    def load_exclusion_zones(self, image_path):
        """Load exclusion zones from JSON file with same name as image"""
        try:
            image_path = Path(image_path)
            json_path = image_path.with_suffix('.json')
            
            if not json_path.exists():
                self.exclusion_zones = []
                return
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            self.exclusion_zones = []
            if 'exclusion_zones' in data:
                for zone in data['exclusion_zones']:
                    bbox = zone.get('bounding_box_pixels', {})
                    
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
        except Exception as e:
            print(f"Warning: Could not load exclusion zones: {e}")
            self.exclusion_zones = []
    
    def is_in_exclusion_zone(self, x, y, width, height):
        """Check if a kernel region overlaps with any exclusion zone"""
        for zone in self.exclusion_zones:
            zone_x1 = min(zone['top_x'], zone['bottom_x'])
            zone_y1 = min(zone['top_y'], zone['bottom_y'])
            zone_x2 = max(zone['top_x'], zone['bottom_x'])
            zone_y2 = max(zone['top_y'], zone['bottom_y'])
            
            kernel_x1 = x - width // 2
            kernel_y1 = y - height // 2
            kernel_x2 = x + width // 2
            kernel_y2 = y + height // 2
            
            if (kernel_x1 < zone_x2 and kernel_x2 > zone_x1 and
                kernel_y1 < zone_y2 and kernel_y2 > zone_y1):
                return True, zone
        
        return False, None
    
    def scan_rows_for_line_edges(self, binary_image, debug=False):
        """Scan each row to find left and right edges of the vertical line
        
        Returns left and right X positions for each Y row
        """
        height, width = binary_image.shape
        left_edge_points = []
        right_edge_points = []
        all_kernel_states = []
        
        # Scan row by row from top to bottom (STACKED)
        y = self.kernel_height // 2
        
        if debug:
            print(f"Scanning rows: Y={y} to {height}, step={self.kernel_height}px")
        
        while y < height - self.kernel_height // 2:
            # Scan from LEFT - ALWAYS scan num_kernels_per_side kernels
            x = self.kernel_width // 2
            left_edge_x = None
            kernels_scanned_left = 0
            
            while x < width - self.kernel_width // 2 and kernels_scanned_left < self.num_kernels_per_side:
                y1 = max(0, y - self.kernel_height // 2)
                y2 = min(height, y + self.kernel_height // 2)
                x1 = max(0, x - self.kernel_width // 2)
                x2 = min(width, x + self.kernel_width // 2)
                
                kernel_region = binary_image[y1:y2, x1:x2]
                black_pixels = np.sum(kernel_region == 0)  # Count BLACK pixels (line)
                total_pixels = (y2 - y1) * (x2 - x1)
                
                if total_pixels > 0:
                    black_ratio = black_pixels / total_pixels
                    has_line = black_ratio > self.line_detection_threshold
                    
                    # If kernel is in exclusion zone, don't register as line
                    in_exclusion, _ = self.is_in_exclusion_zone(x, y, self.kernel_width, self.kernel_height)
                    if in_exclusion:
                        has_line = False  # Override - exclusion zone means ignore this kernel
                    
                    all_kernel_states.append({
                        'x': x, 'y': y, 'has_line': has_line,
                        'bbox': (x1, y1, x2, y2), 'pixel_ratio': black_ratio, 'side': 'left'
                    })
                    
                    if has_line and left_edge_x is None:
                        left_edge_x = x  # First detection from left
                
                x += self.kernel_width
                kernels_scanned_left += 1
            
            # Scan from RIGHT - ALWAYS scan num_kernels_per_side kernels
            x = width - self.kernel_width // 2 - 1
            right_edge_x = None
            kernels_scanned_right = 0
            
            while x > self.kernel_width // 2 and kernels_scanned_right < self.num_kernels_per_side:
                y1 = max(0, y - self.kernel_height // 2)
                y2 = min(height, y + self.kernel_height // 2)
                x1 = max(0, x - self.kernel_width // 2)
                x2 = min(width, x + self.kernel_width // 2)
                
                kernel_region = binary_image[y1:y2, x1:x2]
                black_pixels = np.sum(kernel_region == 0)  # Count BLACK pixels (line)
                total_pixels = (y2 - y1) * (x2 - x1)
                
                if total_pixels > 0:
                    black_ratio = black_pixels / total_pixels
                    has_line = black_ratio > self.line_detection_threshold
                    
                    # If kernel is in exclusion zone, don't register as line
                    in_exclusion, _ = self.is_in_exclusion_zone(x, y, self.kernel_width, self.kernel_height)
                    if in_exclusion:
                        has_line = False  # Override - exclusion zone means ignore this kernel
                    
                    all_kernel_states.append({
                        'x': x, 'y': y, 'has_line': has_line,
                        'bbox': (x1, y1, x2, y2), 'pixel_ratio': black_ratio, 'side': 'right'
                    })
                    
                    if has_line and right_edge_x is None:
                        right_edge_x = x  # First detection from right
                
                x -= self.kernel_width
                kernels_scanned_right += 1
            
            # Record edges if found
            if left_edge_x is not None:
                left_edge_points.append({'x': left_edge_x, 'y': y})
            if right_edge_x is not None:
                right_edge_points.append({'x': right_edge_x, 'y': y})
            
            # Move to next row
            y += self.kernel_height
        
        if debug:
            print(f"Found {len(left_edge_points)} left edges, {len(right_edge_points)} right edges")
        
        return left_edge_points, right_edge_points, all_kernel_states
    
    def create_polygons_from_edges(self, left_points, right_points, image_width, debug=False):
        """Create polygons by tracking X position changes (print head boundaries)
        
        When left or right X shifts > threshold, new print head starts
        """
        if not left_points or not right_points:
            return []
        
        # Sort by Y
        left_sorted = sorted(left_points, key=lambda p: p['y'])
        right_sorted = sorted(right_points, key=lambda p: p['y'])
        
        # Match left and right by Y position
        min_len = min(len(left_sorted), len(right_sorted))
        
        # Use the tunable threshold
        x_shift_threshold = self.x_shift_threshold
        
        polygons = []
        segment_start_idx = 0
        prev_left_x = left_sorted[0]['x'] if left_sorted else None
        prev_right_x = right_sorted[0]['x'] if right_sorted else None
        
        min_width = int(image_width * 0.7)
        
        for i in range(1, min_len):
            left_x = left_sorted[i]['x']
            right_x = right_sorted[i]['x']
            
            # Check if X position shifted significantly
            left_shift = abs(left_x - prev_left_x) if prev_left_x else 0
            right_shift = abs(right_x - prev_right_x) if prev_right_x else 0
            
            # If significant shift, this is a new print head
            if left_shift > x_shift_threshold or right_shift > x_shift_threshold:
                # Create polygon for previous segment
                if i - segment_start_idx >= 2:
                    seg_left = left_sorted[segment_start_idx:i]
                    seg_right = right_sorted[segment_start_idx:i]
                    
                    # Top and bottom Y
                    top_y = seg_left[0]['y']
                    bottom_y = seg_left[-1]['y']
                    
                    # Top and bottom X (median)
                    top_left_x = np.median([p['x'] for p in seg_left[:len(seg_left)//2]]) if len(seg_left) > 1 else seg_left[0]['x']
                    top_right_x = np.median([p['x'] for p in seg_right[:len(seg_right)//2]]) if len(seg_right) > 1 else seg_right[0]['x']
                    bot_left_x = np.median([p['x'] for p in seg_left[len(seg_left)//2:]]) if len(seg_left) > 1 else seg_left[-1]['x']
                    bot_right_x = np.median([p['x'] for p in seg_right[len(seg_right)//2:]]) if len(seg_right) > 1 else seg_right[-1]['x']
                    
                    # Width
                    width = max(min_width, int(abs(top_right_x - top_left_x)))
                    
                    # Create 4 corners
                    top_left = (int(top_left_x), int(top_y))
                    top_right = (int(top_right_x), int(top_y))
                    bottom_right = (int(bot_right_x), int(bottom_y))
                    bottom_left = (int(bot_left_x), int(bottom_y))
                    
                    polygon = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
                    
                    x_shift = abs((bot_left_x + bot_right_x)/2 - (top_left_x + top_right_x)/2)
                    
                    polygons.append({
                        'polygon': polygon,
                        'width': width,
                        'segment_index': len(polygons),
                        'top_x': (top_left_x + top_right_x) / 2,
                        'bottom_x': (bot_left_x + bot_right_x) / 2,
                        'x_shift': x_shift,
                        'is_slanted': x_shift > 5,
                        'top_y': top_y,
                        'bottom_y': bottom_y,
                        'shape_type': 'parallelogram' if x_shift > 5 else 'rectangle'
                    })
                
                if debug:
                    print(f"  Print head boundary at Y={left_sorted[i]['y']} (left shift={left_shift:.0f}px, right shift={right_shift:.0f}px)")
                
                # Start new segment
                segment_start_idx = i
            
            prev_left_x = left_x
            prev_right_x = right_x
        
        # Add final segment
        if min_len - segment_start_idx >= 2:
            seg_left = left_sorted[segment_start_idx:]
            seg_right = right_sorted[segment_start_idx:]
            
            top_y = seg_left[0]['y']
            bottom_y = seg_left[-1]['y']
            
            top_left_x = np.median([p['x'] for p in seg_left[:len(seg_left)//2]]) if len(seg_left) > 1 else seg_left[0]['x']
            top_right_x = np.median([p['x'] for p in seg_right[:len(seg_right)//2]]) if len(seg_right) > 1 else seg_right[0]['x']
            bot_left_x = np.median([p['x'] for p in seg_left[len(seg_left)//2:]]) if len(seg_left) > 1 else seg_left[-1]['x']
            bot_right_x = np.median([p['x'] for p in seg_right[len(seg_right)//2:]]) if len(seg_right) > 1 else seg_right[-1]['x']
            
            width = max(min_width, int(abs(top_right_x - top_left_x)))
            
            top_left = (int(top_left_x), int(top_y))
            top_right = (int(top_right_x), int(top_y))
            bottom_right = (int(bot_right_x), int(bottom_y))
            bottom_left = (int(bot_left_x), int(bottom_y))
            
            polygon = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
            
            x_shift = abs((bot_left_x + bot_right_x)/2 - (top_left_x + top_right_x)/2)
            
            polygons.append({
                'polygon': polygon,
                'width': width,
                'segment_index': len(polygons),
                'top_x': (top_left_x + top_right_x) / 2,
                'bottom_x': (bot_left_x + bot_right_x) / 2,
                'x_shift': x_shift,
                'is_slanted': x_shift > 5,
                'top_y': top_y,
                'bottom_y': bottom_y,
                'shape_type': 'parallelogram' if x_shift > 5 else 'rectangle'
            })
        
        if debug:
            print(f"\nCREATED {len(polygons)} PRINT HEAD POLYGONS")
            for i, p in enumerate(polygons):
                print(f"  Head {i+1}: Y=[{p['top_y']:.0f}-{p['bottom_y']:.0f}], WIDTH={p['width']}px")
        
        return polygons
    
    def detect_vertical_line(self, image, debug=False, image_path=None):
        """
        Detect vertical line - returns MULTIPLE polygons (one per print head)
        
        Args:
            image: Input image
            debug: Print debug info
            image_path: For exclusion zones
            
        Returns:
            tuple: (polygons_list, left_points, right_points, kernel_states)
        """
        if image_path:
            self.load_exclusion_zones(image_path)
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        height, width = gray.shape
        self.update_kernel_dimensions_for_image(width, height, debug)
        
        # Convert to binary: Background=white(255), Line=black(0)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        # Now: background=255 (white), line=0 (black)
        
        # Scan all rows to find left and right edges
        left_points, right_points, kernel_states = self.scan_rows_for_line_edges(binary, debug)
        
        # Create MULTIPLE polygons (one per print head)
        polygons_list = self.create_polygons_from_edges(left_points, right_points, width, debug)
        
        return polygons_list, left_points, right_points, kernel_states, []
    
    def create_visualization(self, image, polygons_list, left_points, right_points, kernel_states, unused):
        """Create debug visualization"""
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
        
        overlay = vis.copy()
        
        # Draw THE polygon (only one)
        if polygons_list:
            poly_data = polygons_list[0]
            polygon = poly_data['polygon']
            
            poly_overlay = overlay.copy()
            cv2.fillPoly(poly_overlay, [polygon], (0, 255, 0))
            overlay = cv2.addWeighted(overlay, 0.8, poly_overlay, 0.2, 0)
            cv2.polylines(overlay, [polygon], True, (0, 255, 0), 3)
        
        # Draw edge points
        for pt in left_points:
            cv2.circle(overlay, (pt['x'], pt['y']), 3, (0, 0, 255), -1)
        
        for pt in right_points:
            cv2.circle(overlay, (pt['x'], pt['y']), 3, (255, 0, 0), -1)
        
        # Draw kernels
        for state in kernel_states:
            x1, y1, x2, y2 = state['bbox']
            if state.get('side') == 'left':
                color = (0, 255, 255) if state['has_line'] else (80, 80, 80)
            else:
                color = (255, 255, 0) if state['has_line'] else (80, 80, 80)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
        
        result = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        
        cv2.putText(result, "Green: Vertical line polygon", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result

