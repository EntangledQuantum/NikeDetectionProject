"""
Line Defect Detection Algorithm
Detects missing lines and jagged/zig-zag lines in island images

Author: Assistant
Date: 2024
"""

import cv2
import numpy as np
from scipy import signal, ndimage
import os


class LineDefectDetector:
    """Detects line defects using kernel-based line tracking"""
    
    def __init__(self, kernel_size=20, search_range=10,
                 min_gap_size=10, sensitivity='medium', debug=False):
        """
        Args:
            kernel_size: Size of the tracking kernel (square)
            search_range: Vertical search range when line is lost
            min_gap_size: Minimum gap size to consider as defect
            sensitivity: Detection sensitivity level
            debug: Whether to draw debug visualization
        """
        self.kernel_size = kernel_size
        self.search_range = search_range
        self.min_gap_size = min_gap_size
        self.debug = debug
        self.step_size = kernel_size  # Horizontal step to avoid overlap
        
        # Adjust parameters based on sensitivity
        if sensitivity == 'high':
            self.kernel_size = 15
            self.search_range = 15
            self.min_gap_size = 5
            self.step_size = 15
        elif sensitivity == 'low':
            self.kernel_size = 25
            self.search_range = 8
            self.min_gap_size = 20
            self.step_size = 25
    
    def scan_for_lines(self, binary_image):
        """Scan image to find all horizontal lines"""
        height, width = binary_image.shape
        line_starts = []
        
        y = self.kernel_size // 2
        while y < height - self.kernel_size // 2:
            # Check if there's a line at this Y position
            x = self.kernel_size // 2
            found_line = False
            
            # Scan horizontally at this Y level
            while x < width - self.kernel_size // 2 and not found_line:
                # Extract kernel region
                y1 = y - self.kernel_size // 2
                y2 = y + self.kernel_size // 2
                x1 = x - self.kernel_size // 2
                x2 = x + self.kernel_size // 2
                
                kernel_region = binary_image[y1:y2, x1:x2]
                
                # Check if there's a line - at least 10% of pixels should be white
                white_pixels = np.sum(kernel_region > 0)
                total_pixels = self.kernel_size * self.kernel_size
                
                if white_pixels > total_pixels * 0.1:  # 10% threshold
                    # Found a line at this Y position
                    found_line = True
                    line_starts.append(y)
                    if self.debug:
                        print(f"Found line at Y={y}")
                
                x += self.kernel_size  # Move by kernel size to check next position
            
            # Move down by kernel size to avoid overlap with previous scan
            y += self.kernel_size
        
        if self.debug:
            print(f"Total lines detected: {len(line_starts)}")
        
        return line_starts
    
    def track_line(self, binary_image, start_y):
        """Track a single line across the image using kernel"""
        height, width = binary_image.shape
        kernel_states = []  # For debug visualization
        defects = []
        
        # Starting position
        x = self.kernel_size // 2
        y = start_y
        gap_start = None
        previous_y = y  # Keep track of last known good Y position
        
        while x < width - self.kernel_size // 2:
            # Extract kernel region
            y1 = max(0, y - self.kernel_size // 2)
            y2 = min(height, y + self.kernel_size // 2)
            x1 = max(0, x - self.kernel_size // 2)
            x2 = min(width, x + self.kernel_size // 2)
            
            kernel_region = binary_image[y1:y2, x1:x2]
            
            # Check if there's line in kernel - count white pixels
            white_pixels = np.sum(kernel_region > 0)
            total_pixels = (y2 - y1) * (x2 - x1)
            has_line = white_pixels > total_pixels * 0.1  # 10% threshold
            
            if has_line:
                # Calculate centroid of line pixels in kernel to follow the line
                y_indices, x_indices = np.where(kernel_region > 0)
                if len(y_indices) > 0:
                    # Update Y position to follow line
                    local_y_center = np.mean(y_indices)
                    y = y1 + int(local_y_center)
                    previous_y = y  # Update last known good position
                    
                    # If we were in a gap, record it
                    if gap_start is not None:
                        gap_size = x - gap_start
                        if gap_size > self.min_gap_size:
                            defects.append({
                                'type': 'missing_line',
                                'start_x': gap_start,
                                'end_x': x,
                                'y': previous_y,
                                'location': ((gap_start + x) // 2, previous_y),
                                'size': gap_size
                            })
                        gap_start = None
                    
                    # Record kernel state for debug (green box)
                    kernel_states.append({
                        'x': x,
                        'y': y,
                        'has_line': True,
                        'bbox': (x1, y1, x2, y2)
                    })
            else:
                # No line found - try searching vertically
                found = False
                best_y = y
                max_pixels = 0
                
                for dy in range(-self.search_range, self.search_range + 1):
                    test_y = y + dy
                    if 0 <= test_y - self.kernel_size // 2 < height and 0 <= test_y + self.kernel_size // 2 < height:
                        test_y1 = test_y - self.kernel_size // 2
                        test_y2 = test_y + self.kernel_size // 2
                        test_region = binary_image[test_y1:test_y2, x1:x2]
                        
                        white_pixels = np.sum(test_region > 0)
                        if white_pixels > max_pixels:
                            max_pixels = white_pixels
                            best_y = test_y
                        
                        if white_pixels > (self.kernel_size * self.kernel_size) * 0.1:
                            # Found line at different Y
                            y = test_y
                            found = True
                            previous_y = y
                            break
                
                if not found:
                    # Mark start of gap if not already in one
                    if gap_start is None:
                        gap_start = x
                    # Keep Y at previous position for red box
                    y = previous_y
                
                # Record kernel state (red box if no line)
                kernel_states.append({
                    'x': x,
                    'y': y,  # Use previous_y to maintain position
                    'has_line': found,
                    'bbox': (x1, y - self.kernel_size // 2, x2, y + self.kernel_size // 2)
                })
            
            # Move to next position horizontally by kernel size to avoid overlap
            x += self.kernel_size
        
        return kernel_states, defects
    
    def detect(self, image):
        """Main detection method"""
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
        
        # Invert if necessary (lines should be white)
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
        
        # Find all horizontal lines
        line_positions = self.scan_for_lines(binary)
        
        # Track each line
        all_defects = []
        all_kernel_states = []
        
        for i, start_y in enumerate(line_positions):
            if self.debug:
                print(f"Tracking line {i+1} starting at Y={start_y}")
            
            kernel_states, defects = self.track_line(binary, start_y)
            all_defects.extend(defects)
            if self.debug:
                all_kernel_states.extend(kernel_states)
        
        # Create visualization
        visualization = self.create_visualization(image, all_defects, all_kernel_states)
        
        # Return tuple format (visualization, defects)
        return visualization, all_defects
    
    def create_visualization(self, original, defects, kernel_states=None):
        """Create visualization with detected defects highlighted"""
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
                
                # Draw kernel box (red if no line, green if line found)
                color = (0, 255, 0) if state['has_line'] else (0, 0, 255)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
                
                # Draw centroid as red dot
                cv2.circle(overlay, (x, y), 2, (0, 0, 255), -1)
            
            # In debug mode, blend lightly to see kernels clearly
            result = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
            return result
        
        # Draw defects (only when not in debug mode)
        for defect in defects:
            if defect['type'] == 'missing_line':
                x1 = defect['start_x']
                x2 = defect['end_x']
                y = defect['y']
                
                # Draw rectangle for missing line
                cv2.rectangle(overlay, (x1, y - 15), (x2, y + 15), (0, 0, 255), -1)
        
        # Blend with original
        result = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
        
        return result 