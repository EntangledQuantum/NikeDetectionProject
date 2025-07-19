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
        print("Sensitivity: ", sensitivity)
        
        # Adjust parameters based on sensitivity
        if sensitivity == 'high':
            self.kernel_size = 15
            self.search_range = 15
            self.min_gap_size = 5
            self.step_size = 15
            self.line_threshold = 0.25  # Require 25% pixels for a valid line (aggressive)
            self.strong_line_threshold = 0.40  # Require 40% pixels to change Y position
            self.max_y_drift = 8  # Maximum Y drift per step
            self.stability_weight = 0.7  # Weight for previous position (higher = more stable)
        elif sensitivity == 'low':
            self.kernel_size = 40
            self.search_range = 10  
            self.min_gap_size = 30
            self.step_size = 25
            self.line_threshold = 0.25  # Only 5% pixels needed (lenient)
            self.strong_line_threshold = 0.25  # Require 35% pixels to change Y position
            self.max_y_drift = 12  # Maximum Y drift per step
            self.stability_weight = 0.6  # Weight for previous position
        else:  # medium
            self.line_threshold = 0.10  # 10% pixels needed (balanced)
            self.strong_line_threshold = 0.20  # Require 20% pixels to change Y position
            self.max_y_drift = 10  # Maximum Y drift per step
            self.stability_weight = 0.65  # Weight for previous position
    
    def scan_for_lines(self, binary_image):
        """Scan image to find all horizontal lines"""
        height, width = binary_image.shape
        line_starts = []
        
        y = self.kernel_size // 2
        while y < height - self.kernel_size // 2:
            # Simply add this Y position as a line to scan
            line_starts.append(y)
            
            if self.debug:
                print(f"Will scan line at Y={y}")
            
            # Move down by kernel size to next scan line
            y += self.kernel_size
        
        if self.debug:
            print(f"Total scan lines: {len(line_starts)}")
            print(f"Line detection threshold: {self.line_threshold * 100:.1f}% of pixels required")
        
        return line_starts
    
    def track_line(self, binary_image, start_y):
        """Track a single line across the image using kernel"""
        height, width = binary_image.shape
        kernel_states = []  # For debug visualization
        defects = []
        
        # Starting position - start from the very left edge
        x = 0  # Start from left edge instead of kernel_size // 2
        y = start_y
        gap_start = None
        previous_y = y  # Keep track of last known good Y position
        consecutive_missing = 0  # Count consecutive missing kernels
        max_consecutive_missing = 30  # Hardcoded limit
        ever_found_line = False  # Track if we ever found a line
        
        while x < width - self.kernel_size // 2:
            # Extract kernel region
            y1 = max(0, y - self.kernel_size // 2)
            y2 = min(height, y + self.kernel_size // 2)
            x1 = max(0, x - self.kernel_size // 2)
            x2 = min(width, x + self.kernel_size // 2)
            
            kernel_region = binary_image[y1:y2, x1:x2]
            
            # Check if there's line in kernel - use sensitivity-based threshold
            white_pixels = np.sum(kernel_region > 0)
            total_pixels = (y2 - y1) * (x2 - x1)
            has_line = white_pixels > total_pixels * self.line_threshold  # Use sensitivity-based threshold
            
            if has_line:
                # We found a line!
                ever_found_line = True
                consecutive_missing = 0  # Reset counter
                
                # Calculate centroid of line pixels in kernel to follow the line
                y_indices, x_indices = np.where(kernel_region > 0)
                if len(y_indices) > 0:
                    # Calculate new Y position from line centroid
                    local_y_center = np.mean(y_indices)
                    new_y = y1 + int(local_y_center)
                    
                    # Check if we have enough strong pixels to justify changing Y position
                    strong_pixels = white_pixels / total_pixels
                    if strong_pixels >= self.strong_line_threshold:
                        # Strong line signal - allow Y position change but limit drift
                        y_drift = abs(new_y - previous_y)
                        if y_drift <= self.max_y_drift:
                            # Apply weighted average for stability
                            y = int(self.stability_weight * previous_y + (1 - self.stability_weight) * new_y)
                        else:
                            # Too much drift - stay at previous position
                            y = previous_y
                            if self.debug:
                                print(f"Prevented large Y drift: {y_drift} pixels at X={x}")
                    else:
                        # Weak line signal - stay at previous Y position to avoid drift
                        y = previous_y
                        if self.debug:
                            print(f"Weak line signal ({strong_pixels:.2%}) - maintaining Y position at X={x}")
                    
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
                        
                        if white_pixels > (self.kernel_size * self.kernel_size) * self.line_threshold:
                            # Found line at different Y
                            y = test_y
                            found = True
                            ever_found_line = True
                            previous_y = y
                            consecutive_missing = 0  # Reset counter
                            break
                
                if not found:
                    # Increment consecutive missing counter
                    consecutive_missing += 1
                    
                    # Check if we've exceeded the limit AND never found a line
                    if consecutive_missing >= max_consecutive_missing and not ever_found_line:
                        # This was a false detection - no line exists here
                        if self.debug:
                            print(f"False line detection at Y={start_y} - no line found in first {consecutive_missing} kernels")
                        return [], []  # Return empty results
                    
                    # Only record states and gaps if we've found a line before
                    if ever_found_line:
                        # Mark start of gap if not already in one
                        if gap_start is None:
                            gap_start = x
                        # Keep Y at previous position for red box
                        y = previous_y
                        
                        # Record kernel state for missing line (red box)
                        kernel_states.append({
                            'x': x,
                            'y': y,
                            'has_line': False,
                            'bbox': (x1, y - self.kernel_size // 2, x2, y + self.kernel_size // 2)
                        })
                else:
                    # Found line after searching - record green box
                    kernel_states.append({
                        'x': x,
                        'y': y,
                        'has_line': True,
                        'bbox': (x1, y - self.kernel_size // 2, x2, y + self.kernel_size // 2)
                    })
            
            # Move to next position horizontally by step size to avoid overlap
            x += self.step_size
        
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
        
        # Draw defects when not in debug mode - only show missing line areas
        for defect in defects:
            if defect['type'] == 'missing_line':
                x1 = defect['start_x']
                x2 = defect['end_x']
                y = defect['y']
                
                # Draw filled red rectangle overlay for missing line segment
                # Make it thick enough to be clearly visible
                thickness = 20  # Thickness of the missing line indicator
                cv2.rectangle(overlay, 
                            (x1, y - thickness), 
                            (x2, y + thickness), 
                            (0, 0, 255),  # Red color
                            -1)  # Filled rectangle
        
        # Blend with original to create overlay effect
        result = cv2.addWeighted(vis, 0.5, overlay, 0.5, 0)
        
        return result 