"""
Line Defect Detection Algorithm using LineDetector
Detects missing lines and jagged/zig-zag lines in island images
Uses the robust LineDetector to find where lines are, then scans along each line

Author: Assistant
Date: 2024
"""

import cv2
import numpy as np
import os
from utils.line_detector import LineDetector


class LineDefectDetector:
    """Detect line defects (missing and jagged segments) using LineDetector.
    
    The detector:
    1. Uses LineDetector to find where horizontal lines are
    2. For each detected line, scans along it with kernels
    3. Detects missing segments (no line found in kernel)
    4. Detects jagged segments (sharp Y position changes)
    """
    
    def __init__(self, sensitivity='medium', debug=False):
        """Initialize the detector with sensitivity settings.

        Args:
            sensitivity: One of {'low', 'medium', 'high'}.
            debug: If True, store additional visualization details.
        """
        # test debug
        self.sensitivity = sensitivity
        self.debug = True
        
        print(f"Line Defect Detection Sensitivity: {sensitivity}")
        
        # Initialize line detector to find where lines are
        self.line_detector = LineDetector(sensitivity)
        
        # Set parameters based on sensitivity
        if sensitivity == 'high':
            self.kernel_size = 25
            self.step_size = 20  # Horizontal step between kernel checks
            self.line_threshold = 0.18  # Require 18% pixels for a valid line
            self.min_gap_size = 80  # Minimum gap size to report missing line
            self.jagged_threshold = 3  # Y delta threshold for jagged detection
        elif sensitivity == 'low':
            self.kernel_size = 35
            self.step_size = 30
            self.line_threshold = 0.22
            self.min_gap_size = 120
            self.jagged_threshold = 7
        else:  # medium (default)
            self.kernel_size = 30
            self.step_size = 25
            self.line_threshold = 0.20
            self.min_gap_size = 100
            self.jagged_threshold = 5
        
        # Store debug images
        self._debug_missing_lines_image = None
        self._debug_jagged_lines_image = None
        self._debug_kernel_image = None
        self._debug_line_detector_kernels = None
        
        print(f"Line Defect Detector Configuration:")
        print(f"  Kernel size: {self.kernel_size}px")
        print(f"  Step size: {self.step_size}px")
        print(f"  Line threshold: {self.line_threshold * 100:.1f}% pixels required")
        print(f"  Min gap size: {self.min_gap_size}px")
        print(f"  Jagged threshold: {self.jagged_threshold}px Y-delta")
    
    def scan_line_for_defects(self, binary_image, line_match):
        """Scan along a detected line to find missing and jagged segments.
        
        Args:
            binary_image: Binary image where lines are white
            line_match: Matched line dict with 'left' and 'right' points
            
        Returns:
            tuple: (defects, kernel_states)
                - defects: List of defect dicts (missing_line, jagged_line)
                - kernel_states: List of kernel placement states for visualization
        """
        left_pt = line_match['left']
        right_pt = line_match['right']
        
        height, width = binary_image.shape
        defects = []
        kernel_states = []
        
        # Calculate expected line trajectory (simple linear interpolation)
        x_start = left_pt['x']
        x_end = right_pt['x']
        y_start = left_pt['y']
        y_end = right_pt['y']
        
        if x_end <= x_start:
            return defects, kernel_states
        
        # Calculate slope
        line_length = x_end - x_start
        slope = (y_end - y_start) / line_length if line_length > 0 else 0
        
        # Scan horizontally along the line
        x = x_start
        last_found_y = y_start
        last_detection_x = x_start  # Track X position of last actual detection
        gap_start = None
        
        while x < x_end:
            # Calculate expected Y position at this X
            expected_y = int(y_start + slope * (x - x_start))
            expected_y = max(self.kernel_size // 2, min(height - self.kernel_size // 2, expected_y))
            
            # Extract kernel region centered at expected position
            y1 = max(0, expected_y - self.kernel_size // 2)
            y2 = min(height, expected_y + self.kernel_size // 2)
            x1 = max(0, x - self.kernel_size // 2)
            x2 = min(width, x + self.kernel_size // 2)
            
            kernel_region = binary_image[y1:y2, x1:x2]
            
            # Check if there's a line in the kernel
            white_pixels = np.sum(kernel_region > 0)
            total_pixels = (y2 - y1) * (x2 - x1)
            
            if total_pixels > 0:
                pixel_ratio = white_pixels / total_pixels
                has_line = pixel_ratio > self.line_threshold
                
                if has_line:
                    # Line found - calculate actual Y position
                    y_indices, x_indices = np.where(kernel_region > 0)
                    if len(y_indices) > 0:
                        local_y_center = np.mean(y_indices)
                        actual_y = y1 + int(local_y_center)
                        
                        # Check for jagged line (sharp Y change from PREVIOUS detection, not expected)
                        # This ignores small gaps and compares actual detections
                        y_delta = abs(actual_y - last_found_y)
                        is_jagged = y_delta > self.jagged_threshold
                        
                        if is_jagged:
                            if self.debug:
                                print(f"    JAGGED at x={x}: Y jumped from {last_found_y} to {actual_y} (delta={y_delta}px > {self.jagged_threshold}px)")
                            
                            defects.append({
                                'type': 'jagged_line',
                                'x': x,
                                'y': actual_y,
                                'previous_y': last_found_y,
                                'location': (x, actual_y),
                                'y_delta': int(y_delta),
                                'threshold': self.jagged_threshold
                            })
                        
                        # Record kernel state
                        kernel_states.append({
                            'x': x,
                            'y': actual_y,
                            'expected_y': expected_y,
                            'has_line': True,
                            'is_jagged': is_jagged,
                            'bbox': (x1, y1, x2, y2),
                            'pixel_ratio': pixel_ratio
                        })
                        
                        # If we were in a gap, close it
                        if gap_start is not None:
                            gap_size = x - gap_start
                            if gap_size >= self.min_gap_size:
                                if self.debug:
                                    print(f"    MISSING from x={gap_start} to x={x} (gap={gap_size}px >= {self.min_gap_size}px)")
                                
                                defects.append({
                                    'type': 'missing_line',
                                    'start_x': gap_start,
                                    'end_x': x,
                                    'y': expected_y,
                                    'location': ((gap_start + x) // 2, expected_y),
                                    'size': gap_size
                                })
                            else:
                                if self.debug:
                                    print(f"    Small gap {gap_size}px (< {self.min_gap_size}px) - ignored for missing, but Y-delta checked for jagged")
                            gap_start = None
                        
                        # Update last detection position (crucial for next jagged check)
                        last_found_y = actual_y
                        last_detection_x = x
                else:
                    # No line found - mark as missing
                    if gap_start is None:
                        gap_start = x
                    
                    kernel_states.append({
                        'x': x,
                        'y': expected_y,
                        'expected_y': expected_y,
                        'has_line': False,
                        'is_jagged': False,
                        'bbox': (x1, y1, x2, y2),
                        'pixel_ratio': pixel_ratio
                    })
            
            # Move to next position
            x += self.step_size
        
        # Close any remaining gap at the end
        if gap_start is not None:
            gap_size = x_end - gap_start
            if gap_size >= self.min_gap_size:
                defects.append({
                    'type': 'missing_line',
                    'start_x': gap_start,
                    'end_x': x_end,
                    'y': y_end,
                    'location': ((gap_start + x_end) // 2, y_end),
                    'size': gap_size
                })
        
        return defects, kernel_states
    
    def detect(self, image, image_path=None):
        """Run the full line defect detection pipeline on an image.

        Steps:
          1) Use LineDetector to find where lines are
          2) For each line, scan along it to find defects
          3) Aggregate defects and produce visualizations

        Args:
            image: Input image as BGR or grayscale.
            image_path: Optional path for loading exclusion zones.

        Returns:
            tuple: (visualization_bgr, defects)
        """
        if self.debug:
            print(f"\n=== LINE DEFECT DETECTION ===")
            print(f"Min gap size: {self.min_gap_size}px")
            print(f"Jagged threshold: {self.jagged_threshold}px")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Use LineDetector to find where lines are
        matched_lines, left_lines, right_lines, left_kernels, right_kernels = \
            self.line_detector.detect_lines(image, self.debug, image_path)
        
        if self.debug:
            print(f"\nLineDetector found {len(matched_lines)} lines")
        
        if not matched_lines:
            # No lines detected
            if len(image.shape) == 2:
                vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                vis = image.copy()
            cv2.putText(vis, "No lines detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return vis, []
        
        # Apply threshold to get binary image (lines should be dark/black)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Scan each detected line for defects
        all_defects = []
        all_kernel_states = []
        
        for i, line_match in enumerate(matched_lines):
            if self.debug:
                left = line_match['left']
                right = line_match['right']
                print(f"\nScanning line {i+1}: ({left['x']}, {left['y']}) -> ({right['x']}, {right['y']})")
            
            defects, kernel_states = self.scan_line_for_defects(binary, line_match)
            
            if self.debug:
                missing = [d for d in defects if d['type'] == 'missing_line']
                jagged = [d for d in defects if d['type'] == 'jagged_line']
                print(f"  Found {len(missing)} missing segments, {len(jagged)} jagged segments")
            
            all_defects.extend(defects)
            all_kernel_states.extend(kernel_states)
        
        if self.debug:
            missing_total = len([d for d in all_defects if d['type'] == 'missing_line'])
            jagged_total = len([d for d in all_defects if d['type'] == 'jagged_line'])
            print(f"\n=== TOTAL DEFECTS ===")
            print(f"Missing line segments: {missing_total}")
            print(f"Jagged line segments: {jagged_total}")
        
        # Always create combined visualization (red + yellow defects)
        combined_vis = self.create_combined_visualization(image, all_defects, all_kernel_states)
        
        if self.debug:
            # In debug mode, create additional visualizations:
            # 1. Separate missing lines only (red)
            self._debug_missing_lines_image = self.create_missing_lines_visualization(image, all_defects)
            # 2. Separate jagged lines only (yellow)
            self._debug_jagged_lines_image = self.create_jagged_lines_visualization(image, all_defects)
            # 3. Kernel visualization showing all kernel boxes (line defect detector)
            self._debug_kernel_image = self.create_kernel_visualization(image, all_kernel_states, all_defects)
            # 4. LineDetector kernel visualization showing how lines were initially detected
            self._debug_line_detector_kernels = self.create_line_detector_kernel_visualization(
                image, left_kernels, right_kernels, matched_lines)
        
        # Always return combined visualization (works for both debug and normal mode)
        return combined_vis, all_defects
    
    def create_combined_visualization(self, original, defects, kernel_states=None):
        """Create visualization showing both missing and jagged defects.
        
        Args:
            original: Original input image
            defects: List of defect dicts
            kernel_states: Optional kernel states for debug visualization
            
        Returns:
            BGR visualization image
        """
        if len(original.shape) == 2:
            vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            vis = original.copy()
        
        overlay = vis.copy()
        
        # Draw missing line segments in RED
        for defect in defects:
            if defect['type'] == 'missing_line':
                x1 = defect['start_x']
                x2 = defect['end_x']
                y = defect['y']
                thickness = 20
                
                cv2.rectangle(overlay,
                            (x1, y - thickness),
                            (x2, y + thickness),
                            (0, 0, 255),  # RED
                            -1)
        
        # Draw jagged line segments in YELLOW
        for defect in defects:
            if defect['type'] == 'jagged_line':
                x = defect['x']
                y = defect['y']
                thickness = 15
                
                cv2.rectangle(overlay,
                            (x - self.kernel_size//2, y - thickness),
                            (x + self.kernel_size//2, y + thickness),
                            (0, 255, 255),  # YELLOW
                            -1)
        
        # Blend with original
        result = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        
        # Add summary text
        missing_count = len([d for d in defects if d['type'] == 'missing_line'])
        jagged_count = len([d for d in defects if d['type'] == 'jagged_line'])
        
        text = f"Missing: {missing_count} | Jagged: {jagged_count}"
        cv2.putText(result, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return result
    
    def create_missing_lines_visualization(self, original, defects):
        """Create visualization showing only missing line segments in RED."""
        if len(original.shape) == 2:
            vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            vis = original.copy()
        
        overlay = vis.copy()
        
        for defect in defects:
            if defect['type'] == 'missing_line':
                x1 = defect['start_x']
                x2 = defect['end_x']
                y = defect['y']
                thickness = 20
                
                cv2.rectangle(overlay,
                            (x1, y - thickness),
                            (x2, y + thickness),
                            (0, 0, 255),  # RED
                            -1)
        
        result = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        
        missing_count = len([d for d in defects if d['type'] == 'missing_line'])
        text = f"Missing Line Segments: {missing_count}"
        cv2.putText(result, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return result
    
    def create_jagged_lines_visualization(self, original, defects):
        """Create visualization showing only jagged line segments in YELLOW."""
        if len(original.shape) == 2:
            vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            vis = original.copy()
        
        overlay = vis.copy()
        
        for defect in defects:
            if defect['type'] == 'jagged_line':
                x = defect['x']
                y = defect['y']
                thickness = 15
                
                cv2.rectangle(overlay,
                            (x - self.kernel_size//2, y - thickness),
                            (x + self.kernel_size//2, y + thickness),
                            (0, 255, 255),  # YELLOW
                            -1)
        
        result = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        
        jagged_count = len([d for d in defects if d['type'] == 'jagged_line'])
        text = f"Jagged Line Segments: {jagged_count}"
        cv2.putText(result, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return result
    
    def create_kernel_visualization(self, original, kernel_states, defects):
        """Create visualization showing all kernel boxes with color coding.
        
        Args:
            original: Original input image
            kernel_states: List of kernel state dicts
            defects: List of defects for context
            
        Returns:
            BGR visualization with kernel boxes overlaid
        """
        if len(original.shape) == 2:
            vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            vis = original.copy()
        
        overlay = vis.copy()
        
        # Draw each kernel box with appropriate color
        for state in kernel_states:
            x1, y1, x2, y2 = state['bbox']
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(vis.shape[1], x2)
            y2 = min(vis.shape[0], y2)
            
            # Choose color based on kernel state
            if state.get('is_jagged', False):
                color = (0, 255, 255)  # Yellow for jagged lines
                thickness = 2
            elif state['has_line']:
                color = (0, 255, 0)  # Green for normal lines
                thickness = 1
            else:
                color = (0, 0, 255)  # Red for missing lines
                thickness = 2
            
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
            
            # Draw center dot
            x = state['x']
            y = state['y']
            cv2.circle(overlay, (x, y), 3, color, -1)
        
        # Blend with original
        result = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        
        # Add legend
        cv2.putText(result, "Green: Line OK", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result, "Red: Missing", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(result, "Yellow: Jagged", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return result
    
    def create_line_detector_kernel_visualization(self, original, left_kernels, right_kernels, matched_lines):
        """Create visualization showing LineDetector's kernel scanning results.
        
        Args:
            original: Original input image
            left_kernels: List of kernel state dicts from left side scan
            right_kernels: List of kernel state dicts from right side scan
            matched_lines: Matched line pairs from LineDetector
            
        Returns:
            BGR visualization with LineDetector kernels and matched lines
        """
        if len(original.shape) == 2:
            vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            vis = original.copy()
        
        overlay = vis.copy()
        
        # Draw matched lines first
        for i, match in enumerate(matched_lines):
            left_pt = match['left']
            right_pt = match['right']
            valid_slope = match.get('valid_slope', True)
            left_type = match.get('left_type', 'real')
            right_type = match.get('right_type', 'real')
            
            # Choose color based on validity and line types
            if not valid_slope:
                line_color = (0, 0, 255)  # Red for invalid slope
                line_thickness = 3
            elif left_type == 'ghost' or right_type == 'ghost':
                line_color = (255, 165, 0)  # Orange for ghost lines
                line_thickness = 2
            else:
                line_color = (0, 255, 0)  # Green for valid real lines
                line_thickness = 3
            
            # Draw the line
            cv2.line(overlay, (left_pt['x'], left_pt['y']), 
                    (right_pt['x'], right_pt['y']), 
                    line_color, line_thickness)
        
        # Draw left side kernels
        for state in left_kernels:
            x1, y1, x2, y2 = state['bbox']
            color = (0, 255, 255) if state['has_line'] else (128, 128, 128)  # Cyan if line, gray if not
            thickness = 2 if state['has_line'] else 1
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        
        # Draw right side kernels
        for state in right_kernels:
            x1, y1, x2, y2 = state['bbox']
            color = (255, 255, 0) if state['has_line'] else (128, 128, 128)  # Yellow if line, gray if not
            thickness = 2 if state['has_line'] else 1
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        
        # Blend with original
        result = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
        
        # Add legend
        cv2.putText(result, "LineDetector Kernel Scan", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(result, "Cyan: Left kernels with lines", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(result, "Yellow: Right kernels with lines", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(result, "Green: Valid matched lines", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result, "Orange: Ghost lines", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        cv2.putText(result, "Red: Invalid slopes", (10, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return result
    
    def save_debug_images(self, output_dir, base_name):
        """Save debug images if debug mode is enabled."""
        debug_paths = []
        
        if self.debug:
            if self._debug_missing_lines_image is not None:
                path = os.path.join(output_dir, f"{base_name}_line_defect_missing.jpg")
                cv2.imwrite(path, self._debug_missing_lines_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                debug_paths.append(path)
                print(f"    Saved missing lines debug: {path}")
            
            if self._debug_jagged_lines_image is not None:
                path = os.path.join(output_dir, f"{base_name}_line_defect_jagged.jpg")
                cv2.imwrite(path, self._debug_jagged_lines_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                debug_paths.append(path)
                print(f"    Saved jagged lines debug: {path}")
            
            if hasattr(self, '_debug_kernel_image') and self._debug_kernel_image is not None:
                path = os.path.join(output_dir, f"{base_name}_line_defect_kernels.jpg")
                cv2.imwrite(path, self._debug_kernel_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                debug_paths.append(path)
                print(f"    Saved line defect kernel visualization: {path}")
            
            if hasattr(self, '_debug_line_detector_kernels') and self._debug_line_detector_kernels is not None:
                path = os.path.join(output_dir, f"{base_name}_line_detector_kernels.jpg")
                cv2.imwrite(path, self._debug_line_detector_kernels, [cv2.IMWRITE_JPEG_QUALITY, 95])
                debug_paths.append(path)
                print(f"    Saved LineDetector kernel visualization: {path}")
        
        return debug_paths if debug_paths else None