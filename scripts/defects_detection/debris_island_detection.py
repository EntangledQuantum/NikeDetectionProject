"""
Debris Detection Algorithm for Island Images
Detects horizontal slanted lines and prepares for debris detection
Specifically designed for island-type images

Author: Assistant
Date: 2024
"""

import cv2
import numpy as np
from detector_base import BaseDetector


class DebrisIslandDetector(BaseDetector):
    """Detects lines in island images to prepare for debris detection"""
    
    def __init__(self, first_line_x=0, first_line_y=100, delta_y=50, 
                 search_step=5, max_batch_checking=5, line_thickness=3,
                 sensitivity='medium', debug=False, invert_y_axis=False):
        """
        Args:
            first_line_x: X coordinate of the first line start point
            first_line_y: Y coordinate of the first line start point
            delta_y: Vertical distance between lines
            search_step: Step size when searching vertically for line end
            max_batch_checking: Number of lines to check for verification
            line_thickness: Thickness of visualized lines
            sensitivity: Detection sensitivity level
            debug: Whether to enable debug visualization
            invert_y_axis: If True, assumes Y increases upward (mathematical convention)
                          and converts to image coordinates (Y increases downward)
        """
        # Store original values for reference
        self.invert_y_axis = invert_y_axis
        self.original_first_line_x = first_line_x
        self.original_first_line_y = first_line_y
        self.original_delta_y = delta_y
        
        # Convert to integers and handle coordinate system
        self.first_line_x = int(round(first_line_x))
        
        # Handle Y coordinate based on coordinate system
        if invert_y_axis:
            # In inverted system, we'll need to transform when we know image height
            # For now, store the original value
            self.first_line_y_needs_transform = True
            self.first_line_y = int(round(abs(first_line_y)))  # Temporary value
        else:
            self.first_line_y_needs_transform = False
            # If Y is negative in normal image coords, use absolute value
            if first_line_y < 0:
                self.first_line_y = int(round(abs(first_line_y)))
            else:
                self.first_line_y = int(round(first_line_y))
        
        self.delta_y = int(round(abs(delta_y)))  # Ensure delta_y is positive integer
        self.search_step = search_step
        self.max_batch_checking = max_batch_checking
        self.line_thickness = line_thickness
        self.debug = debug
        
        print(f"Sensitivity: {sensitivity}")
        print(f"Coordinates after conversion - X: {self.first_line_x}, Y: {self.first_line_y}, Delta Y: {self.delta_y}")
        
        # Adjust parameters based on sensitivity
        if sensitivity == 'high':
            self.search_step = 2
            self.verification_threshold = 0.4  # 40% of batch checks need to pass
        elif sensitivity == 'low':
            self.search_step = 10
            self.verification_threshold = 0.7  # 70% of batch checks need to pass
        else:  # medium
            self.search_step = 5
            self.verification_threshold = 0.5  # 50% of batch checks need to pass
    
    def find_line_end(self, binary_image):
        """Find the end point of the first line starting from top-right"""
        height, width = binary_image.shape
        
        # Start from top-right corner
        start_x = width - 1
        start_y = self.first_line_y
        
        # Search vertically downward
        y = start_y
        line_end_found = False
        line_end_x = start_x
        line_end_y = start_y
        
        while y < height - self.delta_y * self.max_batch_checking:
            # Check if there's a line pixel at this position
            y_int = int(y)
            x_int = int(start_x)
            if 0 <= y_int < height and 0 <= x_int < width and binary_image[y_int, x_int] > 0:
                # Potential line found, verify with batch checking
                if self.verify_line_position(binary_image, x_int, y_int):
                    line_end_found = True
                    line_end_x = x_int
                    line_end_y = y_int
                    break
            
            y += self.search_step
        
        if not line_end_found:
            # If no line found at edge, search inward
            for x_offset in range(0, width // 4, 10):
                x = start_x - x_offset
                if x < 0:
                    break
                
                y = start_y
                while y < height - self.delta_y * self.max_batch_checking:
                    y_int = int(y)
                    x_int = int(x)
                    if 0 <= y_int < height and 0 <= x_int < width and binary_image[y_int, x_int] > 0:
                        if self.verify_line_position(binary_image, x_int, y_int):
                            line_end_found = True
                            line_end_x = x_int
                            line_end_y = y_int
                            break
                    y += self.search_step
                
                if line_end_found:
                    break
        
        if self.debug:
            print(f"Line end found: {line_end_found} at ({line_end_x}, {line_end_y})")
        
        return line_end_found, line_end_x, line_end_y
    
    def verify_line_position(self, binary_image, x, y):
        """Verify if the detected position is actually on a line by checking multiple positions below"""
        height, width = binary_image.shape
        lines_found = 0
        
        # Check multiple positions below
        for i in range(1, self.max_batch_checking + 1):
            check_y = int(y + i * self.delta_y)
            
            if check_y >= height:
                break
            
            # Check in a small window around the expected position
            window_size = 10
            y_start = max(0, int(check_y - window_size))
            y_end = min(height, int(check_y + window_size))
            x_start = max(0, int(x - window_size // 2))
            x_end = min(width, int(x + window_size // 2))
            
            # Check if there's a line in this window
            window_region = binary_image[y_start:y_end, x_start:x_end]
            if np.sum(window_region > 0) > 0:
                lines_found += 1
        
        # Return true if more than threshold of checks passed
        return lines_found >= (self.max_batch_checking * self.verification_threshold)
    
    def calculate_line_slope(self, x1, y1, x2, y2):
        """Calculate the slope of the line"""
        if x2 - x1 == 0:
            return float('inf')
        return (y2 - y1) / (x2 - x1)
    
    def draw_lines(self, image, slope):
        """Draw all detected lines based on the calculated slope"""
        height, width = image.shape[:2] if len(image.shape) > 2 else image.shape
        
        # Create visualization image
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
        
        # Draw lines starting from the first line
        line_count = 0
        y = self.first_line_y
        
        while y < height:
            # Calculate line endpoints
            x1 = self.first_line_x
            y1 = y
            
            # Calculate x2 based on slope
            if slope != float('inf'):
                x2 = width - 1
                y2 = int(y1 + slope * (x2 - x1))
            else:
                x2 = x1
                y2 = height - 1
            
            # Ensure y2 is within bounds
            if y2 >= height:
                # Adjust x2 to keep y2 within bounds
                if slope != 0 and slope != float('inf'):
                    y2 = height - 1
                    x2 = int(x1 + (y2 - y1) / slope)
            
            # Draw the line
            if 0 <= x2 < width and 0 <= y2 < height:
                cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), 
                        (0, 255, 0), self.line_thickness)
                line_count += 1
            
            # Move to next line position
            y += self.delta_y
        
        if self.debug:
            print(f"Drew {line_count} lines with slope {slope:.4f}")
        
        return vis, line_count
    
    def detect(self, image):
        """Main detection method"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold to get binary image (lines should be dark/black)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find the end of the first line
        line_found, line_end_x, line_end_y = self.find_line_end(binary)
        
        defects = []
        
        if line_found:
            # Calculate slope
            slope = self.calculate_line_slope(self.first_line_x, self.first_line_y, 
                                            line_end_x, line_end_y)
            
            # Draw all lines
            visualization, line_count = self.draw_lines(image, slope)
            
            # For now, just report the detected lines
            # Later this will be extended to detect debris by subtracting lines
            defects.append({
                'type': 'lines_detected',
                'line_count': line_count,
                'slope': float(slope) if slope != float('inf') else 999999,
                'first_line_start': (self.first_line_x, self.first_line_y),
                'first_line_end': (line_end_x, line_end_y),
                'delta_y': self.delta_y
            })
            
            if self.debug:
                # Add debug information to visualization
                cv2.putText(visualization, f"Lines: {line_count}, Slope: {slope:.4f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.circle(visualization, (self.first_line_x, self.first_line_y), 
                          10, (0, 0, 255), -1)  # Red dot at start
                cv2.circle(visualization, (line_end_x, line_end_y), 
                          10, (255, 0, 0), -1)  # Blue dot at end
        else:
            # No lines detected
            if len(image.shape) == 2:
                visualization = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                visualization = image.copy()
            
            cv2.putText(visualization, "No lines detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return visualization, defects