"""
Stripe Misalignment Detection Algorithm
Detects misalignment of vertical stripes caused by printer head misalignment

Author: Assistant
Date: 2024
"""

import cv2
import numpy as np
import os
import sys

# Add the utils directory to path to import edge_detector
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from edge_detector import detect_edges_enhanced


class StripeMisalignmentDetector:
    """Detects misalignment in vertical stripe patterns using grid-based kernel scanning"""
    
    def __init__(self, kernel_size=50, step_size=None, line_detection_threshold=0.15,
                 defect_threshold=10, sensitivity='medium', debug=False):
        """
        Args:
            kernel_size: Size of the scanning kernel (square)
            step_size: Horizontal step size (defaults to kernel_size for no overlap)
            line_detection_threshold: Minimum ratio of pixels to classify as line
            defect_threshold: Minimum x position delta to consider as defect
            sensitivity: Detection sensitivity level
            debug: Whether to draw debug visualization
        """
        self.kernel_size = kernel_size
        self.step_size = step_size if step_size else kernel_size
        self.line_detection_threshold = line_detection_threshold
        self.defect_threshold = defect_threshold
        self.debug = debug
        
        # print sensitivity
        print(f"Sensitivity: {sensitivity}")
        
        # Adjust parameters based on sensitivity
        if sensitivity == 'high':
            self.kernel_size = 30
            self.step_size = 30
            self.line_detection_threshold = 0.10  # More sensitive to line detection
            self.defect_threshold = 5  # Smaller misalignment considered defect
        elif sensitivity == 'low':
            self.kernel_size = 70
            self.step_size = 70
            self.line_detection_threshold = 0.20  # Less sensitive to line detection
            self.defect_threshold = 20  # Larger misalignment needed for defect
        else:  # medium
            self.kernel_size = 50
            self.step_size = 30
            self.line_detection_threshold = 0.15
            self.defect_threshold = 10
    
    def preprocess_with_edge_detection(self, image):
        """Apply edge detection preprocessing to enhance vertical edges"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply edge detection with parameters from edge_detector.py
        # Using the exact parameters from the edge_detector.py main function
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # Apply median filter
        processed = cv2.medianBlur(blurred, 51)
        
        # Apply Sobel for vertical edges (we're looking for vertical lines)
        sobel_x = cv2.Sobel(processed, cv2.CV_64F, 1, 0, ksize=5)
        vertical_edges = np.absolute(sobel_x)
        
        # Normalize to 0-255
        edges_normalized = np.uint8(vertical_edges / vertical_edges.max() * 255)
        
        # Apply threshold to remove weak edges
        _, edges_normalized = cv2.threshold(edges_normalized, 30, 255, cv2.THRESH_BINARY)
        
        return edges_normalized
    
    def scan_grid(self, binary_image):
        """Scan image in grid pattern to detect vertical line and misalignments"""
        height, width = binary_image.shape
        kernel_states = []
        defects = []
        
        # Track line detection state
        line_detected = False
        previous_x_pos = None
        
        # Start from top of image
        y = self.kernel_size // 2
        
        while y < height - self.kernel_size // 2:
            row_results = self.scan_row(binary_image, y)
            
            if row_results:
                # Process results for this row
                for result in row_results:
                    x_pos = result['x_center']
                    
                    # If this is the first line detection, just record it
                    if not line_detected:
                        line_detected = True
                        previous_x_pos = x_pos
                        result['is_defect'] = False
                        if self.debug:
                            print(f"First line detected at Y={y}, X={x_pos}")
                    else:
                        # Calculate delta from previous position
                        x_delta = abs(x_pos - previous_x_pos)
                        result['x_delta'] = x_delta
                        
                        # Check if this is a defect
                        if x_delta > self.defect_threshold:
                            result['is_defect'] = True
                            defects.append({
                                'type': 'stripe_misalignment',
                                'y': y,
                                'x': x_pos,
                                'x_delta': int(x_delta),
                                'previous_x': previous_x_pos,
                                'location': (x_pos, y),
                                'threshold': self.defect_threshold
                            })
                            if self.debug:
                                print(f"Misalignment detected at Y={y}: X delta={x_delta} > threshold={self.defect_threshold}")
                        else:
                            result['is_defect'] = False
                        
                        # Update previous position
                        previous_x_pos = x_pos
                    
                    kernel_states.extend(result['kernels'])
            
            # Move to next row
            y += self.kernel_size
        
        return kernel_states, defects
    
    def scan_row(self, binary_image, y):
        """Scan a single row to find vertical line"""
        height, width = binary_image.shape
        row_results = []
        
        # Extract row region
        y1 = max(0, y - self.kernel_size // 2)
        y2 = min(height, y + self.kernel_size // 2)
        
        # Scan horizontally
        x = self.kernel_size // 2
        line_found_in_row = False
        line_start_x = None
        line_end_x = None
        kernels_in_row = []
        
        while x < width - self.kernel_size // 2:
            # Extract kernel region
            x1 = max(0, x - self.kernel_size // 2)
            x2 = min(width, x + self.kernel_size // 2)
            
            kernel_region = binary_image[y1:y2, x1:x2]
            
            # Check if there's line in kernel
            white_pixels = np.sum(kernel_region > 0)
            total_pixels = (y2 - y1) * (x2 - x1)
            has_line = white_pixels > total_pixels * self.line_detection_threshold
            
            if has_line:
                if not line_found_in_row:
                    line_found_in_row = True
                    line_start_x = x
                line_end_x = x
                
                # Record kernel state for debug
                if self.debug:
                    kernels_in_row.append({
                        'x': x,
                        'y': y,
                        'has_line': True,
                        'bbox': (x1, y1, x2, y2)
                    })
            
            # Move to next position
            x += self.step_size
        
        # If line was found in this row, calculate center position
        if line_found_in_row:
            x_center = (line_start_x + line_end_x) // 2
            row_results.append({
                'y': y,
                'x_center': x_center,
                'line_start': line_start_x,
                'line_end': line_end_x,
                'kernels': kernels_in_row
            })
        
        return row_results
    
    def detect(self, image):
        """Main detection method"""
        # Preprocess with edge detection
        edge_image = self.preprocess_with_edge_detection(image)
        
        # Store edge image for debug output
        self.edge_image = edge_image
        
        # Scan grid for misalignments
        kernel_states, defects = self.scan_grid(edge_image)
        
        # Create visualization
        visualization = self.create_visualization(image, defects, kernel_states, edge_image)
        
        # Return tuple format (visualization, defects)
        return visualization, defects
    
    def create_visualization(self, original, defects, kernel_states, edge_image):
        """Create visualization with detected defects highlighted"""
        if len(original.shape) == 2:
            vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            vis = original.copy()
        
        # Create overlay
        overlay = vis.copy()
        
        if self.debug:
            # In debug mode, show kernels
            for state in kernel_states:
                x1, y1, x2, y2 = state['bbox']
                
                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(vis.shape[1], x2)
                y2 = min(vis.shape[0], y2)
                
                # Draw kernel box - green for normal, red for defects
                color = (0, 255, 0)  # Default green
                
                # Check if this kernel is part of a defect
                for defect in defects:
                    if abs(state['y'] - defect['y']) < self.kernel_size // 2:
                        color = (0, 0, 255)  # Red for defect
                        break
                
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
            
            # Blend lightly to see kernels clearly
            result = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
            
            # Save edge detected image
            edge_vis = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)
            self._debug_edge_image = edge_vis
            
            return result
        else:
            # Non-debug mode - highlight defect regions
            for defect in defects:
                y = defect['y']
                x = defect['x']
                
                # Draw filled red rectangle overlay for misalignment
                thickness = 30  # Height of the misalignment indicator
                width = 100  # Width of the misalignment indicator
                
                cv2.rectangle(overlay,
                            (x - width // 2, y - thickness // 2),
                            (x + width // 2, y + thickness // 2),
                            (0, 0, 255),  # Red color
                            -1)  # Filled rectangle
                
                # Add text showing the delta
                cv2.putText(overlay, f"Delta: {defect['x_delta']}px",
                           (x - 40, y - thickness // 2 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Blend with original
            result = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
            
            return result
    
    def save_debug_images(self, output_dir, base_name):
        """Save debug images if debug mode is enabled"""
        if self.debug and hasattr(self, '_debug_edge_image'):
            edge_path = os.path.join(output_dir, f"{base_name}_edge_detected.jpg")
            cv2.imwrite(edge_path, self._debug_edge_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return edge_path
        return None 