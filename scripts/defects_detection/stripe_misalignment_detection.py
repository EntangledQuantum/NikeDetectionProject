"""
Stripe Misalignment Detection Algorithm
Detects misalignment of vertical stripes caused by printer head misalignment
"""

import cv2
import numpy as np
import os

from utils.image_saver import save_image
from utils.edge_detector import detect_edges_enhanced


class StripeMisalignmentDetector:
    """Detect vertical stripe misalignment via kernel scanning and edge preprocessing.

    The algorithm enhances vertical edges, then scans rows with a rectangular
    kernel to find the first strong vertical line per row. It compares the x
    positions across rows to flag significant lateral shifts as defects.
    Sensitivity presets adjust kernel shape, step size, and thresholds.
    """
    
    def __init__(self, kernel_size=50, kernel_width=None, kernel_height=None, 
                 step_size=None, line_detection_threshold=0.15,
                 defect_threshold=10, sensitivity='medium', debug=False):
        """Configure kernel geometry, thresholds, and debug mode.

        Args:
            kernel_size: Square kernel size when width/height are not provided.
            kernel_width: Explicit kernel width in pixels (overrides kernel_size).
            kernel_height: Explicit kernel height in pixels (overrides kernel_size).
            step_size: Horizontal step between kernel positions; defaults to width.
            line_detection_threshold: Min white pixel ratio to qualify as a line.
            defect_threshold: Min change in x between rows to be a misalignment.
            sensitivity: One of {'low', 'medium', 'high'}.
            debug: If True, draw kernel boxes and store edge image.
        """
        # Handle rectangular kernels
        if kernel_width is not None and kernel_height is not None:
            self.kernel_width = kernel_width
            self.kernel_height = kernel_height
        else:
            # Fall back to square kernel
            self.kernel_width = kernel_size
            self.kernel_height = kernel_size
        
        self.step_size = step_size if step_size else self.kernel_width
        self.line_detection_threshold = line_detection_threshold
        self.defect_threshold = defect_threshold
        self.debug = debug
        
        # print sensitivity
        print(f"Sensitivity: {sensitivity}")
        
        # Adjust parameters based on sensitivity
        if sensitivity == 'high':
            self.kernel_width = 30
            self.kernel_height = 30
            self.step_size = 30
            self.line_detection_threshold = 0.10  # More sensitive to line detection
            self.defect_threshold = 5  # Smaller misalignment considered defect
        elif sensitivity == 'low':
            self.kernel_width = 70
            self.kernel_height = 70
            self.step_size = 70
            self.line_detection_threshold = 0.20  # Less sensitive to line detection
            self.defect_threshold = 20  # Larger misalignment needed for defect
        else:  # medium
            self.kernel_width = 5
            self.kernel_height = 60
            self.step_size = 5
            self.line_detection_threshold = 0.20
            self.defect_threshold = 20
    
    def preprocess_with_edge_detection(self, image):
        """Enhance vertical edges to facilitate robust line detection.

        Returns a binary edge map after Gaussian blur, median filtering, Sobel
        gradient on x, normalization, and thresholding.

        Args:
            image: Input image (BGR or grayscale).

        Returns:
            Binary uint8 image emphasizing vertical edges.
        """
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
        """Scan rows and collect misalignment defects across the image.

        Args:
            binary_image: Binary edge image from preprocessing.

        Returns:
            tuple: (kernel_states, defects)
        """
        height, width = binary_image.shape
        kernel_states = []
        defects = []
        
        # Track line detection state
        line_detected = False
        previous_x_pos = None
        
        # Start from top of image
        y = self.kernel_height // 2
        
        while y < height - self.kernel_height // 2:
            row_results = self.scan_row(binary_image, y)
            
            if row_results:
                # Process results for this row
                for result in row_results:
                    x_pos = result['x_center']
                    
                    # If x_pos is None, this row has no line detected
                    if x_pos is not None:
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
                    
                    # Always add kernels for debug visualization
                    kernel_states.extend(result['kernels'])
            
            # Move to next row
            y += self.kernel_height
        
        return kernel_states, defects
    
    def scan_row(self, binary_image, y):
        """Scan a single row for the first strong vertical line position.

        Args:
            binary_image: Binary edge image.
            y: Row center to scan around using kernel height.

        Returns:
            List[dict]: Row result entries including first line x position and
            all kernel states when debug is True.
        """
        height, width = binary_image.shape
        row_results = []
        
        # Extract row region
        y1 = max(0, y - self.kernel_height // 2)
        y2 = min(height, y + self.kernel_height // 2)
        
        # Scan horizontally
        x = self.kernel_width // 2
        line_found_in_row = False
        line_x_position = None
        kernels_in_row = []
        
        while x < width - self.kernel_width // 2:
            # Extract kernel region
            x1 = max(0, x - self.kernel_width // 2)
            x2 = min(width, x + self.kernel_width // 2)
            
            kernel_region = binary_image[y1:y2, x1:x2]
            
            # Check if there's line in kernel
            white_pixels = np.sum(kernel_region > 0)
            total_pixels = (y2 - y1) * (x2 - x1)
            has_line = white_pixels > total_pixels * self.line_detection_threshold
            
            # Record kernel state for debug - ALL kernels scanned
            if self.debug:
                kernels_in_row.append({
                    'x': x,
                    'y': y,
                    'has_line': has_line,
                    'bbox': (x1, y1, x2, y2)
                })
            
            if has_line and not line_found_in_row:
                # First detection in this row - record position and stop scanning
                line_found_in_row = True
                line_x_position = x
                if self.debug:
                    print(f"Line detected at Y={y}, X={x} - stopping row scan")
                
                # If not in debug mode, we can break here since we found the line
                if not self.debug:
                    break
            
            # Move to next position
            x += self.step_size
        
        # Return results
        if line_found_in_row:
            row_results.append({
                'y': y,
                'x_center': line_x_position,  # Using the first detection position
                'line_start': line_x_position,
                'line_end': line_x_position,
                'kernels': kernels_in_row
            })
        else:
            # Even if no line found, still return kernels for debug visualization
            if self.debug and kernels_in_row:
                row_results.append({
                    'y': y,
                    'x_center': None,
                    'line_start': None,
                    'line_end': None,
                    'kernels': kernels_in_row
                })
        
        return row_results
    
    def detect(self, image):
        """Run stripe misalignment detection and produce a visualization.

        Args:
            image: Input image (BGR or grayscale).

        Returns:
            tuple: (visualization_bgr, defects)
        """
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
        """Create a visualization image showing misalignment highlights.

        Draws kernel boxes in debug mode and filled red rectangles for defects
        in non-debug mode. Stores the edge image for optional saving.

        Args:
            original: Original input image (BGR or grayscale).
            defects: List of misalignment defect dicts.
            kernel_states: Kernel placement states (debug only).
            edge_image: Binary edge map from preprocessing.

        Returns:
            BGR visualization image.
        """
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
                
                # Determine color based on kernel state
                if not state.get('has_line', False):
                    # Grey for kernels that didn't detect a line
                    color = (128, 128, 128)
                else:
                    # Green for normal, red for defects
                    color = (0, 255, 0)  # Default green
                    
                    # Check if this kernel is part of a defect
                    for defect in defects:
                        if abs(state['y'] - defect['y']) < self.kernel_height // 2 and abs(state['x'] - defect['x']) < self.kernel_width:
                            color = (0, 0, 255)  # Red for defect
                            break
                
                # Draw with thicker lines (3 pixels instead of 1)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
            
            # Blend lightly to see kernels clearly
            result = cv2.addWeighted(vis, 0.8, overlay, 0.2, 0)
            
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
                thickness = 100  # Height of the misalignment indicator
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
            result = cv2.addWeighted(vis, 0.8, overlay, 0.2, 0)
            
            return result
    
    def save_debug_images(self, output_dir, base_name):
        """Save the preprocessed edge image when debug mode is enabled.

        Args:
            output_dir: Directory to save images.
            base_name: Base filename used for output naming.

        Returns:
            str | None: Saved file path or None if not saved.
        """
        if self.debug:
            image = getattr(self, '_debug_edge_image', None)
            saved_path = save_image(output_dir, base_name, image, 'edge_detected')
            return saved_path
        return None 