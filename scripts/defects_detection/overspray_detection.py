"""
Overspray Detection Algorithm
Detects overspray defects by finding regions with scattered pixels

Author: Assistant
Date: 2024
"""

import cv2
import numpy as np
from scipy import ndimage
import os


class OversprayDetector:
    """Detect overspray defects by grid scanning and scatter analysis.

    The detector preprocesses the image to a binary representation where spray
    pixels are white, then slides a kernel across a grid to measure spatial
    scatter of white pixels. Regions exceeding a scatter threshold are flagged
    as overspray, and adjacent kernels can be merged into larger regions.
    """
    
    def __init__(self, kernel_size=30, step_size=None, scatter_threshold=0.3,
                 min_pixels_threshold=0.05, sensitivity='medium', debug=False):
        """Configure grid kernel size, step, and thresholds.

        Args:
            kernel_size: Square kernel size in pixels for grid scanning.
            step_size: Grid step in pixels; defaults to `kernel_size` (no overlap).
            scatter_threshold: Threshold (0..1) for scatter metric to flag overspray.
            min_pixels_threshold: Minimum white pixel ratio within kernel to analyze.
            sensitivity: One of {'low', 'medium', 'high'}; adjusts defaults.
            debug: If True, stores kernel states for visualization.
        """
        self.kernel_size = kernel_size
        self.step_size = step_size if step_size else kernel_size
        self.scatter_threshold = scatter_threshold
        self.min_pixels_threshold = min_pixels_threshold
        self.debug = debug
        
        print(f"Sensitivity: {sensitivity}")
        
        # Adjust parameters based on sensitivity
        if sensitivity == 'high':
            self.kernel_size = 20
            self.step_size = 15  # More overlap for high sensitivity
            self.scatter_threshold = 0.2  # Lower threshold - more sensitive
            self.min_pixels_threshold = 0.03  # Detect smaller amounts of scatter
        elif sensitivity == 'low':
            self.kernel_size = 50
            self.step_size = 50  # No overlap
            self.scatter_threshold = 0.5  # Higher threshold - less sensitive
            self.min_pixels_threshold = 0.1  # Need more pixels to consider
        else:  # medium
            self.kernel_size = 500
            self.step_size = 500  # Small overlap
            self.scatter_threshold = 0.3
            self.min_pixels_threshold = 0.05
    
    def preprocess_image(self, image):
        """Preprocess input image into a binary mask for overspray analysis.

        Steps: grayscale, CLAHE contrast enhancement, adaptive threshold, and
        optional inversion so overspray pixels are white.

        Args:
            image: Input image (BGR or grayscale).

        Returns:
            Binary uint8 image with overspray pixels as white (255).
        """
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
        
        # Invert if necessary (we want spray pixels to be white)
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
        
        return binary
    
    def calculate_scatter_metric(self, kernel_region):
        """Compute a scatter score for white pixels within a kernel region.

        Combines spread (std of coordinates normalized by kernel size) and
        distance variance between sampled points, with weighting.

        Args:
            kernel_region: Binary kernel crop (uint8) from the preprocessed image.

        Returns:
            float in [0, 1] representing scatter; higher implies overspray-like.
        """
        if kernel_region.size == 0:
            return 0
        
        # Get white pixel positions
        white_pixels = np.where(kernel_region > 0)
        
        if len(white_pixels[0]) == 0:
            return 0
        
        # Calculate metrics
        total_pixels = kernel_region.size
        white_pixel_count = len(white_pixels[0])
        white_ratio = white_pixel_count / total_pixels
        
        # If too few pixels, not overspray
        if white_ratio < self.min_pixels_threshold:
            return 0
        
        # If too many pixels, likely solid region not overspray
        if white_ratio > 0.7:
            return 0
        
        # Calculate spatial distribution - how spread out are the pixels
        if white_pixel_count > 1:
            # Calculate standard deviation of positions
            y_std = np.std(white_pixels[0])
            x_std = np.std(white_pixels[1])
            
            # Normalize by kernel size
            y_spread = y_std / (kernel_region.shape[0] / 2)
            x_spread = x_std / (kernel_region.shape[1] / 2)
            
            # Combined spread metric
            spread = (y_spread + x_spread) / 2
            
            # Also consider the distribution pattern
            # Calculate distances between consecutive pixels
            points = np.column_stack(white_pixels)
            if len(points) > 2:
                # Calculate pairwise distances
                distances = []
                for i in range(min(len(points), 20)):  # Sample up to 20 points
                    for j in range(i + 1, min(len(points), 20)):
                        dist = np.sqrt((points[i][0] - points[j][0])**2 + 
                                     (points[i][1] - points[j][1])**2)
                        distances.append(dist)
                
                if distances:
                    # High variance in distances indicates scattered pattern
                    dist_variance = np.var(distances) / (self.kernel_size**2)
                    
                    # Combine spread and variance metrics
                    scatter_metric = (spread * 0.6 + dist_variance * 0.4)
                else:
                    scatter_metric = spread
            else:
                scatter_metric = spread
            
            # Adjust by pixel ratio - moderate amounts of pixels are more likely overspray
            if 0.05 < white_ratio < 0.3:
                scatter_metric *= 1.2  # Boost score for typical overspray density
            
            return min(scatter_metric, 1.0)  # Cap at 1.0
        
        return 0
    
    def scan_grid(self, binary_image):
        """Slide kernel over a grid to identify overspray kernels.

        Args:
            binary_image: Preprocessed binary image with overspray pixels white.

        Returns:
            tuple: (kernel_states, defects)
                - kernel_states: List of per-kernel diagnostics (when debug)
                - defects: List of kernel-level detections or merged regions
        """
        height, width = binary_image.shape
        kernel_states = []
        defects = []
        
        # Scan in grid pattern
        y = self.kernel_size // 2
        while y < height - self.kernel_size // 2:
            x = self.kernel_size // 2
            while x < width - self.kernel_size // 2:
                # Extract kernel region
                y1 = max(0, y - self.kernel_size // 2)
                y2 = min(height, y + self.kernel_size // 2)
                x1 = max(0, x - self.kernel_size // 2)
                x2 = min(width, x + self.kernel_size // 2)
                
                kernel_region = binary_image[y1:y2, x1:x2]
                
                # Calculate scatter metric
                scatter_metric = self.calculate_scatter_metric(kernel_region)
                
                # Determine if this is overspray
                is_overspray = scatter_metric > self.scatter_threshold
                
                # Record kernel state for debug
                if self.debug:
                    kernel_states.append({
                        'x': x,
                        'y': y,
                        'has_overspray': is_overspray,
                        'scatter_metric': scatter_metric,
                        'bbox': (x1, y1, x2, y2)
                    })
                
                # Record defect if found
                if is_overspray:
                    defects.append({
                        'type': 'overspray',
                        'x': x,
                        'y': y,
                        'location': (x, y),
                        'scatter_metric': float(scatter_metric),
                        'kernel_size': self.kernel_size,
                        'bbox': (x1, y1, x2, y2)
                    })
                    
                    # if self.debug:
                    #     print(f"Overspray detected at X={x}, Y={y}: scatter={scatter_metric:.3f}")
                
                # Move to next position
                x += self.step_size
            
            # Move to next row
            y += self.step_size
        
        # Merge nearby defects to create regions
        if defects and not self.debug:
            defects = self.merge_nearby_defects(defects)
        
        return kernel_states, defects
    
    def merge_nearby_defects(self, defects):
        """Merge spatially adjacent kernel-level detections into regions.

        Args:
            defects: List of kernel-level overspray detections with bboxes.

        Returns:
            List of merged region dicts with bbox, area, center, and metrics.
        """
        if not defects:
            return defects
        
        # Create a mask for defect regions
        # Find image bounds
        max_x = max(d['x'] for d in defects) + self.kernel_size
        max_y = max(d['y'] for d in defects) + self.kernel_size
        
        # Create mask
        defect_mask = np.zeros((max_y, max_x), dtype=np.uint8)
        
        # Fill defect regions
        for defect in defects:
            x1, y1, x2, y2 = defect['bbox']
            defect_mask[y1:y2, x1:x2] = 255
        
        # Apply morphological closing to merge nearby regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (self.kernel_size, self.kernel_size))
        merged_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours of merged regions
        contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        # Create new defect list from merged regions
        merged_defects = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w // 2
            cy = y + h // 2
            
            # Calculate average scatter metric for this region
            region_metrics = [d['scatter_metric'] for d in defects 
                            if x <= d['x'] <= x + w and y <= d['y'] <= y + h]
            avg_metric = np.mean(region_metrics) if region_metrics else 0
            
            merged_defects.append({
                'type': 'overspray_region',
                'x': cx,
                'y': cy,
                'location': (cx, cy),
                'scatter_metric': float(avg_metric),
                'bbox': (x, y, x + w, y + h),
                'area': cv2.contourArea(contour),
                'kernel_count': len(region_metrics)
            })
        
        return merged_defects
    
    def detect(self, image):
        """Run overspray detection and return visualization plus defects.

        Args:
            image: Input image (BGR or grayscale).

        Returns:
            tuple: (visualization_bgr, defects)
        """
        # Preprocess image
        binary = self.preprocess_image(image)
        
        # Scan grid for overspray
        kernel_states, defects = self.scan_grid(binary)
        
        # Create visualization
        visualization = self.create_visualization(image, defects, kernel_states)
        
        # Return tuple format (visualization, defects)
        return visualization, defects
    
    def create_visualization(self, original, defects, kernel_states=None):
        """Create a visualization image highlighting overspray findings.

        Args:
            original: Original image (BGR or grayscale).
            defects: List of defect dicts (kernel-level or merged regions).
            kernel_states: Optional kernel state list for debug overlays.

        Returns:
            BGR image with overlays indicating overspray.
        """
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
                
                # Color based on scatter metric
                if state['has_overspray']:
                    # Red for overspray
                    color = (0, 0, 255)
                elif state['scatter_metric'] > self.scatter_threshold * 0.7:
                    # Yellow for near-threshold
                    color = (0, 255, 255)
                else:
                    # Green for clean areas
                    color = (0, 255, 0)
                
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
                
                # Draw center point
                cv2.circle(overlay, (x, y), 2, (255, 0, 0), -1)
            
            # In debug mode, blend lightly to see kernels clearly
            result = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
            return result
        
        # Non-debug mode - highlight overspray regions
        for defect in defects:
            if 'bbox' in defect:
                x1, y1, x2, y2 = defect['bbox']
                
                # Draw filled red rectangle overlay for overspray region
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
        
        # Blend with original to create overlay effect
        result = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
        
        return result 