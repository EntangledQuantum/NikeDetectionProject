"""
Head Calibration Defect Detection Algorithm
Detects misalignment between print heads at boundaries

Author: Koushik and Assistant
Date: 2024
Version: 1.0 - Robust edge-based detection
"""

import cv2
import numpy as np
from scipy import signal, interpolate
from skimage import morphology, measure, filters
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Optional


class HeadCalibrationDetector:
    """
    Detects head calibration defects characterized by:
    - Misalignment at print head boundaries (1/4, 1/2, 3/4 positions)
    - Stitch errors (vertical position misalignment)
    - Roll errors (angular misalignment)
    """
    
    def __init__(self,
                 edge_threshold: float = 0.7,
                 alignment_tolerance: int = 5,
                 min_edge_length: int = 100,
                 head_positions: List[float] = [0.25, 0.5, 0.75],
                 boundary_width: int = 50):
        """
        Args:
            edge_threshold: Threshold for edge detection strength
            alignment_tolerance: Maximum pixels for acceptable alignment
            min_edge_length: Minimum length of edge to consider
            head_positions: Relative positions where heads meet (0-1)
            boundary_width: Width of region to check around head boundaries
        """
        self.edge_threshold = edge_threshold
        self.alignment_tolerance = alignment_tolerance
        self.min_edge_length = min_edge_length
        self.head_positions = head_positions
        self.boundary_width = boundary_width
    
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Main detection method
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Tuple of (visualization, defects list)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detect edges
        edges = self._detect_stripe_edges(gray)
        
        # Extract edge profiles
        left_edge, right_edge = self._extract_edge_profiles(edges, gray)
        
        # Analyze head boundaries
        calibration_errors = self._analyze_head_boundaries(left_edge, right_edge, gray.shape)
        
        # Detect roll errors
        roll_errors = self._detect_roll_errors(left_edge, right_edge, gray.shape)
        
        # Combine all errors
        all_errors = calibration_errors + roll_errors
        
        # Create defects list
        defects = self._create_defect_list(all_errors)
        
        # Create visualization
        visualization = self._create_visualization(image, edges, left_edge, right_edge, all_errors)
        
        return visualization, defects
    
    def _detect_stripe_edges(self, gray: np.ndarray) -> np.ndarray:
        """Detect vertical edges of the stripe"""
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 50, 50)
        
        # Calculate horizontal gradient (vertical edges)
        grad_x = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=5)
        
        # Apply non-maximum suppression for edge thinning
        # Calculate gradient magnitude and direction
        grad_y = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Non-maximum suppression
        h, w = magnitude.shape
        suppressed = np.zeros_like(magnitude)
        
        for y in range(1, h-1):
            for x in range(1, w-1):
                angle = direction[y, x]
                mag = magnitude[y, x]
                
                # Determine neighbors based on gradient direction
                # For vertical edges, we're interested in horizontal gradients
                if (-np.pi/8 <= angle < np.pi/8) or (7*np.pi/8 <= angle) or (angle < -7*np.pi/8):
                    # Horizontal gradient - check left and right
                    if mag >= magnitude[y, x-1] and mag >= magnitude[y, x+1]:
                        suppressed[y, x] = mag
                elif (3*np.pi/8 <= angle < 5*np.pi/8) or (-5*np.pi/8 <= angle < -3*np.pi/8):
                    # Vertical gradient - check top and bottom
                    if mag >= magnitude[y-1, x] and mag >= magnitude[y+1, x]:
                        suppressed[y, x] = mag
        
        # Threshold to get binary edges
        edge_threshold = np.percentile(suppressed[suppressed > 0], 100 * self.edge_threshold)
        edges = suppressed > edge_threshold
        
        # Clean up edges with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        edges = cv2.morphologyEx(edges.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
        
        return edges
    
    def _extract_edge_profiles(self, edges: np.ndarray, gray: np.ndarray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Extract left and right edge profiles"""
        h, w = edges.shape
        
        # Find left edge (first significant edge from left)
        left_edge = []
        for y in range(h):
            row = edges[y, :]
            edge_positions = np.where(row > 0)[0]
            if len(edge_positions) > 0:
                # Find the leftmost edge that's part of the main stripe
                for x in edge_positions:
                    # Check if this is a significant edge by looking at intensity change
                    if x < w - 10:
                        intensity_change = abs(int(gray[y, x+5]) - int(gray[y, max(0, x-5)]))
                        if intensity_change > 30:  # Significant change
                            left_edge.append((x, y))
                            break
        
        # Find right edge (last significant edge from right)
        right_edge = []
        for y in range(h):
            row = edges[y, :]
            edge_positions = np.where(row > 0)[0]
            if len(edge_positions) > 0:
                # Find the rightmost edge that's part of the main stripe
                for x in reversed(edge_positions):
                    # Check if this is a significant edge
                    if x > 10:
                        intensity_change = abs(int(gray[y, min(w-1, x+5)]) - int(gray[y, x-5]))
                        if intensity_change > 30:  # Significant change
                            right_edge.append((x, y))
                            break
        
        # Smooth edge profiles using moving average
        left_edge = self._smooth_edge_profile(left_edge)
        right_edge = self._smooth_edge_profile(right_edge)
        
        return left_edge, right_edge
    
    def _smooth_edge_profile(self, edge_points: List[Tuple[int, int]], window_size: int = 11) -> List[Tuple[int, int]]:
        """Smooth edge profile using moving average"""
        if len(edge_points) < window_size:
            return edge_points
        
        # Extract x and y coordinates
        x_coords = [p[0] for p in edge_points]
        y_coords = [p[1] for p in edge_points]
        
        # Apply moving average
        smoothed_x = signal.savgol_filter(x_coords, window_size, 3)
        
        # Reconstruct edge points
        smoothed_edge = [(int(smoothed_x[i]), y_coords[i]) for i in range(len(edge_points))]
        
        return smoothed_edge
    
    def _analyze_head_boundaries(self, left_edge: List[Tuple[int, int]], 
                               right_edge: List[Tuple[int, int]], 
                               image_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Analyze misalignment at head boundaries"""
        h, w = image_shape
        calibration_errors = []
        
        # Check each head boundary position
        for head_pos in self.head_positions:
            boundary_y = int(h * head_pos)
            
            # Define region around boundary
            y_start = max(0, boundary_y - self.boundary_width // 2)
            y_end = min(h, boundary_y + self.boundary_width // 2)
            
            # Analyze left edge
            left_error = self._check_edge_alignment(left_edge, y_start, y_end, boundary_y, 'left')
            if left_error:
                left_error['head_position'] = head_pos
                calibration_errors.append(left_error)
            
            # Analyze right edge
            right_error = self._check_edge_alignment(right_edge, y_start, y_end, boundary_y, 'right')
            if right_error:
                right_error['head_position'] = head_pos
                calibration_errors.append(right_error)
        
        return calibration_errors
    
    def _check_edge_alignment(self, edge_profile: List[Tuple[int, int]], 
                            y_start: int, y_end: int, boundary_y: int, 
                            edge_side: str) -> Optional[Dict[str, Any]]:
        """Check for alignment errors at a specific boundary"""
        # Extract edge points in the boundary region
        boundary_points = [(x, y) for x, y in edge_profile if y_start <= y <= y_end]
        
        if len(boundary_points) < self.min_edge_length // 2:
            return None
        
        # Split into upper and lower regions
        upper_points = [(x, y) for x, y in boundary_points if y < boundary_y]
        lower_points = [(x, y) for x, y in boundary_points if y >= boundary_y]
        
        if len(upper_points) < 10 or len(lower_points) < 10:
            return None
        
        # Calculate average x position for each region
        upper_avg_x = np.mean([p[0] for p in upper_points])
        lower_avg_x = np.mean([p[0] for p in lower_points])
        
        # Calculate misalignment
        misalignment = abs(upper_avg_x - lower_avg_x)
        
        if misalignment > self.alignment_tolerance:
            # Fit lines to detect the exact transition point
            upper_x = [p[0] for p in upper_points]
            upper_y = [p[1] for p in upper_points]
            lower_x = [p[0] for p in lower_points]
            lower_y = [p[1] for p in lower_points]
            
            # Find the discontinuity point
            transition_y = boundary_y
            if len(upper_points) > 0 and len(lower_points) > 0:
                # Find the gap between upper and lower segments
                last_upper_y = max(upper_y)
                first_lower_y = min(lower_y)
                transition_y = (last_upper_y + first_lower_y) // 2
            
            return {
                'type': 'stitch_error',
                'edge_side': edge_side,
                'boundary_y': boundary_y,
                'transition_y': transition_y,
                'misalignment': misalignment,
                'upper_position': upper_avg_x,
                'lower_position': lower_avg_x,
                'severity': 'high' if misalignment > 2 * self.alignment_tolerance else 'medium'
            }
        
        return None
    
    def _detect_roll_errors(self, left_edge: List[Tuple[int, int]], 
                          right_edge: List[Tuple[int, int]], 
                          image_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Detect roll (angular) misalignment errors"""
        h, w = image_shape
        roll_errors = []
        
        # Check each head region
        for i, head_pos in enumerate(self.head_positions):
            # Define head region
            if i == 0:
                y_start = 0
                y_end = int(h * head_pos)
            else:
                y_start = int(h * self.head_positions[i-1])
                y_end = int(h * head_pos)
            
            # Get edge points in this region
            left_region = [(x, y) for x, y in left_edge if y_start <= y < y_end]
            right_region = [(x, y) for x, y in right_edge if y_start <= y < y_end]
            
            if len(left_region) < 20 or len(right_region) < 20:
                continue
            
            # Fit lines to edges
            left_x = np.array([p[0] for p in left_region])
            left_y = np.array([p[1] for p in left_region])
            right_x = np.array([p[0] for p in right_region])
            right_y = np.array([p[1] for p in right_region])
            
            # Calculate slopes
            left_slope, left_intercept = np.polyfit(left_y, left_x, 1)
            right_slope, right_intercept = np.polyfit(right_y, right_x, 1)
            
            # Calculate angles
            left_angle = np.arctan(left_slope) * 180 / np.pi
            right_angle = np.arctan(right_slope) * 180 / np.pi
            
            # Check if edges are not parallel (roll error)
            angle_difference = abs(left_angle - right_angle)
            
            if angle_difference > 0.5:  # More than 0.5 degrees
                # Calculate stripe width at top and bottom of region
                top_width = (right_intercept + right_slope * y_start) - (left_intercept + left_slope * y_start)
                bottom_width = (right_intercept + right_slope * y_end) - (left_intercept + left_slope * y_end)
                width_change = abs(top_width - bottom_width)
                
                roll_errors.append({
                    'type': 'roll_error',
                    'head_index': i,
                    'y_range': (y_start, y_end),
                    'angle_difference': angle_difference,
                    'width_change': width_change,
                    'left_angle': left_angle,
                    'right_angle': right_angle,
                    'severity': 'high' if angle_difference > 1.0 else 'medium'
                })
        
        return roll_errors
    
    def _create_defect_list(self, errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create detailed defect list"""
        defects = []
        
        for error in errors:
            if error['type'] == 'stitch_error':
                defects.append({
                    'type': 'head_calibration',
                    'subtype': 'stitch',
                    'location': (0, error['transition_y']),  # Y position of error
                    'edge_side': error['edge_side'],
                    'misalignment_pixels': float(error['misalignment']),
                    'head_boundary': float(error['boundary_y']),
                    'severity': error['severity']
                })
            elif error['type'] == 'roll_error':
                defects.append({
                    'type': 'head_calibration',
                    'subtype': 'roll',
                    'head_index': error['head_index'],
                    'y_range': error['y_range'],
                    'angle_difference': float(error['angle_difference']),
                    'width_variation': float(error['width_change']),
                    'severity': error['severity']
                })
        
        return defects
    
    def _create_visualization(self, original: np.ndarray, edges: np.ndarray,
                            left_edge: List[Tuple[int, int]], right_edge: List[Tuple[int, int]],
                            errors: List[Dict[str, Any]]) -> np.ndarray:
        """Create visualization highlighting calibration errors"""
        if len(original.shape) == 2:
            vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            vis = original.copy()
        
        h, w = vis.shape[:2]
        
        # Draw detected edges
        edge_overlay = np.zeros_like(vis)
        edge_overlay[edges > 0] = [0, 255, 0]  # Green for edges
        vis = cv2.addWeighted(vis, 0.8, edge_overlay, 0.2, 0)
        
        # Draw edge profiles
        for x, y in left_edge:
            cv2.circle(vis, (x, y), 1, [0, 255, 255], -1)  # Yellow for left edge
        for x, y in right_edge:
            cv2.circle(vis, (x, y), 1, [255, 255, 0], -1)  # Cyan for right edge
        
        # Draw head boundaries
        for head_pos in self.head_positions:
            y_pos = int(h * head_pos)
            cv2.line(vis, (0, y_pos), (w, y_pos), [128, 128, 128], 1)  # Gray lines
        
        # Highlight errors
        for error in errors:
            if error['type'] == 'stitch_error':
                # Draw misalignment
                y = error['transition_y']
                if error['edge_side'] == 'left':
                    x1 = int(error['upper_position'])
                    x2 = int(error['lower_position'])
                    cv2.line(vis, (x1, y - 20), (x1, y), [0, 0, 255], 2)
                    cv2.line(vis, (x2, y), (x2, y + 20), [0, 0, 255], 2)
                    cv2.line(vis, (x1, y), (x2, y), [0, 0, 255], 2)
                else:
                    x1 = int(error['upper_position'])
                    x2 = int(error['lower_position'])
                    cv2.line(vis, (x1, y - 20), (x1, y), [0, 0, 255], 2)
                    cv2.line(vis, (x2, y), (x2, y + 20), [0, 0, 255], 2)
                    cv2.line(vis, (x1, y), (x2, y), [0, 0, 255], 2)
                
                # Draw error region
                cv2.rectangle(vis, (0, y - 30), (w, y + 30), [255, 0, 0], 2)
                
            elif error['type'] == 'roll_error':
                # Highlight roll error region
                y_start, y_end = error['y_range']
                cv2.rectangle(vis, (0, y_start), (w, y_end), [255, 0, 255], 2)
        
        return vis 