"""
Void Defect Detection Algorithm
Detects small circular areas without ink where ink should be

Author: Koushik and Assistant
Date: 2024
Version: 1.0 - Robust detection for small voids
"""

import cv2
import numpy as np
from scipy import ndimage, signal
from skimage import morphology, measure, filters, feature, segmentation
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any


class VoidDetector:
    """
    Detects void defects characterized by:
    - Small circular/elliptical areas without ink
    - Typically very small in size
    - High contrast with surrounding printed area
    """
    
    def __init__(self,
                 min_void_size: int = 10,
                 max_void_size: int = 500,
                 circularity_threshold: float = 0.6,
                 contrast_threshold: float = 0.3,
                 edge_sharpness_threshold: float = 0.7):
        """
        Args:
            min_void_size: Minimum area for void detection
            max_void_size: Maximum area for void detection
            circularity_threshold: Minimum circularity (4π*area/perimeter²)
            contrast_threshold: Minimum contrast with surroundings
            edge_sharpness_threshold: Threshold for edge sharpness
        """
        self.min_void_size = min_void_size
        self.max_void_size = max_void_size
        self.circularity_threshold = circularity_threshold
        self.contrast_threshold = contrast_threshold
        self.edge_sharpness_threshold = edge_sharpness_threshold
    
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
        
        # Detect voids using multiple approaches
        blob_voids = self._detect_blob_voids(gray)
        edge_voids = self._detect_edge_based_voids(gray)
        morphological_voids = self._detect_morphological_voids(gray)
        
        # Combine detections
        combined_mask = self._combine_void_detections(blob_voids, edge_voids, morphological_voids)
        
        # Analyze void characteristics
        void_regions = self._analyze_void_regions(combined_mask, gray)
        
        # Filter false positives
        filtered_voids = self._filter_false_positives(void_regions, gray)
        
        # Create defects list
        defects = self._create_defect_list(filtered_voids)
        
        # Create visualization
        visualization = self._create_visualization(image, combined_mask, filtered_voids)
        
        return visualization, defects
    
    def _detect_blob_voids(self, gray: np.ndarray) -> List[Dict[str, Any]]:
        """Detect voids using blob detection"""
        # Invert image to make voids dark
        inverted = cv2.bitwise_not(gray)
        
        # Set up blob detector with parameters optimized for small voids
        params = cv2.SimpleBlobDetector_Params()
        
        # Filter by area
        params.filterByArea = True
        params.minArea = self.min_void_size
        params.maxArea = self.max_void_size
        
        # Filter by circularity
        params.filterByCircularity = True
        params.minCircularity = self.circularity_threshold
        
        # Filter by convexity
        params.filterByConvexity = True
        params.minConvexity = 0.8
        
        # Filter by inertia (elongation)
        params.filterByInertia = True
        params.minInertiaRatio = 0.5
        
        # Create detector
        detector = cv2.SimpleBlobDetector_create(params)
        
        # Detect blobs
        keypoints = detector.detect(inverted)
        
        # Convert keypoints to void info
        blob_voids = []
        for kp in keypoints:
            blob_voids.append({
                'center': (int(kp.pt[0]), int(kp.pt[1])),
                'radius': kp.size / 2,
                'area': np.pi * (kp.size / 2) ** 2,
                'response': kp.response
            })
        
        return blob_voids
    
    def _detect_edge_based_voids(self, gray: np.ndarray) -> np.ndarray:
        """Detect voids using edge detection and circle finding"""
        # Apply adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Multi-scale edge detection
        edges_fine = cv2.Canny(enhanced, 100, 200)
        edges_coarse = cv2.Canny(enhanced, 50, 100)
        
        # Combine edges
        edges = cv2.bitwise_or(edges_fine, edges_coarse)
        
        # Find circles using Hough transform
        circles = cv2.HoughCircles(
            enhanced,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=100,
            param2=15,
            minRadius=int(np.sqrt(self.min_void_size / np.pi)),
            maxRadius=int(np.sqrt(self.max_void_size / np.pi))
        )
        
        # Create mask from detected circles
        void_mask = np.zeros_like(gray)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                # Verify it's actually a void (bright inside)
                x, y, r = circle
                if 0 <= y - r and y + r < gray.shape[0] and 0 <= x - r and x + r < gray.shape[1]:
                    # Create circular mask
                    y_grid, x_grid = np.ogrid[:gray.shape[0], :gray.shape[1]]
                    circle_mask = (x_grid - x)**2 + (y_grid - y)**2 <= r**2
                    
                    # Check if inside is brighter than surroundings
                    inside_mean = np.mean(gray[circle_mask])
                    
                    # Create ring mask for surrounding
                    ring_mask = ((x_grid - x)**2 + (y_grid - y)**2 <= (r + 5)**2) & \
                               ((x_grid - x)**2 + (y_grid - y)**2 > r**2)
                    
                    if np.any(ring_mask):
                        outside_mean = np.mean(gray[ring_mask])
                        
                        # Void should be brighter inside
                        if inside_mean > outside_mean * (1 + self.contrast_threshold):
                            cv2.circle(void_mask, (x, y), r, 255, -1)
        
        return void_mask
    
    def _detect_morphological_voids(self, gray: np.ndarray) -> np.ndarray:
        """Detect voids using morphological operations"""
        # Apply adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 5
        )
        
        # Find small bright regions
        # Opening to remove noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel_small)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
        
        # Create void mask
        void_mask = np.zeros_like(gray)
        
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            
            if self.min_void_size <= area <= self.max_void_size:
                # Check circularity
                component_mask = labels == i
                contours, _ = cv2.findContours(
                    component_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                if contours:
                    contour = contours[0]
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        
                        if circularity >= self.circularity_threshold:
                            # Check contrast
                            x, y = int(centroids[i][0]), int(centroids[i][1])
                            component_mean = np.mean(gray[component_mask])
                            
                            # Get surrounding region
                            x_min = max(0, stats[i, cv2.CC_STAT_LEFT] - 10)
                            x_max = min(gray.shape[1], stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] + 10)
                            y_min = max(0, stats[i, cv2.CC_STAT_TOP] - 10)
                            y_max = min(gray.shape[0], stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] + 10)
                            
                            surrounding_roi = gray[y_min:y_max, x_min:x_max]
                            surrounding_mean = np.mean(surrounding_roi)
                            
                            if component_mean > surrounding_mean * (1 + self.contrast_threshold):
                                void_mask[component_mask] = 255
        
        return void_mask
    
    def _combine_void_detections(self, blob_voids: List[Dict[str, Any]], 
                               edge_mask: np.ndarray, morph_mask: np.ndarray) -> np.ndarray:
        """Combine different void detection methods"""
        # Create combined mask
        combined = np.zeros_like(edge_mask)
        
        # Add blob detections
        for void in blob_voids:
            cv2.circle(combined, void['center'], int(void['radius']), 255, -1)
        
        # Add edge-based detections
        combined = cv2.bitwise_or(combined, edge_mask)
        
        # Add morphological detections
        combined = cv2.bitwise_or(combined, morph_mask)
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        return combined
    
    def _analyze_void_regions(self, mask: np.ndarray, gray: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze characteristics of detected void regions"""
        labeled = measure.label(mask > 0)
        props = measure.regionprops(labeled, intensity_image=gray)
        
        void_regions = []
        
        for prop in props:
            if self.min_void_size <= prop.area <= self.max_void_size:
                # Calculate additional features
                # Edge sharpness
                y, x = prop.centroid
                y, x = int(y), int(x)
                
                # Get bounding box with margin
                minr, minc, maxr, maxc = prop.bbox
                margin = 5
                minr = max(0, minr - margin)
                minc = max(0, minc - margin)
                maxr = min(gray.shape[0], maxr + margin)
                maxc = min(gray.shape[1], maxc + margin)
                
                roi = gray[minr:maxr, minc:maxc]
                
                # Calculate edge strength
                edges = cv2.Canny(roi, 50, 150)
                edge_pixels = np.sum(edges > 0)
                total_pixels = edges.size
                edge_density = edge_pixels / total_pixels if total_pixels > 0 else 0
                
                # Calculate contrast
                void_mean = prop.mean_intensity
                surrounding_mask = np.zeros_like(roi, dtype=bool)
                surrounding_mask[margin:-margin, margin:-margin] = True
                surrounding_mask[prop.coords[:, 0] - minr, prop.coords[:, 1] - minc] = False
                
                if np.any(surrounding_mask):
                    surrounding_mean = np.mean(roi[surrounding_mask])
                    contrast = (void_mean - surrounding_mean) / (surrounding_mean + 1e-6)
                else:
                    contrast = 0
                
                # Calculate circularity
                perimeter = prop.perimeter
                circularity = 4 * np.pi * prop.area / (perimeter ** 2) if perimeter > 0 else 0
                
                void_regions.append({
                    'label': prop.label,
                    'centroid': prop.centroid,
                    'area': prop.area,
                    'bbox': prop.bbox,
                    'circularity': circularity,
                    'eccentricity': prop.eccentricity,
                    'contrast': contrast,
                    'edge_density': edge_density,
                    'mean_intensity': void_mean,
                    'equivalent_diameter': prop.equivalent_diameter
                })
        
        return void_regions
    
    def _filter_false_positives(self, void_regions: List[Dict[str, Any]], 
                              gray: np.ndarray) -> List[Dict[str, Any]]:
        """Filter out false positive detections"""
        filtered = []
        
        for void in void_regions:
            # Check multiple criteria
            is_valid = True
            
            # Circularity check
            if void['circularity'] < self.circularity_threshold:
                is_valid = False
            
            # Contrast check
            if abs(void['contrast']) < self.contrast_threshold:
                is_valid = False
            
            # Edge density check (voids should have sharp edges)
            if void['edge_density'] < 0.1:
                is_valid = False
            
            # Size consistency check
            expected_diameter = 2 * np.sqrt(void['area'] / np.pi)
            actual_diameter = void['equivalent_diameter']
            if abs(expected_diameter - actual_diameter) / expected_diameter > 0.3:
                is_valid = False
            
            if is_valid:
                filtered.append(void)
        
        return filtered
    
    def _create_defect_list(self, void_regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create detailed defect list"""
        defects = []
        
        for void in void_regions:
            defects.append({
                'type': 'void',
                'location': (int(void['centroid'][1]), int(void['centroid'][0])),
                'area': void['area'],
                'diameter': void['equivalent_diameter'],
                'bbox': void['bbox'],
                'circularity': float(void['circularity']),
                'contrast': float(void['contrast']),
                'confidence': float(min(void['circularity'], abs(void['contrast'])))
            })
        
        return defects
    
    def _create_visualization(self, original: np.ndarray, mask: np.ndarray,
                            void_regions: List[Dict[str, Any]]) -> np.ndarray:
        """Create visualization highlighting void defects"""
        if len(original.shape) == 2:
            vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            vis = original.copy()
        
        # Create overlay
        overlay = vis.copy()
        
        # Mark void regions in red
        labeled = measure.label(mask > 0)
        
        for void in void_regions:
            void_mask = labeled == void['label']
            overlay[void_mask] = [0, 0, 255]  # Red for voids
            
            # Draw circle around void for clarity
            center = (int(void['centroid'][1]), int(void['centroid'][0]))
            radius = int(void['equivalent_diameter'] / 2)
            cv2.circle(overlay, center, radius + 2, [0, 255, 0], 1)  # Green circle
        
        # Blend with original
        result = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        
        return result 