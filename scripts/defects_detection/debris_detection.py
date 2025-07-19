"""
Debris Defect Detection Algorithm
Detects foreign particles like dirt, fibers on sheets

Author: Koushik and Assistant
Date: 2024
Version: 2.0 - Complete rewrite for colored prints
"""

import cv2
import numpy as np
from scipy import ndimage, signal
from skimage import morphology, measure, filters, feature, color
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any


class DebrisDetector:
    """
    Detects debris defects characterized by:
    - Dark spots with possible blank rings (pre-print debris)
    - Dark fibers on colored prints
    - Post-print debris particles
    Works on colored images without grayscale conversion
    """
    
    def __init__(self,
                 dark_threshold: float = 0.3,
                 min_debris_size: int = 20,
                 max_debris_size: int = 2000,
                 fiber_min_length: int = 30,
                 halo_detection: bool = True,
                 color_contrast_threshold: float = 0.4):
        """
        Args:
            dark_threshold: Threshold for dark region detection
            min_debris_size: Minimum area for debris particles
            max_debris_size: Maximum area for debris particles
            fiber_min_length: Minimum length for fiber detection
            halo_detection: Whether to detect halos around debris
            color_contrast_threshold: Threshold for color contrast
        """
        self.dark_threshold = dark_threshold
        self.min_debris_size = min_debris_size
        self.max_debris_size = max_debris_size
        self.fiber_min_length = fiber_min_length
        self.halo_detection = halo_detection
        self.color_contrast_threshold = color_contrast_threshold
    
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Main detection method - works on colored images
        
        Args:
            image: Input image (BGR format expected)
            
        Returns:
            Tuple of (visualization, defects list)
        """
        # Ensure we have a color image
        if len(image.shape) == 2:
            # If grayscale, convert to BGR for consistency
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Detect different types of debris
        dark_spots = self._detect_dark_spots(image)
        fibers = self._detect_fibers(image)
        color_anomalies = self._detect_color_anomalies(image)
        
        # Combine all detections
        all_debris = self._combine_detections(dark_spots, fibers, color_anomalies, image.shape[:2])
        
        # Analyze debris characteristics
        debris_regions = self._analyze_debris_regions(all_debris, image)
        
        # Create defects list
        defects = self._create_defect_list(debris_regions)
        
        # Create visualization
        visualization = self._create_visualization(image, debris_regions)
        
        return visualization, defects
    
    def _detect_dark_spots(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect dark spots on colored background"""
        # Convert to LAB color space for better color separation
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]  # Lightness channel
        
        # Find dark regions
        # Adaptive threshold to handle varying background colors
        mean_lightness = np.mean(l_channel)
        dark_threshold = mean_lightness * self.dark_threshold
        
        # Create mask of dark regions
        dark_mask = l_channel < dark_threshold
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark_mask = cv2.morphologyEx(dark_mask.astype(np.uint8) * 255, 
                                    cv2.MORPH_OPEN, kernel)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find connected components
        labeled = measure.label(dark_mask > 0)
        props = measure.regionprops(labeled, intensity_image=l_channel)
        
        dark_spots = []
        for prop in props:
            if self.min_debris_size <= prop.area <= self.max_debris_size:
                # Check if it's really darker than surroundings
                y, x = prop.centroid
                y, x = int(y), int(x)
                
                # Get surrounding region
                margin = 20
                y_min = max(0, y - margin)
                y_max = min(image.shape[0], y + margin)
                x_min = max(0, x - margin)
                x_max = min(image.shape[1], x + margin)
                
                surrounding_roi = l_channel[y_min:y_max, x_min:x_max]
                spot_mean = prop.mean_intensity
                surrounding_mean = np.mean(surrounding_roi)
                
                if spot_mean < surrounding_mean * (1 - self.color_contrast_threshold):
                    # Check for halo if enabled
                    has_halo = False
                    halo_strength = 0
                    
                    if self.halo_detection:
                        has_halo, halo_strength = self._check_for_halo(
                            labeled == prop.label, l_channel, prop
                        )
                    
                    dark_spots.append({
                        'type': 'dark_spot',
                        'centroid': prop.centroid,
                        'area': prop.area,
                        'bbox': prop.bbox,
                        'intensity': spot_mean,
                        'contrast': (surrounding_mean - spot_mean) / surrounding_mean,
                        'has_halo': has_halo,
                        'halo_strength': halo_strength,
                        'coords': prop.coords
                    })
        
        return dark_spots
    
    def _detect_fibers(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect fiber-like structures"""
        # Use multiple color channels to detect fibers
        fibers_all = []
        
        # Process each color channel
        for i, channel_name in enumerate(['blue', 'green', 'red']):
            channel = image[:, :, i]
            
            # Enhance edges
            enhanced = cv2.bilateralFilter(channel, 9, 50, 50)
            
            # Detect edges
            edges = cv2.Canny(enhanced, 30, 90)
            
            # Use morphological operations to connect fiber segments
            kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
            connected_h = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_line, iterations=2)
            
            kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
            connected_v = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_line, iterations=2)
            
            connected = cv2.bitwise_or(connected_h, connected_v)
            
            # Skeletonize to get fiber centerlines
            skeleton = morphology.skeletonize(connected > 0)
            
            # Find connected components
            labeled = measure.label(skeleton)
            props = measure.regionprops(labeled)
            
            for prop in props:
                # Check if it's elongated (fiber-like)
                if prop.area >= self.fiber_min_length:
                    # Calculate elongation
                    if prop.minor_axis_length > 0:
                        elongation = prop.major_axis_length / prop.minor_axis_length
                    else:
                        elongation = prop.major_axis_length
                    
                    if elongation > 3:  # Fiber should be elongated
                        # Check darkness relative to surroundings
                        fiber_mask = labeled == prop.label
                        fiber_mean = np.mean(channel[fiber_mask])
                        
                        # Dilate to get surrounding area
                        dilated = cv2.dilate(fiber_mask.astype(np.uint8) * 255, 
                                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
                        surrounding_mask = (dilated > 0) & (~fiber_mask)
                        
                        if np.any(surrounding_mask):
                            surrounding_mean = np.mean(channel[surrounding_mask])
                            
                            if fiber_mean < surrounding_mean * (1 - self.color_contrast_threshold):
                                fibers_all.append({
                                    'type': 'fiber',
                                    'channel': channel_name,
                                    'centroid': prop.centroid,
                                    'area': prop.area,
                                    'length': prop.major_axis_length,
                                    'width': prop.minor_axis_length,
                                    'orientation': prop.orientation,
                                    'elongation': elongation,
                                    'bbox': prop.bbox,
                                    'coords': prop.coords
                                })
        
        # Merge overlapping fibers from different channels
        fibers = self._merge_overlapping_fibers(fibers_all)
        
        return fibers
    
    def _detect_color_anomalies(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect color anomalies that might indicate debris"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate local color statistics
        window_size = 31
        kernel = np.ones((window_size, window_size), dtype=np.float32) / (window_size ** 2)
        
        # Calculate local mean for each channel
        local_mean_h = cv2.filter2D(hsv[:, :, 0].astype(np.float32), -1, kernel)
        local_mean_s = cv2.filter2D(hsv[:, :, 1].astype(np.float32), -1, kernel)
        local_mean_v = cv2.filter2D(hsv[:, :, 2].astype(np.float32), -1, kernel)
        
        # Calculate deviations
        dev_h = np.abs(hsv[:, :, 0].astype(np.float32) - local_mean_h)
        dev_s = np.abs(hsv[:, :, 1].astype(np.float32) - local_mean_s)
        dev_v = np.abs(hsv[:, :, 2].astype(np.float32) - local_mean_v)
        
        # Normalize deviations
        dev_h = dev_h / (np.std(dev_h) + 1e-6)
        dev_s = dev_s / (np.std(dev_s) + 1e-6)
        dev_v = dev_v / (np.std(dev_v) + 1e-6)
        
        # Combined anomaly score
        anomaly_score = np.sqrt(dev_h**2 + dev_s**2 + dev_v**2)
        
        # Threshold to get anomalous regions
        anomaly_threshold = np.percentile(anomaly_score, 98)
        anomaly_mask = anomaly_score > anomaly_threshold
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        anomaly_mask = cv2.morphologyEx(anomaly_mask.astype(np.uint8) * 255,
                                       cv2.MORPH_OPEN, kernel)
        anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find regions
        labeled = measure.label(anomaly_mask > 0)
        props = measure.regionprops(labeled, intensity_image=anomaly_score)
        
        anomalies = []
        for prop in props:
            if self.min_debris_size <= prop.area <= self.max_debris_size:
                anomalies.append({
                    'type': 'color_anomaly',
                    'centroid': prop.centroid,
                    'area': prop.area,
                    'bbox': prop.bbox,
                    'anomaly_score': prop.mean_intensity,
                    'coords': prop.coords
                })
        
        return anomalies
    
    def _check_for_halo(self, debris_mask: np.ndarray, l_channel: np.ndarray, 
                       prop: Any) -> Tuple[bool, float]:
        """Check if debris has a bright halo around it"""
        # Create ring mask around debris
        dilated = cv2.dilate(debris_mask.astype(np.uint8) * 255,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        ring_mask = (dilated > 0) & (~debris_mask)
        
        if np.any(ring_mask):
            # Compare intensities
            debris_mean = prop.mean_intensity
            ring_mean = np.mean(l_channel[ring_mask])
            
            # Halo should be brighter than both debris and normal background
            if ring_mean > debris_mean * 1.3:
                # Get wider surrounding to compare with
                dilated_wide = cv2.dilate(dilated,
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
                outer_ring = (dilated_wide > 0) & (~(dilated > 0))
                
                if np.any(outer_ring):
                    outer_mean = np.mean(l_channel[outer_ring])
                    if ring_mean > outer_mean * 1.1:
                        halo_strength = (ring_mean - debris_mean) / debris_mean
                        return True, halo_strength
        
        return False, 0
    
    def _merge_overlapping_fibers(self, fibers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge fibers detected in different channels that overlap"""
        if len(fibers) <= 1:
            return fibers
        
        merged = []
        used = set()
        
        for i, fiber1 in enumerate(fibers):
            if i in used:
                continue
            
            # Check for overlaps with other fibers
            overlapping = [fiber1]
            
            for j, fiber2 in enumerate(fibers[i+1:], i+1):
                if j in used:
                    continue
                
                # Check bounding box overlap
                bbox1 = fiber1['bbox']
                bbox2 = fiber2['bbox']
                
                # Calculate intersection
                y_overlap = max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]))
                x_overlap = max(0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))
                
                if y_overlap > 0 and x_overlap > 0:
                    # Check actual pixel overlap
                    overlap_area = y_overlap * x_overlap
                    min_area = min(fiber1['area'], fiber2['area'])
                    
                    if overlap_area > 0.3 * min_area:
                        overlapping.append(fiber2)
                        used.add(j)
            
            # Merge overlapping fibers
            if len(overlapping) > 1:
                # Take the longest fiber as representative
                longest = max(overlapping, key=lambda f: f.get('length', f['area']))
                merged.append(longest)
            else:
                merged.append(fiber1)
        
        return merged
    
    def _combine_detections(self, dark_spots: List[Dict[str, Any]], 
                          fibers: List[Dict[str, Any]], 
                          anomalies: List[Dict[str, Any]], 
                          image_shape: Tuple[int, int]) -> np.ndarray:
        """Combine all debris detections into a single mask"""
        combined_mask = np.zeros(image_shape, dtype=np.uint8)
        
        # Add dark spots
        for spot in dark_spots:
            coords = spot['coords']
            combined_mask[coords[:, 0], coords[:, 1]] = 255
        
        # Add fibers
        for fiber in fibers:
            coords = fiber['coords']
            combined_mask[coords[:, 0], coords[:, 1]] = 255
        
        # Add anomalies
        for anomaly in anomalies:
            coords = anomaly['coords']
            combined_mask[coords[:, 0], coords[:, 1]] = 255
        
        return combined_mask
    
    def _analyze_debris_regions(self, combined_mask: np.ndarray, 
                              image: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze and classify debris regions"""
        labeled = measure.label(combined_mask > 0)
        props = measure.regionprops(labeled)
        
        debris_regions = []
        
        for prop in props:
            # Classify debris type based on shape and characteristics
            debris_type = self._classify_debris(prop)
            
            # Calculate color statistics
            region_mask = labeled == prop.label
            region_pixels = image[region_mask]
            
            color_mean = np.mean(region_pixels, axis=0)
            color_std = np.std(region_pixels, axis=0)
            
            debris_regions.append({
                'label': prop.label,
                'type': debris_type,
                'centroid': prop.centroid,
                'area': prop.area,
                'bbox': prop.bbox,
                'eccentricity': prop.eccentricity,
                'solidity': prop.solidity,
                'color_mean': color_mean,
                'color_std': color_std,
                'coords': prop.coords
            })
        
        return debris_regions
    
    def _classify_debris(self, prop: Any) -> str:
        """Classify debris based on shape characteristics"""
        # High eccentricity and low solidity suggests fiber
        if prop.eccentricity > 0.9 and prop.solidity < 0.5:
            return 'fiber'
        
        # Circular shape suggests particle
        if prop.eccentricity < 0.5 and prop.solidity > 0.8:
            return 'particle'
        
        # Large irregular shape
        if prop.area > 500 and prop.solidity < 0.7:
            return 'irregular_debris'
        
        # Default
        return 'debris'
    
    def _create_defect_list(self, debris_regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create detailed defect list"""
        defects = []
        
        for debris in debris_regions:
            defects.append({
                'type': 'debris',
                'subtype': debris['type'],
                'location': (int(debris['centroid'][1]), int(debris['centroid'][0])),
                'area': debris['area'],
                'bbox': debris['bbox'],
                'eccentricity': float(debris['eccentricity']),
                'solidity': float(debris['solidity']),
                'color_bgr': debris['color_mean'].tolist(),
                'color_std': debris['color_std'].tolist()
            })
        
        return defects
    
    def _create_visualization(self, original: np.ndarray, 
                            debris_regions: List[Dict[str, Any]]) -> np.ndarray:
        """Create visualization highlighting debris"""
        vis = original.copy()
        
        # Create overlay
        overlay = vis.copy()
        
        # Color code by debris type
        for debris in debris_regions:
            coords = debris['coords']
            
            if debris['type'] == 'fiber':
                overlay[coords[:, 0], coords[:, 1]] = [0, 255, 255]  # Yellow for fibers
            elif debris['type'] == 'particle':
                overlay[coords[:, 0], coords[:, 1]] = [0, 0, 255]  # Red for particles
            elif debris['type'] == 'irregular_debris':
                overlay[coords[:, 0], coords[:, 1]] = [255, 0, 255]  # Magenta for irregular
            else:
                overlay[coords[:, 0], coords[:, 1]] = [255, 0, 0]  # Blue for other
        
        # Blend with original
        result = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
        
        # Draw bounding boxes for clarity
        for debris in debris_regions:
            minr, minc, maxr, maxc = debris['bbox']
            color = [0, 255, 0]  # Green boxes
            cv2.rectangle(result, (minc, minr), (maxc, maxr), color, 1)
        
        return result 