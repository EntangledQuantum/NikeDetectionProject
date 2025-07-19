"""
Smudge Defect Detection Algorithm
Detects post-print smudges like fingerprints and contact marks

Author: Koushik and Assistant
Date: 2024
Version: 1.0 - Robust detection using multiple techniques
"""

import cv2
import numpy as np
from scipy import ndimage, signal
from skimage import morphology, measure, filters, feature, transform
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any


class SmudgeDetector:
    """
    Detects smudge defects characterized by:
    - Fingerprint-like patterns
    - Directional smearing of ink
    - Post-print contact marks
    """
    
    def __init__(self, 
                 ridge_wavelength: Tuple[float, float] = (5.0, 15.0),
                 orientation_threshold: float = 0.3,
                 coherence_threshold: float = 0.4,
                 min_smudge_area: int = 500,
                 texture_window: int = 32):
        """
        Args:
            ridge_wavelength: Expected wavelength range for fingerprint ridges
            orientation_threshold: Threshold for orientation consistency
            coherence_threshold: Threshold for pattern coherence
            min_smudge_area: Minimum area to consider as smudge
            texture_window: Window size for texture analysis
        """
        self.ridge_wavelength = ridge_wavelength
        self.orientation_threshold = orientation_threshold
        self.coherence_threshold = coherence_threshold
        self.min_smudge_area = min_smudge_area
        self.texture_window = texture_window
    
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
        
        # Detect smudges using multiple methods
        fingerprint_mask = self._detect_fingerprint_patterns(gray)
        directional_mask = self._detect_directional_smearing(gray)
        texture_mask = self._detect_texture_disruption(gray)
        
        # Combine masks intelligently
        combined_mask = self._combine_masks(fingerprint_mask, directional_mask, texture_mask)
        
        # Analyze smudge characteristics
        smudge_regions = self._analyze_smudge_regions(combined_mask, gray)
        
        # Create defects list
        defects = self._create_defect_list(smudge_regions)
        
        # Create visualization
        visualization = self._create_visualization(image, combined_mask, smudge_regions)
        
        return visualization, defects
    
    def _detect_fingerprint_patterns(self, gray: np.ndarray) -> np.ndarray:
        """Detect fingerprint-like ridge patterns"""
        # Enhance ridges using frequency domain filtering
        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        
        # Create band-pass filter for ridge frequencies
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create frequency mask
        mask = np.zeros((rows, cols), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                dist = np.sqrt((r - crow)**2 + (c - ccol)**2)
                # Band-pass filter for expected ridge frequencies
                if self.ridge_wavelength[0] < dist < self.ridge_wavelength[1] * 2:
                    mask[r, c] = 1
        
        # Apply filter
        f_shift_filtered = f_shift * mask
        f_inverse_shift = np.fft.ifftshift(f_shift_filtered)
        filtered = np.fft.ifft2(f_inverse_shift)
        filtered = np.abs(filtered)
        
        # Normalize
        filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Detect ridge-like structures using Gabor filters
        gabor_responses = []
        for theta in np.linspace(0, np.pi, 8):
            for frequency in np.linspace(0.05, 0.25, 4):
                kernel = cv2.getGaborKernel((31, 31), 4.0, theta, 10.0, frequency, 0.5)
                response = cv2.filter2D(filtered, cv2.CV_32F, kernel)
                gabor_responses.append(np.abs(response))
        
        # Combine Gabor responses
        gabor_max = np.max(gabor_responses, axis=0)
        
        # Threshold to get ridge patterns
        ridge_threshold = np.percentile(gabor_max, 85)
        fingerprint_mask = gabor_max > ridge_threshold
        
        # Morphological operations to connect ridges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fingerprint_mask = cv2.morphologyEx(fingerprint_mask.astype(np.uint8) * 255,
                                          cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return fingerprint_mask
    
    def _detect_directional_smearing(self, gray: np.ndarray) -> np.ndarray:
        """Detect directional smearing patterns"""
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        
        # Calculate gradient magnitude and orientation
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        orientation = np.arctan2(grad_y, grad_x)
        
        # Analyze local orientation coherence
        window = self.texture_window
        h, w = gray.shape
        coherence_map = np.zeros_like(gray, dtype=np.float32)
        
        for y in range(0, h - window, window // 2):
            for x in range(0, w - window, window // 2):
                # Extract local window
                mag_roi = magnitude[y:y+window, x:x+window]
                ori_roi = orientation[y:y+window, x:x+window]
                
                if mag_roi.size > 0:
                    # Calculate orientation coherence
                    # Convert orientations to unit vectors
                    vx = np.cos(ori_roi) * mag_roi
                    vy = np.sin(ori_roi) * mag_roi
                    
                    # Average direction
                    avg_vx = np.mean(vx)
                    avg_vy = np.mean(vy)
                    avg_magnitude = np.sqrt(avg_vx**2 + avg_vy**2)
                    
                    # Coherence is ratio of average magnitude to average of magnitudes
                    avg_individual_mag = np.mean(mag_roi)
                    if avg_individual_mag > 0:
                        coherence = avg_magnitude / avg_individual_mag
                    else:
                        coherence = 0
                    
                    coherence_map[y:y+window, x:x+window] = coherence
        
        # Find regions with high directional coherence
        directional_mask = coherence_map > self.coherence_threshold
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        directional_mask = cv2.morphologyEx(directional_mask.astype(np.uint8) * 255,
                                          cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return directional_mask
    
    def _detect_texture_disruption(self, gray: np.ndarray) -> np.ndarray:
        """Detect areas where normal texture is disrupted"""
        # Calculate local texture features using LBP (Local Binary Patterns)
        # Simple LBP implementation
        h, w = gray.shape
        lbp = np.zeros_like(gray, dtype=np.uint8)
        
        for y in range(1, h-1):
            for x in range(1, w-1):
                center = gray[y, x]
                code = 0
                # 8-neighborhood
                neighbors = [
                    gray[y-1, x-1], gray[y-1, x], gray[y-1, x+1],
                    gray[y, x+1], gray[y+1, x+1], gray[y+1, x],
                    gray[y+1, x-1], gray[y, x-1]
                ]
                for i, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << i)
                lbp[y, x] = code
        
        # Calculate LBP histogram in local windows
        window = self.texture_window
        texture_anomaly = np.zeros_like(gray, dtype=np.float32)
        
        for y in range(0, h - window, window // 2):
            for x in range(0, w - window, window // 2):
                roi = lbp[y:y+window, x:x+window]
                if roi.size > 0:
                    # Calculate histogram
                    hist, _ = np.histogram(roi, bins=256, range=(0, 256))
                    hist = hist.astype(np.float32) / roi.size
                    
                    # Calculate entropy as measure of texture complexity
                    hist_nonzero = hist[hist > 0]
                    if len(hist_nonzero) > 0:
                        entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
                    else:
                        entropy = 0
                    
                    texture_anomaly[y:y+window, x:x+window] = entropy
        
        # Normalize and find anomalies
        texture_anomaly = cv2.normalize(texture_anomaly, None, 0, 1, cv2.NORM_MINMAX)
        
        # Smudges typically have different texture entropy than surroundings
        mean_entropy = np.mean(texture_anomaly)
        std_entropy = np.std(texture_anomaly)
        
        # Detect both low and high entropy regions (smudges can be either)
        anomaly_mask = np.logical_or(
            texture_anomaly < mean_entropy - 1.5 * std_entropy,
            texture_anomaly > mean_entropy + 1.5 * std_entropy
        )
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        texture_mask = cv2.morphologyEx(anomaly_mask.astype(np.uint8) * 255,
                                      cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return texture_mask
    
    def _combine_masks(self, fingerprint: np.ndarray, directional: np.ndarray,
                      texture: np.ndarray) -> np.ndarray:
        """Intelligently combine detection masks"""
        # Convert to binary
        fp_bin = fingerprint > 0
        dir_bin = directional > 0
        tex_bin = texture > 0
        
        # Smudges should be detected by at least 2 methods
        combined = (fp_bin.astype(int) + dir_bin.astype(int) + tex_bin.astype(int)) >= 2
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        combined = cv2.morphologyEx(combined.astype(np.uint8) * 255,
                                   cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Fill holes
        combined_filled = ndimage.binary_fill_holes(combined > 0)
        
        # Remove small regions
        combined_clean = morphology.remove_small_objects(combined_filled, self.min_smudge_area)
        
        return combined_clean.astype(np.uint8) * 255
    
    def _analyze_smudge_regions(self, mask: np.ndarray, gray: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze characteristics of detected smudge regions"""
        labeled = measure.label(mask > 0)
        props = measure.regionprops(labeled, intensity_image=gray)
        
        smudge_regions = []
        
        for prop in props:
            if prop.area >= self.min_smudge_area:
                # Analyze smudge characteristics
                # Check for fingerprint-like patterns
                roi_mask = labeled == prop.label
                roi_gray = gray * roi_mask
                
                # Calculate ridge orientation if present
                grad_x = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
                orientation = np.arctan2(grad_y, grad_x)
                
                # Calculate dominant orientation
                orientation_hist, _ = np.histogram(orientation[roi_mask], bins=36, range=(-np.pi, np.pi))
                dominant_orientation = np.argmax(orientation_hist) * (2 * np.pi / 36) - np.pi
                
                smudge_regions.append({
                    'label': prop.label,
                    'centroid': prop.centroid,
                    'area': prop.area,
                    'bbox': prop.bbox,
                    'eccentricity': prop.eccentricity,
                    'orientation': dominant_orientation,
                    'intensity_std': np.std(gray[roi_mask]),
                    'type': self._classify_smudge_type(prop, orientation_hist)
                })
        
        return smudge_regions
    
    def _classify_smudge_type(self, prop: Any, orientation_hist: np.ndarray) -> str:
        """Classify the type of smudge based on characteristics"""
        # High eccentricity suggests directional smearing
        if prop.eccentricity > 0.8:
            return 'directional_smear'
        
        # Multiple orientation peaks suggest fingerprint
        peaks = signal.find_peaks(orientation_hist, height=np.max(orientation_hist) * 0.3)[0]
        if len(peaks) >= 3:
            return 'fingerprint'
        
        # Default to contact mark
        return 'contact_mark'
    
    def _create_defect_list(self, smudge_regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create detailed defect list"""
        defects = []
        
        for region in smudge_regions:
            defects.append({
                'type': 'smudge',
                'subtype': region['type'],
                'location': (int(region['centroid'][1]), int(region['centroid'][0])),
                'area': region['area'],
                'bbox': region['bbox'],
                'eccentricity': region['eccentricity'],
                'orientation': float(region['orientation']),
                'intensity_variation': float(region['intensity_std'])
            })
        
        return defects
    
    def _create_visualization(self, original: np.ndarray, mask: np.ndarray,
                            smudge_regions: List[Dict[str, Any]]) -> np.ndarray:
        """Create visualization highlighting smudge defects"""
        if len(original.shape) == 2:
            vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            vis = original.copy()
        
        # Create overlay
        overlay = vis.copy()
        
        # Color code by smudge type
        labeled = measure.label(mask > 0)
        
        for region in smudge_regions:
            region_mask = labeled == region['label']
            
            # Different colors for different types
            if region['type'] == 'fingerprint':
                overlay[region_mask] = [255, 0, 0]  # Blue for fingerprints
            elif region['type'] == 'directional_smear':
                overlay[region_mask] = [0, 255, 255]  # Yellow for smears
            else:
                overlay[region_mask] = [255, 0, 255]  # Magenta for contact marks
        
        # Blend with original
        result = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
        
        return result 