"""
Banding Detection Algorithm
Detects horizontal and vertical banding patterns in printed images

Author: Assistant
Date: 2024
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fft
from skimage import filters, measure
import os
from tqdm import tqdm


class BandingDetector:
    """Detects banding defects - periodic horizontal/vertical patterns"""
    
    def __init__(self, min_band_strength=0.15, band_width_range=(5, 50),
                 frequency_threshold=0.1):
        """
        Args:
            min_band_strength: Minimum strength for band detection
            band_width_range: Expected band width range in pixels
            frequency_threshold: Threshold for frequency domain detection
        """
        self.min_band_strength = min_band_strength
        self.band_width_range = band_width_range
        self.frequency_threshold = frequency_threshold
        
    def detect(self, image):
        """
        Detect banding patterns in the image
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            result_image: Visualization of detected bands
            defects: List of detected banding defects
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        h, w = gray.shape
        
        # Detect horizontal and vertical banding separately
        h_bands = self._detect_horizontal_bands(gray)
        v_bands = self._detect_vertical_bands(gray)
        
        # Combine results
        defects = []
        
        for band in h_bands:
            defects.append({
                'type': 'horizontal_banding',
                'position': band['position'],
                'strength': band['strength'],
                'width': band['width'],
                'frequency': band.get('frequency', None)
            })
            
        for band in v_bands:
            defects.append({
                'type': 'vertical_banding',
                'position': band['position'],
                'strength': band['strength'],
                'width': band['width'],
                'frequency': band.get('frequency', None)
            })
            
        # Create visualization
        result_image = self.visualize_detections(image, defects)
        
        return result_image, defects
    
    def _detect_horizontal_bands(self, gray):
        """Detect horizontal banding patterns"""
        h, w = gray.shape
        bands = []
        
        # Method 1: Projection profile analysis
        horizontal_profile = np.mean(gray, axis=1)
        
        # Smooth the profile
        kernel_size = max(3, min(21, h // 50))
        if kernel_size % 2 == 0:
            kernel_size += 1
        smooth_profile = cv2.GaussianBlur(horizontal_profile.reshape(-1, 1), 
                                         (1, kernel_size), 0).flatten()
        
        # Find periodic patterns using autocorrelation
        autocorr = np.correlate(smooth_profile - np.mean(smooth_profile), 
                               smooth_profile - np.mean(smooth_profile), 
                               mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        # Find peaks in autocorrelation
        peaks, properties = signal.find_peaks(autocorr, 
                                            height=self.min_band_strength,
                                            distance=self.band_width_range[0])
        
        # Method 2: FFT analysis
        fft_result = np.fft.fft(horizontal_profile)
        frequencies = np.fft.fftfreq(len(horizontal_profile))
        magnitudes = np.abs(fft_result)
        
        # Find dominant frequencies (excluding DC component)
        freq_peaks, _ = signal.find_peaks(magnitudes[1:len(magnitudes)//2], 
                                         height=np.max(magnitudes[1:]) * self.frequency_threshold)
        
        # Combine results from both methods
        if len(peaks) > 0:
            # Estimate band period from autocorrelation
            band_period = peaks[0] if len(peaks) > 0 else None
            
            if band_period and self.band_width_range[0] <= band_period <= self.band_width_range[1]:
                # Find actual band positions
                gradient = np.gradient(smooth_profile)
                band_edges = signal.find_peaks(np.abs(gradient), 
                                             height=np.std(gradient))[0]
                
                for i in range(0, len(band_edges) - 1, 2):
                    if i + 1 < len(band_edges):
                        band_pos = (band_edges[i] + band_edges[i+1]) // 2
                        band_width = band_edges[i+1] - band_edges[i]
                        
                        if self.band_width_range[0] <= band_width <= self.band_width_range[1]:
                            bands.append({
                                'position': band_pos,
                                'strength': autocorr[peaks[0]] if len(peaks) > 0 else 0,
                                'width': band_width,
                                'frequency': 1.0 / band_period if band_period else None
                            })
        
        return bands
    
    def _detect_vertical_bands(self, gray):
        """Detect vertical banding patterns"""
        # Transpose and use horizontal detection
        transposed = gray.T
        h_bands = self._detect_horizontal_bands(transposed)
        
        # Convert back to vertical coordinates
        v_bands = []
        for band in h_bands:
            v_bands.append({
                'position': band['position'],
                'strength': band['strength'],
                'width': band['width'],
                'frequency': band.get('frequency', None)
            })
            
        return v_bands
    
    def visualize_detections(self, image, defects):
        """Visualize detected banding patterns"""
        result = image.copy()
        
        # Create overlay for bands
        overlay = np.zeros_like(result)
        
        for defect in defects:
            if defect['type'] == 'horizontal_banding':
                # Draw horizontal band
                y = defect['position']
                width = defect['width']
                cv2.rectangle(overlay, 
                            (0, y - width//2), 
                            (image.shape[1], y + width//2),
                            (0, 0, 255), 2)
                
                # Add label
                cv2.putText(overlay, 
                          f"H-Band: {defect['strength']:.2f}",
                          (10, y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                          
            elif defect['type'] == 'vertical_banding':
                # Draw vertical band
                x = defect['position']
                width = defect['width']
                cv2.rectangle(overlay, 
                            (x - width//2, 0), 
                            (x + width//2, image.shape[0]),
                            (255, 0, 0), 2)
                
                # Add label with rotation
                cv2.putText(overlay, 
                          f"V-Band: {defect['strength']:.2f}",
                          (x, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Blend with original
        alpha = 0.7
        result = cv2.addWeighted(result, 1, overlay, alpha, 0)
        
        return result
    
    def process_folder(self, input_folder, output_folder=None):
        """Process all images in a folder"""
        if output_folder is None:
            output_folder = os.path.join(input_folder, 'output', 'banding')
            
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
        image_files = [f for f in os.listdir(input_folder) 
                      if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        results = {}
        
        for img_file in tqdm(image_files, desc="Detecting banding"):
            img_path = os.path.join(input_folder, img_file)
            image = cv2.imread(img_path)
            
            if image is None:
                continue
                
            # Detect banding
            result_img, defects = self.detect(image)
            
            # Save result
            output_path = os.path.join(output_folder, f'banding_{img_file}')
            cv2.imwrite(output_path, result_img)
            
            results[img_file] = {
                'defects': defects,
                'count': len(defects),
                'output_path': output_path
            }
            
        return results 