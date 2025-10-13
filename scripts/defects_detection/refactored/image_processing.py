"""
Image Processing Utilities for Defect Detection Pipeline

This module handles all image loading, preprocessing, and classification
operations with optimized shared processing capabilities.

Author: Refactored Architecture
Date: 2024
"""

import os
import cv2
import numpy as np
from typing import Tuple, Dict, Any
from .data_models import ImageType


class ImageTypeClassifier:
    """Classifies images based on filename patterns"""
    
    @staticmethod
    def classify_image(image_path: str) -> ImageType:
        """Classify image type based on filename"""
        filename = os.path.basename(image_path).lower()
        
        if 'stripe' in filename:
            return ImageType.STRIPE
        elif 'island' in filename:
            return ImageType.ISLAND
        else:
            return ImageType.UNKNOWN


class TiffImageLoader:
    """Handles efficient loading of large TIFF files"""
    
    @staticmethod
    def load_tiff_image(image_path: str) -> np.ndarray:
        """Load TIFF image efficiently, handling large files"""
        try:
            # Try using tifffile for better TIFF support
            import tifffile
            image = tifffile.imread(image_path)
            
            # Convert to BGR format if needed
            if len(image.shape) == 2:
                # Grayscale to BGR
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                # Single channel to BGR
                image = cv2.cvtColor(image.squeeze(), cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                # RGB to BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                # RGBA to BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            
            return image
            
        except ImportError:
            print("    tifffile not available, using OpenCV...")
            # Fallback to OpenCV
            return cv2.imread(image_path)
        except Exception as e:
            print(f"    Error loading with tifffile: {e}, trying OpenCV...")
            # Fallback to OpenCV
            return cv2.imread(image_path)
    
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """Load any image format efficiently"""
        if image_path.lower().endswith(('.tif', '.tiff')):
            return TiffImageLoader.load_tiff_image(image_path)
        else:
            return cv2.imread(image_path)


class ImagePreprocessor:
    """Centralized image preprocessing with common operations"""
    
    @staticmethod
    def load_and_convert_to_grayscale(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load image and convert to grayscale if needed"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        return image, gray
    
    @staticmethod
    def apply_noise_reduction(gray: np.ndarray, method: str = 'bilateral') -> np.ndarray:
        """Apply noise reduction based on method"""
        if method == 'bilateral':
            return cv2.bilateralFilter(gray, 9, 75, 75)
        elif method == 'median':
            return cv2.medianBlur(gray, 5)
        elif method == 'gaussian':
            return cv2.GaussianBlur(gray, (5, 5), 0)
        else:
            return gray
    
    @staticmethod
    def enhance_contrast(gray: np.ndarray, clip_limit: float = 2.0, 
                        grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """Apply CLAHE contrast enhancement"""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        return clahe.apply(gray)
    
    @classmethod
    def preprocess_for_surface_treatment(cls, image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preprocessing pipeline for surface treatment detection"""
        original, gray = cls.load_and_convert_to_grayscale(image_path)
        enhanced = cls.enhance_contrast(gray, clip_limit=2.0)
        return original, gray, enhanced
    
    @classmethod
    def preprocess_for_debris(cls, image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Preprocessing pipeline for debris detection"""
        original, gray = cls.load_and_convert_to_grayscale(image_path)
        denoised = cls.apply_noise_reduction(gray, 'median')
        enhanced = cls.enhance_contrast(denoised, clip_limit=3.0)
        return original, gray, denoised, enhanced


class SharedPreprocessor:
    """
    Optimized preprocessor that performs shared operations once
    to minimize redundant preprocessing across multiple detectors
    """
    
    def __init__(self, verbose_timing: bool = True):
        self.verbose_timing = verbose_timing
    
    def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Perform shared preprocessing operations that multiple detectors need
        
        Args:
            image: Input BGR image
            
        Returns:
            Dictionary containing all preprocessed versions of the image
        """
        import time
        
        shared_data = {
            'original_bgr': image.copy(),
            'timings': {}
        }
        
        # Convert to grayscale (needed by most detectors)
        gray_start = time.time()
        if len(image.shape) == 3:
            shared_data['gray'] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            shared_data['gray'] = image.copy()
        shared_data['timings']['grayscale_conversion'] = time.time() - gray_start
        
        # Apply noise reduction (used by debris and void detectors)
        denoise_start = time.time()
        shared_data['denoised'] = cv2.medianBlur(shared_data['gray'], 5)
        shared_data['timings']['noise_reduction'] = time.time() - denoise_start
        
        # Apply bilateral filter (used by head calibration and surface treatment)
        bilateral_start = time.time()
        shared_data['bilateral'] = cv2.bilateralFilter(shared_data['gray'], 9, 75, 75)
        shared_data['timings']['bilateral_filter'] = time.time() - bilateral_start
        
        # Enhance contrast (used by debris and surface treatment)
        contrast_start = time.time()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        shared_data['enhanced'] = clahe.apply(shared_data['gray'])
        shared_data['timings']['contrast_enhancement'] = time.time() - contrast_start
        
        # Compute gradients (used by head calibration and smudge detectors)
        gradient_start = time.time()
        shared_data['grad_x'] = cv2.Sobel(shared_data['bilateral'], -1, 1, 0, ksize=5)
        shared_data['grad_y'] = cv2.Sobel(shared_data['bilateral'], -1, 0, 1, ksize=5)
        shared_data['gradient_magnitude'] = np.sqrt(
            np.square(shared_data['grad_x']) + np.square(shared_data['grad_y'])
        )
        shared_data['timings']['gradient_computation'] = time.time() - gradient_start
        
        # HSV conversion (used by surface treatment detector)
        hsv_start = time.time()
        shared_data['hsv'] = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        shared_data['timings']['hsv_conversion'] = time.time() - hsv_start
        
        return shared_data 