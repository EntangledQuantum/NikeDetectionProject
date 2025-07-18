"""
Base Detector Class
Provides standard interface for all defect detectors

Author: Assistant
Date: 2024
"""

import cv2
import numpy as np


class BaseDetector:
    """Base class for all defect detectors"""
    
    def detect(self, image):
        """
        Detect defects in the image
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            tuple: (visualization_image, defect_list)
        """
        raise NotImplementedError("Subclasses must implement detect method")
    
    def _standardize_output(self, result):
        """
        Standardize detector output to consistent format
        
        Args:
            result: Raw detector output (dict or other format)
            
        Returns:
            tuple: (visualization_image, defect_list)
        """
        if isinstance(result, dict):
            visualization = result.get('visualization')
            defects = result.get('defects', [])
            return visualization, defects
        elif isinstance(result, tuple) and len(result) == 2:
            return result
        else:
            raise ValueError(f"Unexpected detector output format: {type(result)}")
    
    def detect_wrapper(self, image):
        """
        Wrapper that ensures consistent output format
        
        Args:
            image: Input image
            
        Returns:
            tuple: (visualization_image, defect_list)
        """
        result = self._detect_impl(image)
        return self._standardize_output(result)
    
    def _detect_impl(self, image):
        """Implementation of detection logic - to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement _detect_impl method") 