"""
Optimized Combined Detector

This module contains the optimized detector that combines operations from multiple
detectors to minimize redundant preprocessing and maximize shared computations.

Author: Refactored Architecture
Date: 2024
"""

import time
import cv2
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

from .image_processing import SharedPreprocessor
from .data_models import DetectionResult


class OptimizedCombinedDetector:
    """
    Optimized detector that combines operations from multiple detectors
    to minimize redundant preprocessing and maximize shared computations
    """
    
    def __init__(self, detector_configs: Dict[str, Any], verbose_timing: bool = True):
        """
        Args:
            detector_configs: Dictionary of detector names and their configurations
            verbose_timing: Whether to log detailed timing information
        """
        self.detector_configs = detector_configs
        self.verbose_timing = verbose_timing
        self.timing_data = {}
        self.preprocessor = SharedPreprocessor(verbose_timing)
    
    def detect_all(self, image: np.ndarray) -> Dict[str, Tuple[Optional[np.ndarray], List[Dict[str, Any]], Dict[str, float]]]:
        """
        Run all detectors with optimized preprocessing
        
        Args:
            image: Input BGR image
            
        Returns:
            Dict of detector_name: (visualization, defects, timing_info)
        """
        total_start_time = time.time()
        
        # Step 1: Shared preprocessing (done once for all detectors)
        preprocessing_start = time.time()
        shared_data = self.preprocessor.process_image(image)
        preprocessing_time = time.time() - preprocessing_start
        
        if self.verbose_timing:
            print(f"      📊 Shared preprocessing: {preprocessing_time:.3f}s")
        
        results = {}
        
        # Step 2: Run each detector with shared data
        for detector_name in self.detector_configs.keys():
            detector_start = time.time()
            
            if self.verbose_timing:
                print(f"      🔍 Running {detector_name} detection...")
            
            try:
                visualization, defects, detector_timing = self._run_single_detector(
                    detector_name, shared_data
                )
                
                detector_total_time = time.time() - detector_start
                
                # Combine timing information
                timing_info = {
                    'total_time': detector_total_time,
                    'preprocessing_shared': preprocessing_time / len(self.detector_configs),  # Amortized
                    **detector_timing
                }
                
                results[detector_name] = (visualization, defects, timing_info)
                
                if self.verbose_timing:
                    print(f"        ✅ {detector_name}: {len(defects)} defects in {detector_total_time:.3f}s")
                    for operation, op_time in detector_timing.items():
                        if op_time > 0.01:  # Only show significant operations
                            print(f"           - {operation}: {op_time:.3f}s")
                
            except Exception as e:
                if self.verbose_timing:
                    print(f"        ❌ {detector_name} failed: {str(e)}")
                results[detector_name] = (None, [], {'total_time': 0, 'error': str(e)})
        
        total_time = time.time() - total_start_time
        if self.verbose_timing:
            print(f"      📈 Total combined detection: {total_time:.3f}s")
        
        return results
    
    def _run_single_detector(self, detector_name: str, shared_data: Dict[str, Any]) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]], Dict[str, float]]:
        """
        Run a single detector using shared preprocessing data
        """
        timing = {}
        
        if detector_name == 'debris':
            return self._run_debris_detection(shared_data, timing)
        elif detector_name == 'smudge':
            return self._run_smudge_detection(shared_data, timing)
        elif detector_name == 'void':
            return self._run_void_detection(shared_data, timing)
        elif detector_name == 'head_calibration':
            return self._run_head_calibration_detection(shared_data, timing)
        elif detector_name == 'surface_treatment':
            return self._run_surface_treatment_detection(shared_data, timing)
        else:
            raise ValueError(f"Unknown detector: {detector_name}")
    
    def _run_debris_detection(self, shared_data: Dict[str, Any], timing: Dict[str, float]) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]], Dict[str, float]]:
        """Optimized debris detection using shared preprocessing"""
        config = self.detector_configs['debris']
        
        # Use shared denoised and enhanced images
        enhanced = shared_data['enhanced']
        original = shared_data['original_bgr']
        
        # Use the actual debris detector
        detector_start = time.time()
        try:
            defects = config.detect_defects(enhanced)
            # Clean the defects data
            cleaned_defects = self._clean_defect_data(defects)
            
            # Create simple visualization
            visualization = original.copy()
            for defect in cleaned_defects:
                if 'location' in defect:
                    center = tuple(map(int, defect['location']))
                    cv2.circle(visualization, center, 10, (0, 255, 255), 2)  # Yellow circles
            
            timing['detection'] = time.time() - detector_start
            return visualization, cleaned_defects, timing
            
        except Exception as e:
            timing['detection'] = time.time() - detector_start
            # Don't add error to timing dict as it expects floats
            return None, [], timing
    
    def _run_smudge_detection(self, shared_data: Dict[str, Any], timing: Dict[str, float]) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]], Dict[str, float]]:
        """Optimized smudge detection using shared preprocessing"""
        config = self.detector_configs['smudge']
        
        gray = shared_data['gray']
        original = shared_data['original_bgr']
        
        detector_start = time.time()
        try:
            defects = config.detect_defects(gray)
            cleaned_defects = self._clean_defect_data(defects)
            
            # Create simple visualization
            visualization = original.copy()
            for defect in cleaned_defects:
                if 'location' in defect:
                    center = tuple(map(int, defect['location']))
                    cv2.circle(visualization, center, 15, (255, 0, 255), 2)  # Magenta circles
            
            timing['detection'] = time.time() - detector_start
            return visualization, cleaned_defects, timing
            
        except Exception as e:
            timing['detection'] = time.time() - detector_start
            return None, [], timing
    
    def _run_void_detection(self, shared_data: Dict[str, Any], timing: Dict[str, float]) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]], Dict[str, float]]:
        """Optimized void detection using shared preprocessing"""
        config = self.detector_configs['void']
        
        gray = shared_data['gray']
        original = shared_data['original_bgr']
        
        detector_start = time.time()
        try:
            defects = config.detect_defects(gray)
            cleaned_defects = self._clean_defect_data(defects)
            
            # Create simple visualization
            visualization = original.copy()
            for defect in cleaned_defects:
                if 'location' in defect:
                    center = tuple(map(int, defect['location']))
                    cv2.circle(visualization, center, 10, (0, 0, 255), 2)  # Red circles
            
            timing['detection'] = time.time() - detector_start
            return visualization, cleaned_defects, timing
            
        except Exception as e:
            timing['detection'] = time.time() - detector_start
            return None, [], timing
    
    def _run_head_calibration_detection(self, shared_data: Dict[str, Any], timing: Dict[str, float]) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]], Dict[str, float]]:
        """Optimized head calibration detection using shared preprocessing"""
        config = self.detector_configs['head_calibration']
        
        gray = shared_data['gray']
        original = shared_data['original_bgr']
        
        detector_start = time.time()
        try:
            defects = config.detect_defects(gray)
            cleaned_defects = self._clean_defect_data(defects)
            
            # Create simple visualization
            visualization = original.copy()
            for defect in cleaned_defects:
                if 'location' in defect:
                    center = tuple(map(int, defect['location']))
                    cv2.circle(visualization, center, 12, (255, 255, 0), 2)  # Cyan circles
            
            timing['detection'] = time.time() - detector_start
            return visualization, cleaned_defects, timing
            
        except Exception as e:
            timing['detection'] = time.time() - detector_start
            return None, [], timing
    
    def _run_surface_treatment_detection(self, shared_data: Dict[str, Any], timing: Dict[str, float]) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]], Dict[str, float]]:
        """Optimized surface treatment detection using shared preprocessing"""
        config = self.detector_configs['surface_treatment']
        
        original = shared_data['original_bgr']
        
        detector_start = time.time()
        try:
            defects = config.detect_defects(original)
            cleaned_defects = self._clean_defect_data(defects)
            
            # Create simple visualization
            visualization = original.copy()
            for defect in cleaned_defects:
                if 'location' in defect:
                    center = tuple(map(int, defect['location']))
                    cv2.circle(visualization, center, 15, (0, 255, 0), 2)  # Green circles
            
            timing['detection'] = time.time() - detector_start
            return visualization, cleaned_defects, timing
            
        except Exception as e:
            timing['detection'] = time.time() - detector_start
            return None, [], timing
    
    def _clean_defect_data(self, defects: List[Dict]) -> List[Dict]:
        """Clean defect data for JSON serialization"""
        if not defects:
            return []
        
        cleaned_defects = []
        for defect in defects:
            cleaned_defect = {}
            for key, value in defect.items():
                if isinstance(value, (np.ndarray, tuple)):
                    if isinstance(value, np.ndarray):
                        cleaned_defect[key] = value.tolist()
                    else:
                        cleaned_defect[key] = list(value)
                elif isinstance(value, (np.integer, np.floating)):
                    cleaned_defect[key] = float(value) if isinstance(value, np.floating) else int(value)
                else:
                    cleaned_defect[key] = value
            cleaned_defects.append(cleaned_defect)
        return cleaned_defects 