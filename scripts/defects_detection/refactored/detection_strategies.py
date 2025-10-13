"""
Detection Strategies Module

This module implements the Strategy pattern for different detection
approaches based on image types and user selections.

Author: Refactored Architecture
Date: 2024
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

# Import the existing detector modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from surface_treatment_detection import SurfaceTreatmentDetector
from debris_detection import DebrisDetector
from smudge_detection import SmudgeDetector
from void_detection import VoidDetector
from head_calibration_detection import HeadCalibrationDetector

from .data_models import ImageType


class DetectorFactory:
    """Factory for creating appropriate detectors based on sensitivity"""
    
    @staticmethod
    def create_surface_treatment_detector(sensitivity: str) -> SurfaceTreatmentDetector:
        """Create surface treatment detector with appropriate settings"""
        if sensitivity == 'low':
            return SurfaceTreatmentDetector(
                density_threshold=0.2,
                band_detection_sensitivity=0.3,
                head_comparison_threshold=0.15,
                min_defect_area_ratio=0.4
            )
        elif sensitivity == 'high':
            return SurfaceTreatmentDetector(
                density_threshold=0.1,
                band_detection_sensitivity=0.15,
                head_comparison_threshold=0.05,
                min_defect_area_ratio=0.2
            )
        else:  # medium
            return SurfaceTreatmentDetector()  # Use defaults
    
    @staticmethod
    def create_debris_detector(sensitivity: str) -> DebrisDetector:
        """Create debris detector with appropriate settings"""
        if sensitivity == 'low':
            return DebrisDetector(
                dark_threshold=0.5,  # More conservative
                min_debris_size=100,  # Larger minimum size
                max_debris_size=1000,  # Smaller maximum size
                fiber_min_length=80
            )
        elif sensitivity == 'high':
            return DebrisDetector(
                dark_threshold=0.15,
                min_debris_size=8,
                max_debris_size=4000,
                fiber_min_length=15
            )
        else:  # medium - more conservative defaults
            return DebrisDetector(
                dark_threshold=0.35,  # More conservative than default
                min_debris_size=50,   # Larger than default
                max_debris_size=2000,
                fiber_min_length=40   # Longer than default
            )
    
    @staticmethod
    def create_smudge_detector(sensitivity: str) -> SmudgeDetector:
        """Create smudge detector with appropriate settings"""
        if sensitivity == 'low':
            return SmudgeDetector(
                min_smudge_size=200000,  # 450x450 pixels - very conservative
                background_window_size=120,
                lightness_threshold_factor=2.2,  # More conservative
                consistency_threshold=0.2,  # Stricter consistency requirement
                morphology_kernel_size=20,
                use_fft_analysis=True  # Use fastest method for production
            )
        elif sensitivity == 'high':
            return SmudgeDetector(
                min_smudge_size=100000,  # 316x316 pixels - more sensitive
                background_window_size=80,
                lightness_threshold_factor=1.4,  # More sensitive
                consistency_threshold=0.1,  # Less strict consistency
                morphology_kernel_size=10,
                use_fft_analysis=True  # Use fastest method
            )
        else:  # medium - balanced defaults
            return SmudgeDetector(
                use_fft_analysis=True  # Use fastest method by default
            )  # Use the new defaults (400x400 = 160000 pixels)
    
    @staticmethod
    def create_void_detector(sensitivity: str) -> VoidDetector:
        """Create void detector with appropriate settings"""
        if sensitivity == 'low':
            return VoidDetector(
                min_void_size=50,         # Larger minimum
                max_void_size=200,        # Smaller maximum
                circularity_threshold=0.8, # More strict
                contrast_threshold=0.5    # Higher contrast required
            )
        elif sensitivity == 'high':
            return VoidDetector(
                min_void_size=3,
                max_void_size=1000,
                circularity_threshold=0.4,
                contrast_threshold=0.15
            )
        else:  # medium - more conservative defaults
            return VoidDetector(
                min_void_size=15,         # Larger than default
                max_void_size=400,        # Smaller than default
                circularity_threshold=0.65, # More strict
                contrast_threshold=0.35   # Higher contrast required
            )
    
    @staticmethod
    def create_head_calibration_detector(sensitivity: str) -> HeadCalibrationDetector:
        """Create head calibration detector with appropriate settings"""
        if sensitivity == 'low':
            return HeadCalibrationDetector(
                edge_threshold=0.8,
                alignment_tolerance=8,
                min_edge_length=150
            )
        elif sensitivity == 'high':
            return HeadCalibrationDetector(
                edge_threshold=0.6,
                alignment_tolerance=3,
                min_edge_length=50
            )
        else:  # medium
            return HeadCalibrationDetector()  # Use defaults


class DetectionStrategy(ABC):
    """Abstract strategy for detection based on image type"""
    
    @abstractmethod
    def get_required_detectors(self) -> List[str]:
        """Get list of required detector names"""
        pass
    
    @abstractmethod
    def create_detectors(self, sensitivity: str) -> Dict[str, Any]:
        """Create detector instances"""
        pass


class StripeDetectionStrategy(DetectionStrategy):
    """Detection strategy for stripe images"""
    
    def __init__(self, selected_detectors: Optional[List[str]] = None):
        self.selected_detectors = selected_detectors
    
    def get_required_detectors(self) -> List[str]:
        all_detectors = ['debris', 'smudge', 'void', 'head_calibration']
        if self.selected_detectors:
            # Only return detectors that are both available and selected
            return [d for d in all_detectors if d in self.selected_detectors]
        return all_detectors
    
    def create_detectors(self, sensitivity: str) -> Dict[str, Any]:
        required = self.get_required_detectors()
        detectors = {}
        
        if 'debris' in required:
            detectors['debris'] = DetectorFactory.create_debris_detector(sensitivity)
        if 'smudge' in required:
            detectors['smudge'] = DetectorFactory.create_smudge_detector(sensitivity)
        if 'void' in required:
            detectors['void'] = DetectorFactory.create_void_detector(sensitivity)
        if 'head_calibration' in required:
            detectors['head_calibration'] = DetectorFactory.create_head_calibration_detector(sensitivity)
            
        return detectors


class IslandDetectionStrategy(DetectionStrategy):
    """Detection strategy for island images"""
    
    def __init__(self, selected_detectors: Optional[List[str]] = None):
        self.selected_detectors = selected_detectors
    
    def get_required_detectors(self) -> List[str]:
        # Surface treatment for islands, but only if selected
        all_detectors = ['surface_treatment']
        if self.selected_detectors:
            return [d for d in all_detectors if d in self.selected_detectors]
        return []  # Default to no detectors for islands as overspray is disabled
    
    def create_detectors(self, sensitivity: str) -> Dict[str, Any]:
        required = self.get_required_detectors()
        detectors = {}
        
        if 'surface_treatment' in required:
            detectors['surface_treatment'] = DetectorFactory.create_surface_treatment_detector(sensitivity)
            
        return detectors


class UnknownDetectionStrategy(DetectionStrategy):
    """Detection strategy for unknown image types"""
    
    def __init__(self, selected_detectors: Optional[List[str]] = None):
        self.selected_detectors = selected_detectors
    
    def get_required_detectors(self) -> List[str]:
        # Run all detectors except surface treatment for unknown types
        all_detectors = ['debris', 'smudge', 'void', 'head_calibration']
        if self.selected_detectors:
            return [d for d in all_detectors if d in self.selected_detectors]
        return all_detectors
    
    def create_detectors(self, sensitivity: str) -> Dict[str, Any]:
        required = self.get_required_detectors()
        detectors = {}
        
        if 'debris' in required:
            detectors['debris'] = DetectorFactory.create_debris_detector(sensitivity)
        if 'smudge' in required:
            detectors['smudge'] = DetectorFactory.create_smudge_detector(sensitivity)
        if 'void' in required:
            detectors['void'] = DetectorFactory.create_void_detector(sensitivity)
        if 'head_calibration' in required:
            detectors['head_calibration'] = DetectorFactory.create_head_calibration_detector(sensitivity)
            
        return detectors


class DetectionStrategyFactory:
    """Factory for creating detection strategies"""
    
    _strategies = {
        ImageType.STRIPE: StripeDetectionStrategy,
        ImageType.ISLAND: IslandDetectionStrategy,
        ImageType.UNKNOWN: UnknownDetectionStrategy
    }
    
    @classmethod
    def create_strategy(cls, image_type: ImageType, selected_detectors: Optional[List[str]] = None) -> DetectionStrategy:
        """Create appropriate detection strategy"""
        strategy_class = cls._strategies.get(image_type, UnknownDetectionStrategy)
        return strategy_class(selected_detectors) 