"""
Refactored Defect Detection Pipeline

A clean, modular implementation of the defect detection pipeline
following SOLID principles and clean architecture patterns.

Author: Refactored Architecture
Date: 2024
"""

from .data_models import (
    ImageType, ProcessingConfig, DetectionResult, 
    ImageResult, ClusterInfo, TimingStats
)
from .image_processing import (
    ImageTypeClassifier, TiffImageLoader, 
    ImagePreprocessor, SharedPreprocessor
)
from .detection_strategies import (
    DetectorFactory, DetectionStrategy, DetectionStrategyFactory
)
from .clustering import DefectClusterer, ClusterVisualizer
from .optimized_detector import OptimizedCombinedDetector
from .pipeline import DefectDetectionPipeline, SingleImageProcessor
from .results_manager import ResultsSaver

__version__ = "2.0.0"
__author__ = "Refactored Architecture"

__all__ = [
    # Data Models
    'ImageType', 'ProcessingConfig', 'DetectionResult', 
    'ImageResult', 'ClusterInfo', 'TimingStats',
    
    # Image Processing
    'ImageTypeClassifier', 'TiffImageLoader', 
    'ImagePreprocessor', 'SharedPreprocessor',
    
    # Detection Strategies
    'DetectorFactory', 'DetectionStrategy', 'DetectionStrategyFactory',
    
    # Clustering
    'DefectClusterer', 'ClusterVisualizer',
    
    # Detection
    'OptimizedCombinedDetector',
    
    # Pipeline
    'DefectDetectionPipeline', 'SingleImageProcessor',
    
    # Results
    'ResultsSaver'
] 