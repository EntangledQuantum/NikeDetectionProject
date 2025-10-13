"""
Data Models for Defect Detection Pipeline

This module contains all the data structures and enums used throughout
the defect detection pipeline, following clean architecture principles.

Author: Refactored Architecture
Date: 2024
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum


class ImageType(Enum):
    """Enum for different image types"""
    STRIPE = "stripe"
    ISLAND = "island"
    UNKNOWN = "unknown"


@dataclass
class ProcessingConfig:
    """Configuration for processing parameters"""
    output_base_dir: str = "output"
    sensitivity: str = 'medium'
    generate_report: bool = False
    enable_parallel_processing: bool = True
    max_workers: int = 4
    save_individual_visualizations: bool = False
    verbose_timing: bool = True
    selected_detectors: Optional[List[str]] = None


@dataclass
class DetectionResult:
    """Standardized result format for all detectors"""
    defect_count: int
    visualization_path: Optional[str]
    defects: List[Dict[str, Any]]
    error: Optional[str] = None
    timing: Optional[Dict[str, float]] = None


@dataclass
class ImageResult:
    """Result for a single image"""
    image_name: str
    image_path: str
    image_type: ImageType
    processing_time: str
    file_size_mb: float
    detectors_used: List[str]
    detections: Dict[str, DetectionResult]
    error: Optional[str] = None


@dataclass
class ClusterInfo:
    """Information about a defect cluster"""
    cluster_id: Any
    defect_type: str
    defects: List[Dict[str, Any]]
    centroid: Tuple[int, int]
    bounding_box: Tuple[int, int, int, int]
    cluster_size: int
    cluster_area: float


@dataclass
class TimingStats:
    """Timing statistics for performance analysis"""
    total_images_processed: int
    detector_timing: Dict[str, Dict[str, float]]
    operation_timing: Dict[str, Dict[str, float]]
    optimization_impact: Dict[str, float]


def to_dict(obj) -> Dict[str, Any]:
    """Convert dataclass to dictionary for JSON serialization"""
    if hasattr(obj, '__dict__'):
        result = {}
        for key, value in obj.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif hasattr(value, '__dict__'):
                result[key] = to_dict(value)
            elif isinstance(value, (list, tuple)):
                result[key] = [to_dict(item) if hasattr(item, '__dict__') else item for item in value]
            elif isinstance(value, dict):
                result[key] = {k: to_dict(v) if hasattr(v, '__dict__') else v for k, v in value.items()}
            else:
                result[key] = value
        return result
    return obj 