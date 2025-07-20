"""
Refactored Defect Detection Pipeline
Processes images through appropriate detection algorithms based on image name patterns

Author: Koushik and Cursor Assistant
Date: 2024
Version: 2.0 - Refactored for SOLID principles and efficiency
"""

import os
import cv2
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

# Import detection modules
from surface_treatment_detection import SurfaceTreatmentDetector
from debris_detection import DebrisDetector
from smudge_detection import SmudgeDetector
from void_detection import VoidDetector
from head_calibration_detection import HeadCalibrationDetector


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


@dataclass
class DetectionResult:
    """Standardized result format for all detectors"""
    defect_count: int
    visualization_path: Optional[str]
    defects: List[Dict[str, Any]]
    error: Optional[str] = None


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


class DefectClusterer:
    """Clusters nearby defects of the same type using sophisticated algorithms"""
    
    def __init__(self, eps: float = 50.0, min_samples: int = 2, max_cluster_distance: float = 100.0):
        """
        Args:
            eps: Maximum distance between two samples for clustering (DBSCAN parameter)
            min_samples: Minimum number of samples in a neighborhood for a core point
            max_cluster_distance: Maximum distance to consider defects as clusterable
        """
        self.eps = eps
        self.min_samples = min_samples
        self.max_cluster_distance = max_cluster_distance
    
    def cluster_defects_by_type(self, defects: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Cluster defects by type and proximity using DBSCAN clustering
        
        Args:
            defects: List of defect dictionaries with 'type', 'location', etc.
            
        Returns:
            Dictionary mapping defect types to lists of clustered defects
        """
        if not defects:
            return {}
        
        # Group defects by type first
        defects_by_type = {}
        for defect in defects:
            defect_type = defect.get('type', 'unknown')
            if defect_type not in defects_by_type:
                defects_by_type[defect_type] = []
            defects_by_type[defect_type].append(defect)
        
        # Cluster each type separately
        clustered_results = {}
        for defect_type, type_defects in defects_by_type.items():
            clustered_results[defect_type] = self._cluster_single_type(type_defects, defect_type)
        
        return clustered_results
    
    def _cluster_single_type(self, defects: List[Dict[str, Any]], defect_type: str) -> List[Dict[str, Any]]:
        """Cluster defects of a single type"""
        if len(defects) <= 1:
            # If only one defect, create a single-defect cluster
            if defects:
                return [{
                    'cluster_id': 0,
                    'defect_type': defect_type,
                    'defects': defects,
                    'centroid': defects[0]['location'],
                    'bounding_box': self._calculate_bounding_box([defects[0]]),
                    'cluster_size': 1,
                    'cluster_area': defects[0].get('area', 0)
                }]
            return []
        
        # Extract locations for clustering
        locations = []
        for defect in defects:
            if 'location' in defect:
                locations.append(defect['location'])
            elif 'bbox' in defect and len(defect['bbox']) >= 4:
                # Calculate centroid from bounding box
                minr, minc, maxr, maxc = defect['bbox'][:4]
                centroid = ((minr + maxr) // 2, (minc + maxc) // 2)
                locations.append(centroid)
            else:
                # Skip defects without location info
                continue
        
        if len(locations) < 2:
            # Not enough valid locations for clustering
            valid_defects = [d for d in defects if 'location' in d or 'bbox' in d]
            if valid_defects:
                return [{
                    'cluster_id': 0,
                    'defect_type': defect_type,
                    'defects': valid_defects,
                    'centroid': locations[0] if locations else (0, 0),
                    'bounding_box': self._calculate_bounding_box(valid_defects),
                    'cluster_size': len(valid_defects),
                    'cluster_area': sum(d.get('area', 0) for d in valid_defects)
                }]
            return []
        
        # Apply DBSCAN clustering
        locations_array = np.array(locations)
        
        # Use adaptive eps based on data distribution
        adaptive_eps = min(self.eps, self._calculate_adaptive_eps(locations_array))
        
        clustering = DBSCAN(eps=adaptive_eps, min_samples=self.min_samples)
        cluster_labels = clustering.fit_predict(locations_array)
        
        # Group defects by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label == -1:  # Noise points get individual clusters
                label = f"noise_{i}"
            
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(defects[i])
        
        # Create cluster objects
        result_clusters = []
        for cluster_id, cluster_defects in clusters.items():
            centroid = self._calculate_cluster_centroid(cluster_defects)
            bounding_box = self._calculate_bounding_box(cluster_defects)
            
            result_clusters.append({
                'cluster_id': cluster_id,
                'defect_type': defect_type,
                'defects': cluster_defects,
                'centroid': centroid,
                'bounding_box': bounding_box,
                'cluster_size': len(cluster_defects),
                'cluster_area': sum(d.get('area', 0) for d in cluster_defects)
            })
        
        return result_clusters
    
    def _calculate_adaptive_eps(self, locations: np.ndarray) -> float:
        """Calculate adaptive eps based on data distribution"""
        if len(locations) < 2:
            return self.eps
        
        # Calculate pairwise distances
        distances = cdist(locations, locations)
        
        # Use k-nearest neighbor distance (k=4) as adaptive eps
        k = min(4, len(locations) - 1)
        knn_distances = []
        for i in range(len(locations)):
            row_distances = distances[i]
            row_distances = np.sort(row_distances)[1:k+1]  # Exclude self (distance=0)
            knn_distances.append(np.mean(row_distances))
        
        # Use 75th percentile of k-NN distances
        adaptive_eps = np.percentile(knn_distances, 75)
        return min(adaptive_eps, self.max_cluster_distance)
    
    def _calculate_cluster_centroid(self, cluster_defects: List[Dict[str, Any]]) -> Tuple[int, int]:
        """Calculate the centroid of a cluster"""
        x_coords = []
        y_coords = []
        
        for defect in cluster_defects:
            if 'location' in defect:
                x, y = defect['location']
                x_coords.append(x)
                y_coords.append(y)
            elif 'bbox' in defect and len(defect['bbox']) >= 4:
                minr, minc, maxr, maxc = defect['bbox'][:4]
                x_coords.append((minc + maxc) // 2)
                y_coords.append((minr + maxr) // 2)
        
        if x_coords and y_coords:
            return (int(np.mean(x_coords)), int(np.mean(y_coords)))
        return (0, 0)
    
    def _calculate_bounding_box(self, cluster_defects: List[Dict[str, Any]]) -> Tuple[int, int, int, int]:
        """Calculate the bounding box that encompasses all defects in cluster"""
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        
        for defect in cluster_defects:
            if 'bbox' in defect and len(defect['bbox']) >= 4:
                minr, minc, maxr, maxc = defect['bbox'][:4]
                min_x = min(min_x, minc)
                min_y = min(min_y, minr)
                max_x = max(max_x, maxc)
                max_y = max(max_y, maxr)
            elif 'location' in defect:
                x, y = defect['location']
                # Use a default size if no bbox available
                size = defect.get('area', 100) ** 0.5 / 2  # Approximate radius
                min_x = min(min_x, x - size)
                min_y = min(min_y, y - size)
                max_x = max(max_x, x + size)
                max_y = max(max_y, y + size)
        
        # Handle case where no valid coordinates found
        if min_x == float('inf'):
            return (0, 0, 10, 10)
        
        return (int(min_y), int(min_x), int(max_y), int(max_x))
    
    def create_cluster_visualization(self, image: np.ndarray, 
                                   clustered_defects: Dict[str, List[Dict[str, Any]]]) -> np.ndarray:
        """
        Create improved visualization with bright hollow circles for clustered defects
        
        Args:
            image: Original image
            clustered_defects: Dictionary of clustered defects by type
            
        Returns:
            Visualization image with clustered defects highlighted
        """
        vis = image.copy()
        
        # Color scheme for different defect types
        type_colors = {
            'debris': (0, 255, 255),      # Yellow
            'smudge': (255, 0, 255),      # Magenta
            'void': (0, 0, 255),          # Red
            'head_calibration': (255, 255, 0),  # Cyan
            'surface_treatment': (0, 255, 0),    # Green
            'unknown': (128, 128, 128)    # Gray
        }
        
        cluster_id_counter = 0
        
        for defect_type, clusters in clustered_defects.items():
            color = type_colors.get(defect_type, (255, 255, 255))  # White default
            
            for cluster in clusters:
                cluster_id_counter += 1
                centroid = cluster['centroid']
                bounding_box = cluster['bounding_box']
                cluster_size = cluster['cluster_size']
                
                # Calculate circle radius based on cluster size and bounding box
                minr, minc, maxr, maxc = bounding_box
                bbox_width = maxc - minc
                bbox_height = maxr - minr
                base_radius = max(bbox_width, bbox_height) // 2 + 10
                
                # Scale radius based on cluster size
                size_multiplier = 1 + (cluster_size - 1) * 0.3  # Larger for more defects
                radius = int(base_radius * size_multiplier)
                radius = max(radius, 15)  # Minimum radius for visibility
                radius = min(radius, 100)  # Maximum radius to avoid huge circles
                
                # Draw bright hollow circle
                circle_thickness = max(3, radius // 10)  # Thickness scales with radius
                cv2.circle(vis, centroid, radius, color, circle_thickness)
                
                # Draw inner circle for better visibility
                inner_radius = max(5, radius // 3)
                cv2.circle(vis, centroid, inner_radius, color, 2)
                
                # Add cluster information text
                text = f"{defect_type[:4]}-{cluster_size}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                text_thickness = 1
                
                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
                
                # Position text above the circle
                text_x = centroid[0] - text_width // 2
                text_y = centroid[1] - radius - 10
                
                # Ensure text stays within image bounds
                text_x = max(0, min(text_x, vis.shape[1] - text_width))
                text_y = max(text_height, min(text_y, vis.shape[0]))
                
                # Draw text background
                cv2.rectangle(vis, 
                            (text_x - 2, text_y - text_height - 2),
                            (text_x + text_width + 2, text_y + 2),
                            (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(vis, text, (text_x, text_y), font, font_scale, color, text_thickness)
                
                # Draw connecting lines between defects in the cluster if cluster has multiple defects
                if cluster_size > 1:
                    defects = cluster['defects']
                    for i, defect in enumerate(defects):
                        defect_location = defect.get('location')
                        if defect_location:
                            # Draw line from centroid to defect
                            cv2.line(vis, centroid, defect_location, color, 1)
                            # Mark individual defect with small circle
                            cv2.circle(vis, defect_location, 3, color, -1)
        
        return vis


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
                dark_threshold=0.4,
                min_debris_size=50,
                max_debris_size=1500,
                fiber_min_length=50
            )
        elif sensitivity == 'high':
            return DebrisDetector(
                dark_threshold=0.2,
                min_debris_size=10,
                max_debris_size=3000,
                fiber_min_length=20
            )
        else:  # medium
            return DebrisDetector()  # Use defaults
    
    @staticmethod
    def create_smudge_detector(sensitivity: str) -> SmudgeDetector:
        """Create smudge detector with appropriate settings"""
        if sensitivity == 'low':
            return SmudgeDetector(
                coherence_threshold=0.5,
                min_smudge_area=800
            )
        elif sensitivity == 'high':
            return SmudgeDetector(
                coherence_threshold=0.3,
                min_smudge_area=300
            )
        else:  # medium
            return SmudgeDetector()  # Use defaults
    
    @staticmethod
    def create_void_detector(sensitivity: str) -> VoidDetector:
        """Create void detector with appropriate settings"""
        if sensitivity == 'low':
            return VoidDetector(
                min_void_size=20,
                max_void_size=300,
                circularity_threshold=0.7,
                contrast_threshold=0.4
            )
        elif sensitivity == 'high':
            return VoidDetector(
                min_void_size=5,
                max_void_size=800,
                circularity_threshold=0.5,
                contrast_threshold=0.2
            )
        else:  # medium
            return VoidDetector()  # Use defaults
    
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
    
    def get_required_detectors(self) -> List[str]:
        return ['debris', 'smudge', 'void', 'head_calibration']
    
    def create_detectors(self, sensitivity: str) -> Dict[str, Any]:
        return {
            'debris': DetectorFactory.create_debris_detector(sensitivity),
            'smudge': DetectorFactory.create_smudge_detector(sensitivity),
            'void': DetectorFactory.create_void_detector(sensitivity),
            'head_calibration': DetectorFactory.create_head_calibration_detector(sensitivity)
        }


class IslandDetectionStrategy(DetectionStrategy):
    """Detection strategy for island images"""
    
    def get_required_detectors(self) -> List[str]:
        # Overspray is disabled as requested
        return []
    
    def create_detectors(self, sensitivity: str) -> Dict[str, Any]:
        # Return empty dict since overspray is disabled
        return {}


class UnknownDetectionStrategy(DetectionStrategy):
    """Detection strategy for unknown image types"""
    
    def get_required_detectors(self) -> List[str]:
        # Run all detectors except surface treatment for unknown types
        return ['debris', 'smudge', 'void', 'head_calibration']
    
    def create_detectors(self, sensitivity: str) -> Dict[str, Any]:
        return {
            'debris': DetectorFactory.create_debris_detector(sensitivity),
            'smudge': DetectorFactory.create_smudge_detector(sensitivity),
            'void': DetectorFactory.create_void_detector(sensitivity),
            'head_calibration': DetectorFactory.create_head_calibration_detector(sensitivity)
        }


class DetectionStrategyFactory:
    """Factory for creating detection strategies"""
    
    _strategies = {
        ImageType.STRIPE: StripeDetectionStrategy,
        ImageType.ISLAND: IslandDetectionStrategy,
        ImageType.UNKNOWN: UnknownDetectionStrategy
    }
    
    @classmethod
    def create_strategy(cls, image_type: ImageType) -> DetectionStrategy:
        """Create appropriate detection strategy"""
        strategy_class = cls._strategies.get(image_type, UnknownDetectionStrategy)
        return strategy_class()


class SingleImageProcessor:
    """Processes a single image through appropriate detectors"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.clusterer = DefectClusterer(eps=50.0, min_samples=2, max_cluster_distance=100.0)
    
    def process_image(self, image_path: str, output_dir: str) -> ImageResult:
        """Process a single image through appropriate detectors"""
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(image_output_dir, exist_ok=True)
        
        # Classify image type
        image_type = ImageTypeClassifier.classify_image(image_path)
        
        # Get detection strategy
        strategy = DetectionStrategyFactory.create_strategy(image_type)
        required_detectors = strategy.get_required_detectors()
        
        # Early return if no detectors needed
        if not required_detectors:
            return ImageResult(
                image_name=base_name,
                image_path=image_path,
                image_type=image_type,
                processing_time=datetime.now().isoformat(),
                file_size_mb=self._get_file_size_mb(image_path),
                detectors_used=[],
                detections={},
                error="No detectors available for this image type"
            )
        
        print(f"\n  Processing: {base_name} (type: {image_type.value})")
        print(f"  Required detectors: {required_detectors}")
        
        # Create detectors
        detectors = strategy.create_detectors(self.config.sensitivity)
        
        # Get file size for reporting
        file_size = os.path.getsize(image_path)
        print(f"    Image size: {file_size/(1024*1024):.1f}MB")
        
        # Process the image (always use full image processing)
        return self._process_image(image_path, image_output_dir, detectors, 
                                 image_type, base_name, file_size)
    
    def _process_image(self, image_path: str, output_dir: str, detectors: Dict[str, Any],
                      image_type: ImageType, base_name: str, file_size: int) -> ImageResult:
        """Process image using full image processing with clustering"""
        detections = {}
        all_defects = []  # Collect all defects for clustering
        
        # Load the image once for all detectors
        original = TiffImageLoader.load_image(image_path)
        if original is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        print(f"      Image loaded: {original.shape} pixels")
        
        # Run all detectors
        for detector_name, detector in detectors.items():
            print(f"    Running {detector_name} detection...")
            try:
                # All detectors now accept BGR images and handle preprocessing internally
                result_img, defects = detector.detect(original)
                
                print(f"      Found {len(defects)} defects")
                
                # Add detector name to each defect for clustering
                for defect in defects:
                    defect['detector'] = detector_name
                
                # Collect defects for clustering
                all_defects.extend(defects)
                
                # Clean defects for JSON serialization
                cleaned_defects = self._clean_defect_data(defects)
                
                detections[detector_name] = DetectionResult(
                    defect_count=len(defects),
                    visualization_path=None,  # Will be set after clustering
                    defects=cleaned_defects
                )
                
            except Exception as e:
                print(f"    Error in {detector_name} detection: {str(e)}")
                import traceback
                traceback.print_exc()
                detections[detector_name] = DetectionResult(
                    defect_count=0,
                    visualization_path=None,
                    defects=[],
                    error=str(e)
                )
        
        # Apply clustering to all defects
        print(f"    Applying clustering to {len(all_defects)} total defects...")
        clustered_defects = self.clusterer.cluster_defects_by_type(all_defects)
        
        # Count clustered defects
        total_clusters = sum(len(clusters) for clusters in clustered_defects.values())
        print(f"      Created {total_clusters} clusters from {len(all_defects)} defects")
        
        # Create clustered visualization
        if clustered_defects:
            clustered_vis = self.clusterer.create_cluster_visualization(original, clustered_defects)
            clustered_vis_path = os.path.join(output_dir, "clustered_defects_visualization.jpg")
            cv2.imwrite(clustered_vis_path, clustered_vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"      Saved clustered visualization: {clustered_vis_path}")
        
        # Create individual detector visualizations with clustering overlay
        for detector_name, detection in detections.items():
            if detection.error is None and detection.defect_count > 0:
                # Get defects for this detector only
                detector_defects = [d for d in all_defects if d.get('detector') == detector_name]
                detector_clustered = self.clusterer.cluster_defects_by_type(detector_defects)
                
                # Create visualization for this detector
                detector_vis = self.clusterer.create_cluster_visualization(original, detector_clustered)
                
                # Save detector-specific visualization
                vis_path = os.path.join(output_dir, f"{detector_name}_clustered_visualization.jpg")
                cv2.imwrite(vis_path, detector_vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # Update the detection result with the visualization path
                detections[detector_name] = DetectionResult(
                    defect_count=detection.defect_count,
                    visualization_path=vis_path,
                    defects=detection.defects,
                    error=detection.error
                )
        
        # Save clustering information
        if clustered_defects:
            cluster_info_path = os.path.join(output_dir, "cluster_information.json")
            try:
                with open(cluster_info_path, 'w') as f:
                    json.dump(clustered_defects, f, indent=2, default=self._json_serializer)
                print(f"      Saved cluster information: {cluster_info_path}")
            except Exception as e:
                print(f"      Warning: Could not save cluster information: {e}")
        
        return ImageResult(
            image_name=base_name,
            image_path=image_path,
            image_type=image_type,
            processing_time=datetime.now().isoformat(),
            file_size_mb=file_size / (1024 * 1024),
            detectors_used=list(detectors.keys()),
            detections=detections
        )
    
    def _get_file_size_mb(self, image_path: str) -> float:
        """Get file size in MB"""
        return os.path.getsize(image_path) / (1024 * 1024)
    
    def _clean_defect_data(self, defects: List[Dict]) -> List[Dict]:
        """Clean defect data for JSON serialization"""
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
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for clustering data"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, tuple):
            return list(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)


class ResultsSaver:
    """Handles saving of results in various formats"""
    
    @staticmethod
    def save_image_result(result: ImageResult, output_dir: str) -> None:
        """Save individual image result to JSON"""
        try:
            json_path = os.path.join(output_dir, f"{result.image_name}_results.json")
            with open(json_path, 'w') as f:
                json.dump(asdict(result), f, indent=2, default=ResultsSaver._json_serializer)
        except Exception as e:
            print(f"    Warning: Could not save JSON results: {e}")
            # Save simplified version
            simplified_result = {
                'image_name': result.image_name,
                'processing_time': result.processing_time,
                'image_type': result.image_type.value,
                'detectors_used': result.detectors_used,
                'detection_counts': {name: detection.defect_count 
                                   for name, detection in result.detections.items()}
            }
            try:
                simplified_path = os.path.join(output_dir, f"{result.image_name}_results_simplified.json")
                with open(simplified_path, 'w') as f:
                    json.dump(simplified_result, f, indent=2)
                print(f"    Saved simplified results instead")
            except Exception as e2:
                print(f"    Could not save even simplified results: {e2}")
    
    @staticmethod
    def save_summary_report(results: List[ImageResult], output_dir: str, 
                          config: ProcessingConfig) -> None:
        """Save summary report"""
        # Calculate statistics
        stats = ResultsSaver._calculate_statistics(results)
        
        # Save JSON summary
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(results),
            'detection_sensitivity': config.sensitivity,
            'image_type_distribution': stats['image_type_distribution'],
            'defect_statistics': stats['defect_statistics'],
            'processing_summary': stats['processing_summary'],
            'detailed_results': [asdict(result) for result in results]
        }
        
        summary_path = os.path.join(output_dir, "defect_report.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=ResultsSaver._json_serializer)
        
        print(f"Summary report saved to: {summary_path}")
        
        # Generate PDF report if requested
        if config.generate_report:
            ResultsSaver._generate_pdf_report(results, stats, output_dir)
    
    @staticmethod
    def _calculate_statistics(results: List[ImageResult]) -> Dict[str, Any]:
        """Calculate statistics from results"""
        # Image type distribution
        image_type_distribution = {}
        for result in results:
            image_type = result.image_type.value
            image_type_distribution[image_type] = image_type_distribution.get(image_type, 0) + 1
        
        # Defect statistics
        defect_statistics = {}
        processing_summary = {
            'successful_images': 0,
            'failed_images': 0,
            'total_processing_time': 0,
            'average_file_size_mb': 0.0
        }
        
        total_file_size = 0
        for result in results:
            if result.error:
                processing_summary['failed_images'] += 1
            else:
                processing_summary['successful_images'] += 1
            
            total_file_size += result.file_size_mb
            
            for detector_name, detection in result.detections.items():
                if detector_name not in defect_statistics:
                    defect_statistics[detector_name] = {
                        'total_defects': 0,
                        'affected_images': 0
                    }
                
                defect_statistics[detector_name]['total_defects'] += detection.defect_count
                if detection.defect_count > 0:
                    defect_statistics[detector_name]['affected_images'] += 1
        
        processing_summary['average_file_size_mb'] = float(total_file_size / len(results)) if results else 0.0
        
        return {
            'image_type_distribution': image_type_distribution,
            'defect_statistics': defect_statistics,
            'processing_summary': processing_summary
        }
    
    @staticmethod
    def _generate_pdf_report(results: List[ImageResult], stats: Dict[str, Any], 
                           output_dir: str) -> None:
        """Generate PDF report"""
        pdf_path = os.path.join(output_dir, "defect_detection_report.pdf")
        
        with PdfPages(pdf_path) as pdf:
            # Summary page
            fig, ax = plt.subplots(figsize=(8.5, 11))
            fig.suptitle("Defect Detection Summary Report", fontsize=20, fontweight='bold')
            
            summary_text = f"Total Images Processed: {len(results)}\n\n"
            summary_text += "Image Type Distribution:\n" + "-" * 40 + "\n"
            
            for img_type, count in stats['image_type_distribution'].items():
                summary_text += f"  {img_type.title()}: {count} images\n"
            
            summary_text += "\nDefect Summary:\n" + "-" * 40 + "\n"
            
            for detector_name, detector_stats in stats['defect_statistics'].items():
                summary_text += f"\n{detector_name.replace('_', ' ').title()}:\n"
                summary_text += f"  Total Defects Found: {detector_stats['total_defects']}\n"
                summary_text += f"  Images with Defects: {detector_stats['affected_images']}\n"
                if detector_stats['affected_images'] > 0:
                    avg = detector_stats['total_defects'] / detector_stats['affected_images']
                    summary_text += f"  Average per Affected Image: {avg:.2f}\n"
            
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                   fontsize=12, verticalalignment='top', fontfamily='monospace')
            ax.axis('off')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        print(f"PDF report saved to: {pdf_path}")
    
    @staticmethod
    def _json_serializer(obj):
        """Custom JSON serializer"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)


class DefectDetectionPipeline:
    """Main pipeline orchestrating the detection process"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.processor = SingleImageProcessor(config)
        self.results: List[ImageResult] = []
    
    def process_folder(self, input_folder: str) -> None:
        """Process all images in a folder"""
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(input_folder, f"output_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get image files
        image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')
        image_files = [f for f in os.listdir(input_folder)
                      if f.lower().endswith(image_extensions)]
        
        if not image_files:
            print(f"No image files found in {input_folder}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        for img_file in tqdm(image_files, desc="Processing images"):
            img_path = os.path.join(input_folder, img_file)
            result = self.processor.process_image(img_path, output_dir)
            self.results.append(result)
            
            # Save individual result
            image_output_dir = os.path.join(output_dir, result.image_name)
            ResultsSaver.save_image_result(result, image_output_dir)
            
            print(f"  ✅ Completed processing {result.image_name}")
        
        # Generate summary report
        ResultsSaver.save_summary_report(self.results, output_dir, self.config)
        
        print(f"\n✅ Processing complete! Results saved to: {output_dir}")
        print(f"📊 Processed {len(self.results)} images total")
        
        # Print summary statistics
        self._print_summary_statistics()
    
    def _print_summary_statistics(self) -> None:
        """Print summary statistics to console"""
        if not self.results:
            return
        
        # Count by image type
        type_counts = {}
        detector_counts = {}
        
        for result in self.results:
            # Count image types
            img_type = result.image_type.value
            type_counts[img_type] = type_counts.get(img_type, 0) + 1
            
            # Count detections
            for detector_name, detection in result.detections.items():
                if detector_name not in detector_counts:
                    detector_counts[detector_name] = 0
                detector_counts[detector_name] += detection.defect_count
        
        print("\n📈 Summary Statistics:")
        print("-" * 40)
        
        print("Image Types Processed:")
        for img_type, count in type_counts.items():
            print(f"  {img_type.title()}: {count} images")
        
        print("\nDefects Found:")
        for detector_name, count in detector_counts.items():
            print(f"  {detector_name.replace('_', ' ').title()}: {count} defects")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run defect detection algorithms on images based on filename patterns'
    )
    parser.add_argument('--input_folder', required=True, 
                       help='Path to folder containing images')
    parser.add_argument('--output', '-o', 
                       help='Output base directory (default: creates output folder in input folder)')
    parser.add_argument('--generate_report', action='store_true',
                       help='Generate PDF report with all detections')
    parser.add_argument('--sensitivity', choices=['low', 'medium', 'high'], default='medium',
                       help='Detection sensitivity level (default: medium)')
    
    args = parser.parse_args()
    
    # Validate input folder
    if not os.path.isdir(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist")
        sys.exit(1)
    
    # Create configuration
    config = ProcessingConfig(
        output_base_dir=args.output,
        sensitivity=args.sensitivity,
        generate_report=args.generate_report
    )
    
    # Create and run pipeline
    pipeline = DefectDetectionPipeline(config)
    
    print("=" * 60)
    print("Defect Detection Pipeline v2.0 with Clustering")
    print("=" * 60)
    print(f"Input folder: {args.input_folder}")
    print(f"Detection routing:")
    print(f"  - Stripe images: Debris, Smudge, Void, Head Calibration")
    print(f"  - Island images: (Currently no detectors)")
    print(f"  - Unknown images: Debris, Smudge, Void, Head Calibration")
    print(f"Features:")
    print(f"  - DBSCAN clustering for nearby defects of same type")
    print(f"  - Bright hollow circle visualization for clusters")
    print(f"  - Adaptive clustering parameters based on defect distribution")
    print("=" * 60)
    
    pipeline.process_folder(args.input_folder)


if __name__ == "__main__":
    main() 