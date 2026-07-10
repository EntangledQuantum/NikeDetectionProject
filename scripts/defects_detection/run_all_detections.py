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

# Import detection modules
from surface_treatment_detection import SurfaceTreatmentDetector
from line_defect_detection import LineDefectDetector
from stripe_misalignment_detection import StripeMisalignmentDetector
from overspray_detection import OversprayDetector
from debris_island_detection import DebrisIslandDetector
from overspray_island_detection import OversprayIslandDetector
from void_detection import VoidDetector
from debris_stripe_detector import DebrisStripeDetector
from new_pattern_debris_island_detection import NewPatternDebrisIslandDetector
from new_pattern_overspray_island_detection import NewPatternOversprayIslandDetector
from new_pattern_line_defect_detection import NewPatternLineDefectDetector


class ImageType(Enum):
    """Enumeration of supported image categories used for routing detections.

    - stripe: images with pronounced stripe patterns (e.g., 'blueStripe.tiff')
    - island: images with isolated printed islands or shapes (e.g., 'island-black-blue.tiff')
    - unknown: fallback when filename patterns do not match any rule
    """
    STRIPE = "stripe"
    ISLAND = "island"
    UNKNOWN = "unknown"


@dataclass
class ProcessingConfig:
    """Configuration container for pipeline-level processing parameters.

    Attributes:
        output_base_dir: Base directory for results; if provided, overrides
            the default output folder under the input directory.
        sensitivity: Detector sensitivity level: 'low' | 'medium' | 'high'.
        generate_report: If True, produces a PDF report in addition to JSON.
    """
    output_base_dir: str = "output"
    sensitivity: str = 'medium'
    generate_report: bool = False
    pattern: str = 'legacy'  # island pattern: 'legacy' (single band) or 'new' (dual band)


@dataclass
class DetectionResult:
    """Standardized result payload produced by each detector.

    Attributes:
        defect_count: Number of defects detected.
        visualization_path: Path to a saved visualization image, if any.
        defects: Per-defect dictionaries with detector-specific fields.
        error: Optional error message if detection failed.
    """
    defect_count: int
    visualization_path: Optional[str]
    defects: List[Dict[str, Any]]
    error: Optional[str] = None


@dataclass
class ImageResult:
    """Aggregated results for a single image across all detectors.

    Attributes:
        image_name: Base filename without extension.
        image_path: Full path to the processed image.
        image_type: Classified `ImageType` used to choose detectors.
        processing_time: ISO timestamp when processing completed.
        file_size_mb: File size in megabytes.
        detectors_used: List of detector keys that were run.
        detections: Mapping of detector key to `DetectionResult`.
        error: Optional error message if high-level processing failed.
    """
    image_name: str
    image_path: str
    image_type: ImageType
    processing_time: str
    file_size_mb: float
    detectors_used: List[str]
    detections: Dict[str, DetectionResult]
    error: Optional[str] = None


class ImagePreprocessor:
    """Centralized image preprocessing utilities shared by detectors."""
    
    @staticmethod
    def load_and_convert_to_grayscale(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load an image and return both original (BGR) and grayscale versions.

        Args:
            image_path: Path to the input image readable by OpenCV.

        Returns:
            Tuple of (original_bgr, gray) as NumPy arrays.

        Raises:
            ValueError: If the image cannot be read from disk.
        """
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
        """Denoise a grayscale image using the specified filter.

        Args:
            gray: Grayscale image array (uint8).
            method: One of {'bilateral', 'median', 'gaussian'}.

        Returns:
            Denoised grayscale image as a NumPy array.
        """
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
        """Apply CLAHE contrast enhancement to improve local contrast.

        Args:
            gray: Grayscale image to enhance.
            clip_limit: CLAHE clip limit parameter.
            grid_size: CLAHE tile grid size (columns, rows).

        Returns:
            Enhanced grayscale image as a NumPy array.
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        return clahe.apply(gray)
    
    @classmethod
    def preprocess_for_surface_treatment(cls, image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess an image for surface treatment analysis.

        Loads the image, converts to grayscale, and applies CLAHE.

        Args:
            image_path: Path to the input image.

        Returns:
            Tuple of (original_bgr, gray, enhanced_gray).
        """
        original, gray = cls.load_and_convert_to_grayscale(image_path)
        enhanced = cls.enhance_contrast(gray, clip_limit=2.0)
        return original, gray, enhanced


class ImageTypeClassifier:
    """Helpers to infer `ImageType` from filename heuristics."""
    
    @staticmethod
    def classify_image(image_path: str) -> ImageType:
        """Infer image type from filename heuristics.

        Rules:
        - Contains 'stripe' -> ImageType.STRIPE
        - Contains 'island' -> ImageType.ISLAND
        - Otherwise        -> ImageType.UNKNOWN

        Args:
            image_path: Full path or filename of the image.

        Returns:
            An `ImageType` enum value.
        """
        filename = os.path.basename(image_path).lower()
        
        if 'stripe' in filename:
            return ImageType.STRIPE
        elif 'island' in filename:
            return ImageType.ISLAND
        else:
            return ImageType.UNKNOWN


# ------------------------------------------------------------
# Detection Strategy
# ------------------------------------------------------------

class DetectorFactory:
    """Constructors for detector instances configured by sensitivity.

    Some detectors are initialized with debug visualization enabled to
    aid troubleshooting.
    """
    
    @staticmethod
    def create_surface_treatment_detector(sensitivity: str) -> SurfaceTreatmentDetector:
        """Create a `SurfaceTreatmentDetector` tuned by sensitivity.

        Args:
            sensitivity: 'low' | 'medium' | 'high'.

        Returns:
            Configured `SurfaceTreatmentDetector` instance.
        """
        if sensitivity == 'low':
            return SurfaceTreatmentDetector(
                contrast_threshold=70, 
                void_size_threshold=200, 
                coalescence_threshold=400, 
                kernel_size=15
            )
        elif sensitivity == 'high':
            return SurfaceTreatmentDetector(
                contrast_threshold=30, 
                void_size_threshold=70, 
                coalescence_threshold=150, 
                kernel_size=18
            )
        else:  # medium
            return SurfaceTreatmentDetector(
                void_size_threshold=150, 
                coalescence_threshold=300, 
                kernel_size=12
            )
    
    
    @staticmethod
    def create_line_defect_detector(sensitivity: str) -> LineDefectDetector:
        """Create a `LineDefectDetector` tuned by sensitivity.

        Args:
            sensitivity: 'low' | 'medium' | 'high'.

        Returns:
            Configured `LineDefectDetector` instance.
        """
        # Enable debug mode for high sensitivity to see detected lines
        return LineDefectDetector(sensitivity=sensitivity, debug=False)
    
    @staticmethod
    def create_stripe_misalignment_detector(sensitivity: str) -> StripeMisalignmentDetector:
        """Create a `StripeMisalignmentDetector` tuned by sensitivity.

        Args:
            sensitivity: 'low' | 'medium' | 'high'.

        Returns:
            Configured `StripeMisalignmentDetector` instance.
        """
        return StripeMisalignmentDetector(sensitivity=sensitivity, debug=False)
    
    @staticmethod
    def create_overspray_detector(sensitivity: str) -> OversprayDetector:
        """Create an `OversprayDetector` tuned by sensitivity.

        Args:
            sensitivity: 'low' | 'medium' | 'high'.

        Returns:
            Configured `OversprayDetector` instance.
        """
        return OversprayDetector(sensitivity=sensitivity, debug=False)
    
    @staticmethod
    def create_debris_island_detector(sensitivity: str) -> DebrisIslandDetector:
        """Create a `DebrisIslandDetector` tuned by sensitivity.

        Args:
            sensitivity: 'low' | 'medium' | 'high'.

        Returns:
            Configured `DebrisIslandDetector` instance.
        """
        # All parameters are handled internally based on sensitivity
        return DebrisIslandDetector(
            sensitivity=sensitivity,
            debug=False               # Enable debug visualization
        )
    
    @staticmethod
    def create_overspray_island_detector(sensitivity: str) -> OversprayIslandDetector:
        """Create an `OversprayIslandDetector` tuned by sensitivity.

        Args:
            sensitivity: 'low' | 'medium' | 'high'.

        Returns:
            Configured `OversprayIslandDetector` instance.
        """
        # All parameters are handled internally based on sensitivity
        return OversprayIslandDetector(
            sensitivity=sensitivity,
            debug=False               # Enable debug visualization
        )

    @staticmethod
    def create_new_pattern_debris_island_detector(sensitivity: str) -> NewPatternDebrisIslandDetector:
        """Create a `NewPatternDebrisIslandDetector` tuned by sensitivity.

        Args:
            sensitivity: 'low' | 'medium' | 'high'.

        Returns:
            Configured `NewPatternDebrisIslandDetector` instance.
        """
        return NewPatternDebrisIslandDetector(sensitivity=sensitivity, debug=False)

    @staticmethod
    def create_new_pattern_overspray_island_detector(sensitivity: str) -> NewPatternOversprayIslandDetector:
        """Create a `NewPatternOversprayIslandDetector` tuned by sensitivity.

        Args:
            sensitivity: 'low' | 'medium' | 'high'.

        Returns:
            Configured `NewPatternOversprayIslandDetector` instance.
        """
        return NewPatternOversprayIslandDetector(sensitivity=sensitivity, debug=False)

    @staticmethod
    def create_new_pattern_line_defect_detector(sensitivity: str) -> NewPatternLineDefectDetector:
        """Create a `NewPatternLineDefectDetector` tuned by sensitivity.

        Args:
            sensitivity: 'low' | 'medium' | 'high'.

        Returns:
            Configured `NewPatternLineDefectDetector` instance.
        """
        return NewPatternLineDefectDetector(sensitivity=sensitivity, debug=False)

    @staticmethod
    def create_void_detector(sensitivity: str) -> VoidDetector:
        """Create a `VoidDetector` tuned by sensitivity.

        Args:
            sensitivity: 'low' | 'medium' | 'high'.

        Returns:
            Configured `VoidDetector` instance.
        """
        return VoidDetector(
            sensitivity=sensitivity,
            debug=False
        )

    @staticmethod
    def create_debris_stripe_detector(sensitivity: str) -> DebrisStripeDetector:
        """Create a `DebrisStripeDetector` tuned by sensitivity.

        Args:
            sensitivity: 'low' | 'medium' | 'high'.

        Returns:
            Configured `DebrisStripeDetector` instance.
        """
        return DebrisStripeDetector(
            sensitivity=sensitivity,
            debug=False
        )


class DetectionStrategy(ABC):
    """Abstract strategy for selecting detectors based on image type."""
    
    @abstractmethod
    def get_required_detectors(self) -> List[str]:
        """Return a list of detector keys to execute for this strategy."""
        pass
    
    @abstractmethod
    def create_detectors(self, sensitivity: str) -> Dict[str, Any]:
        """Instantiate detector objects keyed by their detector names.

        Args:
            sensitivity: Global sensitivity level for detector configuration.

        Returns:
            Mapping from detector key to detector instance.
        """
        pass


class StripeDetectionStrategy(DetectionStrategy):
    """Detection strategy for stripe images.

    Currently routes to the `stripe_misalignment` detector.
    """
    
    def get_required_detectors(self) -> List[str]:
        """Detectors to run for stripe images."""
        return ['stripe_misalignment', 'overspray', 'surface_treatment', 'void', 'debris_stripe']
    
    def create_detectors(self, sensitivity: str) -> Dict[str, Any]:
        """Create detector instances for stripe images."""
        return {
            #'surface_treatment': DetectorFactory.create_surface_treatment_detector(sensitivity),
            'stripe_misalignment': DetectorFactory.create_stripe_misalignment_detector(sensitivity),
            'overspray': DetectorFactory.create_overspray_detector(sensitivity),
            'void': DetectorFactory.create_void_detector(sensitivity),
            'debris_stripe': DetectorFactory.create_debris_stripe_detector(sensitivity)
        }


class IslandDetectionStrategy(DetectionStrategy):
    """Detection strategy for island images.

    Routes to `debris_island`, `line_defect`, and `overspray_island` detectors.
    """
    
    def get_required_detectors(self) -> List[str]:
        """Detectors to run for island images."""
        # Run debris_island and overspray_island detectors for island images
        return ['debris_island', 'line_defect', 'overspray_island']
    
    def create_detectors(self, sensitivity: str) -> Dict[str, Any]:
        """Create detector instances for island images."""
        # Return debris_island and overspray_island detectors for island images
        return {
            'debris_island': DetectorFactory.create_debris_island_detector(sensitivity),
            'overspray_island': DetectorFactory.create_overspray_island_detector(sensitivity),
            'line_defect': DetectorFactory.create_line_defect_detector(sensitivity)
        }


class NewPatternIslandDetectionStrategy(DetectionStrategy):
    """Detection strategy for new-pattern (dual-band) island images.

    Routes to the dual-band variants of the debris, overspray, and line-defect
    detectors. Reuses the same detector keys as `IslandDetectionStrategy` so the
    downstream dispatch and output naming are unchanged.
    """

    def get_required_detectors(self) -> List[str]:
        """Detectors to run for new-pattern island images."""
        return ['debris_island', 'line_defect', 'overspray_island']

    def create_detectors(self, sensitivity: str) -> Dict[str, Any]:
        """Create dual-band detector instances for new-pattern island images."""
        return {
            'debris_island': DetectorFactory.create_new_pattern_debris_island_detector(sensitivity),
            'overspray_island': DetectorFactory.create_new_pattern_overspray_island_detector(sensitivity),
            'line_defect': DetectorFactory.create_new_pattern_line_defect_detector(sensitivity)
        }


class UnknownDetectionStrategy(DetectionStrategy):
    """Detection strategy for unknown image types.

    Takes a conservative approach and only runs safe detectors.
    """
    
    def get_required_detectors(self) -> List[str]:
        """Detectors to run for unknown image types."""
        # Only run safe detectors for unknown types
        return ['surface_treatment']
    
    def create_detectors(self, sensitivity: str) -> Dict[str, Any]:
        """Create detector instances for unknown image types."""
        return {
            'surface_treatment': DetectorFactory.create_surface_treatment_detector(sensitivity),
        }


class DetectionStrategyFactory:
    """Factory for creating detection strategies based on `ImageType`."""
    
    _strategies = {
        ImageType.STRIPE: StripeDetectionStrategy,
        ImageType.ISLAND: IslandDetectionStrategy,
        ImageType.UNKNOWN: UnknownDetectionStrategy
    }
    
    @classmethod
    def create_strategy(cls, image_type: ImageType, pattern: str = 'legacy') -> DetectionStrategy:
        """Create an appropriate detection strategy instance.

        Args:
            image_type: The classified `ImageType` for the image.
            pattern: Island pattern selector. When 'new' and the image is an
                island, the dual-band strategy is used instead of the legacy
                single-band one.

        Returns:
            An instance of a `DetectionStrategy` implementation.
        """
        if image_type == ImageType.ISLAND and pattern == 'new':
            return NewPatternIslandDetectionStrategy()
        strategy_class = cls._strategies.get(image_type, UnknownDetectionStrategy)
        return strategy_class()


class SingleImageProcessor:
    """Process a single image through detectors chosen by image type.

    The image type is inferred from the filename and used to select the
    appropriate detection strategy and detectors.
    """
    
    def __init__(self, config: ProcessingConfig):
        """Initialize the processor.

        Args:
            config: Global processing configuration.
        """
        self.config = config
    
    def process_image(self, image_path: str, output_dir: str) -> ImageResult:
        """Dispatch a single image through the selected detectors.

        Args:
            image_path: Path to the input image file.
            output_dir: Directory in which to save per-image results.

        Returns:
            An `ImageResult` describing all detector outputs.
        """
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(image_output_dir, exist_ok=True)
        
        # Classify image type
        image_type = ImageTypeClassifier.classify_image(image_path)
        
        # Get detection strategy
        strategy = DetectionStrategyFactory.create_strategy(image_type, self.config.pattern)
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
        
        # Always process as full image
        file_size = os.path.getsize(image_path)
        return self._process_full_image(image_path, image_output_dir, detectors, 
                                       image_type, base_name, file_size)
    
    def _process_full_image(self, image_path: str, output_dir: str, detectors: Dict[str, Any],
                             image_type: ImageType, base_name: str, file_size: int) -> ImageResult:
        """Run all detectors on the full image and aggregate results.

        Args:
            image_path: Path to the input image.
            output_dir: Directory where visualizations and JSON will be saved.
            detectors: Mapping of detector key to detector instance to run.
            image_type: Classified image type.
            base_name: Base filename without extension for output naming.
            file_size: File size in bytes.

        Returns:
            `ImageResult` containing the consolidated detector outputs.
        """
        print(f"    Full image (size: {file_size/(1024*1024):.1f}MB), using normal processing...")
        
        detections = {}
        
        for detector_name, detector in detectors.items():
            print(f"    Running {detector_name} detection...")
            try:
                # Preprocess based on detector type
                if detector_name == 'surface_treatment':
                    original, gray, enhanced = ImagePreprocessor.preprocess_for_surface_treatment(image_path)
                    result_img, defects = detector.detect(enhanced)
                elif detector_name == 'stripe_misalignment':
                    # For stripe misalignment, pass the original image
                    # The detector has its own edge detection preprocessing
                    original, gray = ImagePreprocessor.load_and_convert_to_grayscale(image_path)
                    result_img, defects = detector.detect(original)
                    
                    # Save debug images if available
                    if hasattr(detector, 'save_debug_images'):
                        detector.save_debug_images(output_dir, base_name)
                elif detector_name == 'overspray':
                    # For overspray, pass the original image
                    # The detector has its own preprocessing for scatter detection
                    original, gray = ImagePreprocessor.load_and_convert_to_grayscale(image_path)
                    result_img, defects = detector.detect(original)
                elif detector_name == 'void':
                    # Void detection works directly on the original stripe image.
                    # It derives stripe and paper color references from the same image
                    # and outputs a TIFF-sized visualization with black bounding boxes.
                    original, gray = ImagePreprocessor.load_and_convert_to_grayscale(image_path)
                    result_img, defects = detector.detect(original, image_path)

                    if hasattr(detector, 'save_debug_images'):
                        detector.save_debug_images(output_dir, base_name)
                elif detector_name == 'debris_stripe':
                    # Stripe debris detection operates only on stripe images.
                    # It uses the original color image to find dark, near-black
                    # debris spots inside the colored stripe region.
                    original, gray = ImagePreprocessor.load_and_convert_to_grayscale(image_path)
                    result_img, defects = detector.detect(original, image_path)

                    if hasattr(detector, 'save_debug_images'):
                        detector.save_debug_images(output_dir, base_name)
                elif detector_name == 'debris_island':
                    # For debris island detection, pass the original image and image path
                    # The detector handles its own preprocessing and can load exclusion zones
                    original, gray = ImagePreprocessor.load_and_convert_to_grayscale(image_path)
                    result_img, defects = detector.detect(original, image_path)
                    
                    # Save debug images if available
                    if hasattr(detector, 'save_debug_images'):
                        detector.save_debug_images(output_dir, base_name)
                elif detector_name == 'overspray_island':
                    # For overspray island detection, pass the original image and image path
                    # The detector handles its own preprocessing and can load exclusion zones
                    original, gray = ImagePreprocessor.load_and_convert_to_grayscale(image_path)
                    result_img, defects = detector.detect(original, image_path)
                    
                    # Save debug images if available
                    if hasattr(detector, 'save_debug_images'):
                        detector.save_debug_images(output_dir, base_name)
                elif detector_name == 'line_defect':
                    # For line defect detection, pass the original image and image path
                    # Uses LineDetector internally to find lines, then scans for defects
                    original, gray = ImagePreprocessor.load_and_convert_to_grayscale(image_path)
                    result_img, defects = detector.detect(original, image_path)
                    
                    # Save debug images if available (separate missing/jagged visualizations)
                    if hasattr(detector, 'save_debug_images'):
                        detector.save_debug_images(output_dir, base_name)
                else:
                    # Fallback to original image loading
                    original, gray = ImagePreprocessor.load_and_convert_to_grayscale(image_path)
                    result_img, defects = detector.detect(original)
                
                # Save visualization
                vis_ext = ".tiff" if detector_name == 'void' else ".jpg"
                vis_path = os.path.join(output_dir, f"{detector_name}_visualization{vis_ext}")
                if detector_name == 'void':
                    operation_success = cv2.imwrite(vis_path, result_img)
                else:
                    operation_success = cv2.imwrite(vis_path, result_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if operation_success:
                    print(f"    Visualization saved to: {vis_path}")
                else:
                    print(f"    Failed to save visualization to: {vis_path}")
                
                # Clean defects for JSON serialization
                cleaned_defects = self._clean_defect_data(defects)
                
                detections[detector_name] = DetectionResult(
                    defect_count=len(defects),
                    visualization_path=vis_path,
                    defects=cleaned_defects
                )
                
            except Exception as e:
                print(f"    Error in {detector_name} detection: {str(e)}")
                detections[detector_name] = DetectionResult(
                    defect_count=0,
                    visualization_path=None,
                    defects=[],
                    error=str(e)
                )
        
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
        """Get file size in megabytes.

        Args:
            image_path: Path to the file.

        Returns:
            File size in megabytes as a float.
        """
        return os.path.getsize(image_path) / (1024 * 1024)
    
    def _clean_defect_data(self, defects: List[Dict]) -> List[Dict]:
        """Normalize detector outputs for JSON serialization.

        - Converts NumPy arrays and tuples to lists
        - Converts NumPy scalars to native Python numbers

        Args:
            defects: List of raw defect dictionaries from a detector.

        Returns:
            A list of JSON-serializable defect dictionaries.
        """
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


class ResultsSaver:
    """Utility functions to save per-image results and summary reports."""
    
    @staticmethod
    def save_image_result(result: ImageResult, output_dir: str) -> None:
        """Save a single image's `ImageResult` to JSON.

        Args:
            result: The `ImageResult` to serialize.
            output_dir: Directory where the JSON file will be written.
        """
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
        """Save a summary report (JSON, and optional PDF) for all images.

        Args:
            results: List of `ImageResult` objects.
            output_dir: Directory where summary artifacts will be saved.
            config: Pipeline configuration controlling report generation.
        """
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
        """Compute aggregate statistics across all processed images.

        Returns:
            A dictionary with keys: 'image_type_distribution',
            'defect_statistics', and 'processing_summary'.
        """
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
            'average_file_size_mb': 0
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
        
        processing_summary['average_file_size_mb'] = total_file_size / len(results) if results else 0
        
        return {
            'image_type_distribution': image_type_distribution,
            'defect_statistics': defect_statistics,
            'processing_summary': processing_summary
        }
    
    @staticmethod
    def _generate_pdf_report(results: List[ImageResult], stats: Dict[str, Any], 
                           output_dir: str) -> None:
        """Generate a human-readable PDF summary report.

        Args:
            results: Per-image results to include.
            stats: Precomputed aggregate statistics.
            output_dir: Directory to write the PDF to.
        """
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
        """Custom JSON serializer for NumPy and enum types."""
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
    """Orchestrates folder-level processing using the image processor and savers."""
    
    def __init__(self, config: ProcessingConfig):
        """Initialize the pipeline with global configuration.

        Args:
            config: Pipeline configuration.
        """
        self.config = config
        self.processor = SingleImageProcessor(config)
        self.results: List[ImageResult] = []
    
    def process_folder(self, input_folder: str) -> None:
        """Process all supported images found in a folder.

        Produces per-image JSON and visualization artifacts, and a folder-level
        summary JSON (and optional PDF).

        Args:
            input_folder: Directory containing images to process.
        """
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
            
            print(f"  [OK] Completed processing {result.image_name}")
        
        # Generate summary report
        ResultsSaver.save_summary_report(self.results, output_dir, self.config)
        
        print(f"\n[OK] Processing complete! Results saved to: {output_dir}")
        print(f"[INFO] Processed {len(self.results)} images total")
        
        # Print summary statistics
        self._print_summary_statistics()
    
    def _print_summary_statistics(self) -> None:
        """Print a concise summary of image counts and total defects to console."""
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
        
        print("\nSummary Statistics:")
        print("-" * 40)
        
        print("Image Types Processed:")
        for img_type, count in type_counts.items():
            print(f"  {img_type.title()}: {count} images")
        
        print("\nDefects Found:")
        for detector_name, count in detector_counts.items():
            print(f"  {detector_name.replace('_', ' ').title()}: {count} defects")


def main():
    """CLI entry point for running the defect detection pipeline.

    Parses command-line arguments and executes processing for the provided
    input folder, generating outputs under a timestamped directory.
    """
    parser = argparse.ArgumentParser(
        description='Run defect detection algorithms on images based on filename patterns'
    )
    parser.add_argument('--input_folder', '-i', required=True, 
                       help='Path to folder containing images')
    parser.add_argument('--output', '-o', 
                       help='Output base directory (default: creates output folder in input folder)')
    parser.add_argument('--generate_report', action='store_true',
                       help='Generate PDF report with all detections')
    parser.add_argument('--sensitivity', choices=['low', 'medium', 'high'], default='medium',
                       help='Detection sensitivity level (default: medium)')
    parser.add_argument('--pattern', choices=['legacy', 'new'], default='legacy',
                       help="Island pattern: 'legacy' single-band or 'new' dual-band (default: legacy)")
    
    args = parser.parse_args()
    
    # Validate input folder
    if not os.path.isdir(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist")
        sys.exit(1)
    
    # Create configuration
    config = ProcessingConfig(
        output_base_dir=args.output,
        sensitivity=args.sensitivity,
        generate_report=args.generate_report,
        pattern=args.pattern
    )
    
    # Create and run pipeline
    pipeline = DefectDetectionPipeline(config)
    
    print("=" * 60)
    print("Defect Detection Pipeline v2.0")
    print("=" * 60)
    print(f"Input folder: {args.input_folder}")
    print(f"Island pattern: {args.pattern}")
    print(f"Detection routing:")
    print(f"  - Stripe images: Stripe Misalignment, Overspray, Surface Treatment, Void, Debris Stripe")
    if args.pattern == 'new':
        print(f"  - Island images (dual-band): Debris Island, Overspray Island, Line Defect")
    else:
        print(f"  - Island images: Debris Island, Overspray Island, Line Defect")
    print(f"  - Unknown images: Surface Treatment")
    print("=" * 60)
    
    pipeline.process_folder(args.input_folder)


if __name__ == "__main__":
    main() 