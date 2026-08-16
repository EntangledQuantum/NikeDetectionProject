import os
import time
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
from stripe_edge_roughness_detection import StripeEdgeRoughnessDetector
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
        pattern: Island pattern selector: 'legacy' | 'new'.
        only_detectors: Optional subset of detector keys to run. None runs
            every detector required by the image-type strategy.
        clear: If True, island detectors adapt to the clear scan material
            (gray background, fainter ink, lower SNR); thresholds are derived
            from the measured background level per image.
    """
    output_base_dir: Optional[str] = None
    sensitivity: str = 'medium'
    generate_report: bool = False
    pattern: str = 'legacy'  # island pattern: 'legacy' (single band) or 'new' (dual band)
    only_detectors: Optional[List[str]] = None
    clear: bool = False


@dataclass
class DetectionResult:
    """Standardized result payload produced by each detector.

    Attributes:
        defect_count: Number of defects detected.
        visualization_path: Path to a saved visualization image, if any.
        defects: Per-defect dictionaries with detector-specific fields.
        error: Optional error message if detection failed.
        elapsed_seconds: Wall-clock seconds for this detector (detect + save).
    """
    defect_count: int
    visualization_path: Optional[str]
    defects: List[Dict[str, Any]]
    error: Optional[str] = None
    elapsed_seconds: float = 0.0


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
        elapsed_seconds: Wall-clock seconds for all detectors on this image.
    """
    image_name: str
    image_path: str
    image_type: ImageType
    processing_time: str
    file_size_mb: float
    detectors_used: List[str]
    detections: Dict[str, DetectionResult]
    error: Optional[str] = None
    elapsed_seconds: float = 0.0


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
    def create_new_pattern_debris_island_detector(sensitivity: str,
                                                  clear: bool = False) -> NewPatternDebrisIslandDetector:
        """Create a `NewPatternDebrisIslandDetector` tuned by sensitivity.

        Args:
            sensitivity: 'low' | 'medium' | 'high'.
            clear: Adapt to the clear scan material (gray background).

        Returns:
            Configured `NewPatternDebrisIslandDetector` instance.
        """
        return NewPatternDebrisIslandDetector(sensitivity=sensitivity, debug=False,
                                              clear=clear)

    @staticmethod
    def create_new_pattern_overspray_island_detector(sensitivity: str,
                                                     clear: bool = False) -> NewPatternOversprayIslandDetector:
        """Create a `NewPatternOversprayIslandDetector` tuned by sensitivity.

        Args:
            sensitivity: 'low' | 'medium' | 'high'.
            clear: Adapt to the clear scan material (gray background).

        Returns:
            Configured `NewPatternOversprayIslandDetector` instance.
        """
        return NewPatternOversprayIslandDetector(sensitivity=sensitivity, debug=False,
                                                 clear=clear)

    @staticmethod
    def create_new_pattern_line_defect_detector(sensitivity: str,
                                                clear: bool = False) -> NewPatternLineDefectDetector:
        """Create a `NewPatternLineDefectDetector` tuned by sensitivity.

        Args:
            sensitivity: 'low' | 'medium' | 'high'.
            clear: Adapt to the clear scan material (gray background).

        Returns:
            Configured `NewPatternLineDefectDetector` instance.
        """
        return NewPatternLineDefectDetector(sensitivity=sensitivity, debug=False,
                                            clear=clear)

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

    @staticmethod
    def create_stripe_edge_roughness_detector(sensitivity: str) -> StripeEdgeRoughnessDetector:
        """Create a `StripeEdgeRoughnessDetector` tuned by sensitivity.

        Args:
            sensitivity: 'low' | 'medium' | 'high'.

        Returns:
            Configured `StripeEdgeRoughnessDetector` instance.
        """
        return StripeEdgeRoughnessDetector(
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
    def detector_factories(self, sensitivity: str) -> Dict[str, Any]:
        """Return name -> zero-arg factory callable for each detector."""
        pass

    def create_detectors(self, sensitivity: str,
                         only: Optional[List[str]] = None) -> Dict[str, Any]:
        """Instantiate detector objects keyed by their detector names.

        Args:
            sensitivity: Global sensitivity level for detector configuration.
            only: Optional subset of detector keys to construct. When None,
                every detector in ``detector_factories`` is built.

        Returns:
            Mapping from detector key to detector instance.
        """
        factories = self.detector_factories(sensitivity)
        keys = only if only is not None else list(factories.keys())
        return {k: factories[k]() for k in keys if k in factories}


class StripeDetectionStrategy(DetectionStrategy):
    """Detection strategy for stripe images.

    Currently routes to the `stripe_misalignment` detector.
    """
    
    def get_required_detectors(self) -> List[str]:
        """Detectors to run for stripe images."""
        return ['stripe_misalignment', 'overspray', 'surface_treatment', 'void',
                'debris_stripe', 'edge_roughness']
    
    def detector_factories(self, sensitivity: str) -> Dict[str, Any]:
        """Lazy factories for stripe detectors."""
        return {
            'stripe_misalignment': lambda: DetectorFactory.create_stripe_misalignment_detector(sensitivity),
            'overspray': lambda: DetectorFactory.create_overspray_detector(sensitivity),
            'void': lambda: DetectorFactory.create_void_detector(sensitivity),
            'debris_stripe': lambda: DetectorFactory.create_debris_stripe_detector(sensitivity),
            'surface_treatment': lambda: DetectorFactory.create_surface_treatment_detector(sensitivity),
            'edge_roughness': lambda: DetectorFactory.create_stripe_edge_roughness_detector(sensitivity),
        }


class IslandDetectionStrategy(DetectionStrategy):
    """Detection strategy for island images.

    Routes to `debris_island`, `line_defect`, and `overspray_island` detectors.
    """
    
    def get_required_detectors(self) -> List[str]:
        """Detectors to run for island images."""
        return ['debris_island', 'line_defect', 'overspray_island']
    
    def detector_factories(self, sensitivity: str) -> Dict[str, Any]:
        """Lazy factories for legacy island detectors."""
        return {
            'debris_island': lambda: DetectorFactory.create_debris_island_detector(sensitivity),
            'overspray_island': lambda: DetectorFactory.create_overspray_island_detector(sensitivity),
            'line_defect': lambda: DetectorFactory.create_line_defect_detector(sensitivity),
        }


class NewPatternIslandDetectionStrategy(DetectionStrategy):
    """Detection strategy for new-pattern (dual-band) island images.

    Routes to the dual-band variants of the debris, overspray, and line-defect
    detectors. Reuses the same detector keys as `IslandDetectionStrategy` so the
    downstream dispatch and output naming are unchanged.
    """

    def __init__(self, clear: bool = False):
        """Args:
            clear: Adapt detectors to the clear scan material (gray
                background, fainter ink, lower SNR).
        """
        self.clear = clear

    def get_required_detectors(self) -> List[str]:
        """Detectors to run for new-pattern island images."""
        return ['debris_island', 'line_defect', 'overspray_island']

    def detector_factories(self, sensitivity: str) -> Dict[str, Any]:
        """Lazy factories for dual-band island detectors."""
        return {
            'debris_island': lambda: DetectorFactory.create_new_pattern_debris_island_detector(sensitivity, self.clear),
            'overspray_island': lambda: DetectorFactory.create_new_pattern_overspray_island_detector(sensitivity, self.clear),
            'line_defect': lambda: DetectorFactory.create_new_pattern_line_defect_detector(sensitivity, self.clear),
        }


class UnknownDetectionStrategy(DetectionStrategy):
    """Detection strategy for unknown image types.

    Takes a conservative approach and only runs safe detectors.
    """
    
    def get_required_detectors(self) -> List[str]:
        """Detectors to run for unknown image types."""
        return ['surface_treatment']
    
    def detector_factories(self, sensitivity: str) -> Dict[str, Any]:
        """Lazy factories for unknown-image detectors."""
        return {
            'surface_treatment': lambda: DetectorFactory.create_surface_treatment_detector(sensitivity),
        }


class DetectionStrategyFactory:
    """Factory for creating detection strategies based on `ImageType`."""
    
    _strategies = {
        ImageType.STRIPE: StripeDetectionStrategy,
        ImageType.ISLAND: IslandDetectionStrategy,
        ImageType.UNKNOWN: UnknownDetectionStrategy
    }
    
    @classmethod
    def create_strategy(cls, image_type: ImageType, pattern: str = 'legacy',
                        clear: bool = False) -> DetectionStrategy:
        """Create an appropriate detection strategy instance.

        Args:
            image_type: The classified `ImageType` for the image.
            pattern: Island pattern selector. When 'new' and the image is an
                island, the dual-band strategy is used instead of the legacy
                single-band one.
            clear: Adapt island detectors to the clear scan material (only
                effective with the new-pattern island strategy).

        Returns:
            An instance of a `DetectionStrategy` implementation.
        """
        if image_type == ImageType.ISLAND and pattern == 'new':
            return NewPatternIslandDetectionStrategy(clear=clear)
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
        strategy = DetectionStrategyFactory.create_strategy(
            image_type, self.config.pattern, self.config.clear)
        required_detectors = strategy.get_required_detectors()

        # Optional --only filter: keep requested detectors that this strategy supports
        if self.config.only_detectors:
            allowed = set(self.config.only_detectors)
            unknown = allowed - set(required_detectors)
            if unknown:
                print(f"  [WARN] Ignoring detectors not available for "
                      f"{image_type.value}/{self.config.pattern}: {sorted(unknown)}")
            required_detectors = [d for d in required_detectors if d in allowed]
        
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
        
        # Create only the detectors we will actually run
        detectors = strategy.create_detectors(self.config.sensitivity,
                                              only=required_detectors)
        
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
        image_t0 = time.perf_counter()
        
        for detector_name, detector in detectors.items():
            print(f"    Running {detector_name} detection...")
            det_t0 = time.perf_counter()
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
                elif detector_name == 'edge_roughness':
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
                elapsed = time.perf_counter() - det_t0
                print(f"    {detector_name}: {elapsed:.2f}s, {len(defects)} defect(s)")
                
                detections[detector_name] = DetectionResult(
                    defect_count=len(defects),
                    visualization_path=vis_path,
                    defects=cleaned_defects,
                    elapsed_seconds=round(elapsed, 3),
                )
                
            except Exception as e:
                elapsed = time.perf_counter() - det_t0
                print(f"    Error in {detector_name} detection ({elapsed:.2f}s): {str(e)}")
                detections[detector_name] = DetectionResult(
                    defect_count=0,
                    visualization_path=None,
                    defects=[],
                    error=str(e),
                    elapsed_seconds=round(elapsed, 3),
                )
        
        image_elapsed = time.perf_counter() - image_t0
        print(f"    Image total: {image_elapsed:.2f}s")
        return ImageResult(
            image_name=base_name,
            image_path=image_path,
            image_type=image_type,
            processing_time=datetime.now().isoformat(),
            file_size_mb=file_size / (1024 * 1024),
            detectors_used=list(detectors.keys()),
            detections=detections,
            elapsed_seconds=round(image_elapsed, 3),
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
                          config: ProcessingConfig,
                          batch_elapsed_seconds: Optional[float] = None) -> None:
        """Save a summary report (JSON, and optional PDF) for all images.

        Args:
            results: List of `ImageResult` objects.
            output_dir: Directory where summary artifacts will be saved.
            config: Pipeline configuration controlling report generation.
            batch_elapsed_seconds: Optional wall-clock time for the whole batch.
        """
        # Calculate statistics
        stats = ResultsSaver._calculate_statistics(results)
        timing = ResultsSaver._calculate_timing(results, batch_elapsed_seconds)
        
        # Save JSON summary
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(results),
            'detection_sensitivity': config.sensitivity,
            'island_pattern': config.pattern,
            'image_type_distribution': stats['image_type_distribution'],
            'defect_statistics': stats['defect_statistics'],
            'processing_summary': stats['processing_summary'],
            'timing': timing,
            'detailed_results': [asdict(result) for result in results]
        }
        
        summary_path = os.path.join(output_dir, "defect_report.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=ResultsSaver._json_serializer)
        
        print(f"Summary report saved to: {summary_path}")
        
        # Generate PDF report if requested
        if config.generate_report:
            ResultsSaver._generate_pdf_report(results, stats, output_dir, timing)

    @staticmethod
    def _calculate_timing(results: List[ImageResult],
                          batch_elapsed_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Aggregate wall-clock timing by image type and by detector."""
        by_image_type: Dict[str, Dict[str, float]] = {}
        by_detector: Dict[str, Dict[str, float]] = {}
        per_image: List[Dict[str, Any]] = []

        for result in results:
            kind = result.image_type.value
            entry = by_image_type.setdefault(kind, {
                'total_seconds': 0.0,
                'image_count': 0,
            })
            entry['total_seconds'] += float(result.elapsed_seconds or 0.0)
            entry['image_count'] += 1

            per_image.append({
                'image_name': result.image_name,
                'image_type': kind,
                'elapsed_seconds': float(result.elapsed_seconds or 0.0),
                'detectors': {
                    name: float(det.elapsed_seconds or 0.0)
                    for name, det in result.detections.items()
                },
            })

            for name, det in result.detections.items():
                d = by_detector.setdefault(name, {
                    'total_seconds': 0.0,
                    'call_count': 0,
                })
                d['total_seconds'] += float(det.elapsed_seconds or 0.0)
                d['call_count'] += 1

        for entry in by_image_type.values():
            n = max(1, int(entry['image_count']))
            entry['total_seconds'] = round(entry['total_seconds'], 3)
            entry['average_seconds_per_image'] = round(
                entry['total_seconds'] / n, 3)

        for entry in by_detector.values():
            n = max(1, int(entry['call_count']))
            entry['total_seconds'] = round(entry['total_seconds'], 3)
            entry['average_seconds'] = round(entry['total_seconds'] / n, 3)

        out: Dict[str, Any] = {
            'by_image_type': by_image_type,
            'by_detector': by_detector,
            'per_image': per_image,
            'sum_of_image_seconds': round(
                sum(float(r.elapsed_seconds or 0.0) for r in results), 3),
        }
        if batch_elapsed_seconds is not None:
            out['batch_elapsed_seconds'] = round(float(batch_elapsed_seconds), 3)
        return out
    
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
            processing_summary['total_processing_time'] += float(
                result.elapsed_seconds or 0.0)
            
            for detector_name, detection in result.detections.items():
                if detector_name not in defect_statistics:
                    defect_statistics[detector_name] = {
                        'total_defects': 0,
                        'affected_images': 0,
                        'total_seconds': 0.0,
                    }
                
                defect_statistics[detector_name]['total_defects'] += detection.defect_count
                defect_statistics[detector_name]['total_seconds'] += float(
                    detection.elapsed_seconds or 0.0)
                if detection.defect_count > 0:
                    defect_statistics[detector_name]['affected_images'] += 1
        
        for det_stats in defect_statistics.values():
            det_stats['total_seconds'] = round(det_stats['total_seconds'], 3)

        processing_summary['total_processing_time'] = round(
            processing_summary['total_processing_time'], 3)
        processing_summary['average_file_size_mb'] = total_file_size / len(results) if results else 0
        
        return {
            'image_type_distribution': image_type_distribution,
            'defect_statistics': defect_statistics,
            'processing_summary': processing_summary
        }
    
    @staticmethod
    def _generate_pdf_report(results: List[ImageResult], stats: Dict[str, Any],
                           output_dir: str,
                           timing: Optional[Dict[str, Any]] = None) -> None:
        """Generate a human-readable PDF summary report.

        Args:
            results: Per-image results to include.
            stats: Precomputed aggregate statistics.
            output_dir: Directory to write the PDF to.
            timing: Optional wall-clock timing aggregates.
        """
        pdf_path = os.path.join(output_dir, "defect_detection_report.pdf")
        if timing is None:
            timing = ResultsSaver._calculate_timing(results)
        
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
                summary_text += f"  Total Time: {detector_stats.get('total_seconds', 0):.2f}s\n"
                if detector_stats['affected_images'] > 0:
                    avg = detector_stats['total_defects'] / detector_stats['affected_images']
                    summary_text += f"  Average per Affected Image: {avg:.2f}\n"

            summary_text += "\nTiming (wall-clock):\n" + "-" * 40 + "\n"
            if timing.get('batch_elapsed_seconds') is not None:
                summary_text += f"  Batch total: {timing['batch_elapsed_seconds']:.2f}s\n"
            for kind in ('stripe', 'island', 'unknown'):
                if kind in timing.get('by_image_type', {}):
                    t = timing['by_image_type'][kind]
                    summary_text += (
                        f"  {kind.title()}: {t['total_seconds']:.2f}s "
                        f"({t['image_count']} image(s), "
                        f"avg {t['average_seconds_per_image']:.2f}s)\n"
                    )
            summary_text += "  Per detector:\n"
            for name, t in sorted(timing.get('by_detector', {}).items(),
                                  key=lambda kv: -kv[1]['total_seconds']):
                summary_text += (
                    f"    {name}: {t['total_seconds']:.2f}s "
                    f"({t['call_count']} run(s))\n"
                )
            
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
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
    
    def _make_output_dir(self, parent: str) -> str:
        """Create a timestamped output directory under parent (or config override)."""
        if self.config.output_base_dir:
            output_dir = self.config.output_base_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(parent, f"output_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @staticmethod
    def _collect_image_files(input_folder: str, recursive: bool = True,
                             stripe_island_only: bool = True) -> List[str]:
        """Collect image paths under a folder for batch processing.

        Args:
            input_folder: Root directory to search.
            recursive: If True, walk subdirectories; else only the top level.
            stripe_island_only: If True, keep only filenames containing
                ``stripe`` or ``island`` (skips full-scan color TIFFs that
                would otherwise route to the unknown/surface-treatment path).

        Returns:
            Sorted list of absolute image paths.
        """
        image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')
        found: List[str] = []

        if recursive:
            for root, _dirs, files in os.walk(input_folder):
                for name in files:
                    if name.lower().endswith(image_extensions):
                        found.append(os.path.join(root, name))
        else:
            for name in os.listdir(input_folder):
                path = os.path.join(input_folder, name)
                if os.path.isfile(path) and name.lower().endswith(image_extensions):
                    found.append(path)

        if stripe_island_only:
            kept = []
            skipped = []
            for path in found:
                kind = ImageTypeClassifier.classify_image(path)
                if kind in (ImageType.STRIPE, ImageType.ISLAND):
                    kept.append(path)
                else:
                    skipped.append(os.path.basename(path))
            if skipped:
                print(f"  Skipping {len(skipped)} non-stripe/island file(s): "
                      f"{', '.join(skipped[:8])}"
                      f"{'...' if len(skipped) > 8 else ''}")
            found = kept

        return sorted(found)

    def process_single_image(self, image_path: str) -> None:
        """Process one island/stripe image (filename drives detector routing).

        Args:
            image_path: Path to a single image file.
        """
        image_path = os.path.abspath(image_path)
        parent = os.path.dirname(image_path) or '.'
        output_dir = self._make_output_dir(parent)

        image_type = ImageTypeClassifier.classify_image(image_path)
        print(f"Found 1 image to process ({image_type.value})")

        batch_t0 = time.perf_counter()
        result = self.processor.process_image(image_path, output_dir)
        batch_elapsed = time.perf_counter() - batch_t0
        self.results.append(result)

        image_output_dir = os.path.join(output_dir, result.image_name)
        ResultsSaver.save_image_result(result, image_output_dir)
        print(f"  [OK] Completed processing {result.image_name} "
              f"({result.elapsed_seconds:.2f}s)")

        ResultsSaver.save_summary_report(
            self.results, output_dir, self.config,
            batch_elapsed_seconds=batch_elapsed)

        print(f"\n[OK] Processing complete! Results saved to: {output_dir}")
        print(f"[INFO] Processed {len(self.results)} images total "
              f"in {batch_elapsed:.2f}s")
        self._print_summary_statistics(batch_elapsed_seconds=batch_elapsed)

    def process_folder(self, input_folder: str, recursive: bool = True,
                       stripe_island_only: bool = True) -> None:
        """Process all supported images found in a folder (optionally recursive).

        Filename heuristics route each file: ``*stripe*`` → stripe detectors,
        ``*island*`` → island detectors. By default only those names are kept
        so full-scan color columns in the same tree are skipped.

        Args:
            input_folder: Directory containing images to process.
            recursive: Walk subdirectories when True.
            stripe_island_only: Skip files that are neither stripe nor island.
        """
        input_folder = os.path.abspath(input_folder)
        output_dir = self._make_output_dir(input_folder)

        image_files = self._collect_image_files(
            input_folder, recursive=recursive,
            stripe_island_only=stripe_island_only)

        if not image_files:
            print(f"No matching image files found in {input_folder}")
            return

        n_stripe = sum(1 for p in image_files
                       if ImageTypeClassifier.classify_image(p) == ImageType.STRIPE)
        n_island = sum(1 for p in image_files
                       if ImageTypeClassifier.classify_image(p) == ImageType.ISLAND)
        print(f"Found {len(image_files)} images to process "
              f"({n_stripe} stripe, {n_island} island"
              f"{'' if stripe_island_only else ', +others'})"
              f"{' [recursive]' if recursive else ''})")

        batch_t0 = time.perf_counter()
        for img_path in tqdm(image_files, desc="Processing images"):
            result = self.processor.process_image(img_path, output_dir)
            self.results.append(result)

            image_output_dir = os.path.join(output_dir, result.image_name)
            ResultsSaver.save_image_result(result, image_output_dir)

            print(f"  [OK] Completed processing {result.image_name} "
                  f"({result.image_type.value}, {result.elapsed_seconds:.2f}s)")

        batch_elapsed = time.perf_counter() - batch_t0
        ResultsSaver.save_summary_report(
            self.results, output_dir, self.config,
            batch_elapsed_seconds=batch_elapsed)

        print(f"\n[OK] Processing complete! Results saved to: {output_dir}")
        print(f"[INFO] Processed {len(self.results)} images total "
              f"in {batch_elapsed:.2f}s")
        self._print_summary_statistics(batch_elapsed_seconds=batch_elapsed)

    def _print_summary_statistics(self,
                                  batch_elapsed_seconds: Optional[float] = None) -> None:
        """Print image counts, defects, and timing to the console."""
        if not self.results:
            return

        timing = ResultsSaver._calculate_timing(self.results, batch_elapsed_seconds)

        type_counts: Dict[str, int] = {}
        detector_counts: Dict[str, int] = {}
        for result in self.results:
            img_type = result.image_type.value
            type_counts[img_type] = type_counts.get(img_type, 0) + 1
            for detector_name, detection in result.detections.items():
                detector_counts[detector_name] = (
                    detector_counts.get(detector_name, 0) + detection.defect_count)

        print("\nSummary Statistics:")
        print("-" * 50)

        print("Image Types Processed:")
        for img_type, count in type_counts.items():
            print(f"  {img_type.title()}: {count} images")

        print("\nDefects Found:")
        for detector_name, count in detector_counts.items():
            print(f"  {detector_name.replace('_', ' ').title()}: {count} defects")

        print("\nTiming (wall-clock):")
        if timing.get('batch_elapsed_seconds') is not None:
            print(f"  Batch total: {timing['batch_elapsed_seconds']:.2f}s")
        by_type = timing.get('by_image_type', {})
        for kind in ('stripe', 'island', 'unknown'):
            if kind in by_type:
                t = by_type[kind]
                print(f"  {kind.title()}: {t['total_seconds']:.2f}s "
                      f"across {t['image_count']} image(s) "
                      f"(avg {t['average_seconds_per_image']:.2f}s/image)")
        print("  Per detector (sum over all images):")
        for name, t in sorted(timing.get('by_detector', {}).items(),
                              key=lambda kv: -kv[1]['total_seconds']):
            print(f"    {name}: {t['total_seconds']:.2f}s "
                  f"({t['call_count']} run(s), avg {t['average_seconds']:.2f}s)")
        print("  Per image:")
        for row in timing.get('per_image', []):
            print(f"    {row['image_name']} [{row['image_type']}]: "
                  f"{row['elapsed_seconds']:.2f}s")
            for det_name, det_sec in sorted(
                    row.get('detectors', {}).items(), key=lambda kv: -kv[1]):
                print(f"      - {det_name}: {det_sec:.2f}s")

# Detector keys accepted by --only (must match strategy keys)
_ALL_DETECTOR_KEYS = [
    'stripe_misalignment', 'overspray', 'surface_treatment', 'void', 'debris_stripe',
    'edge_roughness',
    'debris_island', 'overspray_island', 'line_defect',
]


def main():
    """CLI entry point for running the defect detection pipeline.

    ``-i`` accepts either a single island/stripe image file or a folder of
    them. Filename heuristics pick the detector strategy (``island`` /
    ``stripe`` / unknown); ``--pattern new`` switches island images to the
    dual-band detectors.
    """
    parser = argparse.ArgumentParser(
        description='Run defect detection on a single island/stripe image or a folder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single new-pattern island image (filename must contain 'island')
  python run_all_detections.py -i KeyIsland.tiff --pattern new

  # Only missing-nozzle / misalignment on that island
  python run_all_detections.py -i KeyIsland.tiff --pattern new --only line_defect

  # Single stripe image (filename must contain 'stripe')
  python run_all_detections.py -i blueStripe.tiff

  # Folder of extracted stripe + island crops (recursive; skips non-stripe/island)
  python run_all_detections.py -i /path/July_26 --pattern new -o /path/out
"""
    )
    parser.add_argument('--input', '-i', dest='input_path', default=None,
                       help='Path to a single image file OR a folder of images')
    # Alias kept for main_defect_detection.py and older scripts
    parser.add_argument('--input_folder', dest='input_path_alias', default=None,
                       help=argparse.SUPPRESS)
    parser.add_argument('--output', '-o',
                       help='Output directory (default: output_<timestamp> next to the input)')
    parser.add_argument('--generate_report', action='store_true',
                       help='Generate PDF report with all detections')
    parser.add_argument('--sensitivity', choices=['low', 'medium', 'high'], default='medium',
                       help='Detection sensitivity level (default: medium)')
    parser.add_argument('--pattern', choices=['legacy', 'new'], default='legacy',
                       help="Island pattern: 'legacy' single-band or 'new' dual-band (default: legacy)")
    parser.add_argument('--clear', action='store_true',
                       help='Clear scan material (gray background, fainter ink, lower SNR): '
                            'island thresholds are derived from the measured background '
                            'level per image. Requires --pattern new.')
    parser.add_argument('--only', nargs='+', metavar='DETECTOR', choices=_ALL_DETECTOR_KEYS,
                       help='Run only these detectors (subset of the strategy for this image type)')
    parser.add_argument('--no-recursive', action='store_true',
                       help='When input is a folder, only scan the top level (default: recursive)')
    parser.add_argument('--include-unknown', action='store_true',
                       help='When input is a folder, also process files that are neither '
                            '*stripe* nor *island* (default: skip them)')
    
    args = parser.parse_args()

    if args.clear and args.pattern != 'new':
        parser.error('--clear is only supported with --pattern new')

    input_path = args.input_path or args.input_path_alias
    if not input_path:
        parser.error('one of -i/--input or --input_folder is required')
    if not os.path.exists(input_path):
        print(f"Error: Input path '{input_path}' does not exist")
        sys.exit(1)

    is_file = os.path.isfile(input_path)
    is_dir = os.path.isdir(input_path)
    if not (is_file or is_dir):
        print(f"Error: Input path '{input_path}' is neither a file nor a folder")
        sys.exit(1)

    if is_file:
        ext = os.path.splitext(input_path)[1].lower()
        if ext not in ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'):
            print(f"Error: Unsupported image extension '{ext}'")
            sys.exit(1)
    
    # Create configuration
    config = ProcessingConfig(
        output_base_dir=args.output,
        sensitivity=args.sensitivity,
        generate_report=args.generate_report,
        pattern=args.pattern,
        only_detectors=args.only,
        clear=args.clear
    )
    
    # Create and run pipeline
    pipeline = DefectDetectionPipeline(config)
    
    print("=" * 60)
    print("Defect Detection Pipeline v2.0")
    print("=" * 60)
    print(f"Input: {input_path} ({'file' if is_file else 'folder'})")
    print(f"Island pattern: {args.pattern}"
          f"{' (clear material)' if args.clear else ''}")
    if args.only:
        print(f"Only detectors: {args.only}")
    if is_dir:
        print(f"Folder scan: "
              f"{'top-level only' if args.no_recursive else 'recursive'}; "
              f"{'all images' if args.include_unknown else 'stripe/island filenames only'}")
    print(f"Detection routing (by filename):")
    print(f"  - *stripe*: Stripe Misalignment, Edge Roughness, Overspray, Surface Treatment, Void, Debris Stripe")
    if args.pattern == 'new':
        print(f"  - *island* (dual-band): Debris Island, Overspray Island, Line Defect")
    else:
        print(f"  - *island*: Debris Island, Overspray Island, Line Defect")
    print(f"  - other: Surface Treatment")
    print("=" * 60)

    if is_file:
        pipeline.process_single_image(input_path)
    else:
        pipeline.process_folder(
            input_path,
            recursive=not args.no_recursive,
            stripe_island_only=not args.include_unknown,
        )


if __name__ == "__main__":
    main() 