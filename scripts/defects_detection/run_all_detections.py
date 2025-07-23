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

# Import detection modules
from surface_treatment_detection import SurfaceTreatmentDetector
from debris_detection import DebrisDetector
from line_defect_detection import LineDefectDetector


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
    window_size: int = 2048
    overlap: int = 256
    max_workers: int = 1
    large_file_threshold: int = 20 * 1024 * 1024  # 20MB


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


class DetectorFactory:
    """Factory for creating appropriate detectors based on sensitivity"""
    
    @staticmethod
    def create_surface_treatment_detector(sensitivity: str) -> SurfaceTreatmentDetector:
        """Create surface treatment detector with appropriate settings"""
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
    def create_debris_detector(sensitivity: str) -> DebrisDetector:
        """Create debris detector with appropriate settings"""
        if sensitivity == 'low':
            return DebrisDetector(
                halo_threshold=40, 
                region_size_range=(150, 1500), 
                kernel_size=18
            )
        elif sensitivity == 'high':
            return DebrisDetector(
                halo_threshold=18, 
                region_size_range=(50, 4500), 
                kernel_size=22
            )
        else:  # medium
            return DebrisDetector(
                region_size_range=(100, 1200), 
                kernel_size=15
            )
    
    @staticmethod
    def create_line_defect_detector(sensitivity: str) -> LineDefectDetector:
        """Create line defect detector with appropriate settings"""
        # Enable debug mode for high sensitivity to see detected lines
        debug = False
        return LineDefectDetector(sensitivity=sensitivity, debug=debug)


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
        return ['surface_treatment', 'debris']
    
    def create_detectors(self, sensitivity: str) -> Dict[str, Any]:
        return {
            'surface_treatment': DetectorFactory.create_surface_treatment_detector(sensitivity),
            'debris': DetectorFactory.create_debris_detector(sensitivity)
        }


class IslandDetectionStrategy(DetectionStrategy):
    """Detection strategy for island images"""
    
    def get_required_detectors(self) -> List[str]:
        # Only run line defect detector for island images
        return ['line_defect']
    
    def create_detectors(self, sensitivity: str) -> Dict[str, Any]:
        # Return line defect detector for island images
        return {
            'line_defect': DetectorFactory.create_line_defect_detector(sensitivity)
        }


class UnknownDetectionStrategy(DetectionStrategy):
    """Detection strategy for unknown image types"""
    
    def get_required_detectors(self) -> List[str]:
        # Only run safe detectors for unknown types
        return ['surface_treatment', 'debris']
    
    def create_detectors(self, sensitivity: str) -> Dict[str, Any]:
        return {
            'surface_treatment': DetectorFactory.create_surface_treatment_detector(sensitivity),
            'debris': DetectorFactory.create_debris_detector(sensitivity)
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
        
        # Always process as full image
        file_size = os.path.getsize(image_path)
        return self._process_full_image(image_path, image_output_dir, detectors, 
                                       image_type, base_name, file_size)
    
    def _process_full_image(self, image_path: str, output_dir: str, detectors: Dict[str, Any],
                             image_type: ImageType, base_name: str, file_size: int) -> ImageResult:
        """Process full-sized images"""
        print(f"    Full image (size: {file_size/(1024*1024):.1f}MB), using normal processing...")
        
        detections = {}
        
        for detector_name, detector in detectors.items():
            print(f"    Running {detector_name} detection...")
            try:
                # Preprocess based on detector type
                if detector_name == 'surface_treatment':
                    original, gray, enhanced = ImagePreprocessor.preprocess_for_surface_treatment(image_path)
                    result_img, defects = detector.detect(enhanced)
                elif detector_name == 'debris':
                    original, gray, denoised, enhanced = ImagePreprocessor.preprocess_for_debris(image_path)
                    result_img, defects = detector.detect(enhanced)
                else:
                    # Fallback to original image loading
                    original, gray = ImagePreprocessor.load_and_convert_to_grayscale(image_path)
                    result_img, defects = detector.detect(original)
                
                # Save visualization
                vis_path = os.path.join(output_dir, f"{detector_name}_visualization.jpg")
                cv2.imwrite(vis_path, result_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
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
    print("Defect Detection Pipeline v2.0")
    print("=" * 60)
    print(f"Input folder: {args.input_folder}")
    print(f"Detection routing:")
    print(f"  - Stripe images: Surface Treatment, Debris")
    print(f"  - Island images: (Overspray disabled)")
    print(f"  - Unknown images: Surface Treatment, Debris")
    print("=" * 60)
    
    pipeline.process_folder(args.input_folder)


if __name__ == "__main__":
    main() 