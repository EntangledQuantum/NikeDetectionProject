"""
Detection Pipeline Orchestrator

This module contains the main pipeline that orchestrates the entire
defect detection process with support for parallel processing.

Author: Refactored Architecture
Date: 2024
"""

import os
import time
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

from .data_models import ImageResult, ProcessingConfig, ImageType
from .image_processing import ImageTypeClassifier, TiffImageLoader
from .detection_strategies import DetectionStrategyFactory
from .optimized_detector import OptimizedCombinedDetector
from .clustering import DefectClusterer, ClusterVisualizer
from .results_manager import ResultsSaver


class SingleImageProcessor:
    """Processes a single image through appropriate detectors with optimization"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.clusterer = DefectClusterer(eps=50.0, min_samples=2, max_cluster_distance=100.0)
        self.visualizer = ClusterVisualizer()
    
    def process_image(self, image_path: str, output_dir: str) -> ImageResult:
        """Process a single image through optimized detectors"""
        start_time = time.time()
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(image_output_dir, exist_ok=True)
        
        # Classify image type
        image_type = ImageTypeClassifier.classify_image(image_path)
        
        # Get detection strategy
        strategy = DetectionStrategyFactory.create_strategy(image_type, self.config.selected_detectors)
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
        
        if self.config.verbose_timing:
            print(f"\n  🖼️  Processing: {base_name} (type: {image_type.value})")
            print(f"    Required detectors: {required_detectors}")
        
        # Get file size for reporting
        file_size = os.path.getsize(image_path)
        if self.config.verbose_timing:
            print(f"    📊 Image size: {file_size/(1024*1024):.1f}MB")
        
        # Load image once
        load_start = time.time()
        original = TiffImageLoader.load_image(image_path)
        if original is None:
            raise ValueError(f"Could not read image: {image_path}")
        load_time = time.time() - load_start
        
        if self.config.verbose_timing:
            print(f"    ⏱️  Image loaded: {original.shape} pixels in {load_time:.3f}s")
        
        # Create detector configs for OptimizedCombinedDetector
        detector_configs = strategy.create_detectors(self.config.sensitivity)
        
        # Use optimized combined detector
        combined_detector = OptimizedCombinedDetector(detector_configs, self.config.verbose_timing)
        
        # Run all detections with shared preprocessing
        detection_start = time.time()
        detector_results = combined_detector.detect_all(original)
        detection_time = time.time() - detection_start
        
        # Process results
        detections = {}
        all_defects = []
        
        for detector_name, (visualization, defects, timing_info) in detector_results.items():
            # Add detector name to each defect for clustering
            for defect in defects:
                defect['detector'] = detector_name
            
            # Collect defects for clustering
            all_defects.extend(defects)
            
            from .data_models import DetectionResult
            detections[detector_name] = DetectionResult(
                defect_count=len(defects),
                visualization_path=None,  # Will be set after clustering if needed
                defects=defects,
                timing=timing_info
            )
        
        # Apply clustering to all defects
        clustering_start = time.time()
        clustered_defects = self.clusterer.cluster_defects_by_type(all_defects)
        clustering_time = time.time() - clustering_start
        
        total_clusters = sum(len(clusters) for clusters in clustered_defects.values())
        if self.config.verbose_timing:
            print(f"    🔗 Clustering: {total_clusters} clusters from {len(all_defects)} defects in {clustering_time:.3f}s")
        
        # Create single combined visualization
        viz_start = time.time()
        if clustered_defects:
            # Convert ClusterInfo objects to compatible format for visualization
            compatible_clustered_defects = {}
            for defect_type, clusters in clustered_defects.items():
                compatible_clustered_defects[defect_type] = []
                for cluster in clusters:
                    compatible_clustered_defects[defect_type].append({
                        'cluster_id': cluster.cluster_id,
                        'defect_type': cluster.defect_type,
                        'defects': cluster.defects,
                        'centroid': cluster.centroid,
                        'bounding_box': cluster.bounding_box,
                        'cluster_size': cluster.cluster_size,
                        'cluster_area': cluster.cluster_area
                    })
            
            combined_visualization = self.visualizer.create_cluster_visualization(original, compatible_clustered_defects)
            combined_vis_path = os.path.join(image_output_dir, "all_defects_visualization.jpg")
            
            import cv2
            cv2.imwrite(combined_vis_path, combined_visualization, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Update all detection results to point to the combined visualization
            for detector_name in detections.keys():
                if detections[detector_name].defect_count > 0:
                    detections[detector_name] = DetectionResult(
                        defect_count=detections[detector_name].defect_count,
                        visualization_path=combined_vis_path,
                        defects=detections[detector_name].defects,
                        error=detections[detector_name].error,
                        timing=detections[detector_name].timing
                    )
        
        viz_time = time.time() - viz_start
        
        # Save clustering information
        if clustered_defects:
            cluster_info_path = os.path.join(image_output_dir, "cluster_information.json")
            try:
                import json
                from .data_models import to_dict
                
                # Convert clustered_defects to serializable format
                serializable_clusters = {}
                for defect_type, clusters in clustered_defects.items():
                    serializable_clusters[defect_type] = [to_dict(cluster) for cluster in clusters]
                
                with open(cluster_info_path, 'w') as f:
                    json.dump(serializable_clusters, f, indent=2, default=ResultsSaver._json_serializer)
            except Exception as e:
                if self.config.verbose_timing:
                    print(f"      ⚠️  Warning: Could not save cluster information: {e}")
        
        total_time = time.time() - start_time
        
        if self.config.verbose_timing:
            print(f"    📈 Total processing time: {total_time:.3f}s")
            print(f"       - Image loading: {load_time:.3f}s")
            print(f"       - Detection: {detection_time:.3f}s")
            print(f"       - Clustering: {clustering_time:.3f}s")
            print(f"       - Visualization: {viz_time:.3f}s")
        
        return ImageResult(
            image_name=base_name,
            image_path=image_path,
            image_type=image_type,
            processing_time=datetime.now().isoformat(),
            file_size_mb=file_size / (1024 * 1024),
            detectors_used=list(detector_configs.keys()),
            detections=detections
        )
    
    def _get_file_size_mb(self, image_path: str) -> float:
        """Get file size in MB"""
        return os.path.getsize(image_path) / (1024 * 1024)


# Global function for parallel processing
def process_single_image_parallel(args):
    """
    Function for parallel processing of single images
    This is defined at module level for picklability
    """
    image_path, output_dir, config = args
    
    try:
        processor = SingleImageProcessor(config)
        return processor.process_image(image_path, output_dir)
    except Exception as e:
        import traceback
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return ImageResult(
            image_name=base_name,
            image_path=image_path,
            image_type=ImageType.UNKNOWN,
            processing_time=datetime.now().isoformat(),
            file_size_mb=0.0,
            detectors_used=[],
            detections={},
            error=f"Error in parallel processing: {str(e)} - {traceback.format_exc()}"
        )


class DefectDetectionPipeline:
    """Main pipeline orchestrating the optimized detection process"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.processor = SingleImageProcessor(config)
        self.results: List[ImageResult] = []
    
    def process_folder(self, input_folder: str) -> None:
        """Process all images in a folder with parallel processing support"""
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
        
        print(f"📋 Found {len(image_files)} images to process")
        
        if self.config.enable_parallel_processing and len(image_files) > 1:
            self._process_parallel(input_folder, image_files, output_dir)
        else:
            self._process_sequential(input_folder, image_files, output_dir)
        
        # Generate summary report
        ResultsSaver.save_summary_report(self.results, output_dir, self.config)
        
        print(f"\n✅ Processing complete! Results saved to: {output_dir}")
        print(f"📊 Processed {len(self.results)} images total")
        
        # Print summary statistics
        self._print_summary_statistics()
    
    def _process_parallel(self, input_folder: str, image_files: List[str], output_dir: str) -> None:
        """Process images in parallel"""
        print(f"🚀 Using parallel processing with {self.config.max_workers} workers")
        
        # Prepare arguments for parallel processing
        args_list = [(os.path.join(input_folder, img_file), output_dir, self.config) 
                     for img_file in image_files]
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_image = {executor.submit(process_single_image_parallel, args): args[0] 
                              for args in args_list}
            
            # Process completed tasks
            for future in as_completed(future_to_image):
                image_path = future_to_image[future]
                try:
                    result = future.result()
                    self.results.append(result)
                    
                    # Save individual result
                    image_output_dir = os.path.join(output_dir, result.image_name)
                    ResultsSaver.save_image_result(result, image_output_dir)
                    
                    if self.config.verbose_timing:
                        print(f"  ✅ Completed {result.image_name}")
                    
                except Exception as e:
                    print(f"  ❌ Error processing {os.path.basename(image_path)}: {str(e)}")
    
    def _process_sequential(self, input_folder: str, image_files: List[str], output_dir: str) -> None:
        """Process images sequentially"""
        print("🔄 Using sequential processing")
        
        # Process each image
        for img_file in tqdm(image_files, desc="Processing images"):
            img_path = os.path.join(input_folder, img_file)
            result = self.processor.process_image(img_path, output_dir)
            self.results.append(result)
            
            # Save individual result
            image_output_dir = os.path.join(output_dir, result.image_name)
            ResultsSaver.save_image_result(result, image_output_dir)
            
            if self.config.verbose_timing:
                print(f"  ✅ Completed processing {result.image_name}")
    
    def process_single_file(self, input_file: str) -> None:
        """Process a single image file"""
        # Validate file is an image
        image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')
        if not input_file.lower().endswith(image_extensions):
            print(f"Error: '{input_file}' is not a supported image format")
            print(f"Supported formats: {', '.join(image_extensions)}")
            return
        
        # Create output directory next to the input file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_dir = os.path.dirname(input_file)
        input_name = os.path.splitext(os.path.basename(input_file))[0]
        output_dir = os.path.join(input_dir, f"{input_name}_output_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📋 Processing single file: {os.path.basename(input_file)}")
        
        # Process the single image
        result = self.processor.process_image(input_file, output_dir)
        self.results.append(result)
        
        # Save individual result
        image_output_dir = os.path.join(output_dir, result.image_name)
        ResultsSaver.save_image_result(result, image_output_dir)
        
        # Generate summary report
        ResultsSaver.save_summary_report(self.results, output_dir, self.config)
        
        print(f"\n✅ Processing complete! Results saved to: {output_dir}")
        print(f"📊 Processed 1 image")
        
        # Print summary statistics
        self._print_summary_statistics()

    def process_input(self, input_path: str) -> None:
        """Process input - automatically detects if it's a file or folder"""
        if os.path.isfile(input_path):
            self.process_single_file(input_path)
        elif os.path.isdir(input_path):
            self.process_folder(input_path)
        else:
            print(f"Error: '{input_path}' is neither a file nor a directory")
    
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