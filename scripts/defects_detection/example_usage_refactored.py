"""
Example usage of the refactored defect detection pipeline
DO NOT RUN THIS CODE, THIS IS FOR REFERENCE ONLY
This example shows how to use the new image name-based routing system
"""

import os
import sys

# Add the current directory to path so we can import the modules
sys.path.append(os.path.dirname(__file__))

from run_all_detections import DefectDetectionPipeline, ProcessingConfig


def main():
    """Example usage of the refactored pipeline"""
    
    # Example 1: Basic usage with default settings
    print("Example 1: Basic usage")
    config = ProcessingConfig(
        sensitivity='medium',
        generate_report=True
    )
    
    pipeline = DefectDetectionPipeline(config)
    
    # Replace with your actual image folder path
    # image_folder = "path/to/your/images"
    # pipeline.process_folder(image_folder)
    
    print("Pipeline created successfully!")
    print("Image routing:")
    print("- Files with 'stripe' in name: Surface Treatment + Debris detection")
    print("- Files with 'island' in name: No detectors (Overspray disabled)")
    print("- Other files: Surface Treatment + Debris detection")
    
    
    # Example 2: High sensitivity processing
    print("\nExample 2: High sensitivity processing")
    high_sensitivity_config = ProcessingConfig(
        sensitivity='high',
        generate_report=True,
        window_size=1024,  # Smaller windows for faster processing
        overlap=128
    )
    
    high_sensitivity_pipeline = DefectDetectionPipeline(high_sensitivity_config)
    print("High sensitivity pipeline created!")
    
    
    # Example 3: Memory-efficient processing for large files
    print("\nExample 3: Memory-efficient configuration")
    memory_efficient_config = ProcessingConfig(
        sensitivity='medium',
        generate_report=False,  # Skip PDF to save memory
        window_size=2048,
        overlap=256,
        large_file_threshold=10 * 1024 * 1024  # 10MB threshold
    )
    
    memory_efficient_pipeline = DefectDetectionPipeline(memory_efficient_config)
    print("Memory-efficient pipeline created!")


if __name__ == "__main__":
    main() 