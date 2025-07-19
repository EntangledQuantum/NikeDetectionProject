"""
Example Usage for Print Defect Detection System
Demonstrates how to use the detection algorithms

Author: Assistant
Date: 2024
"""

import os
import cv2
from overspray_detection import OversprayDetector
from surface_treatment_detection import SurfaceTreatmentDetector
from debris_detection import DebrisDetector


def example_single_detector():
    """Example of using a single detector"""
    print("Example 1: Using single detector (Overspray)")
    
    # Load image
    image_path = "path/to/your/image.png"
    # image = cv2.imread(image_path)
    
    # For demo, create a sample image
    import numpy as np
    image = np.ones((500, 500, 3), dtype=np.uint8) * 255
    # Add some black dots (simulating overspray)
    for _ in range(50):
        x, y = np.random.randint(50, 450, 2)
        cv2.circle(image, (x, y), 2, (0, 0, 0), -1)
    
    # Initialize detector
    detector = OversprayDetector(dot_size_range=(2, 10))
    
    # Detect defects
    result_image, defects = detector.detect(image)
    
    # Print results
    print(f"Found {len(defects)} overspray defects")
    for i, defect in enumerate(defects[:5]):  # Show first 5
        print(f"  Defect {i+1}: Location {defect['location']}, Size {defect['size']}")
    
    # Save result
    cv2.imwrite("overspray_result.png", result_image)
    print("Result saved to overspray_result.png")


def example_multiple_detectors():
    """Example of using multiple detectors on one image"""
    print("\nExample 2: Using multiple detectors")
    
    # Create sample image
    import numpy as np
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Add various defects
    # Overspray
    for _ in range(30):
        x, y = np.random.randint(50, 750, 2)
        cv2.circle(image, (x, y), 2, (0, 0, 0), -1)
    
    # Streak
    cv2.line(image, (100, 100), (700, 500), (0, 0, 0), 3)
    
    # Initialize detectors
    detectors = {
        'overspray': OversprayDetector(),
        'surface_treatment': SurfaceTreatmentDetector(),
        'debris': DebrisDetector()
    }
    
    # Run all detectors
    all_defects = {}
    for name, detector in detectors.items():
        result_image, defects = detector.detect(image)
        all_defects[name] = defects
        print(f"{name}: Found {len(defects)} defects")
    
    print(f"\nTotal defects found: {sum(len(d) for d in all_defects.values())}")


def example_custom_parameters():
    """Example of using detectors with custom parameters"""
    print("\nExample 3: Custom detector parameters")
    
    # High sensitivity overspray detection
    high_sensitivity = OversprayDetector(
        dot_size_range=(1, 20),  # Detect smaller and larger dots
        proximity_threshold=100   # Look further from printed areas
    )
    
    # Low sensitivity debris detection
    low_sensitivity = DebrisDetector(
        halo_threshold=50,       # Require stronger halos
        particle_size_range=(50, 1000)  # Only detect larger particles
    )
    
    print("Detectors configured with custom sensitivity")


def example_batch_processing():
    """Example of processing multiple images"""
    print("\nExample 4: Batch processing")
    
    # Setup
    input_folder = "path/to/images"
    output_folder = "path/to/output"
    
    # Example structure (would use actual folder in practice)
    print("Would process all images in input folder:")
    print("  - image1.png")
    print("  - image2.jpg")
    print("  - image3.tiff")
    print("And save results to output folder")
    
    # To actually run:
    # from run_all_detections import DefectDetectionPipeline
    # pipeline = DefectDetectionPipeline()
    # pipeline.process_folder(input_folder)


def example_visualization():
    """Example of custom visualization"""
    print("\nExample 5: Custom visualization")
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create sample image and detect defects
    image = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # Add some defects
    for _ in range(20):
        x, y = np.random.randint(50, 350, 2)
        cv2.circle(image, (x, y), 3, (0, 0, 0), -1)
    
    detector = OversprayDetector()
    result_image, defects = detector.detect(image)
    
    # Create custom visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Original image
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Detection result
    axes[1].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Detected {len(defects)} Defects")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("defect_comparison.png")
    print("Visualization saved to defect_comparison.png")


if __name__ == "__main__":
    print("Print Defect Detection System - Examples")
    print("=" * 50)
    
    # Run examples
    example_single_detector()
    example_multiple_detectors()
    example_custom_parameters()
    example_batch_processing()
    example_visualization()
    
    print("\n" + "=" * 50)
    print("Examples complete!")
    print("\nTo run on your images:")
    print("1. For single image: Use individual detector classes")
    print("2. For batch processing: python run_all_detections.py --input_folder /path/to/images") 