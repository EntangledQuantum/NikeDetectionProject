"""
Example usage of Stripe Misalignment Detection
Demonstrates how to use the stripe misalignment detector independently

Author: Assistant
Date: 2024
"""

import cv2
import os
from stripe_misalignment_detection import StripeMisalignmentDetector


def main():
    # Configuration
    image_path = "path/to/your/stripe_image.tiff"  # Update with your image path
    output_dir = "stripe_misalignment_output"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create detector with different sensitivity levels
    sensitivities = ['low', 'medium', 'high']
    
    for sensitivity in sensitivities:
        print(f"\n{'='*60}")
        print(f"Running stripe misalignment detection with {sensitivity} sensitivity")
        print(f"{'='*60}")
        
        # Create detector
        detector = StripeMisalignmentDetector(
            sensitivity=sensitivity,
            debug=True  # Enable debug mode to see kernels and edge detection
        )
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return
        
        # Run detection
        visualization, defects = detector.detect(image)
        
        # Save results
        base_name = f"stripe_misalignment_{sensitivity}"
        
        # Save visualization
        vis_path = os.path.join(output_dir, f"{base_name}_visualization.jpg")
        cv2.imwrite(vis_path, visualization, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Saved visualization to: {vis_path}")
        
        # Save debug images if available
        debug_path = detector.save_debug_images(output_dir, base_name)
        if debug_path:
            print(f"Saved edge detection debug image to: {debug_path}")
        
        # Print results
        print(f"\nDetection Results:")
        print(f"Total misalignments found: {len(defects)}")
        
        if defects:
            print("\nMisalignment Details:")
            for i, defect in enumerate(defects, 1):
                print(f"  {i}. Y={defect['y']}, X delta={defect['x_delta']}px (threshold={defect['threshold']}px)")
        else:
            print("No misalignments detected.")
    
    print(f"\n{'='*60}")
    print("Example completed. Check the output directory for results.")


def test_custom_parameters():
    """Test with custom parameters instead of sensitivity presets"""
    print("\nTesting with custom parameters...")
    
    image_path = "path/to/your/stripe_image.tiff"  # Update with your image path
    output_dir = "stripe_misalignment_custom"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create detector with custom parameters
    detector = StripeMisalignmentDetector(
        kernel_size=40,              # Custom kernel size
        step_size=40,                # No overlap
        line_detection_threshold=0.12,  # 12% of pixels needed to detect line
        defect_threshold=8,          # 8 pixel delta considered misalignment
        debug=True                   # Enable debug visualization
    )
    
    # Load and process image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    visualization, defects = detector.detect(image)
    
    # Save results
    cv2.imwrite(os.path.join(output_dir, "custom_params_result.jpg"), 
                visualization, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    print(f"Custom parameter test completed. Found {len(defects)} misalignments.")


if __name__ == "__main__":
    # Run main example
    main()
    
    # Uncomment to test custom parameters
    # test_custom_parameters() 