#!/usr/bin/env python3
"""
Test script for the new smudge detection algorithm
Demonstrates the improved detection based on background consistency analysis
"""

import cv2
import numpy as np
import os
import sys
from typing import List, Dict, Any

# Add the scripts directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts', 'defects_detection'))

from smudge_detection import SmudgeDetector


def test_smudge_detection_on_image(image_path: str, sensitivity: str = 'medium') -> None:
    """
    Test the new smudge detection algorithm on a single image
    
    Args:
        image_path: Path to the test image
        sensitivity: Detection sensitivity ('low', 'medium', 'high')
    """
    print(f"\n🔍 Testing smudge detection on: {os.path.basename(image_path)}")
    print(f"   Sensitivity: {sensitivity}")
    
    # Load image
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file '{image_path}' not found!")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image '{image_path}'!")
        return
    
    print(f"   📊 Image size: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Configure detector based on sensitivity
    if sensitivity == 'low':
        detector = SmudgeDetector(
            min_smudge_size=200000,  # 450x450 pixels - very conservative
            background_window_size=120,
            lightness_threshold_factor=2.2,
            consistency_threshold=0.2,
            morphology_kernel_size=20
        )
    elif sensitivity == 'high':
        detector = SmudgeDetector(
            min_smudge_size=100000,  # 316x316 pixels - more sensitive
            background_window_size=80,
            lightness_threshold_factor=1.4,
            consistency_threshold=0.1,
            morphology_kernel_size=10
        )
    else:  # medium
        detector = SmudgeDetector()  # Use defaults
    
    # Run detection
    print("   🔄 Running smudge detection...")
    try:
        visualization, defects = detector.detect(image)
        
        # Print results
        print(f"   ✅ Detection completed!")
        print(f"   📈 Found {len(defects)} smudge(s)")
        
        if defects:
            print("\n   📋 Detected smudges:")
            for i, defect in enumerate(defects, 1):
                print(f"      {i}. Type: {defect['subtype']}")
                print(f"         Location: {defect['location']}")
                print(f"         Area: {defect['area']} pixels ({defect['area']/1000:.1f}K)")
                print(f"         Severity: {defect['severity']}")
                print(f"         Lightness ratio: {defect['lightness_ratio']:.2f}")
                if defect['area'] >= 160000:  # 400x400
                    equivalent_size = int(np.sqrt(defect['area']))
                    print(f"         Equivalent size: ~{equivalent_size}x{equivalent_size} pixels")
                print()
        
        # Save visualization
        output_dir = "smudge_test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_smudge_detection_{sensitivity}.jpg")
        
        cv2.imwrite(output_path, visualization, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"   💾 Visualization saved to: {output_path}")
        
    except Exception as e:
        print(f"   ❌ Error during detection: {str(e)}")
        import traceback
        traceback.print_exc()


def create_synthetic_test_image() -> str:
    """
    Create a synthetic test image with simulated smudges for testing
    """
    print("🎨 Creating synthetic test image...")
    
    # Create a base image with consistent background (simulating elephant gray)
    width, height = 1200, 800
    base_intensity = 120  # Gray background
    
    # Create base image with slight texture
    base_image = np.full((height, width), base_intensity, dtype=np.uint8)
    
    # Add subtle texture to simulate paper texture
    noise = np.random.normal(0, 5, (height, width))
    base_image = np.clip(base_image + noise, 0, 255).astype(np.uint8)
    
    # Add some simulated smudges (lighter areas)
    
    # Large smudge (fingerprint-like)
    center1 = (300, 200)
    size1 = 100
    for y in range(max(0, center1[1] - size1), min(height, center1[1] + size1)):
        for x in range(max(0, center1[0] - size1), min(width, center1[0] + size1)):
            dist = np.sqrt((x - center1[0])**2 + (y - center1[1])**2)
            if dist < size1:
                # Make it lighter (smudge effect)
                lightening = int(40 * (1 - dist / size1))
                base_image[y, x] = min(255, base_image[y, x] + lightening)
    
    # Medium smudge (directional smear)
    center2 = (700, 400)
    width2, height2 = 150, 50
    for y in range(max(0, center2[1] - height2), min(height, center2[1] + height2)):
        for x in range(max(0, center2[0] - width2), min(width, center2[0] + width2)):
            # Elliptical smudge
            dx = (x - center2[0]) / width2
            dy = (y - center2[1]) / height2
            if dx**2 + dy**2 < 1:
                lightening = int(30 * (1 - (dx**2 + dy**2)))
                base_image[y, x] = min(255, base_image[y, x] + lightening)
    
    # Small area that should NOT be detected (too small)
    center3 = (900, 600)
    size3 = 30
    for y in range(max(0, center3[1] - size3), min(height, center3[1] + size3)):
        for x in range(max(0, center3[0] - size3), min(width, center3[0] + size3)):
            dist = np.sqrt((x - center3[0])**2 + (y - center3[1])**2)
            if dist < size3:
                lightening = int(50 * (1 - dist / size3))
                base_image[y, x] = min(255, base_image[y, x] + lightening)
    
    # Convert to BGR for OpenCV
    test_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    
    # Save synthetic image
    os.makedirs("smudge_test_output", exist_ok=True)
    synthetic_path = "smudge_test_output/synthetic_test_image.jpg"
    cv2.imwrite(synthetic_path, test_image)
    
    print(f"   💾 Synthetic test image saved to: {synthetic_path}")
    print(f"   📊 Expected results:")
    print(f"      - Should detect 2 large smudges (fingerprint-like and directional)")
    print(f"      - Should NOT detect the small area (below 400x400 threshold)")
    
    return synthetic_path


def main():
    """Main test function"""
    print("🚀 Testing New Smudge Detection Algorithm")
    print("=" * 50)
    
    # Test on synthetic image first
    synthetic_path = create_synthetic_test_image()
    
    print("\n" + "="*50)
    print("Testing on synthetic image with different sensitivities:")
    
    for sensitivity in ['low', 'medium', 'high']:
        test_smudge_detection_on_image(synthetic_path, sensitivity)
    
    # Test on real images if available
    test_images_dir = "test_images"
    if os.path.exists(test_images_dir):
        print(f"\n" + "="*50)
        print(f"Testing on real images from {test_images_dir}:")
        
        image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')
        test_files = [f for f in os.listdir(test_images_dir)
                     if f.lower().endswith(image_extensions)]
        
        if test_files:
            for image_file in test_files[:3]:  # Test first 3 images
                image_path = os.path.join(test_images_dir, image_file)
                test_smudge_detection_on_image(image_path, 'medium')
        else:
            print("   📝 No test images found in test_images directory")
    else:
        print(f"\n   📝 No {test_images_dir} directory found, skipping real image tests")
    
    print("\n" + "="*50)
    print("✅ Testing completed!")
    print("📁 Check the 'smudge_test_output' directory for visualization results")
    print("\n🔧 Algorithm Summary:")
    print("   - Detects lighter areas on consistent backgrounds")
    print("   - Minimum size: 400x400 pixels (160,000 pixels)")
    print("   - Uses background consistency analysis")
    print("   - Color-coded by severity (Red=High, Orange=Medium, Yellow=Low)")


if __name__ == "__main__":
    main() 