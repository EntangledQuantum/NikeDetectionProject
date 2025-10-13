#!/usr/bin/env python3
"""
Test script for validating the improved defect detection algorithm
Place your test images in the test_images folder and run this script

Expected results:
- blackStripe-smudge.jpg: Should detect fingerprint smudge
- blueStripe-smudge.jpg: Should detect fingerprint smudge  
- blackStripe-void.jpg: Should detect significant white void
- blackStripe-st-non-defect.jpg: Should NOT detect any defects (clean surface)
"""

import os
import sys
import cv2
import numpy as np

# Add the scripts directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts', 'defects_detection'))

def test_algorithm():
    """Test the improved algorithm on sample images"""
    
    # Expected results for validation
    expected_results = {
        'blackStripe-smudge': {'should_detect': True, 'expected_type': 'smudge', 'description': 'Fingerprint smudge'},
        'blueStripe-smudge': {'should_detect': True, 'expected_type': 'smudge', 'description': 'Fingerprint smudge'},
        'blackStripe-void': {'should_detect': True, 'expected_type': 'void', 'description': 'Significant white void'},
        'blackStripe-st-non-defect': {'should_detect': False, 'expected_type': None, 'description': 'Clean surface - no defects'}
    }
    
    test_images_dir = 'test_images'
    
    if not os.path.exists(test_images_dir):
        print(f"❌ Test images directory '{test_images_dir}' not found!")
        print("\n📋 To test the algorithm:")
        print("1. Create a 'test_images' folder")
        print("2. Add your test images with the following names:")
        for img_name, details in expected_results.items():
            print(f"   - {img_name}.jpg (or .png): {details['description']}")
        print("3. Run this script again")
        return
    
    # Look for test images
    image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']
    test_files = []
    
    for ext in image_extensions:
        for img_name in expected_results.keys():
            img_path = os.path.join(test_images_dir, img_name + ext)
            if os.path.exists(img_path):
                test_files.append((img_name, img_path))
                break
    
    if not test_files:
        print(f"❌ No test images found in '{test_images_dir}'!")
        print("\n📋 Expected image names:")
        for img_name in expected_results.keys():
            print(f"   - {img_name}.(jpg|png|tiff)")
        return
    
    print("🧪 Testing Improved Defect Detection Algorithm")
    print("=" * 60)
    
    # Test each image
    for img_name, img_path in test_files:
        print(f"\n📸 Testing: {img_name}")
        print(f"   Path: {img_path}")
        
        expected = expected_results[img_name]
        print(f"   Expected: {'DETECT ' + expected['expected_type'] if expected['should_detect'] else 'NO DEFECTS'}")
        print(f"   Description: {expected['description']}")
        
        # Load image to verify it exists and is readable
        image = cv2.imread(img_path)
        if image is None:
            print(f"   ❌ Error: Could not read image!")
            continue
            
        print(f"   ✅ Image loaded: {image.shape} pixels")
        
        # Here you would run the actual detection algorithm
        # For now, we'll just show what the test expects
        if expected['should_detect']:
            print(f"   🔍 This image should be detected as {expected['expected_type']}")
        else:
            print(f"   ✅ This image should show NO defects (clean surface)")
    
    print("\n" + "=" * 60)
    print("🚀 To run the actual algorithm:")
    print(f"   cd scripts/defects_detection")
    print(f"   python run_all_detections.py --input_folder ../../test_images")
    print("\n📊 Check the results in the generated output folder!")
    print("   - Look for 'all_defects_visualization.jpg' in each image folder")
    print("   - Check the JSON reports for defect counts")
    print("   - Verify timing performance improvements")

def main():
    """Main test function"""
    print("🔬 Defect Detection Algorithm Validation")
    print("Version: 3.0 - Optimized with Conservative Sensitivity")
    print()
    
    # Show improvements made
    print("🔧 Recent Improvements:")
    print("   ✅ Parallel processing for faster execution")
    print("   ✅ Shared preprocessing to reduce redundant operations")
    print("   ✅ More conservative sensitivity to reduce false positives")
    print("   ✅ Improved smudge detection for fingerprints")
    print("   ✅ Better void detection with contrast analysis")
    print("   ✅ Enhanced debris detection with adaptive thresholding")
    print("   ✅ Single combined visualization instead of multiple files")
    print("   ✅ Detailed timing analysis for performance monitoring")
    print()
    
    test_algorithm()

if __name__ == "__main__":
    main() 