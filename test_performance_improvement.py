#!/usr/bin/env python3
"""
Performance comparison between old and new smudge detection methods
"""

import cv2
import numpy as np
import time
import sys
import os

# Add the scripts directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'scripts', 'defects_detection'))

from smudge_detection import SmudgeDetector


def create_test_image(size=(2000, 1500)):
    """Create a test image with consistent background and some smudges"""
    height, width = size
    
    # Create consistent gray background
    background = np.full((height, width), 120, dtype=np.uint8)
    
    # Add some texture noise
    noise = np.random.normal(0, 3, (height, width))
    background = np.clip(background + noise, 0, 255).astype(np.uint8)
    
    # Add a few smudges (lighter areas)
    smudge_centers = [(500, 400), (1200, 800), (800, 1100)]
    
    for center in smudge_centers:
        cx, cy = center
        size_x, size_y = 250, 180
        
        for y in range(max(0, cy - size_y), min(height, cy + size_y)):
            for x in range(max(0, cx - size_x), min(width, cx + size_x)):
                dx = (x - cx) / size_x
                dy = (y - cy) / size_y
                dist = dx*dx + dy*dy
                
                if dist <= 1.0:
                    lightening = int(40 * (1 - dist))
                    background[y, x] = min(255, background[y, x] + lightening)
    
    return cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)


def benchmark_method(detector, image, method_name):
    """Benchmark a detection method"""
    print(f"\n🔄 Testing {method_name}...")
    
    start_time = time.time()
    try:
        visualization, defects = detector.detect(image)
        total_time = time.time() - start_time
        
        print(f"   ✅ {method_name} completed in {total_time:.3f}s")
        print(f"   📈 Found {len(defects)} smudge(s)")
        
        return total_time, len(defects)
        
    except Exception as e:
        print(f"   ❌ {method_name} failed: {str(e)}")
        return None, 0


def main():
    """Performance comparison test"""
    print("🚀 Smudge Detection Performance Comparison")
    print("=" * 50)
    
    # Create test image
    print("🎨 Creating test image (2000x1500 pixels)...")
    test_image = create_test_image()
    print(f"   📊 Test image size: {test_image.shape[1]}×{test_image.shape[0]} pixels")
    
    # Save test image
    os.makedirs("performance_test_output", exist_ok=True)
    cv2.imwrite("performance_test_output/test_image.jpg", test_image)
    print("   💾 Test image saved: performance_test_output/test_image.jpg")
    
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON")
    print("="*50)
    
    # Test 1: Old method (without FFT)
    detector_old = SmudgeDetector(use_fft_analysis=False)
    time_old, detections_old = benchmark_method(detector_old, test_image, "OLD METHOD (Box Filter)")
    
    # Test 2: New FFT method
    detector_fft = SmudgeDetector(use_fft_analysis=True)
    time_fft, detections_fft = benchmark_method(detector_fft, test_image, "NEW METHOD (FFT-based)")
    
    # Calculate improvement
    if time_old and time_fft:
        speedup = time_old / time_fft
        time_saved = time_old - time_fft
        
        print(f"\n📊 PERFORMANCE RESULTS:")
        print(f"   Old Method:     {time_old:.3f}s")
        print(f"   New FFT Method: {time_fft:.3f}s")
        print(f"   ⚡ Speedup:     {speedup:.1f}x faster")
        print(f"   ⏱️  Time Saved:  {time_saved:.3f}s ({(time_saved/time_old)*100:.1f}% faster)")
        
        print(f"\n🎯 ACCURACY COMPARISON:")
        print(f"   Old Method detections: {detections_old}")
        print(f"   New Method detections: {detections_fft}")
        
        if abs(detections_old - detections_fft) <= 1:
            print("   ✅ Detection accuracy: Equivalent")
        else:
            print("   ⚠️  Detection difference noted")
        
        print(f"\n💡 CONCLUSION:")
        if speedup > 2:
            print(f"   🚀 Significant improvement! {speedup:.1f}x faster with FFT method")
        elif speedup > 1.2:
            print(f"   ⚡ Good improvement! {speedup:.1f}x faster with FFT method")
        else:
            print(f"   📝 Modest improvement: {speedup:.1f}x faster")
        
        print(f"\n🔧 RECOMMENDATION:")
        print(f"   For consistent backgrounds (elephant gray/cyan):")
        print(f"   ✅ Use FFT method (use_fft_analysis=True)")
        print(f"   ⚡ Expected speedup: {speedup:.1f}x")
        print(f"   💡 Ideal for production environments")
    
    else:
        print(f"\n❌ Could not complete performance comparison")
    
    print(f"\n📁 Test outputs saved to: performance_test_output/")


if __name__ == "__main__":
    main() 