"""
Quick test script for stripe debris/void detection
Run this on a single stripe image to test the detector
"""
import cv2
import sys
import os

# Add scripts directory to path
sys.path.insert(0, 'scripts/defects_detection')

from stripe_debris_void_detection import StripeDebrisVoidDetector

def test_stripe_debris_detection(image_path, sensitivity='medium'):
    """Test debris/void detection on a single stripe image"""
    
    print("="*80)
    print(f"Testing Stripe Debris/Void Detection")
    print(f"Image: {image_path}")
    print(f"Sensitivity: {sensitivity}")
    print("="*80)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image {image_path}")
        return
    
    print(f"✓ Image loaded: {image.shape}")
    
    # Create detector with debug enabled
    detector = StripeDebrisVoidDetector(sensitivity=sensitivity, debug=True)
    
    # Run detection
    visualization, defects = detector.detect(image, image_path)
    
    # Save results
    output_dir = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save main visualization
    vis_path = os.path.join(output_dir, f"{base_name}_debris_void_test.jpg")
    cv2.imwrite(vis_path, visualization, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"\n✓ Visualization saved: {vis_path}")
    
    # Save debug images
    debug_dir = os.path.join(output_dir, f"{base_name}_debris_void_debug")
    os.makedirs(debug_dir, exist_ok=True)
    detector.save_debug_images(debug_dir, base_name)
    print(f"✓ Debug images saved to: {debug_dir}")
    
    # Print summary
    print("\n" + "="*80)
    print("DETECTION SUMMARY")
    print("="*80)
    
    debris_count = sum(1 for d in defects if d['type'] == 'debris')
    void_count = sum(1 for d in defects if d['type'] == 'void')
    
    print(f"Total Defects: {len(defects)}")
    print(f"  - Debris: {debris_count}")
    print(f"  - Voids: {void_count}")
    
    if len(defects) > 0:
        print("\nDefect Details:")
        for i, defect in enumerate(defects[:10], 1):  # Show first 10
            print(f"\n  Defect {i}:")
            print(f"    Type: {defect['type']}")
            print(f"    Area: {defect['area']:.1f} pixels")
            print(f"    Centroid: {defect['centroid']}")
            print(f"    Severity: {defect['severity']}")
            print(f"    Intensity deviation: {defect['intensity_deviation']:.2f}")
        
        if len(defects) > 10:
            print(f"\n  ... and {len(defects) - 10} more defects")
    else:
        print("\n⚠ No defects detected!")
        print("\nTroubleshooting tips:")
        print("  1. Check debug images to see if masks are being created")
        print("  2. Try higher sensitivity: 'high'")
        print("  3. Verify the image actually has debris/voids")
        print("  4. Check if image is too uniform (no anomalies)")
    
    print("="*80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_debris_detection.py <stripe_image_path> [sensitivity]")
        print("Example: python test_debris_detection.py Images/Newer_High_DPI/test_Paper2400_extracted/blackStripe.tiff medium")
        sys.exit(1)
    
    image_path = sys.argv[1]
    sensitivity = sys.argv[2] if len(sys.argv) > 2 else 'medium'
    
    test_stripe_debris_detection(image_path, sensitivity)

