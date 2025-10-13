#!/usr/bin/env python3
"""
Standalone Smudge Detection Tool
Takes a TIFF file input and runs smudge detection in all sensitivity modes
"""

import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path

# Add the scripts directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'scripts', 'defects_detection'))

from smudge_detection import SmudgeDetector


def load_tiff_image(tiff_path: str) -> np.ndarray:
    """
    Load a TIFF image file
    
    Args:
        tiff_path: Path to the TIFF file
        
    Returns:
        Loaded image as numpy array
    """
    if not os.path.exists(tiff_path):
        raise FileNotFoundError(f"TIFF file not found: {tiff_path}")
    
    # Try loading with OpenCV first
    image = cv2.imread(tiff_path, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        raise ValueError(f"Could not load TIFF file: {tiff_path}")
    
    # If grayscale, convert to BGR for consistency
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        # If RGBA, convert to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    
    return image


def create_detector_for_sensitivity(sensitivity: str, use_fft: bool = True) -> SmudgeDetector:
    """
    Create smudge detector with specified sensitivity settings
    
    Args:
        sensitivity: 'low', 'medium', or 'high'
        use_fft: Use ultra-fast FFT method (recommended for consistent backgrounds)
        
    Returns:
        Configured SmudgeDetector instance
    """
    if sensitivity == 'low':
        return SmudgeDetector(
            min_smudge_size=200000,  # 450x450 pixels - very conservative
            background_window_size=120,
            lightness_threshold_factor=2.2,  # More conservative
            consistency_threshold=0.2,  # Stricter consistency requirement
            morphology_kernel_size=20,
            use_fft_analysis=use_fft
        )
    elif sensitivity == 'high':
        return SmudgeDetector(
            min_smudge_size=100000,  # 316x316 pixels - more sensitive
            background_window_size=80,
            lightness_threshold_factor=1.4,  # More sensitive
            consistency_threshold=0.1,  # Less strict consistency
            morphology_kernel_size=10,
            use_fft_analysis=use_fft
        )
    else:  # medium
        return SmudgeDetector(
            use_fft_analysis=use_fft
        )  # Use defaults (400x400 = 160,000 pixels)


def run_smudge_detection(image: np.ndarray, sensitivity: str, output_dir: str, base_name: str) -> dict:
    """
    Run smudge detection with specified sensitivity
    
    Args:
        image: Input image
        sensitivity: Detection sensitivity level
        output_dir: Output directory for results
        base_name: Base name for output files
        
    Returns:
        Dictionary with detection results
    """
    print(f"\n🔍 Running {sensitivity.upper()} sensitivity detection...")
    
    # Create detector (use FFT method by default for speed)
    detector = create_detector_for_sensitivity(sensitivity, use_fft=True)
    
    # Print detector settings
    print(f"   ⚙️  Settings:")
    print(f"      - Min size: {detector.min_smudge_size:,} pixels (~{int(np.sqrt(detector.min_smudge_size))}×{int(np.sqrt(detector.min_smudge_size))})")
    print(f"      - Background window: {detector.background_window_size}×{detector.background_window_size}")
    print(f"      - Lightness factor: {detector.lightness_threshold_factor}")
    print(f"      - Consistency threshold: {detector.consistency_threshold}")
    
    # Run detection
    try:
        visualization, defects = detector.detect(image)
        
        # Save visualization
        output_filename = f"{base_name}_smudge_detection_{sensitivity}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, visualization, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Print results
        print(f"   📈 Found {len(defects)} smudge(s)")
        print(f"   💾 Saved: {output_filename}")
        
        # Print defect details
        if defects:
            print(f"   📋 Detected smudges:")
            for i, defect in enumerate(defects, 1):
                area_k = defect['area'] / 1000
                equiv_size = int(np.sqrt(defect['area']))
                print(f"      {i}. {defect['subtype']} at {defect['location']}")
                print(f"         Area: {defect['area']:,} pixels ({area_k:.1f}K, ~{equiv_size}×{equiv_size})")
                print(f"         Severity: {defect['severity']}, Lightness: {defect['lightness_ratio']:.2f}x")
        
        return {
            'sensitivity': sensitivity,
            'defect_count': len(defects),
            'defects': defects,
            'output_file': output_path
        }
        
    except Exception as e:
        print(f"   ❌ Error during {sensitivity} detection: {str(e)}")
        return {
            'sensitivity': sensitivity,
            'defect_count': 0,
            'defects': [],
            'error': str(e)
        }


def create_summary_report(results: list, output_dir: str, base_name: str, image_info: dict):
    """
    Create a summary report of all detection results
    
    Args:
        results: List of detection results from all sensitivity levels
        output_dir: Output directory
        base_name: Base name for files
        image_info: Information about the input image
    """
    report_path = os.path.join(output_dir, f"{base_name}_smudge_detection_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("SMUDGE DETECTION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Input Image: {image_info['filename']}\n")
        f.write(f"Image Size: {image_info['width']}×{image_info['height']} pixels\n")
        f.write(f"File Size: {image_info['file_size_mb']:.1f} MB\n")
        f.write(f"Analysis Date: {image_info['timestamp']}\n\n")
        
        f.write("DETECTION RESULTS BY SENSITIVITY:\n")
        f.write("-" * 40 + "\n\n")
        
        for result in results:
            sensitivity = result['sensitivity'].upper()
            f.write(f"{sensitivity} SENSITIVITY:\n")
            f.write(f"  Smudges Found: {result['defect_count']}\n")
            
            if 'error' in result:
                f.write(f"  Error: {result['error']}\n")
            elif result['defects']:
                f.write(f"  Details:\n")
                for i, defect in enumerate(result['defects'], 1):
                    area_k = defect['area'] / 1000
                    equiv_size = int(np.sqrt(defect['area']))
                    f.write(f"    {i}. Type: {defect['subtype']}\n")
                    f.write(f"       Location: {defect['location']}\n")
                    f.write(f"       Area: {defect['area']:,} pixels ({area_k:.1f}K, ~{equiv_size}×{equiv_size})\n")
                    f.write(f"       Severity: {defect['severity']}\n")
                    f.write(f"       Lightness Ratio: {defect['lightness_ratio']:.2f}x\n")
                    f.write(f"       Eccentricity: {defect['eccentricity']:.3f}\n\n")
            f.write("\n")
        
        f.write("ALGORITHM INFORMATION:\n")
        f.write("-" * 25 + "\n")
        f.write("The smudge detection algorithm identifies lighter areas on\n")
        f.write("consistent backgrounds, typical of post-print smudges.\n\n")
        f.write("Sensitivity Levels:\n")
        f.write("- LOW: Very conservative (450×450+ pixels minimum)\n")
        f.write("- MEDIUM: Balanced (400×400+ pixels minimum)\n")
        f.write("- HIGH: More sensitive (316×316+ pixels minimum)\n\n")
        f.write("For 2400+ DPI printing, smudges should be at least 400×400 pixels\n")
        f.write("to be considered significant quality defects.\n")
    
    print(f"   📄 Summary report saved: {os.path.basename(report_path)}")


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Standalone Smudge Detection Tool for TIFF images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_smudge_detection.py input.tiff
  python demo_smudge_detection.py path/to/image.tif -o custom_output_dir
  python demo_smudge_detection.py sample.tiff --verbose

The tool will run smudge detection in all three sensitivity modes:
- LOW: Very conservative detection (450×450+ pixels)
- MEDIUM: Balanced detection (400×400+ pixels) 
- HIGH: More sensitive detection (316×316+ pixels)

Output files created:
- {basename}_smudge_detection_low.jpg
- {basename}_smudge_detection_medium.jpg  
- {basename}_smudge_detection_high.jpg
- {basename}_smudge_detection_report.txt
        """
    )
    
    parser.add_argument('input_tiff', 
                       help='Path to input TIFF file')
    parser.add_argument('-o', '--output-dir', 
                       default='smudge_detection_output',
                       help='Output directory (default: smudge_detection_output)')
    parser.add_argument('-v', '--verbose', 
                       action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_tiff)
    if not input_path.exists():
        print(f"❌ Error: Input file '{args.input_tiff}' not found!")
        sys.exit(1)
    
    if not input_path.suffix.lower() in ['.tiff', '.tif']:
        print(f"⚠️  Warning: File extension '{input_path.suffix}' is not .tiff or .tif")
        print("   Attempting to load anyway...")
    
    print("🚀 Standalone Smudge Detection Tool")
    print("=" * 50)
    print(f"📁 Input: {input_path.name}")
    print(f"📂 Output: {args.output_dir}")
    
    try:
        # Load image
        print(f"\n📖 Loading TIFF image...")
        image = load_tiff_image(str(input_path))
        
        # Get image info
        file_size_mb = input_path.stat().st_size / (1024 * 1024)
        image_info = {
            'filename': input_path.name,
            'width': image.shape[1],
            'height': image.shape[0],
            'file_size_mb': file_size_mb,
            'timestamp': __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print(f"   ✅ Image loaded: {image_info['width']}×{image_info['height']} pixels")
        print(f"   📊 File size: {file_size_mb:.1f} MB")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        print(f"   📁 Output directory: {output_dir}")
        
        # Save original image for reference
        base_name = input_path.stem
        original_path = output_dir / f"{base_name}_original.jpg"
        cv2.imwrite(str(original_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"   💾 Original saved: {original_path.name}")
        
        # Run detection in all sensitivity modes
        print(f"\n🔍 Running smudge detection in all sensitivity modes...")
        print(f"   (This may take a few minutes for large images)")
        
        results = []
        sensitivities = ['low', 'medium', 'high']
        
        for sensitivity in sensitivities:
            result = run_smudge_detection(image, sensitivity, str(output_dir), base_name)
            results.append(result)
        
        # Create summary report
        print(f"\n📄 Creating summary report...")
        create_summary_report(results, str(output_dir), base_name, image_info)
        
        # Print final summary
        print(f"\n✅ Analysis Complete!")
        print(f"📂 All files saved to: {output_dir}")
        print(f"\n📊 Summary:")
        for result in results:
            if 'error' not in result:
                print(f"   {result['sensitivity'].upper():>6}: {result['defect_count']} smudge(s) detected")
            else:
                print(f"   {result['sensitivity'].upper():>6}: Error occurred")
        
        print(f"\n🔧 Recommendation:")
        total_detections = sum(r['defect_count'] for r in results if 'error' not in r)
        if total_detections == 0:
            print("   No smudges detected at any sensitivity level.")
            print("   The image appears to be free of significant smudge defects.")
        else:
            medium_count = next(r['defect_count'] for r in results if r['sensitivity'] == 'medium')
            if medium_count > 0:
                print(f"   MEDIUM sensitivity detected {medium_count} smudge(s).")
                print("   This represents a balanced assessment for quality control.")
            else:
                high_count = next(r['defect_count'] for r in results if r['sensitivity'] == 'high')
                print(f"   Only HIGH sensitivity detected smudges ({high_count} found).")
                print("   These may be minor defects - review manually if needed.")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 