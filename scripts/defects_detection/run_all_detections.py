"""
Main Script for Running All Defect Detection Algorithms
Processes images through all detection algorithms and generates comprehensive reports

Author: Assistant
Date: 2024
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

# Import all detection modules
from overspray_detection import OversprayDetector
from surface_treatment_detection import SurfaceTreatmentDetector
from debris_detection import DebrisDetector
from edge_defect_detection import EdgeDefectDetector
from banding_detection import BandingDetector
from streak_detection import StreakDetector
from window_processor import WindowProcessor


class DefectDetectionPipeline:
    """Main pipeline for running all defect detection algorithms"""
    
    def __init__(self, output_base_dir="output", sensitivity='medium', generate_report=False):
        self.output_base_dir = output_base_dir
        self.sensitivity = sensitivity
        self.generate_report = generate_report
        self.detectors = self._initialize_detectors()
        self.results_summary = []
        # Initialize window processor for large images - DISABLED MULTITHREADING
        self.window_processor = WindowProcessor(
            window_size=2048,  # Optimal for your tall images
            overlap=256,
            max_workers=1  # Single thread mode for debugging
        )
        
    def _initialize_detectors(self):
        """Initialize all defect detection algorithms with sensitivity settings"""
        if self.sensitivity == 'low':
            # Conservative detection
            detectors = {
                'overspray': OversprayDetector(dot_size_range=(5, 20), proximity_threshold=30),
                'surface_treatment': SurfaceTreatmentDetector(contrast_threshold=70, void_size_threshold=30),
                'debris': DebrisDetector(halo_threshold=40, particle_size_range=(20, 800)),
                'edge_defect': EdgeDefectDetector(smoothness_threshold=7, min_defect_length=15),
                'banding': BandingDetector(min_band_strength=0.2, band_width_range=(8, 60)),
                'streak': StreakDetector(min_streak_length=70, contrast_threshold=30)
            }
        elif self.sensitivity == 'high':
            # Aggressive detection
            detectors = {
                'overspray': OversprayDetector(dot_size_range=(2, 25), proximity_threshold=70),
                'surface_treatment': SurfaceTreatmentDetector(contrast_threshold=30, void_size_threshold=10),
                'debris': DebrisDetector(halo_threshold=20, particle_size_range=(5, 1000)),
                'edge_defect': EdgeDefectDetector(smoothness_threshold=3, min_defect_length=5),
                'banding': BandingDetector(min_band_strength=0.1, band_width_range=(3, 40)),
                'streak': StreakDetector(min_streak_length=30, contrast_threshold=15)
            }
        else:  # medium (default)
            detectors = {
                'overspray': OversprayDetector(),
                'surface_treatment': SurfaceTreatmentDetector(),
                'debris': DebrisDetector(),
                'edge_defect': EdgeDefectDetector(),
                'banding': BandingDetector(),
                'streak': StreakDetector()
            }
        return detectors
    
    def process_folder(self, input_folder):
        """Process all images in a folder"""
        # Create output directory within input folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(input_folder, f"output_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
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
            self.process_single_image(img_path, output_dir)
        
        # Generate summary report
        self.generate_summary_report(output_dir)
        
        print(f"\nProcessing complete! Results saved to: {output_dir}")
        
    def process_single_image(self, image_path, output_dir):
        """Process a single image through all detectors"""
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(image_output_dir, exist_ok=True)
        
        print(f"\n  Processing: {base_name}")
        
        # Initialize results
        image_results = {
            'image_name': base_name,
            'image_path': image_path,
            'processing_time': datetime.now().isoformat(),
            'defects': {}
        }
        
        # Check if it's a large image
        try:
            # Get image size without loading full image
            file_size = os.path.getsize(image_path)
            
            # Use windowed processing for files > 20MB OR TIFF files
            if file_size > 20 * 1024 * 1024 or image_path.lower().endswith(('.tif', '.tiff')):
                # Use window processor for large images and all TIFF files
                print(f"    Large/TIFF image detected (size: {file_size/(1024*1024):.1f}MB), using windowed processing...")
                detector_results = self.window_processor.process_image_windowed(
                    image_path, self.detectors, image_output_dir
                )
                
                # Store results
                if detector_results:
                    for detector_name, results in detector_results.items():
                        image_results['defects'][detector_name] = {
                            'defect_count': results.get('defect_count', 0),
                            'visualization_path': results.get('visualization_path'),
                            'defects': results.get('defects', [])
                        }
            else:
                # Process normally for smaller non-TIFF images
                print(f"    Regular image (size: {file_size/(1024*1024):.1f}MB), using normal processing...")
                self._process_regular_image(image_path, image_output_dir, image_results)
                
        except Exception as e:
            print(f"    Error processing image: {str(e)}")
            image_results['error'] = str(e)
        
        # Save results (JSON only)
        with open(os.path.join(image_output_dir, f"{base_name}_results.json"), 'w') as f:
            json.dump(image_results, f, indent=2)
        
        self.results_summary.append(image_results)
    
    def _process_regular_image(self, image_path, output_dir, image_results):
        """Process regular-sized images"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"    Error: Could not read image")
            return
        
        # Run each detector
        for detector_name, detector in self.detectors.items():
            print(f"    Running {detector_name} detection...")
            try:
                # Run detection - all detectors now return (visualization, defects)
                result_img, defects = detector.detect(image)
                
                # Save only visualization (not mask or original)
                vis_path = os.path.join(output_dir, f"{detector_name}_visualization.jpg")
                cv2.imwrite(vis_path, result_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # Store results
                image_results['defects'][detector_name] = {
                    'defect_count': len(defects),
                    'visualization_path': vis_path,
                    'defects': defects
                }
                
            except Exception as e:
                print(f"    Error in {detector_name} detection: {str(e)}")
                image_results['defects'][detector_name] = {'error': str(e)}
    
    def create_combined_visualization(self, original, results, output_dir):
        """Create a combined visualization showing all defects"""
        h, w = original.shape[:2]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Defect Detection Results - {results['image_name']}", fontsize=16)
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Load and display each detector's visualization
        detector_positions = {
            'overspray': (0, 1),
            'surface_treatment': (0, 2),
            'debris': (1, 0),
            'gray_spot': (1, 1),
            'edge_defect': (1, 2)
        }
        
        for detector_name, pos in detector_positions.items():
            if detector_name in results['defects']:
                vis_path = results['defects'][detector_name].get('visualization_path')
                if vis_path and os.path.exists(vis_path):
                    vis_img = cv2.imread(vis_path)
                    axes[pos].imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
                    
                    # Add defect count to title
                    count = results['defects'][detector_name].get('defect_count', 0)
                    axes[pos].set_title(f"{detector_name.replace('_', ' ').title()}\n({count} defects)")
                else:
                    axes[pos].text(0.5, 0.5, 'Error', ha='center', va='center')
                    axes[pos].set_title(detector_name.replace('_', ' ').title())
            else:
                axes[pos].text(0.5, 0.5, 'Not processed', ha='center', va='center')
                axes[pos].set_title(detector_name.replace('_', ' ').title())
            
            axes[pos].axis('off')
        
        # Save combined visualization
        plt.tight_layout()
        combined_path = os.path.join(output_dir, f"{results['image_name']}_combined_results.png")
        plt.savefig(combined_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create defect overlay
        self.create_defect_overlay(original, results, output_dir)
    
    def create_defect_overlay(self, original, results, output_dir):
        """Create an overlay showing all defects on one image"""
        overlay = original.copy()
        combined_mask = np.zeros(original.shape[:2], dtype=np.uint8)
        
        # Define colors for each defect type
        defect_colors = {
            'overspray': [0, 0, 255],      # Red
            'surface_treatment': [255, 0, 0],  # Blue
            'debris': [0, 255, 0],          # Green
            'gray_spot': [255, 255, 0],     # Cyan
            'edge_defect': [255, 0, 255]    # Magenta
        }
        
        # Combine all defect masks
        for detector_name in self.detectors.keys():
            if detector_name in results['defects']:
                mask_path = results['defects'][detector_name].get('mask_path')
                if mask_path and os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        # Color the defects
                        colored_region = np.zeros_like(overlay)
                        colored_region[mask > 0] = defect_colors[detector_name]
                        overlay = cv2.addWeighted(overlay, 1, colored_region, 0.3, 0)
                        combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Save overlay
        overlay_path = os.path.join(output_dir, f"{results['image_name']}_defect_overlay.jpg")
        cv2.imwrite(overlay_path, overlay)
        
        # Save combined mask
        combined_mask_path = os.path.join(output_dir, f"{results['image_name']}_combined_mask.png")
        cv2.imwrite(combined_mask_path, combined_mask)
    
    def generate_summary_report(self, output_dir):
        """Generate a summary report of all processed images"""
        # Calculate statistics first
        total_images = len(self.results_summary)
        defect_counts = {detector: 0 for detector in self.detectors.keys()}
        images_with_defects = {detector: 0 for detector in self.detectors.keys()}
        
        for result in self.results_summary:
            for detector_name in self.detectors.keys():
                if detector_name in result['defects']:
                    count = result['defects'][detector_name].get('defect_count', 0)
                    if count > 0:
                        defect_counts[detector_name] += count
                        images_with_defects[detector_name] += 1
        
        # Generate PDF report if requested
        if self.generate_report:
            pdf_path = os.path.join(output_dir, "defect_detection_report.pdf")
            
            with PdfPages(pdf_path) as pdf:
                # Summary page
                fig, ax = plt.subplots(figsize=(8.5, 11))
                fig.suptitle("Defect Detection Summary Report", fontsize=20, fontweight='bold')
                
                # Create summary text
                summary_text = f"Total Images Processed: {total_images}\n\n"
                summary_text += "Defect Summary:\n" + "-" * 40 + "\n"
                
                for detector_name in self.detectors.keys():
                    summary_text += f"\n{detector_name.replace('_', ' ').title()}:\n"
                    summary_text += f"  Total Defects Found: {defect_counts[detector_name]}\n"
                    summary_text += f"  Images with Defects: {images_with_defects[detector_name]}\n"
                    if images_with_defects[detector_name] > 0:
                        avg = defect_counts[detector_name] / images_with_defects[detector_name]
                        summary_text += f"  Average per Affected Image: {avg:.2f}\n"
                
                ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
                       fontsize=12, verticalalignment='top', fontfamily='monospace')
                ax.axis('off')
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
                # Add defect distribution chart
                if any(defect_counts.values()):
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 11))
                    
                    # Bar chart of defect counts
                    detector_names = list(defect_counts.keys())
                    counts = list(defect_counts.values())
                    
                    ax1.bar(detector_names, counts, color=['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'orange'])
                    ax1.set_xlabel('Defect Type')
                    ax1.set_ylabel('Total Count')
                    ax1.set_title('Total Defects by Type')
                    ax1.tick_params(axis='x', rotation=45)
                    
                    # Pie chart of affected images
                    affected_counts = [images_with_defects[d] for d in detector_names if images_with_defects[d] > 0]
                    affected_labels = [d.replace('_', ' ').title() for d in detector_names if images_with_defects[d] > 0]
                    
                    if affected_counts:
                        ax2.pie(affected_counts, labels=affected_labels, autopct='%1.1f%%')
                        ax2.set_title('Distribution of Images with Defects')
                    
                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
            
            print(f"\nPDF report saved to: {pdf_path}")
        
        # Save JSON summary
        summary_json_path = os.path.join(output_dir, "defect_report.json")
        with open(summary_json_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_images': total_images,
                'detection_sensitivity': self.sensitivity,
                'defect_statistics': {
                    detector: {
                        'total_defects': defect_counts[detector],
                        'affected_images': images_with_defects[detector]
                    } for detector in self.detectors.keys()
                },
                'detailed_results': self.results_summary
            }, f, indent=2)
        
        print(f"Detailed results saved to: {summary_json_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run all defect detection algorithms on a folder of images'
    )
    parser.add_argument('--input_folder', required=True, help='Path to folder containing images')
    parser.add_argument('--output', '-o', help='Output base directory (default: creates output folder in input folder)')
    parser.add_argument('--generate_report', action='store_true', help='Generate PDF report with all detections')
    parser.add_argument('--sensitivity', choices=['low', 'medium', 'high'], default='medium', 
                       help='Detection sensitivity level (default: medium)')
    
    args = parser.parse_args()
    
    # Check if input folder exists
    if not os.path.isdir(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist")
        sys.exit(1)
    
    # Create pipeline and process
    pipeline = DefectDetectionPipeline(
        output_base_dir=args.output,
        sensitivity=args.sensitivity,
        generate_report=args.generate_report
    )
    
    print("=" * 60)
    print("Defect Detection Pipeline")
    print("=" * 60)
    print(f"Input folder: {args.input_folder}")
    print(f"Detection algorithms:")
    for detector_name in pipeline.detectors.keys():
        print(f"  - {detector_name.replace('_', ' ').title()}")
    print("=" * 60)
    
    # Process folder
    pipeline.process_folder(args.input_folder)


if __name__ == "__main__":
    main() 