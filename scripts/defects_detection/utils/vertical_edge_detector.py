"""
Vertical Edge Detection for Stripe Images
Detects vertical edges to identify the main vertical line, which may be slanted.
Currently implements edge detection kernel; Hough transform to be added later.

Author: Assistant
Date: 2024
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path
import tifffile  # For reading large TIFF files
import sys
from pathlib import Path

# Add the parent directory to sys.path to enable relative imports
sys.path.append(str(Path(__file__).parent.parent))

from detector_base import BaseDetector
from image_saver import save_image

class VerticalEdgeDetector(BaseDetector):
    """Detector for vertical edges in stripe images.
    
    Parameters can be adjusted for kernel size and type.
    Debug mode saves visualizations for each processing step.
    """
    
    def __init__(self, sensitivity='medium', debug=True):
        super().__init__()
        self.debug = debug
        self.sensitivity = sensitivity
        self.set_parameters_based_on_sensitivity()
        self.binary_edges = None
    
    def set_parameters_based_on_sensitivity(self):
        if self.sensitivity == 'low':
            self.kernel_size = 5  # Larger kernel for less sensitivity
            self.edge_threshold_low = 100
            self.edge_threshold_high = 200
        elif self.sensitivity == 'medium':
            self.kernel_size = 3
            self.edge_threshold_low = 200
            self.edge_threshold_high = 110
        elif self.sensitivity == 'high':
            self.kernel_size = 3  # Smaller or same, but lower thresholds
            self.edge_threshold_low = 30
            self.edge_threshold_high = 100
        else:
            raise ValueError("Invalid sensitivity level. Choose 'low', 'medium', or 'high'.")
    
    def detect(self, image, image_path=None):
        """Detect vertical edges in the image.
        
        Args:
            image: Input image (BGR or grayscale).
            image_path: Optional path for loading exclusion zones.
        
        Returns:
            tuple: (visualization_bgr, defects) - defects is empty for now as it's just edge detection.
        """
        if image_path:
            self.load_exclusion_zones(image_path)
        
        # Step 1: Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if self.debug:
            self.debug_images = {}  # To store debug images
            self.debug_images['grayscale'] = gray
        
        # Step 2: Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.debug:
            self.debug_images['blurred'] = blurred
        
        # Remove Sobel
        # # Step 3: Detect vertical edges using Sobel
        # edges = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=self.kernel_size)
        # edges = cv2.convertScaleAbs(edges)
        # if self.debug:
        #     self.debug_images['sobel_vertical'] = edges
        
        # Step 3: Apply Canny directly on blurred
        binary_edges = cv2.Canny(blurred, self.edge_threshold_low, self.edge_threshold_high)
        
        if self.debug:
            self.debug_images['canny_edges'] = binary_edges
        
        self.binary_edges = binary_edges
        
        # Create edge overlay always
        overlay = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        red_channel = overlay[:, :, 2]
        red_channel[binary_edges == 255] = 255
        overlay[binary_edges == 255, 0] = 0
        overlay[binary_edges == 255, 1] = 0
        
        if self.debug:
            self.debug_images = {
                'grayscale': gray,
                'blurred': blurred,
                'canny_edges': binary_edges,
                'edge_overlay': overlay
            }
        else:
            self.debug_images = {'edge_overlay': overlay}
        
        visualization = overlay
        
        defects = []
        
        return visualization, defects
    
    def save_debug_images(self, output_dir: str, base_name: str):
        """Save all debug images using image_saver."""
        if not hasattr(self, 'debug_images') or not self.debug_images:
            return
        
        for suffix, img in self.debug_images.items():
            save_image(output_dir, base_name, img, suffix)

# Standalone running
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vertical Edge Detection on Stripe TIFF Image")
    parser.add_argument("image_path", help="Path to the input TIFF image")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: same as input)")
    parser.add_argument("--sensitivity", default='medium', choices=['low', 'medium', 'high'], help="Detection sensitivity")
    
    args = parser.parse_args()
    
    # Load large TIFF using tifffile
    try:
        image = tifffile.imread(args.image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        exit(1)
    
    # Set output dir
    output_dir = args.output_dir or os.path.dirname(args.image_path)
    base_name = Path(args.image_path).stem
    
    # Create detector
    detector = VerticalEdgeDetector(sensitivity=args.sensitivity, debug=True)
    
    # Run detection
    visualization, defects = detector.detect(image, args.image_path)
    
    # Save main visualization
    save_image(output_dir, base_name, visualization, "vertical_edges_visualization")
    
    # Save debug images
    detector.save_debug_images(output_dir, base_name)
    
    # Save defects if any (empty for now)
    if defects:
        json_path = os.path.join(output_dir, f"{base_name}_vertical_edges_results.json")
        with open(json_path, 'w') as f:
            json.dump(defects, f, indent=2)
    
    print(f"Processing complete. Outputs saved to {output_dir}")
