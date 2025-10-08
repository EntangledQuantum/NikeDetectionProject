"""
Probabilistic Hough Transform for Vertical Lines in Stripe Images
Uses vertical edge detection and applies HoughLinesP to detect roughly vertical, slightly tilted, possibly broken lines.
Handles orientation tolerance and gap bridging.

Author: Assistant
Date: 2024
"""

import cv2
import numpy as np
import os
from pathlib import Path

from .vertical_edge_detector import VerticalEdgeDetector
from .image_saver import save_image

class VerticalHoughDetector:
    """Detector using Probabilistic Hough Transform for vertical-ish lines."""
    
    def __init__(self, sensitivity='medium', debug=True):
        self.debug = debug
        self.sensitivity = sensitivity
        self.edge_detector = VerticalEdgeDetector(sensitivity=sensitivity, debug=debug)
        self.set_parameters_based_on_sensitivity()
    
    def set_parameters_based_on_sensitivity(self):
        if self.sensitivity == 'low':
            self.rho = 1
            self.theta = np.pi / 180
            self.threshold = 100  # Higher for fewer false positives
            self.minLineLength = 100  # Larger to ignore short segments
            self.maxLineGap = 10  # Smaller gap for conservative bridging
        elif self.sensitivity == 'medium':
            self.rho = 1
            self.theta = np.pi / 180
            self.threshold = 50
            self.minLineLength = 50
            self.maxLineGap = 20
        elif self.sensitivity == 'high':
            self.rho = 1
            self.theta = np.pi / 180
            self.threshold = 30  # Lower for more detections
            self.minLineLength = 30  # Smaller to catch short segments
            self.maxLineGap = 40  # Larger gap to bridge broken lines
        else:
            raise ValueError("Invalid sensitivity level. Choose 'low', 'medium', or 'high'.")
    
    def detect(self, image, image_path=None):
        """Apply edge detection then Hough transform to detect vertical lines.
        
        Returns:
            tuple: (visualization_bgr, lines) where lines is list of detected segments.
        """
        # Get binary edges from edge detector
        _, _ = self.edge_detector.detect(image, image_path)
        if not hasattr(self.edge_detector, 'debug_images') or 'binary_edges' not in self.edge_detector.debug_images:
            raise ValueError("Edge detection must be run with debug=True to get binary_edges.")
        binary_edges = self.edge_detector.debug_images['binary_edges']
        
        # Apply Probabilistic Hough Transform
        lines = cv2.HoughLinesP(
            binary_edges, 
            self.rho, 
            self.theta, 
            self.threshold, 
            minLineLength=self.minLineLength, 
            maxLineGap=self.maxLineGap
        )
        
        # Filter to nearly vertical lines (angle close to 90 degrees, e.g., 70-110 degrees)
        filtered_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if 70 <= abs(angle) <= 110:  # Roughly vertical
                    filtered_lines.append((x1, y1, x2, y2))
        
        # Create visualization: draw lines on original image
        vis = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2 in filtered_lines:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if self.debug:
            self.debug_images = self.edge_detector.debug_images  # Inherit edge debug images
            self.debug_images['hough_lines'] = vis
        
        return vis, filtered_lines
    
    def save_debug_images(self, output_dir: str, base_name: str):
        """Save all debug images."""
        if not hasattr(self, 'debug_images') or not self.debug_images:
            return
        
        for suffix, img in self.debug_images.items():
            save_image(output_dir, base_name, img, suffix)

# Standalone running
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vertical Hough Transform on Stripe TIFF Image")
    parser.add_argument("image_path", help="Path to the input TIFF image")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: same as input)")
    parser.add_argument("--sensitivity", default='medium', choices=['low', 'medium', 'high'], help="Detection sensitivity")
    
    args = parser.parse_args()
    
    # Load image
    try:
        image = cv2.imread(args.image_path, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print(f"Error loading image: {e}")
        exit(1)
    
    output_dir = args.output_dir or os.path.dirname(args.image_path)
    base_name = Path(args.image_path).stem
    
    detector = VerticalHoughDetector(sensitivity=args.sensitivity, debug=True)
    visualization, lines = detector.detect(image, args.image_path)
    
    save_image(output_dir, base_name, visualization, "vertical_hough_visualization")
    
    detector.save_debug_images(output_dir, base_name)
    
    # Save lines to JSON
    json_path = os.path.join(output_dir, f"{base_name}_vertical_lines.json")
    with open(json_path, 'w') as f:
        json.dump(lines, f, indent=2)
    
    print(f"Processing complete. Detected {len(lines)} vertical lines. Outputs saved to {output_dir}")
