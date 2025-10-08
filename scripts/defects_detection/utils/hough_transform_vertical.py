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
import sys
import os
import json

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from scripts.defects_detection.utils.vertical_edge_detector import VerticalEdgeDetector
from scripts.defects_detection.utils.image_saver import save_image

class VerticalHoughDetector:
    """Detector using Probabilistic Hough Transform for vertical-ish lines."""
    
    def __init__(self, sensitivity='medium', debug=True, edge_detector=None):
        self.debug = debug
        self.sensitivity = sensitivity
        if edge_detector is None:
            self.edge_detector = VerticalEdgeDetector(sensitivity=sensitivity, debug=debug)
        else:
            self.edge_detector = edge_detector
        self.set_parameters_based_on_sensitivity()
    
    def set_parameters_based_on_sensitivity(self):
        if self.sensitivity == 'low':
            self.rho = 1
            self.theta = np.pi / 180
            self.threshold = 100  # Higher for fewer false positives
            self.minLineLength = 100  # Larger to ignore short segments
            self.maxLineGap = 10  # Smaller gap for conservative bridging
            self.angle_tolerance = 5  # degrees
        elif self.sensitivity == 'medium':
            self.rho = 1
            self.theta = np.pi / 180
            self.threshold = 50
            self.minLineLength = 50
            self.maxLineGap = 20
            self.angle_tolerance = 10
        elif self.sensitivity == 'high':
            self.rho = 1
            self.theta = np.pi / 180
            self.threshold = 30  # Lower for more detections
            self.minLineLength = 30  # Smaller to catch short segments
            self.maxLineGap = 40  # Larger gap to bridge broken lines
            self.angle_tolerance = 15
        else:
            raise ValueError("Invalid sensitivity level. Choose 'low', 'medium', or 'high'.")
    
    def detect(self, image, image_path=None):
        """Apply edge detection then Hough transform to detect vertical lines.
        
        Handles large images by processing in windows along height.
        
        Returns:
            tuple: (visualization_bgr, lines) where lines is list of detected segments.
        """
        if self.debug:
            self.debug_images = {}  # Initialize here
        
        # Get binary edges from edge_detector attribute (always available)
        _, _ = self.edge_detector.detect(image, image_path)
        binary_edges = self.edge_detector.binary_edges
        if binary_edges is None:
            raise ValueError("Failed to get binary_edges from edge detector.")
        
        height, width = binary_edges.shape
        all_lines = []
        
        # Define window parameters
        window_height = 5000  # Process 5000 px at a time
        overlap = 500  # Overlap to connect lines across windows
        
        if height <= window_height:
            # Small image: process whole
            lines = cv2.HoughLinesP(
                binary_edges, 
                self.rho, 
                self.theta, 
                self.threshold, 
                minLineLength=self.minLineLength, 
                maxLineGap=self.maxLineGap
            )
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    if (90 - self.angle_tolerance) <= abs(angle) <= (90 + self.angle_tolerance):
                        all_lines.append((x1, y1, x2, y2))
        else:
            # Large image: process in windows
            start_y = 0
            window_num = 0
            while start_y < height:
                end_y = min(start_y + window_height, height)
                window = binary_edges[start_y:end_y, :]
                
                lines = cv2.HoughLinesP(
                    window, 
                    self.rho, 
                    self.theta, 
                    self.threshold, 
                    minLineLength=self.minLineLength, 
                    maxLineGap=self.maxLineGap
                )
                
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                        if (90 - self.angle_tolerance) <= abs(angle) <= (90 + self.angle_tolerance):
                            # Adjust to global coordinates
                            all_lines.append((x1, y1 + start_y, x2, y2 + start_y))
                
                if self.debug:
                    # Save window visualization
                    window_vis = cv2.cvtColor(window, cv2.COLOR_GRAY2BGR)
                    if lines is not None:
                        for line in lines:
                            x1, y1, x2, y2 = line[0]
                            cv2.line(window_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    self.debug_images[f'hough_window_{window_num}'] = window_vis
                
                start_y += window_height - overlap
                window_num += 1
        
        # Move vis creation here, outside if
        vis = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2 in all_lines:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if self.debug:
            self.debug_images = {}  
            if hasattr(self.edge_detector, 'debug_images'):
                self.debug_images.update(self.edge_detector.debug_images)
            self.debug_images['hough_lines'] = vis
        
        return vis, all_lines
    
    def save_debug_images(self, output_dir: str, base_name: str):
        """Save all debug images."""
        if not hasattr(self, 'debug_images') or not self.debug_images:
            return
        
        for suffix, img in self.debug_images.items():
            save_image(output_dir, base_name, img, suffix)

# Standalone running
if __name__ == "__main__":
    import argparse
    import tifffile  # Add this import
    
    parser = argparse.ArgumentParser(description="Vertical Hough Transform on Stripe TIFF Image")
    parser.add_argument("image_path", help="Path to the input TIFF image")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: same as input)")
    parser.add_argument("--sensitivity", default='medium', choices=['low', 'medium', 'high'], help="Detection sensitivity")
    
    args = parser.parse_args()
    
    # Load image with tifffile for large TIFF support
    try:
        image = tifffile.imread(args.image_path)
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
        # Convert numpy ints to python ints for serialization
        if lines:
            serializable_lines = [[int(coord) for coord in line] for line in lines]
        else:
            serializable_lines = []
        json.dump(serializable_lines, f, indent=2)
    
    print(f"Processing complete. Detected {len(lines)} vertical lines. Outputs saved to {output_dir}")
