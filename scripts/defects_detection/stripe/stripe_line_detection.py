"""
Stripe Line Detector
Detects lines in stripe images using vertical edge detection and Hough transform.
Custom parameters per sensitivity for both components.

Author: Assistant
Date: 2024
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path
import tifffile
import sys
import matplotlib.pyplot as plt

# Add the utils directory to the Python path
utils_dir = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(0, os.path.abspath(utils_dir))

from vertical_edge_detector import VerticalEdgeDetector
from hough_transform_vertical import VerticalHoughDetector
from image_saver import save_image

class StripeLineDetector:
    """Detector for lines in stripe images using edge and Hough."""
    
    def __init__(self, sensitivity='medium', debug=True):
        self.debug = debug
        self.sensitivity = sensitivity
        self.edge_params, self.hough_params = self.get_params_for_sensitivity()
        self.edge_detector = VerticalEdgeDetector(debug=self.debug)  # No sensitivity, we'll set params
        for key, value in self.edge_params.items():
            setattr(self.edge_detector, key, value)
        
        self.hough_detector = VerticalHoughDetector(debug=self.debug, edge_detector=self.edge_detector)
        for key, value in self.hough_params.items():
            setattr(self.hough_detector, key, value)
    
    def get_params_for_sensitivity(self):
        if self.sensitivity == 'low':
            edge_params = {
                'kernel_size': 5,
                'edge_threshold_low': 100,
                'edge_threshold_high': 200
            }
            hough_params = {
                'rho': 1,
                'theta': np.pi / 180,
                'threshold': 100,
                'minLineLength': 100,
                'maxLineGap': 10,
                'angle_tolerance': 5
            }
        elif self.sensitivity == 'medium':
            edge_params = {
                'kernel_size': 3,
                'edge_threshold_low': 200,
                'edge_threshold_high': 110
            }
            hough_params = {
                'rho': 1,
                'theta': np.pi / 180,
                'threshold': 50,
                'minLineLength': 50,
                'maxLineGap': 20,
                'angle_tolerance': 10
            }
        elif self.sensitivity == 'high':
            edge_params = {
                'kernel_size': 3,
                'edge_threshold_low': 30,
                'edge_threshold_high': 100
            }
            hough_params = {
                'rho': 1,
                'theta': np.pi / 180,
                'threshold': 30,
                'minLineLength': 30,
                'maxLineGap': 40,
                'angle_tolerance': 15
            }
        else:
            raise ValueError("Invalid sensitivity.")
        
        return edge_params, hough_params
    
    def detect(self, image, image_path=None):
        vis, lines = self.hough_detector.detect(image, image_path)
        return vis, lines
    
    def save_debug_images(self, output_dir, base_name):
        self.hough_detector.save_debug_images(output_dir, base_name)

    def create_line_graph(self, lines, image_height, image_width, output_dir, base_name):
        """Create and save graph of line x vs image y."""
        fig, ax = plt.subplots(figsize=(4, 12))  # Tall narrow figure
        
        ax.set_xlim(0, image_height)
        ax.set_ylim(0, image_width)
        ax.set_xlabel('Image Y (Top to Bottom)')
        ax.set_ylabel('Line X Position')
        ax.invert_yaxis()  # Image y=0 at top
        
        if lines:
            for x1, y1, x2, y2 in lines:
                ax.plot([y1, y2], [x1, x2], 'b-', linewidth=1)
        
        graph_path = os.path.join(output_dir, f"{base_name}_line_x_profile.png")
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Line profile graph saved to {graph_path}")
        return graph_path

# Standalone
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stripe Line Detection on TIFF Image")
    parser.add_argument("image_path", help="Path to input TIFF")
    parser.add_argument("--output_dir", default=None, help="Output dir")
    parser.add_argument("--sensitivity", default='medium', choices=['low', 'medium', 'high'])
    
    args = parser.parse_args()
    
    image = tifffile.imread(args.image_path)
    output_dir = args.output_dir or os.path.dirname(args.image_path)
    base_name = Path(args.image_path).stem
    
    detector = StripeLineDetector(sensitivity=args.sensitivity, debug=True)
    vis, lines = detector.detect(image, args.image_path)
    
    save_image(output_dir, base_name, vis, "stripe_lines_visualization")
    detector.save_debug_images(output_dir, base_name)
    
    # Save lines to JSON
    json_path = os.path.join(output_dir, f"{base_name}_stripe_lines.json")
    with open(json_path, 'w') as f:
        serializable_lines = [[int(c) for c in line] for line in lines] if lines else []
        json.dump(serializable_lines, f, indent=2)
    
    height, width = image.shape[:2] if len(image.shape) > 2 else image.shape
    detector.create_line_graph(lines, height, width, output_dir, base_name)

    print(f"Done. Detected {len(lines)} lines. Outputs in {output_dir}")
