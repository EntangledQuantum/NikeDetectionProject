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
        
        self.stripe_width = 1010
        self.width_tolerance = 100
        self.max_x_dev = 15  # Deviation tolerance for continuous
    
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
                'edge_threshold_low': 80,
                'edge_threshold_high': 100
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
                'edge_threshold_low': 120,
                'edge_threshold_high': 160
            }
            hough_params = {
                'rho': 1,
                'theta': np.pi / 180,
                'threshold': 40,
                'minLineLength': 80,
                'maxLineGap': 60,
                'angle_tolerance': 10
            }
        else:
            raise ValueError("Invalid sensitivity.")
        
        return edge_params, hough_params
    
    def detect(self, image, image_path=None):
        height = image.shape[0]
        _, lines = self.hough_detector.detect(image, image_path)  # Get only lines, skip early vis
        filtered_lines = self.filter_stripe_lines(lines, height)
        
        # Create visualization with filtered lines
        vis = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2 in filtered_lines:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        vis, regions = self.create_regions_and_overlay(filtered_lines, vis)
        return vis, filtered_lines  # Or return vis, filtered_lines, regions if needed
    
    def filter_stripe_lines(self, lines, image_height):
        """Per-y filtering for valid line pairs."""
        if len(lines) < 2:
            print("    Fewer than 2 lines - skipping")
            return []
        
        kept = set()
        
        for y in range(image_height):
            # Collect x at this y for intersecting lines
            x_at_y = []
            for idx, (x1, y1, x2, y2) in enumerate(lines):
                y_min, y_max = min(y1, y2), max(y1, y2)
                if y_min <= y <= y_max:
                    if y1 == y2:  # Horizontal, skip or avg x
                        continue
                    # Interpolate x
                    t = (y - y1) / (y2 - y1)
                    x = x1 + t * (x2 - x1)
                    x_at_y.append((x, idx))  # (x, line_index)
            
            if len(x_at_y) < 2:
                continue
            
            # Sort by x
            x_at_y.sort(key=lambda p: p[0])
            
            # Check pairwise
            found = False
            for i in range(len(x_at_y)):
                for j in range(i+1, len(x_at_y)):
                    dist = abs(x_at_y[j][0] - x_at_y[i][0])
                    if abs(dist - self.stripe_width) <= self.width_tolerance:
                        # Keep these two lines
                        kept.add(tuple(lines[x_at_y[i][1]]))
                        kept.add(tuple(lines[x_at_y[j][1]]))
                        found = True
                        break  # Skip further pairs for this y
                if found:
                    break  # Move to next y
        
        filtered = [list(l) for l in kept]  # Convert back to lists
        print(f"    Filtered to {len(filtered)} unique lines from per-y pairs")
        return filtered
    
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

    def create_regions_and_overlay(self, filtered_lines, vis):
        """Create regions from filtered lines and overlay on vis."""
        if not filtered_lines:
            return vis, []
        
        # Split into left and right groups
        avg_xs = [(l, (l[0] + l[2])/2) for l in filtered_lines]
        median_avg_x = np.median([ax for _, ax in avg_xs])
        left_lines = [l for l, ax in avg_xs if ax <= median_avg_x]
        right_lines = [l for l, ax in avg_xs if ax > median_avg_x]
        
        # Sort by starting y
        left_lines.sort(key=lambda l: min(l[1], l[3]))
        right_lines.sort(key=lambda l: min(l[1], l[3]))
        
        # Build continuous chains for left and right
        def build_chains(lines_group):
            if not lines_group:
                return []
            chains = []
            current_chain = [lines_group[0]]
            for line in lines_group[1:]:
                prev = current_chain[-1]
                prev_end_x = prev[2] if prev[3] > prev[1] else prev[0]
                curr_start_x = line[0] if line[1] < line[3] else line[2]
                if abs(curr_start_x - prev_end_x) < self.max_x_dev:
                    current_chain.append(line)
                else:
                    chains.append(current_chain)
                    current_chain = [line]
            if current_chain:
                chains.append(current_chain)
            return chains
        
        left_chains = build_chains(left_lines)
        right_chains = build_chains(right_lines)
        
        # For simplicity, assume parallel chains; create regions for matching pairs by index
        regions = []
        num_regions = min(len(left_chains), len(right_chains))
        for i in range(num_regions):
            l_chain = left_chains[i]
            r_chain = right_chains[i]
            
            # Get overall top/bottom y, avg top/bottom x
            top_y = min(min(min(l[1],l[3]) for l in l_chain), min(min(r[1],r[3]) for r in r_chain))
            bot_y = max(max(max(l[1],l[3]) for l in l_chain), max(max(r[1],r[3]) for r in r_chain))
            
            # Get top/bottom x from chain endpoints
            l_chain.sort(key=lambda l: min(l[1], l[3]))  # Ensure sorted
            r_chain.sort(key=lambda l: min(l[1], l[3]))
            
            # Left top: first segment's top endpoint x
            l_first = l_chain[0]
            l_top_y_idx = 1 if l_first[1] < l_first[3] else 3
            l_top_x = l_first[l_top_y_idx -1]  # x1 if y1 top, x2 if y2 top
            
            # Left bottom: last segment's bottom endpoint x
            l_last = l_chain[-1]
            l_bot_y_idx = 3 if l_last[3] > l_last[1] else 1
            l_bot_x = l_last[l_bot_y_idx -1]
            
            # Similarly for right
            r_first = r_chain[0]
            r_top_y_idx = 1 if r_first[1] < r_first[3] else 3
            r_top_x = r_first[r_top_y_idx -1]
            
            r_last = r_chain[-1]
            r_bot_y_idx = 3 if r_last[3] > r_last[1] else 1
            r_bot_x = r_last[r_bot_y_idx -1]
            
            # top_y = min(l_first[l_top_y_idx], r_first[r_top_y_idx])
            # bot_y = max(l_last[l_bot_y_idx], r_last[r_bot_y_idx])
            # But keep original top_y bot_y as min/max of all
            
            # Create polygon
            polygon = np.array([
                (int(l_top_x), int(top_y)),
                (int(r_top_x), int(top_y)),
                (int(r_bot_x), int(bot_y)),
                (int(l_bot_x), int(bot_y))
            ], np.int32)
            
            regions.append(polygon)
        
        # Overlay on vis with different random colors, semi-transparent
        overlay = vis.copy()
        for poly in regions:
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.fillPoly(overlay, [poly], color)
        
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        
        return vis, regions

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
