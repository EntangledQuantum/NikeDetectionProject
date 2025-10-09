"""
Island Line Detector
Detects lines in island images using vertical edge detection and Hough transform.
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
import time

# Add the utils directory to the Python path
utils_dir = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(0, os.path.abspath(utils_dir))

from vertical_edge_detector import VerticalEdgeDetector
from hough_transform_vertical import VerticalHoughDetector
from image_saver import save_image

class IslandLineDetector:
    """Detector for lines in island images using edge and Hough."""
    
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
        
        self.line_width = 2400
        self.tolerance = 10
        self.pair_dist = 200
        self.pair_tolerance = 10
        self.deviation_factor = 50
        self.window_size = 50
    
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
    
    def filter_island_lines(self, lines, image_width, image_height):
        import time
        start_time = time.time()
        
        if len(lines) < 4:
            print("    Fewer than 4 lines - skipping")
            return []
        
        main_lines = [tuple(l) for l in lines]  # List of tuples
        
        # Step 1: Extract left_start
        left_start = [l for l in main_lines if (l[0] + l[2])/2 <= 0.3 * image_width]
        main_lines = [l for l in main_lines if l not in left_start]
        print(f"    Left start: {len(left_start)} lines")
        print(f"    Time for left_start: {time.time() - start_time:.2f}s")
        
        left_end = set()
        step_time = time.time()
        for y in range(0, image_height, 10):
            left_start_x = [self.interpolate_x(l, y) for l in left_start if self.intersects_y(l, y)]
            candidates_x = [(self.interpolate_x(l, y), l) for l in main_lines if self.intersects_y(l, y)]
            
            for lsx in left_start_x:
                for cx, cl in candidates_x:
                    dist = abs(cx - lsx)
                    if abs(dist - self.line_width) <= self.tolerance:
                        left_end.add(tuple(cl))
                        if tuple(cl) in [tuple(ml) for ml in main_lines]:
                            for ml in main_lines:
                                if tuple(ml) == cl:
                                    main_lines.remove(ml)
                                    break
                        break
        
        print(f"    Left end: {len(left_end)} lines")
        print(f"    Time for left_end: {time.time() - step_time:.2f}s")
        
        right_start = set()
        step_time = time.time()
        for y in range(0, image_height, 10):
            left_end_x = [self.interpolate_x(l, y) for l in left_end if self.intersects_y(l, y)]
            candidates_x = [(self.interpolate_x(l, y), l) for l in main_lines if self.intersects_y(l, y)]
            
            for lex in left_end_x:
                for cx, cl in candidates_x:
                    dist = abs(cx - lex)
                    if cx > lex and abs(dist - self.pair_dist) <= self.pair_tolerance:
                        right_start.add(tuple(cl))
                        if tuple(cl) in [tuple(ml) for ml in main_lines]:
                            for ml in main_lines:
                                if tuple(ml) == cl:
                                    main_lines.remove(ml)
                                    break
                        break
        
        print(f"    Right start: {len(right_start)} lines")
        print(f"    Time for right_start: {time.time() - step_time:.2f}s")
        
        right_end = set()
        step_time = time.time()
        for y in range(0, image_height, 10):
            right_start_x = [self.interpolate_x(l, y) for l in right_start if self.intersects_y(l, y)]
            candidates_x = [(self.interpolate_x(l, y), l) for l in main_lines if self.intersects_y(l, y)]
            
            for rsx in right_start_x:
                for cx, cl in candidates_x:
                    dist = abs(cx - rsx)
                    if abs(dist - self.line_width) <= self.tolerance:
                        right_end.add(tuple(cl))
                        if tuple(cl) in [tuple(ml) for ml in main_lines]:
                            for ml in main_lines:
                                if tuple(ml) == cl:
                                    main_lines.remove(ml)
                                    break
                        break
        
        print(f"    Right end: {len(right_end)} lines")
        print(f"    Time for right_end: {time.time() - step_time:.2f}s")
        
        # Collect with groups
        filtered = []
        for l in left_start:
            filtered.append((l, 'left_start'))
        for l in [list(t) for t in left_end]:
            filtered.append((l, 'left_end'))
        for l in right_start:
            filtered.append((l, 'right_start'))
        for l in right_end:
            filtered.append((l, 'right_end'))
        
        print(f"    Total filtered: {len(filtered)} lines")
        print(f"    Total time: {time.time() - start_time:.2f}s")
        return filtered

    def remove_outliers(self, group_lines):
        if len(group_lines) < self.window_size + 1:
            return group_lines
        
        # Sort by avg y
        sorted_group = sorted(group_lines, key=lambda l: (l[1] + l[3])/2)
        
        avg_xs = [(l[0] + l[2])/2 for l in sorted_group]
        outliers = set()
        
        for i in range(self.window_size, len(avg_xs)):
            past_mean = np.mean(avg_xs[i - self.window_size : i])
            if abs(avg_xs[i] - past_mean) > self.deviation_factor:
                outliers.add(i)
        
        # Remove outliers (go back and remove)
        cleaned = [l for j, l in enumerate(sorted_group) if j not in outliers]
        
        # Optional: recursive or iterative to remove chains, but for now single pass
        return cleaned

    def create_line_graph(self, lines, image_height, image_width, output_dir, base_name):
        fig, ax = plt.subplots(figsize=(4, 12))
        ax.set_xlim(0, image_height)
        ax.set_ylim(0, image_width)
        ax.set_xlabel('Image Y')
        ax.set_ylabel('Line X')
        ax.invert_yaxis()
        
        for line, group in lines:
            x1, y1, x2, y2 = line
            if group == 'left_start':
                color = 'blue'
            elif group == 'left_end':
                color = 'cyan'
            elif group == 'right_start':
                color = 'green'
            else:
                color = 'yellow'
            ax.plot([y1, y2], [x1, x2], color=color, linewidth=1)
        
        graph_path = os.path.join(output_dir, f"{base_name}_line_x_profile.png")
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Line profile graph saved to {graph_path}")
        return graph_path

    def detect(self, image, image_path=None):
        height, width = image.shape[:2] if len(image.shape)>1 else (image.shape[0], image.shape[1] if len(image.shape)>1 else 1)
        _, lines = self.hough_detector.detect(image, image_path)
        filtered_lines = self.filter_island_lines(lines, width, height)
        
        # Remove outliers per group
        groups = {'left_start': [], 'left_end': [], 'right_start': [], 'right_end': []}
        for line, group in filtered_lines:
            groups[group].append(line)
        
        for g in groups:
            groups[g] = self.remove_outliers(groups[g])
        
        filtered_lines = [(line, g) for g in groups for line in groups[g]]

        # vis with colors
        vis = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for line, group in filtered_lines:
            x1, y1, x2, y2 = map(int, line)
            if group == 'left_start':
                color = (255, 0, 0)
            elif group == 'left_end':
                color = (255, 255, 0)
            elif group == 'right_start':
                color = (0, 255, 0)
            else:  # right_end
                color = (0, 255, 255)
            cv2.line(vis, (x1, y1), (x2, y2), color, 2)
        
        return vis, filtered_lines
    
    def save_debug_images(self, output_dir, base_name):
        self.hough_detector.save_debug_images(output_dir, base_name)

    def intersects_y(self, line, y):
        y1, y2 = line[1], line[3]
        return min(y1, y2) <= y <= max(y1, y2)

    def interpolate_x(self, line, y):
        x1, y1, x2, y2 = line
        if y1 == y2: return (x1 + x2) / 2
        t = (y - y1) / (y2 - y1)
        return x1 + t * (x2 - x1)

# Standalone
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Island Line Detection on TIFF Image")
    parser.add_argument("image_path", help="Path to input TIFF")
    parser.add_argument("--output_dir", default=None, help="Output dir")
    parser.add_argument("--sensitivity", default='medium', choices=['low', 'medium', 'high'])
    
    args = parser.parse_args()
    
    image = tifffile.imread(args.image_path)
    output_dir = args.output_dir or os.path.dirname(args.image_path)
    base_name = Path(args.image_path).stem
    
    detector = IslandLineDetector(sensitivity=args.sensitivity, debug=True)
    vis, lines = detector.detect(image, args.image_path)
    
    save_image(output_dir, base_name, vis, "island_lines_visualization")
    detector.save_debug_images(output_dir, base_name)
    
    # Save lines to JSON
    json_path = os.path.join(output_dir, f"{base_name}_island_lines.json")
    with open(json_path, 'w') as f:
        serializable_lines = [[int(c) for c in line] for line, _ in lines] if lines else []
        json.dump(serializable_lines, f, indent=2)
    
    height, width = image.shape[:2] if len(image.shape) > 2 else image.shape
    detector.create_line_graph(lines, height, width, output_dir, base_name)
    print(f"Graph saved.")

    print(f"Done. Detected {len(lines)} lines. Outputs in {output_dir}")
