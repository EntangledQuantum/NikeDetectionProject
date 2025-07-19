"""
Edge Defect Detection Algorithm
Detects irregularities along printed edges and jagged boundaries

Author: Assistant
Date: 2024
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from skimage import morphology, measure, filters
import os
from tqdm import tqdm


class EdgeDefectDetector:
    """Detects REAL edge defects - misalignment and significant irregularities only"""
    
    def __init__(self, deviation_threshold=15, min_defect_length=50, 
                 straightness_tolerance=5):
        """
        Args:
            deviation_threshold: Maximum allowed deviation from straight line (pixels)
            min_defect_length: Minimum length to consider as defect (pixels)
            straightness_tolerance: Tolerance for what's considered "straight"
        """
        self.deviation_threshold = deviation_threshold
        self.min_defect_length = min_defect_length
        self.straightness_tolerance = straightness_tolerance
        
    def preprocess_image(self, image):
        """Preprocess image for edge defect detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(gray, 9, 50, 50)
        
        return gray, denoised
    
    def extract_edges(self, image):
        """Extract edges using multiple methods"""
        # Canny edge detection
        canny_edges = cv2.Canny(image, 50, 150)
        
        # Sobel edge detection
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_edges = (sobel_magnitude > 50).astype(np.uint8) * 255
        
        # Combine edges
        combined_edges = cv2.bitwise_or(canny_edges, sobel_edges)
        
        # Clean up edges
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
        
        # Thinning to get single-pixel edges
        # Convert to binary for skeletonization
        _, binary = cv2.threshold(cleaned, 1, 1, cv2.THRESH_BINARY)
        thinned = morphology.skeletonize(binary.astype(bool)).astype(np.uint8) * 255
        
        return canny_edges, sobel_edges, combined_edges, thinned, sobel_x, sobel_y
    
    def analyze_edge_smoothness(self, edges, grad_x, grad_y):
        """Analyze edge smoothness to detect irregularities"""
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        edge_defects = []
        defect_mask = np.zeros_like(edges)
        
        for contour in contours:
            if len(contour) < self.min_defect_length:
                continue
                
            # Analyze contour smoothness
            defects = self.detect_contour_irregularities(contour, grad_x, grad_y)
            
            if defects:
                edge_defects.extend(defects)
                
                # Draw defects on mask
                for defect in defects:
                    start_idx, end_idx = defect['indices']
                    defect_points = contour[start_idx:end_idx+1]
                    cv2.polylines(defect_mask, [defect_points], False, 255, 3)
                    
        return defect_mask, edge_defects
    
    def detect_contour_irregularities(self, contour, grad_x, grad_y):
        """Detect irregularities in a single contour"""
        contour = contour.squeeze()
        if len(contour.shape) == 1:
            return []
            
        defects = []
        
        # Calculate local curvature
        window_size = 5
        for i in range(window_size, len(contour) - window_size):
            # Get local segment
            segment = contour[i-window_size:i+window_size+1]
            
            # Fit line to segment
            if len(segment) > 2:
                vx, vy, x, y = cv2.fitLine(segment, cv2.DIST_L2, 0, 0.01, 0.01)
                
                # Calculate distances from points to fitted line
                distances = []
                for point in segment:
                    # Point-to-line distance
                    d = abs((point[1] - y) * vx - (point[0] - x) * vy) / np.sqrt(vx**2 + vy**2)
                    distances.append(d)
                    
                max_deviation = max(distances)
                
                # Check if deviation exceeds threshold
                if max_deviation > self.smoothness_threshold:
                    # Check gradient consistency
                    grad_consistent = self.check_gradient_consistency(
                        segment, grad_x, grad_y
                    )
                    
                    if not grad_consistent:
                        defects.append({
                            'type': 'jagged_edge',
                            'position': contour[i],
                            'deviation': max_deviation,
                            'indices': (i-window_size, i+window_size)
                        })
                        
        # Merge nearby defects
        merged_defects = self.merge_nearby_defects(defects, contour)
        
        return merged_defects
    
    def check_gradient_consistency(self, segment, grad_x, grad_y):
        """Check if gradient direction is consistent along segment"""
        gradients = []
        
        for point in segment:
            x, y = point
            if 0 <= y < grad_x.shape[0] and 0 <= x < grad_x.shape[1]:
                gx = grad_x[y, x]
                gy = grad_y[y, x]
                angle = np.arctan2(gy, gx)
                gradients.append(angle)
                
        if len(gradients) > 2:
            # Check consistency
            grad_std = np.std(gradients)
            return grad_std < np.pi / 6  # 30 degrees tolerance
            
        return True
    
    def merge_nearby_defects(self, defects, contour):
        """Merge nearby defects into continuous regions"""
        if not defects:
            return []
            
        # Sort by position along contour
        defects.sort(key=lambda d: d['indices'][0])
        
        merged = []
        current = defects[0]
        
        for defect in defects[1:]:
            # Check if defects overlap or are adjacent
            if current['indices'][1] >= defect['indices'][0] - 5:
                # Merge
                current['indices'] = (current['indices'][0], defect['indices'][1])
                current['deviation'] = max(current['deviation'], defect['deviation'])
            else:
                # Check if merged defect is long enough
                length = current['indices'][1] - current['indices'][0]
                if length >= self.min_defect_length:
                    merged.append(current)
                current = defect
                
        # Add last defect
        length = current['indices'][1] - current['indices'][0]
        if length >= self.min_defect_length:
            merged.append(current)
            
        return merged
    
    def detect_edge_breaks(self, edges):
        """Detect breaks or gaps in edges"""
        # Distance transform from edges
        dist_transform = cv2.distanceTransform(
            cv2.bitwise_not(edges), cv2.DIST_L2, 5
        )
        
        # Find potential break points
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Areas that should have edges but don't
        edge_region = cv2.dilate(edges, kernel, iterations=3)
        missing_edges = cv2.bitwise_and(edge_region, cv2.bitwise_not(dilated_edges))
        
        # Clean up small gaps
        cleaned_gaps = morphology.remove_small_objects(
            missing_edges.astype(bool), min_size=20
        )
        
        return cleaned_gaps.astype(np.uint8) * 255
    
    def detect(self, image):
        """Main detection method - focus on REAL edge problems only"""
        # Preprocess
        gray, denoised = self.preprocess_image(image)
        
        # Find the main printed strip boundaries
        strip_edges = self.find_strip_edges(denoised)
        
        # Check if edges are straight and properly aligned
        edge_defects = self.check_edge_straightness(strip_edges, gray)
        
        # Create visualization
        visualization = self.create_visualization(image, strip_edges, edge_defects)
        
        # Return tuple format (visualization, defects)
        return visualization, edge_defects
    
    def find_strip_edges(self, image):
        """Find the main edges of the printed strip"""
        h, w = image.shape
        
        # Use Otsu thresholding to separate printed from non-printed areas
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # For tall strips, we mainly care about left and right edges
        edges = {
            'left': [],
            'right': [],
            'top': [],
            'bottom': []
        }
        
        # Find left and right edges by scanning horizontally
        for y in range(0, h, 20):  # Sample every 20 pixels
            row = binary[y, :]
            
            # Find left edge (first white pixel)
            left_edge = np.where(row > 128)[0]
            if len(left_edge) > 0:
                edges['left'].append((left_edge[0], y))
            
            # Find right edge (last white pixel)
            if len(left_edge) > 0:
                edges['right'].append((left_edge[-1], y))
        
        # Find top and bottom edges by scanning vertically
        for x in range(0, w, 20):  # Sample every 20 pixels
            col = binary[:, x]
            
            # Find top edge (first white pixel)
            top_edge = np.where(col > 128)[0]
            if len(top_edge) > 0:
                edges['top'].append((x, top_edge[0]))
            
            # Find bottom edge (last white pixel)
            if len(top_edge) > 0:
                edges['bottom'].append((x, top_edge[-1]))
        
        return edges
    
    def check_edge_straightness(self, edges, image):
        """Check if edges are straight and mark only REAL problems"""
        defects = []
        
        for edge_name, points in edges.items():
            if len(points) < 10:  # Need enough points to analyze
                continue
                
            # Convert to numpy array
            points_array = np.array(points)
            
            # Fit a line through the edge points
            if edge_name in ['left', 'right']:
                # For vertical edges, fit x = my + b
                if len(points_array) > 1:
                    y_coords = points_array[:, 1]
                    x_coords = points_array[:, 0]
                    
                    # Fit line
                    coeffs = np.polyfit(y_coords, x_coords, 1)
                    fitted_x = np.polyval(coeffs, y_coords)
                    
                    # Calculate deviations
                    deviations = np.abs(x_coords - fitted_x)
                    
            else:  # top, bottom edges
                # For horizontal edges, fit y = mx + b
                if len(points_array) > 1:
                    x_coords = points_array[:, 0]
                    y_coords = points_array[:, 1]
                    
                    # Fit line
                    coeffs = np.polyfit(x_coords, y_coords, 1)
                    fitted_y = np.polyval(coeffs, x_coords)
                    
                    # Calculate deviations
                    deviations = np.abs(y_coords - fitted_y)
            
            # Find areas where deviation exceeds threshold
            problem_indices = np.where(deviations > self.deviation_threshold)[0]
            
            if len(problem_indices) > 0:
                # Group consecutive problem areas
                problem_groups = []
                current_group = [problem_indices[0]]
                
                for i in range(1, len(problem_indices)):
                    if problem_indices[i] - problem_indices[i-1] <= 3:  # Close enough
                        current_group.append(problem_indices[i])
                    else:
                        if len(current_group) >= 3:  # Minimum group size
                            problem_groups.append(current_group)
                        current_group = [problem_indices[i]]
                
                # Add last group
                if len(current_group) >= 3:
                    problem_groups.append(current_group)
                
                # Create defects for significant problem areas
                for group in problem_groups:
                    start_idx = group[0]
                    end_idx = group[-1]
                    
                    # Calculate length of defective area
                    start_point = points_array[start_idx]
                    end_point = points_array[end_idx]
                    length = np.sqrt((end_point[0] - start_point[0])**2 + 
                                   (end_point[1] - start_point[1])**2)
                    
                    if length >= self.min_defect_length:
                        max_deviation = np.max(deviations[group])
                        
                        defects.append({
                            'type': f'{edge_name}_edge_misalignment',
                            'start_point': tuple(start_point),
                            'end_point': tuple(end_point),
                            'length': length,
                            'max_deviation': max_deviation,
                            'edge_side': edge_name
                        })
        
        print(f"    Found {len(defects)} REAL edge defects (not {len(defects)*1000}!)")
        return defects
    
    def create_visualization(self, original, strip_edges, edge_defects):
        """Create visualization with detected edge defects highlighted"""
        if len(original.shape) == 2:
            vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            vis = original.copy()
            
        # Draw the detected strip edges in light blue (for reference)
        for edge_name, points in strip_edges.items():
            if len(points) > 1:
                pts = np.array(points, dtype=np.int32)
                cv2.polylines(vis, [pts], False, (255, 200, 100), 1)
        
        # Highlight REAL edge defects in bright red
        for defect in edge_defects:
            start_point = tuple(map(int, defect['start_point']))
            end_point = tuple(map(int, defect['end_point']))
            
            # Draw thick red line for the defective edge section
            cv2.line(vis, start_point, end_point, (0, 0, 255), 4)
            
            # Add text showing deviation
            mid_point = ((start_point[0] + end_point[0]) // 2, 
                        (start_point[1] + end_point[1]) // 2)
            
            text = f"Edge: {defect['max_deviation']:.1f}px off"
            cv2.putText(vis, text, 
                       (mid_point[0] - 50, mid_point[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
        return vis


def process_single_image(image_path, output_dir, detector):
    """Process a single image for edge defect detection"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    # Detect edge defects
    results = detector.detect(image)
    
    # Save results
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save visualization
    vis_path = os.path.join(output_dir, f"{base_name}_edge_defect_detection.jpg")
    cv2.imwrite(vis_path, results['visualization'])
    
    # Save masks
    mask_path = os.path.join(output_dir, f"{base_name}_edge_defect_mask.png")
    cv2.imwrite(mask_path, results['defect_mask'])
    
    edge_path = os.path.join(output_dir, f"{base_name}_edges.png")
    cv2.imwrite(edge_path, results['edge_mask'])
    
    return results


def main():
    """Example usage of edge defect detection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect edge defects in printed materials')
    parser.add_argument('input', help='Input image or directory path')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    parser.add_argument('--smoothness', type=float, default=5,
                       help='Smoothness threshold for edge irregularities')
    parser.add_argument('--min-length', type=int, default=10,
                       help='Minimum defect length')
    parser.add_argument('--edge-width', type=int, default=20,
                       help='Width of edge region to analyze')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize detector
    detector = EdgeDefectDetector(
        smoothness_threshold=args.smoothness,
        min_defect_length=args.min_length,
        edge_width=args.edge_width
    )
    
    # Process images
    if os.path.isfile(args.input):
        results = process_single_image(args.input, args.output, detector)
        if results:
            print(f"Detected {results['defect_count']} edge defects")
    else:
        image_files = [f for f in os.listdir(args.input) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        
        for img_file in tqdm(image_files, desc="Processing images"):
            img_path = os.path.join(args.input, img_file)
            process_single_image(img_path, args.output, detector)


if __name__ == "__main__":
    main() 