"""
Overspray Detection Algorithm
Detects scattered ink dots outside the main printed areas

Author: Assistant
Date: 2024
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology, measure
import os
from tqdm import tqdm


class OversprayDetector:
    """Detects overspray defects - scattered ink dots outside intended areas"""
    
    def __init__(self, dot_size_range=(3, 15), proximity_threshold=50):
        """
        Args:
            dot_size_range: Tuple of (min, max) area for scattered dots
            proximity_threshold: Max distance from main printed area to consider as overspray
        """
        self.dot_size_range = dot_size_range
        self.proximity_threshold = proximity_threshold
        
    def preprocess_image(self, image):
        """Preprocess image for overspray detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        return gray, denoised
    
    def find_main_printed_regions(self, image):
        """Identify main printed regions using morphological operations"""
        # Threshold to get binary image
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if necessary (ensure printed areas are white)
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
            
        # Apply morphological closing to connect nearby printed regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Fill holes
        filled = ndimage.binary_fill_holes(closed).astype(np.uint8) * 255
        
        # Erode slightly to separate overspray from main regions
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        main_regions = cv2.erode(filled, kernel_small, iterations=2)
        
        return main_regions
    
    def detect_scattered_dots(self, image, main_regions):
        """Detect small scattered dots that could be overspray"""
        # Threshold to get all printed areas
        _, all_printed = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if necessary
        if np.mean(all_printed) > 127:
            all_printed = cv2.bitwise_not(all_printed)
            
        # Subtract main regions to get potential overspray
        potential_overspray = cv2.subtract(all_printed, main_regions)
        
        # Remove noise
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(potential_overspray, cv2.MORPH_OPEN, kernel)
        
        # Label connected components
        labeled = measure.label(cleaned, connectivity=2)
        props = measure.regionprops(labeled)
        
        overspray_mask = np.zeros_like(cleaned)
        overspray_dots = []
        
        for prop in props:
            # Check if component size is within dot range
            if self.dot_size_range[0] <= prop.area <= self.dot_size_range[1]:
                # Check proximity to main regions
                y, x = prop.centroid
                
                # Create distance transform from main regions
                dist_transform = cv2.distanceTransform(
                    cv2.bitwise_not(main_regions), 
                    cv2.DIST_L2, 
                    5
                )
                
                # Check if dot is within proximity threshold
                if dist_transform[int(y), int(x)] <= self.proximity_threshold:
                    overspray_dots.append({
                        'centroid': (int(x), int(y)),
                        'area': prop.area,
                        'bbox': prop.bbox
                    })
                    
                    # Add to mask
                    coords = prop.coords
                    overspray_mask[coords[:, 0], coords[:, 1]] = 255
                    
        return overspray_mask, overspray_dots
    
    def detect(self, image):
        """Main detection method"""
        # Preprocess
        gray, denoised = self.preprocess_image(image)
        
        # Find main printed regions
        main_regions = self.find_main_printed_regions(denoised)
        
        # Detect scattered dots
        overspray_mask, overspray_dots = self.detect_scattered_dots(denoised, main_regions)
        
        # Create visualization
        visualization = self.create_visualization(image, overspray_mask, overspray_dots)
        
        # Return tuple format (visualization, defects)
        return visualization, overspray_dots
    
    def create_visualization(self, original, mask, dots):
        """Create visualization with detected overspray highlighted"""
        if len(original.shape) == 2:
            vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            vis = original.copy()
            
        # Create overlay
        overlay = vis.copy()
        
        # Highlight overspray areas in red
        overlay[mask > 0] = [0, 0, 255]
        
        # Blend with original
        result = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        
        # Draw circles around detected dots
        for dot in dots:
            cv2.circle(result, dot['centroid'], 10, (0, 255, 0), 2)
            cv2.putText(result, 'O', 
                       (dot['centroid'][0] - 5, dot['centroid'][1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        return result


def process_single_image(image_path, output_dir, detector):
    """Process a single image for overspray detection"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    # Detect overspray
    results = detector.detect(image)
    
    # Save results
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save visualization
    vis_path = os.path.join(output_dir, f"{base_name}_overspray_detection.jpg")
    cv2.imwrite(vis_path, results['visualization'])
    
    # Save mask
    mask_path = os.path.join(output_dir, f"{base_name}_overspray_mask.png")
    cv2.imwrite(mask_path, results['defect_mask'])
    
    return results


def main():
    """Example usage of overspray detection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect overspray defects in printed materials')
    parser.add_argument('input', help='Input image or directory path')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    parser.add_argument('--min-dot-size', type=int, default=3, help='Minimum dot size')
    parser.add_argument('--max-dot-size', type=int, default=15, help='Maximum dot size')
    parser.add_argument('--proximity', type=int, default=50, help='Proximity threshold')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize detector
    detector = OversprayDetector(
        dot_size_range=(args.min_dot_size, args.max_dot_size),
        proximity_threshold=args.proximity
    )
    
    # Process images
    if os.path.isfile(args.input):
        results = process_single_image(args.input, args.output, detector)
        if results:
            print(f"Detected {results['defect_count']} overspray dots")
    else:
        image_files = [f for f in os.listdir(args.input) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        
        for img_file in tqdm(image_files, desc="Processing images"):
            img_path = os.path.join(args.input, img_file)
            process_single_image(img_path, args.output, detector)


if __name__ == "__main__":
    main() 