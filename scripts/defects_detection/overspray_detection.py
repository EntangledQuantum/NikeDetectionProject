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
    """Detects overspray defects - scattered ink regions outside intended areas"""
    
    def __init__(self, region_size_range=(50, 500), proximity_threshold=50, kernel_size=15):
        """
        Args:
            region_size_range: Tuple of (min, max) area for overspray regions
            proximity_threshold: Max distance from main printed area to consider as overspray
            kernel_size: Size of morphological kernel for region analysis
        """
        self.region_size_range = region_size_range
        self.proximity_threshold = proximity_threshold
        self.kernel_size = kernel_size
        
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
    
    def detect_scattered_regions(self, image, main_regions):
        """Detect scattered ink regions that could be overspray"""
        # Threshold to get all printed areas
        _, all_printed = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if necessary
        if np.mean(all_printed) > 127:
            all_printed = cv2.bitwise_not(all_printed)
            
        # Subtract main regions to get potential overspray
        potential_overspray = cv2.subtract(all_printed, main_regions)
        
        # Use morphological operations to group nearby scattered dots into regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        
        # Close small gaps to group scattered dots
        closed = cv2.morphologyEx(potential_overspray, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Open to remove very small noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, 
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        
        # Label connected components
        labeled = measure.label(opened, connectivity=2)
        props = measure.regionprops(labeled)
        
        overspray_mask = np.zeros_like(opened)
        overspray_regions = []
        
        for prop in props:
            # Check if component size is within region range
            if self.region_size_range[0] <= prop.area <= self.region_size_range[1]:
                # Check proximity to main regions
                y, x = prop.centroid
                
                # Create distance transform from main regions
                dist_transform = cv2.distanceTransform(
                    cv2.bitwise_not(main_regions), 
                    cv2.DIST_L2, 
                    5
                )
                
                # Check if region is within proximity threshold
                if dist_transform[int(y), int(x)] <= self.proximity_threshold:
                    # Expand the region slightly for better visibility
                    expanded_coords = []
                    for coord in prop.coords:
                        y_coord, x_coord = coord
                        # Add surrounding pixels
                        for dy in range(-3, 4):
                            for dx in range(-3, 4):
                                new_y = max(0, min(image.shape[0]-1, y_coord + dy))
                                new_x = max(0, min(image.shape[1]-1, x_coord + dx))
                                expanded_coords.append([new_y, new_x])
                    
                    overspray_regions.append({
                        'centroid': (int(x), int(y)),
                        'area': prop.area,
                        'bbox': prop.bbox,
                        'expanded_area': len(expanded_coords)
                    })
                    
                    # Add expanded region to mask
                    for coord in expanded_coords:
                        overspray_mask[coord[0], coord[1]] = 255
                    
        return overspray_mask, overspray_regions
    
    def detect(self, image):
        """Main detection method"""
        # Preprocess
        gray, denoised = self.preprocess_image(image)
        
        # Find main printed regions
        main_regions = self.find_main_printed_regions(denoised)
        
        # Detect scattered regions
        overspray_mask, overspray_regions = self.detect_scattered_regions(denoised, main_regions)
        
        # Create visualization
        visualization = self.create_visualization(image, overspray_mask, overspray_regions)
        
        # Return tuple format (visualization, defects)
        return visualization, overspray_regions
    
    def create_visualization(self, original, mask, regions):
        """Create visualization with detected overspray regions highlighted - NO TEXT, JUST REGIONS"""
        if len(original.shape) == 2:
            vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            vis = original.copy()
            
        # Create overlay - mark ENTIRE overspray regions in red
        overlay = vis.copy()
        overlay[mask > 0] = [0, 0, 255]  # Red for overspray regions
        
        # Blend with original to show the ENTIRE affected areas
        result = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
            
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