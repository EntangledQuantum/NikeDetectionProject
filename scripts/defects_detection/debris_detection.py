"""
Debris Detection Algorithm
Detects foreign particles on sheets including pre-print and post-print debris

Author: Assistant
Date: 2024
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology, measure, filters, feature
import os
from tqdm import tqdm


class DebrisDetector:
    """Detects debris defects - foreign particles with characteristic patterns"""
    
    def __init__(self, halo_threshold=30, particle_size_range=(10, 500),
                 contrast_threshold=40):
        """
        Args:
            halo_threshold: Minimum intensity difference for halo detection
            particle_size_range: Min and max size for particle detection
            contrast_threshold: Minimum contrast for debris detection
        """
        self.halo_threshold = halo_threshold
        self.particle_size_range = particle_size_range
        self.contrast_threshold = contrast_threshold
        
    def preprocess_image(self, image):
        """Preprocess image for debris detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply median filter to reduce noise
        denoised = cv2.medianBlur(gray, 5)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        return gray, denoised, enhanced
    
    def detect_dark_spots_with_halos(self, image):
        """Detect dark spots with bright halos (pre-print debris)"""
        # Find dark spots
        _, dark_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Clean up small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(dark_thresh, cv2.MORPH_OPEN, kernel)
        
        # Label dark regions
        labeled = measure.label(cleaned)
        props = measure.regionprops(labeled, intensity_image=image)
        
        debris_with_halos = []
        halo_mask = np.zeros_like(image)
        
        for prop in props:
            if self.particle_size_range[0] <= prop.area <= self.particle_size_range[1]:
                # Check for halo around dark spot
                y, x = prop.centroid
                
                # Create ring mask around the spot
                radius_inner = int(np.sqrt(prop.area / np.pi))
                radius_outer = int(radius_inner * 2.5)
                
                # Create masks
                y_grid, x_grid = np.ogrid[:image.shape[0], :image.shape[1]]
                inner_mask = (x_grid - x)**2 + (y_grid - y)**2 <= radius_inner**2
                outer_mask = (x_grid - x)**2 + (y_grid - y)**2 <= radius_outer**2
                ring_mask = outer_mask & ~inner_mask
                
                # Calculate intensities
                if np.any(ring_mask):
                    ring_mean = np.mean(image[ring_mask])
                    spot_mean = prop.mean_intensity
                    
                    # Check if ring is brighter than spot
                    if ring_mean - spot_mean > self.halo_threshold:
                        debris_with_halos.append({
                            'centroid': (int(x), int(y)),
                            'area': prop.area,
                            'bbox': prop.bbox,
                            'spot_intensity': spot_mean,
                            'halo_intensity': ring_mean,
                            'type': 'pre-print'
                        })
                        
                        # Add to mask
                        coords = prop.coords
                        halo_mask[coords[:, 0], coords[:, 1]] = 255
                        
        return halo_mask, debris_with_halos
    
    def detect_surface_particles(self, image):
        """Detect surface particles (post-print debris)"""
        # Use gradient information to find particles
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Threshold gradient to find edges
        grad_thresh = gradient_magnitude > np.percentile(gradient_magnitude, 90)
        
        # Fill regions
        filled = ndimage.binary_fill_holes(grad_thresh)
        
        # Remove large regions (likely not particles)
        labeled = measure.label(filled)
        props = measure.regionprops(labeled, intensity_image=image)
        
        surface_particles = []
        particle_mask = np.zeros_like(image)
        
        for prop in props:
            if self.particle_size_range[0] <= prop.area <= self.particle_size_range[1]:
                # Check contrast with surroundings
                y, x = prop.centroid
                
                # Get bounding box with margin
                minr, minc, maxr, maxc = prop.bbox
                margin = 10
                minr = max(0, minr - margin)
                minc = max(0, minc - margin)
                maxr = min(image.shape[0], maxr + margin)
                maxc = min(image.shape[1], maxc + margin)
                
                # Calculate contrast
                roi = image[minr:maxr, minc:maxc]
                particle_mean = prop.mean_intensity
                surrounding_mean = np.mean(roi) 
                
                contrast = abs(particle_mean - surrounding_mean)
                
                if contrast > self.contrast_threshold:
                    surface_particles.append({
                        'centroid': (int(x), int(y)),
                        'area': prop.area,
                        'bbox': prop.bbox,
                        'contrast': contrast,
                        'type': 'post-print'
                    })
                    
                    coords = prop.coords
                    particle_mask[coords[:, 0], coords[:, 1]] = 255
                    
        return particle_mask, surface_particles
    
    def detect_fiber_like_debris(self, image):
        """Detect elongated fiber-like debris"""
        # Edge detection
        edges = feature.canny(image, sigma=2)
        
        # Hough line detection for fibers
        lines = cv2.HoughLinesP(edges.astype(np.uint8), 1, np.pi/180, 
                               threshold=30, minLineLength=20, maxLineGap=5)
        
        fiber_mask = np.zeros_like(image)
        fibers = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if length > 30:  # Minimum fiber length
                    # Check if it's actually a fiber (not just an edge)
                    # Sample points along the line
                    num_samples = int(length)
                    xs = np.linspace(x1, x2, num_samples)
                    ys = np.linspace(y1, y2, num_samples)
                    
                    # Check consistency of darkness along the line
                    values = []
                    for i in range(len(xs)):
                        if 0 <= int(ys[i]) < image.shape[0] and 0 <= int(xs[i]) < image.shape[1]:
                            values.append(image[int(ys[i]), int(xs[i])])
                    
                    if values and np.std(values) < 30:  # Consistent darkness
                        fibers.append({
                            'start': (x1, y1),
                            'end': (x2, y2),
                            'length': length,
                            'type': 'fiber'
                        })
                        
                        cv2.line(fiber_mask, (x1, y1), (x2, y2), 255, 2)
                        
        return fiber_mask, fibers
    
    def detect(self, image):
        """Main detection method"""
        # Preprocess
        gray, denoised, enhanced = self.preprocess_image(image)
        
        # Detect different types of debris
        halo_mask, debris_with_halos = self.detect_dark_spots_with_halos(enhanced)
        particle_mask, debris_without_halos = self.detect_surface_particles(denoised)
        fiber_mask, fibers = self.detect_fiber_like_debris(enhanced)
        
        # Combine all debris
        combined_mask = cv2.bitwise_or(halo_mask, particle_mask)
        combined_mask = cv2.bitwise_or(combined_mask, fiber_mask)
        
        # Create visualization
        visualization = self.create_visualization(
            image, halo_mask, particle_mask, fiber_mask,
            debris_with_halos, debris_without_halos, fibers
        )
        
        # Combine all defects
        all_defects = []
        for debris in debris_with_halos:
            all_defects.append({
                'type': 'debris_with_halo',
                'location': debris.get('centroid', (0, 0)),
                'size': debris.get('area', 0),
                'halo_strength': debris.get('halo_intensity', 0) - debris.get('spot_intensity', 0)
            })
        for debris in debris_without_halos:
            all_defects.append({
                'type': 'post_print_debris',
                'location': debris.get('centroid', (0, 0)),
                'size': debris.get('area', 0)
            })
        for fiber in fibers:
            all_defects.append({
                'type': 'fiber',
                'location': fiber.get('start', (0, 0)),  # Use start point as location
                'end_point': fiber.get('end', (0, 0)),
                'length': fiber.get('length', 0)
            })
        
        # Return tuple format (visualization, defects)
        return visualization, all_defects
    
    def create_visualization(self, original, halo_mask, particle_mask, 
                           fiber_mask, halo_debris, particles, fibers):
        """Create visualization with detected debris highlighted"""
        if len(original.shape) == 2:
            vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            vis = original.copy()
            
        # Create overlay
        overlay = vis.copy()
        
        # Highlight pre-print debris (with halos) in red
        overlay[halo_mask > 0] = [0, 0, 255]
        
        # Highlight post-print particles in green
        overlay[particle_mask > 0] = [0, 255, 0]
        
        # Highlight fibers in blue
        overlay[fiber_mask > 0] = [255, 0, 0]
        
        # Blend with original
        result = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        
        # Mark pre-print debris
        for debris in halo_debris:
            cv2.circle(result, debris['centroid'], 15, (0, 0, 255), 2)
            cv2.putText(result, 'H', 
                       (debris['centroid'][0] - 5, debris['centroid'][1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
        # Mark post-print particles
        for particle in particles:
            cv2.rectangle(result, 
                         (particle['bbox'][1], particle['bbox'][0]),
                         (particle['bbox'][3], particle['bbox'][2]),
                         (0, 255, 0), 2)
            cv2.putText(result, 'P', 
                       (particle['centroid'][0] - 5, particle['centroid'][1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        # Mark fibers
        for fiber in fibers:
            cv2.line(result, fiber['start'], fiber['end'], (255, 0, 0), 3)
            cv2.putText(result, 'F', 
                       fiber['start'],
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
        return result


def process_single_image(image_path, output_dir, detector):
    """Process a single image for debris detection"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    # Detect debris
    results = detector.detect(image)
    
    # Save results
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save visualization
    vis_path = os.path.join(output_dir, f"{base_name}_debris_detection.jpg")
    cv2.imwrite(vis_path, results['visualization'])
    
    # Save masks
    mask_path = os.path.join(output_dir, f"{base_name}_debris_mask.png")
    cv2.imwrite(mask_path, results['defect_mask'])
    
    return results


def main():
    """Example usage of debris detection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect debris on printed materials')
    parser.add_argument('input', help='Input image or directory path')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    parser.add_argument('--halo-threshold', type=int, default=30,
                       help='Minimum intensity difference for halo detection')
    parser.add_argument('--min-size', type=int, default=10,
                       help='Minimum particle size')
    parser.add_argument('--max-size', type=int, default=500,
                       help='Maximum particle size')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize detector
    detector = DebrisDetector(
        halo_threshold=args.halo_threshold,
        particle_size_range=(args.min_size, args.max_size)
    )
    
    # Process images
    if os.path.isfile(args.input):
        results = process_single_image(args.input, args.output, detector)
        if results:
            print(f"Detected {len(results['debris'])} debris particles")
            print(f"Detected {len(results['fibers'])} fibers")
    else:
        image_files = [f for f in os.listdir(args.input) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        
        for img_file in tqdm(image_files, desc="Processing images"):
            img_path = os.path.join(args.input, img_file)
            process_single_image(img_path, args.output, detector)


if __name__ == "__main__":
    main() 