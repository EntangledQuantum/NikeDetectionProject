"""
Surface Treatment Defect Detection Algorithm
Detects bad surface energy causing ink coalescence and missing ink areas

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


class SurfaceTreatmentDetector:
    """Detects surface treatment defects - irregular ink drops and voids"""
    
    def __init__(self, contrast_threshold=50, void_size_threshold=20, 
                 coalescence_threshold=100):
        """
        Args:
            contrast_threshold: Minimum contrast for high-contrast drops
            void_size_threshold: Minimum size for void areas
            coalescence_threshold: Minimum size for coalesced ink drops
        """
        self.contrast_threshold = contrast_threshold
        self.void_size_threshold = void_size_threshold
        self.coalescence_threshold = coalescence_threshold
        
    def preprocess_image(self, image):
        """Preprocess image for surface treatment detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return gray, enhanced
    
    def detect_high_contrast_drops(self, image):
        """Detect irregular high-contrast ink drops"""
        # Calculate local standard deviation
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        
        # Local mean
        local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
        
        # Local variance
        squared_img = image.astype(np.float32) ** 2
        local_mean_sq = cv2.filter2D(squared_img, -1, kernel)
        local_variance = local_mean_sq - local_mean ** 2
        local_std = np.sqrt(np.maximum(local_variance, 0))
        
        # High contrast regions
        high_contrast = local_std > self.contrast_threshold
        
        # Clean up and label regions
        cleaned = morphology.remove_small_objects(high_contrast, min_size=30)
        labeled = measure.label(cleaned)
        props = measure.regionprops(labeled, intensity_image=image)
        
        high_contrast_drops = []
        drop_mask = np.zeros_like(image)
        
        for prop in props:
            # Check for irregular shape (high eccentricity or low solidity)
            if prop.eccentricity > 0.7 or prop.solidity < 0.8:
                if prop.area >= self.coalescence_threshold:
                    high_contrast_drops.append({
                        'centroid': prop.centroid,
                        'area': prop.area,
                        'eccentricity': prop.eccentricity,
                        'solidity': prop.solidity,
                        'bbox': prop.bbox,
                        'mean_intensity': prop.mean_intensity
                    })
                    
                    coords = prop.coords
                    drop_mask[coords[:, 0], coords[:, 1]] = 255
                    
        return drop_mask, high_contrast_drops
    
    def detect_void_areas(self, image):
        """Detect areas with missing ink (voids)"""
        # Apply adaptive thresholding to find light areas
        adaptive_thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10
        )
        
        # Find main printed areas
        _, global_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(global_thresh) > 127:
            global_thresh = cv2.bitwise_not(global_thresh)
            
        # Dilate to get expected coverage
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        expected_coverage = cv2.dilate(global_thresh, kernel, iterations=2)
        
        # Find voids within expected coverage
        potential_voids = cv2.bitwise_and(adaptive_thresh, expected_coverage)
        
        # Clean up small regions
        cleaned_voids = morphology.remove_small_objects(
            potential_voids.astype(bool), 
            min_size=self.void_size_threshold
        )
        
        # Label and analyze
        labeled_voids = measure.label(cleaned_voids)
        void_props = measure.regionprops(labeled_voids, intensity_image=image)
        
        void_areas = []
        void_mask = np.zeros_like(image)
        
        for prop in void_props:
            # Check if it's genuinely a void (high intensity compared to surroundings)
            y, x = prop.centroid
            roi_y = slice(max(0, int(y)-20), min(image.shape[0], int(y)+20))
            roi_x = slice(max(0, int(x)-20), min(image.shape[1], int(x)+20))
            
            surrounding_mean = np.mean(image[roi_y, roi_x])
            
            if prop.mean_intensity > surrounding_mean * 1.2:
                void_areas.append({
                    'centroid': prop.centroid,
                    'area': prop.area,
                    'bbox': prop.bbox,
                    'intensity_ratio': prop.mean_intensity / surrounding_mean
                })
                
                coords = prop.coords
                void_mask[coords[:, 0], coords[:, 1]] = 255
                
        return void_mask, void_areas
    
    def detect_texture_irregularities(self, image):
        """Detect texture irregularities using frequency analysis"""
        # Apply FFT to detect texture irregularities
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Create radial profile
        center = (magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2)
        y, x = np.ogrid[:magnitude_spectrum.shape[0], :magnitude_spectrum.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r = r.astype(int)
        
        # Calculate radial average
        radial_prof = ndimage.mean(magnitude_spectrum, labels=r, index=np.arange(0, r.max()))
        
        # Find anomalies in radial profile
        smooth_prof = signal.savgol_filter(radial_prof, 11, 3)
        anomalies = np.abs(radial_prof - smooth_prof) > np.std(radial_prof) * 2
        
        return anomalies, radial_prof
    
    def detect(self, image):
        """Main detection method"""
        # Preprocess
        gray, enhanced = self.preprocess_image(image)
        
        # Detect high contrast drops
        drop_mask, drops = self.detect_high_contrast_drops(enhanced)
        
        # Detect void areas
        void_mask, voids = self.detect_void_areas(gray)
        
        # Detect texture irregularities
        texture_anomalies, radial_prof = self.detect_texture_irregularities(gray)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(drop_mask, void_mask)
        
        # Create visualization
        visualization = self.create_visualization(
            image, drop_mask, void_mask, drops, voids
        )
        
        # Combine all defects
        all_defects = []
        for drop in drops:
            all_defects.append({
                'type': 'high_contrast_drop',
                'location': drop.get('centroid', drop.get('location', (0, 0))),
                'contrast': drop.get('contrast', 0),
                'size': drop.get('size', drop.get('area', 0))
            })
        for void in voids:
            all_defects.append({
                'type': 'void_area',
                'location': void.get('centroid', void.get('location', (0, 0))),
                'area': void.get('area', 0)
            })
        
        # Return tuple format (visualization, defects)
        return visualization, all_defects
    
    def create_visualization(self, original, drop_mask, void_mask, drops, voids):
        """Create visualization with detected defects highlighted"""
        if len(original.shape) == 2:
            vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            vis = original.copy()
            
        # Create overlay
        overlay = vis.copy()
        
        # Highlight high contrast drops in blue
        overlay[drop_mask > 0] = [255, 0, 0]
        
        # Highlight voids in yellow
        overlay[void_mask > 0] = [0, 255, 255]
        
        # Blend with original
        result = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        
        # Mark drops
        for drop in drops:
            y, x = drop['centroid']
            cv2.drawContours(result, [np.array([[
                drop['bbox'][1], drop['bbox'][0],
                drop['bbox'][3], drop['bbox'][2]
            ]]).reshape((-1, 1, 2))], -1, (0, 0, 255), 2)
            cv2.putText(result, 'D', 
                       (int(x) - 5, int(y) + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
        # Mark voids
        for void in voids:
            y, x = void['centroid']
            cv2.circle(result, (int(x), int(y)), 15, (0, 255, 255), 2)
            cv2.putText(result, 'V', 
                       (int(x) - 5, int(y) + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
        return result


def process_single_image(image_path, output_dir, detector):
    """Process a single image for surface treatment defects"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    # Detect defects
    results = detector.detect(image)
    
    # Save results
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save visualization
    vis_path = os.path.join(output_dir, f"{base_name}_surface_treatment_detection.jpg")
    cv2.imwrite(vis_path, results['visualization'])
    
    # Save masks
    mask_path = os.path.join(output_dir, f"{base_name}_surface_treatment_mask.png")
    cv2.imwrite(mask_path, results['defect_mask'])
    
    drop_mask_path = os.path.join(output_dir, f"{base_name}_drops_mask.png")
    cv2.imwrite(drop_mask_path, results['drop_mask'])
    
    void_mask_path = os.path.join(output_dir, f"{base_name}_voids_mask.png")
    cv2.imwrite(void_mask_path, results['void_mask'])
    
    return results


def main():
    """Example usage of surface treatment detection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect surface treatment defects')
    parser.add_argument('input', help='Input image or directory path')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    parser.add_argument('--contrast-threshold', type=int, default=50,
                       help='Minimum contrast for high-contrast drops')
    parser.add_argument('--void-size', type=int, default=20,
                       help='Minimum size for void areas')
    parser.add_argument('--coalescence-size', type=int, default=100,
                       help='Minimum size for coalesced ink drops')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize detector
    detector = SurfaceTreatmentDetector(
        contrast_threshold=args.contrast_threshold,
        void_size_threshold=args.void_size,
        coalescence_threshold=args.coalescence_size
    )
    
    # Process images
    if os.path.isfile(args.input):
        results = process_single_image(args.input, args.output, detector)
        if results:
            print(f"Detected {len(results['drops'])} high-contrast drops")
            print(f"Detected {len(results['voids'])} void areas")
    else:
        image_files = [f for f in os.listdir(args.input) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        
        for img_file in tqdm(image_files, desc="Processing images"):
            img_path = os.path.join(args.input, img_file)
            process_single_image(img_path, args.output, detector)


if __name__ == "__main__":
    main() 