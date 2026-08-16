"""
Surface Treatment Defect Detection Algorithm
Detects bad surface energy causing ink coalescence and missing ink areas
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from skimage import morphology, measure, filters
import os
from tqdm import tqdm


class SurfaceTreatmentDetector:
    """Detect surface-treatment issues: irregular drops and void (no-ink) areas.

    This detector enhances contrast, finds high-contrast irregular drops and
    missing-ink voids within expected printed coverage, and returns a
    visualization that highlights the entirety of affected regions.
    """
    
    def __init__(self, contrast_threshold=50, void_size_threshold=150, 
                 coalescence_threshold=300, kernel_size=10):
        """Configure thresholds for drop/void detection and morphology.

        Args:
            contrast_threshold: Local stddev threshold for high-contrast drops.
            void_size_threshold: Minimum connected area to consider as void.
            coalescence_threshold: Minimum area to keep high-contrast regions.
            kernel_size: Morphological kernel size (ellipse) for region shaping.
        """
        self.contrast_threshold = contrast_threshold
        self.void_size_threshold = void_size_threshold
        self.coalescence_threshold = coalescence_threshold
        self.kernel_size = kernel_size
        
    def preprocess_image(self, image):
        """Convert to grayscale and apply CLAHE for contrast enhancement.

        Args:
            image: Input image (BGR or grayscale).

        Returns:
            Tuple[ndarray, ndarray]: (gray, enhanced_gray).
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return gray, enhanced
    
    def detect_high_contrast_drops(self, image):
        """Detect irregular high-contrast drops using local std deviation.

        Args:
            image: Enhanced grayscale image.

        Returns:
            Tuple[ndarray, List[dict]]: (drop_mask, drop_properties)
        """
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
        
        # Use enhanced morphological operations - moderate grouping
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
        high_contrast_uint8 = high_contrast.astype(np.uint8) * 255
        closed = cv2.morphologyEx(high_contrast_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Clean up and label regions with PROPER threshold
        cleaned = morphology.remove_small_objects(closed.astype(bool), min_size=self.coalescence_threshold)
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
        """Detect missing-ink voids within expected printed coverage.

        Args:
            image: Grayscale image.

        Returns:
            Tuple[ndarray, List[dict]]: (void_mask, void_properties)
        """
        # Apply adaptive thresholding to find light areas
        adaptive_thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10
        )
        
        # Find main printed areas
        _, global_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(global_thresh) > 127:
            global_thresh = cv2.bitwise_not(global_thresh)
            
        # Dilate moderately to get expected coverage with slight bump
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
        expected_coverage = cv2.dilate(global_thresh, kernel, iterations=3)
        
        # Find voids within expected coverage
        potential_voids = cv2.bitwise_and(adaptive_thresh, expected_coverage)
        
        # Use moderate morphological closing - connect nearby areas
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16))
        potential_voids = cv2.morphologyEx(potential_voids, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        
        # Clean up small regions (but keep threshold low to catch large areas)
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
        """Detect texture anomalies via FFT radial profile analysis.

        Args:
            image: Grayscale image.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (anomalies_boolean_mask, radial_profile)
        """
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
        """Run surface-treatment detection and return visualization and defects.

        Args:
            image: Input image (BGR or grayscale).

        Returns:
            tuple: (visualization_bgr, defects)
        """
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
        """Create visualization marking entire no-ink and high-contrast regions.

        Args:
            original: Original input image (BGR or grayscale).
            drop_mask: Binary mask of high-contrast drops.
            void_mask: Binary mask of void areas.
            drops: List of detected drop properties.
            voids: List of detected void properties.

        Returns:
            BGR visualization image with regions highlighted.
        """
        if len(original.shape) == 2:
            vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            vis = original.copy()
            
        # Create overlay - mark ENTIRE problem areas
        overlay = vis.copy()
        
        # Mark areas with NO INK (voids) in green - this is the main surface treatment issue
        overlay[void_mask > 0] = [0, 255, 0]  # Green for areas with no ink
        
        # Mark irregular high-contrast drops in green as well (they indicate surface energy problems)
        overlay[drop_mask > 0] = [0, 255, 0]  # Green for irregular drops
        
        # Blend with original to show the ENTIRE affected areas
        result = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
            
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