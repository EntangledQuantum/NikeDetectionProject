"""
Stripe Debris and Void Detection
Detects anomalies in stripe printing: debris (dark spots) and voids (missing ink)

This detector analyzes stripe images to find:
- Debris: Foreign particles or dark spots that shouldn't be in the clean fill
- Voids: Areas where color is not filled properly (missing ink)

Algorithm:
1. Measure stripe intensity baseline (mean and std of grayscale values)
2. Threshold for dark (debris) and bright (void) anomalies
3. Clean masks with morphological operations
4. Extract blobs via connected components
5. Filter by size and report detections

Author: Assistant
Date: 2024
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Tuple, Any, Optional
from detector_base import BaseDetector


class StripeDebrisVoidDetector(BaseDetector):
    """Detector for debris and voids in stripe images.
    
    Identifies anomalies in printed stripes by comparing pixel intensities
    against an adaptive baseline derived from the stripe's mean intensity.
    """
    
    def __init__(self, sensitivity: str = 'medium', debug: bool = False):
        """Initialize the stripe debris/void detector.
        
        Args:
            sensitivity: Detection sensitivity level ('low', 'medium', 'high').
            debug: If True, print debug information and save intermediate images.
        """
        super().__init__()
        self.sensitivity = sensitivity
        self.debug = debug
        
        # Sensitivity-dependent parameters
        self._configure_parameters()
        
        # Storage for debug images
        self.debug_images = {}
    
    def _configure_parameters(self):
        """Configure detection parameters based on sensitivity level."""
        if self.sensitivity == 'low':
            # Conservative detection - fewer false positives
            self.dark_factor = 0.60        # More restrictive for dark anomalies
            self.bright_factor = 1.45      # More restrictive for bright anomalies
            self.min_blob_area = 100       # Larger minimum size
            self.morph_kernel_size = 4     # Smaller morphological operations
            self.morph_iterations = 1
            
        elif self.sensitivity == 'high':
            # Aggressive detection - catch more defects
            self.dark_factor = 0.88        # Less restrictive for dark anomalies
            self.bright_factor = 1.15      # Less restrictive for bright anomalies
            self.min_blob_area = 20        # Smaller minimum size
            self.morph_kernel_size = 5     # Larger morphological operations
            self.morph_iterations = 2
            
        else:  # medium (default)
            # Balanced detection - ADJUSTED FOR BETTER DEBRIS DETECTION
            self.dark_factor = 0.82        # More sensitive to debris (was 0.65)
            self.bright_factor = 1.22      # More sensitive to voids (was 1.35)
            self.min_blob_area = 40        # Lower minimum size to catch smaller debris (was 80)
            self.morph_kernel_size = 3     # Smaller kernel to preserve details (was 6)
            self.morph_iterations = 1      # Less aggressive cleaning (was 2)
    
    def detect(self, image: np.ndarray, image_path: Optional[str] = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Detect debris and voids in a stripe image.
        
        Args:
            image: Input image (BGR format).
            image_path: Optional path to the image (used to load exclusion zones).
        
        Returns:
            Tuple of (visualization_image, defects_list).
            - visualization_image: BGR image with detected anomalies marked.
            - defects_list: List of defect dictionaries with keys:
                - type: 'debris' or 'void'
                - centroid: (cx, cy) tuple
                - area: blob area in pixels
                - bbox: (x, y, w, h) bounding box
                - intensity_deviation: how far from mean (for reference)
        """
        # Load exclusion zones if image path provided
        if image_path is not None:
            self.load_exclusion_zones(image_path)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Store original for visualization
        visualization = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Store grayscale in debug images
        if self.debug:
            self.debug_images['01_grayscale'] = gray.copy()
        
        if self.debug:
            print(f"\n=== Stripe Debris/Void Detection (Sensitivity: {self.sensitivity}) ===")
            print(f"Image shape: {gray.shape}")
            print(f"Parameters:")
            print(f"  - dark_factor: {self.dark_factor}")
            print(f"  - bright_factor: {self.bright_factor}")
            print(f"  - min_blob_area: {self.min_blob_area}")
            print(f"  - morph_kernel_size: {self.morph_kernel_size}")
            print(f"  - morph_iterations: {self.morph_iterations}")
        
        # Step 1: Measure stripe intensity baseline
        mean_intensity, std_intensity = self._compute_baseline(gray)
        
        if self.debug:
            print(f"Baseline - Mean: {mean_intensity:.2f}, Std: {std_intensity:.2f}")
            print(f"  Dark threshold will be: {mean_intensity * self.dark_factor:.2f}")
            print(f"  Bright threshold will be: {mean_intensity * self.bright_factor:.2f}")
            print(f"  Min pixel value in image: {np.min(gray)}")
            print(f"  Max pixel value in image: {np.max(gray)}")
        
        # Step 2: Threshold for dark and bright anomalies
        dark_mask, bright_mask = self._threshold_anomalies(gray, mean_intensity)
        
        if self.debug:
            self.debug_images['02_dark_mask_raw'] = dark_mask.copy()
            self.debug_images['03_bright_mask_raw'] = bright_mask.copy()
            print(f"Dark pixels: {np.sum(dark_mask > 0)}, Bright pixels: {np.sum(bright_mask > 0)}")
        
        # Step 3: Clean masks with morphological operations
        print(f"Cleaning dark mask...")
        dark_mask_clean = self._clean_mask(dark_mask)
        print(f"Cleaning bright mask...")
        bright_mask_clean = self._clean_mask(bright_mask)
        
        if self.debug:
            self.debug_images['04_dark_mask_clean'] = dark_mask_clean.copy()
            self.debug_images['05_bright_mask_clean'] = bright_mask_clean.copy()
            print(f"After cleaning - Dark pixels: {np.sum(dark_mask_clean > 0)}, Bright pixels: {np.sum(bright_mask_clean > 0)}")
        
        # Step 4: Extract blobs (connected components)
        print(f"\nExtracting debris blobs...")
        debris_defects = self._extract_blobs(dark_mask_clean, gray, mean_intensity, defect_type='debris')
        print(f"\nExtracting void blobs...")
        void_defects = self._extract_blobs(bright_mask_clean, gray, mean_intensity, defect_type='void')
        
        # Step 5: Filter and combine results
        all_defects = debris_defects + void_defects
        
        # Draw detections on visualization
        visualization = self._draw_detections(visualization, debris_defects, void_defects)
        
        # Draw exclusion zones if any
        if self.exclusion_zones:
            visualization = self.draw_exclusion_zones(visualization)
        
        # Save visualization to debug images as well
        if self.debug:
            self.debug_images['06_final_visualization'] = visualization.copy()
            
            # Create a combined debug visualization showing masks overlaid on original
            debug_overlay = gray.copy()
            debug_overlay = cv2.cvtColor(debug_overlay, cv2.COLOR_GRAY2BGR)
            
            # Overlay dark mask in red (debris)
            red_overlay = np.zeros_like(debug_overlay)
            red_overlay[:, :, 2] = dark_mask_clean  # Red channel
            debug_overlay = cv2.addWeighted(debug_overlay, 0.7, red_overlay, 0.3, 0)
            
            # Overlay bright mask in cyan (voids)
            cyan_overlay = np.zeros_like(debug_overlay)
            cyan_overlay[:, :, 0] = bright_mask_clean  # Blue channel
            cyan_overlay[:, :, 1] = bright_mask_clean  # Green channel
            debug_overlay = cv2.addWeighted(debug_overlay, 0.7, cyan_overlay, 0.3, 0)
            
            self.debug_images['07_masks_overlay'] = debug_overlay
        
        if self.debug:
            print(f"Total defects found: {len(all_defects)} (Debris: {len(debris_defects)}, Voids: {len(void_defects)})")
        
        return visualization, all_defects
    
    def _compute_baseline(self, gray: np.ndarray) -> Tuple[float, float]:
        """Compute adaptive baseline statistics for the stripe.
        
        Args:
            gray: Grayscale image of the stripe.
        
        Returns:
            Tuple of (mean_intensity, std_intensity).
        """
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        return mean_intensity, std_intensity
    
    def _threshold_anomalies(self, gray: np.ndarray, mean_intensity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create binary masks for dark and bright anomalies.
        
        Args:
            gray: Grayscale image.
            mean_intensity: Computed mean intensity of the stripe.
        
        Returns:
            Tuple of (dark_mask, bright_mask) as binary images.
        """
        # Dark anomalies (debris): pixels darker than mean × DARK_FACTOR
        dark_threshold = mean_intensity * self.dark_factor
        dark_mask = (gray < dark_threshold).astype(np.uint8) * 255
        
        # Bright anomalies (voids): pixels brighter than mean × BRIGHT_FACTOR
        bright_threshold = mean_intensity * self.bright_factor
        bright_mask = (gray > bright_threshold).astype(np.uint8) * 255
        
        if self.debug:
            print(f"Thresholding complete:")
            print(f"  Dark threshold: {dark_threshold:.2f} -> {np.sum(dark_mask > 0)} pixels")
            print(f"  Bright threshold: {bright_threshold:.2f} -> {np.sum(bright_mask > 0)} pixels")
        
        return dark_mask, bright_mask
    
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Clean a binary mask using morphological operations.
        
        Removes speckle noise and fills small holes.
        
        Args:
            mask: Binary mask (0 or 255).
        
        Returns:
            Cleaned binary mask.
        """
        pixels_before = np.sum(mask > 0)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (self.morph_kernel_size, self.morph_kernel_size))
        
        # Morphological opening: removes small noise
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self.morph_iterations)
        
        # Morphological closing: fills small holes
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=self.morph_iterations)
        
        pixels_after = np.sum(closed > 0)
        
        if self.debug:
            print(f"  Morphology: {pixels_before} pixels -> {pixels_after} pixels (removed {pixels_before - pixels_after})")
        
        return closed
    
    def _extract_blobs(self, mask: np.ndarray, gray: np.ndarray, mean_intensity: float, 
                      defect_type: str) -> List[Dict[str, Any]]:
        """Extract connected components from a binary mask.
        
        Args:
            mask: Cleaned binary mask.
            gray: Original grayscale image (for computing intensity deviation).
            mean_intensity: Baseline mean intensity.
            defect_type: Either 'debris' or 'void'.
        
        Returns:
            List of defect dictionaries.
        """
        # Find contours (connected components)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if self.debug:
            print(f"  Found {len(contours)} {defect_type} contours before filtering")
        
        defects = []
        filtered_by_size = 0
        filtered_by_exclusion = 0
        
        for contour in contours:
            # Compute blob properties
            area = cv2.contourArea(contour)
            
            # Filter by minimum area
            if area < self.min_blob_area:
                filtered_by_size += 1
                continue
            
            # Compute centroid
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if blob is in exclusion zone
            in_exclusion, zone = self.is_region_in_exclusion_zone(x, y, w, h)
            if in_exclusion:
                filtered_by_exclusion += 1
                if self.debug:
                    print(f"  {defect_type.capitalize()} at ({cx}, {cy}) excluded by zone '{zone['name']}'")
                continue
            
            # Compute intensity deviation from mean
            mask_roi = np.zeros_like(gray)
            cv2.drawContours(mask_roi, [contour], -1, 255, -1)
            blob_pixels = gray[mask_roi > 0]
            
            if len(blob_pixels) > 0:
                blob_mean = np.mean(blob_pixels)
                intensity_deviation = blob_mean - mean_intensity
            else:
                intensity_deviation = 0.0
            
            # Create defect record
            defect = {
                'type': defect_type,
                'centroid': (cx, cy),
                'area': float(area),
                'bbox': (x, y, w, h),
                'intensity_deviation': float(intensity_deviation),
                'severity': self._compute_severity(area, intensity_deviation, defect_type)
            }
            
            defects.append(defect)
        
        if self.debug:
            print(f"  {defect_type.capitalize()} extraction complete:")
            print(f"    - Filtered by size (<{self.min_blob_area}px): {filtered_by_size}")
            print(f"    - Filtered by exclusion zones: {filtered_by_exclusion}")
            print(f"    - Final {defect_type} count: {len(defects)}")
            if len(defects) > 0:
                areas = [d['area'] for d in defects]
                print(f"    - Area range: {min(areas):.1f} to {max(areas):.1f} pixels")
        
        return defects
    
    def _compute_severity(self, area: float, intensity_deviation: float, defect_type: str) -> str:
        """Classify defect severity based on area and intensity deviation.
        
        Args:
            area: Defect area in pixels.
            intensity_deviation: Deviation from mean intensity.
            defect_type: 'debris' or 'void'.
        
        Returns:
            Severity level: 'low', 'medium', or 'high'.
        """
        # Larger area = more severe
        # Greater intensity deviation = more severe
        
        if defect_type == 'debris':
            # For debris, negative deviation (darker) is more severe
            deviation_score = abs(intensity_deviation)
        else:  # void
            # For voids, positive deviation (brighter) is more severe
            deviation_score = abs(intensity_deviation)
        
        # Combined score (normalized)
        # Area thresholds: small < 200, medium < 500, large >= 500
        # Deviation thresholds depend on typical ranges (0-50 intensity units)
        
        area_score = 0
        if area >= 500:
            area_score = 3
        elif area >= 200:
            area_score = 2
        else:
            area_score = 1
        
        deviation_score_norm = 0
        if deviation_score >= 50:
            deviation_score_norm = 3
        elif deviation_score >= 20:
            deviation_score_norm = 2
        else:
            deviation_score_norm = 1
        
        combined_score = area_score + deviation_score_norm
        
        if combined_score >= 5:
            return 'high'
        elif combined_score >= 3:
            return 'medium'
        else:
            return 'low'
    
    def _draw_detections(self, visualization: np.ndarray, 
                        debris_defects: List[Dict], void_defects: List[Dict]) -> np.ndarray:
        """Draw detected debris and voids on the visualization image.
        
        Args:
            visualization: BGR image to draw on.
            debris_defects: List of debris defect dictionaries.
            void_defects: List of void defect dictionaries.
        
        Returns:
            Visualization image with detections marked.
        """
        # Draw debris in red
        for defect in debris_defects:
            x, y, w, h = defect['bbox']
            cx, cy = defect['centroid']
            severity = defect['severity']
            
            # Color intensity based on severity
            if severity == 'high':
                color = (0, 0, 255)  # Bright red
                thickness = 3
            elif severity == 'medium':
                color = (0, 64, 255)  # Orange-red
                thickness = 2
            else:
                color = (0, 128, 255)  # Light red
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(visualization, (x, y), (x + w, y + h), color, thickness)
            
            # Draw centroid
            cv2.circle(visualization, (cx, cy), 5, color, -1)
            
            # Add label
            label = f"Debris ({severity})"
            cv2.putText(visualization, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw voids in cyan
        for defect in void_defects:
            x, y, w, h = defect['bbox']
            cx, cy = defect['centroid']
            severity = defect['severity']
            
            # Color intensity based on severity
            if severity == 'high':
                color = (255, 255, 0)  # Bright cyan
                thickness = 3
            elif severity == 'medium':
                color = (255, 200, 0)  # Light cyan
                thickness = 2
            else:
                color = (255, 150, 0)  # Pale cyan
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(visualization, (x, y), (x + w, y + h), color, thickness)
            
            # Draw centroid
            cv2.circle(visualization, (cx, cy), 5, color, -1)
            
            # Add label
            label = f"Void ({severity})"
            cv2.putText(visualization, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add summary text
        summary = f"Debris: {len(debris_defects)} | Voids: {len(void_defects)}"
        cv2.putText(visualization, summary, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(visualization, summary, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        
        return visualization
    
    def save_debug_images(self, output_dir: str, base_name: str):
        """Save debug images to the output directory.
        
        Args:
            output_dir: Directory to save debug images.
            base_name: Base filename for the debug images.
        """
        if not self.debug_images:
            if self.debug:
                print("  No debug images to save")
            return
        
        if self.debug:
            print(f"  Saving {len(self.debug_images)} debug images...")
        
        for name, img in self.debug_images.items():
            debug_path = os.path.join(output_dir, f"{base_name}_debris_void_{name}.jpg")
            
            # Handle both grayscale and color images
            if len(img.shape) == 2:
                # Grayscale - save directly
                cv2.imwrite(debug_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                # Color - save directly
                cv2.imwrite(debug_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if self.debug:
                print(f"    ✓ {name}: {debug_path}")


# For standalone testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python stripe_debris_void_detection.py <image_path> [sensitivity]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    sensitivity = sys.argv[2] if len(sys.argv) > 2 else 'medium'
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        sys.exit(1)
    
    # Create detector
    detector = StripeDebrisVoidDetector(sensitivity=sensitivity, debug=True)
    
    # Run detection
    visualization, defects = detector.detect(image, image_path)
    
    # Save result
    output_path = image_path.replace('.tif', '_debris_void_result.jpg').replace('.tiff', '_debris_void_result.jpg')
    cv2.imwrite(output_path, visualization)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Total defects: {len(defects)}")
    
    # Print defect details
    for i, defect in enumerate(defects, 1):
        print(f"\nDefect {i}:")
        print(f"  Type: {defect['type']}")
        print(f"  Centroid: {defect['centroid']}")
        print(f"  Area: {defect['area']:.1f} pixels")
        print(f"  Severity: {defect['severity']}")

