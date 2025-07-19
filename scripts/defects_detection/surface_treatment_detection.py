"""
Surface Treatment Defect Detection Algorithm
Detects bad surface energy causing ink coalescence and missing ink areas

Author: Koushik and Assistant
Date: 2024
Version: 3.0 - Corrected for vertical head segmentation
"""

import cv2
import numpy as np
from scipy import ndimage, signal
from skimage import morphology, measure, filters, feature
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Optional
import os


class SurfaceTreatmentDetector:
    """
    Detects surface treatment defects characterized by:
    - Bands of light areas or pores throughout an entire print head
    - Affects one of 4 vertically stacked print heads
    - Must be uniform across the entire head region, not isolated spots
    """
    
    def __init__(self, 
                 uniformity_threshold: float = 0.7,
                 porosity_threshold: float = 0.15,
                 brightness_variation_threshold: float = 0.3,
                 min_head_coverage: float = 0.4):
        """
        Args:
            uniformity_threshold: Threshold for uniformity across head (0-1)
            porosity_threshold: Threshold for detecting porous regions (0-1)
            brightness_variation_threshold: Threshold for brightness variations
            min_head_coverage: Minimum coverage of defects to consider head defective
        """
        self.uniformity_threshold = uniformity_threshold
        self.porosity_threshold = porosity_threshold
        self.brightness_variation_threshold = brightness_variation_threshold
        self.min_head_coverage = min_head_coverage
    
    def detect(self, image: np.ndarray, image_path: str = "unknown") -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Main detection method for vertical head analysis
        
        Args:
            image: Input image (BGR format expected for colored stripes)
            
        Returns:
            Tuple of (visualization, defects list)
        """
        print("  🔍 Surface Treatment: Analyzing vertical head segments...")
        
        # Step 1: Segment the colored region
        colored_mask, colored_bbox = self._segment_colored_region(image)
        
        if colored_bbox is None:
            print("  ⚠️  No colored region found")
            return image, []
        
        # Step 2: Extract colored region and split into 4 vertical heads
        head_regions = self._split_into_vertical_heads(image, colored_bbox, image_path=image_path)
        
        if not head_regions:
            print("  ⚠️  No valid head regions found")
            return image, []
            
        # Step 3: Analyze each head for surface treatment errors
        head_defects = []
        for i, (head_image, head_bbox) in enumerate(head_regions):
            print("\n" + "="*50)
            print(f"📐 Head {i+1}/4 Analysis")
            print(f"Size: {head_image.shape[1]}x{head_image.shape[0]} pixels")
            print("="*50)
            
            defect_info = self._analyze_head_uniformity(head_image, head_bbox, i)
            if defect_info:
                head_defects.append(defect_info)
                print("❌ Result: Surface Treatment Error")
                print(f"   Coverage: {defect_info['coverage']:.1%}")
            else:
                print("✅ Result: Normal Surface Treatment")
        
        # Step 4: Create visualization and defects list
        defects = self._create_defect_list(head_defects)
        visualization = self._create_visualization(image, head_regions, head_defects)
        
        return visualization, defects
    
    def _segment_colored_region(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Segment the colored printed region from the background
        
        Returns:
            Tuple of (colored_mask, bounding_box)
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for non-white regions (printed areas)
        # White background typically has high V (value) and low S (saturation)
        lower_bound = np.array([0, 30, 0])      # Low saturation, any hue, any value
        upper_bound = np.array([180, 255, 240]) # High saturation, any hue, not too bright
        
        colored_mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        colored_mask = cv2.morphologyEx(colored_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        colored_mask = cv2.morphologyEx(colored_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find the largest connected component (main colored region)
        contours, _ = cv2.findContours(colored_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return colored_mask, (0, 0, 0, 0)
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Create clean mask with only the largest region
        clean_mask = np.zeros_like(colored_mask)
        cv2.fillPoly(clean_mask, [largest_contour], (255,))
        
        return clean_mask, (x, y, w, h)
    
    def _split_into_vertical_heads(self, image: np.ndarray, 
                                 colored_bbox: Tuple[int, int, int, int],
                                 image_path: str = "unknown") -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Split the entire image into 4 equal vertical segments (print heads)
        
        Args:
            image: Original image
            colored_bbox: Bounding box of colored region (x, y, w, h) - for reference
            
        Returns:
            List of (head_image, head_bbox) tuples
        """
        x, y, w, h = colored_bbox  # Keep for reference but split entire image
        
        # ===== TEMPORARY VISUALIZATION CODE - START =====
        # Create visualization image for debugging head splits
        vis_image = image.copy()
        head_colors = [
            (255, 0, 0),   # Blue
            (0, 255, 0),   # Green
            (0, 0, 255),   # Red
            (255, 255, 0)  # Cyan
        ]
        # ============================================
        
        # Split ENTIRE IMAGE into 4 equal vertical segments for visualization
        img_height, img_width = image.shape[:2]
        segment_height = img_height // 4
        head_regions = []
        
        for i in range(4):
            # Calculate head boundaries for ENTIRE IMAGE
            head_y_start = i * segment_height
            head_y_end = (i + 1) * segment_height if i < 3 else img_height  # Last head gets remainder
            
            # Extract head segment from ENTIRE IMAGE (full width)
            head_image = image[head_y_start:head_y_end, :].copy()
            
            # Debug print head dimensions
            print(f"[DEBUG] Head {i+1} dimensions: {head_image.shape}")
            print(f"[DEBUG] Head {i+1} bounds: y={head_y_start} to {head_y_end}")
            
            if head_image.size == 0:
                print(f"[WARNING] Empty head region detected for head {i+1}")
                continue
                
            # Calculate absolute coordinates (entire image width)
            abs_head_bbox = (0, head_y_start, img_width, head_y_end - head_y_start)
            
            # ===== TEMPORARY VISUALIZATION CODE - CONTINUED =====
            # Create colored overlay for the ENTIRE IMAGE segment (not just colored region)
            overlay = np.zeros_like(vis_image)
            cv2.rectangle(overlay, (0, head_y_start), (img_width, head_y_end), head_colors[i], -1)
            cv2.addWeighted(overlay, 0.3, vis_image, 1.0, 0, vis_image)
            
            # Add head number label in the center of the segment
            text_x = img_width // 2
            text_y = head_y_start + (head_y_end - head_y_start) // 2
            cv2.putText(vis_image, f"Head {i+1}", (text_x - 50, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
            # ================================================
            
            head_regions.append((head_image, abs_head_bbox))
        
        # ===== TEMPORARY VISUALIZATION CODE - END =====
        # Get input image name without extension
        input_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Get the directory where the input image is located
        input_dir = os.path.dirname(image_path)
        
        # Create visualizations directory next to the input image
        output_dir = os.path.join(input_dir, "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save visualization with input image name
        output_path = os.path.join(output_dir, f"{input_name}_head_segmentation.jpg")
        cv2.imwrite(output_path, vis_image)
        print(f"\n[DEBUG] Head segmentation saved: {os.path.basename(output_path)}")
        # ================================================
        
        return head_regions
    
    def _analyze_head_uniformity(self, head_image: np.ndarray, 
                               head_bbox: Tuple[int, int, int, int], 
                               head_index: int) -> Optional[Dict[str, Any]]:
        """
        Analyze a single head for uniform surface treatment errors
        
        Args:
            head_image: Image of the head region
            head_bbox: Bounding box coordinates
            head_index: Index of the head (0-3)
            
        Returns:
            Defect info dictionary if defect found, None otherwise
        """
        # Convert to LAB color space for better lightness analysis
        lab = cv2.cvtColor(head_image, cv2.COLOR_BGR2LAB)
        lightness = lab[:, :, 0]
        
        # Step 1: Detect porous/light regions
        porous_regions = self._detect_porous_regions(lightness)
        
        # Step 2: Analyze uniformity across the head
        h, w = head_image.shape[:2]
        shape_2d = (h, w)  # Explicit 2-tuple
        uniformity_score = self._calculate_uniformity_score(porous_regions, shape_2d)
        
        # Step 3: Check if defects are distributed across the head (not just isolated spots)
        coverage_score = self._calculate_coverage_score(porous_regions, shape_2d)
        
        # Step 4: Analyze brightness variations (bands of light areas)
        brightness_bands = self._detect_brightness_bands(lightness)
        band_score = np.sum(brightness_bands > 0) / brightness_bands.size
        
        # Decision logic: Surface treatment error if:
        # 1. High coverage of porous regions across the head, OR
        # 2. Significant brightness bands throughout the head
        is_surface_treatment_error = (
            coverage_score > self.min_head_coverage or 
            band_score > self.brightness_variation_threshold
        )
        print("\nMetrics:")
        print(f"├── Coverage Score: {coverage_score:.3f} (threshold: {self.min_head_coverage:.3f})")
        print(f"├── Band Score: {band_score:.3f} (threshold: {self.brightness_variation_threshold:.3f})")
        print(f"└── Uniformity Score: {uniformity_score:.3f}")
        print(f"\nFinal Decision: {'❌ DEFECT' if is_surface_treatment_error else '✅ NORMAL'}")

        if is_surface_treatment_error:
            return {
                'head_index': head_index,
                'head_bbox': head_bbox,
                'coverage': coverage_score,
                'uniformity': uniformity_score,
                'brightness_bands': band_score,
                'severity': 'high' if coverage_score > 0.6 else 'medium',
                'defect_mask': porous_regions | brightness_bands
            }
        
        return None
    
    def _detect_porous_regions(self, lightness: np.ndarray) -> np.ndarray:
        """Detect porous regions (areas with missing ink)"""
        # Find regions that are significantly brighter than the mean
        mean_lightness = np.mean(lightness)
        std_lightness = np.std(lightness)
        
        # Threshold for bright spots (pores)
        pore_threshold = mean_lightness + 1.5 * std_lightness
        
        # Create binary mask of porous regions
        porous_mask = lightness > pore_threshold
        
        # Clean up small isolated spots (these might be debris, not surface treatment)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        porous_mask = cv2.morphologyEx(porous_mask.astype(np.uint8) * 255, 
                                     cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Connect nearby porous regions
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        porous_mask = cv2.morphologyEx(porous_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        
        return np.greater(porous_mask.astype(np.uint8), 0)
    
    def _detect_brightness_bands(self, lightness: np.ndarray) -> np.ndarray:
        """Detect horizontal bands of varying brightness"""
        h, w = lightness.shape
        
        # Calculate horizontal intensity profiles
        horizontal_profile = np.mean(lightness, axis=1)
        
        # Smooth the profile
        smoothed_profile = signal.savgol_filter(horizontal_profile, 
                                              min(21, h//10*2+1), 3)
        
        # Find significant deviations from the mean
        mean_intensity = float(np.mean(smoothed_profile.astype(np.float64)))
        std_intensity = float(np.std(smoothed_profile.astype(np.float64)))
        
        # Create mask for bands with significant brightness variations
        band_mask = np.zeros_like(lightness, dtype=bool)
        
        for y in range(h):
            if abs(smoothed_profile[y] - mean_intensity) > std_intensity:
                band_mask[y, :] = True
        
        # Clean up the band mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        band_mask = cv2.morphologyEx(band_mask.astype(np.uint8) * 255, 
                                   cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return np.greater(band_mask.astype(np.uint8), 0)
    
    def _calculate_uniformity_score(self, defect_mask: np.ndarray, shape: Tuple[int, int]) -> float:
        """Calculate how uniformly defects are distributed across the head"""
        h, w = shape
        
        # Divide head into grid and check defect presence in each cell
        grid_size = 8
        cell_h = h // grid_size
        cell_w = w // grid_size
        
        cells_with_defects = 0
        total_cells = 0
        
        for i in range(grid_size):
            for j in range(grid_size):
                y_start = i * cell_h
                y_end = min((i + 1) * cell_h, h)
                x_start = j * cell_w
                x_end = min((j + 1) * cell_w, w)
                
                cell = defect_mask[y_start:y_end, x_start:x_end]
                if cell.size > 0:
                    total_cells += 1
                    if np.any(cell):
                        cells_with_defects += 1
        
        return cells_with_defects / total_cells if total_cells > 0 else 0
    
    def _calculate_coverage_score(self, defect_mask: np.ndarray, shape: Tuple[int, int]) -> float:
        """Calculate the percentage of head area covered by defects"""
        total_pixels = shape[0] * shape[1]
        defect_pixels = np.sum(defect_mask)
        return defect_pixels / total_pixels
    
    def _create_defect_list(self, head_defects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create detailed defect list"""
        defects = []
        
        for head_defect in head_defects:
            x, y, w, h = head_defect['head_bbox']
            
            defects.append({
                'type': 'surface_treatment',
                'subtype': 'head_defect',
                'head_index': head_defect['head_index'],
                'location': (x + w//2, y + h//2),  # Center of head
                'bbox': head_defect['head_bbox'],
                'coverage': float(head_defect['coverage']),
                'uniformity': float(head_defect['uniformity']),
                'brightness_bands': float(head_defect['brightness_bands']),
                'severity': head_defect['severity']
            })
        
        return defects
    
    def _create_visualization(self, original: np.ndarray, 
                            head_regions: List[Tuple[np.ndarray, Tuple[int, int, int, int]]],
                            head_defects: List[Dict[str, Any]]) -> np.ndarray:
        """Create visualization highlighting surface treatment defects"""
        vis = original.copy()
        
        # Draw head boundaries
        for i, (_, head_bbox) in enumerate(head_regions):
            x, y, w, h = head_bbox
            # Draw head boundary in gray
            cv2.rectangle(vis, (x, y), (x + w, y + h), (128, 128, 128), 2)
            # Add head number
            cv2.putText(vis, f"Head {i+1}", (x + 5, y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        
        # Highlight defective heads
        for head_defect in head_defects:
            x, y, w, h = head_defect['head_bbox']
            
            # Create overlay for defective head
            overlay = vis.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), -1)  # Green overlay
            
            # Blend overlay
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
            
            # Draw thick red border for defective head
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 4)
            
            # Add defect information
            severity_color = (0, 0, 255) if head_defect['severity'] == 'high' else (0, 165, 255)
            text = f"DEFECT: {head_defect['coverage']:.1%}"
            cv2.putText(vis, text, (x + 5, y + h - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, severity_color, 2)
        
        return vis 