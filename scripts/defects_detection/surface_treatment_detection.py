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
    - Lighter bands, pores, or reduced color density across entire print head regions
    - Affects one of 4 vertically stacked print heads
    - Distinguishes from localized defects (which are smudges/debris)
    """
    
    def __init__(self, 
                 density_threshold: float = 0.25,  # Threshold for color density difference
                 band_detection_sensitivity: float = 0.2,  # Sensitivity for detecting bands
                 head_comparison_threshold: float = 0.1,  # Threshold for comparing heads
                 min_defect_area_ratio: float = 0.4):  # Minimum area ratio to consider head defective
        """
        Args:
            density_threshold: Threshold for detecting low density regions
            band_detection_sensitivity: Sensitivity for band detection
            head_comparison_threshold: Threshold for comparing heads against each other
            min_defect_area_ratio: Minimum ratio of defective area in head
        """
        self.density_threshold = density_threshold
        self.band_detection_sensitivity = band_detection_sensitivity
        self.head_comparison_threshold = head_comparison_threshold
        self.min_defect_area_ratio = min_defect_area_ratio
    
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
                print(f"   Average Density: {defect_info['average_density']:.3f}")
                print(f"   Density Variation: {defect_info['density_std']:.3f}")
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
        SIMPLIFIED: Analyze head for surface treatment defects
        Good heads = solid, uniform density
        Bad heads = lighter, less dense, irregular
        """
        # Step 1: Get only the colored region (ignore white background)
        colored_mask = self._get_simple_colored_mask(head_image)
        colored_pixels = colored_mask > 0
        
        if np.sum(colored_pixels) < 1000:  # Need sufficient colored area
            print("   ⚠️  Insufficient colored region")
            return None
        
        # Step 2: Calculate simple density (darkness × saturation)
        density_map = self._calculate_simple_density(head_image)
        colored_density_values = density_map[colored_pixels]
        
        # Step 3: Calculate key metrics
        avg_density = np.mean(colored_density_values)
        density_std = np.std(colored_density_values)
        low_density_ratio = np.sum(colored_density_values < 0.3) / len(colored_density_values)
        
        print(f"\n📊 Head {head_index + 1} Metrics:")
        print(f"├── Average Density: {avg_density:.3f}")
        print(f"├── Density Std: {density_std:.3f}")
        print(f"└── Low Density Ratio: {low_density_ratio:.3f}")
        
        # Step 4: Simple decision logic
        # Bad if: Low average density OR high variation OR too many low-density pixels
        is_defective = (
            avg_density < 0.35 or           # Overall too light
            density_std > 0.15 or           # Too much variation (irregular)
            low_density_ratio > 0.4         # Too many light pixels
        )
        
        print(f"Decision: {'❌ DEFECT' if is_defective else '✅ NORMAL'}")
        print(f"  Reasons: {'Low avg' if avg_density < 0.35 else ''} {'High var' if density_std > 0.15 else ''} {'Many light' if low_density_ratio > 0.4 else ''}")

        if is_defective:
            return {
                'head_index': head_index,
                'head_bbox': head_bbox,
                'average_density': float(avg_density),
                'density_std': float(density_std),
                'low_density_ratio': float(low_density_ratio),
                'severity': 'high' if avg_density < 0.25 else 'medium',
                'defect_mask': colored_density_values < 0.3
            }
        
        return None
    
    def _get_simple_colored_mask(self, head_image: np.ndarray) -> np.ndarray:
        """Simple method to get colored (non-white) pixels"""
        # Convert to HSV 
        hsv = cv2.cvtColor(head_image, cv2.COLOR_BGR2HSV)
        
        # Simple threshold: any pixel with reasonable saturation
        colored_mask = np.greater(hsv[:, :, 1], 30)  # Saturation > 30
        
        return colored_mask.astype(np.uint8) * 255
    
    def _calculate_simple_density(self, head_image: np.ndarray) -> np.ndarray:
        """Simple density calculation: darkness × saturation"""
        # Convert to HSV
        hsv = cv2.cvtColor(head_image, cv2.COLOR_BGR2HSV)
        
        # Darkness = 1 - (Value/255)
        darkness = 1.0 - (hsv[:, :, 2].astype(np.float32) / 255.0)
        
        # Saturation normalized
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0
        
        # Simple density = darkness × saturation
        density = darkness * saturation
        
        return density
    
    def _detect_low_density_regions(self, density_map: np.ndarray) -> np.ndarray:
        """Detect regions with significantly lower color density"""
        # Calculate statistics
        mean_density = np.mean(density_map)
        std_density = np.std(density_map)
        
        # Threshold for low density (surface treatment problems)
        low_density_threshold = mean_density - (std_density * 1.5)
        
        # Create binary mask
        low_density_mask = density_map < low_density_threshold
        
        # Clean up small isolated regions (these might be noise)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_mask = cv2.morphologyEx(low_density_mask.astype(np.uint8) * 255, 
                                       cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Connect nearby low-density regions
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        connected_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        
        return np.greater(connected_mask, 0)
    
    def _detect_horizontal_bands(self, density_map: np.ndarray) -> np.ndarray:
        """Detect horizontal bands of varying density (common in surface treatment defects)"""
        h, w = density_map.shape
        
        # Calculate horizontal density profiles
        horizontal_profile = np.mean(density_map, axis=1)
        
        # Smooth the profile to reduce noise
        if h > 10:
            window_length = min(21, h // 5 * 2 + 1)  # Ensure odd number
            if window_length >= 3:
                smoothed_profile = signal.savgol_filter(horizontal_profile, window_length, 2)
            else:
                smoothed_profile = horizontal_profile
        else:
            smoothed_profile = horizontal_profile
        
        # Find significant deviations from the median (more robust than mean)
        median_density = float(np.median(smoothed_profile.astype(np.float64)))
        mad = float(np.median(np.abs(smoothed_profile.astype(np.float64) - median_density)))
        
        # Create mask for bands with significant density variations
        band_mask = np.zeros_like(density_map, dtype=bool)
        
        if mad > 0:
            threshold = self.band_detection_sensitivity * mad
            for y in range(h):
                if abs(smoothed_profile[y] - median_density) > threshold:
                    band_mask[y, :] = True
        
        # Clean up the band mask
        if np.any(band_mask):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
            band_mask_uint8 = cv2.morphologyEx(band_mask.astype(np.uint8) * 255, 
                                              cv2.MORPH_CLOSE, kernel, iterations=2)
            band_mask = np.greater(band_mask_uint8, 0)
        
        return band_mask
    
    def _calculate_defect_distribution(self, defect_mask: np.ndarray) -> float:
        """Calculate how widely distributed defects are across the head region"""
        h, w = defect_mask.shape
        
        # Divide into a grid and check how many grid cells contain defects
        grid_size = 6  # 6x6 grid for good coverage
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
        
        return cells_with_defects / total_cells if total_cells > 0 else 0.0
    
    def _extract_colored_region_from_head(self, head_image: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        """Extract only the colored region from the head image to avoid analyzing white background"""
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(head_image, cv2.COLOR_BGR2HSV)
        
        # Create mask for non-white regions (printed areas)
        # More conservative thresholds to focus on actual printed areas
        lower_bound = np.array([0, 40, 0])      # Higher saturation threshold
        upper_bound = np.array([180, 255, 220]) # Avoid very bright areas
        
        colored_mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        colored_mask = cv2.morphologyEx(colored_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        colored_mask = cv2.morphologyEx(colored_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find the largest connected component
        contours, _ = cv2.findContours(colored_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros_like(colored_mask), None
        
        # Keep only significant contours
        significant_contours = [c for c in contours if cv2.contourArea(c) > 500]
        
        if not significant_contours:
            return np.zeros_like(colored_mask), None
        
        # Create final mask with all significant colored regions
        final_mask = np.zeros_like(colored_mask)
        for contour in significant_contours:
            cv2.fillPoly(final_mask, [contour], (255,))
        
        # Get bounding box of colored region
        x, y, w, h = cv2.boundingRect(np.vstack(significant_contours))
        
        return final_mask, (x, y, w, h)
    
    def _detect_horizontal_bands_conservative(self, density_map: np.ndarray, colored_mask: np.ndarray) -> np.ndarray:
        """Conservative horizontal band detection focused on colored regions"""
        h, w = density_map.shape
        
        # Calculate horizontal density profiles only for colored pixels
        band_mask = np.zeros_like(density_map, dtype=bool)
        
        for y in range(h):
            row_colored = colored_mask[y, :] > 0
            if np.sum(row_colored) < w * 0.1:  # Skip rows with too few colored pixels
                continue
                
            row_density = density_map[y, row_colored]
            if len(row_density) == 0:
                continue
                
            row_mean = np.mean(row_density)
            
            # Compare with neighboring rows
            neighbor_densities = []
            for dy in [-2, -1, 1, 2]:  # Check 2 rows above and below
                ny = y + dy
                if 0 <= ny < h:
                    neighbor_colored = colored_mask[ny, :] > 0
                    if np.sum(neighbor_colored) > 0:
                        neighbor_density = density_map[ny, neighbor_colored]
                        neighbor_densities.extend(neighbor_density)
            
            if len(neighbor_densities) > 0:
                neighbor_mean = np.mean(neighbor_densities)
                
                # Only flag as band if significantly different from neighbors
                if abs(row_mean - neighbor_mean) > 0.1:  # Conservative threshold
                    band_mask[y, row_colored] = True
        
        return band_mask
    
    def _calculate_defect_distribution_in_colored_region(self, defect_mask: np.ndarray, colored_mask: np.ndarray) -> float:
        """Calculate distribution of defects within the colored region only"""
        # Get bounding box of colored region
        colored_coords = np.where(colored_mask > 0)
        if len(colored_coords[0]) == 0:
            return 0.0
            
        min_y, max_y = np.min(colored_coords[0]), np.max(colored_coords[0])
        min_x, max_x = np.min(colored_coords[1]), np.max(colored_coords[1])
        
        colored_height = max_y - min_y + 1
        colored_width = max_x - min_x + 1
        
        # Divide colored region into grid
        grid_size = 4  # Smaller grid for more focused analysis
        cell_h = colored_height // grid_size
        cell_w = colored_width // grid_size
        
        cells_with_defects = 0
        total_cells = 0
        
        for i in range(grid_size):
            for j in range(grid_size):
                y_start = min_y + i * cell_h
                y_end = min(min_y + (i + 1) * cell_h, max_y + 1)
                x_start = min_x + j * cell_w
                x_end = min(min_x + (j + 1) * cell_w, max_x + 1)
                
                # Check if this cell has colored pixels
                cell_colored = colored_mask[y_start:y_end, x_start:x_end]
                if np.sum(cell_colored) < cell_colored.size * 0.3:  # Skip cells with too few colored pixels
                    continue
                    
                total_cells += 1
                
                # Check if this cell has defects
                cell_defects = defect_mask[y_start:y_end, x_start:x_end]
                if np.any(cell_defects):
                    cells_with_defects += 1
        
        return cells_with_defects / total_cells if total_cells > 0 else 0.0
    
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
                'average_density': float(head_defect['average_density']),
                'density_std': float(head_defect['density_std']),
                'low_density_ratio': float(head_defect['low_density_ratio']),
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
            text = f"DEFECT: {head_defect['average_density']:.2f}"
            cv2.putText(vis, text, (x + 5, y + h - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, severity_color, 2)
        
        return vis 