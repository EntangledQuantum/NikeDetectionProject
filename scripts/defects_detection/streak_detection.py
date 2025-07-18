"""
Streak Detection Algorithm
Detects linear streaks and lines in printed images

Author: Assistant
Date: 2024
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from skimage import morphology, measure, transform
import os
from tqdm import tqdm


class StreakDetector:
    """Detects streak defects - continuous linear marks in prints"""
    
    def __init__(self, min_streak_length=50, max_streak_width=10,
                 angle_tolerance=5, contrast_threshold=20):
        """
        Args:
            min_streak_length: Minimum length for streak detection
            max_streak_width: Maximum width of streaks
            angle_tolerance: Tolerance for line angle grouping
            contrast_threshold: Minimum contrast for streak detection
        """
        self.min_streak_length = min_streak_length
        self.max_streak_width = max_streak_width
        self.angle_tolerance = angle_tolerance
        self.contrast_threshold = contrast_threshold
        
    def detect(self, image):
        """
        Detect streaks in the image
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            result_image: Visualization of detected streaks
            defects: List of detected streak defects
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply multiple detection methods
        hough_streaks = self._detect_using_hough(gray)
        morphology_streaks = self._detect_using_morphology(gray)
        gradient_streaks = self._detect_using_gradients(gray)
        
        # Merge and filter results
        all_streaks = hough_streaks + morphology_streaks + gradient_streaks
        filtered_streaks = self._filter_and_merge_streaks(all_streaks)
        
        # Create defect list
        defects = []
        for streak in filtered_streaks:
            defects.append({
                'type': 'streak',
                'start_point': streak['start'],
                'end_point': streak['end'],
                'angle': streak['angle'],
                'length': streak['length'],
                'width': streak.get('width', 1),
                'contrast': streak.get('contrast', 0),
                'detection_method': streak.get('method', 'unknown')
            })
            
        # Create visualization
        result_image = self.visualize_detections(image, defects)
        
        return result_image, defects
    
    def _detect_using_hough(self, gray):
        """Detect streaks using Hough Line Transform"""
        streaks = []
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Apply Hough transform
        lines = cv2.HoughLinesP(edges, 
                               rho=1, 
                               theta=np.pi/180, 
                               threshold=50,
                               minLineLength=self.min_streak_length,
                               maxLineGap=10)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate line properties
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                # Check if it's a valid streak
                if length >= self.min_streak_length:
                    # Estimate contrast along the line
                    line_mask = np.zeros_like(gray)
                    cv2.line(line_mask, (x1, y1), (x2, y2), 255, 3)
                    
                    # Get pixels along and around the line
                    dilated_mask = cv2.dilate(line_mask, np.ones((5, 5), np.uint8))
                    line_pixels = gray[line_mask > 0]
                    surrounding_pixels = gray[(dilated_mask > 0) & (line_mask == 0)]
                    
                    if len(line_pixels) > 0 and len(surrounding_pixels) > 0:
                        contrast = abs(np.mean(line_pixels) - np.mean(surrounding_pixels))
                        
                        if contrast >= self.contrast_threshold:
                            streaks.append({
                                'start': (x1, y1),
                                'end': (x2, y2),
                                'angle': angle,
                                'length': length,
                                'contrast': contrast,
                                'method': 'hough'
                            })
        
        return streaks
    
    def _detect_using_morphology(self, gray):
        """Detect streaks using morphological operations"""
        streaks = []
        
        # Try different orientations
        angles = np.arange(0, 180, 15)
        
        for angle in angles:
            # Create oriented structuring element
            length = max(5, self.min_streak_length // 2)  # Ensure minimum length
            if length % 2 == 0:
                length += 1  # Make odd for symmetry
            center = (float(length//2), 1.0)
            kernel = cv2.getRotationMatrix2D(center, float(angle), 1.0)
            
            # Create line kernel
            line_kernel = np.zeros((3, length), dtype=np.uint8)
            line_kernel[1, :] = 1
            
            # Rotate kernel
            rotated_kernel = cv2.warpAffine(line_kernel, kernel, (length, 3))
            
            # Apply morphological operations
            closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, rotated_kernel)
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, rotated_kernel)
            
            # Find differences
            diff = cv2.absdiff(gray, opened)
            _, binary = cv2.threshold(diff, self.contrast_threshold, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Fit line to contour
                if len(contour) >= 5:
                    [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
                    
                    # Calculate line extent
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    
                    # Get length and width
                    width = min(rect[1])
                    length = max(rect[1])
                    
                    if length >= self.min_streak_length and width <= self.max_streak_width:
                        # Calculate endpoints  
                        # Avoid division by zero
                        if abs(vx) > 0.001:
                            lefty = int((-x * vy / vx) + y)
                            righty = int(((gray.shape[1] - x) * vy / vx) + y)
                        else:
                            # Vertical line
                            lefty = 0
                            righty = gray.shape[0] - 1
                        
                        streaks.append({
                            'start': (0, lefty),
                            'end': (gray.shape[1] - 1, righty),
                            'angle': angle,
                            'length': length,
                            'width': width,
                            'method': 'morphology'
                        })
        
        return streaks
    
    def _detect_using_gradients(self, gray):
        """Detect streaks using gradient analysis"""
        streaks = []
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Threshold gradient magnitude
        _, binary = cv2.threshold(magnitude, np.mean(magnitude) + 2*np.std(magnitude), 
                                 255, cv2.THRESH_BINARY)
        binary = binary.astype(np.uint8)
        
        # Apply morphological operations to connect streak segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Skeletonize to get streak centerlines
        skeleton = morphology.skeletonize(closed > 0)
        
        # Find connected components
        labeled = measure.label(skeleton)
        
        for region in measure.regionprops(labeled):
            if region.major_axis_length >= self.min_streak_length:
                # Get orientation and endpoints
                y0, x0 = region.centroid
                orientation = region.orientation
                
                # Calculate endpoints based on major axis
                dx = region.major_axis_length / 2 * np.cos(orientation)
                dy = region.major_axis_length / 2 * np.sin(orientation)
                
                x1 = int(x0 - dx)
                y1 = int(y0 - dy)
                x2 = int(x0 + dx)
                y2 = int(y0 + dy)
                
                streaks.append({
                    'start': (x1, y1),
                    'end': (x2, y2),
                    'angle': orientation * 180 / np.pi,
                    'length': region.major_axis_length,
                    'width': region.minor_axis_length,
                    'method': 'gradient'
                })
        
        return streaks
    
    def _filter_and_merge_streaks(self, streaks):
        """Filter and merge similar streaks"""
        if not streaks:
            return []
            
        # Group streaks by angle
        filtered = []
        processed = set()
        
        for i, streak1 in enumerate(streaks):
            if i in processed:
                continue
                
            # Find similar streaks
            similar = [streak1]
            for j, streak2 in enumerate(streaks[i+1:], i+1):
                if j in processed:
                    continue
                    
                # Check angle similarity
                angle_diff = abs(streak1['angle'] - streak2['angle'])
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                    
                if angle_diff <= self.angle_tolerance:
                    # Check proximity
                    dist = self._line_distance(streak1, streak2)
                    if dist < 20:  # pixels
                        similar.append(streak2)
                        processed.add(j)
            
            # Merge similar streaks
            if len(similar) > 1:
                merged = self._merge_streaks(similar)
                filtered.append(merged)
            else:
                filtered.append(streak1)
                
        return filtered
    
    def _line_distance(self, streak1, streak2):
        """Calculate distance between two line segments"""
        # Simple point-to-line distance
        x1, y1 = streak1['start']
        x2, y2 = streak1['end']
        x3, y3 = streak2['start']
        
        # Distance from point to line
        num = abs((y2 - y1)*x3 - (x2 - x1)*y3 + x2*y1 - y2*x1)
        den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        
        return num / den if den > 0 else float('inf')
    
    def _merge_streaks(self, streaks):
        """Merge multiple streaks into one"""
        # Get all points
        all_points = []
        for streak in streaks:
            all_points.append(streak['start'])
            all_points.append(streak['end'])
        
        # Find extreme points
        all_points = np.array(all_points)
        
        # Fit line through all points
        mean_point = np.mean(all_points, axis=0)
        _, _, vt = np.linalg.svd(all_points - mean_point)
        direction = vt[0]
        
        # Project points onto line
        projections = np.dot(all_points - mean_point, direction)
        min_proj = np.min(projections)
        max_proj = np.max(projections)
        
        # Calculate endpoints
        start = mean_point + min_proj * direction
        end = mean_point + max_proj * direction
        
        return {
            'start': tuple(start.astype(int)),
            'end': tuple(end.astype(int)),
            'angle': np.arctan2(direction[1], direction[0]) * 180 / np.pi,
            'length': np.linalg.norm(end - start),
            'width': np.mean([s.get('width', 1) for s in streaks]),
            'contrast': np.max([s.get('contrast', 0) for s in streaks]),
            'method': 'merged'
        }
    
    def visualize_detections(self, image, defects):
        """Visualize detected streaks"""
        result = image.copy()
        
        # Create overlay
        overlay = np.zeros_like(result)
        
        for i, defect in enumerate(defects):
            # Draw streak line
            start = tuple(map(int, defect['start_point']))
            end = tuple(map(int, defect['end_point']))
            
            # Color based on detection method
            color = (0, 255, 255)  # Yellow for streaks
            
            # Draw the streak with width
            width = max(1, int(defect.get('width', 2)))  # Ensure minimum width of 1
            cv2.line(overlay, start, end, color, width)
            
            # Draw endpoints
            cv2.circle(overlay, start, 5, (0, 255, 0), -1)
            cv2.circle(overlay, end, 5, (255, 0, 0), -1)
            
            # Add label
            mid_point = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
            cv2.putText(overlay, 
                       f"Streak {i+1}: {defect['length']:.0f}px",
                       mid_point,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Blend with original
        alpha = 0.7
        result = cv2.addWeighted(result, 1, overlay, alpha, 0)
        
        return result
    
    def process_folder(self, input_folder, output_folder=None):
        """Process all images in a folder"""
        if output_folder is None:
            output_folder = os.path.join(input_folder, 'output', 'streaks')
            
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
        image_files = [f for f in os.listdir(input_folder) 
                      if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        results = {}
        
        for img_file in tqdm(image_files, desc="Detecting streaks"):
            img_path = os.path.join(input_folder, img_file)
            image = cv2.imread(img_path)
            
            if image is None:
                continue
                
            # Detect streaks
            result_img, defects = self.detect(image)
            
            # Save result
            output_path = os.path.join(output_folder, f'streaks_{img_file}')
            cv2.imwrite(output_path, result_img)
            
            results[img_file] = {
                'defects': defects,
                'count': len(defects),
                'output_path': output_path
            }
            
        return results 