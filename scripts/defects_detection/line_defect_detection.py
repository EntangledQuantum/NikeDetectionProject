"""
Line Defect Detection Algorithm
Detects missing lines and jagged/zig-zag lines in island images

Author: Assistant
Date: 2024
"""

import cv2
import numpy as np
from scipy import signal, ndimage
import os


class LineDefectDetector:
    """Detects line defects - missing segments and jagged/zig-zag patterns"""
    
    def __init__(self, min_gap_size=10, angle_tolerance=30, 
                 vertical_deviation_threshold=15, sensitivity='medium'):
        """
        Args:
            min_gap_size: Minimum gap size to consider as defect
            angle_tolerance: Maximum angle deviation from horizontal (degrees)
            vertical_deviation_threshold: Max vertical deviation for jagged detection
            sensitivity: Detection sensitivity level
        """
        self.min_gap_size = min_gap_size
        self.angle_tolerance = angle_tolerance
        self.vertical_deviation_threshold = vertical_deviation_threshold
        
        # Adjust parameters based on sensitivity
        if sensitivity == 'high':
            self.min_gap_size = 5
            self.angle_tolerance = 45
            self.vertical_deviation_threshold = 10
        elif sensitivity == 'low':
            self.min_gap_size = 20
            self.angle_tolerance = 20
            self.vertical_deviation_threshold = 20
    
    def detect_line_regions(self, binary_image):
        """Detect connected line regions using connected components"""
        # Use connected components to find all line segments
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        
        line_segments = []
        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Filter for line-like components (wider than tall)
            if width > height * 2 and area > 50:
                line_segments.append({
                    'label': i,
                    'bbox': (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]),
                    'centroid': centroids[i],
                    'area': area
                })
        
        return labels, line_segments
    
    def group_line_segments(self, line_segments, image_width):
        """Group line segments that belong to the same logical line"""
        if not line_segments:
            return []
        
        # Sort segments by Y coordinate
        sorted_segments = sorted(line_segments, key=lambda s: s['centroid'][1])
        
        # Group segments that are close in Y coordinate
        line_groups = []
        current_group = [sorted_segments[0]]
        
        for segment in sorted_segments[1:]:
            # Check if segment belongs to current group
            avg_y = np.mean([s['centroid'][1] for s in current_group])
            if abs(segment['centroid'][1] - avg_y) < 50:  # Within 50 pixels vertically
                current_group.append(segment)
            else:
                line_groups.append(current_group)
                current_group = [segment]
        
        if current_group:
            line_groups.append(current_group)
        
        return line_groups
    
    def analyze_line_continuity(self, labels, line_group, image_shape):
        """Analyze a line group for gaps and discontinuities"""
        height, width = image_shape
        defects = []
        
        # Get Y range for this line group
        y_coords = [seg['centroid'][1] for seg in line_group]
        avg_y = int(np.mean(y_coords))
        y_range = max(20, int(np.std(y_coords) * 3 + 10))
        
        # Calculate line slope by fitting a line through segment centroids
        x_coords = [seg['centroid'][0] for seg in line_group]
        if len(x_coords) > 1:
            # Fit a line to get slope
            coeffs = np.polyfit(x_coords, y_coords, 1)
            slope = coeffs[0]  # dy/dx
            intercept = coeffs[1]
        else:
            slope = 0
            intercept = avg_y
        
        # Create a mask for this line group
        line_mask = np.zeros((height, width), dtype=np.uint8)
        for segment in line_group:
            line_mask[labels == segment['label']] = 255
        
        # Scan horizontally with a wider band
        y_start = max(0, avg_y - y_range)
        y_end = min(height, avg_y + y_range)
        
        # Project line onto horizontal axis
        horizontal_projection = np.max(line_mask[y_start:y_end, :], axis=0)
        
        # Find gaps in the projection
        in_line = False
        gap_start = 0
        
        for x in range(width):
            if horizontal_projection[x] > 0:
                if not in_line:
                    # End of gap
                    if x - gap_start > self.min_gap_size and gap_start > 0:
                        # Check if there was line before the gap
                        has_line_before = np.any(horizontal_projection[max(0, gap_start-50):gap_start] > 0)
                        has_line_after = np.any(horizontal_projection[x:min(width, x+50)] > 0)
                        
                        if has_line_before and has_line_after:
                            defects.append({
                                'type': 'missing_line',
                                'start_x': gap_start,
                                'end_x': x,
                                'slope': slope,
                                'intercept': intercept,
                                'location': ((gap_start + x) // 2, int(slope * ((gap_start + x) // 2) + intercept)),
                                'size': x - gap_start
                            })
                    in_line = True
            else:
                if in_line:
                    gap_start = x
                    in_line = False
        
        # Check for jagged sections by analyzing vertical deviations
        for segment in line_group:
            x, y, w, h = segment['bbox']
            if h > self.vertical_deviation_threshold:
                # This segment has significant vertical variation
                segment_mask = (labels == segment['label']).astype(np.uint8)
                segment_region = segment_mask[y:y+h, x:x+w]
                
                # Analyze vertical distribution
                for col in range(0, w, 5):  # Sample every 5 pixels
                    if col < segment_region.shape[1]:
                        col_pixels = segment_region[:, col]
                        if np.sum(col_pixels) > 0:
                            pixel_positions = np.where(col_pixels > 0)[0]
                            if len(pixel_positions) > 1:
                                spread = np.max(pixel_positions) - np.min(pixel_positions)
                                if spread > self.vertical_deviation_threshold:
                                    defects.append({
                                        'type': 'jagged_line',
                                        'start_x': x + col - 10,
                                        'end_x': x + col + 10,
                                        'y_from': y + np.min(pixel_positions),
                                        'y_to': y + np.max(pixel_positions),
                                        'location': (x + col, y + h//2),
                                        'deviation': spread
                                    })
                                    break
        
        return defects
    
    def detect(self, image):
        """Main detection method"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Use adaptive threshold for better line detection
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, 21, -5)
        
        # Invert if necessary (lines should be white)
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
        
        # Clean up noise
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Detect line segments
        labels, line_segments = self.detect_line_regions(binary)
        
        # Group segments into logical lines
        line_groups = self.group_line_segments(line_segments, image.shape[1])
        
        # Analyze each line group for defects
        all_defects = []
        for line_group in line_groups:
            line_defects = self.analyze_line_continuity(labels, line_group, binary.shape)
            all_defects.extend(line_defects)
        
        # Create visualization
        visualization = self.create_visualization(image, all_defects)
        
        # Return tuple format (visualization, defects)
        return visualization, all_defects
    
    def create_visualization(self, original, defects):
        """Create visualization with detected defects highlighted"""
        if len(original.shape) == 2:
            vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            vis = original.copy()
        
        # Create overlay
        overlay = vis.copy()
        
        for defect in defects:
            if defect['type'] == 'missing_line':
                # Draw rectangle that follows the line slope
                x1 = defect['start_x']
                x2 = defect['end_x']
                
                # Calculate Y positions at x1 and x2 using the line equation
                if 'slope' in defect and 'intercept' in defect:
                    slope = defect['slope']
                    intercept = defect['intercept']
                    y1 = int(slope * x1 + intercept)
                    y2 = int(slope * x2 + intercept)
                else:
                    # Fallback to center location
                    y1 = y2 = defect['location'][1]
                
                # Draw a parallelogram that follows the slope
                # Calculate perpendicular offset for thickness
                thickness = 15
                if abs(slope) > 0.001:
                    # Perpendicular slope is -1/slope
                    perp_slope = -1.0 / slope
                    # Normalize to get unit perpendicular vector
                    length = np.sqrt(1 + perp_slope * perp_slope)
                    dx = thickness / length
                    dy = perp_slope * thickness / length
                else:
                    # Nearly horizontal line
                    dx = 0
                    dy = thickness
                
                # Four corners of the parallelogram
                pts = np.array([
                    [int(x1 - dx), int(y1 - dy)],
                    [int(x2 - dx), int(y2 - dy)],
                    [int(x2 + dx), int(y2 + dy)],
                    [int(x1 + dx), int(y1 + dy)]
                ], np.int32)
                
                cv2.fillPoly(overlay, [pts], (0, 0, 255))
                
            elif defect['type'] == 'jagged_line':
                # Draw yellow region for jagged/zig-zag sections
                x1 = max(0, defect['start_x'])
                x2 = min(overlay.shape[1]-1, defect['end_x'])
                y1 = max(0, defect['y_from'] - 5)
                y2 = min(overlay.shape[0]-1, defect['y_to'] + 5)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)
        
        # Blend with original
        result = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
        
        return result 