"""
Vertical Line Dislocation Detection Algorithm
Detects vertical line edge dislocations in stripe images using kernel-based tracking

Author: Assistant
Date: 2024
"""

import cv2
import numpy as np
from scipy import signal, ndimage
import os


class VerticalLineDislocationDetector:
    """Detects vertical line dislocations using kernel-based vertical tracking"""
    
    def __init__(self, kernel_size=20, search_range=10,
                 delta_x_threshold=15, sensitivity='medium', debug=False):
        """
        Args:
            kernel_size: Size of the tracking kernel (square)
            search_range: Horizontal search range when line is lost
            delta_x_threshold: Maximum allowed deviation from mean X position
            sensitivity: Detection sensitivity level
            debug: Whether to draw debug visualization
        """
        self.kernel_size = kernel_size
        self.search_range = search_range
        self.delta_x_threshold = delta_x_threshold
        self.debug = debug
        self.step_size = kernel_size  # Vertical step to avoid overlap
        
        print(f"Vertical Line Dislocation Detector - Sensitivity: {sensitivity}")
        
        # Adjust parameters based on sensitivity
        if sensitivity == 'high':
            self.kernel_size = 15
            self.search_range = 15
            self.delta_x_threshold = 10
            self.step_size = 15
            self.line_threshold = 0.25  # Require 25% pixels for a valid line (aggressive)
            self.strong_line_threshold = 0.40  # Require 40% pixels to change X position
            self.max_x_drift = 8  # Maximum X drift per step
            self.stability_weight = 0.7  # Weight for previous position (higher = more stable)
        elif sensitivity == 'low':
            self.kernel_size = 50
            self.search_range = 10  
            self.delta_x_threshold = 25
            self.step_size = 45
            self.line_threshold = 0.20  # Only 20% pixels needed (lenient)
            self.strong_line_threshold = 0.05  # Require 5% pixels to change X position
            self.max_x_drift = 5  # Maximum X drift per step
            self.stability_weight = 0.7  # Weight for previous position
        else:  # medium
            self.line_threshold = 0.10  # 10% pixels needed (balanced)
            self.strong_line_threshold = 0.20  # Require 20% pixels to change X position
            self.max_x_drift = 10  # Maximum X drift per step
            self.stability_weight = 0.65  # Weight for previous position
    
    def scan_for_vertical_lines(self, binary_image):
        """Scan image to find vertical lines by scanning horizontally"""
        height, width = binary_image.shape
        
        # For stripe images, we expect one main vertical line
        # Scan horizontally across the image to find the vertical line
        x_positions = []
        
        # Sample at different Y positions to find where the vertical line is
        sample_y_positions = np.linspace(self.kernel_size // 2, 
                                       height - self.kernel_size // 2, 
                                       min(10, height // self.kernel_size))
        
        for y in sample_y_positions:
            y = int(y)
            # Scan horizontally at this Y position
            for x in range(self.kernel_size // 2, width - self.kernel_size // 2, self.step_size):
                # Extract kernel region
                y1 = max(0, y - self.kernel_size // 2)
                y2 = min(height, y + self.kernel_size // 2)
                x1 = max(0, x - self.kernel_size // 2)
                x2 = min(width, x + self.kernel_size // 2)
                
                kernel_region = binary_image[y1:y2, x1:x2]
                
                # Check if there's a vertical line in kernel
                white_pixels = np.sum(kernel_region > 0)
                total_pixels = (y2 - y1) * (x2 - x1)
                
                if white_pixels > total_pixels * self.line_threshold:
                    # Found a line, calculate X centroid
                    y_indices, x_indices = np.where(kernel_region > 0)
                    if len(x_indices) > 0:
                        local_x_center = np.mean(x_indices)
                        global_x = x1 + int(local_x_center)
                        x_positions.append(global_x)
                        break  # Found the line at this Y, move to next Y
        
        if x_positions:
            # Calculate the expected X position of the vertical line
            expected_x = int(np.mean(x_positions))
            if self.debug:
                print(f"Found vertical line at expected X position: {expected_x}")
            return expected_x
        else:
            if self.debug:
                print("No vertical line found in initial scan")
            return width // 2  # Default to center if no line found
    
    def track_vertical_line(self, binary_image, expected_x):
        """Track the vertical line from top to bottom using kernel"""
        height, width = binary_image.shape
        kernel_states = []  # For debug visualization
        defects = []
        
        # Starting position - start from the top
        y = self.kernel_size // 2
        x = expected_x
        
        # Track X positions for mean calculation
        x_positions = []  # Store X positions for mean calculation
        running_mean_x = expected_x  # Running mean of X positions
        
        # Track if we're currently in a dislocation
        in_dislocation = False
        dislocation_start_y = None
        
        while y < height - self.kernel_size // 2:
            # Extract kernel region
            y1 = max(0, y - self.kernel_size // 2)
            y2 = min(height, y + self.kernel_size // 2)
            x1 = max(0, x - self.kernel_size // 2)
            x2 = min(width, x + self.kernel_size // 2)
            
            kernel_region = binary_image[y1:y2, x1:x2]
             
            # Check if there's a line in kernel
            white_pixels = np.sum(kernel_region > 0)
            total_pixels = (y2 - y1) * (x2 - x1)
            has_line = white_pixels > total_pixels * self.line_threshold
            
            if has_line:
                # Calculate centroid of line pixels in kernel
                y_indices, x_indices = np.where(kernel_region > 0)
                if len(x_indices) > 0:
                    # Calculate new X position from line centroid
                    local_x_center = np.mean(x_indices)
                    new_x = x1 + int(local_x_center)
                    
                    # Apply stability logic for X position
                    if white_pixels > total_pixels * self.strong_line_threshold:
                        # Strong signal - allow X position change but limit drift
                        x_drift = abs(new_x - x)
                        if x_drift <= self.max_x_drift:
                            # Apply weighted average for stability
                            x = int(self.stability_weight * x + (1 - self.stability_weight) * new_x)
                        else:
                            # Too much drift - stay at previous position
                            if self.debug:
                                print(f"Prevented large X drift: {x_drift} pixels at Y={y}")
                    else:
                        # Weak signal - stay at previous X position
                        if self.debug:
                            print(f"Weak line signal - maintaining X position at Y={y}")
                    
                    # Check for dislocation
                    deviation = abs(x - running_mean_x)
                    is_dislocated = deviation > self.delta_x_threshold
                    
                    if is_dislocated:
                        if not in_dislocation:
                            # Start of new dislocation
                            in_dislocation = True
                            dislocation_start_y = y
                            if self.debug:
                                print(f"Dislocation started at Y={y}, X={x}, deviation={deviation:.1f}")
                    else:
                        if in_dislocation:
                            # End of dislocation - record it
                            defects.append({
                                'type': 'vertical_line_dislocation',
                                'start_y': dislocation_start_y,
                                'end_y': y,
                                'x_position': x,
                                'location': (x, (dislocation_start_y + y) // 2),
                                'deviation': deviation,
                                'length': y - dislocation_start_y
                            })
                            
                            # Reset mean calculation from this point
                            x_positions = [x]  # Reset with current position
                            running_mean_x = x  # Reset mean
                            in_dislocation = False
                            
                            if self.debug:
                                print(f"Dislocation ended at Y={y}, recorded defect")
                        else:
                            # Normal line - update running mean
                            x_positions.append(x)
                            # Keep only recent positions for mean calculation
                            if len(x_positions) > 10:
                                x_positions = x_positions[-10:]
                            running_mean_x = np.mean(x_positions)
                    
                    # Record kernel state for visualization
                    kernel_states.append({
                        'x': x,
                        'y': y,
                        'has_line': True,
                        'is_dislocated': is_dislocated,
                        'deviation': deviation,
                        'bbox': (x1, y1, x2, y2)
                    })
            else:
                # No line found - try searching horizontally
                found = False
                best_x = x
                max_pixels = 0
                
                for dx in range(-self.search_range, self.search_range + 1):
                    test_x = x + dx
                    if 0 <= test_x - self.kernel_size // 2 and test_x + self.kernel_size // 2 < width:
                        test_x1 = test_x - self.kernel_size // 2
                        test_x2 = test_x + self.kernel_size // 2
                        test_region = binary_image[y1:y2, test_x1:test_x2]
                        
                        white_pixels = np.sum(test_region > 0)
                        if white_pixels > max_pixels:
                            max_pixels = white_pixels
                            best_x = test_x
                        
                        if white_pixels > (self.kernel_size * self.kernel_size) * self.line_threshold:
                            # Found line at different X
                            x = test_x
                            found = True
                            break
                
                if not found:
                    # No line found even after searching - stay at previous position
                    if self.debug:
                        print(f"No line found at Y={y}, maintaining X={x}")
                
                # Record kernel state for missing line (red box)
                kernel_states.append({
                    'x': x,
                    'y': y,
                    'has_line': found,
                    'is_dislocated': False,
                    'deviation': 0,
                    'bbox': (x - self.kernel_size//2, y1, x + self.kernel_size//2, y2)
                })
            
            # Move to next position vertically
            y += self.step_size
        
        # Handle case where dislocation extends to end of image
        if in_dislocation:
            defects.append({
                'type': 'vertical_line_dislocation',
                'start_y': dislocation_start_y,
                'end_y': y,
                'x_position': x,
                'location': (x, (dislocation_start_y + y) // 2),
                'deviation': abs(x - running_mean_x),
                'length': y - dislocation_start_y
            })
        
        return kernel_states, defects
    
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
        
        # Binary threshold
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, 21, -5)
        
        # Invert if necessary (lines should be white)
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
        
        # Find the expected vertical line position
        expected_x = self.scan_for_vertical_lines(binary)
        
        if self.debug:
            print(f"Starting vertical line tracking at X={expected_x}")
        
        # Track the vertical line
        kernel_states, defects = self.track_vertical_line(binary, expected_x)
        
        # Create visualization
        visualization = self.create_visualization(image, defects, kernel_states)
        
        # Return tuple format (visualization, defects)
        return visualization, defects
    
    def create_visualization(self, original, defects, kernel_states=None):
        """Create visualization with detected defects highlighted"""
        if len(original.shape) == 2:
            vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            vis = original.copy()
        
        # Create overlay
        overlay = vis.copy()
        
        # Draw debug kernels if enabled
        if self.debug and kernel_states:
            for state in kernel_states:
                x = state['x']
                y = state['y']
                x1, y1, x2, y2 = state['bbox']
                
                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(vis.shape[1], x2)
                y2 = min(vis.shape[0], y2)
                
                # Draw kernel box with appropriate color
                if state.get('is_dislocated', False):
                    color = (0, 0, 255)  # Red for dislocated sections
                elif state['has_line']:
                    color = (0, 255, 0)  # Green for normal vertical line
                else:
                    color = (0, 165, 255)  # Orange for missing line
                
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                
                # Draw centroid as small circle
                cv2.circle(overlay, (x, y), 3, (0, 0, 255), -1)
            
            # In debug mode, blend lightly to see kernels clearly
            result = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
            return result
        
        # Draw defects when not in debug mode
        for defect in defects:
            if defect['type'] == 'vertical_line_dislocation':
                start_y = defect['start_y']
                end_y = defect['end_y']
                x = defect['x_position']
                
                # Draw filled red rectangle overlay for dislocation segment
                thickness = 30  # Thickness of the dislocation indicator
                cv2.rectangle(overlay, 
                            (x - thickness, start_y), 
                            (x + thickness, end_y), 
                            (0, 0, 255),  # Red color
                            -1)  # Filled rectangle
        
        # Blend with original to show the affected areas
        result = cv2.addWeighted(vis, 0.5, overlay, 0.5, 0)
        
        return result


def process_single_image(image_path, output_dir, detector):
    """Process a single image for vertical line dislocation detection"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    # Detect defects
    result_img, defects = detector.detect(image)
    
    # Save results
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save visualization
    vis_path = os.path.join(output_dir, f"{base_name}_vertical_line_dislocation_detection.jpg")
    cv2.imwrite(vis_path, result_img)
    
    return {'defects': defects, 'visualization_path': vis_path}


def main():
    """Example usage of vertical line dislocation detection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect vertical line dislocations in stripe images')
    parser.add_argument('input', help='Input image or directory path')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    parser.add_argument('--sensitivity', choices=['low', 'medium', 'high'], default='medium',
                       help='Detection sensitivity level')
    parser.add_argument('--delta-x-threshold', type=int, default=15,
                       help='Maximum allowed X deviation from mean position')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug visualization')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize detector
    detector = VerticalLineDislocationDetector(
        delta_x_threshold=args.delta_x_threshold,
        sensitivity=args.sensitivity,
        debug=args.debug
    )
    
    # Process images
    if os.path.isfile(args.input):
        results = process_single_image(args.input, args.output, detector)
        if results:
            print(f"Detected {len(results['defects'])} vertical line dislocations")
            for i, defect in enumerate(results['defects']):
                print(f"  Dislocation {i+1}: Y={defect['start_y']}-{defect['end_y']}, "
                      f"X={defect['x_position']}, deviation={defect['deviation']:.1f}")
    else:
        from tqdm import tqdm
        image_files = [f for f in os.listdir(args.input) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'))]
        
        total_defects = 0
        for img_file in tqdm(image_files, desc="Processing images"):
            img_path = os.path.join(args.input, img_file)
            results = process_single_image(img_path, args.output, detector)
            if results:
                total_defects += len(results['defects'])
        
        print(f"Total vertical line dislocations found: {total_defects}")


if __name__ == "__main__":
    main() 