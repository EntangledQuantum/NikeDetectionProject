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
    
    def __init__(self, kernel_size=20, delta_x_threshold=15, 
                 vertical_line_strength=0.3, sensitivity='medium', debug=True):
        """
        Args:
            kernel_size: Size of the tracking kernel (square)
            delta_x_threshold: Maximum allowed deviation from last X position
            vertical_line_strength: Minimum strength required for vertical line detection
            sensitivity: Detection sensitivity level
            debug: Whether to draw debug visualization (always True for kernel boxes)
        """
        self.kernel_size = kernel_size
        self.delta_x_threshold = delta_x_threshold
        self.vertical_line_strength = vertical_line_strength
        self.debug = True  # Always show debug for kernel visualization
        self.step_size = kernel_size  # Vertical step to avoid overlap
        
        print(f"Vertical Line Dislocation Detector - Sensitivity: {sensitivity}")
        
        # Adjust parameters based on sensitivity
        if sensitivity == 'high':
            self.kernel_size = 15
            self.delta_x_threshold = 10
            self.step_size = 15
            self.vertical_line_strength = 0.02  # Much lower threshold for Sobel responses
        elif sensitivity == 'low':
            self.kernel_size = 60  # Updated from user's change
            self.delta_x_threshold = 500
            self.step_size = 60
            self.vertical_line_strength = 0.3  # Much lower threshold for Sobel responses
            print("low sensitivity")
        else:  # medium
            self.vertical_line_strength = 0.03  # Much lower threshold for Sobel responses
        
        # Create vertical line detection filter
        self.vertical_filter = self._create_vertical_line_filter()
    
    def _create_vertical_line_filter(self):
        """Create standard Sobel vertical edge detection filter"""
        # Use the standard Sobel vertical edge detector from web research
        # This is the proven method for detecting vertical edges
        vertical_filter = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32)
        
        if self.debug:
            print(f"Created standard Sobel vertical edge filter:")
            print(vertical_filter)
        
        return vertical_filter
    

    
    def track_vertical_line_with_filter_response(self, filter_response):
        """Track vertical line using the pre-computed filter response across entire image"""
        height, width = filter_response.shape
        kernel_states = []  # For debug visualization - show all kernel positions
        defects = []
        
        # Track last encountered X position for deviation calculation
        last_x = None
        
        # Track if we're currently in a dislocation
        in_dislocation = False
        dislocation_start_y = None
        
        # Start from top and move down by kernel size each time
        y = self.kernel_size // 2
        
        while y < height - self.kernel_size // 2:
            # Scan horizontally across the entire width using kernel-sized steps
            best_x = None
            best_response = 0.0
            
            # Apply kernels across the width until we find a line
            x = self.kernel_size // 2
            found_line_in_row = False
            
            while x < width - self.kernel_size // 2 and not found_line_in_row:
                # Extract kernel-sized region from filter response
                y1 = max(0, y - self.kernel_size // 2)
                y2 = min(height, y + self.kernel_size // 2)
                x1 = max(0, x - self.kernel_size // 2)
                x2 = min(width, x + self.kernel_size // 2)
                
                # Get the filter response for this kernel region
                kernel_region = filter_response[y1:y2, x1:x2]
                
                # Calculate average response in this kernel region
                avg_response = np.mean(kernel_region)
                
                # Check if this kernel detects a vertical line
                has_line = avg_response > self.vertical_line_strength
                
                # Record kernel state for visualization
                kernel_states.append({
                    'x': x,
                    'y': y,
                    'has_line': has_line,
                    'is_scanning': True,
                    'is_best': has_line,  # First detection is the best
                    'response': avg_response,
                    'bbox': (x1, y1, x2, y2)
                })
                
                # If we found a line, stop scanning this row
                if has_line:
                    best_response = avg_response
                    best_x = x
                    found_line_in_row = True
                    if self.debug:
                        print(f"Y={y}: Found line at X={x}, response={avg_response:.3f} - STOPPING row scan")
                else:
                    # Move to next kernel position horizontally
                    x += self.kernel_size
            
            # Process the best detection for this Y level
            if best_x is not None:
                # Found vertical line at best_x
                current_x = best_x
                
                if last_x is not None:
                    # Calculate deviation from last known X position
                    deviation = abs(current_x - last_x)
                    is_dislocated = deviation > self.delta_x_threshold
                    
                    # Mark the best kernel as dislocated if needed
                    for state in kernel_states:
                        if state['y'] == y and state.get('is_best', False):
                            state['is_dislocated'] = is_dislocated
                            state['deviation'] = deviation
                    
                    if is_dislocated:
                        if not in_dislocation:
                            # Start of new dislocation
                            in_dislocation = True
                            dislocation_start_y = y
                            if self.debug:
                                print(f"DISLOCATION STARTED: Y={y}, X={current_x}, deviation={deviation:.1f} from last_x={last_x}")
                    else:
                        if in_dislocation:
                            # End of dislocation - record it
                            defects.append({
                                'type': 'vertical_line_dislocation',
                                'start_y': dislocation_start_y,
                                'end_y': y,
                                'x_position': current_x,
                                'location': (current_x, (dislocation_start_y + y) // 2),
                                'deviation': deviation,
                                'length': y - dislocation_start_y
                            })
                            
                            if self.debug:
                                print(f"DISLOCATION ENDED: Y={y}, recorded defect")
                            
                            in_dislocation = False
                else:
                    # First line found - no deviation to calculate
                    deviation = 0
                    is_dislocated = False
                    # Mark the best kernel as first line
                    for state in kernel_states:
                        if state['y'] == y and state.get('is_best', False):
                            state['is_dislocated'] = False
                            state['deviation'] = 0
                    
                    if self.debug:
                        print(f"FIRST LINE: Y={y}, X={current_x}")
                
                # Update last known X position
                last_x = current_x
                
                if self.debug:
                    status = "DISLOCATED" if is_dislocated else "NORMAL"
                    print(f"RESULT: X={current_x}, deviation={deviation:.1f}, response={best_response:.3f} [{status}]")
            
            # else:
            #     # No vertical line found at this Y level
            #     if self.debug:
            #         print(f"NO LINE FOUND at Y={y}")
            
            # Move to next position vertically by kernel size
            y += self.kernel_size  # Move by full kernel size
        
        # Handle case where dislocation extends to end of image
        if in_dislocation:
            defects.append({
                'type': 'vertical_line_dislocation',
                'start_y': dislocation_start_y,
                'end_y': y,
                'x_position': last_x,
                'location': (last_x, (dislocation_start_y + y) // 2),
                'deviation': abs(last_x - (defects[0]['x_position'] if defects else width//2)),
                'length': y - dislocation_start_y
            })
            
            if self.debug:
                print(f"FINAL DISLOCATION: Extended to end of image")
        
        return kernel_states, defects
    
    def apply_vertical_filter_to_entire_image(self, image):
        """Apply vertical edge filter to entire image using proper convolution"""
        # Apply the vertical edge filter using OpenCV filter2D (proper convolution)
        filter_response = cv2.filter2D(image, cv2.CV_32F, self.vertical_filter)
        
        # Take absolute value to get edge strength (traditional approach)
        filter_response = np.abs(filter_response)
        
        if self.debug:
            print(f"Applied vertical edge filter to entire image, response range: {filter_response.min():.3f} to {filter_response.max():.3f}")
        
        return filter_response
    
    def save_filter_response_image(self, filter_response, output_path="vertical_filter_response.jpg"):
        """Save the vertical filter response as black background with white edges"""
        # Apply threshold to get binary edge map
        # Use a low threshold to capture all significant edges
        edge_threshold = np.percentile(filter_response, 95)  # Top 5% of responses
        
        # Create binary edge image: white edges on black background
        edge_image = np.zeros_like(filter_response, dtype=np.uint8)
        edge_image[filter_response > edge_threshold] = 255
        
        # Apply morphological closing to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))  # Vertical kernel
        edge_image = cv2.morphologyEx(edge_image, cv2.MORPH_CLOSE, kernel)
        
        # Save the binary edge image
        cv2.imwrite(output_path, edge_image)
        
        if self.debug:
            print(f"Saved binary edge image (black bg, white edges) to: {output_path}")
            print(f"Edge threshold used: {edge_threshold:.6f}")
            print(f"Original response range: {filter_response.min():.6f} to {filter_response.max():.6f}")
            print(f"Edges detected: {np.sum(edge_image > 0)} pixels")
        
        return output_path

    def detect(self, image):
        """Main detection method - expects grayscale image from run_all_detections.py"""
        # Image should already be grayscale from ImagePreprocessor.load_and_convert_to_grayscale()
        if len(image.shape) == 3:
            raise ValueError("Expected grayscale image, but received color image. Check run_all_detections.py preprocessing.")
        
        gray = image.copy()
        
        # Apply Gaussian blur to reduce noise before edge detection
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Enhance contrast for better vertical edge detection
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # Normalize image to 0-1 range for filter convolution
        normalized = enhanced.astype(np.float32) / 255.0
        
        # Apply vertical line filter to entire image
        filter_response = self.apply_vertical_filter_to_entire_image(normalized)
        
        # Save filter response image for debugging
        filter_image_path = self.save_filter_response_image(filter_response)
        
        if self.debug:
            print(f"Image shape: {image.shape}, processing with kernel size: {self.kernel_size}")
            print(f"Vertical line strength threshold: {self.vertical_line_strength}")
            print(f"Delta X threshold: {self.delta_x_threshold}")
            print(f"Applied convolution filter to entire image")
            print(f"Filter response image saved to: {filter_image_path}")
        
        # Track the vertical line through the entire image using the filter response
        kernel_states, defects = self.track_vertical_line_with_filter_response(filter_response)
        
        # Create visualization
        visualization = self.create_visualization(image, defects, kernel_states)
        
        if self.debug:
            print(f"Detection complete: Found {len(defects)} dislocations")
            for i, defect in enumerate(defects):
                print(f"  Dislocation {i+1}: Y={defect['start_y']}-{defect['end_y']}, "
                      f"X={defect['x_position']}, deviation={defect['deviation']:.1f}")
        
        # Return tuple format (visualization, defects)
        return visualization, defects
    
    def create_visualization(self, original, defects, kernel_states=None):
        """Create visualization with detected defects highlighted - ALWAYS show kernel boxes"""
        if len(original.shape) == 2:
            vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            vis = original.copy()
        
        # Create overlay
        overlay = vis.copy()
        
        # Always draw kernel boxes to show the scanning process from LEFT TO RIGHT
        if kernel_states:
            for state in kernel_states:
                x = state['x']
                y = state['y']
                x1, y1, x2, y2 = state['bbox']
                
                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(vis.shape[1], x2)
                y2 = min(vis.shape[0], y2)
                
                # Determine color and thickness based on scanning state
                if state.get('is_scanning', False):
                    # This is a scanning kernel - show the scanning process
                    if state.get('is_best', False):
                        # This is the best response kernel
                        if state.get('is_dislocated', False):
                            color = (0, 0, 255)  # RED for dislocated best kernel
                            thickness = 4  # Extra thick for dislocated best
                        else:
                            color = (0, 255, 0)  # GREEN for normal best kernel
                            thickness = 3  # Thick for best kernel
                    elif state.get('has_line', False):
                        color = (0, 255, 255)  # YELLOW for detected but not best
                        thickness = 2
                    else:
                        color = (128, 128, 128)  # GRAY for scanning with no detection
                        thickness = 1
                else:
                    # Legacy kernel state handling
                    if state.get('is_dislocated', False):
                        color = (0, 0, 255)  # RED for dislocated sections
                        thickness = 3
                    elif state['has_line']:
                        color = (0, 255, 0)  # GREEN for normal vertical line
                        thickness = 2
                    else:
                        color = (0, 165, 255)  # ORANGE for missing line
                        thickness = 2
                
                # Draw kernel box
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
                
                # Draw centroid as small circle for important kernels
                if state.get('is_best', False) or state.get('is_dislocated', False):
                    cv2.circle(overlay, (x, y), 4, color, -1)
                
                # Only show text in debug mode and only for important kernels
                if self.debug:
                    # Add deviation text for dislocated kernels
                    if state.get('is_dislocated', False) and state.get('deviation', 0) > 0:
                        text = f"D{int(state['deviation'])}"  # Use 'D' instead of delta symbol
                        cv2.putText(overlay, text, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.4, (0, 0, 255), 1)
                    
                    # Add response value only for kernels that detect lines
                    if 'response' in state and state.get('has_line', False):
                        response_text = f"{state['response']:.3f}"
                        text_y = y2 + 12
                        
                        # Color code based on response strength
                        if state['response'] > self.vertical_line_strength:
                            text_color = (0, 255, 0)  # Green for above threshold
                        else:
                            text_color = (128, 128, 128)  # Gray for below threshold
                        
                        cv2.putText(overlay, response_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.3, text_color, 1)
        
        # Only draw defect regions when NOT in debug mode
        if not self.debug:
            for defect in defects:
                if defect['type'] == 'vertical_line_dislocation':
                    start_y = defect['start_y']
                    end_y = defect['end_y']
                    x = defect['x_position']
                    
                    # Draw filled red rectangle overlay for dislocation segment
                    thickness = 20  # Thickness of the dislocation indicator
                    cv2.rectangle(overlay, 
                                (x - thickness, start_y), 
                                (x + thickness, end_y), 
                                (0, 0, 255),  # Red color
                                -1)  # Filled rectangle
        
        # Blend with original to show both kernels and defects
        result = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
        
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
        vertical_line_strength=0.3,  # Default strength
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