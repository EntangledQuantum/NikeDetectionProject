"""
Window-based Image Processor for Efficient Defect Detection
Handles huge images by processing them in overlapping windows using multithreading

Author: Assistant
Date: 2024
"""

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
import os


class WindowProcessor:
    """Processes large images in overlapping windows for memory efficiency"""
    
    def __init__(self, window_size=2048, overlap=256, max_workers=4):
        """
        Args:
            window_size: Size of processing window (height for tall images)
            overlap: Overlap between windows to avoid edge artifacts
            max_workers: Maximum number of parallel threads
        """
        self.window_size = window_size
        self.overlap = overlap
        self.max_workers = max_workers
        self.lock = threading.Lock()
        
    def process_image_windowed(self, image_path, detectors, output_dir):
        """
        Process a large image using windowed approach
        
        Args:
            image_path: Path to input image
            detectors: Dictionary of detector instances
            output_dir: Output directory for results
        """
        # Get image dimensions without loading full image
        img_info = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img_info is None:
            return None
            
        # For very large images, read header only
        if os.path.getsize(image_path) > 100 * 1024 * 1024:  # > 100MB
            # Read just the header to get dimensions
            with open(image_path, 'rb') as f:
                # This is a simplified approach - in production use tifffile
                height, width = self._get_image_dimensions_from_header(image_path)
        else:
            height, width = img_info.shape[:2]
            del img_info  # Free memory
        
        # Determine if image needs windowed processing
        if height * width < 10_000_000:  # < 10 megapixels
            # Small enough to process directly
            return self._process_full_image(image_path, detectors)
        
        # Calculate windows
        windows = self._calculate_windows(height, width)
        
        # Initialize result accumulators
        all_results = {name: {'defects': [], 'visualizations': []} 
                      for name in detectors.keys()}
        
        # Process windows
        print(f"Processing {len(windows)} windows with {self.max_workers} worker(s)...")
        
        if self.max_workers == 1:
            # Single-threaded processing for debugging
            for i, window in enumerate(windows):
                print(f"    Processing window {i+1}/{len(windows)}: {window}")
                try:
                    window_results = self._process_window(image_path, window, detectors)
                    if window_results:
                        self._merge_window_results(all_results, window_results, window)
                        print(f"    Window {i+1} completed successfully")
                    else:
                        print(f"    Window {i+1} returned no results")
                except Exception as e:
                    print(f"    Error processing window {window}: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            # Multi-threaded processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                
                for window in windows:
                    future = executor.submit(
                        self._process_window,
                        image_path, window, detectors
                    )
                    futures.append((future, window))
                
                # Collect results with progress bar
                for future, window in tqdm(futures, desc="Processing windows"):
                    try:
                        window_results = future.result()
                        self._merge_window_results(all_results, window_results, window)
                    except Exception as e:
                        print(f"Error processing window {window}: {e}")
        
        # Create combined visualizations
        final_results = self._create_final_visualizations(
            image_path, all_results, height, width, output_dir
        )
        
        return final_results
    
    def _get_image_dimensions_from_header(self, image_path):
        """Get image dimensions without loading full image"""
        # For TIFF files, we can use tifffile to read just metadata
        try:
            import tifffile
            with tifffile.TiffFile(image_path) as tif:
                page = tif.pages[0]
                return page.shape[0], page.shape[1]
        except:
            # Fallback to opencv
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            shape = img.shape
            del img
            return shape[0], shape[1]
    
    def _calculate_windows(self, height, width):
        """Calculate overlapping windows for image processing"""
        windows = []
        
        # For very tall images (like your examples), use horizontal strips
        if height > width * 3:  # Very tall image
            y = 0
            while y < height:
                y_end = min(y + self.window_size, height)
                windows.append({
                    'x': 0,
                    'y': y,
                    'width': width,
                    'height': y_end - y
                })
                y += self.window_size - self.overlap
                if y >= height:
                    break
        else:
            # For regular images, use grid of windows
            y = 0
            while y < height:
                y_end = min(y + self.window_size, height)
                x = 0
                while x < width:
                    x_end = min(x + self.window_size, width)
                    windows.append({
                        'x': x,
                        'y': y,
                        'width': x_end - x,
                        'height': y_end - y
                    })
                    x += self.window_size - self.overlap
                    if x >= width:
                        break
                y += self.window_size - self.overlap
                if y >= height:
                    break
        
        return windows
    
    def _process_window(self, image_path, window, detectors):
        """Process a single window of the image"""
        # Read only the window region
        x, y, w, h = window['x'], window['y'], window['width'], window['height']
        
        # For TIFF files, use memory-mapped reading
        if image_path.lower().endswith(('.tif', '.tiff')):
            try:
                import tifffile
                with tifffile.TiffFile(image_path) as tif:
                    # Read just the window region
                    window_img = tif.asarray()[y:y+h, x:x+w]
                    if len(window_img.shape) == 2:
                        window_img = cv2.cvtColor(window_img, cv2.COLOR_GRAY2BGR)
            except:
                # Fallback
                img = cv2.imread(image_path)
                window_img = img[y:y+h, x:x+w]
                del img
        else:
            # For other formats
            img = cv2.imread(image_path)
            window_img = img[y:y+h, x:x+w]
            del img
        
        # Run all detectors on this window
        results = {}
        for name, detector in detectors.items():
            try:
                # Call detection method - all detectors should return (visualization, defects)
                result_img, defects = detector.detect(window_img)
                
                # Adjust defect coordinates to global image space
                adjusted_defects = []
                for defect in defects:
                    defect_copy = defect.copy()
                    # Adjust coordinates based on window position
                    if 'location' in defect_copy:
                        defect_copy['location'] = (
                            defect_copy['location'][0] + x,
                            defect_copy['location'][1] + y
                        )
                    if 'start_point' in defect_copy:
                        defect_copy['start_point'] = (
                            defect_copy['start_point'][0] + x,
                            defect_copy['start_point'][1] + y
                        )
                    if 'end_point' in defect_copy:
                        defect_copy['end_point'] = (
                            defect_copy['end_point'][0] + x,
                            defect_copy['end_point'][1] + y
                        )
                    if 'position' in defect_copy:
                        if name == 'banding' and defect_copy.get('type') == 'horizontal_banding':
                            defect_copy['position'] += y
                        elif name == 'banding' and defect_copy.get('type') == 'vertical_banding':
                            defect_copy['position'] += x
                    
                    adjusted_defects.append(defect_copy)
                
                results[name] = {
                    'defects': adjusted_defects,
                    'visualization': result_img
                }
            except Exception as e:
                print(f"Error in {name} detector for window: {e}")
                results[name] = {'defects': [], 'visualization': None}
        
        return results
    
    def _merge_window_results(self, all_results, window_results, window):
        """Merge results from a window into the overall results"""
        with self.lock:
            for name, result in window_results.items():
                # Add defects
                all_results[name]['defects'].extend(result['defects'])
                
                # Store visualization info for later combining
                if result['visualization'] is not None:
                    all_results[name]['visualizations'].append({
                        'window': window,
                        'image': result['visualization']
                    })
    
    def _process_full_image(self, image_path, detectors):
        """Process image without windowing (for smaller images)"""
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        results = {}
        for name, detector in detectors.items():
            try:
                # Call detection method - all detectors should return (visualization, defects)
                result_img, defects = detector.detect(image)
                
                results[name] = {
                    'visualization': result_img,
                    'defects': defects,
                    'defect_count': len(defects)
                }
                    
            except Exception as e:
                print(f"Error in {name} detector: {e}")
                results[name] = {
                    'visualization': image,
                    'defects': [],
                    'defect_count': 0
                }
        
        return results
    
    def _create_final_visualizations(self, image_path, all_results, height, width, output_dir):
        """Create final visualizations by combining window results"""
        final_results = {}
        
        for name, result in all_results.items():
            # Remove duplicate defects from overlapping regions
            defects = self._remove_duplicate_defects(result['defects'], name)
            
            # Create visualization
            if result['visualizations']:
                # For very large images, create a scaled visualization
                if height * width > 50_000_000:  # > 50 megapixels
                    scale_factor = np.sqrt(20_000_000 / (height * width))
                    vis_height = int(height * scale_factor)
                    vis_width = int(width * scale_factor)
                else:
                    vis_height = height
                    vis_width = width
                    scale_factor = 1.0
                
                # Load and scale the original image as background
                try:
                    if image_path.lower().endswith(('.tif', '.tiff')):
                        import tifffile
                        with tifffile.TiffFile(image_path) as tif:
                            # Read image with scaling if needed
                            if scale_factor < 1.0:
                                # Read a subset for very large images
                                step_y = max(1, int(1 / scale_factor))
                                step_x = max(1, int(1 / scale_factor))
                                image_data = tif.pages[0].asarray()[::step_y, ::step_x]
                            else:
                                image_data = tif.pages[0].asarray()
                    else:
                        image_data = cv2.imread(image_path)
                        if scale_factor < 1.0:
                            image_data = cv2.resize(image_data, (vis_width, vis_height))
                    
                    # Convert to BGR if needed
                    if len(image_data.shape) == 2:
                        visualization = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)
                    else:
                        visualization = image_data.copy()
                        
                    # Ensure correct size
                    if visualization.shape[:2] != (vis_height, vis_width):
                        visualization = cv2.resize(visualization, (vis_width, vis_height))
                        
                except Exception as e:
                    print(f"Warning: Could not load image for visualization, using white background: {e}")
                    # Fallback to white canvas
                    visualization = np.ones((vis_height, vis_width, 3), dtype=np.uint8) * 255
                
                # Draw defects on visualization
                for defect in defects:
                    self._draw_defect_on_visualization(
                        visualization, defect, name, scale_factor
                    )
                
                # Save visualization
                output_path = os.path.join(output_dir, f"{name}_defects.jpg")
                cv2.imwrite(output_path, visualization, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                final_results[name] = {
                    'defects': defects,
                    'defect_count': len(defects),
                    'visualization_path': output_path
                }
            else:
                final_results[name] = {
                    'defects': defects,
                    'defect_count': len(defects),
                    'visualization_path': None
                }
        
        return final_results
    
    def _remove_duplicate_defects(self, defects, detector_name):
        """Remove duplicate defects from overlapping windows"""
        if not defects:
            return []
        
        # Simple deduplication based on proximity
        unique_defects = []
        
        for defect in defects:
            is_duplicate = False
            
            # Get defect position
            if 'location' in defect:
                pos = defect['location']
            elif 'start_point' in defect:
                pos = defect['start_point']
            elif 'position' in defect:
                if detector_name == 'banding':
                    pos = (0, defect['position']) if defect.get('type') == 'horizontal_banding' else (defect['position'], 0)
                else:
                    pos = (defect['position'], defect['position'])
            else:
                unique_defects.append(defect)
                continue
            
            # Check against existing defects
            for existing in unique_defects:
                if 'location' in existing:
                    existing_pos = existing['location']
                elif 'start_point' in existing:
                    existing_pos = existing['start_point']
                elif 'position' in existing:
                    if detector_name == 'banding':
                        existing_pos = (0, existing['position']) if existing.get('type') == 'horizontal_banding' else (existing['position'], 0)
                    else:
                        existing_pos = (existing['position'], existing['position'])
                else:
                    continue
                
                # Check distance
                dist = np.sqrt((pos[0] - existing_pos[0])**2 + (pos[1] - existing_pos[1])**2)
                if dist < self.overlap / 4:  # Within overlap region
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_defects.append(defect)
        
        return unique_defects
    
    def _draw_defect_on_visualization(self, visualization, defect, detector_name, scale_factor):
        """Draw a defect on the visualization canvas"""
        # Scale coordinates
        def scale_point(point):
            return (int(point[0] * scale_factor), int(point[1] * scale_factor))
        
        # Color scheme for different detectors
        colors = {
            'overspray': (0, 0, 255),      # Red
            'surface_treatment': (0, 255, 0), # Green
            'debris': (255, 0, 0),          # Blue
            'gray_spot': (0, 255, 255),     # Yellow
            'edge_defect': (255, 0, 255),   # Magenta
            'banding': (255, 128, 0),       # Orange
            'streak': (0, 255, 255)         # Cyan
        }
        
        color = colors.get(detector_name, (128, 128, 128))
        
        # Draw based on defect type
        if 'location' in defect:
            # Point defect
            center = scale_point(defect['location'])
            radius = max(2, int(defect.get('size', 5) * scale_factor))
            cv2.circle(visualization, center, radius, color, -1)
            
        elif 'start_point' in defect and 'end_point' in defect:
            # Line defect
            start = scale_point(defect['start_point'])
            end = scale_point(defect['end_point'])
            thickness = max(1, int(defect.get('width', 2) * scale_factor))
            cv2.line(visualization, start, end, color, thickness)
            
        elif 'position' in defect and detector_name == 'banding':
            # Banding defect
            if defect.get('type') == 'horizontal_banding':
                y = int(defect['position'] * scale_factor)
                cv2.line(visualization, (0, y), (visualization.shape[1], y), color, 2)
            else:
                x = int(defect['position'] * scale_factor)
                cv2.line(visualization, (x, 0), (x, visualization.shape[0]), color, 2) 