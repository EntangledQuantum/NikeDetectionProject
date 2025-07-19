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
        total_pixels = height * width
        print(f"Image dimensions: {width}x{height} = {total_pixels:,} pixels ({total_pixels/1_000_000:.1f} megapixels)")
        
        if total_pixels < 10_000_000:  # < 10 megapixels
            # Small enough to process directly
            print(f"Using direct processing (< 10 megapixels)")
            return self._process_full_image(image_path, detectors, output_dir)
        else:
            print(f"Using windowed processing (>= 10 megapixels)")
        
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
        print(f"Creating final visualizations...")
        final_results = self._create_final_visualizations(
            image_path, all_results, height, width, output_dir
        )
        print(f"Final visualizations completed.")
        
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
        
        # For TIFF files, use memory-mapped reading to load ONLY the window region
        if image_path.lower().endswith(('.tif', '.tiff')):
            try:
                import tifffile
                with tifffile.TiffFile(image_path) as tif:
                    # Read ONLY the window region - this is the key optimization!
                    window_img = tif.pages[0].asarray()[y:y+h, x:x+w]
                    # Convert to BGR if grayscale
                    if len(window_img.shape) == 2:
                        window_img = cv2.cvtColor(window_img, cv2.COLOR_GRAY2BGR)
                    elif len(window_img.shape) == 3 and window_img.shape[2] == 1:
                        window_img = cv2.cvtColor(window_img.squeeze(), cv2.COLOR_GRAY2BGR)
            except Exception as e:
                print(f"        Error reading TIFF window: {e}")
                # Fallback to OpenCV (slower but more compatible)
                img = cv2.imread(image_path)
                if img is not None:
                    window_img = img[y:y+h, x:x+w]
                    del img
                else:
                    return None
        else:
            # For other formats, still need to load full image (OpenCV limitation)
            img = cv2.imread(image_path)
            if img is not None:
                window_img = img[y:y+h, x:x+w]
                del img
            else:
                return None
        
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
    
    def _process_full_image(self, image_path, detectors, output_dir):
        """Process image without windowing (for smaller images)"""
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        print(f"Processing full image directly...")
        results = {}
        
        for name, detector in detectors.items():
            print(f"  Running {name} detection...")
            try:
                # Call detection method - all detectors should return (visualization, defects)
                result_img, defects = detector.detect(image)
                
                # Save visualization to disk
                output_path = os.path.join(output_dir, f"{name}_defects.jpg")
                cv2.imwrite(output_path, result_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"    Saved {name} visualization: {len(defects)} defects found")
                
                results[name] = {
                    'visualization_path': output_path,
                    'defects': defects,
                    'defect_count': len(defects)
                }
                    
            except Exception as e:
                print(f"    Error in {name} detector: {e}")
                results[name] = {
                    'visualization_path': None,
                    'defects': [],
                    'defect_count': 0
                }
        
        print(f"✅ Full image processing completed")
        return results
    
    def _create_final_visualizations(self, image_path, all_results, height, width, output_dir):
        """Create final visualizations by combining window results"""
        final_results = {}
        
        print(f"Processing {len(all_results)} detector results...")
        
        for i, (name, result) in enumerate(all_results.items(), 1):
            print(f"  [{i}/{len(all_results)}] Creating visualization for {name}...")
            # Remove duplicate defects from overlapping regions
            defects = self._remove_duplicate_defects(result['defects'], name)
            print(f"    Found {len(defects)} defects after deduplication")
            
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
                
                # Load the actual image as background, but do it smartly
                print(f"    Creating visualization with real image background ({vis_width}x{vis_height}, scale: {scale_factor:.3f})...")
                try:
                    if image_path.lower().endswith(('.tif', '.tiff')):
                        import tifffile
                        with tifffile.TiffFile(image_path) as tif:
                            if scale_factor < 1.0:
                                # For large images, read with subsampling directly
                                step_y = max(1, int(1 / scale_factor))
                                step_x = max(1, int(1 / scale_factor))
                                print(f"    Reading TIFF with subsampling (step: {step_y}x{step_x})...")
                                image_data = tif.pages[0].asarray()[::step_y, ::step_x]
                            else:
                                print(f"    Reading full TIFF...")
                                image_data = tif.pages[0].asarray()
                        
                        # Convert to BGR if needed
                        if len(image_data.shape) == 2:
                            visualization = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)
                        elif len(image_data.shape) == 3 and image_data.shape[2] == 1:
                            visualization = cv2.cvtColor(image_data.squeeze(), cv2.COLOR_GRAY2BGR)
                        else:
                            visualization = image_data.copy()
                            
                        # Resize to exact target size if needed
                        if visualization.shape[:2] != (vis_height, vis_width):
                            visualization = cv2.resize(visualization, (vis_width, vis_height))
                            
                        print(f"    Successfully loaded TIFF background: {visualization.shape}")
                        
                    else:
                        # For regular images
                        print(f"    Reading regular image...")
                        image_data = cv2.imread(image_path)
                        if scale_factor < 1.0:
                            visualization = cv2.resize(image_data, (vis_width, vis_height))
                        else:
                            visualization = image_data.copy()
                        print(f"    Successfully loaded regular image background: {visualization.shape}")
                        
                except Exception as e:
                    print(f"    Warning: Could not load image background ({e}), using gray background")
                    visualization = np.ones((vis_height, vis_width, 3), dtype=np.uint8) * 128
                
                # Draw defects on visualization
                for defect in defects:
                    self._draw_defect_on_visualization(
                        visualization, defect, name, scale_factor
                    )
                
                # Save visualization
                output_path = os.path.join(output_dir, f"{name}_defects.jpg")
                cv2.imwrite(output_path, visualization, [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"    Saved {name} visualization: {len(defects)} defects found")
                
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
        
        # Create a legend image explaining the color scheme
        self._create_legend(output_dir, final_results)
        
        print(f"✅ All visualizations completed for {len(final_results)} detectors")
        return final_results
    
    def _create_legend(self, output_dir, results):
        """Create a legend image explaining what each color means"""
        try:
            # Create legend canvas
            legend_height = 400
            legend_width = 600
            legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255
            
            # Color scheme (same as in drawing function)
            colors = {
                'overspray': (0, 0, 255),        # Bright Red
                'surface_treatment': (0, 255, 0), # Bright Green  
                'debris': (255, 255, 0)          # Bright Yellow
            }
            
            # Descriptions
            descriptions = {
                'overspray': 'Ink scattered outside intended areas, appearing as dots trailing printed regions',
                'surface_treatment': 'Poor surface energy causing ink to combine into irregular drops, leaving areas with no ink',
                'debris': 'Foreign particles (dirt, fibers, etc.) causing dark spots with blank rings or contamination'
            }
            
            # Title
            cv2.putText(legend, "DEFECT DETECTION LEGEND", 
                       (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            
            y_pos = 80
            for detector_name, color in colors.items():
                if detector_name in results and results[detector_name]['defect_count'] > 0:
                    # Draw color sample
                    cv2.circle(legend, (70, y_pos), 15, color, -1)
                    cv2.circle(legend, (70, y_pos), 15, (0, 0, 0), 2)
                    
                    # Add label
                    label = f"{detector_name.upper().replace('_', ' ')}"
                    cv2.putText(legend, label, (100, y_pos - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    
                    # Add description
                    desc = descriptions.get(detector_name, '')
                    cv2.putText(legend, desc, (100, y_pos + 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    
                    # Add count
                    count_text = f"Found: {results[detector_name]['defect_count']} defects"
                    cv2.putText(legend, count_text, (100, y_pos + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
                    
                    y_pos += 50
            
            # Save legend
            legend_path = os.path.join(output_dir, "defect_legend.jpg")
            cv2.imwrite(legend_path, legend, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"    Created defect legend: {legend_path}")
            
        except Exception as e:
            print(f"    Warning: Could not create legend: {e}")
    
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
        """Draw defect regions - NO TEXT, JUST COLORED AREAS"""
        try:
            # Scale coordinates safely
            def scale_point(point):
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    x = max(0, min(int(point[0] * scale_factor), visualization.shape[1] - 1))
                    y = max(0, min(int(point[1] * scale_factor), visualization.shape[0] - 1))
                    return (x, y)
                return (0, 0)
            
            # Color scheme matching individual detectors
            colors = {
                'overspray': (0, 0, 255),        # Red
                'surface_treatment': (0, 255, 0), # Green  
                'debris': (0, 255, 255)          # Yellow
            }
            
            color = colors.get(detector_name, (128, 128, 128))
            
            # Draw simple filled regions - NO TEXT, NO BORDERS
            if 'location' in defect:
                # Point defect - draw filled circle
                center = scale_point(defect['location'])
                radius = max(8, min(20, int(defect.get('size', 10) * scale_factor)))
                cv2.circle(visualization, center, radius, color, -1)
                
            elif 'start_point' in defect and 'end_point' in defect:
                # Line defect - draw thick line
                start = scale_point(defect['start_point'])
                end = scale_point(defect['end_point'])
                thickness = max(3, min(8, int(defect.get('width', 4) * scale_factor)))
                cv2.line(visualization, start, end, color, thickness)
                    
        except Exception as e:
            print(f"    Warning: Could not draw defect for {detector_name}: {e}") 