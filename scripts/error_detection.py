# %%
import cv2
import numpy as np
import os
import math
import argparse
import sys
from pathlib import Path
from skimage import feature, measure, morphology
from skimage.filters import threshold_otsu
from scipy import ndimage
from collections import defaultdict
from PIL import Image
import tifffile
from tqdm import tqdm

def read_image_file(image_path: str, input_type: str | None = None) -> np.ndarray | None:
    """
    Reads an image file supporting both regular formats (PNG, JPG) and TIFF files.
    Preserves full resolution and metadata for TIFF files.
    
    Args:
        image_path: Path to the input image
        input_type: Pre-detected input type ('tiff' or 'image'), if available
        
    Returns:
        np.ndarray | None: Loaded image in BGR format (compatible with OpenCV), or None if failed
    """
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' was not found.")
        return None
    
    # Use pre-detected type if available, otherwise check extension
    if input_type is None:
        file_ext = os.path.splitext(image_path)[1].lower()
        is_tiff = file_ext in ['.tiff', '.tif']
    else:
        is_tiff = input_type == 'tiff'
    
    if is_tiff:
        try:
            with tifffile.TiffFile(image_path) as tif:
                page = tif.pages[0]
                
                # Try memory mapping for large files first
                if hasattr(page, 'is_memmappable') and page.is_memmappable:
                    img = page.asarray(out='memmap')
                else:
                    img = page.asarray()
                
                # Handle different TIFF formats and bit depths
                if len(img.shape) == 2:  # Grayscale
                    if img.dtype != np.uint8:
                        img_float = img.astype(np.float32)
                        img = ((img_float - img_float.min()) * (255.0 / (img_float.max() - img_float.min()))).astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif len(img.shape) == 3:
                    if img.shape[2] == 3:  # RGB
                        if img.dtype != np.uint8:
                            img_float = img.astype(np.float32)
                            img = np.zeros_like(img_float, dtype=np.uint8)
                            for c in range(3):
                                channel = img_float[:,:,c]
                                img[:,:,c] = ((channel - channel.min()) * (255.0 / (channel.max() - channel.min()))).astype(np.uint8)
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    elif img.shape[2] == 4:  # RGBA
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                
                print(f"Successfully loaded TIFF: {os.path.basename(image_path)}")
                print(f"- Dimensions: {img.shape}")
                print(f"- Original dtype: {page.dtype}")
                return img
            
        except Exception as e:
            print(f"Error reading TIFF file {image_path} with tifffile: {e}")
            print("Trying with OpenCV...")
    
    # Fall back to OpenCV for regular formats or if TIFF failed
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return None
        
    return img

def save_image(image: np.ndarray, output_path: str, original_path: str | None = None) -> bool:
    """Saves image with quality preservation."""
    try:
        if output_path.lower().endswith(('.tiff', '.tif')):
            # Save as TIFF with original metadata
            tifffile.imwrite(
                output_path,
                image,
                compression=None  # No compression
            )
        else:
            # For PNG output, use maximum quality
            cv2.imwrite(output_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False

def split_image_vertically(image_path: str, output_dir: str | None = None) -> str | None:
    """
    Splits an image vertically and saves the left half.
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save the left half image (default: creates 'vertical_halves' in image directory)
        
    Returns:
        Path to the left half image if successful, None otherwise
    """
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' was not found.")
        return None
    
    # If no output directory specified, create it in the same directory as the input image
    if output_dir is None:
        image_dir = os.path.dirname(image_path)
        output_dir = os.path.join(image_dir, "vertical_halves")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Generate output filename
    base_name = os.path.basename(image_path)
    file_name, ext = os.path.splitext(base_name)
    left_half_path = os.path.join(output_dir, f"{file_name}_left_half{ext}")
    
    # Check if left half already exists
    if os.path.exists(left_half_path):
        print(f"Left half already exists at: {left_half_path}")
        return left_half_path
    
    # Read the image (supports TIFF and other formats)
    img = read_image_file(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return None
    
    # Get dimensions
    height, width, _ = img.shape
    
    # Calculate the midpoint
    mid_width = width // 2
    
    # Extract the left half
    left_half = img[:, :mid_width]
    
    # Save the left half using save_image
    if save_image(left_half, left_half_path, image_path):
        print(f"Saved left half of image to: {left_half_path}")
        print(f"Left half dimensions: {left_half.shape[1]}x{left_half.shape[0]}")
        return left_half_path
    else:
        print(f"Failed to save left half to: {left_half_path}")
        return None

def split_image(image_path: str, num_splits: int = 50, output_dir: str | None = None) -> tuple[str | None, list]:
    """
    Splits an image into a specified number of equal parts (e.g., 50).
    Saves the split images into a directory.
    
    Args:
        image_path: Path to the input image
        num_splits: Number of parts to split into
        output_dir: Directory to save split images (optional)
        
    Returns:
        Tuple of (split_directory_path, list of (split_file_path, offset))
    """
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' was not found.")
        return None, []

    img = read_image_file(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return None, []

    img_height, img_width, _ = img.shape
    
    # Create output directory if not provided
    if output_dir is None:
        base_name = os.path.basename(image_path)
        file_name, _ = os.path.splitext(base_name)
        output_dir = os.path.join(os.path.dirname(image_path), f"split_{file_name}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # For 50 parts, we can do a 5x10 grid
    rows = 5
    cols = 10
    if num_splits != rows * cols:
        # Fallback for different number of splits, find factors
        factors = [i for i in range(1, int(math.sqrt(num_splits)) + 1) if num_splits % i == 0]
        if not factors:
            rows = 1
            cols = num_splits
        else:
            rows = factors[-1]
            cols = num_splits // rows

    tile_width = img_width // cols
    tile_height = img_height // rows
    
    split_files = []
    for r in range(rows):
        for c in range(cols):
            y0 = r * tile_height
            y1 = y0 + tile_height
            x0 = c * tile_width
            x1 = x0 + tile_width
            
            tile = img[y0:y1, x0:x1]
            
            out_path = os.path.join(output_dir, f"split_{r}_{c}.png")
            # Save without compression using save_image
            if save_image(tile, out_path, image_path):
                split_files.append((out_path, (x0, y0)))
            else:
                print(f"Warning: Failed to save split {r}_{c}")
            
    print(f"Split image into {len(split_files)} parts in '{output_dir}'")
    return output_dir, split_files


def detect_overspray(image):
    """
    Detects overspray defects: small scattered ink dots in non-printed areas.
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        List of dictionaries with defect information
    """
    defects = []
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create background mask using Otsu's thresholding
    thresh_val = threshold_otsu(gray)
    background_mask = gray > thresh_val
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    background_mask = morphology.binary_opening(background_mask, kernel)
    
    # Find dark spots in background areas
    background_region = gray.copy()
    background_region[~background_mask] = 255  # Set non-background to white
    
    # Adaptive thresholding to detect dark spots
    adaptive_thresh = cv2.adaptiveThreshold(
        background_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Find contours of potential overspray
    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if 5 < area < 200:  # Filter by size
            # Calculate aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            if 0.3 < aspect_ratio < 3.0:  # Reasonable aspect ratio
                # Calculate center
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Calculate confidence based on contrast
                    roi = gray[max(0, cy-5):cy+5, max(0, cx-5):cx+5]
                    if roi.size > 0:
                        contrast = np.std(roi.astype(np.float32))
                        confidence = min(contrast / 50.0, 1.0)
                        
                        defects.append({
                            'defect_type': 'overspray',
                            'coordinates': (cx, cy),
                            'confidence_score': confidence,
                            'area': area
                        })
    
    return defects


def detect_surface_treatment(image):
    """
    Detects surface treatment defects: mottled appearance within printed stripes.
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        List of dictionaries with defect information
    """
    defects = []
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create stripe mask (dark areas)
    thresh_val = threshold_otsu(gray)
    stripe_mask = gray < thresh_val
    
    # Clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    stripe_mask = morphology.binary_closing(stripe_mask, kernel)
    
    # Sliding window texture analysis
    window_size = 20
    step_size = 10
    
    height, width = gray.shape
    
    for y in range(0, height - window_size, step_size):
        for x in range(0, width - window_size, step_size):
            # Check if window is mostly within stripe
            window_mask = stripe_mask[y:y+window_size, x:x+window_size]
            if np.sum(window_mask) < 0.7 * window_size * window_size:
                continue
            
            # Extract window
            window = gray[y:y+window_size, x:x+window_size]
            
            # Calculate texture features
            # Local Binary Pattern
            lbp = feature.local_binary_pattern(window, 8, 1, method='uniform')
            lbp_hist = np.histogram(lbp, bins=10)[0]
            lbp_variance = np.var(lbp_hist)
            
            # Standard deviation as texture measure
            texture_std = np.std(window.astype(np.float32))
            
            # Combine features for mottling detection
            mottling_score = lbp_variance * 0.001 + texture_std * 0.1
            
            if mottling_score > 8.0:  # Threshold for mottling
                center_x = x + window_size // 2
                center_y = y + window_size // 2
                
                confidence = min(mottling_score / 20.0, 1.0)
                
                defects.append({
                    'defect_type': 'surface_treatment',
                    'coordinates': (center_x, center_y),
                    'confidence_score': confidence,
                    'mottling_score': mottling_score
                })
    
    return defects


def detect_debris(image):
    """
    Detects debris defects: dark spots with light halos.
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        List of dictionaries with defect information
    """
    defects = []
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use Difference of Gaussians for blob detection
    sigma1, sigma2 = 1.0, 2.0
    gaussian1 = cv2.GaussianBlur(gray, (0, 0), sigma1)
    gaussian2 = cv2.GaussianBlur(gray, (0, 0), sigma2)
    dog = gaussian1 - gaussian2
    
    # Threshold to find potential debris
    _, binary = cv2.threshold(dog, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if 10 < area < 500:  # Reasonable size for debris
                # Check if this contour has a parent (halo structure)
                parent_idx = hierarchy[i][3]
                
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > 0.3:  # Reasonably circular
                        # Get center
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            # Calculate confidence based on circularity and contrast
                            roi = gray[max(0, cy-10):cy+10, max(0, cx-10):cx+10]
                            if roi.size > 0:
                                contrast = np.std(roi.astype(np.float32))
                                confidence = min((circularity + contrast/100.0) / 2.0, 1.0)
                                
                                defects.append({
                                    'defect_type': 'debris',
                                    'coordinates': (cx, cy),
                                    'confidence_score': confidence,
                                    'circularity': circularity,
                                    'area': area
                                })
    
    return defects


def detect_calibration_error(image):
    """
    Detects calibration errors: sudden horizontal shifts in vertical edges.
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        List of dictionaries with defect information
    """
    defects = []
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find vertical edges using morphological operations
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    vertical_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
    
    # Find contours of vertical edges
    contours, _ = cv2.findContours(vertical_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if len(contour) > 20:  # Need sufficient points for analysis
            # Extract x, y coordinates
            points = contour.reshape(-1, 2)
            points = points[np.argsort(points[:, 1])]  # Sort by y-coordinate
            
            if len(points) > 10:
                # Calculate first derivative of x with respect to y
                x_coords = points[:, 0]
                y_coords = points[:, 1]
                
                # Smooth the x coordinates
                x_smooth = cv2.GaussianBlur(x_coords.reshape(-1, 1), (5, 1), 0).flatten()
                
                # Calculate derivative
                dx_dy = np.diff(x_smooth)
                
                # Find sudden changes (spikes in derivative)
                threshold = np.std(dx_dy) * 3
                spikes = np.where(np.abs(dx_dy) > threshold)[0]
                
                for spike_idx in spikes:
                    if spike_idx < len(y_coords) - 1:
                        spike_y = y_coords[spike_idx]
                        spike_x = x_coords[spike_idx]
                        
                        # Calculate confidence based on spike magnitude
                        spike_magnitude = abs(dx_dy[spike_idx])
                        confidence = float(min(float(spike_magnitude / (threshold * 2)), 1.0))
                        
                        defects.append({
                            'defect_type': 'calibration',
                            'coordinates': (int(spike_x.item()), int(spike_y.item())),
                            'confidence_score': confidence,
                            'spike_magnitude': spike_magnitude
                        })
    
    return defects


def detect_smudge(image):
    """
    Detects smudge defects: blurry, low-contrast extensions from main stripe.
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        List of dictionaries with defect information
    """
    defects = []
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find edges
    edges = cv2.Canny(gray, 50, 150)
    
    # Calculate gradient magnitude
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Sliding window analysis for edge sharpness
    window_size = 15
    step_size = 5
    
    height, width = gray.shape
    
    for y in range(0, height - window_size, step_size):
        for x in range(0, width - window_size, step_size):
            window_edges = edges[y:y+window_size, x:x+window_size]
            window_gradient = gradient_magnitude[y:y+window_size, x:x+window_size]
            
            # Check if there are edges in this window
            if np.sum(window_edges) > 0:
                # Calculate edge sharpness metrics
                avg_gradient = np.mean(window_gradient[window_edges > 0])
                gradient_std = np.std(window_gradient[window_edges > 0])
                
                # Low gradient and high std indicate smudging
                if avg_gradient < 30 and gradient_std > 10:
                    center_x = x + window_size // 2
                    center_y = y + window_size // 2
                    
                    # Calculate confidence (inverse relationship with sharpness)
                    sharpness = avg_gradient / 100.0
                    confidence = max(0.1, 1.0 - sharpness)
                    
                    defects.append({
                        'defect_type': 'smudge',
                        'coordinates': (center_x, center_y),
                        'confidence_score': confidence,
                        'avg_gradient': avg_gradient,
                        'gradient_std': gradient_std
                    })
    
    return defects


def detect_void(image):
    """
    Detects void defects: small circular areas where ink is missing within printed stripes.
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        List of dictionaries with defect information
    """
    defects = []
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create stripe mask (dark areas)
    thresh_val = threshold_otsu(gray)
    stripe_mask = gray < thresh_val
    
    # Clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    stripe_mask = morphology.binary_closing(stripe_mask, kernel)
    
    # Within stripe areas, find bright spots (voids)
    stripe_region = gray.copy()
    stripe_region[~stripe_mask] = 0  # Set non-stripe areas to black
    
    # Invert to make voids dark for easier detection
    inverted = 255 - stripe_region
    
    # Threshold to find bright spots in original (dark spots in inverted)
    _, void_binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Remove small noise
    kernel = np.ones((2, 2), np.uint8)
    void_binary = cv2.morphologyEx(void_binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(void_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if 5 < area < 200:  # Reasonable size for voids
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity > 0.5:  # Reasonably circular
                    # Get center
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Verify it's actually within a stripe
                        if stripe_mask[cy, cx]:
                            # Calculate confidence based on circularity and contrast
                            roi = gray[max(0, cy-5):cy+5, max(0, cx-5):cx+5]
                            if roi.size > 0:
                                contrast = np.std(roi.astype(np.float32))
                                confidence = min((circularity + contrast/100.0) / 2.0, 1.0)
                                
                                defects.append({
                                    'defect_type': 'void',
                                    'coordinates': (cx, cy),
                                    'confidence_score': confidence,
                                    'circularity': circularity,
                                    'area': area
                                })
    
    return defects


def comprehensive_defect_detection(image_path, window_size=20, output_path="result.png", 
                                 original_image_for_drawing=None, offset=(0, 0)):
    """
    Comprehensive defect detection using all six defect detection functions.
    
    Args:
        image_path: Path to input image
        window_size: Size of analysis window for visualization
        output_path: Path for output image
        original_image_for_drawing: Original image to draw on
        offset: Offset for mapping coordinates back to original image
        
    Returns:
        Tuple of (all_defects, output_image)
    """
    # Load image
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' was not found.")
        return [], None

    image = read_image_file(image_path)
    if image is None:
        print(f"Error: Could not read the image from '{image_path}'. Check the file format.")
        return [], None

    print(f"Processing image: {image_path}")

    # Prepare output image
    if original_image_for_drawing is None:
        output_image = image.copy()
    else:
        output_image = original_image_for_drawing

    # Run all defect detection functions
    all_defects = []
    
    # Define colors for different defect types
    defect_colors = {
        'overspray': (0, 255, 255),      # Yellow
        'surface_treatment': (0, 255, 0), # Green
        'debris': (255, 0, 0),           # Blue
        'calibration': (255, 0, 255),    # Magenta
        'smudge': (0, 128, 255),         # Orange
        'void': (255, 255, 0)            # Cyan
    }
    
    detection_functions = [
        detect_overspray,
        detect_surface_treatment,
        detect_debris,
        detect_calibration_error,
        detect_smudge,
        detect_void
    ]
    
    for detect_func in detection_functions:
        try:
            defects = detect_func(image)
            all_defects.extend(defects)
        except Exception as e:
            print(f"Error in {detect_func.__name__}: {str(e)}")
    
    # Draw defects on output image
    for defect in all_defects:
        original_x = defect['coordinates'][0] + offset[0]
        original_y = defect['coordinates'][1] + offset[1]
        
        color = defect_colors.get(defect['defect_type'], (0, 0, 255))
        
        # Draw circle for defect location
        cv2.circle(output_image, (original_x, original_y), 5, color, 2)
        
        # Draw rectangle for context
        cv2.rectangle(output_image, 
                     (original_x - window_size//2, original_y - window_size//2),
                     (original_x + window_size//2, original_y + window_size//2),
                     color, 1)
        
        # Add text label
        label = f"{defect['defect_type'][:4]}: {defect['confidence_score']:.2f}"
        cv2.putText(output_image, label, 
                   (original_x + 10, original_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Group defects by type for reporting
    defect_counts = defaultdict(int)
    for defect in all_defects:
        defect_counts[defect['defect_type']] += 1
    
    # Print summary
    print(f"Found {len(all_defects)} total defects:")
    for defect_type, count in defect_counts.items():
        print(f"  - {defect_type}: {count}")
    
    return all_defects, output_image


def detect_input_type(input_path):
    """
    Detects the input type based on file extension.
    
    Args:
        input_path: Path to the input file
        
    Returns:
        Tuple of (input_type, is_valid) where:
        - input_type: 'tiff', 'image', 'directory', or 'unknown'
        - is_valid: Boolean indicating if the input is valid
    """
    if not os.path.exists(input_path):
        return 'unknown', False
    
    if os.path.isdir(input_path):
        return 'directory', True
    
    if os.path.isfile(input_path):
        file_ext = os.path.splitext(input_path)[1].lower()
        
        if file_ext in ['.tiff', '.tif']:
            return 'tiff', True
        elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']:
            return 'image', True
        else:
            return 'unknown', False
    
    return 'unknown', False


def main(input_path: str, split_large_image: bool = True, analysis_window_size: int = 20) -> None:
    """
    Main function to process images for defect detection.
    
    Args:
        input_path: Path to input image, TIFF file, or directory containing images.
        split_large_image: Whether to split large images for processing, regardless of format.
        analysis_window_size: Size of the window for defect analysis.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input path '{input_path}' does not exist.")
        return
    
    # --- Input Type Detection ---
    input_type, is_valid = detect_input_type(input_path)
    
    if not is_valid:
        print(f"Error: Unsupported file format. Please provide:")
        print("  - TIFF files (.tiff, .tif)")
        print("  - Image files (.png, .jpg, .jpeg, .bmp, .webp)")
        print("  - Directory containing image files")
        return
    
    print(f"Detected input type: {input_type}")
    print(f"Processing: {input_path}")
    print(f"Split large image mode: {'enabled' if split_large_image else 'disabled'}")
    
    all_defects = []
    failed_files = []
    successful_files = []
    
    # --- Handle Directory Input ---
    if input_type == 'directory':
        # Create output directory with prefix
        input_dir_name = os.path.basename(os.path.abspath(input_path))
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(input_path)),
            f"output_{input_dir_name}"
        )
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
        
        # Get all supported image files in directory (ignore subdirectories)
        image_files = []
        for file in os.listdir(input_path):
            file_path = os.path.join(input_path, file)
            if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp')):
                image_files.append(file_path)
        
        if not image_files:
            print(f"No supported image files found in: {input_path}")
            print("Supported formats: .png, .jpg, .jpeg, .tiff, .tif, .bmp, .webp")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image file
        for img_path in tqdm(image_files, desc="Processing images"):
            try:
                file_type, _ = detect_input_type(img_path)
                file_name = os.path.basename(img_path)
                print(f"\nProcessing {file_type} file: {file_name}")
                
                # Generate output path with prefix
                base_name, _ = os.path.splitext(file_name)
                output_file_path = os.path.join(output_dir, f"output_{base_name}.png")
                
                # Check if we should process in chunks
                is_large = process_large_image_in_chunks(img_path)
                if is_large:
                    print(f"Large image detected, using memory-efficient processing")
                
                if split_large_image:
                    # Process with splitting
                    split_dir, split_files_with_offsets = split_image(img_path, num_splits=50)
                    
                    if split_dir: # Only proceed if splitting was successful
                        # Load original image for drawing defects
                        original_image = read_image_file(img_path, file_type)
                        if original_image is not None:
                            output_image = original_image.copy()
                            
                            # Process each split
                            for split_path, offset in split_files_with_offsets:
                                defects, updated_image = comprehensive_defect_detection(
                                    split_path,
                                    analysis_window_size,
                                    original_image_for_drawing=output_image,
                                    offset=offset
                                )
                                all_defects.extend(defects)
                                if updated_image is not None:
                                    output_image = updated_image
                            
                            # Save result
                            if save_image(output_image, output_file_path, img_path):
                                print(f"Saved: {output_file_path}")
                                successful_files.append(file_name)
                            else:
                                print(f"Failed to save: {output_file_path}")
                                failed_files.append((file_name, "Failed to save output"))
                            
                            # Cleanup temporary files
                            cleanup_temp_files(split_files_with_offsets)
                            if split_dir:
                                cleanup_split_directory(split_dir)
                else:
                    # Process without splitting
                    defects, output_image = comprehensive_defect_detection(
                        img_path,
                        analysis_window_size
                    )
                    all_defects.extend(defects)
                    
                    if output_image is not None:
                        if save_image(output_image, output_file_path, img_path):
                            print(f"Saved: {output_file_path}")
                            successful_files.append(file_name)
                        else:
                            print(f"Failed to save: {output_file_path}")
                            failed_files.append((file_name, "Failed to save output"))
            
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
                failed_files.append((file_name, str(e)))
                continue
    
    # --- Handle Single File Input ---
    else:
        try:
            # Generate output path for single file
            input_dir = os.path.dirname(os.path.abspath(input_path))
            file_name = os.path.basename(input_path)
            base_name, _ = os.path.splitext(file_name)
            output_file_path = os.path.join(input_dir, f"output_{base_name}.png")
            
            # Check if we should process in chunks
            is_large = process_large_image_in_chunks(input_path)
            if is_large:
                print(f"Large image detected, using memory-efficient processing")
            
            if split_large_image:
                print(f"Processing {input_type} file with splitting...")
                
                # Split the image
                split_dir, split_files_with_offsets = split_image(input_path, num_splits=50)
                
                if not split_dir: # Check if splitting failed
                    print("Image splitting failed.")
                    return
                
                # Load original image for drawing defects
                original_image = read_image_file(input_path, input_type)
                if original_image is None:
                    print("Could not load original image for defect visualization.")
                    return
                
                output_image = original_image.copy()
                
                # Process each split image
                print(f"Processing {len(split_files_with_offsets)} image segments...")
                for img_path, offset in split_files_with_offsets:
                    defects, updated_image = comprehensive_defect_detection(
                        img_path,
                        analysis_window_size,
                        original_image_for_drawing=output_image,
                        offset=offset
                    )
                    all_defects.extend(defects)
                    if updated_image is not None:
                        output_image = updated_image
                
                # Save the final result
                if save_image(output_image, output_file_path, input_path):
                    print(f"Final analysis result saved to: {output_file_path}")
                    successful_files.append(file_name)
                else:
                    print(f"Failed to save: {output_file_path}")
                    failed_files.append((file_name, "Failed to save output"))
                
                # Cleanup temporary files
                cleanup_temp_files(split_files_with_offsets)
                if split_dir:
                    cleanup_split_directory(split_dir)
            
            else:
                print(f"Processing {input_type} file without splitting...")
                
                # Process without splitting
                defects, output_image = comprehensive_defect_detection(
                    input_path,
                    analysis_window_size
                )
                all_defects.extend(defects)
                
                if output_image is not None:
                    if save_image(output_image, output_file_path, input_path):
                        print(f"Analysis result saved to: {output_file_path}")
                        successful_files.append(file_name)
                    else:
                        print(f"Failed to save: {output_file_path}")
                        failed_files.append((file_name, "Failed to save output"))
        
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            failed_files.append((file_name, str(e)))
    
    # --- Final Report ---
    print(f"\n=== Processing Summary ===")
    print(f"Successfully processed: {len(successful_files)} files")
    if failed_files:
        print(f"Failed to process: {len(failed_files)} files")
        print("\nFailed files:")
        for filename, error in failed_files:
            print(f"  - {filename}: {error}")
    
    if all_defects:
        print(f"\nFound {len(all_defects)} total defective area(s).")
        
        # Group by defect type for final summary
        defect_summary = defaultdict(int)
        for defect in all_defects:
            defect_summary[defect['defect_type']] += 1
        
        print("\nDefect Summary:")
        for defect_type, count in defect_summary.items():
            print(f"  - {defect_type.title()}: {count}")
    else:
        print("\nNo defects were found matching the criteria.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Nike Defect Detection - Analyze TIFF and image files for manufacturing defects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single TIFF file with splitting
  python error_detection.py /path/to/image.tiff

  # Process a single image without splitting
  python error_detection.py /path/to/image.png --no-split

  # Process all images in a directory
  python error_detection.py /path/to/images/

  # Process with custom window size
  python error_detection.py /path/to/image.tiff --window-size 30
        """
    )
    
    parser.add_argument(
        'input_path',
        help='Path to input image file, TIFF file, or directory containing images'
    )
    
    parser.add_argument(
        '--no-split',
        action='store_true',
        help='Process images without splitting them into segments (default: split large images)'
    )
    
    parser.add_argument(
        '--window-size',
        type=int,
        default=20,
        help='Size of analysis window for defect detection (default: 20)'
    )
    
    return parser.parse_args()


def cleanup_temp_files(split_files: list) -> None:
    """Clean up temporary split files after processing."""
    for file_path, _ in split_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Warning: Could not remove temporary file {file_path}: {e}")

def cleanup_split_directory(split_dir: str) -> None:
    """Remove the entire split directory and its contents."""
    try:
        import shutil
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
            print(f"Cleaned up temporary directory: {split_dir}")
    except Exception as e:
        print(f"Warning: Could not remove temporary directory {split_dir}: {e}")

def process_large_image_in_chunks(image_path: str, chunk_size: int = 1024) -> bool:
    """Check if image should be processed in chunks based on size."""
    try:
        with tifffile.TiffFile(image_path) as tif:
            page = tif.pages[0]
            height, width = page.shape[:2]
            # If image is larger than 10000x10000 pixels, consider it large
            return height > 10000 or width > 10000
    except:
        return False

if __name__ == "__main__":
    args = parse_arguments()
    
    # Convert --no-split to split_large_image boolean
    split_large_image = not args.no_split
    
    try:
        main(
            input_path=args.input_path,
            split_large_image=split_large_image,
            analysis_window_size=args.window_size
        )
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
