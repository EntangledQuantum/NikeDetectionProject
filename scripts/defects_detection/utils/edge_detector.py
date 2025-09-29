import cv2
import numpy as np
import os

def detect_edges_enhanced(image_path, output_vertical, output_horizontal, output_combined, 
                         blur_kernel_size, blur_sigma, sobel_kernel_size,
                         use_bilateral_filter, bilateral_d, bilateral_sigma_color, bilateral_sigma_space,
                         use_morphology, morph_kernel_size, use_threshold, threshold_value,
                         use_median_filter, median_kernel_size):
    """
    Enhanced edge detection with advanced noise reduction techniques.
    """
    
    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    # Load the image
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image '{image_path}'. Please check if it's a valid image file.")
        return
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"Image loaded successfully. Size: {gray.shape}")
    
    # Step 1: Advanced noise reduction
    processed = gray.copy()
    
    # Apply median filter first (removes salt-and-pepper noise)
    if use_median_filter:
        processed = cv2.medianBlur(processed, median_kernel_size)
        print(f"Applied median filter with kernel size {median_kernel_size}")
    
    # Apply bilateral filter (preserves edges while reducing noise)
    if use_bilateral_filter:
        processed = cv2.bilateralFilter(processed, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
        print(f"Applied bilateral filter: d={bilateral_d}, sigmaColor={bilateral_sigma_color}, sigmaSpace={bilateral_sigma_space}")
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(processed, blur_kernel_size, blur_sigma)
    print(f"Applied Gaussian blur with kernel {blur_kernel_size} and sigma {blur_sigma}")
    
    # Step 2: Edge detection
    # 1. Vertical Edge Detection
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
    vertical_edges = np.absolute(sobel_x)
    
    # 2. Horizontal Edge Detection
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)
    horizontal_edges = np.absolute(sobel_y)
    
    # 3. Combined Edge Detection
    combined_edges = np.sqrt(sobel_x**2 + sobel_y**2)
    
    print(f"Applied Sobel filters with kernel size {sobel_kernel_size}")
    
    # Step 3: Post-processing to reduce artifacts
    def post_process_edges(edges):
        # Normalize to 0-255
        edges_normalized = np.uint8(edges / edges.max() * 255)
        
        # Apply threshold to remove weak edges (reduces curvy artifacts)
        if use_threshold:
            _, edges_normalized = cv2.threshold(edges_normalized, threshold_value, 255, cv2.THRESH_BINARY)
            print(f"Applied threshold: {threshold_value}")
        
        # Apply morphological operations to clean up edges
        if use_morphology:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
            # Opening: removes small noise
            edges_normalized = cv2.morphologyEx(edges_normalized, cv2.MORPH_OPEN, kernel)
            # Closing: fills small gaps in edges
            edges_normalized = cv2.morphologyEx(edges_normalized, cv2.MORPH_CLOSE, kernel)
            print(f"Applied morphological operations with kernel size {morph_kernel_size}")
        
        return edges_normalized
    
    # Process each edge type
    vertical_final = post_process_edges(vertical_edges)
    horizontal_final = post_process_edges(horizontal_edges)
    combined_final = post_process_edges(combined_edges)
    
    # Save the results
    output_files = {
        output_vertical: vertical_final,
        output_horizontal: horizontal_final,
        output_combined: combined_final
    }
    
    for filename, edge_image in output_files.items():
        cv2.imwrite(filename, edge_image)
        print(f"Saved: {filename}")
    
    print("\nEnhanced edge detection completed successfully!")
    print("Generated files:")
    print(f"- {output_vertical}: Shows vertical edges (vertical lines and boundaries)")
    print(f"- {output_horizontal}: Shows horizontal edges (horizontal lines and boundaries)")
    print(f"- {output_combined}: Shows all edges combined (complete edge map)")

def detect_edges_with_params(image_path, output_vertical, output_horizontal, output_combined, 
                           blur_kernel_size, blur_sigma, sobel_kernel_size):
    """
    Legacy function - calls enhanced version with default noise reduction settings.
    """
    detect_edges_enhanced(image_path, output_vertical, output_horizontal, output_combined,
                         blur_kernel_size, blur_sigma, sobel_kernel_size,
                         True, 9, 75, 75, True, 3, True, 50, True, 5)

def detect_edges(image_path):
    """
    Legacy function for backward compatibility.
    """
    detect_edges_with_params(image_path, "vertical_edges.tiff", "horizontal_edges.tiff", 
                           "combined_edges.tiff", (5, 5), 0, 3)

def main():
    """Main function to run edge detection on blueStripe.tiff"""
    
    # ========== CONFIGURABLE PARAMETERS ==========
    # You can modify these values before running the script
    
    # Input/Output Settings
    input_image = "blueStripe.tiff"              # Input image filename
    output_vertical = "vertical_edges.tiff"      # Vertical edges output filename
    output_horizontal = "horizontal_edges.tiff"  # Horizontal edges output filename
    output_combined = "combined_edges.tiff"      # Combined edges output filename
    
    # Basic Image Processing Parameters
    blur_kernel_size = (15, 15)                 # Gaussian blur kernel size (width, height)
    blur_sigma = 0                               # Gaussian blur sigma (0 = auto-calculate)
    sobel_kernel_size = 5                        # Sobel filter kernel size (3, 5, or 7)
    
    # ===== ADVANCED NOISE REDUCTION PARAMETERS =====
    # These help eliminate curvy artifacts and noise
    
    use_bilateral_filter = False                  # Use bilateral filter (preserves edges while reducing noise)
    bilateral_d = 9                              # Bilateral filter diameter (9-15, larger = more smoothing)
    bilateral_sigma_color = 75                   # Color sigma (50-150, larger = more colors averaged)
    bilateral_sigma_space = 75                   # Space sigma (50-150, larger = more distant pixels averaged)
    
    use_morphology = False                        # Use morphological operations to clean edges
    morph_kernel_size = 7                        # Morphological kernel size (3, 5, or 7)
    
    use_threshold = True                         # Apply threshold to remove weak edges (reduces artifacts!)
    threshold_value = 30                         # Threshold value (20-80, higher = fewer but cleaner edges)
    
    use_median_filter = True                     # Use median filter for additional noise reduction
    median_kernel_size = 51                      # Median filter kernel size (3, 5, 7, 9)
    
    # ============================================
    
    print("=== Enhanced Edge Detection Script ===")
    print("This script uses advanced noise reduction to eliminate curvy artifacts.")
    print(f"\nCurrent Parameters:")
    print(f"  Input image: {input_image}")
    print(f"  Blur kernel: {blur_kernel_size}, sigma: {blur_sigma}")
    print(f"  Sobel kernel size: {sobel_kernel_size}")
    print(f"\nNoise Reduction Settings:")
    print(f"  Bilateral filter: {use_bilateral_filter} (d={bilateral_d})")
    print(f"  Morphology: {use_morphology} (kernel={morph_kernel_size})")
    print(f"  Threshold: {use_threshold} (value={threshold_value})")
    print(f"  Median filter: {use_median_filter} (kernel={median_kernel_size})")
    print(f"\nProcessing...\n")
    
    # Run enhanced edge detection
    detect_edges_enhanced(input_image, output_vertical, output_horizontal, output_combined,
                         blur_kernel_size, blur_sigma, sobel_kernel_size,
                         use_bilateral_filter, bilateral_d, bilateral_sigma_color, bilateral_sigma_space,
                         use_morphology, morph_kernel_size, use_threshold, threshold_value,
                         use_median_filter, median_kernel_size)

if __name__ == "__main__":
    main() 