"""
Image Saving Utility
Provides a robust function for saving images, automatically handling large image dimensions.
"""

import cv2
import os
import numpy as np

def save_image(output_dir: str, base_name: str, image: np.ndarray, suffix: str) -> str:
    """
    Safely saves an image, automatically switching to TIFF for large images.

    Args:
        output_dir: The directory to save the image in.
        base_name: The base filename for the image.
        image: The image data (numpy array).
        suffix: The suffix to append to the base name (e.g., 'visualization', 'kernel_debug').

    Returns:
        The path to the saved image, or an empty string if saving failed.
    """
    if image is None:
        print(f"    WARNING: Image for '{suffix}' is None. Skipping save.")
        return ""

    h, w = image.shape[:2]
    
    # JPEG has a dimension limit of 65535. Use TIFF for larger images.
    is_large_image = w > 65535 or h > 65535
    file_extension = ".tif" if is_large_image else ".jpg"
    
    # Sanitize suffix to ensure it's a valid part of a filename
    clean_suffix = suffix.replace(' ', '_').replace('.', '')
    output_path = os.path.join(output_dir, f"{base_name}_{clean_suffix}{file_extension}")

    print(f"    Saving '{clean_suffix}' to {os.path.basename(output_path)}. Shape: {image.shape}, Dtype: {image.dtype}")
    
    if is_large_image:
        print(f"    INFO: Image dimensions ({w}x{h}) exceed JPEG limit. Saving as TIFF is required.")

    try:
        image_to_save = image
        
        if file_extension == ".jpg":
            # Convert to uint8 for JPG compatibility if needed
            if image_to_save.dtype != np.uint8:
                 print(f"    Converting image from {image_to_save.dtype} to uint8 for JPEG saving.")
                 if image_to_save.dtype == np.uint16:
                     image_to_save = (image_to_save / 256).astype(np.uint8)
                 elif 'float' in str(image_to_save.dtype).lower():
                     # Handle float images (often in range 0.0-1.0)
                     if np.max(image_to_save) <= 1.0:
                         image_to_save = (image_to_save * 255).astype(np.uint8)
                     else:
                         image_to_save = image_to_save.astype(np.uint8)
                 else:
                     # Fallback conversion for other types
                     image_to_save = image_to_save.astype(np.uint8)
            
            operation_success = cv2.imwrite(output_path, image_to_save, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:  # .tif
            operation_success = cv2.imwrite(output_path, image_to_save)

        if operation_success:
            print(f"    Image saved successfully.")
            return output_path
        else:
            print(f"    !!! FAILED to save image to: {output_path}")
            return ""
            
    except Exception as e:
        print(f"    !!! EXCEPTION while saving image to {output_path}: {e}")
        return ""
