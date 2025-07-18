#!/usr/bin/env python3
"""
Example usage of the TIFF Region Extractor
"""

from tiff_extractor import process_tiff_with_config

# Example 1: Using a dictionary configuration
config_dict = {
    "original_image_path": "F:/DeepLearning/NikePProject/test_images/test_image.tiff",
    "sub_images": [
        {
            "name": "redBox",
            "bounding_box_pixels": {
                "top_x": 42.97521,
                "top_y": -38.26446,
                "bottom_x": 247.69044,
                "bottom_y": -194.04729
            }
        },
        {
            "name": "blue_Box",
            "bounding_box_pixels": {
                "top_x": 515.83093,
                "top_y": -514.77457,
                "bottom_x": 731.38465,
                "bottom_y": -674.09688
            }
        }
    ]
}

# Process the TIFF file
print("Processing TIFF file with configuration...")
results = process_tiff_with_config(config_dict)

# Print results
print("\nExtraction Results:")
for name, success in results.items():
    status = "Success" if success else "Failed"
    print(f"  {name}: {status}")

# Example 2: Using a JSON file path
# results = process_tiff_with_config("path/to/config.json") 