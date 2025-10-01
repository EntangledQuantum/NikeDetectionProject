#!/usr/bin/env python3
"""
TIFF Region Extractor

This script extracts regions from large TIFF files based on bounding box coordinates.
It's designed to handle very large files (25GB+) efficiently by only loading the
required regions into memory.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Union, Tuple
import tifffile
import numpy as np
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_region_from_tiff(
    tiff_path: str,
    top_x: int,
    top_y: int,
    bottom_x: int,
    bottom_y: int,
    output_path: str
) -> bool:
    """
    Extract a specific region from a TIFF file and save it as a new TIFF.
    
    Args:
        tiff_path: Path to the input TIFF file
        top_x: X coordinate of top-left corner
        top_y: Y coordinate of top-left corner
        bottom_x: X coordinate of bottom-right corner
        bottom_y: Y coordinate of bottom-right corner
        output_path: Path where the extracted region will be saved
        
    Returns:
        bool: True if extraction was successful, False otherwise
    """
    try:
        # Convert coordinates to positive values if negative
        # and ensure proper ordering
        x1 = int(min(abs(top_x), abs(bottom_x)))
        y1 = int(min(abs(top_y), abs(bottom_y)))
        x2 = int(max(abs(top_x), abs(bottom_x)))
        y2 = int(max(abs(top_y), abs(bottom_y)))
        
        logger.info(f"Extracting region: ({x1}, {y1}) to ({x2}, {y2})")
        
        # Open the TIFF file using memory mapping for efficiency
        with tifffile.TiffFile(tiff_path) as tif:
            # Get the first page (assuming single-page TIFF)
            page = tif.pages[0]
            
            # Get image metadata
            metadata = {}
            
            # Safely extract metadata
            if hasattr(page, 'photometric'):
                metadata['photometric'] = page.photometric
            
            if hasattr(page, 'compression'):
                metadata['compression'] = page.compression
            
            if hasattr(page, 'planarconfig'):
                metadata['planarconfig'] = page.planarconfig
            
            # Extract resolution metadata
            resolution = None
            resolutionunit = None
            
            if hasattr(page, 'tags'):
                if 'XResolution' in page.tags and 'YResolution' in page.tags:
                    x_res = page.tags['XResolution'].value
                    y_res = page.tags['YResolution'].value
                    resolution = (x_res, y_res)
                
                if 'ResolutionUnit' in page.tags:
                    resolutionunit = page.tags['ResolutionUnit'].value
            
            # Read the entire image or use memmap for large files
            # Check if we can use memmap
            if hasattr(page, 'is_memmappable') and page.is_memmappable:
                # Use memory mapping for efficiency
                full_image = page.asarray(out='memmap')
            else:
                # Load normally
                full_image = page.asarray()
            
            # Extract the region
            region = full_image[y1:y2, x1:x2]
            
            # Copy the region to ensure it's not a view
            region = np.array(region, copy=True)
            
            # Prepare kwargs for imwrite
            imwrite_kwargs = {
                'compression': 'none',  # Use no compression for speed
            }
            
            # Add metadata if available
            if 'photometric' in metadata:
                imwrite_kwargs['photometric'] = metadata['photometric']
            
            if resolution is not None:
                imwrite_kwargs['resolution'] = resolution
            
            if resolutionunit is not None:
                imwrite_kwargs['resolutionunit'] = resolutionunit
            
            # Save the extracted region
            tifffile.imwrite(
                output_path,
                region,
                **imwrite_kwargs
            )
            
            logger.info(f"Successfully saved region to: {output_path}")
            return True
            
    except Exception as e:
        logger.error(f"Error extracting region: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def process_tiff_with_config(config: Union[str, Dict]) -> Dict[str, bool]:
    """
    Process a TIFF file based on configuration containing bounding boxes.
    
    Args:
        config: Either a path to JSON file or a dictionary with configuration
        
    Returns:
        Dict: Dictionary with extraction results for each sub-image
    """
    # Parse configuration
    if isinstance(config, str):
        if os.path.isfile(config):
            # It's a file path
            with open(config, 'r') as f:
                config_data = json.load(f)
        else:
            # Try to parse as JSON string
            try:
                config_data = json.loads(config)
            except json.JSONDecodeError:
                logger.error("Invalid JSON string provided")
                return {}
    else:
        config_data = config
    
    # Extract configuration values
    original_image_path = config_data.get('original_image_path')
    sub_images = config_data.get('sub_images', [])
    
    if not original_image_path or not os.path.exists(original_image_path):
        logger.error(f"Original image not found: {original_image_path}")
        return {}
    
    # Create output directory
    image_name = Path(original_image_path).stem
    output_dir = Path(original_image_path).parent / f"{image_name}_output"
    output_dir.mkdir(exist_ok=True)
    
    logger.info(f"Processing {len(sub_images)} sub-images from {original_image_path}")
    logger.info(f"Output directory: {output_dir}")
    
    results = {}
    
    # Process each sub-image with progress bar
    for sub_image in tqdm(sub_images, desc="Extracting regions", unit="image"):
        name = sub_image.get('name', 'unnamed')
        bbox = sub_image.get('bounding_box_pixels', {})
        
        # Extract coordinates
        top_x = bbox.get('top_x', 0)
        top_y = bbox.get('top_y', 0)
        bottom_x = bbox.get('bottom_x', 0)
        bottom_y = bbox.get('bottom_y', 0)
        
        # Create output path
        output_path = output_dir / f"{name}.tiff"
        
        # Extract region
        success = extract_region_from_tiff(
            original_image_path,
            top_x, top_y,
            bottom_x, bottom_y,
            str(output_path)
        )
        
        results[name] = success
    
    return results


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Extract regions from large TIFF files based on bounding box coordinates"
    )
    parser.add_argument(
        'config',
        help='Path to JSON configuration file or JSON string'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Process the TIFF file
    results = process_tiff_with_config(args.config)
    
    # Print results
    print("\nExtraction Results:")
    print("-" * 50)
    for name, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"{name}: {status}")
    
    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main() 