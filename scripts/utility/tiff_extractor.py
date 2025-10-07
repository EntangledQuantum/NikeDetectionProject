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


def normalize_bbox(bbox: Dict[str, float]) -> Tuple[int, int, int, int]:
    """
    Normalize bounding box coordinates to positive integers in correct order.
    
    Args:
        bbox: Dictionary with top_x, top_y, bottom_x, bottom_y keys
        
    Returns:
        Tuple of (x1, y1, x2, y2) where x1 < x2 and y1 < y2
    """
    x1 = int(min(abs(bbox['top_x']), abs(bbox['bottom_x'])))
    y1 = int(min(abs(bbox['top_y']), abs(bbox['bottom_y'])))
    x2 = int(max(abs(bbox['top_x']), abs(bbox['bottom_x'])))
    y2 = int(max(abs(bbox['top_y']), abs(bbox['bottom_y'])))
    return x1, y1, x2, y2


def check_intersection(sub_bbox: Tuple[int, int, int, int], 
                       exclusion_bbox: Tuple[int, int, int, int]) -> bool:
    """
    Check if two bounding boxes intersect.
    
    Args:
        sub_bbox: (x1, y1, x2, y2) of sub-image
        exclusion_bbox: (x1, y1, x2, y2) of exclusion zone
        
    Returns:
        True if boxes intersect, False otherwise
    """
    sub_x1, sub_y1, sub_x2, sub_y2 = sub_bbox
    ex_x1, ex_y1, ex_x2, ex_y2 = exclusion_bbox
    
    # Check if boxes don't overlap (then return False)
    if sub_x2 < ex_x1 or ex_x2 < sub_x1:
        return False
    if sub_y2 < ex_y1 or ex_y2 < sub_y1:
        return False
    
    return True


def convert_exclusion_to_local(sub_bbox: Tuple[int, int, int, int],
                               exclusion_bbox: Tuple[int, int, int, int],
                               exclusion_name: str) -> Dict:
    """
    Convert global exclusion zone coordinates to local sub-image coordinates.
    Clips the exclusion zone to the sub-image boundaries.
    
    Args:
        sub_bbox: (x1, y1, x2, y2) of sub-image in global coordinates
        exclusion_bbox: (x1, y1, x2, y2) of exclusion zone in global coordinates
        exclusion_name: Name of the exclusion zone
        
    Returns:
        Dictionary with local exclusion zone definition, or None if no overlap
    """
    sub_x1, sub_y1, sub_x2, sub_y2 = sub_bbox
    ex_x1, ex_y1, ex_x2, ex_y2 = exclusion_bbox
    
    # Check if they intersect
    if not check_intersection(sub_bbox, exclusion_bbox):
        return None
    
    # Calculate intersection (clipped to sub-image boundaries)
    local_x1 = max(0, ex_x1 - sub_x1)
    local_y1 = max(0, ex_y1 - sub_y1)
    local_x2 = min(sub_x2 - sub_x1, ex_x2 - sub_x1)
    local_y2 = min(sub_y2 - sub_y1, ex_y2 - sub_y1)
    
    # Ensure valid bounds
    local_x1 = max(0, local_x1)
    local_y1 = max(0, local_y1)
    local_x2 = max(local_x1, local_x2)
    local_y2 = max(local_y1, local_y2)
    
    return {
        "name": exclusion_name,
        "bounding_box_pixels": {
            "top_x": local_x1,
            "top_y": local_y1,
            "bottom_x": local_x2,
            "bottom_y": local_y2
        }
    }


def process_tiff_with_config(config: Union[str, Dict]) -> Dict[str, bool]:
    """
    Process a TIFF file based on configuration containing bounding boxes.
    Also generates per-image exclusion zone JSON files if exclusion zones are defined.
    
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
    exclusion_zones = config_data.get('exclusion_zones', [])
    origin_x = float(config_data.get('origin_x'))
    origin_y = float(config_data.get('origin_y'))
    offsets_path = config_data.get('offset_path')

    sub_images = localize_sub_images(origin_x, origin_y, sub_images, offsets_path)
    exclusion_zones = localize_exclusion_zones(origin_x, origin_y, exclusion_zones, offsets_path)

    
    if not original_image_path or not os.path.exists(original_image_path):
        logger.error(f"Original image not found: {original_image_path}")
        return {}
    
    # Create output directory
    image_name = Path(original_image_path).stem
    output_dir = Path(original_image_path).parent / f"{image_name}_output"
    output_dir.mkdir(exist_ok=True)
    
    logger.info(f"Processing {len(sub_images)} sub-images from {original_image_path}")
    if exclusion_zones:
        logger.info(f"Found {len(exclusion_zones)} global exclusion zones to process")
    logger.info(f"Output directory: {output_dir}")
    
    results = {}
    
    # Normalize all exclusion zones to global coordinates once
    normalized_exclusions = []
    if exclusion_zones:
        for ez in exclusion_zones:
            ez_name = ez.get('name', 'unnamed_exclusion')
            ez_bbox = ez.get('bounding_box_pixels', {})
            if ez_bbox:
                normalized_exclusions.append({
                    'name': ez_name,
                    'bbox': normalize_bbox(ez_bbox)
                })
    
    # Process each sub-image with progress bar
    for sub_image in tqdm(sub_images, desc="Extracting regions", unit="image"):
        name = sub_image.get('name', 'unnamed')
        bbox = sub_image.get('bounding_box_pixels', {})
        
        # Extract and normalize coordinates
        sub_x1, sub_y1, sub_x2, sub_y2 = normalize_bbox(bbox)
        
        # Create output path
        output_path = output_dir / f"{name}.tiff"
        
        # Extract region
        success = extract_region_from_tiff(
            original_image_path,
            bbox.get('top_x', 0),
            bbox.get('top_y', 0),
            bbox.get('bottom_x', 0),
            bbox.get('bottom_y', 0),
            str(output_path)
        )
        
        results[name] = success
        
        # Process exclusion zones for this sub-image
        if success and normalized_exclusions:
            local_exclusions = []
            sub_bbox = (sub_x1, sub_y1, sub_x2, sub_y2)
            
            for ez in normalized_exclusions:
                local_ez = convert_exclusion_to_local(
                    sub_bbox,
                    ez['bbox'],
                    ez['name']
                )
                if local_ez:
                    local_exclusions.append(local_ez)
            
            # Save exclusion zones JSON if any zones intersect with this sub-image
            if local_exclusions:
                exclusion_json_path = output_dir / f"{name}.json"
                exclusion_data = {
                    "exclusion_zones": local_exclusions
                }
                try:
                    with open(exclusion_json_path, 'w') as f:
                        json.dump(exclusion_data, f, indent=2)
                    logger.info(f"Created exclusion zones file for {name}: {len(local_exclusions)} zones")
                except Exception as e:
                    logger.error(f"Failed to save exclusion zones for {name}: {e}")
    
    return results

def localize_sub_images(x, y, sub_images, offsets_path):
    # TODO: Add comments
    # TODO: Generalize the logic for both sub images and exclusion zones as a single helper function
    obj_list = []

    # Load in the offsets json file
    offset_data = {}
    with open(offsets_path, 'r') as f:
        offset_data = json.load(f)
    sub_image_offsets = offset_data.get('sub_images')

    for sub_image, offsets in zip(sub_images, sub_image_offsets):
        # Convert the x coordinates
        sub_image['bounding_box_pixels']['top_x'] = offsets['bounding_box_pixels']['top_x'] + x
        sub_image['bounding_box_pixels']['bottom_x'] = offsets['bounding_box_pixels']['bottom_x'] + x
        
        # Convert the y coordinates
        sub_image['bounding_box_pixels']['top_y'] = offsets['bounding_box_pixels']['top_y'] - y
        sub_image['bounding_box_pixels']['bottom_y'] = offsets['bounding_box_pixels']['bottom_y'] - y

        # Write to the JSON object
        obj_list.append(sub_image)

    return obj_list

def localize_exclusion_zones(x, y, exclusion_zones, offsets_path):
    # TODO: Add comments
    # TODO: Generalize the logic for both sub images and exclusion zones as a single helper function

    obj_list = []

    # Load in the offsets json file
    offset_data = {}
    with open(offsets_path, 'r') as f:
        offset_data = json.load(f)
    
    ez_offsets = offset_data.get('exclusion_zones')

    for ez, offsets in zip(exclusion_zones, ez_offsets):
        # Convert the x coordinates
        ez['bounding_box_pixels']['top_x'] = offsets['bounding_box_pixels']['top_x'] + x
        ez['bounding_box_pixels']['bottom_x'] = offsets['bounding_box_pixels']['bottom_x'] + x
        
        # Convert the y coordinates
        ez['bounding_box_pixels']['top_y'] = offsets['bounding_box_pixels']['top_y'] - y
        ez['bounding_box_pixels']['bottom_y'] = offsets['bounding_box_pixels']['bottom_y'] - y

        # Write to the JSON object
        obj_list.append(ez)

    return obj_list

def generate_offsets_json(x: int, y: int, json_path: str):
    """
   Generates offsets for sub images and exclusion zones in JSON format given a reference point.  
    
    Args:
        x: x coordinate of the top left scan
        y: y coordinate of the top left scan
        json_path: path of the json file where sub-regions are defined
        
    Returns:
        bool: True if detection completed successfully, False otherwise
    """

    # Load the json file
    try:
        logger.info(f'Opening the file: {json_path}')    
        json_data = {}
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        # Iterate through the sub images and localize to the ROI
        for indx, sub_image in enumerate(json_data['sub_images']):
            # Convert the x coordinates
            sub_image['bounding_box_pixels']['top_x'] = sub_image['bounding_box_pixels']['top_x'] - x
            sub_image['bounding_box_pixels']['bottom_x'] = sub_image['bounding_box_pixels']['bottom_x'] - x
            
            # Convert the y coordinates
            sub_image['bounding_box_pixels']['top_y'] = sub_image['bounding_box_pixels']['top_y'] - y
            sub_image['bounding_box_pixels']['bottom_y'] = sub_image['bounding_box_pixels']['bottom_y'] - y

            # Write to the JSON object
            json_data['sub_images'][indx] = sub_image

        # Iterate through the sub images and localize to the ROI
        for indx, exclusion_zone in enumerate(json_data['exclusion_zones']):
            # Convert the x coordinates
            exclusion_zone['bounding_box_pixels']['top_x'] = exclusion_zone['bounding_box_pixels']['top_x'] - x
            exclusion_zone['bounding_box_pixels']['bottom_x'] = exclusion_zone['bounding_box_pixels']['bottom_x'] - x
            
            # Convert the y coordinates
            exclusion_zone['bounding_box_pixels']['top_y'] = exclusion_zone['bounding_box_pixels']['top_y'] - y
            exclusion_zone['bounding_box_pixels']['bottom_y'] = exclusion_zone['bounding_box_pixels']['bottom_y'] - y

            # Write to the JSON object
            json_data['exclusion_zones'][indx] = exclusion_zone
        
        # TODO: Add check to see if the value falls out of bounds of the image

        # TODO: Add a check to see if its a negative x coordinate

        # TODO: Add a check to see if the values are invalid

        # Create a output path
        folder_path = 'offsets/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Output to JSON file        
        with open(f"{folder_path}/offset.json", "w") as json_file:
            json.dump(json_data, json_file, indent=4)

        # Success!
        return True
    
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON file: {e}")
        return False
    except Exception as e:
        logger.error(f"Error reading configuration: {e}")
        return False


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