#!/usr/bin/env python3
"""
Main Defect Detection Orchestrator

This script orchestrates the complete defect detection workflow:
1. Extract regions from a large TIFF image based on JSON configuration
2. Run defect detection on the extracted regions

Usage:
    python main_defect_detection.py --image path/to/image.tif --config path/to/config.json
    python main_defect_detection.py --image path/to/image.tif --config path/to/config.json --sensitivity high
    python main_defect_detection.py --image path/to/image.tif --config path/to/config.json --generate_report
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_step(step_num: int, message: str):
    """Print a step header with formatting."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}STEP {step_num}: {message}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.ENDC}\n")


def print_success(message: str):
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")


def print_error(message: str):
    """Print an error message."""
    print(f"{Colors.RED}✗ {message}{Colors.ENDC}")


def print_info(message: str):
    """Print an info message."""
    print(f"{Colors.BLUE}ℹ {message}{Colors.ENDC}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.ENDC}")


def load_config_for_dpi(dpi: str, custom_config_path: Optional[str] = None) -> Optional[str]:
    """
    Load the appropriate configuration file based on DPI or custom path.
    
    Args:
        dpi: DPI value ('2400' or '4800')
        custom_config_path: Optional path to custom config (overrides DPI template)
        
    Returns:
        str: Path to the configuration file to use, or None if not found
    """
    # If custom config is provided, use it
    if custom_config_path:
        if os.path.exists(custom_config_path):
            print_info(f"Using custom configuration: {custom_config_path}")
            return custom_config_path
        else:
            print_error(f"Custom configuration file not found: {custom_config_path}")
            return None
    
    # Otherwise, use DPI template
    template_map = {
        '2400': 'regions_json/template-2400-configs.json',
        '4800': 'regions_json/template-4800-configs.json'
    }
    
    template_path = Path(__file__).parent / template_map[dpi]
    
    if template_path.exists():
        print_info(f"Using {dpi} DPI template: {template_path}")
        return str(template_path)
    else:
        print_error(f"DPI template not found: {template_path}")
        return None


def validate_inputs(image_path: str, config_path: str) -> bool:
    """
    Validate that required input files exist and are properly formatted.
    
    Args:
        image_path: Path to the TIFF image file
        config_path: Path to the JSON configuration file
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    print_info("Validating input files...")
    
    if not os.path.exists(image_path):
        print_error(f"Image file not found: {image_path}")
        return False
    
    if not os.path.exists(config_path):
        print_error(f"Configuration file not found: {config_path}")
        return False
    
    # Validate it's a TIFF file
    if not image_path.lower().endswith(('.tif', '.tiff')):
        print_error(f"Image must be a TIFF file (.tif or .tiff)")
        return False
    
    # Validate it's a JSON file
    if not config_path.lower().endswith('.json'):
        print_error(f"Configuration must be a JSON file (.json)")
        return False
    
    # Try to parse the JSON
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Validate JSON structure
        if 'sub_images' not in config_data:
            print_error("JSON configuration must contain 'sub_images' field")
            return False
        
        if not isinstance(config_data['sub_images'], list):
            print_error("'sub_images' must be a list")
            return False
        
        if len(config_data['sub_images']) == 0:
            print_error("'sub_images' list is empty")
            return False
        
        print_success(f"Found {len(config_data['sub_images'])} regions to extract")
        
        # Check for exclusion zones (optional)
        if 'exclusion_zones' in config_data and config_data['exclusion_zones']:
            print_info(f"Found {len(config_data['exclusion_zones'])} exclusion zones (will be converted to local coordinates)")
        
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON file: {e}")
        return False
    except Exception as e:
        print_error(f"Error reading configuration: {e}")
        return False
    
    print_success("Input validation passed")
    return True


def extract_regions(image_path: str, config_path: str, origin_coordinates: dict) -> Optional[str]:
    """
    Extract regions from the TIFF image using tiff_extractor.py.
    
    Args:
        image_path: Path to the TIFF image file
        config_path: Path to the JSON configuration file
        origin_coordinates: (x, y) Top-left coordinates of the black stripe 
        
    Returns:
        str: Path to the extraction output directory, or None if failed
    """
    print_step(1, "EXTRACTING REGIONS FROM TIFF IMAGE")
    
    # Read the config and add the image path at runtime
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
    except Exception as e:
        print_error(f"Failed to read configuration: {e}")
        return None
    
    # Add the original_image_path at runtime (not required in JSON)
    # This allows users to have cleaner JSON files with just region definitions
    abs_image_path = os.path.abspath(image_path)
    config_data['original_image_path'] = abs_image_path

    # Add the origin coordinates to the config file
    config_data['origin_x'] = origin_coordinates[0]
    config_data['origin_y'] = origin_coordinates[1]
    
    # Add the offsets json file to config file. TODO: Revamp this part so that it can choose either 2400 dpi or 4800 dpi offset json file based on user input
    offset_path = Path(__file__).parent / 'regions_json/offsets.json'
    
    if offset_path.exists():
        config_data['offset_path'] = str(offset_path)
    else:
        print_error(f"Unable to open the offset JSON file: {offset_path}")
        return None
    
    # Create a temporary config file with updated path
    temp_config_path = config_path + '.tmp'
    try:
        with open(temp_config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    except Exception as e:
        print_error(f"Failed to create temporary config: {e}")
        return None
    
    # Construct the output directory name
    image_name = Path(image_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_name = f"{image_name}_extracted_regions_{timestamp}"
    output_dir = Path(image_path).parent / output_dir_name
    
    print_info(f"Source image: {image_path}")
    print_info(f"Configuration: {config_path}")
    print_info(f"Output directory: {output_dir}")
    print_info(f"Number of regions: {len(config_data['sub_images'])}")
    
    # Run tiff_extractor.py
    extractor_script = Path(__file__).parent / "scripts" / "utility" / "tiff_extractor.py"
    
    if not extractor_script.exists():
        print_error(f"Extractor script not found: {extractor_script}")
        # Clean up temp config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        return None
    
    print_info("Running TIFF region extraction...")
    print(f"{Colors.YELLOW}--- Extraction Logs Start ---{Colors.ENDC}")
    
    try:
        # Run the extractor script
        result = subprocess.run(
            [sys.executable, str(extractor_script), temp_config_path],
            capture_output=False,
            text=True,
            check=True
        )
        
        print(f"{Colors.YELLOW}--- Extraction Logs End ---{Colors.ENDC}\n")
        
        # Check if output directory was created
        expected_output = Path(image_path).parent / f"{image_name}_output"
        
        if expected_output.exists():
            # Rename to our timestamped directory
            expected_output.rename(output_dir)
            print_success(f"Extraction completed successfully")
            print_success(f"Extracted regions saved to: {output_dir}")
            
            # List extracted files
            extracted_files = list(output_dir.glob("*.tiff")) + list(output_dir.glob("*.tif"))
            print_info(f"Extracted {len(extracted_files)} region files:")
            for f in extracted_files:
                print(f"  • {f.name}")
            
            # Clean up temp config
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            
            return str(output_dir)
        else:
            print_error("Extraction output directory not found")
            # Clean up temp config
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"{Colors.YELLOW}--- Extraction Logs End ---{Colors.ENDC}\n")
        print_error(f"Extraction failed with exit code {e.returncode}")
        # Clean up temp config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        return None
    except Exception as e:
        print(f"{Colors.YELLOW}--- Extraction Logs End ---{Colors.ENDC}\n")
        print_error(f"Extraction failed: {e}")
        # Clean up temp config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        return None


def run_defect_detection(extracted_dir: str, sensitivity: str = 'medium', 
                        generate_report: bool = False) -> bool:
    """
    Run defect detection on the extracted regions.
    
    Args:
        extracted_dir: Path to directory containing extracted region images
        sensitivity: Detection sensitivity level ('low', 'medium', 'high')
        generate_report: Whether to generate a PDF report
        
    Returns:
        bool: True if detection completed successfully, False otherwise
    """
    print_step(2, "RUNNING DEFECT DETECTION")
    
    print_info(f"Input folder: {extracted_dir}")
    print_info(f"Sensitivity: {sensitivity}")
    print_info(f"Generate PDF report: {generate_report}")
    
    # Construct the detection script path
    detection_script = Path(__file__).parent / "scripts" / "defects_detection" / "run_all_detections.py"
    
    if not detection_script.exists():
        print_error(f"Detection script not found: {detection_script}")
        return False
    
    # Build command
    cmd = [
        sys.executable,
        str(detection_script),
        '--input_folder', extracted_dir,
        '--sensitivity', sensitivity
    ]
    
    if generate_report:
        cmd.append('--generate_report')
    
    print_info("Starting defect detection pipeline...")
    print(f"{Colors.YELLOW}--- Detection Logs Start ---{Colors.ENDC}\n")
    
    try:
        # Run the detection script
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            check=True,
            cwd=Path(detection_script).parent  # Run from detection script directory
        )
        
        print(f"\n{Colors.YELLOW}--- Detection Logs End ---{Colors.ENDC}")
        print_success("Defect detection completed successfully")
        
        # Find and report the output directory
        output_dirs = list(Path(extracted_dir).glob("output_*"))
        if output_dirs:
            latest_output = max(output_dirs, key=lambda p: p.stat().st_mtime)
            print_success(f"Detection results saved to: {latest_output}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n{Colors.YELLOW}--- Detection Logs End ---{Colors.ENDC}")
        print_error(f"Detection failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n{Colors.YELLOW}--- Detection Logs End ---{Colors.ENDC}")
        print_error(f"Detection failed: {e}")
        return False


def main():
    """Main entry point for the orchestrator script."""
    parser = argparse.ArgumentParser(
        description='Complete defect detection workflow: Extract regions → Detect defects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with DPI template
  python main_defect_detection.py --image test.tif --dpi 2400
  
  # With high sensitivity
  python main_defect_detection.py --image test.tif --dpi 4800 --sensitivity high
  
  # With custom config (overrides DPI template)
  python main_defect_detection.py --image test.tif --dpi 2400 --config custom.json
  
  # Generate PDF report
  python main_defect_detection.py --image test.tif --dpi 2400 --generate_report
  
  # Full options
  python main_defect_detection.py --image test.tif --dpi 4800 --config custom.json --sensitivity high --generate_report
        """
    )
    
    parser.add_argument(
        '--image', '-i',
        required=True,
        help='Path to the TIFF image file to process'
    )
    
    parser.add_argument(
        '--dpi', '-d',
        required=True,
        choices=['2400', '4800'],
        help='Image DPI resolution (required, determines which template to use)'
    )

    parser.add_argument(
        '--origin_x', '-x',
        required=True,
        help='Starting x-coordinate of the top-left black stripe (required)'
    )

    parser.add_argument(
        '--origin_y', '-y',
        required=True,
        help='Starting y-coordinate of the top-left black stripe (required)'
    )
    
    parser.add_argument(
        '--config', '-c',
        required=False,
        help='Path to custom JSON configuration file (optional, overrides DPI template)'
    )
    
    parser.add_argument(
        '--sensitivity', '-s',
        choices=['low', 'medium', 'high'],
        default='medium',
        help='Detection sensitivity level (default: medium)'
    )
    
    parser.add_argument(
        '--generate_report',
        action='store_true',
        help='Generate PDF report with all detections'
    )
    
    args = parser.parse_args()
    
    # Print header
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'DEFECT DETECTION ORCHESTRATOR':^80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}\n")
    
    print_info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_info(f"Image: {args.image}")
    print_info(f"DPI: {args.dpi}")
    
    # Load appropriate configuration (DPI template or custom)
    config_path = load_config_for_dpi(args.dpi, args.config)
    if not config_path:
        print_error("Failed to load configuration. Exiting.")
        sys.exit(1)
    
    # Validate inputs
    if not validate_inputs(args.image, config_path):
        print_error("Input validation failed. Exiting.")
        sys.exit(1)
    
    # Step 1: Extract regions
    extracted_dir = extract_regions(args.image, config_path, (args.origin_x, args.origin_y))
    
    if not extracted_dir:
        print_error("Region extraction failed. Exiting.")
        sys.exit(1)
    
    # Step 2: Run defect detection
    detection_success = run_defect_detection(
        extracted_dir,
        sensitivity=args.sensitivity,
        generate_report=args.generate_report
    )
    
    if not detection_success:
        print_error("Defect detection failed. Exiting.")
        sys.exit(1)
    
    # Final summary
    print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'WORKFLOW COMPLETED SUCCESSFULLY':^80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'='*80}{Colors.ENDC}\n")
    
    print_success(f"Extracted regions directory: {extracted_dir}")
    
    # Find and display output directory
    output_dirs = list(Path(extracted_dir).glob("output_*"))
    if output_dirs:
        latest_output = max(output_dirs, key=lambda p: p.stat().st_mtime)
        print_success(f"Detection results directory: {latest_output}")
    
    print_info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n{Colors.BOLD}Next steps:{Colors.ENDC}")
    print(f"  1. Review the detection visualizations in the output folder")
    print(f"  2. Check the JSON report for detailed defect information")
    if args.generate_report:
        print(f"  3. Open the PDF report for a comprehensive overview")
    
    sys.exit(0)


if __name__ == "__main__":
    main()

