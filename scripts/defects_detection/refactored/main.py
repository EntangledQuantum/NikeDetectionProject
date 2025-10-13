"""
Main Entry Point for Defect Detection Pipeline

This script provides the command-line interface and orchestrates
the execution of the refactored defect detection pipeline.

Author: Refactored Architecture
Date: 2024
"""

import os
import sys
import argparse

# Add the parent directory to the path to import detection modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_models import ProcessingConfig
from pipeline import DefectDetectionPipeline


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser"""
    parser = argparse.ArgumentParser(
        description='Refactored defect detection pipeline with clean architecture and parallel processing'
    )
    
    parser.add_argument('--input', required=True, 
                       help='Path to folder containing images or single image file')
    
    parser.add_argument('--output', '-o', 
                       help='Output base directory (default: creates output folder in input folder)')
    
    parser.add_argument('--generate_report', action='store_true',
                       help='Generate PDF report with all detections')
    
    parser.add_argument('--sensitivity', choices=['low', 'medium', 'high'], default='medium',
                       help='Detection sensitivity level (default: medium)')
    
    parser.add_argument('--detectors', nargs='+', 
                       choices=['debris', 'smudge', 'void', 'head_calibration', 'surface_treatment'],
                       help='Select specific detectors to run (default: all based on image type). '
                            'Example: --detectors smudge void')
    
    parser.add_argument('--no_parallel', action='store_true',
                       help='Disable parallel processing')
    
    parser.add_argument('--max_workers', type=int, default=4,
                       help='Maximum number of parallel workers (default: 4)')
    
    parser.add_argument('--individual_visualizations', action='store_true',
                       help='Save individual detector visualizations')
    
    parser.add_argument('--quiet', action='store_true',
                       help='Disable verbose timing output')
    
    return parser


def validate_input_path(input_path: str) -> None:
    """Validate that the input path exists"""
    if not os.path.exists(input_path):
        print(f"Error: Input path '{input_path}' does not exist")
        sys.exit(1)


def create_processing_config(args: argparse.Namespace) -> ProcessingConfig:
    """Create processing configuration from command line arguments"""
    return ProcessingConfig(
        output_base_dir=args.output,
        sensitivity=args.sensitivity,
        generate_report=args.generate_report,
        enable_parallel_processing=not args.no_parallel,
        max_workers=args.max_workers,
        save_individual_visualizations=args.individual_visualizations,
        verbose_timing=not args.quiet,
        selected_detectors=args.detectors
    )


def print_configuration_summary(args: argparse.Namespace, config: ProcessingConfig) -> None:
    """Print a summary of the configuration and input"""
    # Determine input type and display appropriate message
    if os.path.isfile(args.input):
        print(f"Input file: {args.input}")
        input_type = "Single file"
    elif os.path.isdir(args.input):
        print(f"Input folder: {args.input}")
        input_type = "Folder"
    else:
        print(f"Error: '{args.input}' is neither a file nor directory")
        sys.exit(1)

    print(f"Input type: {input_type}")

    # Show selected detectors
    if args.detectors:
        print(f"🎯 Selected detectors: {', '.join(args.detectors)}")
    else:
        print(f"Detection routing:")
        print(f"  - Stripe images: Debris, Smudge, Void, Head Calibration")
        print(f"  - Island images: Surface Treatment (if selected)")
        print(f"  - Unknown images: Debris, Smudge, Void, Head Calibration")

    print(f"🔧 Optimizations enabled:")
    print(f"  - Parallel processing: {'Yes' if config.enable_parallel_processing else 'No'}")
    if config.enable_parallel_processing:
        print(f"  - Max workers: {config.max_workers}")
    print(f"  - Shared preprocessing: Yes")
    print(f"  - Combined visualizations: Yes")
    print(f"  - Individual visualizations: {'Yes' if config.save_individual_visualizations else 'No'}")
    print(f"  - Detailed timing analysis: {'Yes' if config.verbose_timing else 'No'}")
    print("=" * 60)


def main():
    """Main entry point"""
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate input path
    validate_input_path(args.input)
    
    # Create configuration
    config = create_processing_config(args)
    
    # Print configuration summary
    print_configuration_summary(args, config)
    
    # Create and run pipeline
    pipeline = DefectDetectionPipeline(config)
    pipeline.process_input(args.input)


if __name__ == "__main__":
    main() 