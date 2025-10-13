"""
Refactored Defect Detection Pipeline - Main Entry Point

This is the new, clean main script that replaces the massive run_all_detections.py.
It uses the refactored modular architecture with SOLID principles.

Key improvements:
- Clean separation of concerns
- Modular design with clear responsibilities
- Type safety and proper error handling
- Optimized shared preprocessing
- Advanced clustering and visualization
- Comprehensive timing analysis

Usage:
    python run_all_detections_refactored.py --input /path/to/images
    python run_all_detections_refactored.py --input /path/to/single_image.tiff --detectors debris void

Author: Refactored Architecture Team
Date: 2024
Version: 2.0 - Clean Architecture Implementation
"""

import os
import sys
import argparse
from pathlib import Path

# Add the refactored modules to the path
current_dir = Path(__file__).parent
refactored_dir = current_dir / "refactored"
sys.path.insert(0, str(refactored_dir))

# Import the clean, modular components
try:
    from refactored.data_models import ProcessingConfig
    from refactored.pipeline import DefectDetectionPipeline
except ImportError as e:
    print(f"Error importing refactored modules: {e}")
    print("Please ensure all refactored modules are present in the 'refactored' directory")
    sys.exit(1)


class CLI:
    """Command Line Interface handler - Single Responsibility Principle"""
    
    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        """Create and configure the argument parser"""
        parser = argparse.ArgumentParser(
            description='🚀 Refactored Defect Detection Pipeline v2.0\n'
                       'Clean architecture with modular design and optimized performance',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s --input ./test_images                     # Process all images in folder
  %(prog)s --input ./image.tiff                      # Process single image
  %(prog)s --input ./images --detectors debris void  # Run specific detectors
  %(prog)s --input ./images --sensitivity high       # High sensitivity detection
  %(prog)s --input ./images --no_parallel            # Disable parallel processing
  %(prog)s --input ./images --generate_report        # Generate PDF report
            """)
        
        # Required arguments
        parser.add_argument('--input', required=True, metavar='PATH',
                           help='Path to folder containing images or single image file')
        
        # Optional configuration
        parser.add_argument('--output', '-o', metavar='DIR',
                           help='Output directory (default: creates timestamped folder in input location)')
        
        parser.add_argument('--sensitivity', choices=['low', 'medium', 'high'], default='medium',
                           help='Detection sensitivity level (default: %(default)s)')
        
        parser.add_argument('--detectors', nargs='+', metavar='DETECTOR',
                           choices=['debris', 'smudge', 'void', 'head_calibration', 'surface_treatment'],
                           help='Select specific detectors to run. Choices: %(choices)s')
        
        # Processing options
        parser.add_argument('--no_parallel', action='store_true',
                           help='Disable parallel processing (use sequential processing)')
        
        parser.add_argument('--max_workers', type=int, default=4, metavar='N',
                           help='Maximum number of parallel workers (default: %(default)s)')
        
        # Output options
        parser.add_argument('--generate_report', action='store_true',
                           help='Generate comprehensive PDF report')
        
        parser.add_argument('--individual_visualizations', action='store_true',
                           help='Save individual detector visualizations (in addition to combined)')
        
        parser.add_argument('--quiet', action='store_true',
                           help='Disable verbose timing and progress output')
        
        return parser
    
    @staticmethod
    def validate_args(args: argparse.Namespace) -> None:
        """Validate command line arguments"""
        # Check input path exists
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input path does not exist: {args.input}")
        
        # Validate worker count
        if args.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        
        # Warn about parallel processing with single file
        if os.path.isfile(args.input) and not args.no_parallel:
            if not args.quiet:
                print("ℹ️  Note: Parallel processing disabled for single file input")


class ConfigBuilder:
    """Configuration builder - Builder Pattern"""
    
    @staticmethod
    def build_config(args: argparse.Namespace) -> ProcessingConfig:
        """Build processing configuration from command line arguments"""
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


class ConfigurationDisplay:
    """Display configuration information - Single Responsibility"""
    
    @staticmethod
    def show_header():
        """Display application header"""
        print("🔬 Defect Detection Pipeline v2.0")
        print("=" * 50)
        print("🧹 Clean Architecture | 🚀 Optimized Performance | 📊 Advanced Analytics")
        print()
    
    @staticmethod
    def show_input_info(input_path: str):
        """Display input information"""
        if os.path.isfile(input_path):
            file_size = os.path.getsize(input_path) / (1024 * 1024)
            print(f"📁 Input: Single file")
            print(f"   Path: {input_path}")
            print(f"   Size: {file_size:.1f} MB")
        else:
            # Count image files
            image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')
            image_files = [f for f in os.listdir(input_path) 
                          if f.lower().endswith(image_extensions)]
            print(f"📁 Input: Folder with {len(image_files)} images")
            print(f"   Path: {input_path}")
    
    @staticmethod
    def show_detector_routing(selected_detectors):
        """Display detector routing information"""
        print("🎯 Detection Strategy:")
        if selected_detectors:
            print(f"   Selected: {', '.join(selected_detectors)}")
        else:
            print("   Stripe images: Debris, Smudge, Void, Head Calibration")
            print("   Island images: Surface Treatment (if selected)")
            print("   Unknown images: Debris, Smudge, Void, Head Calibration")
    
    @staticmethod
    def show_optimization_info(config: ProcessingConfig):
        """Display optimization settings"""
        print("⚡ Optimizations:")
        print(f"   Shared preprocessing: ✅ Enabled")
        print(f"   Combined visualizations: ✅ Enabled")
        print(f"   Parallel processing: {'✅ Enabled' if config.enable_parallel_processing else '❌ Disabled'}")
        if config.enable_parallel_processing:
            print(f"   Max workers: {config.max_workers}")
        print(f"   Timing analysis: {'✅ Enabled' if config.verbose_timing else '❌ Disabled'}")


def main():
    """
    Main entry point - Orchestrates the entire pipeline
    
    This function only handles:
    1. CLI argument parsing and validation
    2. Configuration creation
    3. Pipeline orchestration
    4. Error handling and user feedback
    
    All actual detection work is delegated to the modular components.
    """
    try:
        # Parse and validate command line arguments
        parser = CLI.create_parser()
        args = parser.parse_args()
        CLI.validate_args(args)
        
        # Build configuration
        config = ConfigBuilder.build_config(args)
        
        # Display configuration (unless quiet mode)
        if not args.quiet:
            ConfigurationDisplay.show_header()
            ConfigurationDisplay.show_input_info(args.input)
            ConfigurationDisplay.show_detector_routing(args.detectors)
            ConfigurationDisplay.show_optimization_info(config)
            print("=" * 50)
        
        # Create and execute pipeline - This is where all the work happens
        pipeline = DefectDetectionPipeline(config)
        pipeline.process_input(args.input)
        
        # Success message
        if not args.quiet:
            print("\n🎉 Pipeline execution completed successfully!")
            print("📊 Check the output directory for detailed results and visualizations")
    
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        sys.exit(1)
    
    except ValueError as e:
        print(f"❌ Invalid argument: {e}")
        sys.exit(1)
    
    except ImportError as e:
        print(f"❌ Module import error: {e}")
        print("Please ensure all dependencies are installed and the refactored modules are present")
        sys.exit(1)
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if not args.quiet if 'args' in locals() else True:
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 