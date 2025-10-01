# Print Defect Detection System

A comprehensive suite of computer vision algorithms for detecting various printing defects in scanned images, optimized for handling extremely large TIFF files.

## Overview

This system detects three critical types of printing defects:
- **Overspray**: Ink scattered outside intended areas, appearing as dots trailing printed regions
- **Surface Treatment Issues**: Poor surface energy causing ink to combine into irregular drops, leaving areas with no ink
- **Debris**: Foreign particles (dirt, fibers, etc.) causing dark spots with blank rings or contamination patterns

## Features

- **Optimized for Large Images**: Handles extremely large TIFF files (e.g., 1230×44167, 5163×44228 pixels) efficiently
- **Window-based Processing**: Processes huge images in overlapping windows to minimize memory usage
- **Multithreading**: Parallel processing of image windows for faster detection
- **Memory Efficient**: Uses memory-mapped file reading for TIFF files
- **Smart Visualization**: Creates scaled visualizations for very large images

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8 or higher
- See `requirements.txt` for package dependencies

## Usage

### Input Folder Structure

Your input folder should contain the images you want to analyze. Optionally, you can provide per-image exclusion zone definitions:

```
input_folder/
├── image1.tif              # Image to analyze
├── image1.json             # Optional: exclusion zones for image1
├── blueStripe.tiff         # Another image (auto-classified as 'stripe')
├── blueStripe.json         # Optional: exclusion zones for blueStripe
├── island-black-blue.tiff  # Another image (auto-classified as 'island')
└── island-black-blue.json  # Optional: exclusion zones for island image
```

**Important**: 
- Image classification is based on filename patterns ('stripe' → stripe images, 'island' → island images)
- JSON exclusion zone files must have the **exact same name** as the image file (only extension differs)
- If no JSON file is provided, the detector runs without exclusion zones

### Exclusion Zones

**What are exclusion zones?**
Exclusion zones are rectangular regions in your images that should be **ignored during defect detection**. This is useful for:
- **Stamps, watermarks, or labels** that should not be flagged as defects
- **Intentional marks or special characters** in the printed design
- **Edge artifacts** from scanning or image capture
- **Reference marks or registration targets** used in printing

**How to define exclusion zones:**

Create a JSON file with the same name as your image (e.g., `image.tiff` → `image.json`):

```json
{
  "exclusion_zones": [
    {
      "name": "stamp_top_left",
      "bounding_box_pixels": {
        "top_x": 100,
        "top_y": 50,
        "bottom_x": 300,
        "bottom_y": 200
      }
    },
    {
      "name": "watermark_bottom_right",
      "bounding_box_pixels": {
        "top_x": 4000,
        "top_y": 3000,
        "bottom_x": 4500,
        "bottom_y": 3300
      }
    }
  ]
}
```

See `example_exclusion_zones.json` in the project root for a complete example.

**Coordinate System:**
- `top_x`, `top_y`: Top-left corner of the exclusion rectangle (in pixels)
- `bottom_x`, `bottom_y`: Bottom-right corner of the exclusion rectangle (in pixels)
- Origin (0,0) is at the top-left of the image
- Coordinates can be negative (they are converted to absolute values automatically)

**Which detectors use exclusion zones?**
- ✅ **Debris Island Detection**: Ignores debris particles inside exclusion zones
- ✅ **Overspray Island Detection**: Ignores overspray regions inside exclusion zones  
- ✅ **Line Detection** (used internally by island detectors): Excludes line detection kernels in zones
- ❌ Other detectors currently do not support exclusion zones sa of now

### Basic Usage

```bash
# Run all detection algorithms on a folder of images
python scripts/defects_detection/run_all_detections.py --input_folder path/to/images
```

### Command Line Options

```bash
python scripts/defects_detection/run_all_detections.py \
    --input_folder path/to/images \
    --generate_report \
    --sensitivity high
```

Options:
- `--input_folder`: Path to folder containing images to analyze (required)
- `--generate_report`: Generate a PDF report with all detections (optional)
- `--sensitivity`: Detection sensitivity level: low, medium, high (default: medium)

### Output Structure

The system creates an `output` folder within your input folder containing:
```
input_folder/
└── output_YYYYMMDD_HHMMSS/
    ├── image_name/
    │   ├── overspray_visualization.jpg
    │   ├── surface_treatment_visualization.jpg
    │   ├── debris_visualization.jpg
    │   ├── gray_spots_visualization.jpg
    │   ├── edge_defects_visualization.jpg
    │   ├── banding_visualization.jpg
    │   ├── streak_visualization.jpg
    │   └── image_name_results.json
    ├── defect_report.json      # Detailed JSON report
    └── defect_report.pdf       # Visual PDF report (if requested)
```

## Handling Large Images

The system automatically detects large images and processes them efficiently:

- **Automatic Detection**: Images > 50MB are processed using windowed approach
- **Window Size**: Default 2048×2048 pixels with 256 pixel overlap
- **Parallel Processing**: Up to 4 threads process windows simultaneously
- **Memory Optimization**: Only loads required image regions into memory
- **Scaled Visualizations**: Large images get intelligently scaled output visualizations

## Detection Algorithms

### 1. Overspray Detection (`overspray_detection.py`)
- **Purpose**: Detects ink scattered outside intended print areas
- **Method**: Uses kernel-based region analysis to group scattered dots into meaningful regions
- **Parameters**: Region size (50-1000 pixels), proximity to main print areas, morphological kernels
- **Output**: Highlighted regions showing where ink has scattered beyond intended boundaries

### 2. Surface Treatment Detection (`surface_treatment_detection.py`)
- **Purpose**: Identifies poor surface energy causing irregular ink behavior
- **Method**: Detects high-contrast ink drops and void areas where ink is missing
- **Parameters**: Contrast thresholds, void size limits, coalescence detection
- **Output**: Regions showing irregular ink drops and missing ink areas

### 3. Debris Detection (`debris_island_detection.py`)
- **Purpose**: Finds foreign particles and contamination on the substrate in island images
- **Method**: Removes slanted lines first, then detects dark debris using thresholding and light morphology
- **Parameters**: Background threshold, debris area limits, morphological kernel sizes
- **Output**: Contaminated regions with debris particles highlighted after line removal

### 4. Line Defect Detection (`line_defect_detection.py`)
- **Purpose**: Detects missing line segments and jagged/zig-zag lines in horizontal line patterns
- **Method**: Kernel-based tracking across scanlines; identifies gaps (missing segments) and large Y deltas (jagged lines)
- **Parameters**: Kernel size, search range, minimum gap size, jagged threshold
- **Output**: Missing line segments highlighted in red, jagged segments in yellow

### 5. Overspray Island Detection (`overspray_island_detection.py`)
- **Purpose**: Detects overspray (scattered ink) in island images after removing intended printed lines
- **Method**: Removes slanted lines using line detector, then detects colored non-white regions and groups them by proximity
- **Parameters**: Background threshold, minimum area, maximum grouping distance, line thickness
- **Output**: Grouped overspray regions highlighted with area and density metrics

### 6. Stripe Misalignment Detection (`stripe_misalignment_detection.py`)
- **Purpose**: Identifies vertical stripe misalignment caused by printer head issues
- **Method**: Enhances vertical edges, scans rows with a kernel to find first strong vertical line, flags lateral X-position shifts
- **Parameters**: Kernel width/height, step size, line detection threshold, defect threshold
- **Output**: Misaligned stripe positions highlighted with X-delta measurements

## Individual Algorithm Usage

Each detection algorithm can also be used standalone:

```python
from overspray_detection import OversprayDetector

# Initialize detector
detector = OversprayDetector(dot_size_range=(3, 15))

# Process single image
image = cv2.imread('path/to/image.png')
result, defects = detector.detect(image)

# Visualize results
visualization = detector.visualize_detections(image, defects)
cv2.imwrite('output.png', visualization)
```

## Sensitivity Levels

- **Low**: Conservative detection, fewer false positives
- **Medium**: Balanced detection (default)
- **High**: Aggressive detection, may include more false positives

## Performance Optimization

### For Large Images
- The system automatically switches to windowed processing for large files
- Adjust window size and overlap in `WindowProcessor` initialization
- Increase `max_workers` for more CPU cores (default: 4)

### Memory Usage
- Typical memory usage: 2-4GB for standard images
- Large image processing: Memory usage stays constant regardless of image size
- Uses memory-mapped file reading for TIFF files

### Processing Speed
- Standard images (< 10MP): 2-5 seconds per image
- Large images (> 50MP): 10-30 seconds depending on size and defect count
- Multithreading provides 2-4x speedup on multi-core systems

## Defect Report Format

The JSON report includes:
```json
{
  "image_name": "scan001.png",
  "timestamp": "2024-01-20T10:30:00",
  "processing_time": "2024-01-20T10:30:15",
  "defects": {
    "overspray": {
      "count": 45,
      "defects": [{"location": [x, y], "size": 5}, ...],
      "visualization_path": "output/scan001/overspray_visualization.jpg"
    },
            "surface_treatment": {...},
        "debris": {...},
        "edge_defects": {...},
    "banding": {...},
    "streak": {...}
  }
}
```

## Troubleshooting

### Common Issues

1. **Memory errors with large images**: The windowed processor should handle this automatically
2. **"No module named cv2"**: Ensure opencv-python is installed
3. **Slow processing**: Increase `max_workers` or reduce `window_size`
4. **False positives**: Adjust sensitivity or algorithm parameters

### Debug Mode

Enable debug output:
```python
detector = OversprayDetector(debug=True)
```

## Examples

See `scripts/example_usage.py` for usage examples.

## Algorithm Parameters

Each algorithm has tunable parameters. See individual algorithm files for detailed documentation of parameters and their effects.

## Contributing

To add new defect detection algorithms:
1. Create a new file in `scripts/defects_detection/`
2. Implement the base detector interface with `detect()` method
3. Add import to `run_all_detections.py`
4. Update this README

## License

This project is proprietary. All rights reserved. 

## Module Reference

- **scripts/defects_detection/run_all_detections.py**: Orchestrator CLI that routes images to the right detectors based on filename patterns (stripe/island/unknown), executes detections, and saves per-image JSON plus a summary report (and optional PDF).
- **scripts/defects_detection/detector_base.py**: Common base with exclusion zone support (`load_exclusion_zones`, `is_point_in_exclusion_zone`, `is_region_in_exclusion_zone`, `draw_exclusion_zones`) and a small adapter to standardize detector outputs.
- **scripts/defects_detection/debris_island_detection.py**: `DebrisIslandDetector` for island images. Removes slanted lines using `utils/line_detector.py`, thresholds for dark debris, applies light morphology, and returns debris regions; supports rich debug artifacts.
- **scripts/defects_detection/overspray_island_detection.py**: `OversprayIslandDetector` for island images. Removes slanted lines, detects colored (non-white) regions below a background threshold, aggressively connects nearby regions, and groups them into overspray shapes.
- **scripts/defects_detection/line_defect_detection.py**: `LineDefectDetector` that tracks horizontal lines to find two defect types: `missing_line` (gaps) and `jagged_line` (large Y deltas). Uses contrast enhancement and adaptive thresholding.
- **scripts/defects_detection/stripe_misalignment_detection.py**: `StripeMisalignmentDetector` for vertical stripe patterns. Enhances vertical edges and scans rows to flag significant X-position shifts as misalignment defects.
- **scripts/defects_detection/overspray_detection.py**: `OversprayDetector` that grid-scans the image and computes a pixel scatter metric per kernel; optionally merges adjacent kernels into larger overspray regions.
- **scripts/defects_detection/surface_treatment_detection.py**: `SurfaceTreatmentDetector` detecting irregular high-contrast drops and missing-ink voids within expected coverage; produces whole-region overlays for easy review.
- **scripts/defects_detection/utils/edge_detector.py**: Helpers for enhanced edge detection with noise reduction and optional CLI usage.
- **scripts/defects_detection/utils/image_saver.py**: Robust image saving that automatically switches to TIFF for very large dimensions; handles dtype conversions safely.
- **scripts/defects_detection/utils/line_detector.py**: Robust slanted line detection for island images; supports dynamic kernel scaling and per-image exclusion zones.
- **scripts/defects_detection/stripe_misalignment_README.md**: Additional notes and tuning tips for stripe misalignment detection.

Note: In earlier documentation you may see references like `debris_detection.py`. The current module name in this repo is `debris_island_detection.py`.