# Print Defect Detection System

A comprehensive suite of computer vision algorithms for detecting various printing defects in scanned images, optimized for handling extremely large TIFF files.

## Overview

This system detects four critical types of printing defects:
- **Overspray**: Ink scattered outside intended areas, appearing as dots trailing printed regions
- **Surface Treatment Issues**: Poor surface energy causing ink to combine into irregular drops, leaving areas with no ink
- **Debris**: Foreign particles (dirt, fibers, etc.) causing dark spots with blank rings or contamination patterns
- **Vertical Line Dislocation**: Displacement or shifting of vertical line edges in stripe images

## Features

- **Optimized for Large Images**: Handles extremely large TIFF files (e.g., 1230×44167, 5163×44228 pixels) efficiently
- **Window-based Processing**: Processes huge images in overlapping windows to minimize memory usage
- **Multithreading**: Parallel processing of image windows for faster detection
- **Memory Efficient**: Uses memory-mapped file reading for TIFF files
- **Smart Visualization**: Creates scaled visualizations for very large images

## Installation

```bash
# Create and activate conda environment
conda create -n tiff_extractor python=3.10
conda activate tiff_extractor

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Run all detection algorithms on a folder of images
conda activate tiff_extractor
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
    │   ├── vertical_line_dislocation_visualization.jpg
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

### 3. Debris Detection (`debris_detection.py`)
- **Purpose**: Finds foreign particles and contamination on the substrate
- **Method**: Detects characteristic halo patterns and dark spots caused by debris
- **Parameters**: Halo detection thresholds, particle size ranges, contrast analysis
- **Output**: Contaminated regions with debris particles and their effects

### 4. Vertical Line Dislocation Detection (`vertical_line_dislocation_detection.py`)
- **Purpose**: Detects displacement or shifting of vertical line edges in stripe images
- **Method**: Uses kernel-based vertical tracking to follow vertical lines and detect deviations from expected position
- **Parameters**: Kernel size, delta X threshold (maximum allowed deviation), search range, sensitivity level
- **Output**: Highlighted regions showing where vertical lines have shifted from their expected position

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

For vertical line dislocation detection in stripe images:

```python
from vertical_line_dislocation_detection import VerticalLineDislocationDetector

# Initialize detector
detector = VerticalLineDislocationDetector(
    delta_x_threshold=15,  # Maximum allowed deviation
    sensitivity='medium',
    debug=False
)

# Process single image
image = cv2.imread('path/to/stripe_image.tiff')
result_img, defects = detector.detect(image)

# Save results
cv2.imwrite('dislocation_result.jpg', result_img)
print(f"Found {len(defects)} vertical line dislocations")
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
    "vertical_line_dislocation": {
      "count": 3,
      "defects": [{"location": [x, y], "start_y": 100, "end_y": 200, "deviation": 12.5}, ...],
      "visualization_path": "output/scan001/vertical_line_dislocation_visualization.jpg"
    },
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