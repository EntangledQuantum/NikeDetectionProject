# Print Defect Detection System

A comprehensive suite of computer vision algorithms for detecting various printing defects in scanned images, optimized for handling extremely large TIFF files.

## Overview

This system detects multiple types of printing defects including:
- **Overspray**: Scattered ink dots outside intended areas
- **Surface Treatment Issues**: Irregular ink drops and missing ink areas due to poor surface energy
- **Debris**: Foreign particles on sheets (pre-print and post-print)
- **Edge Defects**: Irregularities and jaggedness along printed boundaries
- **Banding**: Horizontal and vertical periodic patterns
- **Streaks**: Linear marks and lines in prints

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
- Detects scattered ink dots outside main printed areas
- Uses morphological operations and connected component analysis
- Parameters: dot size range, proximity threshold

### 2. Surface Treatment Detection (`surface_treatment_detection.py`)
- Identifies areas with irregular ink coalescence
- Detects voids and missing ink regions
- Uses local contrast analysis and texture features

### 3. Debris Detection (`debris_detection.py`)
- Detects foreign particles on sheets
- Distinguishes pre-print vs post-print debris
- Identifies characteristic halo patterns around debris



### 4. Edge Defect Detection (`edge_defect_detection.py`)
- Analyzes edge smoothness and regularity
- Detects jagged edges and boundary irregularities
- Uses contour analysis and edge deviation metrics

### 5. Banding Detection (`banding_detection.py`)
- Detects periodic horizontal/vertical patterns
- Uses FFT and autocorrelation analysis
- Identifies band frequency and strength

### 6. Streak Detection (`streak_detection.py`)
- Finds linear marks and streaks
- Uses Hough transform and morphological operations
- Detects streak angle, length, and contrast

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