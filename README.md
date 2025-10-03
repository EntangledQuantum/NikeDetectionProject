# Print Defect Detection System

A comprehensive suite of computer vision algorithms for detecting various printing defects in large TIFF scanned images.

## Overview

This system automatically detects critical printing defects in high-resolution scanned images:
- **Overspray**: Ink scattered outside intended areas
- **Surface Treatment Issues**: Poor surface energy causing irregular ink drops and missing ink areas
- **Debris**: Foreign particles causing dark spots and contamination patterns
- **Line Defects**: Missing or jagged horizontal lines
- **Stripe Misalignment**: Vertical stripe positioning errors

## Features

- **Fully Automated Workflow**: Single command extracts regions and runs all detections
- **DPI Template Support**: Built-in configurations for 2400 DPI and 4800 DPI images
- **Exclusion Zones**: Define regions to ignore during detection (stamps, marks, artifacts)
- **Large Image Optimized**: Memory-efficient processing with windowed scanning
- **Comprehensive Reports**: JSON and PDF outputs with visualizations

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8 or higher
- OpenCV, NumPy, scikit-image, matplotlib, tifffile (see `requirements.txt`)

## Quick Start

```bash
# Process a 2400 DPI image with default template
python main_defect_detection.py --image path/to/scan.tif --dpi 2400

# Process a 4800 DPI image with high sensitivity and PDF report
python main_defect_detection.py --image path/to/scan.tif --dpi 4800 --sensitivity high --generate_report
```

## What You Need

### 1. **TIFF Image File** (Required)
Your input must be a **TIFF** (`.tif` or `.tiff`) file containing the full scanned print. The system will automatically extract individual regions based on your DPI template or custom configuration.

**Supported DPI:**
- 2400 DPI
- 4800 DPI

### 2. **DPI Value** (Required)
Specify your image resolution using `--dpi` or `-d`:
- `2400` - Uses the 2400 DPI template configuration
- `4800` - Uses the 4800 DPI template configuration

This tells the system which built-in template to use for extracting stripe and island regions from your full image.

### 3. **Custom Configuration** (Optional)
If you want to override the DPI templates or define your own regions, provide a custom JSON configuration file using `--config` or `-c`.

## Configuration Files

### Using Built-in DPI Templates (Recommended)

The system includes pre-configured templates for standard print layouts:

- **2400 DPI Template**: `regions_json/template-2400-configs.json`
- **4800 DPI Template**: `regions_json/template-4800-configs.json`

These templates define:
- Standard stripe and island region locations
- Example exclusion zones (customize as needed)

**Simply specify `--dpi 2400` or `--dpi 4800` and the appropriate template is used automatically.**

### Custom Configuration (Optional)

Create your own JSON configuration file if you need custom regions or exclusion zones:

```json
{
  "sub_images": [
    {
      "name": "blackStripe",
      "bounding_box_pixels": {
        "top_x": 9129.36,
        "top_y": -902.24,
        "bottom_x": 10359.60,
        "bottom_y": -45069.65
      }
    },
    {
      "name": "island-black-blue",
      "bounding_box_pixels": {
        "top_x": 10574.72,
        "top_y": -786.27,
        "bottom_x": 15737.95,
        "bottom_y": -45014.72
      }
    },
    {
      "name": "blueStripe",
      "bounding_box_pixels": {
        "top_x": 15776.07,
        "top_y": -887.92,
        "bottom_x": 16989.58,
        "bottom_y": -45023.19
      }
    }
  ],
  "exclusion_zones": [
    {
      "name": "stamp_area",
      "bounding_box_pixels": {
        "top_x": 9129.36,
        "top_y": -902.24,
        "bottom_x": 9500.00,
        "bottom_y": -1200.00
      }
    }
  ]
}
```

**Configuration Structure:**

#### `sub_images` (Required)
List of regions to extract from the full TIFF image. Each region must have:
- `name`: Region name (should contain 'stripe' or 'island' for auto-classification)
- `bounding_box_pixels`: Rectangle coordinates in pixels
  - `top_x`, `top_y`: Top-left corner
  - `bottom_x`, `bottom_y`: Bottom-right corner
  - Note: Coordinates can be negative (automatically converted to absolute values)

**Region naming conventions:**
- Names containing **'stripe'** → Runs stripe detectors (overspray, surface treatment, misalignment)
- Names containing **'island'** → Runs island detectors (debris, overspray island, line defects)

#### `exclusion_zones` (Optional)
List of regions to **ignore during defect detection**. Useful for:
- Stamps, watermarks, or labels
- Intentional marks or characters
- Edge artifacts from scanning
- Registration marks or alignment targets

Each exclusion zone has the same structure as sub-images (name + bounding_box_pixels).

**Which detectors support exclusion zones?**
- ✅ Debris Island Detection
- ✅ Overspray Island Detection  
- ✅ Line Detection (internal)
- ❌ Other detectors (coming soon)

**Example templates available in `regions_json/` folder:**
- `template-2400-configs.json` - Standard 2400 DPI layout
- `template-4800-configs.json` - Standard 4800 DPI layout
- `test_Paper_2400 Black pt0.json` - Example custom config
- `test_Paper_4800_BlackPt215.json` - Example custom config

See `example_exclusion_zones.json` in project root for detailed exclusion zone examples.

## Usage

### Command-Line Options

```bash
python main_defect_detection.py [options]
```

**Required Arguments:**
- `--image`, `-i`: Path to TIFF image file
- `--dpi`, `-d`: Image DPI (choices: `2400`, `4800`)

**Optional Arguments:**
- `--config`, `-c`: Path to custom JSON configuration (overrides DPI template)
- `--sensitivity`, `-s`: Detection sensitivity level - `low`, `medium`, `high` (default: `medium`)
- `--generate_report`: Generate PDF report with all detections

### Usage Examples

```bash
# Basic usage with 2400 DPI template
python main_defect_detection.py --image scan.tif --dpi 2400

# With short flags
python main_defect_detection.py -i scan.tif -d 4800

# High sensitivity with PDF report
python main_defect_detection.py -i scan.tif -d 2400 -s high --generate_report

# Custom configuration (overrides DPI template)
python main_defect_detection.py -i scan.tif -d 2400 -c my_custom_regions.json

# Full options
python main_defect_detection.py \
  --image scan.tif \
  --dpi 4800 \
  --config custom.json \
  --sensitivity high \
  --generate_report
```

### What Happens When You Run It

**Step 1: Region Extraction**
- Reads your TIFF image
- Loads DPI template (or custom config if provided)
- Extracts individual stripe and island regions
- Saves extracted regions to timestamped folder: `{image_name}_extracted_regions_YYYYMMDD_HHMMSS/`

**Step 2: Defect Detection**
- Automatically runs appropriate detectors on each extracted region
- Stripe regions → Overspray, Surface Treatment, Stripe Misalignment detectors
- Island regions → Debris, Overspray Island, Line Defect detectors
- Saves visualizations and JSON results for each region
- Generates summary report

## Output Structure

After running, you'll find:

```
image_directory/
└── scan_extracted_regions_20250103_143022/
    ├── blackStripe.tiff                       # Extracted stripe region
    ├── blueStripe.tiff                        # Extracted stripe region
    ├── island-black-blue.tiff                 # Extracted island region
    ├── island-blue-pink.tiff                  # Extracted island region
    ├── pinkStripe.tiff                        # Extracted stripe region
    └── output_20250103_143045/                # Detection results
        ├── blackStripe/
        │   ├── stripe_misalignment_visualization.jpg
        │   ├── overspray_visualization.jpg
        │   ├── surface_treatment_visualization.jpg
        │   └── blackStripe_results.json       # Per-region detection data
        ├── blueStripe/
        │   ├── stripe_misalignment_visualization.jpg
        │   ├── overspray_visualization.jpg
        │   ├── surface_treatment_visualization.jpg
        │   └── blueStripe_results.json
        ├── island-black-blue/
        │   ├── debris_island_visualization.jpg
        │   ├── line_defect_visualization.jpg
        │   ├── overspray_island_visualization.jpg
        │   └── island-black-blue_results.json
        ├── island-blue-pink/
        │   ├── debris_island_visualization.jpg
        │   ├── line_defect_visualization.jpg
        │   ├── overspray_island_visualization.jpg
        │   └── island-blue-pink_results.json
        ├── pinkStripe/
        │   ├── stripe_misalignment_visualization.jpg
        │   ├── overspray_visualization.jpg
        │   ├── surface_treatment_visualization.jpg
        │   └── pinkStripe_results.json
        ├── defect_report.json                 # Summary JSON report
        └── defect_detection_report.pdf        # Summary PDF (if --generate_report used)
```

## Sensitivity Levels

Choose the appropriate sensitivity for your use case:

- **`low`**: Conservative detection, fewer false positives, may miss subtle defects
- **`medium`**: Balanced detection (recommended for most cases)
- **`high`**: Aggressive detection, catches more defects but may include false positives

## Detection Algorithms

### For Stripe Images

#### 1. Stripe Misalignment Detection
- **Purpose**: Identifies vertical stripe positioning errors
- **Method**: Edge enhancement and row scanning to detect lateral shifts
- **Output**: Misaligned regions with X-delta measurements

#### 2. Overspray Detection
- **Purpose**: Detects scattered ink outside intended print areas
- **Method**: Kernel-based grid scanning with scatter analysis
- **Output**: Regions showing ink scattered beyond boundaries

#### 3. Surface Treatment Detection
- **Purpose**: Identifies poor surface energy issues
- **Method**: Detects irregular ink drops and void areas
- **Output**: Areas with ink coalescence and missing ink

### For Island Images

#### 4. Debris Island Detection
- **Purpose**: Finds foreign particles and contamination
- **Method**: Line removal + thresholding + morphology
- **Output**: Contaminated regions with debris highlighted

#### 5. Overspray Island Detection
- **Purpose**: Detects scattered ink in island regions
- **Method**: Line removal + colored region detection + grouping
- **Output**: Grouped overspray regions with metrics

#### 6. Line Defect Detection
- **Purpose**: Detects missing/jagged horizontal lines
- **Method**: Kernel-based line tracking across scanlines
- **Output**: Missing segments (red) and jagged lines (yellow)

## Performance & Optimization

### Large Image Handling
- **Automatic Detection**: Images > 50MB use windowed processing
- **Window Size**: 2048×2048 pixels with 256px overlap
- **Parallel Processing**: Up to 4 threads process windows simultaneously
- **Memory Efficient**: Memory-mapped TIFF reading, constant memory usage

### Typical Processing Times
- Extraction: 5-15 seconds for full TIFF (depends on size)
- Detection per region: 2-10 seconds (depends on size and defect count)
- Full workflow: 30-90 seconds for typical 2400/4800 DPI scans

### Memory Usage
- Extraction: < 1GB
- Detection: 2-4GB for standard regions
- Large images: Memory usage stays constant (windowed processing)

## Troubleshooting

### Common Issues

**"Image must be a TIFF file"**
→ Ensure your input file has `.tif` or `.tiff` extension

**"No module named cv2"**
→ Install dependencies: `pip install -r requirements.txt`

**"'sub_images' list is empty"**
→ Check your JSON configuration file has valid region definitions

**Memory errors**
→ The system should handle this automatically with windowed processing

**Too many false positives**
→ Reduce sensitivity: `--sensitivity low`

**Missing defects**
→ Increase sensitivity: `--sensitivity high`

## Advanced Usage

### Using Individual Detection Scripts

If you already have extracted individual stripe/island images, you can run the detection script directly:

```bash
python scripts/defects_detection/run_all_detections.py --input_folder path/to/extracted_images
```

**What `run_all_detections.py` does:**
- Takes a folder of pre-extracted images (already separated into individual stripes/islands)
- Classifies each image by filename pattern ('stripe' or 'island')
- Runs appropriate detectors on each image
- Saves results to `output_YYYYMMDD_HHMMSS/` folder inside input directory

**Options:**
- `--input_folder`: Path to folder containing images (required)
- `--sensitivity`: Detection sensitivity (low/medium/high, default: medium)
- `--generate_report`: Generate PDF report

**Note:** This is for advanced users who have their own extraction pipeline. Most users should use `main_defect_detection.py`.

## Module Reference

### Core Scripts
- **`main_defect_detection.py`**: Main orchestrator - extracts regions and runs detections automatically
- **`scripts/defects_detection/run_all_detections.py`**: Detection orchestrator - routes images to detectors based on type
- **`scripts/utility/tiff_extractor.py`**: Region extraction from large TIFF files (called by main script)

### Detection Modules
- **`detector_base.py`**: Base class with exclusion zone support
- **`stripe_misalignment_detection.py`**: Vertical stripe alignment detector
- **`overspray_detection.py`**: Scattered ink detector (stripe images)
- **`surface_treatment_detection.py`**: Irregular drops and void detector
- **`debris_island_detection.py`**: Foreign particle detector (island images)
- **`overspray_island_detection.py`**: Scattered ink detector (island images)
- **`line_defect_detection.py`**: Missing/jagged line detector

### Utility Modules
- **`utils/edge_detector.py`**: Enhanced edge detection with noise reduction
- **`utils/image_saver.py`**: Smart image saving (auto-switches to TIFF for large images)
- **`utils/line_detector.py`**: Slanted line detection for island images

## License

This project is proprietary. All rights reserved.
