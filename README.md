# Print Defect Detection System

A comprehensive suite of computer vision algorithms for detecting various printing defects in large TIFF scanned images.

## Overview

This system automatically detects critical printing defects in high-resolution scanned images:

**Island regions**
- **Debris**: Foreign particles causing dark spots and contamination
- **Overspray**: Ink scattered outside intended line areas
- **Line Defects**: Missing nozzles (gaps) and misaligned / doubled prints

**Stripe regions**
- **Stripe Misalignment**: Stitch steps and roll drift between successive print heads
- **Overspray**: Scattered ink outside the stripe
- **Surface Treatment Issues**: Irregular ink drops and missing ink areas
- **Voids**: Compact low-ink / missing-ink patches inside the solid stripe
- **Debris Stripe**: Dark debris spots inside the colored stripe

## Features

- **Fully Automated Workflow**: Single command extracts regions and runs all detections
- **Standalone region testing**: Run detectors on a single island or stripe crop without re-extracting
- **Legacy + new island patterns**: Single-band (`--pattern legacy`) or dual-band with 4 vertical lines (`--pattern new`)
- **Clear scan material**: `--clear` adapts island thresholds for gray paper / fainter ink (requires `--pattern new`)
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
# Legacy layout (default): extract + detect with 2400 DPI template
python main_defect_detection.py --image path/to/scan.tif --dpi 2400

# New dual-band island pattern (4 vertical boundary lines → 2 print bands)
python main_defect_detection.py --image path/to/scan.tif --dpi 2400 --pattern new

# Clear scan material (gray background, fainter ink) — requires --pattern new
python main_defect_detection.py --image path/to/scan.tif --dpi 2400 --pattern new --clear

# High sensitivity + PDF report
python main_defect_detection.py --image path/to/scan.tif --dpi 4800 --sensitivity high --generate_report
```

### Quick test of one island or stripe crop

If you already have an extracted region image, skip the full scan and run detectors directly.
Filenames must contain `island` or `stripe` so the correct detector stack is chosen:

```bash
cd scripts/defects_detection

# New-pattern island (all island detectors)
python run_all_detections.py -i "C:\path\KeyIsland.tiff" --pattern new

# Island: missing nozzles / line misalignment only
python run_all_detections.py -i "C:\path\KeyIsland.tiff" --pattern new --only line_defect

# Clear-material island
python run_all_detections.py -i "C:\path\ClearIsland.tiff" --pattern new --clear

# Stripe: all stripe detectors
python run_all_detections.py -i "C:\path\CyanStripe.tiff"

# Stripe: stitch/roll calibration only
python run_all_detections.py -i "C:\path\CyanStripe.tiff" --only stripe_misalignment

# Stripe: voids only
python run_all_detections.py -i "C:\path\CyanStripe.tiff" --only void
```

See [`scripts/defects_detection/RUN_ALL_DETECTIONS_README.md`](scripts/defects_detection/RUN_ALL_DETECTIONS_README.md) for the full standalone CLI.

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
- Names containing **'stripe'** → Stripe detectors: misalignment (stitch/roll), overspray, surface treatment, void, debris stripe
- Names containing **'island'** → Island detectors: debris, overspray island, line defects

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
- `template-2400-configs.json` - Standard 2400 DPI layout (legacy)
- `template-4800-configs.json` - Standard 4800 DPI layout (legacy)
- `new_pattern_2400.json` - New dual-band layout (3 heads; used with `--pattern new`)
- `test_Paper_2400 Black pt0.json` - Example custom config
- `test_Paper_4800_BlackPt215.json` - Example custom config

See `example_exclusion_zones.json` in project root for detailed exclusion zone examples.

## Usage

### Command-Line Options (`main_defect_detection.py`)

```bash
python main_defect_detection.py [options]
```

**Required:**
- `--image`, `-i`: Path to TIFF image file
- `--dpi`, `-d`: Image DPI (`2400` or `4800`)

**Optional:**
- `--config`, `-c`: Custom JSON configuration (overrides the DPI template)
- `--sensitivity`, `-s`: `low` \| `medium` \| `high` (default: `medium`)
- `--pattern`: `legacy` (single-band islands, default) \| `new` (dual-band islands with 4 vertical boundary lines)
- `--clear`: Clear scan material (gray background, fainter ink, lower SNR). Island thresholds are derived from the measured background level per image. **Requires `--pattern new`.**
- `--generate_report`: Generate PDF report with all detections

### Usage Examples

```bash
# Basic — legacy pattern, 2400 DPI template
python main_defect_detection.py --image scan.tif --dpi 2400

# Short flags
python main_defect_detection.py -i scan.tif -d 4800

# New dual-band island pattern
python main_defect_detection.py -i scan.tif -d 2400 --pattern new

# Clear material + new pattern
python main_defect_detection.py -i scan.tif -d 2400 --pattern new --clear

# High sensitivity with PDF report
python main_defect_detection.py -i scan.tif -d 2400 -s high --generate_report

# Custom regions config (overrides DPI template)
python main_defect_detection.py -i scan.tif -d 2400 -c my_custom_regions.json

# New-pattern config is selected automatically when --pattern new
# (e.g. regions_json/new_pattern_2400.json for 2400 DPI)

# Full options
python main_defect_detection.py \
  --image scan.tif \
  --dpi 2400 \
  --pattern new \
  --clear \
  --sensitivity high \
  --generate_report
```

### What Happens When You Run It

**Step 1: Region Extraction**
- Reads your TIFF image
- Loads DPI template / new-pattern config / custom config
- Extracts individual stripe and island regions
- Saves extracted regions to: `{image_name}_extracted_regions_YYYYMMDD_HHMMSS/`

**Step 2: Defect Detection**
- Routes each extracted region by filename (`island` / `stripe`)
- Island regions → debris, overspray island, line defect
  (dual-band detectors when `--pattern new`; adaptive thresholds when `--clear`)
- Stripe regions → stripe misalignment (stitch/roll), overspray, surface treatment, void, debris stripe
- Saves visualizations and JSON results per region
- Writes `defect_report.json` (and optional PDF)

### Island vs Stripe — which detectors run?

| Filename contains | Detectors |
|---|---|
| `island` | `debris_island`, `overspray_island`, `line_defect` |
| `stripe` | `stripe_misalignment`, `overspray`, `surface_treatment`, `void`, `debris_stripe` |
| neither | `surface_treatment` only |

`--pattern` and `--clear` only affect **island** detectors. Stripe detection is the same for legacy and new layouts (the extractor already uses `num_heads` from the region config, e.g. 3 heads in `new_pattern_2400.json`).

### Processing Pre-Extracted Images (Skip Extraction)

Point the standalone runner at a single crop or a folder:

```bash
cd scripts/defects_detection

# Single island
python run_all_detections.py -i "C:\path\KeyIsland.tiff" --pattern new

# Single stripe
python run_all_detections.py -i "C:\path\CyanStripe.tiff" --only stripe_misalignment void

# Whole extracted folder
python run_all_detections.py -i path/to/extracted_images --pattern new
```

**Naming rule:** classification is based solely on the filename (`island` / `stripe`).

Full CLI for the standalone runner (including `--only`, `-o`, `--clear`) is documented in
[`scripts/defects_detection/RUN_ALL_DETECTIONS_README.md`](scripts/defects_detection/RUN_ALL_DETECTIONS_README.md).

## Output Structure

After running, you'll find:

```
image_directory/
└── scan_extracted_regions_20250103_143022/
    ├── KeyStripe.tiff                         # Extracted stripe region
    ├── CyanStripe.tiff                        # Extracted stripe region
    ├── KeyIsland.tiff                         # Extracted island region
    ├── CyanIsland.tiff                        # Extracted island region
    └── output_20250103_143045/                # Detection results
        ├── CyanStripe/
        │   ├── stripe_misalignment_visualization.jpg
        │   ├── overspray_visualization.jpg
        │   ├── surface_treatment_visualization.jpg
        │   ├── void_visualization.tiff
        │   ├── debris_stripe_visualization.jpg
        │   └── CyanStripe_results.json
        ├── KeyIsland/
        │   ├── debris_island_visualization.jpg
        │   ├── line_defect_visualization.jpg
        │   ├── overspray_island_visualization.jpg
        │   └── KeyIsland_results.json
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

#### 1. Stripe Misalignment (`stripe_misalignment`)
- **Purpose**: Head calibration — stitch steps and roll drift between successive print heads
- **Method**: Per-row left/right edge profiles of the stripe; abrupt steps → stitch; gradual drift across a segment → roll
- **Output**: `stripe_misalignment` (`kind=stitch`, signed `step_px`) and `roll_error` (`drift_px`, slope)

#### 2. Overspray Detection (`overspray`)
- **Purpose**: Scattered ink outside intended print areas
- **Method**: Kernel-based grid scanning with scatter analysis
- **Output**: Regions showing ink scattered beyond boundaries

#### 3. Surface Treatment Detection (`surface_treatment`)
- **Purpose**: Poor surface energy issues
- **Method**: Irregular ink drops and void areas
- **Output**: Areas with ink coalescence and missing ink

#### 4. Void Detection (`void`)
- **Purpose**: Compact low-ink / missing-ink patches inside the solid stripe
- **Method**: LAB projection toward paper color + hysteresis threshold + morphology
- **Output**: Bounding boxes with `mean_voidness` / area

#### 5. Debris Stripe (`debris_stripe`)
- **Purpose**: Dark debris spots inside the colored stripe
- **Method**: Multi-cue darkness / saturation score with strong/weak hysteresis
- **Output**: Bounding boxes for dark contaminants

### For Island Images

#### 6. Debris Island Detection (`debris_island`)
- **Purpose**: Foreign particles and contamination
- **Method**: Line removal (horizontal + vertical when `--pattern new`) → threshold → morphology
- **Output**: Contaminated regions with debris highlighted

#### 7. Overspray Island Detection (`overspray_island`)
- **Purpose**: Scattered ink in island regions
- **Method**: Line removal → colored region detection → grouping
- **Output**: Grouped overspray regions with metrics

#### 8. Line Defect Detection (`line_defect`)
- **Purpose**: Missing nozzles (gaps) and misaligned / doubled prints
- **Method** (`--pattern new`): Detect 4 vertical lines → segregate 2 bands → refine horizontal lines → scan for gaps (red) and splits (yellow); density hotspots for clustered missing nozzles
- **Method** (`legacy`): Kernel-based line tracking across scanlines
- **Output**: Missing segments (red), misaligned lines (yellow), optional density regions

With `--clear`, island detectors derive ink/debris/overspray thresholds from the measured background gray level and apply light despeckling for lower SNR.

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

### Standalone detection on one island or stripe image

Prefer this for algorithm iteration — no full-scan extraction needed:

```bash
cd scripts/defects_detection

# Island — new pattern, all detectors
python run_all_detections.py -i KeyIsland.tiff --pattern new

# Island — clear material
python run_all_detections.py -i ClearIsland.tiff --pattern new --clear --only line_defect

# Stripe — stitch/roll + voids
python run_all_detections.py -i CyanStripe.tiff --only stripe_misalignment void -o ..\..\out_stripe

# Folder of extracted regions
python run_all_detections.py -i path/to/extracted --pattern new --sensitivity high --generate_report
```

**CLI summary for `run_all_detections.py`:**

| Flag | Meaning |
|---|---|
| `-i` / `--input` | Single image **or** folder |
| `-o` / `--output` | Output directory (default: timestamped next to input) |
| `--pattern` | `legacy` \| `new` (island only) |
| `--clear` | Clear material; requires `--pattern new` |
| `--only` | Subset of detector keys for that image type |
| `--sensitivity` | `low` \| `medium` \| `high` |
| `--generate_report` | PDF summary |

**`--only` keys**

- Island: `debris_island`, `overspray_island`, `line_defect`
- Stripe: `stripe_misalignment`, `overspray`, `surface_treatment`, `void`, `debris_stripe`

**PowerShell:** separate `cd` and `python` with `;` — do not concatenate them:

```powershell
cd scripts/defects_detection; python run_all_detections.py -i "C:\path\KeyIsland.tiff" --pattern new --only line_defect
```

Details: [`scripts/defects_detection/RUN_ALL_DETECTIONS_README.md`](scripts/defects_detection/RUN_ALL_DETECTIONS_README.md).
Algorithm notes: [`scripts/defects_detection/ALGORITHMS.md`](scripts/defects_detection/ALGORITHMS.md).

## Module Reference

### Core Scripts
- **`main_defect_detection.py`**: Full workflow — extract regions then run detections (`--pattern`, `--clear`)
- **`scripts/defects_detection/run_all_detections.py`**: Standalone runner for one image or a folder (`-i`, `--only`, `--pattern`, `--clear`)
- **`scripts/utility/tiff_extractor.py`**: Legacy region extraction
- **`scripts/utility/new_pattern_tiff_extractor.py`**: New-pattern extraction (`num_heads`, dual-band islands)

### Detection Modules
- **`detector_base.py`**: Base class with exclusion zone support
- **`stripe_misalignment_detection.py`**: Stitch / roll calibration on stripe edges
- **`overspray_detection.py`**: Scattered ink detector (stripe images)
- **`surface_treatment_detection.py`**: Irregular drops and voids
- **`void_detection.py`**: Compact voids inside solid stripes
- **`debris_stripe_detector.py`**: Dark debris inside colored stripes
- **`debris_island_detection.py`** / **`new_pattern_debris_island_detection.py`**: Island debris
- **`overspray_island_detection.py`** / **`new_pattern_overspray_island_detection.py`**: Island overspray
- **`line_defect_detection.py`** / **`new_pattern_line_defect_detection.py`**: Missing nozzles / misaligned lines

### Utility Modules
- **`utils/vertical_band_detector.py`**: 4 vertical lines + dual-band segregation (new pattern)
- **`utils/band_line_refiner.py`**: Full-width horizontal line trajectories
- **`utils/island_line_extractor.py`**: Per-line ink / gap / split analysis
- **`utils/material_profile.py`**: Background level for `--clear` material
- **`utils/edge_detector.py`**: Enhanced edge detection with noise reduction
- **`utils/image_saver.py`**: Smart image saving (auto-switches to TIFF for large images)
- **`utils/line_detector.py`**: Slanted line detection for legacy islands

## Try It Yourself / Example Use Cases

Assuming you have access to the example TIFF images, you can try the following commands to see how the system detects various defects. (Make sure to replace the `<location to tif file...>` placeholders with the actual file paths on your system before running.)

### 1. Bad/Missing Nozzles (Island — legacy)
Detects the misprint of individual horizontal lines on island sections.
- **Example Image:** `test_Paper2400_BlackPt210_Sharpen1_R4_T1.tif`
- **Command:**
  ```bash
  python .\main_defect_detection.py --image "<location to tif file test_Paper2400_BlackPt210_Sharpen1_R4_T1.tif>" --dpi 2400 --config '.\regions_json\template-2400-configs.json'
  ```
- **Where to Check:** Open the output folder next to the input image → `island-black-blue` → `line_defect_visualization.jpg` (missing lines in red).

### 2. New-pattern island (dual-band) — standalone crop
```bash
cd scripts/defects_detection
python run_all_detections.py -i "C:\path\KeyIsland.tiff" --pattern new --only line_defect
```
- **Where to Check:** `line_defect_visualization.jpg` — red = missing nozzles, yellow = misaligned / doubled prints.

### 3. Clear-material island
```bash
cd scripts/defects_detection
python run_all_detections.py -i "C:\path\ClearIsland.tiff" --pattern new --clear
```

### 4. Debris / Overspray on Island
Same full-scan run as Example 1, or on a single crop:
```bash
python run_all_detections.py -i "C:\path\KeyIsland.tiff" --pattern new --only debris_island overspray_island
```
- **Where to Check:** `debris_island_visualization.jpg`, `overspray_island_visualization.jpg`.

### 5. Calibration Error (Stitch / Roll on Stripe)
```bash
cd scripts/defects_detection
python run_all_detections.py -i "C:\path\CyanStripe.tiff" --only stripe_misalignment
```
- **Where to Check:** `stripe_misalignment_visualization.jpg` — red stitch lines with signed step (px), orange roll arrows.

Or via the full scan:
```bash
python .\main_defect_detection.py --image "<location to tif file test_Paper2400.tif>" --dpi 2400 --config '.\regions_json\template-2400-configs.json'
```
Open the `blueStripe` (or `CyanStripe`) folder → `stripe_misalignment_visualization.jpg`.

### 6. Voids inside a Stripe
```bash
cd scripts/defects_detection
python run_all_detections.py -i "C:\path\CyanStripe.tiff" --only void
```
- **Where to Check:** `void_visualization.tiff` — black boxes around low-ink patches.

## License

This project is proprietary. All rights reserved.
