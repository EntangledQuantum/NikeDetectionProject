# Print Defect Detection System

A comprehensive suite of computer vision algorithms for detecting various printing defects in large TIFF scanned images.

## Overview

This system automatically detects critical printing defects in high-resolution scanned images:

**Island regions**
- **Debris**: Foreign particles causing dark spots and contamination
- **Overspray**: Ink scattered outside intended line areas
- **Line Defects**: Missing nozzles (gaps), hazy/smudged print, and stitch error (jagged zig-zag where successive heads join)

**Stripe regions**
- **Stripe Misalignment**: Stitch steps and roll drift between successive print heads
- **Edge Roughness**: High-frequency jaggedness of the left/right stripe edges (independent of stitch/roll)
- **Overspray**: Scattered ink outside the stripe
- **Surface Treatment Issues**: Irregular ink drops and missing ink areas
- **Voids**: Compact low-ink / missing-ink patches inside the solid stripe
- **Debris Stripe**: Dark debris spots inside the colored stripe

## Features

- **Toggle detectors in config**: `detector_sets` in `config/detection_2400.json` (`true` / `false` per detector)
- **Full-scan defect overlay + summary**: annotated `{prefix}_full_defects.jpg` plus `defect_summary.txt` / `.csv` at the result-folder root
- **Standalone region testing**: Run detectors on a single island, stripe, or `full` crop
- **Legacy + new island patterns**: Single-band (`--pattern legacy`) or dual-band with 4 vertical lines (`--pattern new`)
- **Clear scan material**: `--clear` adapts island thresholds for gray paper / fainter ink (requires `--pattern new`)
- **2400 DPI only**: All pixel thresholds and geometry live in [`config/detection_2400.json`](config/detection_2400.json)
- **Shared preprocessing**: Each region is decoded once; stripe edges / island bands are built once and shared
- **Parallel detectors**: Threads inside one image, process workers across a folder
- **Exclusion Zones**: Loaded once onto the image context and applied to detections
- **Separated I/O**: Detectors score defects; JSON and visualizations are written afterwards

## Installation

You need **Python 3.10, 3.11, 3.12, or 3.13** and a **virtual environment kept outside this project folder** (so large TIFF outputs and git do not mix with your Python packages).

`requirements.txt` uses **automatic OS detection**: when you run `pip install -r requirements.txt`, pip only installs the lines that match your operating system and Python version (for example, headless OpenCV on Linux, regular OpenCV on Windows and macOS).

| OS | Recommended tool | Why |
|---|---|---|
| **Windows** | [uv](https://docs.astral.sh/uv/) or built-in `venv` | Fast installs, works well with OpenCV wheels |
| **macOS** | [uv](https://docs.astral.sh/uv/) | Fastest setup; one command creates the env and installs deps |
| **Linux** | [Conda](https://docs.conda.io/) (or uv / `venv`) | Conda handles system libraries on many distros; uv is a lighter alternative |

### Step 0 — Get Python

1. Download Python from [python.org/downloads](https://www.python.org/downloads/) (pick **3.11** or **3.12** if you can).
2. During install on Windows, check **“Add python.exe to PATH”**.
3. Open a **new** terminal and confirm:

```bash
python --version
# or on macOS/Linux:
python3 --version
```

You should see `Python 3.10` or newer.

---

### Windows

**Option A — uv (recommended)**

```powershell
# 1) Install uv (one time)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Close and reopen PowerShell, then:

# 2) Create env OUTSIDE the repo (example path)
uv venv $env:USERPROFILE\.venvs\nike-detection --python 3.12

# 3) Activate
.\$env:USERPROFILE\.venvs\nike-detection\Scripts\Activate.ps1

# 4) Go to the project and install deps
cd C:\path\to\NikeDetectionProject
uv pip install -r requirements.txt
```

**Option B — built-in venv (no extra tools)**

```powershell
python -m venv $env:USERPROFILE\.venvs\nike-detection
.\$env:USERPROFILE\.venvs\nike-detection\Scripts\Activate.ps1
cd C:\path\to\NikeDetectionProject
python -m pip install --upgrade pip wheel
pip install -r requirements.txt
```

**Automated script (Windows):**

```powershell
cd C:\path\to\NikeDetectionProject
.\scripts\setup_env.ps1
```

---

### macOS

**Option A — uv (recommended)**

```bash
# 1) Install uv (one time)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Restart the terminal, then:

# 2) Create env outside the repo
uv venv ~/.venvs/nike-detection --python python3

# 3) Activate
source ~/.venvs/nike-detection/bin/activate

# 4) Install
cd /path/to/NikeDetectionProject
uv pip install -r requirements.txt
```

**Option B — built-in venv**

```bash
python3 -m venv ~/.venvs/nike-detection
source ~/.venvs/nike-detection/bin/activate
cd /path/to/NikeDetectionProject
python -m pip install --upgrade pip wheel
pip install -r requirements.txt
```

**Automated script (macOS / Linux):**

```bash
cd /path/to/NikeDetectionProject
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

---

### Linux

**Option A — Conda (recommended on Linux)**

```bash
# 1) Install Miniconda if needed: https://docs.anaconda.com/miniconda/

# 2) Create env outside the repo
conda create -y -p ~/.venvs/nike-detection python=3.11 pip
conda activate ~/.venvs/nike-detection

# 3) Install
cd /path/to/NikeDetectionProject
pip install -r requirements.txt
```

**Option B — uv**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv ~/.venvs/nike-detection --python python3
source ~/.venvs/nike-detection/bin/activate
cd /path/to/NikeDetectionProject
uv pip install -r requirements.txt
```

**Option C — built-in venv**

```bash
python3 -m venv ~/.venvs/nike-detection
source ~/.venvs/nike-detection/bin/activate
cd /path/to/NikeDetectionProject
python -m pip install --upgrade pip wheel
pip install -r requirements.txt
```

---

### Verify the install

With the virtual environment **activated** and your shell in the project folder:

```bash
python -c "import cv2, numpy, scipy, skimage, tifffile, matplotlib; print('OK')"
python -m nike_detection -i data/blackStripe.tiff --only stripe_misalignment --no-vis
```

If you see `OK` and a new timestamped output folder next to the TIFF, everything is working.

**Optional — run tests:**

```bash
pip install -r requirements-dev.txt
pytest tests/ -q
```

**Requirements (from `requirements.txt`):**
- Python 3.10+
- NumPy, SciPy, OpenCV, scikit-image, matplotlib, tifffile, Pillow, tqdm

## Quick Start

```bash
# Preferred: python -m nike_detection  (2400 DPI, --pattern new by default)

# 1) Inspect island/stripe boxes on a full scan (no detectors)
python -m nike_detection -i Cyan_full.tiff --regions-only
python -m nike_detection -i scan_KCMY.tif --regions-only --workers 1

# 2) Measure boxes, then run enabled detectors (see detector_sets in config)
python -m nike_detection -i Cyan_full.tiff --pattern new

# 3) Already-extracted crop
python -m nike_detection -i KeyIsland.tiff --pattern new --only line_defect
python -m nike_detection -i CyanStripe.tiff --only stripe_misalignment -s high

# 4) Clear / gray paper (islands only)
python -m nike_detection -i ClearIsland.tiff --pattern new --clear

# 5) Template extract then detect (legacy path — not the seed-free boxes)
python -m nike_detection -i path/to/scan.tif --extract --pattern new
```

Full flag table, detector keys, and every script shim: [Algorithm.md §11](Algorithm.md#11-how-to-invoke). Algorithms: [Algorithm.md](Algorithm.md). Thresholds: [`config/detection_2400.json`](config/detection_2400.json).

`main_defect_detection.py` always adds `--extract`. `scripts/defects_detection/run_all_detections.py` is a shim for already-extracted crops.

## What You Need

### 1. **TIFF Image File** (Required)
Your input must be a **TIFF** (`.tif` or `.tiff`) file containing the full scanned print. The system will automatically extract individual regions based on your DPI template or custom configuration.

**Supported DPI:** 2400 only. 4800 templates are not used by this pipeline.

Change thresholds by editing [`config/detection_2400.json`](config/detection_2400.json) (`sensitivity.low|medium|high`) rather than detector source files.

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

### Command-Line Options (`python -m nike_detection`)

```bash
python -m nike_detection -i <image-or-folder> [options]
```

**Required:**
- `--input`, `-i`: Image file, `full` TIFF, or folder of crops

**Optional:**
- `--config`: Unified JSON (default: `config/detection_2400.json`)
- `--sensitivity`, `-s`: `low` \| `medium` \| `high`
- `--pattern`: `legacy` \| `new` (default from config: `new`)
- `--clear`: Clear scan material. **Requires `--pattern new`.**
- `--only`: Subset of detector keys
- `--regions-only`: Measure the island/stripe boxes on a `full` scan and write the overlay + corner crops, without running detectors
- `--regions`: Operator-supplied bounding boxes for a `full` TIFF (overrides automatic detection)
- `--extract`: Extract regions from a press scan first
- `--no-vis` / `--downscale-vis` / `--debug`
- `--workers` / `--detector-threads`
- `--generate_report`: PDF summary
- `--dpi`: Must be `2400`

### Usage Examples

```bash
python -m nike_detection -i scan.tif --extract --pattern new
python -m nike_detection -i scan.tif --extract --pattern new --clear
python -m nike_detection -i KeyIsland.tiff --pattern new --only line_defect
python -m nike_detection -i CyanStripe.tiff --only void stripe_misalignment -s high
python -m nike_detection -i Cyan_full.tiff --pattern new --regions regions.json
python -m nike_detection -i extracted_folder --pattern new --workers 2 --no-vis

# Check the island/stripe boxes on a folder of full scans before detecting
python -m nike_detection -i full_scans_folder --pattern new --regions-only

# Measure every colour's boxes on one multi-colour sheet (KCMY, 6.7 GB)
python -m nike_detection -i scan_KCMY.tif --config config/detection_2400.json \
    --regions-only --workers 1
```

On a `full` scan the island and stripe boxes of every colour are measured from the print
itself — no seed coordinates and no positional priors, so a sheet holding one colour and a
sheet holding Key, Cyan, Magenta and Yellow are handled the same way. Colours are named left
to right from `geometry.colors`, and each gets its own y range since successive colours can
sit ±200 px higher or lower. The nominal layout (island 5100 px, gap 580 px, stripe 1050 px,
height 33000 px, 250 px between colours, at 2400 DPI) lives in
`config/detection_2400.json` → `region_reference` and is used only to disambiguate, validate,
and fill in an edge that failed to print. Sheets over 1.5 GB are memory-mapped rather than
decoded, so a full 56000 × 40000 KCMY scan runs in about a minute without exhausting RAM.
See [Algorithm.md §6](Algorithm.md#6-region-extraction-feeds-the-detectors).

```bash
# Clear material + new pattern (always --extract)
python main_defect_detection.py -i scan.tif -d 2400 --pattern new --clear

# High sensitivity with PDF report
python main_defect_detection.py -i scan.tif -d 2400 -s high --generate_report
```

### What Happens When You Run It

**`full` TIFF (filename contains `full`), no `--extract`**
- Memory-maps the sheet if it is larger than 1.5 GB
- Measures island + stripe boxes for every colour from the print (no seeds)
- Writes `{prefix}_full_regions.json` / `.jpg` / `_corners.jpg`
- Unless `--regions-only`, crops each box in memory and runs the **enabled** island and stripe detectors
- Writes `{prefix}_full_defects.jpg` (all findings on the full scan) plus `defect_summary.txt` / `.csv` at the result-folder root
- Per-region subfolders are off unless `write_region_folders` or `--region-folders`

**`--extract`**
- Template-crops from `config.geometry` into `{stem}_extracted_regions_<timestamp>/extracted/`
- Then runs detectors on those files (this is **not** the seed-free path)

**Island or stripe crop**
- Filename token selects the detector set; shared `ImageContext` runs geometry once
- Writes `defect_report.json` plus `defect_summary.txt` / `.csv` at the result-folder root (optional PDF)

### Island vs Stripe — which detectors run?

Flip detectors on or off in `config/detection_2400.json` → `detector_sets` (`true` / `false`). Current defaults:

| Filename contains | Detectors (config default) |
|---|---|
| `full` | Measure boxes, then both sets below |
| `island` | `line_defect` (missing nozzles + stitch / calibration) |
| `stripe` | `stripe_misalignment`, `edge_roughness`, `void` |
| neither | none (`surface_treatment` is off) |

`--pattern` and `--clear` only affect **island** detectors. Surface treatment is opt-in. Stripe detection is the same for legacy and new layouts.

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
{image_name}_MM_DD_YY_HH_MM_SS/          # Detection results (next to the TIFF)
    ├── {prefix}_full_regions.jpg            # Segmented island/stripe boxes
    ├── {prefix}_full_defects.jpg            # Same scan with defects annotated
    ├── defect_summary.txt                   # Readable per-region metrics
    ├── defect_summary.csv
    └── defect_report.json                   # Full detector payload
```

Per-region subfolders (`CyanStripe/line_defect_visualization.jpg`, …) are **off** unless you set `write_region_folders: true` or pass `--region-folders`.

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

#### 5b. Edge Roughness (`edge_roughness`)
- **Purpose**: Jagged / saw-tooth roughness on the left and right stripe edges
- **Method**: Sub-pixel edge trace → remove stitch/roll → MAD + P95 of the high-pass residual, per edge
- **Output**: Per-edge quantification (`sigma_px`, `mad_px`, `p95_px`) and flagged rough spans (red)

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
→ Activate your virtual environment, then run `pip install -r requirements.txt` (see [Installation](#installation))

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
- Stripe: `stripe_misalignment`, `edge_roughness`, `overspray`, `surface_treatment`, `void`, `debris_stripe`

**PowerShell:** separate `cd` and `python` with `;` — do not concatenate them:

```powershell
cd scripts/defects_detection; python run_all_detections.py -i "C:\path\KeyIsland.tiff" --pattern new --only line_defect
```

Details: [`scripts/defects_detection/RUN_ALL_DETECTIONS_README.md`](scripts/defects_detection/RUN_ALL_DETECTIONS_README.md).
Algorithm notes: [`Algorithm.md`](Algorithm.md).

## Module Reference

### Core
- **`python -m nike_detection`**: Single CLI — detect crops, `full` TIFFs, or `--extract` a press scan
- **`config/detection_2400.json`**: All geometry, detector sets, and sensitivity tables
- **`nike_detection/pipeline/runner.py`**: Load once → shared `ImageContext` → parallel detectors → JSON/vis
- **`main_defect_detection.py`**: Shim that calls `--extract`
- **`scripts/defects_detection/run_all_detections.py`**: Shim for already-extracted crops
- **`scripts/utility/tiff_extractor.py`**: Legacy bbox extraction (used in-process)
- **`scripts/utility/new_pattern_tiff_extractor.py`**: New-pattern extraction (used in-process)

### Detection
- **`nike_detection/detectors/stripe/`**: Misalignment, roughness, void, debris, overspray, surface treatment
- **`nike_detection/detectors/island_legacy/`**: Single-band debris / overspray / line defect
- **`nike_detection/detectors/island_new/`**: Dual-band debris / overspray / line defect
- **`nike_detection/geometry/`**: Vertical bands, island line extractor, legacy `LineDetector`

Algorithm notes: [`Algorithm.md`](Algorithm.md).

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
