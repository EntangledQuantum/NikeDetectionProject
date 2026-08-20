# User Guide — How to Run Defect Detection Scripts

Practical guide for operators and developers: which scripts you can run directly, when to use each one, and every CLI argument.

**Prerequisites**

```bash
pip install -r requirements.txt
```

Run commands from the **project root** unless a section says otherwise.

---

## Which script should I use?

| I want to… | Use this | Notes |
|---|---|---|
| Process a **full scan TIFF** end-to-end (extract + detect) | `main_defect_detection.py` | Primary entry point |
| Test detectors on an **already-extracted** island/stripe crop (or a folder of them) | `scripts/defects_detection/run_all_detections.py` | Fast iteration; no re-extraction |
| Extract **legacy** regions from a full TIFF using bbox JSON | `scripts/utility/tiff_extractor.py` | Needs `original_image_path` inside the JSON |
| Extract **new-pattern** color columns (and optionally split stripe/island) | `scripts/utility/new_pattern_tiff_extractor.py` | Parametric geometry config |
| Run **only** stripe debris on one image | `scripts/defects_detection/debris_stripe_detector.py` | Single-detector CLI |
| Run **only** surface treatment on one image/folder | `scripts/defects_detection/surface_treatment_detection.py` | Single-detector CLI |

Everything else under `scripts/defects_detection/` (void, line defect, overspray, band utils, …) is a **library module** — import it or call it through `run_all_detections.py` / `main_defect_detection.py`. They are not meant to be run as standalone CLIs.

---

## Directly executable scripts

| Script | Runnable? | CLI? |
|---|---|---|
| `main_defect_detection.py` | Yes | Full argparse |
| `scripts/defects_detection/run_all_detections.py` | Yes | Full argparse |
| `scripts/utility/tiff_extractor.py` | Yes | Positional config path |
| `scripts/utility/new_pattern_tiff_extractor.py` | Yes | Positional TIFF + config |
| `scripts/defects_detection/debris_stripe_detector.py` | Yes | Full argparse |
| `scripts/defects_detection/surface_treatment_detection.py` | Yes | Full argparse |
| `scripts/defects_detection/utils/edge_detector.py` | Yes* | No CLI — hardcoded paths inside the file (dev utility) |
| Other detector `.py` files | No | Use via `run_all_detections.py` |

\* Editable demo only; not a production CLI.

---

## User stories

### Story 1 — “I have a full scan and want all defects”

As an operator, I drop a TIFF on disk and want extraction + detection in one command.

```bash
# Legacy layout (single-band islands) — default
python main_defect_detection.py --image path/to/scan.tif --dpi 2400

# New dual-band island pattern (July visit layout)
python main_defect_detection.py --image path/to/scan.tif --dpi 2400 --pattern new

# Clear / gray paper material (requires --pattern new)
python main_defect_detection.py --image path/to/scan.tif --dpi 2400 --pattern new --clear

# High sensitivity + PDF report
python main_defect_detection.py --image path/to/scan.tif --dpi 2400 --pattern new -s high --generate_report
```

**What happens**

1. Loads a DPI template (`regions_json/template-*-configs.json` or `new_pattern_*.json`)
2. Extracts stripe/island crops next to the TIFF
3. Runs the matching detector stack per filename
4. Writes visualizations + JSON (and optional PDF) under an `output_*` folder

---

### Story 2 — “I already extracted crops; I want to re-run detectors”

As a developer tuning thresholds, I skip extraction and point at one crop or a folder.

```bash
cd scripts/defects_detection

# New-pattern island — all island detectors
python run_all_detections.py -i /path/KeyIsland.tiff --pattern new

# Clear material island
python run_all_detections.py -i /path/ClearIsland.tiff --pattern new --clear

# Only missing-nozzle / jagged lines
python run_all_detections.py -i /path/KeyIsland.tiff --pattern new --only line_defect

# Stripe — all stripe detectors
python run_all_detections.py -i /path/CyanStripe.tiff

# Stripe — voids + stitch/roll only
python run_all_detections.py -i /path/CyanStripe.tiff --only void stripe_misalignment

# Whole extracted folder
python run_all_detections.py -i /path/extracted_regions --pattern new -o /path/out

# Recursive parent folder that contains stripe + island crops (skips full-scan color TIFFs)
python run_all_detections.py \
  -i /home/koushik/DigitalAirCv/data/July_26 \
  -o /home/koushik/DigitalAirCv/DigitalAirCvProject/tmp/july26_all \
  --pattern new
```

**Filename routing (required)**

| Filename contains | Detectors |
|---|---|
| `island` | `debris_island`, `overspray_island`, `line_defect` |
| `stripe` | `stripe_misalignment`, `edge_roughness`, `overspray`, `surface_treatment`, `void`, `debris_stripe` |
| neither | skipped in folder mode (unless `--include-unknown`) |

Folder mode is recursive by default. Timing (stripe vs island totals, and per-detector seconds) is printed at the end and written under `timing` in `defect_report.json`.

---

### Story 3 — “I only need to extract regions”

**New pattern** (parametric color columns → optional stripe/island split):

```bash
python scripts/utility/new_pattern_tiff_extractor.py \
  path/to/scan.tif \
  regions_json/new_pattern_2400.json \
  --split-stripe-island \
  -o path/to/output_base
```

**Legacy** (bbox list in JSON; JSON must include `original_image_path`):

```bash
python scripts/utility/tiff_extractor.py path/to/config_with_image_path.json
```

Then run Story 2 on the extracted folder.

---

### Story 4 — “I want one specific detector only”

Prefer `run_all_detections.py --only …` for most cases. Two detectors also have their own CLIs:

```bash
# Stripe debris only
python scripts/defects_detection/debris_stripe_detector.py path/to/CyanStripe.tiff -o out.jpg --sensitivity medium --debug

# Surface treatment only
python scripts/defects_detection/surface_treatment_detection.py path/to/image_or_folder -o output
```

---

## CLI reference

### 1. `main_defect_detection.py` — full-scan orchestrator

```bash
python main_defect_detection.py --image <TIFF> --dpi {2400|4800} [options]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `-i`, `--image` | Yes | — | Path to the full-scan TIFF |
| `-d`, `--dpi` | Yes | — | `2400` or `4800` — selects the built-in template |
| `-c`, `--config` | No | DPI template | Custom JSON; overrides the DPI template |
| `-s`, `--sensitivity` | No | `medium` | `low` \| `medium` \| `high` |
| `--pattern` | No | `legacy` | `legacy` = single-band islands; `new` = dual-band (4 vertical lines) |
| `--clear` | No | off | Clear/gray paper material; **requires `--pattern new`** |
| `--generate_report` | No | off | Also write a PDF summary |

**Template selection**

| `--pattern` | `--dpi` | Config used (if `-c` omitted) |
|---|---|---|
| `legacy` | `2400` | `regions_json/template-2400-configs.json` |
| `legacy` | `4800` | `regions_json/template-4800-configs.json` |
| `new` | `2400` | `regions_json/new_pattern_2400.json` |
| `new` | `4800` | `regions_json/new_pattern_4800.json` (if present) |

---

### 2. `run_all_detections.py` — standalone detector runner

```bash
python scripts/defects_detection/run_all_detections.py -i <file_or_folder> [options]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `-i`, `--input` | Yes* | — | Single image **or** folder of images |
| `--input_folder` | Yes* | — | Hidden alias for `-i` (used by the orchestrator) |
| `-o`, `--output` | No | `output_<timestamp>/` next to input | Output directory |
| `--sensitivity` | No | `medium` | `low` \| `medium` \| `high` |
| `--pattern` | No | `legacy` | Island pattern: `legacy` \| `new` |
| `--clear` | No | off | Clear material; **requires `--pattern new`**; islands only |
| `--only` | No | all for that image type | One or more detector keys (see below) |
| `--no-recursive` | No | off | Folder mode: top-level only (default walks subfolders) |
| `--include-unknown` | No | off | Folder mode: also process non-stripe/non-island names |
| `--generate_report` | No | off | Also write a PDF summary |

\* One of `-i` / `--input` or `--input_folder` is required.

**`--only` detector keys**

| Key | Image type |
|---|---|
| `debris_island` | Island |
| `overspray_island` | Island |
| `line_defect` | Island |
| `stripe_misalignment` | Stripe |
| `overspray` | Stripe |
| `surface_treatment` | Stripe / unknown |
| `void` | Stripe |
| `debris_stripe` | Stripe |
| `edge_roughness` | Stripe |

Supported image extensions: `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.bmp`.

---

### 3. `new_pattern_tiff_extractor.py` — new-pattern region extraction

```bash
python scripts/utility/new_pattern_tiff_extractor.py <input_tiff> <config> [options]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `input_tiff` | Yes | — | Full-scan TIFF path |
| `config` | Yes | — | New-pattern JSON (e.g. `regions_json/new_pattern_2400.json`) |
| `-o`, `--output-dir` | No | TIFF’s parent dir | Base output dir; writes under an `extracted` subfolder |
| `--split-stripe-island` | No | off | After each color column, also write `*Stripe.tiff` / `*Island.tiff` |
| `-v`, `--verbose` | No | off | Debug logging |

Example config fields: `colors`, `color_width`, `x_offset`, `num_heads`, `head_height`, `y_offset`, `island_front`, `island_width`, `stripe_width`, `buffer`.

---

### 4. `tiff_extractor.py` — legacy bbox extraction

```bash
python scripts/utility/tiff_extractor.py <config> [-v]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `config` | Yes | — | Path to JSON **or** a JSON string |
| `-v`, `--verbose` | No | off | Debug logging |

The JSON must include:

- `original_image_path` — absolute/relative path to the TIFF
- `sub_images[]` — each with `name` + `bounding_box_pixels` (`top_x/y`, `bottom_x/y`)
- optional `exclusion_zones[]` — same bbox shape; written as sibling JSON per crop

Output: `{image_stem}_output/` next to the TIFF.

> Built-in templates under `regions_json/template-*.json` do **not** embed `original_image_path`. Prefer `main_defect_detection.py`, which injects the image path for you. If you call this script alone, add that field to a copy of the config.

---

### 5. `debris_stripe_detector.py` — stripe debris only

```bash
python scripts/defects_detection/debris_stripe_detector.py <image> [options]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `image` | Yes | — | Path to a stripe image |
| `-o`, `--output` | No | `{stem}_debris_stripe.jpg` | Visualization path |
| `--sensitivity` | No | `medium` | `low` \| `medium` \| `high` |
| `--debug` | No | off | Save debug artifacts next to the output |

---

### 6. `surface_treatment_detection.py` — surface treatment only

```bash
python scripts/defects_detection/surface_treatment_detection.py <input> [options]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `input` | Yes | — | Image file **or** directory |
| `-o`, `--output` | No | `output` | Output directory |
| `--contrast-threshold` | No | `50` | Min contrast for high-contrast drops |
| `--void-size` | No | `20` | Min size for void areas |
| `--coalescence-size` | No | `100` | Min size for coalesced ink drops |

---

## Sensitivity cheat sheet

| Level | Effect |
|---|---|
| `low` | Fewer detections (stricter / larger min areas) |
| `medium` | Default balance |
| `high` | More detections (looser thresholds / smaller min areas) |

Passed through from both `main_defect_detection.py` and `run_all_detections.py` into the detectors that support it.

---

## Outputs (typical)

After a full run you get something like:

```
scan_extracted_regions_YYYYMMDD_HHMMSS/
├── KeyStripe.tiff / CyanStripe.tiff / …
├── KeyIsland.tiff / CyanIsland.tiff / …
└── output_YYYYMMDD_HHMMSS/
    ├── *_visualization.jpg (or .tiff)
    ├── *_results.json
    ├── defect_report.json
    └── defect_report.pdf          # if --generate_report
```

---

## Quick decision tree

```
Have a full scan TIFF?
├─ Yes → main_defect_detection.py  (-i, -d, [--pattern new] [--clear] …)
└─ No (already have crops)
   ├─ Want all / several detectors → run_all_detections.py  (-i, [--only …])
   ├─ Stripe debris only           → debris_stripe_detector.py
   └─ Surface treatment only       → surface_treatment_detection.py

Only need extraction?
├─ New pattern → new_pattern_tiff_extractor.py  (--split-stripe-island)
└─ Legacy bboxes → tiff_extractor.py  (JSON with original_image_path)
```

---

## Related docs

| Doc | Contents |
|---|---|
| `README.md` | System overview, configs, workflows |
| `scripts/defects_detection/RUN_ALL_DETECTIONS_README.md` | Standalone runner deep-dive |
| `Algorithm.md` | How each detector works, shared work, parallelization, speed-ups |
