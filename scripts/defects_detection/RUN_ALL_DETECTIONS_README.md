# `run_all_detections.py`

This script runs the detection pipeline on either a **single** island/stripe
image or a **folder** of already-extracted region images.

## What it expects

- Input (`-i` / `--input`): a single image file **or** a folder of images
  (e.g. `KeyIsland.tiff`, `blueStripe.tiff`, `island-black-blue.tiff`)
- Filenames drive routing (same rules as the full-image workflow):
  - names containing `stripe` -> stripe detector stack
  - names containing `island` -> island detector stack
  - anything else -> fallback `surface_treatment`

## Quick standalone tests

```bash
# New-pattern island (all island detectors)
python run_all_detections.py -i "C:\path\KeyIsland.tiff" --pattern new

# Only line defects (missing nozzles / misalignment) — faster iteration
python run_all_detections.py -i "C:\path\KeyIsland.tiff" --pattern new --only line_defect

# Clear scan material (gray background, fainter ink, lower SNR)
python run_all_detections.py -i "C:\path\ClearIsland.tiff" --pattern new --clear

# Stripe image
python run_all_detections.py -i "C:\path\blueStripe.tiff"

# Legacy island
python run_all_detections.py -i "C:\path\island-black-blue.tiff"

# Whole extracted folder
python run_all_detections.py -i extracted_dir --pattern new
```

Outputs land in `output_YYYYMMDD_HHMMSS/` next to the input (or under `-o`).

## How it works

1. The script parses CLI arguments:
   - `--input` / `-i` (file or folder; `--input_folder` still accepted)
   - `--sensitivity` (`low`, `medium`, `high`)
   - `--pattern` (`legacy`, `new`) — selects dual-band island detectors when `new`
   - `--clear` — clear scan material: island binarization/debris/overspray
     thresholds are derived from the measured background gray level of each
     image instead of the fixed white-paper values, and inputs are
     despeckled for the lower SNR (requires `--pattern new`)
   - `--only` — optional subset of detectors for the chosen strategy
   - `--generate_report`
2. It classifies each image by filename using `ImageTypeClassifier`.
3. A strategy object is chosen by `DetectionStrategyFactory`:
   - `StripeDetectionStrategy`
   - `IslandDetectionStrategy` / `NewPatternIslandDetectionStrategy`
   - `UnknownDetectionStrategy`
4. That strategy asks `DetectorFactory` to build detector instances for the selected sensitivity.
5. `SingleImageProcessor` runs each detector, saves a visualization image, and stores structured JSON-safe defect data.
6. `ResultsSaver` writes:
   - per-image result JSON
   - a folder-level `defect_report.json`
   - optional PDF summary report

## Current routing

### Stripe images

- `stripe_misalignment`
- `overspray`
- `surface_treatment`
- `void`
- `debris_stripe`

### Island images (`--pattern legacy`)

- `debris_island`
- `overspray_island`
- `line_defect`

### Island images (`--pattern new`)

Same keys, wired to the dual-band detectors:

- `NewPatternDebrisIslandDetector`
- `NewPatternOversprayIslandDetector`
- `NewPatternLineDefectDetector`

### Unknown images

- `surface_treatment`

## Relationship to `main_defect_detection.py`

`main_defect_detection.py` is the full workflow entrypoint:

1. choose region config from DPI or custom JSON
2. extract stripe/island sub-images using `tiff_extractor.py`
3. call `run_all_detections.py` on the extracted folder

So:

- use `main_defect_detection.py` for full-image TIFF workflows
- use `run_all_detections.py -i <image>` for a quick single-region test
- use `run_all_detections.py -i <folder>` when you already have extracted stripe/island region images
