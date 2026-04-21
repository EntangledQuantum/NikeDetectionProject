# `run_all_detections.py`

This script is the detection-only orchestrator for a folder of already extracted region images.

## What it expects

- Input: a folder containing image files such as `blueStripe.tiff`, `blackStripe.tiff`, `pinkStripe.tiff`, or island images like `island-black-blue.tiff`
- Filenames drive routing:
  - names containing `stripe` -> stripe detector stack
  - names containing `island` -> island detector stack
  - anything else -> fallback `surface_treatment`

## How it works

1. The script parses CLI arguments:
   - `--input_folder`
   - `--sensitivity` (`low`, `medium`, `high`)
   - `--generate_report`
2. It scans the folder for supported image extensions.
3. Each image is classified by filename using `ImageTypeClassifier`.
4. A strategy object is chosen by `DetectionStrategyFactory`:
   - `StripeDetectionStrategy`
   - `IslandDetectionStrategy`
   - `UnknownDetectionStrategy`
5. That strategy asks `DetectorFactory` to build detector instances for the selected sensitivity.
6. `SingleImageProcessor` runs each detector, saves a visualization image, and stores structured JSON-safe defect data.
7. `ResultsSaver` writes:
   - per-image result JSON
   - a folder-level `defect_report.json`
   - optional PDF summary report

## Current routing

### Stripe images

Stripe images run these detectors:

- `stripe_misalignment`
- `overspray`
- `surface_treatment`
- `void`

The new `void` detector is only called for stripe images because the routing is filename-based and only the stripe strategy includes it.

### Island images

Island images run:

- `debris_island`
- `overspray_island`
- `line_defect`

### Unknown images

Unknown images run:

- `surface_treatment`

## Preprocessing behavior

`run_all_detections.py` does not force a single preprocessing pipeline for every detector. It dispatches inputs based on detector needs:

- `surface_treatment` receives a CLAHE-enhanced grayscale image
- `stripe_misalignment`, `overspray`, and `void` receive the original image
- island detectors receive the original image and optionally the image path so they can load exclusion zones

## Output layout

For each input image, the script creates a subfolder inside a timestamped `output_YYYYMMDD_HHMMSS` directory.

Typical outputs:

- `*_results.json` with detector counts and defect metadata
- `stripe_misalignment_visualization.jpg`
- `overspray_visualization.jpg`
- `surface_treatment_visualization.jpg`
- `void_visualization.tiff`

`void_visualization.tiff` preserves the stripe image dimensions and contains black bounding boxes around detected voids.

## Relationship to `main_defect_detection.py`

`main_defect_detection.py` is the full workflow entrypoint:

1. choose region config from DPI or custom JSON
2. extract stripe/island sub-images using `tiff_extractor.py`
3. call `run_all_detections.py` on the extracted folder

So:

- use `main_defect_detection.py` for full-image TIFF workflows
- use `run_all_detections.py` when you already have extracted stripe/island region images
