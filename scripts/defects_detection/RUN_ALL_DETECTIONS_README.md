# `run_all_detections.py`

Standalone runner for already-extracted **island** or **stripe** images.
Accepts a **single image file** or a **folder** of images.

Filename drives routing (same rules as the full-image workflow):

| Filename contains | Image type | Detectors run |
|---|---|---|
| `island` | Island | `debris_island`, `overspray_island`, `line_defect` |
| `stripe` | Stripe | `stripe_misalignment`, `edge_roughness`, `overspray`, `surface_treatment`, `void`, `debris_stripe` |
| neither | Unknown | `surface_treatment` only |

## CLI reference

```bash
python run_all_detections.py -i <file_or_folder> [options]
```

| Flag | Description |
|---|---|
| `-i` / `--input` | Single image **or** folder of images (required; `--input_folder` still accepted) |
| `-o` / `--output` | Output directory (default: `output_YYYYMMDD_HHMMSS/` next to the input) |
| `--sensitivity` | `low` \| `medium` \| `high` (default: `medium`) |
| `--pattern` | `legacy` (single-band islands) \| `new` (dual-band islands). Default: `legacy`. Stripe images ignore this. |
| `--clear` | Clear scan material (gray background, fainter ink, lower SNR). Requires `--pattern new`. Island-only. |
| `--only` | Run only these detectors (subset of the strategy for this image type) |
| `--generate_report` | Also write a PDF summary report |

### `--only` detector keys

**Island:** `debris_island` `overspray_island` `line_defect`

**Stripe:** `stripe_misalignment` `edge_roughness` `overspray` `surface_treatment` `void` `debris_stripe`

## How to run — Island images

Island filenames must contain `island` (e.g. `KeyIsland.tiff`, `CyanIsland.tiff`).

```bash
cd scripts/defects_detection

# Legacy single-band island (default)
python run_all_detections.py -i "C:\path\island-black-blue.tiff"

# New dual-band island (4 vertical lines → 2 print regions)
python run_all_detections.py -i "C:\path\KeyIsland.tiff" --pattern new

# Clear material + new pattern (gray paper, fainter ink)
python run_all_detections.py -i "C:\path\ClearIsland.tiff" --pattern new --clear

# Faster iteration: missing nozzles / line misalignment only
python run_all_detections.py -i "C:\path\KeyIsland.tiff" --pattern new --only line_defect

# Debris + overspray only
python run_all_detections.py -i "C:\path\KeyIsland.tiff" --pattern new --only debris_island overspray_island

# Explicit output folder
python run_all_detections.py -i "C:\path\KeyIsland.tiff" --pattern new -o "C:\path\out_island"
```

With `--pattern new`, island detectors use the dual-band path
(`NewPatternDebrisIslandDetector`, `NewPatternOversprayIslandDetector`,
`NewPatternLineDefectDetector`). With `--clear`, thresholds are derived from
the measured background gray level of each image.

## How to run — Stripe images

Stripe filenames must contain `stripe` (e.g. `CyanStripe.tiff`, `KeyStripe.tiff`).
`--pattern` / `--clear` are not used for stripes.

```bash
cd scripts/defects_detection

# All stripe detectors
python run_all_detections.py -i "C:\path\CyanStripe.tiff"

# Stitch / roll (head calibration) only
python run_all_detections.py -i "C:\path\CyanStripe.tiff" --only stripe_misalignment

# Voids only
python run_all_detections.py -i "C:\path\CyanStripe.tiff" --only void

# Stitch/roll + voids together
python run_all_detections.py -i "C:\path\CyanStripe.tiff" --only stripe_misalignment void

# Debris inside the colored stripe
python run_all_detections.py -i "C:\path\MagentaStripe.tiff" --only debris_stripe

# Explicit output folder
python run_all_detections.py -i "C:\path\CyanStripe.tiff" -o "C:\path\out_stripe"
```

`stripe_misalignment` reports:

- **stitch** steps — abrupt lateral edge jumps at head boundaries
- **roll** errors — gradual lateral drift across a head segment

## Folder mode

Point `-i` at an extracted folder; each file is routed by its name:

```bash
python run_all_detections.py -i "C:\path\extracted" --pattern new
python run_all_detections.py -i "C:\path\extracted" --pattern new --clear
python run_all_detections.py --input_folder "C:\path\extracted" --sensitivity high --generate_report
```

## PowerShell note

Use a semicolon between `cd` and `python` (do not glue them together):

```powershell
cd scripts/defects_detection; python run_all_detections.py -i "C:\path\KeyIsland.tiff" --pattern new --only line_defect
```

## Outputs

Results land in `output_YYYYMMDD_HHMMSS/` next to the input (or under `-o`):

- per-image visualizations (`*_visualization.jpg` / `.tiff`)
- per-image `*_results.json`
- folder-level `defect_report.json`
- optional PDF when `--generate_report` is set

## How it works

1. Parse CLI (`-i`, `--pattern`, `--clear`, `--only`, …).
2. Classify each image with `ImageTypeClassifier` (filename heuristics).
3. Pick a strategy via `DetectionStrategyFactory`:
   - `StripeDetectionStrategy`
   - `IslandDetectionStrategy` / `NewPatternIslandDetectionStrategy`
   - `UnknownDetectionStrategy`
4. Build only the requested detectors (`DetectorFactory`).
5. `SingleImageProcessor` runs each detector and saves visualizations + defect dicts.
6. `ResultsSaver` writes JSON (and optional PDF).

## Relationship to `main_defect_detection.py`

| Use this | When |
|---|---|
| `main_defect_detection.py` | Full TIFF scan → extract regions → detect |
| `run_all_detections.py -i <image>` | Quick test of one island or stripe crop |
| `run_all_detections.py -i <folder>` | You already have extracted region images |

`main_defect_detection.py` forwards `--pattern` and `--clear` into this script
after extraction. See the project root `README.md` for the full-scan workflow.
