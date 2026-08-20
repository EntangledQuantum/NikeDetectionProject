# Defect Detection Algorithms

This document is the source of truth for **what the code does today**: every detector, the seed-free region extractor, shared preprocessing, and how to run it. Pixel thresholds live in [`config/detection_2400.json`](config/detection_2400.json). The runtime is `python -m nike_detection` (`nike_detection/cli.py`). Older scripts under `scripts/` are shims around that CLI.

The default island layout is **`--pattern new`** (dual-band, four verticals). `--pattern legacy` is the older single-band path. `--clear` adapts island thresholds for gray paper and requires `--pattern new`. Sensitivity presets (`low` / `medium` / `high`) tighten or loosen thresholds.

**Contents:** [scan pattern](#1-scan-pattern-change-why-new-algorithms-exist) · [shared infrastructure](#2-shared-infrastructure) · [new-pattern islands](#3-new-pattern-dual-band-island-algorithms) · [legacy islands](#4-legacy-island-algorithms) · [stripes](#5-stripe-region-algorithms) · [region boxes](#6-region-extraction-feeds-the-detectors) · [clear material](#7-clear-scan-material---clear) · [robustness](#8-robustness-legacy-vs-dual-band-on-the-new-pattern) · [still needed](#9-cross-cutting-updates-still-needed) · [pipeline](#10-shared-processing-parallelization-and-speed-ups) · [how to run](#11-how-to-invoke) · [file map](#12-file-map)

| Image type (filename token) | What runs |
|---|---|
| **`full`** | Measure island + stripe boxes for every colour (no seeds), then run the island and stripe detector sets on those crops |
| **`island`** | `debris_island`, `overspray_island`, `line_defect` — dual-band implementations when `--pattern new` |
| **`stripe`** | `stripe_misalignment`, `edge_roughness`, `void`, `debris_stripe`, `overspray` |
| **Unknown** | `surface_treatment` only (opt-in; not in the default stripe set) |

`--pattern` and `--clear` affect **islands only**. Stripe detectors are the same for both layouts. `--regions-only` stops after the boxes are written and does not run detectors.

---

## 1. Scan-pattern change (why new algorithms exist)

### Legacy island

A single print band spanning the image width, flanked by two vertical lines:

```
[V  horizontal print lines  V]
```

`LineDetector` scans inward from the extreme left and right edges, matches left/right endpoints by index, and treats the pair as one slanted line.

### New island (July visit)

Two print bands separated by a white gap, each flanked by a pair of vertical boundary lines (four vertical lines total):

```
[V  prints  V]   gap   [V  prints  V]
```

Applying the legacy stack unchanged fails in three ways:

1. Left-band endpoints are paired with right-band endpoints **across the empty gap** → invalid slopes and wrong trajectories.
2. The four vertical boundary lines are dark/continuous and get flagged as **debris or overspray**.
3. The print is **stippled** (dotted lines ~10 px thick). The legacy kernel-ratio walk produces mass false positives whenever the trajectory is a few pixels off or a stipple gap is locally sparse.

The dual-band detectors (`--pattern new`) add a band-splitting front end, mask the vertical lines, and replace the kernel walk with column-level ink evidence. Legacy detectors are left unchanged and selected with `--pattern legacy`. Config default is `--pattern new`.

---

## 2. Shared infrastructure

### 2.1 `BaseDetector` (`nike_detection/detectors/base_legacy.py`)

Common interface for detectors that support exclusion zones:

- Loads sibling JSON (`image.json`) with `exclusion_zones[].bounding_box_pixels`
- Point / region overlap checks
- Output as `(visualization_bgr, defects_list)`

**Limitation:** exclusion zones are only honored by the **legacy** island detectors (debris, overspray, line discovery). New-pattern island detectors accept `image_path` for interface compatibility but do not load or apply exclusion zones. Stripe detectors do not use them at all.

**Update needed:** wire exclusion zones into the dual-band detectors and at least into stripe misalignment / void (stamps, pen marks, crop artifacts).

### 2.2 `LineDetector` (`nike_detection/geometry/line_detector.py`) — legacy island primitive

Used by legacy Debris Island, Overspray Island, and Line Defect.

**Pipeline**

1. Scale kernels to image size vs a fixed ideal reference (`5163 × 44228`); also scale `Y_DELTA` with height.
2. Binarize (dark lines → white) at a **fixed** threshold of 127.
3. Scan left and right with multiple vertical columns of kernels:
   - Require `min_detection_count` hits at a Y to accept a line
   - After each hit, jump to the next expected Y window (`Y_DELTA_MIN`–`Y_DELTA_MAX`)
   - If nothing is found, insert a **ghost** line at the expected midpoint Y
   - Shift scan X around exclusion zones
4. Match left/right detections by index; compute slope  
   `slope = (y_right - y_left) / (x_right - x_left)`  
   Valid if `SLOPE_MIN ≤ slope ≤ SLOPE_MAX` (≈ `0.0150`–`0.0219`).

**Limitations**

- Assumes **one** band spanning the full width.
- Slope window is a calibration assumption; print-head drift outside that window is rejected.
- Ghost lines guess spacing from neighboring hits — a fully missing line in a sparse region can be placed wrong.
- Fixed 127 binarization **inverts on gray clear material**.
- Kernel occupancy is brittle on stippled (dotted) ink.

**Update needed:** do not extend this primitive to the new pattern. New-pattern line work should keep using `IslandLineExtractor`. If legacy scans still matter, make the binarization threshold image-adaptive.

### 2.3 `material_profile.py` (`nike_detection/geometry/material_profile.py`) — clear-scan background

Clear material has a mid-gray background (~140 instead of white), fainter ink, and lower SNR. Fixed white-paper thresholds are meaningless.

- **Background level** = median of a subsampled grayscale view (ink is a small area, so the median is robust).
- Thresholds are then `background − offset`.

`--clear` is only valid with `--pattern new`.

---

## 3. New-pattern (dual-band) island algorithms

Selected with `--pattern new`. Same detector keys (`debris_island`, `overspray_island`, `line_defect`) so routing and output names stay identical.

```
New-pattern island image
    │
    ├─ VerticalBandDetector → 4 vertical lines + 2 band crops
    │
    ├─ Debris / Overspray
    │     BandLineDetector + BandLineRefiner  (or IslandLineExtractor if --clear)
    │     paint horizontal lines + vertical envelopes white
    │     run legacy debris / overspray helpers on the full cleaned image
    │
    └─ Line Defect
          IslandLineExtractor per band (shear + per-line slope/intercept + local path)
          missing (red) / hazy (yellow) / stitch (blue) / high-density regions
```

### 3.1 Vertical band detector (`nike_detection/geometry/vertical_band_detector.py`)

Finds the **4 vertical boundary lines** first, then derives the two bands. Vertical lines can be speckled, have imperfect ends, and drift sideways, so detection is morphological rather than a raw column-density profile.

**Pipeline**

1. Grayscale → `THRESH_BINARY_INV` with a generous ink threshold (default 170). If fewer than 4 candidates survive, retry at 200 then 230. In `--clear` mode the ladder is `bg−25 → bg−18 → bg−12` (never past the background — a white-paper threshold like 200 would turn the whole gray field into “ink”).
2. Small **horizontal CLOSE** to connect the sideways spread of a speckled vertical line.
3. **Vertical CLOSE** with a kernel well **below** the horizontal-line spacing (bridges gaps in a dotted vertical without merging two adjacent print lines). Spacing is **measured** from the image’s row ink profile.
4. **Vertical OPEN** with a kernel well **above** horizontal-line thickness — removes every horizontal crossing, keeping only tall vertical structures.
5. Horizontal dilation + column coverage profile → candidates with center, x-extent envelope `x0..x1`, y-extent, and coverage. Must cover ≥ `min_coverage` of the height and not be wider than `max_thickness_fraction`.
6. **4-of-N selection:** if more than 4 candidates, every 4-combination is scored by coverage and pattern geometry (two similar-width bands, narrower central gap).
7. **Bands** = region between each selected pair (v1–v2 and v3–v4), trimmed `inner_margin` px inside the measured envelopes.
8. Fallbacks: if 4 lines cannot be selected → bands from ink-content runs (found vlines still trim the crops); no content → single full-width band (legacy degrade).

Also provides `paint_vertical_line_regions(gray, vlines, pad)` which paints the **measured x-extent envelope** plus a speckle-halo pad so noisy verticals are never counted as debris/overspray.

**Limitations**

- Geometry is hard-wired to **exactly two bands / four verticals**. A 3-band or 1-band new layout would need a new selector.
- Heavily damaged or missing verticals fall back to content runs, which can crop too tightly or include the gap.
- Drift that is large compared to the horizontal CLOSE kernel can split one vertical into two candidates and confuse 4-of-N selection.
- Does not inspect the vertical lines themselves for defects (by design).

**Updates needed**

- Validate 4-of-N scoring on noisy Key / Cyan / Magenta / Yellow islands and on clear material; tune `min_coverage` if a faint vertical is dropped.
- Add an explicit “band count” config if future patterns are not always two bands.
- Persist detected band bounds in the per-image JSON so operators can audit crops without re-running.

### 3.2 Band line detector (`nike_detection/geometry/band_line_detector.py`)

Thin `LineDetector` subclass: runs the legacy kernel scan on a **band crop**, but scales kernel width and number of scan columns from the **full image width** (`reference_width`) so per-band sensitivity matches the full-image detector. Height-based `Y_DELTA` and slope validation are inherited.

**Limitations:** inherits every `LineDetector` limitation (fixed 127 binarization, slope window, ghosts). On white paper this is acceptable for *painting lines out*; it is **not** used for new-pattern missing-nozzle decisions.

**Update needed:** in `--clear` mode this class is already skipped (debris/overspray use `IslandLineExtractor`). Consider using the extractor for white-paper debris/overspray as well, so there is one line-finding path.

### 3.3 Band line refiner (`nike_detection/geometry/band_line_refiner.py`)

Turns coarse left/right matches into **full-width fitted trajectories**:

1. Sample actual ink along each coarse trajectory (center-of-mass inside a vertical window < ½ spacing).
2. Robust-fit a straight line per trajectory (least squares + iterative sigma-clipping). Each line gets its **own** slope/intercept — constant slope/spacing is not assumed.
3. Extrapolate each fit to the band’s inner bounds `x0..x1`. Missing ink at a line start/end still gets a full-width path (the legacy walk silently started at the first ink it saw).
4. Lines with too little ink are interpolated from nearest fitted neighbors; if the coarse detector found nothing, trajectories are seeded from the row ink profile.
5. Near-duplicate trajectories (< ½ spacing apart mid-band) are deduplicated.

Used by new-pattern **debris and overspray** (to paint lines out). New-pattern **line defect** no longer uses this; it uses `IslandLineExtractor` instead.

**Limitations**

- Depends on a decent coarse `BandLineDetector` seed. If the coarse scan misses a whole family of lines, interpolation can invent trajectories in the wrong place.
- Straight-line model cannot represent strongly curved or kinked print.

**Update needed:** optional quadratic/piecewise fit if roll inside a band is large enough to leave residual ink after painting.

### 3.4 Island line extractor (`nike_detection/geometry/island_line_extractor.py`)

Precursor for new-pattern **line defects** (and for clear-mode debris/overspray line masking). Avoids kernel scanning entirely.

**Pipeline**

1. **Ink binarization:** Otsu on a subsample, clamped to `[120, 200]`. Clear mode: 3×3 median despeckle, clamp to `[bg−85, bg−15]`. Clear mode also requires **2** ink pixels per column (one speck must not bridge a real gap).
2. **Global slope by shear search:** candidate slopes shear the binary image column-wise; the slope that maximizes sharpness (sum of squares) of the row ink profile wins. Coarse sweep (−0.005…0.035, step 0.002) then fine (step 0.0002). No calibration table.
3. **Shear** so every line becomes a narrow horizontal row band.
4. **Line rows:** runs of the sheared row profile above 25% of its P99; weighted centroids = line rows; median distance = **measured spacing**. Peak gaps ≈ 1.6× spacing get synthetic *inserted* rows so **fully missing lines are still evaluated**.
5. **Per-line residual fit + local trajectory:** ±0.45·spacing window, per-column ink centroids, sigma-clipped least squares — each line gets its own slope/intercept. A **local centroid path** (interpolated, lightly median-smoothed) is then used as the mask center so stitch zig-zags and roll stay inside the corridor.
6. **Per-column statistics** inside ±0.3·spacing of that local path: ink presence/count, number of separate ink runs, hollow interior, centroid deviation from the *straight* fit (jaggedness). Mapped back to original coordinates via the stored shear shift. `line_y` follows the local path so debris/overspray paint-out uses the same mask.

   Jaggedness is two numbers per line, used later for stitch (not for missing nozzles):
   - `jagged_rms` — RMS of (local trajectory − straight fit). A slow bow from roll scores high here.
   - `jagged_hf` — RMS after subtracting a ~0.25·spacing smoother. Only the zig-zag at a head join survives.

**Limitations**

- Slope search range is bounded; a print with inverted or much steeper slant will pick a wrong global slope and scramble every line.
- Inserted (fully missing) lines assume locally regular spacing. A true double-gap or a compressed region can insert a row on the wrong Y.
- Corridor of 0.3·spacing can pick up a neighbor if spacing is mis-measured (e.g. every other line faint).
- Otsu + clamps can still fail on extremely low-contrast clear scans or on Yellow (very light ink on white).

**Updates needed**

- Widen or auto-expand the slope search if the sharpness peak sits on the range edge.
- Special-case **Yellow** (and possibly Magenta) with a chroma-aware binarization instead of grayscale Otsu.
- Expose measured `slope` / `spacing` in the results JSON (partially done via band summaries) and fail loudly when extraction returns `None` rather than silently skipping a band.

### 3.5 New-pattern debris (`nike_detection/detectors/island_new/debris.py`)

Debris can occur **anywhere** — inside bands, in the gap, or outside the boundary lines — so this runs on the **full image** after masking printed structure.

**Pipeline**

1. Detect bands + verticals.
2. Per band, refine horizontal lines and paint them white on the full grayscale image (legacy thickness ×2), extended ~20 px past band bounds. Clear mode: locate lines with `IslandLineExtractor` and paint with the **background gray** (painting white on gray would create bright artifacts).
3. Paint vertical envelopes white (`paint_vertical_line_regions`).
4. Reuse `DebrisIslandDetector.detect_debris`: threshold → 3×3 close → contours → area filter. Clear mode replaces the fixed `background_threshold` with `bg − 35/45/55` (high/medium/low) and median-filters before thresholding.

**Limitations**

- Incomplete line painting (missed lines, under-thick paint) leaves print ink that is then scored as debris.
- Over-thick paint can hide real debris sitting on a line.
- Full-image grouping can flag scanner dirt in the paper margins that operators may not care about.
- No exclusion-zone support.
- White-paper path still uses `BandLineDetector` (fixed 127), not the extractor.

**Updates needed**

- Unify line masking on `IslandLineExtractor` for both white and clear.
- Honor exclusion zones.
- Optional “in-band only” mode if margin detections are too noisy.
- Tune paint thickness per color / DPI; 20 px extension is a guess.

### 3.6 New-pattern overspray (`nike_detection/detectors/island_new/overspray.py`)

Same masking front end as debris (paint thickness ×3). Then reuse `detect_colored_regions` + `group_nearby_regions` on the full cleaned image.

Clear mode: color threshold `bg − 35/45/60`; 5×5 median before region growing.

**Limitations**

- Proximity grouping (`overspray_max_distance` up to 1000 px at medium) can merge unrelated blobs across the central gap into one “overspray”.
- Medium min area is 5000 px² — small spray is missed; high (100 px²) can flood with noise.
- Colored-mask logic is “darker than white,” not “wrong hue,” so dark debris and overspray overlap as classes.
- Same exclusion-zone and Yellow-ink gaps as debris.

**Updates needed**

- Restrict grouping to within a band / gap / margin region so gap-crossing merges stop.
- Hue- or chroma-based overspray vs darkness-based debris, especially on Cyan/Magenta.
- Recalibrate min-area / max-distance on new-pattern samples; the legacy medium settings were for a different crop size.

### 3.7 New-pattern line defect (`nike_detection/detectors/island_new/line_defect.py`)

Scoring runs **only inside the two print bands** — never on the verticals or the gap. Config `geometry.num_heads` (3) caps how many stitch zones can be reported.

#### Why this is measured in ink density, not grayscale

Every earlier version thresholded grayscale, which quietly made the detector colour dependent. On white paper the luminance contrast between paper and a saturated line is:

| Ink | Grayscale contrast | Contrast in the channel it absorbs |
|---|---|---|
| Key | ~180 | ~180 (neutral) |
| Magenta | 127 | 188 (green) |
| Cyan | 85 | 219 (red) |
| **Yellow** | **21** | **155 (blue)** |

Yellow is the pathological case: its Otsu threshold lands *above* the `[120, 200]` clamp the extractor applied, so it marked **0.06%** of yellow pixels as ink — the region was effectively invisible. Meanwhile the same settings flooded Cyan and Magenta with false haze. No single grayscale number can serve both.

`nike_detection/geometry/ink_density.py` removes colour from the problem instead of tuning around it. Per region it measures the paper white, fits the ink's own **absorbance direction** from the darkest pixels, and normalizes by the saturated-core level, producing

```
alpha = 0.0  paper          alpha = 1.0  healthy, fully saturated line core
```

Because paper, direction and core are all *measured per region*, thresholds expressed in alpha carry the same meaning on every ink — **there is no per-colour constant anywhere in the detector.** The map is stored as uint8 (`SCALE` counts per unit alpha), so it costs exactly what the binary mask it replaced cost, while carrying 256 density levels instead of 1 bit. Band and vertical-line geometry run on the same map (via `as_pseudo_gray`), so a Yellow island's boundary lines are as findable as a Key island's.

#### The two ratios that decide everything

`nike_detection/geometry/line_profile.py` reduces each line to three per-column series in alpha units — `mass` (total ink), `peak` (densest pixel) and `center` (ink-weighted row) — then normalizes the first two against a baseline measured from the band itself:

```
coverage = mass / baseline_mass     ~1 healthy, ~0 no ink
density  = peak / baseline_peak     ~1 healthy, low = pale and spread
```

Two ratios separate the defects because a missing nozzle and a smear differ in *kind*, not degree:

| Condition | coverage | density | Verdict |
|---|---|---|---|
| healthy print | ~1.0 | ~1.0 | — |
| nozzle not firing | ~0.0 | ~0.0 | `missing_line` (red) |
| ink fired, smeared | ~0.5–1.0 | low | `misaligned_line` (yellow) |
| thin but crisp | lower | ~1.0 | — (not a defect) |

That last row is what the previous rule could not express: it keyed haze off ink *thickness*, so Cyan and Magenta stipple constantly tripped it. Peak density does not care how much ink landed, only whether it stayed concentrated — which is exactly what a smear destroys.

The baseline is the band's own upper-quartile level with a bounded per-line correction, so slow drift in illumination or ink laydown cancels out, while a *globally* faint line is still flagged rather than normalizing itself into looking healthy.

#### Defect decisions

| Type | Color | Rule |
|---|---|---|
| `missing_line` | red | `coverage < missing_level` over a run longer than `min_gap_fraction × expected_gap`. `expected_gap` is **90 px at 2400 DPI**, scaled by `measured_spacing / 100`. Because coverage is a *continuous* ink measure, a faint ghost line that a binary threshold counted as "ink present" now correctly reads as missing — this is what fixed the previous false misses. Fully missing lines are inserted from spacing and scored as a whole-line gap. |
| `misaligned_line` | yellow | Ink landed but lost its core: `density < haze_level` while `coverage ≥ missing_level`, persisting ≥ `haze_min_fraction × spacing`. Carries a continuous `severity` (0–1) and a `mild`/`moderate`/`severe` grade, so a faint haze and a heavy smear are separable downstream. |
| `stitch_error` | blue | Short-range trajectory wander at a head join. Waviness is the RMS of the trajectory residual (anything slower than ~4 spacings removed, so paper roll does not count), normalized by spacing, and **measured only over healthy columns** — smeared ink drags the centroid about without the printed core moving, which is how a hazy band used to be mistaken for a join. A line qualifies by standing out from its own band (`median + 3×MAD`, and ≥1.6× median); the absolute `stitch_wave` floor only applies to crops too short to estimate that spread. Adjacent candidates are clustered and at most `num_heads − 1` are kept, so the physical constraint is enforced by construction. Expected 1/3 and 2/3 positions act as a tie-break only, never a hard gate. |
| `high_density_region` | orange | Missing pixels splatted into a 16×-downsampled accumulator, Gaussian-smoothed over ~2 spacings, thresholded at 40% of peak density. |

**Five thresholds, down from thirteen.** `missing_level`, `haze_level`, `min_gap_fraction`, `haze_min_fraction`, `stitch_wave` — every one a dimensionless ratio, so none needs retuning for a different ink, exposure or scan resolution. (Replaced: `split_min_fraction`, `dev_min_fraction`, `split_min_hollow`, `dev_abs_floor`, `dev_thickness_factor`, `hazy_thickness_factor`, `hazy_weak_factor`, `stitch_rms_fraction`, `stitch_score_ratio`, `stitch_max_lines_per_zone`.)

A `*_newpattern_detected_lines.jpg` debug image is always saved (verticals magenta, local trajectories green/orange, inserted fully-missing lines red).

#### Validation

Synthetic islands with known defects (`nike_detection/testing/synthetic_island.py`, scored by `testing/scoring.py`) sweep all four inks — run `python scripts/line_defect_sweep.py`. Results are **identical across Key / Cyan / Magenta / Yellow**, which is the colour-independence claim tested directly rather than inferred:

| Defect | Precision | Recall |
|---|---|---|
| missing | 1.00 | 1.00 |
| haze | 0.92 | 1.00 |
| stitch | 1.00 | 1.00 |

Healthy print reports **zero** defects on every ink. Stitch detection starts at ~2 px of wander with severity rising smoothly (2 px → 0.07, 3 px → 0.34, 5 px → 0.91).

On the real 2400-DPI scans all three colours independently place both stitch zones at lines 113 and 229 of ~343 — positions **0.33 and 0.67**, the expected head joins — and both bands of each scan agree. Haze on the Cyan island fell from **1405 undifferentiated marks to 411, of which 22 are moderate-or-severe**. Runtime for a 169-megapixel region is **~1.0 s, down from ~29.7 s**.

`python scripts/line_defect_lab.py <folder>` renders annotated overlays for eyeballing real crops.

**Limitations**

- Vertical lines and the central gap are never inspected (by design).
- `image_path` / exclusion zones unused; a stamp overlapping a band will be scored as missing ink.
- Inserted whole-line defects depend on spacing regularity.
- Density blobs are relative to the *peak* in that image, so a uniformly bad print may produce no hotspot.
- A crop with fewer than ~6 lines per band cannot estimate its own waviness spread, so stitch falls back to the absolute floor and is correspondingly less reliable there.
- Very long defects (beyond a whole line) are bounded by the per-line baseline correction; a band that is uniformly hazy end to end normalizes partly against itself.

**Updates needed**

- Exclusion zones and optional ignore-margins.
- Absolute density floor so uniformly bad bands still report a hotspot.
- Report nozzle estimates in physical units (need DPI + nozzle pitch calibration).
- Consider scoring the vertical boundary lines for breaks (currently structural only).

---

## 4. Legacy island algorithms

Still selected with `--pattern legacy`. Shared `LineDetector` locates slanted print lines. Config default is `--pattern new`.

```
Legacy island
    │
    ├─ LineDetector → matched left/right endpoints
    │
    ├─ Debris ── paint valid-slope lines white → dark threshold → contours
    ├─ Overspray ── paint lines white → colored mask → group blobs
    └─ Line Defect ── binary line map → walk each line → missing / jagged
```

### 4.1 Debris Island (`nike_detection/detectors/island_legacy/debris.py`)

1. Locate lines via `LineDetector` (exclusion zones from sibling JSON).
2. Paint `valid_slope` lines white, thickness `line_thickness * 2`. Invalid-slope matches are **left in place**.
3. `THRESH_BINARY_INV` at `background_threshold` — darker than paper/ink grey → candidate debris.
4. Elliptical close `(3, 3)`; contours with area > `debris_min_area`; drop exclusion overlaps.

| Sensitivity | Background thresh | Min area (px²) | Paint thickness |
|---|---|---|---|
| high | 140 | 5 | 5 |
| medium | 120 | 10 | 20 |
| low | 100 | 50 | 5 |

**Limitations:** invalid-slope lines become false debris; paint-thickness jump (5 vs 20) is inconsistent across sensitivities; fixed greyscale threshold fails on clear/gray paper.

**Updates needed:** do not use this detector on dual-band images (already gated by `--pattern`). If legacy remains in production, make paint thickness monotonic with sensitivity and skip invalid-slope leftovers more carefully (or paint them with a dashed mask).

### 4.2 Overspray Island (`nike_detection/detectors/island_legacy/overspray.py`)

Same line removal with thickness ×3. Colored mask at `color_threshold = 180 - background_threshold`. Aggressive dilate/close/erode, then greedy centroid grouping within `overspray_max_distance`.

| Sensitivity | `background_threshold` | Min area | Max group distance | Paint thickness |
|---|---|---|---|---|
| high | 5 | 100 | 300 | 5 |
| medium | 50 | 5000 | 1000 | 15 |
| low | 20 | 1000 | 800 | 5 |

**Limitations:** sensitivity table is not monotonic (medium min-area 5000 is *stricter* than low at 1000). Grouping distance of 1000 px can swallow half an island. Same “dark ≠ colored” confusion as the new-pattern variant.

**Updates needed:** rebuild the sensitivity ladder so high ⊃ medium ⊃ low; add hue/chroma; cap merge distance by island width.

### 4.3 Line Defect (`nike_detection/detectors/island_legacy/line_defect.py`)

Does **not** remove lines. Walks each matched trajectory:

1. Linear interpolation from left to right endpoint (including **ghost** endpoints and **invalid-slope** matches).
2. Square kernel occupancy vs `line_threshold`.
3. **Missing:** ink-free run ≥ `min_gap_size`.
4. **Jagged:** `|actual_y − last_found_y| > jagged_threshold` (vs previous detection, not vs expected Y).

| Sensitivity | kernel | step | occupancy | min gap | jagged ΔY |
|---|---|---|---|---|---|
| high | 15 | 10 | 0.25 | 50 | 10 |
| medium | 30 | 25 | 0.20 | 100 | 5 |
| low | 50 | 40 | 0.15 | 200 | 25 |

**Limitations**

- Mass false positives on stippled new-pattern ink (reason the extractor replaced this).
- Medium jagged threshold (5 px) is *more* sensitive than high (10 px) — another non-monotonic ladder.
- Exclusion zones affect discovery placement, not individual missing/jagged hits.
- Ghost trajectories can invent gaps where no line was ever printed.

**Updates needed:** keep this path for legacy single-band only. If it stays, fix the jagged ladder and skip invalid-slope walks (new-pattern already does the equivalent).

---

## 5. Stripe region algorithms

Stripe detectors are **layout-agnostic**: they derive geometry from the crop itself. On a `full` scan the seed-free extractor (§6.1) supplies the stripe boxes; `--extract` still uses `geometry.num_heads` (3) and `head_height` from config. Misalignment works at 1 px resolution for both 3-head and 4-head patterns.

### 5.1 Stripe misalignment (`nike_detection/detectors/stripe/misalignment.py`)

Head-calibration errors from the stripe’s edge trajectory:

- **Stitch:** abrupt lateral step at a head boundary.
- **Roll:** gradual lateral drift across a head segment.

**Pipeline**

1. Binarize stripe vs paper at the mid-level between interior and paper (from the column intensity profile).
2. Per row, trace left/right boundary of the contiguous ink run **anchored from inside the stripe, walking outward** (pen marks on the paper cannot hijack the edge). Median-filter profiles (31-row window).
3. **Stitch:** at sampled rows, step = `median(edge below) − median(edge above)` (300-row windows + guard). Local extrema with `|step| ≥ step_threshold`; NMS within 2 windows. Both-edge steps are confident; single-edge steps must exceed `1.6×` threshold (ragged fringes are one-sided).
4. **Roll:** between consecutive stitches, sigma-clipped line-fit of stripe center vs y; report `roll_error` when total drift > `roll_threshold`.

| Sensitivity | Stitch step (px) | Roll drift (px) |
|---|---|---|
| high | 3 | 5 |
| medium | 5 | 8 |
| low | 10 | 15 |

**Limitations**

- Window of 300 rows assumes head height is large compared to that; a much shorter crop can miss stitches or invent them at the image ends.
- Severe voids or fade-outs that break the interior-anchored walk produce NaN edges and silent gaps in the profile.
- Single-edge 1.6× rule can miss a true one-sided stitch (one edge ragged, the other real).
- No exclusion zones; a label overlapping the stripe edge will look like a stitch.
- Thresholds are in pixels, not mm — they should scale with DPI (currently the same at 2400 and 4800).

**Updates needed**

- Scale `step_threshold` / `roll_threshold` / window size with DPI (or with measured stripe width).
- Convert reported `step_px` to mm in the JSON.
- Optional known head-boundary priors from `num_heads` + `head_height` to suppress mid-head false stitches, without requiring them (the detector is intentionally head-count agnostic).

### 5.1b Stripe Edge Roughness (`nike_detection/detectors/stripe/roughness.py`)

**Goal:** Flag and quantify high-frequency jaggedness (saw-tooth / scalloped fringe) on the left and right stripe edges. A rough left edge with a smooth right edge is the intended case; both edges are scored independently.

**Must not confuse roughness with calibration:** stitch (step at a head joint) and roll (slow drift along a head) are removed before scoring. Detection runs along the **entire** stripe length, not only at stitch locations.

**Pipeline**

1. Trace raw **sub-pixel** left/right edges with the same interior-anchored walk as misalignment (paper-side marks cannot hijack the edge). Do **not** apply the 31-row median that misalignment uses to erase fringe.
2. Detect stitch rows on a slow (median-31) copy of the profile. Split the stripe into per-head segments at those rows.
3. Per segment, **robust linear fit** (sigma-clipped) = roll. Residual = raw edge − fit, with a short guard around each stitch so the step itself never enters the roughness residual.
4. Wide median **high-pass** (81 rows) removes leftover slow bow.
5. **Quantify** the residual with robust stats: MAD, σ = 1.4826·MAD, P95 of |residual|, peak-to-peak (P5–P95), and RMS (reported only).
6. Sliding windows (256 rows, stride 64) flag a span when `MAD ≥ mad_threshold` **and** `P95 ≥ p95_threshold`. Windows where P95 is many times MAD (a debris spike, not a saw-tooth) are ignored. Adjacent hits merge.

| Sensitivity | MAD (px) | P95 (px) |
|---|---|---|
| high | 1.30 | 4.2 |
| medium | 1.70 | 5.0 |
| low | 2.50 | 7.2 |

**Outputs**

- `edge_roughness_summary` — always; `left` / `right` metrics plus which edges were flagged.
- `edge_roughness` — one per flagged (edge, y0–y1) span, with `roughness_sigma_px`, `mad_px`, `p95_px`, `rms_px`, `peak_to_peak_px`.

Visualization: green edge traces, **red** overpaint on flagged spans, per-edge score labels (`ROUGH` vs `smooth`).

**Limitations**

- Thresholds are in pixels (not mm); 4800 DPI teeth look twice as large.
- Sub-pixel refine assumes a monotonic grey crossing; a very blurry halo can shrink measured amplitude.
- A single large scallop longer than the 81-row high-pass is treated as roll-like and attenuated.
- No exclusion zones.

**Updates needed:** DPI-scale the MAD/P95 thresholds (or report mm).

### 5.2 Overspray — scatter grid (`nike_detection/detectors/stripe/overspray.py`)

Scattered spray (not solid ink) via spatial scatter of white pixels.

1. Grayscale → CLAHE → adaptive threshold → invert.
2. Slide a square kernel on a grid.
3. Scatter metric from coordinate std + pairwise distance variance; boost mid-density (~5–30%); reject too-sparse or too-solid kernels.
4. Flag scatter > `scatter_threshold`; morphologically merge adjacent hits.

| Sensitivity | Kernel / step | Scatter thresh | Min white ratio |
|---|---|---|---|
| high | 20 / 15 | 0.2 | 0.03 |
| medium | **500 / 500** | 0.3 | 0.05 |
| low | 50 / 50 | 0.5 | 0.10 |

**Limitations**

- Medium kernel of **500 px** is an outlier (likely leftover from an experiment). It will miss small spray that high/low would see, and is not a monotonic sensitivity ladder.
- Adaptive threshold + CLAHE can turn paper texture or stripe speckle into “spray.”
- Does not know where the stripe is, so interior texture of a noisy stripe can flag as overspray.
- No DPI scaling.

**Updates needed:** rebuild the medium preset (something between 20 and 50, with overlap); optionally mask to a band *outside* the solid stripe; add DPI-aware kernel size.

### 5.3 Void (`nike_detection/detectors/stripe/void.py`)

Compact missing-ink patches inside a **solid-color** vertical stripe (color drifted toward paper).

**Pipeline**

1. LAB; stripe left/right from column chroma (fallback: dark columns for black / Key).
2. Shrink inward (`inner_pad`) to ignore ragged edges.
3. Median LAB interior = stripe reference; median LAB outside = paper reference.
4. Per-pixel **voidness** = projection of `(pixel − stripe_ref)` onto `(paper_ref − stripe_ref)`, normalized to `[0, 1]`.
5. Hysteresis on a 5×5-blurred voidness map: strong seeds at `median + k·MAD·1.4826` (floor 0.12), grown into `median + 0.55·k·MAD`; keep weak components only if they contain a strong seed.
6. Morph close then open; kernels scale with stripe width.
7. Keep components with area / extent / aspect consistent with voids; reject huge fade-outs. Min area scales with `stripe_width²` (DPI proxy), **not** crop height.

| Sensitivity | MAD multiplier `k` | Min area floor / frac (× stripe_w²) |
|---|---|---|
| high | 5.0 | 80 / 1e-4 |
| medium | 7.0 | 150 / 2e-4 |
| low | 10.0 | 300 / 4e-4 |

**Limitations**

- Designed for **solid** stripes. A textured or stippled stripe will score every speckle gap as void.
- Inner pad hides voids that sit on the stripe edge (often the real defect).
- Yellow on white has weak chroma; bounds and voidness axis can collapse.
- Clear / gray paper: paper reference is closer to the stripe, shrinking the voidness dynamic range.
- `max_area_frac` of 10% of the *image* can still admit large fade-outs on short crops.

**Updates needed:** validate on new-pattern Cyan/Magenta/Key/Yellow stripes; add a clear-material path (paper/stripe refs from `material_profile`); optional edge-aware pad that still scores the last few pixels with a different threshold.

### 5.4 Debris Stripe (`nike_detection/detectors/stripe/debris.py`)

Dark, near-black spots inside colored stripes (color-agnostic).

**Pipeline**

1. Stripe bounds from LAB chroma (dark fallback); pad inward.
2. Combined **debris score** inside the stripe:
   - 0.40 multi-scale black-hat (L and HSV-V)
   - 0.35 absolute darkness vs stripe luminance
   - 0.20 saturation drop vs stripe S
   - 0.05 chroma drop
3. Strong / weak masks from `median + k·σ` with floors; promote weak components that contain ≥1 strong pixel.
4. Morph open + close (sizes scale with stripe width); filter by area and darkness/saturation.

| Sensitivity | Score `k` | Strong / weak floors | Min area floor / frac |
|---|---|---|---|
| high | 4.5 | 0.34 / 0.22 | 4 / 2e-5 |
| medium | 5.5 | 0.40 / 0.26 | 8 / 5e-5 |
| low | 7.0 | 0.48 / 0.32 | 20 / 1.2e-4 |

**Limitations**

- On **Key / black** stripes, “darker than stripe” has little headroom; black-hat still helps but false negatives rise.
- Saturation/chroma terms are weak on already-unsaturated Key.
- Inner pad can hide debris on the stripe edge.
- Can confuse voids (missing ink → lighter) if a void has a dark speck; generally the two detectors are complementary (void = toward paper, debris = toward black).

**Updates needed:** Key-specific weights (drop sat/chroma, raise black-hat); clear-material luminance stats; exclusion zones for handwritten marks.

### 5.5 Surface treatment (`nike_detection/detectors/stripe/surface_treatment.py`)

Poor surface energy: irregular coalesced drops and missing-ink areas. Used for **unknown** filenames and currently also listed in the stripe strategy.

**Pipeline**

1. CLAHE.
2. **High-contrast drops:** local stddev → morph close → keep irregular (high eccentricity / low solidity) large regions.
3. **Void areas:** adaptive light regions ∩ dilated Otsu print coverage; intensity must exceed surroundings.
4. Optional FFT radial-profile texture anomalies (computed; not primary output).
5. Highlight in green.

**Limitations**

- Overlaps conceptually with stripe void detection; running both on every stripe produces duplicate / conflicting boxes.
- Thresholds (`contrast_threshold=50`, sizes 150/300) are not sensitivity- or DPI-aware.
- FFT path is unused in the reported defects.
- Easy false positives on any high-texture region (including healthy stipple if this were ever pointed at an island).

**Updates needed:** decide whether surface treatment stays on the default stripe set or becomes `--only`; add sensitivity/DPI; drop or wire up the FFT feature.

---

## 6. Region extraction (feeds the detectors)

On a `full` scan the pipeline **measures** island and stripe boxes from the print, then crops those boxes in memory and runs the detector sets. There is no seed coordinate and no requirement that Key start at a known pixel. `--extract` is a separate, older path that writes TIFF crops from a parametric template; use it only when you want files on disk, not when you want accurate boxes.

### 6.1 Seed-free full-scan boxes (`nike_detection/geometry/full_region_detector.py`)

When the input filename contains `full` (or `--regions-only` is used), the island and stripe rectangles of **every colour on the sheet** are measured from the print. **There are no seed boxes and no positional priors**: the search starts at (0, 0), and a sheet holding one colour and a sheet holding four are handled by the same code.

The nominal layout lives in `config/detection_2400.json` → `region_reference` and is only used to disambiguate candidates, validate the result, and predict an edge whose print is missing:

```
   colour k                                      colour k+1
|<---- island 5100 ---->|<-gap 580->|<-stripe->|<-250->|<---- island ...
| V dashed jet lines V | V ... V |  |##########|
```

Both regions of one colour are 33000 px tall and share a y range. Colours are named left to right from `geometry.colors` (Key, Cyan, Magenta, Yellow); successive colours may sit up to `color_y_tolerance` (±200 px) higher or lower, so **each colour gets its own y range**.

**Step 0 — ink is the darkest channel, not luminance.** Yellow is nearly invisible in grayscale (≈226 against 250 paper) but its blue channel is as dark as any other ink. Taking `min(B, G, R)` per pixel puts black, cyan, magenta and yellow on the same footing, which is what allows one global threshold to work across a four-colour sheet.

**Step 1 — coarse ink map.** The scan is area-downsampled (÷8 at 2400 DPI, driven by `stripe_width` so the algorithm is not tied to a DPI) and converted to ink density, `0` = paper, `1` = the darkest ink on the sheet. Paper level and contrast are measured per scan (p95 / p2). The downsample runs a band of rows at a time, so a memory-mapped multi-gigabyte sheet is profiled without ever being held in RAM.

**Step 2 — the solid bars.** Column density near `1.0` over the full height happens nowhere else, so *every* stripe is found first, and each one anchors one colour's block. A run only qualifies as a bar if its *mean* density is high, which is what rules out two island verticals whose low-ink gutter sits between them.

**Step 3 — the vertical lines.** On real scans these are faint, speckled and drift sideways over 33000 rows, so their raw column density is no better than the dashed horizontal print. Instead the coarse ink map is binarized and **opened along y** with a kernel far taller than a horizontal line (~height/100) and far shorter than a region: the horizontal print disappears entirely, and what is left is per-column *vertical structure*. Candidates are collected at a permissive floor so a faint colour is not lost beside a strong one, but each line's extent is taken at half of its own peak, which keeps the envelope tight.

**Step 4 — assembling the colour blocks, both directions.** Two independent pairings are merged:

- **Left-to-right:** each solid bar anchors a colour; the island is the vertical pair whose separation matches `island_width` *and* whose inner edge sits `gap` from that bar. Both constraints are needed on a multi-colour sheet: one colour's inner line and the next colour's outer line are also roughly an island apart, so matching on width alone would straddle two colours.
- **Right-to-left:** island verticals are paired first, then a bar is attached if one sits where the stripe should be; otherwise the stripe is predicted from the island.

Overlapping LTR/RTL blocks are merged by keeping **measured** edges (vertical-line / solid-bar) over predicted ones, and among measured edges taking the **outer** value so a missing corner cannot shrink the block. Duplicate blocks from dual-band inner verticals or a large stitch are collapsed. Then:

- a colour whose **bar never printed** is recovered from any vertical pair not already claimed, and its stripe is predicted from its island;
- a colour that **printed nothing at all** is inserted from the lattice pitch of its neighbours, so the remaining colours keep their correct names.

**Step 5 — y extent, per colour.** A colour's island and stripe are printed in one pass, so they share one y range. The covering union of the island cluster and the bar is used when that height still matches the reference (tolerance `0.12`); if the union would *inflate* the box (overspray above the island), the bar wins. Missing dashed lines at the top or bottom make the island cluster shorter, not taller, so the union stays on the stripe/verticals. A colour recovered from the lattice borrows the median y of the colours that did print.

**Step 6 — station snap, four corners, constraint cover.** Coarse axis-aligned edges are not trusted at the corners: a feeble vertical, a blank top strip of dashed lines, a slanted print, or a mid-height head stitch will pull a single full-height snap inward and clip the print. Instead:

1. **Edge stations.** Each of the four x-edges is re-measured in ~12 y-bands along the region, including the stitch rows implied by `geometry.num_heads`. In each band the **local solid bar** is found first, then the island inner/outer are walked from that bar by the nominal gap/width. A station more than ~200 px from the coarse guess is dropped (stray dual-band line or speck), not allowed to jump the box.
2. **Covering x.** Sparse outliers are discarded; among the rest the **outer** value is kept (min of left stations, max of right). That is what covers a parallelogram and a dogleg stitch: four corners alone would miss a mid-height slant.
3. **Covering y.** Stripe and vertical-line bands vote, plus snaps at the x-extremes of the station polylines (the true top-left / bottom-left of a rotated rectangle are not the same x). A singleton far outside the majority is treated as overspray; a singleton *within* `height/200` is treated as slant and kept. If the measured height is still short of the reference, it is extended from the stripe's reliable end, not from the remaining dashed lines.
4. **Constraint fill, both ways, only when an edge failed to measure.** LTR: stripe → gap → island inner → width → island outer. RTL: island outer → width → island inner → gap → stripe. Layout prediction is **not** applied to a well-sampled edge: real islands are ~5028 px against the nominal 5100, and forcing the nominal would expand the box. Prediction is used only when fewer than three stations returned ink (the missing-corner case).
5. **Covering AABB.** Downstream detectors still take a rectangle crop. `bounding_box_pixels` is the min/max of the four corners **and** the stitch stations, then `geometry.buffer` (50 px) is applied. Extra paper is acceptable; clipping print is not.

Four corners and the station polylines are written per colour so a dogleg is visible in JSON. `sources` / `warnings` flag `corner_missing`, `slant_px` / `stitch_offset_px`, and `predicted-cover`.

`--regions-only` skips detectors and writes, per scan:

| File | Contents |
|---|---|
| `{prefix}_full_regions.json` | boxes, four corners, edge stations, measurements vs reference, per-edge source, warnings |
| `{prefix}_full_regions.jpg` | whole-scan overlay: island green, stripe blue, verticals magenta |
| `{prefix}_full_regions_corners.jpg` | full-resolution crops of every region's four corners (TL, TR, BL, BR) |

`{prefix}` is the ink colour when the filename names one, otherwise the file stem.

`--regions` (or a sibling `<image>.json`) still overrides detection with operator-supplied boxes.

**Very large sheets.** A 2400 DPI four-colour scan is ~56000 × 40000 px (6.7 GB), past OpenCV's decode limit. Files above 1.5 GB are memory-mapped with `tifffile` instead, and every pass — the coarse profile, the station snaps, the overlay, the corner crops, and the per-region crops handed to the detectors — reads only the rows it needs. Folder walks skip a `synthetic/` directory so generated fixtures are not processed as scans.

**Accuracy (layout).** On a synthetic four-colour sheet every one of the 24 edges lands exactly, including a colour whose stripe never printed, a colour with a blank top-left corner, a colour with voids through its second band, and a near-invisible yellow. With a colour that printed nothing at all, its boxes come from the neighbours' pitch to within 1 px in x and are flagged. On the real 6.7 GB KCMY scan the four blocks measure island 5027–5065 px, stripe 1023–1036 px, gap 559–588 px, colour gap 258–272 px and height 32842–32926 px against nominals of 5100 / 1050 / 580 / 250 / 33000.

#### Validation (corner defects, slant, stitch)

The failure modes this snap is built for are missing/feeble ink at island corners, overspray just outside an edge, a slanted rectangle, and a 3-head stitch that puts a dogleg in the middle of an otherwise rectangular region. Ground truth is the **original print corners**: a paper-coloured mask or added spray must not change the box.

Catalog (no full-size mutated TIFFs — overlays are applied in memory on the memmapped original):

- Generator: `scripts/utility/generate_region_synthetics.py`
- Data: `data/20260617_P1_WhitePaper_KCM-updated-folder/synthetic/` (`ground_truth/`, `recipes/`, `geometric/`, `corner_crops/`, `manifest.json`)
- Overlay loader: `nike_detection/geometry/synthetic_overlays.py`
- Metrics / eval: `nike_detection/geometry/region_metrics.py`, `python -m nike_detection.tools.eval_regions`
- Fast tests: `tests/geometry/test_region_synthetics.py`

Pass rules: **inward clip > 15 px fails** (print would be cropped); outward expansion is capped (~80 px, buffer is already 50); print coverage ≥ 99.5% on missing-ink/overspray; IoU ≥ 0.95 on ±1° slant / stitch.

```bash
python scripts/utility/generate_region_synthetics.py
python -m pytest tests/geometry/test_region_synthetics.py -m "not slow"
python -m nike_detection.tools.eval_regions              # geometric canvases
python -m nike_detection.tools.eval_regions --baselines  # unmodified key/cyan/magenta TIFFs
python -m nike_detection.tools.eval_regions --full       # overlay recipes on the three TIFFs (slow)
```

Checked results: 10/10 fast geometric tests; 17/17 geometric eval cases (clean, missing top lines, missing TL, feeble outer vertical, overspray below, ±0.2/0.5/1.0° rotate, shear, stitch ±30/80/150 px); unmodified `key_full` / `cyan_full` / `magenta_full` vs frozen GT with no inward clip; overlay recipes on `key_full` (missing 20 top lines, 400 px TL mask, overspray blob 200 px below, feeble outer vertical, combined missing-TL + overspray-BR) all with 0 px inward clip.

### 6.2 Optional template extract (`--extract`)

Writes `ColorStripe.tiff` / `ColorIsland.tiff` to disk from `config.geometry` via `scripts/utility/new_pattern_tiff_extractor.py`. Invoked by `python -m nike_detection -i scan.tif --extract` or the `main_defect_detection.py` shim.

- Colors: Key, Cyan, Magenta, Yellow (`geometry.colors`)
- `color_width`, `x_offset`, `num_heads=3`, `head_height`, `y_offset`
- `island_front=true` → `[island][stripe]` inside each colour column

This is a **template**, not content-detected. A shifted or rotated scan will crop the wrong place. Prefer §6.1 on any filename containing `full`. `geometry.island_width` is still `0` in the 2400 config (the template uses buffers); the seed-free detector uses `region_reference.island_width` (5100) instead.

### 6.3 Legacy bbox extract (`scripts/utility/tiff_extractor.py`)

Hard-coded bounding boxes in `regions_json/template-2400-configs.json`. Used only with `--extract --pattern legacy --extract-config …`. Optional `exclusion_zones` are written as sibling JSON per crop.

**Limitation:** boxes are layout-specific; a shifted scan or a new pattern requires a new JSON.

---


## 7. Clear scan material (`--clear`)

Requires `--pattern new`. Adapts **island** detectors only.

| Stage | White paper | Clear (`--clear`) |
|---|---|---|
| Background | assumed ~255 | measured median gray |
| `VerticalBandDetector` ink ladder | 170 / 200 / 230 | `bg−25 / −18 / −12` |
| `IslandLineExtractor` | Otsu clamp `[120, 200]` | despeckle + clamp `[bg−85, bg−15]`; 2 px/col |
| Debris threshold | fixed 100–140 | `bg − 35/45/55`; paint with bg gray |
| Overspray threshold | `180 − background_threshold` | `bg − 35/45/60` + 5×5 median |
| Line-defect decisions | spacing-relative | **unchanged** (extraction is what changes) |
| Stripe detectors | — | **not adapted** |

**Limitations / updates needed**

- Stripe void, debris, misalignment, and overspray still assume white paper. Clear stripe crops will mis-binarize.
- `--clear` cannot be combined with `--pattern legacy`.
- Offsets (35/45/55) are empirical; they need a small labeled clear set to re-fit.
- Very dark clear scans (background near ink) will collapse the usable threshold range.

---

## 8. Robustness: legacy vs dual-band on the new pattern

| Concern | Legacy on two bands | Dual-band (`--pattern new`) |
|---|---|---|
| Cross-band line matching | Pairs across the gap → invalid slopes | Independent per-band detection |
| Vertical boundary lines | Flagged as debris/overspray | Morphology + envelope paint-out |
| Missing ink at line start/end | Walk starts at first ink | Trajectory to band bounds / full-width extractor |
| Slope/spacing drift | Fixed global slope window | Per-line robust fit; shear-search global slope |
| Fully missing lines | Ghost with guessed spacing | Inserted rows from measured spacing |
| Debris/overspray coverage | Band interior (and false verticals) | Full image after structure mask |
| Stippled print | Kernel occupancy FPs | Measured stipple close + column evidence |
| Clear / gray paper | Fixed 127 inverts | `--clear` background-adaptive island path |

---

## 9. Cross-cutting updates still needed

Work that is **not** done:

1. **4800 DPI** — no 4800 config; all thresholds are 2400-pixel.
2. **Stripe detectors on clear material** — void, debris stripe, and misalignment still assume white paper.
3. **Yellow (and light Magenta) in the island extractor / void** — the *region* detector already uses min-channel ink; line extraction and void still binarize grayscale.
4. **Exclusion zones on new-pattern islands and on stripes.**
5. **Non-monotonic sensitivity ladders** — stripe overspray medium kernel; legacy overspray min-area; legacy jagged threshold.
6. **Unify line finding** — use `IslandLineExtractor` for new-pattern debris/overspray on white paper too, not only `--clear`.
7. **DPI-scaled stripe misalignment thresholds** (pixels → mm).
8. **Physical nozzle reporting** — `missing_pixels` is columns, not calibrated nozzle IDs.
9. **Automated regression set** — golden *detector* crops for legacy, new-pattern white, and new-pattern clear. Region-*box* regression for corner defects / slant / stitch is in place (§6.1 Validation).
10. **Vertical-line defect inspection** — currently structural only; breaks in the four boundary lines are invisible.

Done since this list was first written: seed-free multi-colour region boxes (§6.1), bidirectional station snap + four-corner covering AABB with a synthetic catalog, shared `ImageContext` + parallel detectors (§10), memmap of multi-GB TIFFs, island stitch_error as its own class, surface treatment dropped from the default stripe set (still available via `--only surface_treatment`).

---

## 10. Shared processing, parallelization, and speed-ups

The runtime is `nike_detection/pipeline/runner.py`. `--workers` is a process pool across **regions of a full scan** (or across images in a folder). `--detector-threads` is a thread pool inside each region after shared geometry is built. Worker count is capped by CPU and an 8 GB RAM budget so eight island crops cannot oversubscribe the machine.

```
folder / full scan
    │
    ├─ collect files (skip previous result folders, synthetic/, and *_full_regions / *_visualization artifacts)
    ├─ if full sheet:
    │     open_scan (memmap if > 1.5 GB) → detect_full_regions
    │     process pool: one worker per colour region
    │
    └─ per region (process worker):
          memmap + crop
          ImageContext(bgr, gray)
          lazy layers: bands / extractor / paint-out  or  stripe edges / LAB
          thread pool: detectors
          write JSON + vis
```

**Shared layers today**

| Layer | Built by | Consumed by |
|---|---|---|
| `bgr` / `gray` | load once | all |
| `background_level` | `--clear` | new-pattern island |
| `bands` + `vlines` | `VerticalBandDetector` once | debris, overspray, line_defect |
| `extractor_per_band` | `IslandLineExtractor` once per band | line_defect (required); debris/overspray paint-out in `--clear` |
| `lines_removed_gray` | paint verticals + horizontals once | debris, overspray |
| `matched_lines` | `LineDetector` once | legacy island trio |
| stripe edges + stitch Ys | one interior-anchored walk | misalignment, edge_roughness |
| LAB + stripe bounds | once | void, debris_stripe |
| CLAHE | once | overspray, surface_treatment |

**Still worth doing**

- Reuse `IslandLineExtractor` trajectories for white-paper debris/overspray paint-out (today those still run `BandLineDetector` + `BandLineRefiner`).
- Fix the stripe overspray medium kernel if it is still far larger than high.
- `--no-vis` / `--downscale-vis` already exist; default writes full-res overlays.

---

## 11. How to invoke

Install once from the repo root:

```bash
pip install -r requirements.txt
```

All commands below are from the repo root. Thresholds, detector sets, geometry, and `region_reference` live in `config/detection_2400.json`. Override the file with `--config path/to.json`. Set a detector to `true` or `false` under `detector_sets.island` / `detector_sets.stripe` to include or skip it; detection algorithms themselves are unchanged.

### 11.1 One CLI, three inputs

| Input | How it is classified | What happens |
|---|---|---|
| File whose name contains `full` | `ImageType.FULL` | Measure boxes (§6.1), then run **enabled** island + stripe detectors on each crop |
| File whose name contains `island` | island | Island detector set |
| File whose name contains `stripe` | stripe | Stripe detector set |
| Folder | walk `*.tif`/`*.tiff`/`*.png`/`*.jpg` | Same classification per file; skips `output_*` folders and pipeline artifacts |
| `--extract` on a press scan | writes crops, then detects | Template extract (§6.2), **not** the seed-free boxes |

### 11.2 Flags

| Flag | Meaning |
|---|---|
| `-i` / `--input` | Image, `full` TIFF, or folder (**required**) |
| `-o` / `--output` | Optional override. Default: `{image_name}_MM_DD_YY_HH_MM_SS` **next to the input TIFF** |
| `--config` | Unified JSON (default `config/detection_2400.json`) |
| `-s` / `--sensitivity` | `low` \| `medium` \| `high` (default from config: `medium`) |
| `--pattern` | `legacy` \| `new` (default from config: `new`) |
| `--clear` | Gray paper / fainter ink. **Requires `--pattern new`.** Islands only. |
| `--only KEY …` | Subset of detector keys (see table below) |
| `--regions-only` | Measure boxes, write overlay + corners, skip detectors |
| `--regions FILE.json` | Operator boxes for a `full` TIFF (overrides automatic detection) |
| `--extract` | Template-extract crops from a press scan, then detect |
| `--extract-config` | Legacy bbox JSON used with `--extract --pattern legacy` |
| `--write-crops` | Also write the measured region TIFFs when processing `full` |
| `--region-folders` | Also write per-region subfolders with visualizations (off by default) |
| `--no-full-overlay` | Skip `{prefix}_full_defects.jpg` |
| `--no-vis` / `--downscale-vis` / `--debug` | Skip overlays, smaller overlays, extra debug images |
| `--workers N` | Parallel **processes** across regions of a full scan, or across images in a folder. Capped by CPU and an 8 GB RAM budget (default 2) |
| `--detector-threads N` | Threads per region after shared geometry. Reduced automatically when many region processes are running (default 4) |
| `--generate_report` | PDF summary |
| `--include-unknown` | Also process files that are not stripe/island/full |
| `--no-recursive` | Folder input: only the top level |
| `-v` / `--verbose` | DEBUG logs |

Detector keys for `--only`. Enable/disable the default set in `config/detection_2400.json` → `detector_sets` (`true` / `false`):

| Island | Stripe |
|---|---|
| **on:** `line_defect` | **on:** `stripe_misalignment` `edge_roughness` `void` |
| off: `debris_island` `overspray_island` | off: `debris_stripe` `overspray` `surface_treatment` |

### 11.3 Region boxes (no detectors)

Use this first on a new scan so you can inspect the overlay and corner montage before spending time on detectors.

```bash
# One colour per TIFF (filename should contain key / cyan / magenta / yellow)
python -m nike_detection -i data/20260617_P1_WhitePaper_KCM-updated-folder \
    --config config/detection_2400.json --regions-only

# Four colours on one sheet (KCMY). --workers 1 keeps the 6.7 GB memmap in one process.
python -m nike_detection -i data/rotated-KCMY-QualTest8Exp-BlkPt100-13.53.22.tif \
    --config config/detection_2400.json --regions-only --workers 1
```

Writes `{prefix}_full_regions.json` (boxes, four corners, edge stations), `{prefix}_full_regions.jpg`, `{prefix}_full_regions_corners.jpg` (TL/TR/BL/BR). `{prefix}` is the ink colour when the filename names one, otherwise the file stem.

Region-box regression (geometric canvases + frozen WhitePaper GT):

```bash
python -m pytest tests/geometry/test_region_synthetics.py -m "not slow"
python -m nike_detection.tools.eval_regions --baselines
```

Operator override (skip automatic detection):

```bash
python -m nike_detection -i Cyan_full.tiff --pattern new --regions my_boxes.json
```

### 11.4 Detect on a `full` scan

```bash
# Measure boxes, then run the default island + stripe sets
python -m nike_detection -i Cyan_full.tiff --pattern new

# Same, keep the region TIFFs on disk
python -m nike_detection -i Cyan_full.tiff --pattern new --write-crops

# Only missing nozzles / stitch / haze on the island crops
python -m nike_detection -i Cyan_full.tiff --pattern new --only line_defect

# Four-colour sheet: 8 regions in parallel, results next to the TIFF
python -m nike_detection -i data/rotated-KCMY-QualTest8Exp-BlkPt100-13.53.22.tif \
    --pattern new --workers 8
```

### 11.5 Already-extracted island or stripe crop

Filenames must contain `island` or `stripe`.

```bash
# All island detectors (new dual-band)
python -m nike_detection -i KeyIsland.tiff --pattern new

# Missing nozzles + stitch error only
python -m nike_detection -i KeyIsland.tiff --pattern new --only line_defect

# Clear-material island
python -m nike_detection -i ClearIsland.tiff --pattern new --clear

# All default stripe detectors
python -m nike_detection -i CyanStripe.tiff

# Stitch/roll only, high sensitivity
python -m nike_detection -i CyanStripe.tiff --only stripe_misalignment -s high

# Voids + misalignment
python -m nike_detection -i CyanStripe.tiff --only void stripe_misalignment

# Surface treatment (not in the default stripe set)
python -m nike_detection -i CyanStripe.tiff --only surface_treatment
```

### 11.6 Folder of crops

```bash
python -m nike_detection -i extracted_folder --pattern new --workers 2 --no-vis
python -m nike_detection -i extracted_folder --pattern new --only line_defect -s high --debug
```

### 11.7 Template extract then detect (legacy path)

Produces `ColorStripe.tiff` / `ColorIsland.tiff` from `geometry.*` seeds, then runs detectors on those files. Do **not** use this when you want the seed-free boxes of §6.1.

```bash
python -m nike_detection -i scan.tif --extract --pattern new
python -m nike_detection -i scan.tif --extract --pattern new --clear
python -m nike_detection -i scan.tif --extract --pattern new -s high --generate_report
```

### 11.8 Shims (same CLI underneath)

```bash
# Always adds --extract
python main_defect_detection.py -i scan.tif -d 2400 --pattern new --clear

# Already-extracted crop or folder (no --extract)
python scripts/defects_detection/run_all_detections.py -i KeyIsland.tiff --pattern new --only line_defect
```

Prefer `python -m nike_detection`. The copies under `scripts/defects_detection/*.py` are the old standalone detectors; they are not the runtime.

### 11.9 Output

Each run writes a folder **next to the input TIFF**:

`{image_name}_MM_DD_YY_HH_MM_SS/`

Example: `data/rotated-KCMY-QualTest8Exp-BlkPt100-13.53.22.tif` → `data/rotated-KCMY-QualTest8Exp-BlkPt100-13.53.22_08_18_26_13_16_00/`. Pass `-o` only if you want a different location.

| File | Contents |
|---|---|
| `{prefix}_full_regions.jpg` / `.json` / `_corners.jpg` | Segmented island/stripe boxes, four corners, edge stations (on `full` input) |
| `{prefix}_full_defects.jpg` | Same scan with every finding annotated (on unless `--no-full-overlay`) |
| `defect_summary.txt` | Readable per-region metrics (missing nozzles, stitch/calibration, roughness, voids) |
| `defect_summary.csv` | Same metrics, one row per metric |
| `defect_report.json` | Full per-region detections and timings |
| `<Region>/…` | Per-region visualizations (only if `--region-folders` or `write_region_folders: true`) |

---

## 12. File map

| Role | Path |
|---|---|
| Package CLI | `nike_detection/cli.py` (`python -m nike_detection`) |
| Unified 2400 config | `config/detection_2400.json` (`region_reference`, `geometry`, sensitivity, detector sets) |
| Extract-then-detect shim | `main_defect_detection.py` |
| Crop-folder shim | `scripts/defects_detection/run_all_detections.py` |
| Shared context / runner | `nike_detection/pipeline/context.py`, `runner.py`, `registry.py` |
| Detector adapters | `nike_detection/detectors/adapters.py` |
| Stripe algorithms | `nike_detection/detectors/stripe/` |
| Legacy island algorithms | `nike_detection/detectors/island_legacy/` |
| Dual-band island algorithms | `nike_detection/detectors/island_new/` (`line_defect.py` = missing nozzles + stitch) |
| Seed-free island/stripe boxes | `nike_detection/geometry/full_region_detector.py` |
| Region synthetic overlays / GT | `nike_detection/geometry/synthetic_overlays.py`, `region_metrics.py` |
| Region eval CLI | `python -m nike_detection.tools.eval_regions` |
| Region synthetic generator | `scripts/utility/generate_region_synthetics.py` |
| Region box tests | `tests/geometry/test_region_synthetics.py` |
| Vertical bands | `nike_detection/geometry/vertical_band_detector.py` |
| Line extract (new) | `nike_detection/geometry/island_line_extractor.py` |
| Legacy line scan | `nike_detection/geometry/line_detector.py` |
| Large-scan I/O | `nike_detection/io/image_loader.py` (`open_scan` / memmap) |
| JSON / vis / summary writers | `nike_detection/io/results.py`, `visualization.py`, `defect_summary.py` |
| Template extract | `scripts/utility/new_pattern_tiff_extractor.py` |
| Legacy bbox extract | `scripts/utility/tiff_extractor.py` |
