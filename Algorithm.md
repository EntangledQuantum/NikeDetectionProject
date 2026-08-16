# Defect Detection Algorithms

This document describes every detection algorithm: what it does, how it works, where it fails, what still needs updating, **what work is shared across detectors**, and **how to speed the pipeline up**. It covers the **legacy** single-band island layout and the **new** dual-band scan pattern on `july_visit`.

Routing is filename-based via `scripts/defects_detection/run_all_detections.py`. Sensitivity presets (`low` / `medium` / `high`) tighten or loosen thresholds.

**Contents:** [scan-pattern change](#1-scan-pattern-change-why-new-algorithms-exist) · [shared infrastructure](#2-shared-infrastructure) · [new-pattern islands](#3-new-pattern-dual-band-island-algorithms) · [legacy islands](#4-legacy-island-algorithms) · [stripes](#5-stripe-region-algorithms) · [extraction](#6-region-extraction-feeds-the-detectors) · [clear material](#7-clear-scan-material---clear) · [robustness](#8-robustness-legacy-vs-dual-band-on-the-new-pattern) · [updates needed](#9-cross-cutting-updates-still-needed) · [**shared work / parallelization / speed-ups**](#10-shared-processing-parallelization-and-speed-ups) · [invoke](#11-how-to-invoke) · [file map](#12-file-map)

| Image type (filename) | Detectors run |
|---|---|
| **Island** (`island` in name) | Debris, Overspray, Line Defect — dual-band variants when `--pattern new` |
| **Stripe** (`stripe` in name) | Stripe Misalignment, Edge Roughness, Overspray (scatter), Surface Treatment, Void, Debris Stripe |
| **Unknown** | Surface Treatment only |

`--pattern` and `--clear` affect **islands only**. Stripe detectors are the same for both layouts.

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

The dual-band detectors (`--pattern new`) add a band-splitting front end, mask the vertical lines, and replace the kernel walk with column-level ink evidence. Legacy detectors are left unchanged; `--pattern` defaults to `legacy`.

---

## 2. Shared infrastructure

### 2.1 `BaseDetector` (`detector_base.py`)

Common interface for detectors that support exclusion zones:

- Loads sibling JSON (`image.json`) with `exclusion_zones[].bounding_box_pixels`
- Point / region overlap checks
- Output as `(visualization_bgr, defects_list)`

**Limitation:** exclusion zones are only honored by the **legacy** island detectors (debris, overspray, line discovery). New-pattern island detectors accept `image_path` for interface compatibility but do not load or apply exclusion zones. Stripe detectors do not use them at all.

**Update needed:** wire exclusion zones into the dual-band detectors and at least into stripe misalignment / void (stamps, pen marks, crop artifacts).

### 2.2 `LineDetector` (`utils/line_detector.py`) — legacy island primitive

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

### 2.3 `material_profile.py` — clear-scan background

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
          IslandLineExtractor per band (shear + per-column ink stats)
          missing / misaligned / high-density regions
```

### 3.1 Vertical band detector (`utils/vertical_band_detector.py`)

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

### 3.2 Band line detector (`utils/band_line_detector.py`)

Thin `LineDetector` subclass: runs the legacy kernel scan on a **band crop**, but scales kernel width and number of scan columns from the **full image width** (`reference_width`) so per-band sensitivity matches the full-image detector. Height-based `Y_DELTA` and slope validation are inherited.

**Limitations:** inherits every `LineDetector` limitation (fixed 127 binarization, slope window, ghosts). On white paper this is acceptable for *painting lines out*; it is **not** used for new-pattern missing-nozzle decisions.

**Update needed:** in `--clear` mode this class is already skipped (debris/overspray use `IslandLineExtractor`). Consider using the extractor for white-paper debris/overspray as well, so there is one line-finding path.

### 3.3 Band line refiner (`utils/band_line_refiner.py`)

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

### 3.4 Island line extractor (`utils/island_line_extractor.py`)

Precursor for new-pattern **line defects** (and for clear-mode debris/overspray line masking). Avoids kernel scanning entirely.

**Pipeline**

1. **Ink binarization:** Otsu on a subsample, clamped to `[120, 200]`. Clear mode: 3×3 median despeckle, clamp to `[bg−85, bg−15]`. Clear mode also requires **2** ink pixels per column (one speck must not bridge a real gap).
2. **Global slope by shear search:** candidate slopes shear the binary image column-wise; the slope that maximizes sharpness (sum of squares) of the row ink profile wins. Coarse sweep (−0.005…0.035, step 0.002) then fine (step 0.0002). No calibration table.
3. **Shear** so every line becomes a narrow horizontal row band.
4. **Line rows:** runs of the sheared row profile above 25% of its P99; weighted centroids = line rows; median distance = **measured spacing**. Peak gaps ≈ 1.6× spacing get synthetic *inserted* rows so **fully missing lines are still evaluated**.
5. **Per-line residual fit:** ±0.4·spacing window, per-column ink centroids, sigma-clipped least squares — each line gets its own slope correction.
6. **Per-column statistics** inside ±0.3·spacing: ink presence/count, number of separate ink runs, hollow interior, centroid deviation from the fit. Mapped back to original coordinates via the stored shear shift.

**Limitations**

- Slope search range is bounded; a print with inverted or much steeper slant will pick a wrong global slope and scramble every line.
- Inserted (fully missing) lines assume locally regular spacing. A true double-gap or a compressed region can insert a row on the wrong Y.
- Corridor of 0.3·spacing can pick up a neighbor if spacing is mis-measured (e.g. every other line faint).
- Otsu + clamps can still fail on extremely low-contrast clear scans or on Yellow (very light ink on white).

**Updates needed**

- Widen or auto-expand the slope search if the sharpness peak sits on the range edge.
- Special-case **Yellow** (and possibly Magenta) with a chroma-aware binarization instead of grayscale Otsu.
- Expose measured `slope` / `spacing` in the results JSON (partially done via band summaries) and fail loudly when extraction returns `None` rather than silently skipping a band.

### 3.5 New-pattern debris (`new_pattern_debris_island_detection.py`)

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

### 3.6 New-pattern overspray (`new_pattern_overspray_island_detection.py`)

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

### 3.7 New-pattern line defect (`new_pattern_line_defect_detection.py`)

Missing-nozzle detection runs **only inside the two bands** — never on the verticals or the gap. Does **not** use the legacy kernel-ratio scan.

**Defect decisions (spacing-relative, resolution-independent)**

| Type | Color | Rule |
|---|---|---|
| `missing_line` | red | Stipple gap length is *measured* from healthy lines (P90 of intra-line ink-free runs). A 1-D close of ~3× that length bridges dotted texture; remaining ink-free runs longer than `min_gap` (0.35 / 0.60 / 1.20 × spacing for high / medium / low) are defects. `missing_pixels` = raw ink-free columns ≈ **missing nozzles**. |
| `misaligned_line` | yellow | (a) *split*: 2+ ink runs/column with hollow interior ≥ 3 px, or (b) *offset*: centroid deviation > max(3 px, 0.6 × thickness). Must persist ≥ ~0.5 × spacing. Overlapping segments merged. |
| `high_density_region` | orange | All missing pixels splatted into a 16×-downsampled accumulator, Gaussian-smoothed over ~2 spacings, thresholded at 40% of peak density. Blobs report defect count + missing-pixel sum. |

A `*_newpattern_detected_lines.jpg` debug image is always saved (verticals magenta, fitted trajectories green/blue, inserted fully-missing lines red).

**Limitations**

- Vertical lines and the central gap are never inspected (by design) — a defect that only lives there is invisible.
- `image_path` / exclusion zones unused; a stamp overlapping a band will be scored as missing ink.
- Inserted whole-line defects depend on spacing regularity.
- Density blobs are relative to the *peak* in that image, so a uniformly bad print may produce no hotspot (everything is equally dense).
- Split detection can fire on legitimate stipple that looks like two runs inside the corridor.
- Clear-mode decisions are unchanged (spacing-relative), but extraction quality still gates everything.

**Updates needed**

- Exclusion zones and optional ignore-margins.
- Absolute density floor so uniformly bad bands still report a hotspot.
- Per-color validation, especially Yellow and clear.
- Report nozzle estimates in physical units (need DPI + nozzle pitch calibration).
- Consider scoring the vertical boundary lines for breaks (currently structural only).

---

## 4. Legacy island algorithms

Still the default (`--pattern legacy`). Shared `LineDetector` locates slanted print lines.

```
Legacy island
    │
    ├─ LineDetector → matched left/right endpoints
    │
    ├─ Debris ── paint valid-slope lines white → dark threshold → contours
    ├─ Overspray ── paint lines white → colored mask → group blobs
    └─ Line Defect ── binary line map → walk each line → missing / jagged
```

### 4.1 Debris Island (`debris_island_detection.py`)

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

### 4.2 Overspray Island (`overspray_island_detection.py`)

Same line removal with thickness ×3. Colored mask at `color_threshold = 180 - background_threshold`. Aggressive dilate/close/erode, then greedy centroid grouping within `overspray_max_distance`.

| Sensitivity | `background_threshold` | Min area | Max group distance | Paint thickness |
|---|---|---|---|---|
| high | 5 | 100 | 300 | 5 |
| medium | 50 | 5000 | 1000 | 15 |
| low | 20 | 1000 | 800 | 5 |

**Limitations:** sensitivity table is not monotonic (medium min-area 5000 is *stricter* than low at 1000). Grouping distance of 1000 px can swallow half an island. Same “dark ≠ colored” confusion as the new-pattern variant.

**Updates needed:** rebuild the sensitivity ladder so high ⊃ medium ⊃ low; add hue/chroma; cap merge distance by island width.

### 4.3 Line Defect (`line_defect_detection.py`)

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

Stripe detectors are **layout-agnostic**: they derive geometry from the crop itself. The new extractor’s `--split-stripe-island` plus `num_heads` in `new_pattern_2400.json` (3 heads) is what changed for stripes, not the defect math. Misalignment was rewritten on this branch to work at 1 px resolution for both 3-head and 4-head patterns.

### 5.1 Stripe misalignment (`stripe_misalignment_detection.py`)

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

### 5.1b Stripe Edge Roughness (`stripe_edge_roughness_detection.py`)

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

### 5.2 Overspray — scatter grid (`overspray_detection.py`)

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

### 5.3 Void (`void_detection.py`)

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

### 5.4 Debris Stripe (`debris_stripe_detector.py`)

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

### 5.5 Surface treatment (`surface_treatment_detection.py`)

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

Detectors never see the full press scan. Extraction quality is part of the algorithm.

### 6.1 Legacy (`scripts/utility/tiff_extractor.py`)

Hard-coded bounding boxes in JSON (`template-2400-configs.json` / `template-4800-configs.json`). Optional `exclusion_zones` written as sibling JSON per crop.

**Limitation:** boxes are layout-specific; a shifted scan or a new pattern requires a new JSON.

### 6.2 New pattern (`scripts/utility/new_pattern_tiff_extractor.py`)

Parametric geometry from `regions_json/new_pattern_2400.json`:

- Colors: Key, Cyan, Magenta, Yellow
- `color_width`, `x_offset`, `num_heads=3`, `head_height`, `y_offset`
- `island_front=true` → `[island][stripe]` inside each color column
- `--split-stripe-island` writes `ColorStripe.tiff` and `ColorIsland.tiff`

`main_defect_detection.py --pattern new` always splits stripe/island so dual-band island detectors receive `*Island.tiff`.

**Limitations**

- Coordinates are still a **template**, not content-detected. A shifted or rotated scan will crop the wrong place.
- `island_width: 0` in the 2400 config: horizontal gap between color columns is not modeled; buffers (`horizontal` / `stripe_island` = 50) have to absorb it.
- **No `new_pattern_4800.json`**, though the orchestrator looks for it when `--dpi 4800 --pattern new`.
- Yellow/Key naming must contain `island` / `stripe` after the split or routing fails.

**Updates needed**

- Add a 4800 DPI new-pattern config (scale widths/heights ×2 from 2400, then verify on a real scan).
- Content-based column finding (chroma peaks) as a check against the template, with a warning on mismatch.
- Measure and set a real `island_width` instead of relying on buffers.

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

Priority-ordered list of work that is **not** done on `july_visit`:

1. **4800 DPI new-pattern config** (`regions_json/new_pattern_4800.json`) and a real 4800 validation scan.
2. **Stripe detectors on clear material** — void, debris stripe, and misalignment still assume white paper.
3. **Yellow (and light Magenta) binarization** — grayscale Otsu / LAB chroma are weak; chroma-aware ink masks are needed in the extractor, void, and band detector.
4. **Exclusion zones on new-pattern islands and on stripes.**
5. **Fix non-monotonic sensitivity ladders** — stripe overspray medium kernel 500; legacy overspray min-area; legacy jagged threshold.
6. **Unify line finding** — use `IslandLineExtractor` for new-pattern debris/overspray on white paper too, not only `--clear`.
7. **DPI-scaled stripe misalignment thresholds** (pixels → mm) so 2400 and 4800 share physical meaning.
8. **Surface treatment vs void** — overlapping stripe detectors; decide default set.
9. **Template vs content extraction** — warn when parametric crops do not match ink columns.
10. **Physical nozzle reporting** — `missing_pixels` is columns, not calibrated nozzle IDs.
11. **Automated regression set** — golden island/stripe crops for legacy, new-pattern white, and new-pattern clear, with expected defect counts.
12. **Vertical-line defect inspection** — currently structural only; breaks in the four boundary lines are invisible.
13. **Shared front-end + parallel detectors** — see [§10](#10-shared-processing-parallelization-and-speed-ups). Today every detector reloads the image and recomputes geometry that its siblings already have.

---

## 10. Shared processing, parallelization, and speed-ups

`run_all_detections.py` currently runs detectors **one after another** on one image, and each detector **reloads the TIFF and rebuilds its own geometry**. That is the main reason a Cyan stripe (~107 MB, 33k rows) takes many minutes, and why a folder of islands (~555 MB each) is dominated by repeated I/O rather than unique math.

This section is a design note only (no code changes landed with it).

### 10.1 What the pipeline does today (sequential, no sharing)

For every detector key, `_process_full_image` calls `cv2.imread` again, converts BGR→gray (or LAB/HSV) again, then the detector class repeats its own geometry:

| Image type | Detectors (default) | Reloads per image |
|---|---|---|
| Stripe | 6 (`stripe_misalignment`, `overspray`, `surface_treatment`, `void`, `debris_stripe`, `edge_roughness`) | 6× decode + 6× gray |
| Island (legacy or `--pattern new`) | 3 (`debris_island`, `overspray_island`, `line_defect`) | 3× decode + 3× gray |

`ImagePreprocessor.load_and_convert_to_grayscale` is already a shared helper, but it is **called per detector**, not once per image.

### 10.2 Shared preprocessing (do once per image)

These steps are identical (or differ only by a constant) across several classes.

**All detectors**

| Step | Who repeats it | Share as |
|---|---|---|
| Decode TIFF / PNG | Every detector via `cv2.imread` | `bgr` array |
| BGR → gray | Almost every detector | `gray` |
| File size / stem / output dir | Pipeline already | — |

**Stripe**

| Step | Used by | Notes |
|---|---|---|
| Column intensity profile + mid-level binarize | Misalignment, edge roughness | Same `col_mean`, `mid`, `gray < mid` mask |
| Interior-anchored left/right edge walk | Misalignment (then 31-row median), edge roughness (sub-pixel, no median) | **One raw integer walk**; roughness interpolates, misalignment median-filters |
| Both-edge stitch Ys | Misalignment reports stitches; roughness splits residual at the same joints | Compute stitch list once |
| LAB + chroma stripe bounds | Void, debris stripe | `_find_stripe_bounds` is duplicated in spirit |
| Inner pad of stripe | Void, debris stripe | Same “ignore ragged fringe” idea, slightly different pad fractions |
| CLAHE on gray | Overspray (scatter), surface treatment | Same contrast boost, different later math |

**Island — legacy**

| Step | Used by | Notes |
|---|---|---|
| `LineDetector.detect_lines` (kernel scan, ghosts, slope match) | Debris, overspray, line defect | **Run once**; debris/overspray paint `valid_slope` lines; line defect walks all matches |
| Exclusion-zone JSON | Same three (via `LineDetector`) | Load once |
| Binarize @ 127 | Line detector + line defect walk | One binary map |

**Island — `--pattern new`**

| Step | Used by | Notes |
|---|---|---|
| `estimate_background_level` (`--clear`) | Band detector, extractor, debris, overspray | One median of subsampled gray |
| `VerticalBandDetector.detect` → 4 verticals + 2 band crops | Debris, overspray, **and** line defect | **Heaviest shared geometry**; today run 3× |
| Per-band horizontal trajectories | Debris + overspray (`BandLineDetector` + `BandLineRefiner`); line defect (`IslandLineExtractor`) | Extractor trajectories can **replace** the coarse+refine path for painting (already done in `--clear`) |
| Paint vertical envelopes white | Debris, overspray | Same `paint_vertical_line_regions` |
| Paint horizontal lines | Debris (×2 thickness), overspray (×3) | One trajectory set; two paint thicknesses (or paint once at ×3 and use a distance map) |

### 10.3 Intermediate data worth caching (a per-image “context”)

A single `ImageContext` built before any detector would look like this.

```
ImageContext
  bgr, gray
  [clear] background_level
  ── stripe ──────────────────────────────────────────
  col_mean, mid, ink_mask
  bounds: x_left, x_right          # from col_mean or LAB chroma
  lab, hsv                         # void + debris stripe
  raw_left[], raw_right[]          # integer edge vs row
  subpixel_left[], subpixel_right[]
  stitch_ys[]                      # both-edge steps
  ── island / new pattern ────────────────────────────
  bands[{x0,x1,vline_xs,…}]
  vlines[]
  per_band: extractor_result       # slope, spacing, lines, ink_cols
            or refined LineDetector matches
  lines_removed_gray               # printed structure painted out
```

**Who consumes what**

| Artifact | Consumers |
|---|---|
| `bgr` / `gray` | All |
| Stripe `bounds` + `lab` | Void, debris stripe (and optionally overspray: mask *outside* the bar) |
| Raw / subpixel edges + `stitch_ys` | Misalignment, edge roughness |
| `bands` + `vlines` | All three new-pattern island detectors |
| Per-band line trajectories / `IslandLineExtractor` | Line defect (required); debris + overspray (paint-out) |
| `lines_removed_gray` | Debris + overspray (threshold on the same cleaned image; only the color/darkness cut differs) |
| Legacy `matched_lines` | All three legacy island detectors |

Detectors that **cannot** share a residual mask: stripe overspray (scatter grid on CLAHE+adaptive binary) and surface treatment (local stddev / coalescence) are different signals. They still share `bgr`/`gray` and should not reload the file.

### 10.4 What can run in parallel

OpenCV and NumPy release the GIL in the heavy loops, so **threads** help inside one process for C++/Fortran work; **processes** help for folder batches (avoid GIL + get more RAM isolation).

```
Folder of images                    one image
─────────────────                   ─────────────────────────────
pool.map(process_image)             load once → ImageContext
  CyanStripe  ──┐                     │
  KeyStripe   ──┼─ independent        ├─ shared geometry (serial, cheap vs I/O)
  MagentaStripe─┘                     │
  CyanIsland  ──┐                     ├─ then parallel:
  KeyIsland   ──┼─ independent        │     stripe:  [misalign ‖ roughness]  [void ‖ debris]
  MagentaIsland─┘                     │               [overspray]  [surface]   (weaker sharing)
                                      │     island:  [debris ‖ overspray] after paint-out
                                      │               [line_defect] after extractor
                                      └─ join → JSON + vis
```

**Safe to parallelize today (after a shared context)**

| Grain | Independent units | Caveat |
|---|---|---|
| Folder | Each `*Stripe*` / `*Island*` file | RAM: one 555 MB island decoded ~1.5 GB BGR; cap workers (2–4) |
| Stripe after context | Misalignment vs roughness vs void vs debris vs overspray vs surface | Misalignment + roughness need the same edges; void + debris need the same LAB bounds |
| New-pattern island | Left band vs right band inside `IslandLineExtractor` | Same slope search can be shared (global shear) |
| `IslandLineExtractor` | Per-line residual fit + per-column stats | After shear + row peaks |
| Stripe overspray grid | Each kernel cell | Only after fixing the 500 px medium kernel |
| Dual-band debris/overspray | Already one full-image pass; parallelizing them is easy once lines are painted |

**Do not bother parallelizing**

- Tiny vectorized NumPy on a 1-D edge profile (already microseconds).
- `VerticalBandDetector` 4-of-N combinations (N is small).
- Visualization `cv2.imwrite` of one JPEG at a time (I/O bound; can overlap with the *next* detector via a writer thread).

**Ordering constraint (must stay serial)**

1. Decode → gray → (optional) background level.
2. Geometry: stripe bounds/edges **or** island bands/lines.
3. Then independent defect scoring.

### 10.5 How to speed up — ranked

Highest impact first. None of this is implemented yet.

1. **Load each image once.** Passing `(bgr, gray)` into `detect()` removes 5 extra `cv2.imread` calls on a stripe and 2 on an island. For `KeyIsland.tiff` (~555 MB) that is the largest single win.
2. **Shared stripe geometry.** One edge walk + one LAB bounds object feeds misalignment, roughness, void, and debris. Roughness already duplicates misalignment’s walk.
3. **Shared new-pattern island front-end.** Run `VerticalBandDetector` **once**. Run `IslandLineExtractor` **once per band** and reuse trajectories for debris/overspray paint-out (drop the second `BandLineDetector`+`BandLineRefiner` pass on white paper).
4. **Parallelize the folder**, not the inner pixels. `ProcessPoolExecutor` over the 7 July_26 crops (3 stripe + 4 island) with `--pattern new`. Limit concurrency so islands do not RAM-thrash.
5. **Fix stripe overspray medium kernel (500×500).** Pairwise scatter on a 500 px window is the pathological case (~10 min on CyanStripe). High uses 20 px; medium should sit near that, not 25× larger. This is also a correctness bug (non-monotonic sensitivity).
6. **Make surface treatment optional or coarse.** It overlaps void conceptually, runs extra CLAHE + local stddev on the full height, and is slow. Default stripe set could drop it unless `--only surface_treatment`.
7. **Skip or downsample visualizations.** Writing 33k-row JPEGs/TIFFs is costly; `--no-vis` or downscaled overlays for the PDF, full-res only on request.
8. **Memory-map TIFFs** (`tifffile.memmap`) so folder workers do not each hold a full decode if they only need a stripe crop — extraction already cropped, so this is secondary.
9. **Two-band extractor:** shear-search slope on one band (or a height subsample), apply to the other if slopes match.
10. **CLAHE / adaptive threshold tiles** for overspray and surface treatment: they do not need the full 33k height in one shot; windowed processing already exists in README for >50 MB but is **not** used by these detectors.

### 10.6 Suggested target architecture (not built)

```
process_image(path):
  bgr, gray = load_once(path)
  ctx = build_context(bgr, gray, image_type, pattern, clear)
      # stripe: bounds, lab, edges, stitch_ys
      # island new: bands, vlines, extractor per band, lines_removed
      # island legacy: LineDetector matches, lines_removed
  results = parallel_map(detectors, ctx)   # threads inside one image
  save_json_and_vis(results)
```

Folder batch: `parallel_map(process_image, files)` with a worker cap.

Expected effect if (1)+(2)+(3)+(4)+(5) land: stripe wall-clock dominated by unique scoring instead of six decodes and a 500 px scatter grid; island wall-clock dominated by one band detect + one extractor instead of three copies of each.

---

## 11. How to invoke

```bash
# Full scan, new dual-band pattern
python main_defect_detection.py --image scan.tif --dpi 2400 --pattern new

# Clear / gray paper (islands only)
python main_defect_detection.py --image scan.tif --dpi 2400 --pattern new --clear

# Already-extracted island crop
python scripts/defects_detection/run_all_detections.py -i KeyIsland.tiff --pattern new --only line_defect

# Stripe crop (pattern flag ignored)
python scripts/defects_detection/run_all_detections.py -i CyanStripe.tiff --only void stripe_misalignment
```

Operator CLI details: `USER_STORY_README.md`. Pipeline overview: `README.md`.

---

## 12. File map

| Role | Path |
|---|---|
| Orchestrator | `main_defect_detection.py` |
| Detector runner | `scripts/defects_detection/run_all_detections.py` |
| New-pattern extract | `scripts/utility/new_pattern_tiff_extractor.py` |
| Legacy extract | `scripts/utility/tiff_extractor.py` |
| Dual-band debris / overspray / line defect | `scripts/defects_detection/new_pattern_*.py` |
| Vertical bands | `scripts/defects_detection/utils/vertical_band_detector.py` |
| Line extract (new) | `scripts/defects_detection/utils/island_line_extractor.py` |
| Band refine | `scripts/defects_detection/utils/band_line_refiner.py` |
| Clear background | `scripts/defects_detection/utils/material_profile.py` |
| Legacy line scan | `scripts/defects_detection/utils/line_detector.py` |
| Stripe void / debris / stitch / roughness | `void_detection.py`, `debris_stripe_detector.py`, `stripe_misalignment_detection.py`, `stripe_edge_roughness_detection.py` |
| 2400 new-pattern geometry | `regions_json/new_pattern_2400.json` |
