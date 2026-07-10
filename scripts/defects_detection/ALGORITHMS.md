# Defect Detection Algorithms Summary

Overview of all detection methods under `scripts/defects_detection/`, grouped by **island** and **stripe** image regions. Routing is filename-based (`island` / `stripe` in the name) via `run_all_detections.py`.

| Image type | Detectors run |
|---|---|
| **Island** | Debris Island, Overspray Island, Line Defect |
| **Stripe** | Stripe Misalignment, Overspray (scatter), Void, Debris Stripe |
| **Unknown** | Surface Treatment only |

Sensitivity presets (`low` / `medium` / `high`) tighten or loosen thresholds across detectors.

---

## Shared infrastructure

### BaseDetector (`detector_base.py`)

Common interface for detectors that support exclusion zones:

- Loads sibling JSON (`image.json`) with `exclusion_zones[].bounding_box_pixels`
- Point / region overlap checks against those zones
- Standardizes output as `(visualization_bgr, defects_list)`

### LineDetector (`utils/line_detector.py`) — island shared primitive

Used by **Debris Island**, **Overspray Island**, and **Line Defect** to find the expected slanted horizontal print lines.

**Pipeline:**

1. **Scale kernels** to image size vs ideal reference (`5163 × 44228`); also scale `Y_DELTA` with height.
2. **Binarize** (dark lines → white).
3. **Scan left and right** with multiple vertical columns of kernels:
   - Require `min_detection_count` hits at a Y to accept a line
   - After each hit, jump to the next expected Y window (`Y_DELTA_MIN`–`Y_DELTA_MAX`)
   - If nothing is found in the window, insert a **ghost** line at the expected midpoint Y
   - Shift scan X around exclusion zones
4. **Match** left/right detections by index; compute slope  
   `slope = (y_right - y_left) / (x_right - x_left)`  
   Valid if `SLOPE_MIN ≤ slope ≤ SLOPE_MAX` (≈ `0.0150`–`0.0219`).

---

## Island region algorithms

Island images contain arrays of slanted horizontal print lines. All three island detectors share `LineDetector` to establish where those lines are. Debris and overspray then **remove** the lines so residual dark/colored pixels can be measured; line defect instead **follows** each line and looks for breaks or zig-zags.

```
Island image
    │
    ├─ LineDetector → matched left/right endpoints per line
    │
    ├─ Debris Island ── paint lines white → dark threshold → debris contours
    ├─ Overspray Island ── paint lines white → colored mask → group blobs
    └─ Line Defect ── binary line map → walk each line → missing / jagged
```

---

### 1. Debris Island (`debris_island_detection.py`)

**Goal:** Find dark debris spots on island prints without mistaking print lines for debris.

**Why line removal first:** Print lines are dark and would otherwise dominate a darkness threshold. Painting them white leaves only non-line dark material as candidates.

**Detailed pipeline:**

1. **Locate lines** via `LineDetector.detect_lines(image, debug, image_path)` (loads exclusion zones from sibling JSON when `image_path` is set).
2. **Remove lines:** for each match with `valid_slope=True`, draw a white line from left→right endpoint with thickness `line_thickness * 2`. Invalid-slope matches are left in place (not painted out).
3. **Debris mask:** `cv2.threshold(..., background_threshold, 255, THRESH_BINARY_INV)` — anything darker than the threshold becomes white (candidate debris). Background reference: typical paper/ink greys ≈ 133–180; the threshold sits below that so only darker spots survive.
4. **Morphology:** single elliptical close `(3, 3)` to fill tiny holes and connect nearby debris pixels.
5. **Contours:** `findContours(RETR_EXTERNAL)`; keep area > `debris_min_area`; reject any contour whose bbox overlaps an exclusion zone (zones come from the shared `LineDetector`).
6. **Visualization:** red filled contours + red bboxes + area labels; magenta exclusion overlays when present.

| Sensitivity | Background thresh | Min area (px²) | Line paint thickness |
|---|---|---|---|
| high | 140 (more aggressive) | 5 | 5 |
| medium | 120 | 10 | 20 |
| low | 100 (darker only) | 50 | 5 |

**Outputs:**

- `lines_detected` — counts, avg/std Y-delta, real vs ghost left/right, valid vs invalid slopes
- `debris_detected` — `debris_count` + per-region `bbox`, `area`, `center`

**Debug artifacts (when enabled):** kernel scan overlay, lines-removed grayscale, enhanced debris mask, left/right line-point dots.

---

### 2. Overspray Island (`overspray_island_detection.py`)

**Goal:** Find larger colored / non-white overspray regions after line removal. Unlike debris (small dark spots), overspray is treated as substantial colored area that may be fragmented and needs grouping.

**Detailed pipeline:**

1. **Locate + remove lines** (same as debris, but paint thickness `line_thickness * 3` for fuller coverage).
2. **Colored mask:** threshold at `color_threshold = 180 - background_threshold` with `THRESH_BINARY_INV`. Lower `background_threshold` → higher color threshold → less sensitive; higher sensitivity lowers the difference from white needed to count as colored. A secondary “strong colored” mask at fixed 120 is kept for debug only.
3. **Aggressive morphology** (connect spray fragments into blobs):
   - Dilate ellipse `(5, 5)`, 2 iterations
   - Close ellipse `(11, 11)`
   - Close again ellipse `(7, 7)` (hole fill)
   - Erode ellipse `(3, 3)`, 1 iteration (shrink back toward true size)
4. **Filter contours:** area ≥ `overspray_min_area`; drop exclusion-zone overlaps. Each kept region stores contour, area, centroid, bbox, and density `area / (w*h)`.
5. **Proximity grouping** (`group_nearby_regions`):
   - Greedy: for each unused region, collect others whose centroid distance ≤ `overspray_max_distance`
   - Multi-region groups → convex hull of all contour points; store `merged_count`, hull area, and `original_area` (sum of parts)
   - Singletons kept as-is
6. **Visualization:** red outlines/fills labeled `OVERSPRAY N` with area (and merge count if merged).

| Sensitivity | `background_threshold` (color sensitivity) | Min area (px²) | Max group distance (px) | Line paint thickness |
|---|---|---|---|---|
| high | 5 (very sensitive) | 100 | 300 | 5 |
| medium | 50 | 5000 | 1000 | 15 |
| low | 20 | 1000 | 800 | 5 |

**Outputs:** `lines_detected` + `overspray_detected` (`bbox`, `area`, `center`, `density`, `merged_count`, `original_area`, thresholds used).

**Debug artifacts:** lines-removed image, colored mask, region overlay, intermediate masks, line-point dots.

---

### 3. Line Defect (`line_defect_detection.py`)

**Goal:** Along each known print line, find **missing** segments (ink gaps) and **jagged** segments (sudden vertical jumps / zig-zags). This detector does **not** remove lines; it uses them as trajectories to inspect.

**High-level flow:**

```
detect()
  │
  ├─ LineDetector.detect_lines()  → matched_lines (left/right endpoints)
  ├─ Binarize grayscale @ 127 (THRESH_BINARY_INV)  → lines white
  ├─ For each matched line:
  │     scan_line_for_defects(binary, line_match)
  │       → missing_line / jagged_line defects + kernel_states
  └─ create_combined_visualization()  → red missing + yellow jagged
```

#### Stage A — Establish line trajectories

1. Call `LineDetector` with the same sensitivity (and optional `image_path` for exclusion zones).
2. Each `matched_line` provides:
   - `left` / `right`: `{x, y, type}` where `type` is `real` or `ghost`
   - `slope`, `y_delta`, `valid_slope`
3. If no matches: return early with “No lines detected.”
4. **Note:** Line Defect currently scans **all** matched lines (including invalid-slope ones). Debris/overspray only paint out `valid_slope` lines.

#### Stage B — Binary line map

- Convert to grayscale if needed.
- `cv2.threshold(gray, 127, 255, THRESH_BINARY_INV)` so dark print ink becomes white for kernel occupancy checks.

#### Stage C — Walk each line (`scan_line_for_defects`)

For one matched line from left endpoint `(x_start, y_start)` to right `(x_end, y_end)`:

1. **Trajectory model** — linear interpolation:
   - `slope = (y_end - y_start) / (x_end - x_start)`
   - At scan X: `expected_y = y_start + slope * (x - x_start)` (clamped so the kernel stays in-bounds)
2. **Horizontal scan** — start at `x = x_start`, advance by `step_size` until `x_end`.
3. **Kernel occupancy** — square kernel of size `kernel_size` centered at `(x, expected_y)`:
   - `pixel_ratio = white_pixels / kernel_area`
   - `has_line = pixel_ratio > line_threshold`
4. **If line present:**
   - Compute `actual_y` = mean Y of white pixels inside the kernel (true ink center, not just expected Y).
   - **Jagged check** (vs previous *detection*, not vs expected Y):
     - `y_delta = |actual_y - last_found_y|`
     - If `y_delta > jagged_threshold` → emit `jagged_line`
   - **Close an open gap** (if `gap_start` was set):
     - `gap_size = x - gap_start`
     - If `gap_size >= min_gap_size` → emit `missing_line` from `gap_start` to current `x`
     - Smaller gaps are ignored for missing (but the Y jump across them can still trigger jagged)
   - Update `last_found_y = actual_y`
5. **If line absent:**
   - Start a gap at current `x` if not already in one (`gap_start = x`)
6. **End-of-line:** if still in a gap, close against `x_end` with the same `min_gap_size` rule.

```
X along line →
  [ink][ink][  gap ≥ min_gap  ][ink][ink↑Y jump][ink]
   OK   OK      MISSING          OK    JAGGED     OK
```

**Design choices worth noting:**

| Choice | Behavior |
|---|---|
| Jagged vs previous detection | Small missing gaps do not reset the Y baseline; when ink reappears, ΔY is measured from the last real hit, so a zig-zag across a short break still counts |
| Expected Y only for placement | Kernel is centered on the linear path; `actual_y` is measured from ink inside the kernel |
| Ghost endpoints | A ghost left/right from `LineDetector` still defines a trajectory; the walk can find real ink (or gaps) along that path |
| No exclusion filtering in the walk | Exclusion zones affect `LineDetector` scan placement, but individual missing/jagged hits are not re-checked against zones |

#### Stage D — Visualization

| Mode | Content |
|---|---|
| Combined (always returned) | Red filled bars for missing spans; yellow bars for jagged points; summary `Missing: N \| Jagged: M` |
| Debug: missing only | Red missing segments |
| Debug: jagged only | Yellow jagged segments |
| Debug: scan kernels | Green = line OK, red = no line, yellow = jagged |
| Debug: LineDetector kernels | Left/right discovery kernels + matched lines (green valid / orange ghost / red invalid slope) |

#### Sensitivity parameters

| Sensitivity | `kernel_size` | `step_size` | `line_threshold` | `min_gap_size` | `jagged_threshold` |
|---|---|---|---|---|---|
| high | 15 | 10 | 0.25 (25% white) | 50 px | 10 px |
| medium | 30 | 25 | 0.20 | 100 px | 5 px |
| low | 50 | 40 | 0.15 | 200 px | 25 px |

Interpretation:

- **high** — finer steps, smaller kernels, shorter gaps reported, moderate jagged ΔY; stricter occupancy (25%)
- **medium** — default balance; jagged fires at only 5 px Y jump (most sensitive to zig-zag)
- **low** — coarser sampling; only large gaps (≥200 px) and large Y jumps (≥25 px)

#### Defect output schema

**`missing_line`**

| Field | Meaning |
|---|---|
| `start_x`, `end_x` | Horizontal span of the gap |
| `y` | Expected Y at close (or `y_end` for trailing gaps) |
| `location` | Midpoint `( (start_x+end_x)/2 , y )` |
| `size` | `end_x - start_x` in pixels |

**`jagged_line`**

| Field | Meaning |
|---|---|
| `x`, `y` | Current detection position (`y` = `actual_y`) |
| `previous_y` | Y of previous successful detection |
| `location` | `(x, actual_y)` |
| `y_delta` | Absolute jump in pixels |
| `threshold` | `jagged_threshold` used |

#### Relationship to other island detectors

| | Debris / Overspray | Line Defect |
|---|---|---|
| Uses `LineDetector` | Yes (to erase lines) | Yes (to define paths) |
| Operates on | Line-removed grayscale | Binary line map |
| Defect signal | Dark / colored residual area | Gaps and Y discontinuities along the path |
| Typical failure mode caught | Foreign particles, spray blobs | Broken or wavy print lines |

---

## New-Pattern (Dual-Band) Island region

The island pattern changed from a single print band flanked by two vertical
lines to **two** print bands separated by a white gap, each flanked by a pair
of vertical boundary lines (four vertical lines total):

```
[V  prints  V]   gap   [V  prints  V]
```

The three legacy island detectors assume one band spanning the full image
width and scan horizontal lines inward from the extreme left/right edges. On a
two-band image that pairs a left-band endpoint with a right-band endpoint
across the empty gap (invalid slopes, wrong trajectories), and the vertical
boundary lines would be flagged as debris/overspray.

The dual-band detectors **reuse the legacy per-image helpers unchanged** and
add a band-splitting front end. They are selected with `--pattern new` in
`run_all_detections.py` (default is `--pattern legacy`); the same detector keys
(`debris_island`, `overspray_island`, `line_defect`) are reused, so routing and
output naming are identical.

### Shared band front end

**`utils/vertical_band_detector.py` — VerticalBandDetector**

Auto-detects bands and vertical lines from a column ink-density profile (no
hard-coded coordinates):

1. Grayscale → `THRESH_BINARY_INV` (dark ink → white).
2. `col_frac[x] = dark_pixels_in_column / height`, smoothed with a small box
   filter. Print columns are sparse (dashed lines ≈ a few %), vertical lines
   are near-solid (≈ 1.0), the central gap is ≈ 0.
3. **Bands** = contiguous runs with `col_frac > content_threshold`, merging
   small gaps and dropping runs narrower than `min_band_fraction`.
4. **Vertical lines** = columns within a band run with `col_frac > vline_threshold`,
   grouped into line objects (center + thickness).
5. Per band, the **crop** (`x0`, `x1`) is trimmed just inside the outermost
   vertical lines so the boundary lines are excluded from all processing.
6. Fallbacks: fewer than two vertical lines → keep the run edges; no runs at all
   → degrade to a single full-width band (legacy behavior) with a warning.

Also provides `paint_vertical_lines_white(gray, vline_xs, thickness)` as defense
so any vertical line surviving inside a crop is painted out before thresholding.

**`utils/band_line_detector.py` — BandLineDetector(LineDetector)**

Thin subclass that runs the standard `LineDetector` on a band crop but scales
width-dependent parameters (kernel width, number of vertical scan columns)
using the **full image width** (`reference_width`) rather than the narrow crop
width, so per-band line detection matches full-image sensitivity. Height-based
behavior (`Y_DELTA`, slope validation) is inherited unchanged.

### Dual-band flow (all three detectors)

```
Island image
    │
    ├─ VerticalBandDetector → bands [(x0,x1, vline_xs), ...]
    │
    └─ For each band:
          crop = image[:, x0:x1]
          BandLineDetector.detect_lines(crop)      # horizontal lines, band-local
          reuse legacy helper on crop              # debris / overspray / line walk
          offset results by +x0 → full-image coords
    │
    └─ Composite one visualization + merged defect list
```

### 3a. New-Pattern Debris Island (`new_pattern_debris_island_detection.py`)

Per band: `BandLineDetector.detect_lines` → reuse `DebrisIslandDetector.remove_lines_from_image`
→ `paint_vertical_lines_white` → reuse `detect_debris` → offset contours by `+x0`.
Final visualization via the reused `create_debris_visualization`.

### 3b. New-Pattern Overspray Island (`new_pattern_overspray_island_detection.py`)

Per band: line removal + vertical-line removal → reuse `detect_colored_regions`
and `group_nearby_regions` (**grouping stays within a band**, so overspray is
never merged across the central gap) → offset contour, `center`, and `bbox` by
`+x0`. Final visualization via the reused `create_overspray_visualization`.

### 3c. New-Pattern Line Defect (`new_pattern_line_defect_detection.py`)

Per band: binarize crop, then for each matched line reuse
`LineDefectDetector.scan_line_for_defects`. Robustness improvements over the
legacy walk:

- **Invalid-slope matches are skipped** (legacy walks all matches).
- **Ghost endpoints are repaired** using the band's median slope so the walk
  follows a stable trajectory.

Missing/jagged `start_x`/`end_x`/`x`/`location` are offset by `+x0`; final
visualization via the reused `create_combined_visualization`.

### Robustness vs. legacy on the new pattern

| Concern | Legacy on two bands | Dual-band detectors |
|---|---|---|
| Cross-band line matching | Pairs across the gap → invalid slopes | Independent per-band detection |
| Vertical boundary lines | Flagged as debris/overspray | Excluded by cropping + painted out |
| Line-detection sensitivity | N/A | Kernels scaled to full image width |
| Overspray grouping | Could merge across gap | Grouped within each band |

---

## Stripe region algorithms

### 4. Stripe Misalignment (`stripe_misalignment_detection.py`)

**Goal:** Detect lateral jumps in a vertical stripe edge (printer-head misalignment).

**Pipeline:**

1. **Edge preprocess:** Gaussian blur → median blur → Sobel-X → normalize → threshold (vertical edges).
2. Scan row-by-row with a tall thin kernel; take the **first** strong vertical edge X in each row.
3. Compare consecutive row X positions; if `|Δx| > defect_threshold`, flag misalignment.
4. Visualize defects as red markers (or kernel overlays in debug).

| Sensitivity | Kernel (W×H) / step | Line thresh | Defect Δx |
|---|---|---|---|
| high | 30×30 / 30 | 0.10 | 5 |
| medium | 5×60 / 5 | 0.20 | 20 |
| low | 70×70 / 70 | 0.20 | 20 |

**Outputs:** `stripe_misalignment` (`y`, `x`, `x_delta`, `previous_x`).

---

### 5. Overspray — scatter grid (`overspray_detection.py`)

**Goal:** Find scattered spray (not solid ink) via spatial scatter metrics. Used on stripe images in the pipeline.

**Pipeline:**

1. Grayscale → CLAHE → adaptive threshold → invert so spray pixels are white.
2. Slide a square kernel over a grid.
3. Per kernel, compute **scatter metric** from white-pixel spread (std of coords) and pairwise distance variance; boost mid-density ratios (~5–30%); reject too-sparse or too-solid kernels.
4. Flag kernels with scatter > `scatter_threshold`.
5. (Non-debug) morphologically merge adjacent hits into regions.

| Sensitivity | Kernel / step | Scatter thresh | Min white ratio |
|---|---|---|---|
| high | 20 / 15 | 0.2 | 0.03 |
| medium | 500 / 500 | 0.3 | 0.05 |
| low | 50 / 50 | 0.5 | 0.10 |

**Outputs:** `overspray` or merged `overspray_region` (`scatter_metric`, `bbox`, `area`).

---

### 6. Void (`void_detection.py`)

**Goal:** Find compact “missing ink” voids inside a solid-color vertical stripe (color drifted toward paper/white).

**Pipeline:**

1. Convert to LAB; find stripe left/right from column chroma (fallback: dark columns for black stripes).
2. Shrink inward (`inner_pad`) to ignore ragged edges.
3. Median LAB of stripe interior = stripe reference; median LAB outside = paper reference.
4. Per-pixel **voidness** = projection of `(pixel − stripe_ref)` onto `(paper_ref − stripe_ref)`, normalized to `[0, 1]`.
5. Robust threshold inside stripe: `median + k·MAD·1.4826` (floor 0.12).
6. Morph close then open; kernel sizes scale with stripe width.
7. Keep components with area / extent / aspect consistent with voids; reject huge fade-outs.
8. Draw black bounding boxes.

| Sensitivity | MAD multiplier `k` | Min area floor / frac |
|---|---|---|
| high | 5.0 | 80 / 5e-5 |
| medium | 7.0 | 150 / 1e-4 |
| low | 10.0 | 300 / 3e-4 |

**Outputs:** `void` (`bbox`, `area`, `centroid`, `mean_voidness`, `threshold`).

---

### 7. Debris Stripe (`debris_stripe_detector.py`)

**Goal:** Find dark, near-black debris spots inside colored stripes (color-agnostic).

**Pipeline:**

1. Find stripe bounds from LAB chroma (dark fallback); pad inward.
2. Build a combined **debris score** inside the stripe:
   - `0.40` multi-scale black-hat local darkness (L and HSV-V)
   - `0.35` absolute darkness vs stripe luminance stats
   - `0.20` saturation drop vs stripe S stats
   - `0.05` chroma drop
3. Strong / weak masks from robust score thresholds (`median + k·σ`, with floors).
4. Promote weak connected components that contain ≥1 strong pixel.
5. Morph open + close (sizes scale with stripe width).
6. Filter by area and darkness/saturation checks; draw black boxes.

| Sensitivity | Score `k` | Strong / weak floors | Min area floor / frac |
|---|---|---|---|
| high | 4.5 | 0.34 / 0.22 | 4 / 2e-5 |
| medium | 5.5 | 0.40 / 0.26 | 8 / 5e-5 |
| low | 7.0 | 0.48 / 0.32 | 20 / 1.2e-4 |

**Outputs:** `debris_stripe` (`bbox`, `area`, `centroid`, score / darkness metrics).

---

## Other / fallback

### Surface Treatment (`surface_treatment_detection.py`)

Used for **unknown** image types (and listed in stripe strategy docs; currently not wired in the active stripe detector set).

**Pipeline:**

1. CLAHE contrast enhancement.
2. **High-contrast drops:** local stddev → morph close → keep irregular (high eccentricity / low solidity) large regions (ink coalescence).
3. **Void areas:** adaptive light regions ∩ dilated Otsu print coverage; intensity must exceed surroundings.
4. Optional FFT radial-profile texture anomalies (computed; not primary output).
5. Highlight affected regions in green.

---

## Algorithm comparison (quick reference)

| Detector | Region | Core idea | Key signal |
|---|---|---|---|
| Debris Island | Island | Remove lines → dark threshold | Grayscale darkness |
| Overspray Island | Island | Remove lines → colored blobs + proximity merge | Non-white area + grouping |
| Line Defect | Island | Walk known lines | Gaps / Y jumps along line |
| Stripe Misalignment | Stripe | Track vertical edge X down the image | Lateral Δx between rows |
| Overspray (scatter) | Stripe | Grid scatter of spray pixels | Spatial spread metric |
| Void | Stripe | LAB projection toward paper | Voidness score |
| Debris Stripe | Stripe | Multi-feature dark score in stripe | Darkness + chroma/sat drop |
| Surface Treatment | Unknown | Local contrast + missing ink | Stddev / adaptive voids |

---

## Pipeline notes

- Entry point: `run_all_detections.py` (`DefectDetectionPipeline`).
- Per-image results: visualizations + JSON; optional PDF summary.
- Island detectors that use `LineDetector` can honor per-image exclusion-zone JSON.
- Stripe void / debris detectors derive stripe geometry from the image itself (no line model).
