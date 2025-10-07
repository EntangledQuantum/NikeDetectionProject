# Stripe Debris/Void Detection - Testing Guide

## What I Changed

### ✅ **Enabled Debug Mode by Default**
- Debug is now ON by default in `run_all_detections.py`
- Console output shows detailed information about each step
- Debug images are automatically saved for inspection

### ✅ **Improved Medium Sensitivity Parameters for Debris Detection**

**OLD Parameters (Not Detecting Debris):**
```python
dark_factor = 0.65        # Too restrictive
bright_factor = 1.35      # Too restrictive
min_blob_area = 80        # Too large
morph_kernel_size = 6     # Too aggressive, removing details
morph_iterations = 2      # Too aggressive
```

**NEW Parameters (Optimized for Debris):**
```python
dark_factor = 0.82        # More sensitive - detects darker anomalies
bright_factor = 1.22      # More sensitive - detects brighter anomalies
min_blob_area = 40        # Smaller minimum (catches smaller debris)
morph_kernel_size = 3     # Smaller kernel (preserves details)
morph_iterations = 1      # Less aggressive cleaning
```

### ✅ **Enhanced Debug Output**

**Console Output Now Shows:**
1. Detection parameters being used
2. Baseline statistics (mean, std, min, max)
3. Threshold values calculated
4. Pixel counts at each step
5. Contour counts before/after filtering
6. Detailed filtering statistics
7. Final defect counts and area ranges

**Debug Images Saved (7 total):**
1. `01_grayscale.jpg` - Original grayscale image
2. `02_dark_mask_raw.jpg` - Raw dark anomaly mask (before cleaning)
3. `03_bright_mask_raw.jpg` - Raw bright anomaly mask (before cleaning)
4. `04_dark_mask_clean.jpg` - Cleaned dark mask (after morphology)
5. `05_bright_mask_clean.jpg` - Cleaned bright mask (after morphology)
6. `06_final_visualization.jpg` - Final result with bounding boxes
7. `07_masks_overlay.jpg` - Overlay showing red (debris) and cyan (voids) on grayscale

---

## How to Test

### **Option 1: Test Single Stripe Image (Recommended for Debugging)**

```bash
python test_debris_detection.py path/to/stripe.tiff medium
```

**Example:**
```bash
python test_debris_detection.py Images/Newer_High_DPI/test_Paper2400_extracted_regions_20251007_002118/blackStripe.tiff medium
```

**What You'll Get:**
- Main visualization: `blackStripe_debris_void_test.jpg`
- Debug folder: `blackStripe_debris_void_debug/` with 7 debug images
- Detailed console output showing all steps

### **Option 2: Run Full Pipeline**

```bash
python main_defect_detection.py --image Images/Newer_High_DPI/test_Paper2400.tif --dpi 2400 --sensitivity medium
```

This will:
1. Extract all regions (blackStripe, blueStripe, etc.)
2. Run ALL detectors including debris/void detection
3. Save results in timestamped output folder
4. Each stripe will have debug images saved

---

## What to Look For in Debug Images

### **1. Grayscale Image (`01_grayscale.jpg`)**
- Should show the stripe clearly
- Check if there are visible dark spots (debris) or bright areas (voids)

### **2. Dark Mask Raw (`02_dark_mask_raw.jpg`)**
- **White pixels = potential debris** (darker than threshold)
- If completely black → threshold too low, no debris detected
- If mostly white → threshold too high, everything is debris
- **Expected:** Some white spots where debris exists

### **3. Dark Mask Clean (`04_dark_mask_clean.jpg`)**
- Cleaned version after morphology
- Should still have white spots if debris exists
- If this goes from white to black → morphology is too aggressive

### **4. Masks Overlay (`07_masks_overlay.jpg`)**
- Red regions = debris candidates
- Cyan regions = void candidates
- This is the best image to see what's being detected before filtering

### **5. Final Visualization (`06_final_visualization.jpg`)**
- Red bounding boxes = debris defects
- Cyan bounding boxes = void defects
- Only shows defects that passed all filters (size, exclusion zones)

---

## Troubleshooting

### ❌ **Problem: No debris detected**

**Check the console output:**

```
Baseline - Mean: 195.50, Std: 12.30
  Dark threshold will be: 160.31  <-- This is the cutoff
  Min pixel value in image: 45
  Max pixel value in image: 255
```

**Analysis:**
- If debris pixels are > 160, they won't be detected
- If your darkest debris is at value 180, increase `dark_factor` to 0.90

**Check dark mask raw:**
- Open `02_dark_mask_raw.jpg`
- Are there any white pixels? 
  - **No white pixels** → Threshold too low, increase `dark_factor`
  - **Lots of white** → Threshold OK, check next step

**Check dark mask clean:**
- Open `04_dark_mask_clean.jpg`
- Did the white pixels survive?
  - **Disappeared** → Morphology too aggressive, reduce `morph_kernel_size` or `morph_iterations`
  - **Still there** → Good, check contour extraction output

**Check console output:**
```
Found 45 debris contours before filtering
  - Filtered by size (<40px): 42
  - Filtered by exclusion zones: 0
  - Final debris count: 3
```

**Analysis:**
- If many filtered by size → Reduce `min_blob_area`
- If many filtered by exclusion zones → Check exclusion zone JSON file

### 🔧 **Quick Parameter Adjustments**

Edit `scripts/defects_detection/stripe_debris_void_detection.py`, line 69-75:

**To detect MORE debris (more sensitive):**
```python
self.dark_factor = 0.90        # Increase (was 0.82)
self.min_blob_area = 25        # Decrease (was 40)
self.morph_kernel_size = 2     # Decrease (was 3)
```

**To detect LESS debris (reduce false positives):**
```python
self.dark_factor = 0.75        # Decrease (was 0.82)
self.min_blob_area = 60        # Increase (was 40)
self.morph_kernel_size = 4     # Increase (was 3)
```

---

## Understanding the Algorithm

### **Step 1: Measure Baseline**
- Computes mean intensity of the entire stripe
- Example: Mean = 200 (light gray stripe)

### **Step 2: Threshold**
- Dark threshold = Mean × dark_factor = 200 × 0.82 = 164
- Any pixel < 164 is marked as potential debris
- Bright threshold = Mean × bright_factor = 200 × 1.22 = 244
- Any pixel > 244 is marked as potential void

### **Step 3: Clean Masks**
- Morphological opening: Removes tiny noise speckles
- Morphological closing: Fills small holes in blobs
- Smaller kernel = preserve more detail but keep noise
- Larger kernel = remove noise but lose small defects

### **Step 4: Extract Blobs**
- Find connected white regions (contours)
- Measure area of each blob
- Filter out blobs smaller than `min_blob_area`
- Filter out blobs in exclusion zones

### **Step 5: Report**
- Each blob becomes a defect record
- Includes: type, centroid, area, bbox, severity

---

## Expected Console Output (Debug Mode)

```
=== Stripe Debris/Void Detection (Sensitivity: medium) ===
Image shape: (44127, 1230)
Parameters:
  - dark_factor: 0.82
  - bright_factor: 1.22
  - min_blob_area: 40
  - morph_kernel_size: 3
  - morph_iterations: 1

Baseline - Mean: 198.45, Std: 15.23
  Dark threshold will be: 162.73
  Bright threshold will be: 242.11
  Min pixel value in image: 12
  Max pixel value in image: 255

Thresholding complete:
  Dark threshold: 162.73 -> 125847 pixels
  Bright threshold: 242.11 -> 3456 pixels

Cleaning dark mask...
  Morphology: 125847 pixels -> 98234 pixels (removed 27613)
Cleaning bright mask...
  Morphology: 3456 pixels -> 2891 pixels (removed 565)

Extracting debris blobs...
  Found 156 debris contours before filtering
  Debris extraction complete:
    - Filtered by size (<40px): 143
    - Filtered by exclusion zones: 2
    - Final debris count: 11
    - Area range: 45.0 to 1250.5 pixels

Extracting void blobs...
  Found 8 void contours before filtering
  Void extraction complete:
    - Filtered by size (<40px): 5
    - Filtered by exclusion zones: 0
    - Final void count: 3
    - Area range: 52.0 to 340.0 pixels

Total defects found: 14 (Debris: 11, Voids: 3)

Saving 7 debug images...
    ✓ 01_grayscale: path/to/debug/blackStripe_debris_void_01_grayscale.jpg
    ✓ 02_dark_mask_raw: path/to/debug/blackStripe_debris_void_02_dark_mask_raw.jpg
    ...
```

---

## Quick Command Reference

```bash
# Test single stripe (fastest)
python test_debris_detection.py path/to/stripe.tiff medium

# Test with high sensitivity
python test_debris_detection.py path/to/stripe.tiff high

# Run full pipeline
python main_defect_detection.py -i scan.tif -d 2400 -s medium

# Run on extracted regions only
python scripts/defects_detection/run_all_detections.py --input_folder path/to/extracted_regions --sensitivity medium
```

---

## Files Modified

1. ✅ `scripts/defects_detection/stripe_debris_void_detection.py` - Updated parameters and debug output
2. ✅ `scripts/defects_detection/run_all_detections.py` - Enabled debug by default
3. ✅ `test_debris_detection.py` - NEW test script for quick testing
4. ✅ `DEBRIS_DETECTION_TESTING.md` - THIS documentation file

---

**🎯 Start Here:** Run the test script on a single stripe image and examine the debug output and images!

