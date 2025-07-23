# Stripe Misalignment Detection Algorithm

## Overview

The Stripe Misalignment Detection algorithm is designed to detect misalignments in vertical stripes that occur when a printer head takes multiple turns and becomes misaligned from its previous position. This is specifically designed for "stripe" type images which consist of one thick vertical line.

## How It Works

### 1. **Edge Detection Preprocessing**
   - The algorithm first applies edge detection to enhance vertical edges
   - Uses parameters from `edge_detector.py`: Gaussian blur (15x15), median filter (51), Sobel filter (5), and threshold (30)
   - This preprocessing helps isolate the vertical stripe from background noise

### 2. **Grid-Based Kernel Scanning**
   - Unlike line defect detection which searches for multiple lines, this uses a grid approach
   - Scans row by row, moving down by kernel height
   - In each row, scans horizontally to find the vertical line
   - No vertical search is performed - either finds a line or doesn't

### 3. **Misalignment Detection**
   - Once a line is detected for the first time, records its X position
   - For subsequent rows, calculates the X position delta from the previous position
   - If the delta exceeds the defect threshold, marks it as a misalignment

## Parameters

### Sensitivity Presets

| Parameter | Low | Medium | High |
|-----------|-----|--------|------|
| kernel_size | 70 | 50 | 30 |
| step_size | 70 | 50 | 30 |
| line_detection_threshold | 0.20 | 0.15 | 0.10 |
| defect_threshold | 20 | 10 | 5 |

### Custom Parameters

- **kernel_size**: Size of the scanning kernel (square)
- **step_size**: Horizontal step size (defaults to kernel_size for no overlap)
- **line_detection_threshold**: Minimum ratio of pixels to classify as line (0.0-1.0)
- **defect_threshold**: Minimum X position delta to consider as defect (in pixels)
- **debug**: Enable debug visualization

## Output

### Debug Mode (debug=True)
- Saves edge-detected image
- Shows kernel visualization with:
  - Green boxes: Normal line detection
  - Red boxes: Misalignment detected
- Shows kernel positions and detection status

### Normal Mode (debug=False)
- Highlights defective regions in red
- Shows X position delta values
- Clean visualization focusing only on defects

## Integration with Pipeline

The detector is automatically integrated into `run_all_detections.py` and will run on all images classified as "stripe" type.

## Standalone Usage

```python
from stripe_misalignment_detection import StripeMisalignmentDetector
import cv2

# Create detector
detector = StripeMisalignmentDetector(
    sensitivity='medium',  # or 'low', 'high'
    debug=True
)

# Load image and detect
image = cv2.imread('stripe_image.tiff')
visualization, defects = detector.detect(image)

# Save results
cv2.imwrite('result.jpg', visualization)

# Process defects
for defect in defects:
    print(f"Misalignment at Y={defect['y']}, X delta={defect['x_delta']}px")
```

## Example Results

The algorithm will detect:
- Sudden shifts in the vertical line position
- Gradual drift that exceeds the threshold
- Multiple misalignments in a single image

Each defect includes:
- Y position where misalignment occurred
- X position delta from previous position
- Previous X position for reference
- Threshold used for detection 