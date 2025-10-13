# Standalone Smudge Detection Tool - Usage Guide

## Overview
The `demo_smudge_detection.py` script is a standalone tool for detecting smudges in TIFF images. It runs the new reliable smudge detection algorithm in all three sensitivity modes and produces comprehensive output with defect markings.

## Features
- ✅ **TIFF File Input**: Specifically designed for TIFF images
- ✅ **Triple Analysis**: Runs LOW, MEDIUM, and HIGH sensitivity detection
- ✅ **Visual Output**: Creates marked images showing detected smudges
- ✅ **Detailed Reports**: Generates comprehensive text reports
- ✅ **Size Validation**: Enforces 400×400+ pixel minimum for significant smudges
- ✅ **Command Line Interface**: Easy to use from terminal/command prompt

## Installation Requirements
```bash
# Ensure you have the required Python packages:
pip install opencv-python numpy scikit-image scipy
```

## Basic Usage

### Simple Usage
```bash
python demo_smudge_detection.py input_image.tiff
```

### Custom Output Directory
```bash
python demo_smudge_detection.py input_image.tiff -o my_output_folder
```

### Verbose Mode
```bash
python demo_smudge_detection.py input_image.tiff --verbose
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `input_tiff` | Path to input TIFF file | *Required* |
| `-o, --output-dir` | Output directory for results | `smudge_detection_output` |
| `-v, --verbose` | Enable verbose error reporting | *Off* |
| `-h, --help` | Show help message | |

## Output Files

The tool creates several output files for comprehensive analysis:

### 1. Detection Images
- **`{basename}_smudge_detection_low.jpg`** - Conservative detection results
- **`{basename}_smudge_detection_medium.jpg`** - Balanced detection results  
- **`{basename}_smudge_detection_high.jpg`** - Sensitive detection results

### 2. Reference Image
- **`{basename}_original.jpg`** - Original image for comparison

### 3. Summary Report
- **`{basename}_smudge_detection_report.txt`** - Detailed text report

## Detection Sensitivity Levels

### LOW Sensitivity (Conservative)
- **Minimum Size**: 450×450 pixels (202,500 pixels)
- **Purpose**: Only detect very obvious, large smudges
- **Use Case**: Final quality control, critical applications

### MEDIUM Sensitivity (Balanced) ⭐ **Recommended**
- **Minimum Size**: 400×400 pixels (160,000 pixels) 
- **Purpose**: Balanced detection for standard quality control
- **Use Case**: Regular production monitoring

### HIGH Sensitivity (Detailed)
- **Minimum Size**: 316×316 pixels (100,000 pixels)
- **Purpose**: Detect smaller potential defects
- **Use Case**: Detailed inspection, research

## Understanding the Output

### Visual Markings
Detected smudges are marked with color-coded overlays:
- 🔴 **Red**: High severity smudges (large or very light)
- 🟠 **Orange**: Medium severity smudges
- 🟡 **Yellow**: Low severity smudges

### Report Information
Each detected smudge includes:
- **Type**: `directional_smear`, `contact_mark`, or `general_smudge`
- **Location**: X,Y coordinates of the center
- **Area**: Total area in pixels
- **Equivalent Size**: Approximate width×height dimensions
- **Severity**: High, Medium, or Low
- **Lightness Ratio**: How much lighter than surrounding background

## Example Workflow

### 1. Run Detection
```bash
python demo_smudge_detection.py sample_print.tiff -o analysis_results
```

### 2. Check Output
```
analysis_results/
├── sample_print_original.jpg              # Reference image
├── sample_print_smudge_detection_low.jpg    # Conservative results
├── sample_print_smudge_detection_medium.jpg # Balanced results
├── sample_print_smudge_detection_high.jpg   # Sensitive results
└── sample_print_smudge_detection_report.txt # Detailed report
```

### 3. Review Results
- **Start with MEDIUM sensitivity** results as baseline
- **Check LOW sensitivity** for critical defects only
- **Review HIGH sensitivity** for comprehensive inspection
- **Read the report** for detailed measurements

## Interpretation Guidelines

### No Defects Found
```
LOW:    0 smudge(s) detected
MEDIUM: 0 smudge(s) detected  
HIGH:   0 smudge(s) detected
```
**→ Image is clean, no significant smudge defects**

### Minor Defects Only
```
LOW:    0 smudge(s) detected
MEDIUM: 0 smudge(s) detected
HIGH:   2 smudge(s) detected
```
**→ Minor defects present, may be acceptable depending on quality standards**

### Quality Issues
```
LOW:    1 smudge(s) detected
MEDIUM: 3 smudge(s) detected
HIGH:   5 smudge(s) detected
```
**→ Significant smudge defects found, quality control action recommended**

## Technical Specifications

### Supported Formats
- ✅ `.tiff` files (primary)
- ✅ `.tif` files
- ⚠️ Other formats may work but are not optimized

### Algorithm Basis
- **Principle**: Detects lighter areas on consistent backgrounds
- **Background Analysis**: 100×100 pixel sliding window analysis
- **Consistency Filtering**: Eliminates printing artifacts
- **Size Enforcement**: Minimum 400×400 pixels for MEDIUM sensitivity

### Performance
- **Processing Time**: ~1-3 minutes for typical high-resolution images
- **Memory Usage**: Proportional to image size
- **Quality**: Optimized for 2400+ DPI printing

## Troubleshooting

### Common Issues

#### "Input file not found"
- Check file path is correct
- Ensure file has `.tiff` or `.tif` extension
- Verify file permissions

#### "Could not load TIFF file"
- File may be corrupted
- Try opening in image viewer first
- Check if file is actually a TIFF format

#### No smudges detected when expected
- Try HIGH sensitivity mode
- Check if defects meet minimum size requirements
- Verify image has consistent background areas

#### Too many false positives
- Use LOW sensitivity mode
- Check if image has printing artifacts or noise
- Ensure background is relatively consistent

### Getting Help
1. Use `--verbose` flag for detailed error information
2. Check the generated report for algorithm details
3. Review the NEW_SMUDGE_ALGORITHM_SUMMARY.md for technical details

## Integration with Existing Workflow

This standalone tool can be integrated into existing quality control processes:

1. **Batch Processing**: Use shell scripts to process multiple files
2. **Automated QC**: Integrate into production monitoring systems  
3. **Manual Inspection**: Use for detailed analysis of problem areas
4. **Validation**: Cross-reference with other defect detection methods

## Example Commands

```bash
# Basic analysis
python demo_smudge_detection.py print_sample.tiff

# Custom output location
python demo_smudge_detection.py print_sample.tiff -o QC_Analysis_20241201

# Verbose debugging
python demo_smudge_detection.py problematic_print.tiff --verbose

# Help information
python demo_smudge_detection.py --help
```

## Quality Control Recommendations

For **Nike printing quality control**, we recommend:

1. **Standard Analysis**: Use MEDIUM sensitivity as baseline
2. **Critical Products**: Use LOW sensitivity for final approval
3. **Research/Development**: Use HIGH sensitivity for detailed analysis
4. **Size Threshold**: 400×400+ pixels aligns with 2400 DPI requirements
5. **Workflow**: Process samples from each print run

The tool is specifically designed for detecting **post-print smudges** that appear as **lighter areas on consistent backgrounds**, making it ideal for Nike's elephant gray and cyan printing applications. 