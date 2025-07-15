# Wavelet Transform Visualization Script

## Overview

This script visualizes different wavelet transform decompositions for images, supporting:
- Discrete Wavelet Transform (DWT)
- Stationary Wavelet Transform (SWT)
- Quaternion Wavelet Transform (QWT)

## Installation

```bash
# Install optional image dependencies
pip install .[image,visualize]
```

## Usage

```bash
# Basic usage
python visualize_wavelet_transforms.py image.jpg

# Customize wavelet and levels
python visualize_wavelet_transforms.py \
    --wavelet sym4 \
    --level 4 \
    image.jpg

# Save visualization
python visualize_wavelet_transforms.py \
    image.jpg \
    --save-path wavelet_transforms.png
```

## Arguments

- `image`: Path to input image (required)
- `--wavelet`: Wavelet family (default: db4)
- `--level`: Decomposition levels (default: 3)
- `--grayscale`: Convert image to grayscale
- `--save-path`: Path to save visualization

## Visualization Details

The script generates a multi-panel visualization:
- First row: Original image
- Subsequent rows: Wavelet coefficients for each transform type
  - LL (Low-Low): Approximation coefficients
  - LH (Low-High): Horizontal detail coefficients
  - HL (High-Low): Vertical detail coefficients
  - HH (High-High): Diagonal detail coefficients

For Quaternion Wavelet Transform, only the real component is visualized.