# Wavelet Loss Image Scripts

## `wavelet_loss_image.py`

A script to compute Wavelet Loss between two images.

### Installation

```bash
# Install optional image dependencies
pip install .[image,visualize]
```

### Usage

```bash
# Basic usage
python wavelet_loss_image.py image1.jpg image2.jpg

# With custom wavelet and transform type
python wavelet_loss_image.py \
    --wavelet db4 \
    --level 3 \
    --transform-type dwt \
    --target-size 256 256 \
    image1.jpg image2.jpg

# Save loss visualization
python wavelet_loss_image.py \
    image1.jpg image2.jpg \
    --save-plot loss_plot.png
```

### Arguments

- `image1`, `image2`: Paths to input images (required)
- `--wavelet`: Wavelet family (default: db4)
- `--level`: Wavelet decomposition level (default: 3)
- `--transform-type`: Wavelet transform type (dwt, swt, qwt)
- `--grayscale`: Convert images to grayscale
- `--target-size`: Resize images (width height)
- `--device`: Compute device (cuda/cpu)
- `--save-plot`: Save loss visualization plot