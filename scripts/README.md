# Scripts

This directory contains utility scripts for working with wavelet transforms and losses.

## Loss Computation Scripts

### `wavelet_loss_image.py`
Compute Wavelet Loss between two images directly in pixel space.

**Usage:**
```bash
uv run python scripts/wavelet_loss_image.py image1.png image2.png [options]
```

**Arguments:**
- `image1` - Path to first input image
- `image2` - Path to second input image

**Options:**
- `--wavelet WAVELET` - Wavelet family to use (default: db4)
- `--level LEVEL` - Wavelet decomposition level (default: 3)
- `--transform-type {dwt,swt,qwt}` - Wavelet transform type (default: dwt)
- `--grayscale` - Convert images to grayscale
- `--target-size WIDTH HEIGHT` - Resize images to specified size
- `--device DEVICE` - Compute device (default: cuda if available)
- `--save-plot SAVE_PLOT` - Path to save loss visualization plot

### `wavelet_loss_latent.py`
Compute Wavelet Loss between two images using VAE latent representations.

**Usage:**
```bash
uv run python scripts/wavelet_loss_latent.py image1.png image2.png [options]
```

**Arguments:**
- `image1` - Path to first input image
- `image2` - Path to second input image

**Options:**
- `--vae-model VAE_MODEL` - Hugging Face VAE model name or path (default: stabilityai/sd-vae-ft-mse)
- `--wavelet WAVELET` - Wavelet family to use (default: db4)
- `--level LEVEL` - Wavelet decomposition level (default: 3)
- `--transform-type {dwt,swt,qwt}` - Wavelet transform type (default: dwt)
- `--grayscale` - Convert images to grayscale
- `--device DEVICE` - Compute device (auto, cuda, cpu) (default: auto)
- `--save-plot SAVE_PLOT` - Path to save loss visualization plot
- `--output-dir OUTPUT_DIR` - Directory to save results (default: wavelet_loss_results)

**Features:**
- Encodes images to VAE latent space before computing wavelet loss
- Generates comprehensive visualizations with 9 different charts
- Saves detailed text results with component breakdowns
- Uses proper VAE preprocessing from Diffusers library

## Visualization Scripts

### `visualize_wavelet_transforms.py`
Visualize wavelet transforms applied directly to images.

**Usage:**
```bash
uv run python scripts/visualize_wavelet_transforms.py image.png [options]
```

**Arguments:**
- `image` - Path to input image

**Options:**
- `--wavelet WAVELET` - Wavelet family to use (default: db4)
- `--level LEVEL` - Wavelet decomposition levels (default: 3)
- `--transforms {dwt,swt,qwt} [...]` - Wavelet transform types to visualize (default: all)
- `--output-dir OUTPUT_DIR` - Directory to save visualization images (default: visualizations)
- `--output-formats {png,jpg,webp,avif} [...]` - Output image formats (default: png)
- `--grayscale` - Convert image to grayscale
- `--quality {1-100}` - Output image quality (default: 95)

### `visualize_vae_latent_transforms.py`
Visualize wavelet transforms applied to VAE latent representations.

**Usage:**
```bash
uv run python scripts/visualize_vae_latent_transforms.py image.png [options]
```

**Arguments:**
- `image` - Path to input image

**Options:**
- `--vae-model VAE_MODEL` - Hugging Face VAE model name or path (default: stabilityai/sd-vae-ft-mse)
- `--device DEVICE` - Device to use (cpu, cuda, auto) (default: auto)
- `--wavelet WAVELET` - Wavelet family to use (default: db4)
- `--level LEVEL` - Wavelet decomposition levels (default: 2)
- `--transforms {dwt,swt,qwt} [...]` - Wavelet transform types to visualize (default: all)
- `--output-dir OUTPUT_DIR` - Directory to save visualization images (default: vae_latent_visualizations)
- `--output-formats {png,jpg,webp,avif} [...]` - Output image formats (default: png)
- `--grayscale` - Convert image to grayscale
- `--quality {1-100}` - Output image quality (default: 95)

**Features:**
- Shows original image, VAE reconstruction, and wavelet transforms
- Displays each VAE latent channel (4 channels) separately in rows
- Maintains proper aspect ratios throughout the pipeline
- Uses Diffusers VaeImageProcessor for consistent preprocessing

## Utility Modules

### `utils/image_processing.py`
Common image loading and processing utilities:
- `load_image()` - Load images with OpenCV or PIL fallback
- `generate_image_hash()` - Generate unique hashes for images

### `utils/vae_utils.py`
VAE model and preprocessing utilities:
- `load_vae_model()` - Load VAE model and processor from Hugging Face
- `preprocess_image_with_vae_processor()` - Process images for VAE input
- `encode_image_to_latent()` - Encode images to latent space with proper scaling

## Examples

**Compare two images with basic wavelet loss:**
```bash
uv run python scripts/wavelet_loss_image.py test_image1.png test_image2.png
```

**Compare images in VAE latent space with detailed analysis:**
```bash
uv run python scripts/wavelet_loss_latent.py test_image1.png test_image2.png --level 2 --wavelet haar
```

**Visualize wavelet transforms of an image:**
```bash
uv run python scripts/visualize_wavelet_transforms.py test_image.png --transforms dwt swt
```

**Visualize VAE latent wavelet transforms:**
```bash
uv run python scripts/visualize_vae_latent_transforms.py test_image.png --level 3
```

## Output

- **Loss scripts** generate visualization plots and detailed text results
- **Visualization scripts** save transform visualizations in specified formats
- All outputs include unique identifiers based on parameters and image hashes
- Results are organized in dedicated output directories