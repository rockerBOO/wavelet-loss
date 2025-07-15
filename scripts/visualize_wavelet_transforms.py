#!/usr/bin/env python3
"""
Visualize different wavelet transform decompositions for images.
Supports DWT, SWT, and QWT transforms.
"""

import os
from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# Optional image loading libraries
try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None

from wavelet_transform import (
    DiscreteWaveletTransform,
    StationaryWaveletTransform,
    QuaternionWaveletTransform,
)


def load_image(image_path, grayscale=False):
    """
    Load an image using either OpenCV or PIL, with optional grayscale conversion.

    Args:
        image_path (str): Path to the image file
        grayscale (bool): Whether to convert image to grayscale

    Returns:
        numpy.ndarray: Loaded image as a numpy array
    """
    # Prioritize OpenCV if available
    if cv2 is not None:
        # Read in BGR, optionally convert to grayscale
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Fall back to PIL if OpenCV is not available
    elif Image is not None:
        img = Image.open(image_path)
        if grayscale:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        img = np.array(img)

    else:
        raise ImportError("Neither OpenCV nor PIL is available for image loading")

    return img


def next_power_of_two(x):
    """
    Find the next power of two for a given number.

    Args:
        x (int): Input number

    Returns:
        int: Next power of two
    """
    return 2 ** (x - 1).bit_length()


def preprocess_image(
    img,
    target_size=None,
    normalize=True,
    force_power_of_two=True,
    min_size=512,
    max_size=1024,
):
    """
    Preprocess image for wavelet transform visualization.

    Args:
        img (numpy.ndarray): Input image
        target_size (tuple, optional): Resize image to this size
        normalize (bool): Normalize image to [0, 1] range
        force_power_of_two (bool): Ensure image dimensions are powers of two
        min_size (int): Minimum size for image dimensions
        max_size (int): Maximum size for image dimensions

    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Resize image while maintaining aspect ratio
    height, width = img.shape[:2]
    aspect_ratio = width / height

    # Determine target size if not specified
    if target_size is None:
        # Calculate dimensions that fit within min and max sizes
        if width > height:
            new_width = min(max_size, width)
            new_height = int(new_width / aspect_ratio)
            if new_height < min_size:
                new_height = min_size
                new_width = int(new_height * aspect_ratio)
        else:
            new_height = min(max_size, height)
            new_width = int(new_height * aspect_ratio)
            if new_width < min_size:
                new_width = min_size
                new_height = int(new_width / aspect_ratio)
    else:
        # Use provided target size, maintaining aspect ratio
        target_aspect = target_size[0] / target_size[1]
        if target_aspect > aspect_ratio:
            # Height is the limiting factor
            new_height = target_size[1]
            new_width = int(new_height * aspect_ratio)
        else:
            # Width is the limiting factor
            new_width = target_size[0]
            new_height = int(new_width / aspect_ratio)

    # Resize image
    if cv2 is not None:
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    elif Image is not None:
        img = np.array(
            Image.fromarray(img).resize(
                (new_width, new_height), Image.Resampling.LANCZOS
            )
        )

    # Ensure power of two dimensions if requested
    if force_power_of_two:
        # Find next power of two for both dimensions
        new_height_pow2 = next_power_of_two(new_height)
        new_width_pow2 = next_power_of_two(new_width)

        # Create padded canvas
        if len(img.shape) == 3:
            canvas = np.zeros(
                (new_height_pow2, new_width_pow2, img.shape[2]), dtype=img.dtype
            )
        else:
            canvas = np.zeros((new_height_pow2, new_width_pow2), dtype=img.dtype)

        # Center the image on the canvas
        y_offset = (new_height_pow2 - new_height) // 2
        x_offset = (new_width_pow2 - new_width) // 2
        canvas[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = img
        img = canvas

    # Add batch and channel dimensions if needed
    if len(img.shape) == 2:
        img = img[np.newaxis, np.newaxis, :, :]
    elif len(img.shape) == 3:
        # Convert RGB to (B, C, H, W)
        if img.shape[2] == 3:
            img = img.transpose(2, 0, 1)
        img = img[np.newaxis, :, :, :]

    # Convert to float and normalize
    img = img.astype(np.float32)
    if normalize:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Convert to PyTorch tensor
    return torch.from_numpy(img)


def visualize_wavelet_transforms(
    img_tensor,
    wavelet="db4",
    level=3,
    save_paths: list[Path] | None = None,
    quality=95,
    transform_class=None,
    transform_desc=None,
    **kwargs,
):
    """
    Visualize different wavelet transform decompositions.

    Args:
        img_tensor (torch.Tensor): Input image tensor
        wavelet (str): Wavelet family to use
        level (int): Decomposition levels
        save_paths (list, optional): Paths to save the visualization in different formats
        quality (int): Output image quality (1-100)
        transform_class (type, optional): Specific wavelet transform class to use
        transform_desc (str, optional): Description for the transform
    """
    # Prepare transforms
    device = img_tensor.device

    # If a specific transform is provided, use it
    if transform_class is not None:
        transforms = {
            transform_desc or transform_class.__name__: transform_class(wavelet, device)
        }
    else:
        # Default transforms
        transforms = {
            "DWT (Discrete Wavelet Transform)": DiscreteWaveletTransform(
                wavelet, device
            ),
            "SWT (Stationary Wavelet Transform)": StationaryWaveletTransform(
                wavelet, device
            ),
            "QWT (Quaternion Wavelet Transform)": QuaternionWaveletTransform(
                wavelet, device
            ),
        }

    # Set up visualization
    bands = ["ll", "lh", "hl", "hh"]

    # Determine number of channels
    n_channels = img_tensor.shape[1]

    # Create a figure for each transform type
    for transform_name, transform in transforms.items():
        # Decompose the image
        if "QWT" in transform_name:
            # For QWT, process all four components
            coeffs = transform.decompose(img_tensor, level)
            components = ["r", "i", "j", "k"]
        else:
            # For other transforms, use single band
            coeffs = transform.decompose(img_tensor, level)
            components = [None]  # Placeholder for single-band transforms

        # Create a figure with rows for components and columns for bands
        total_cols = level * len(bands) + 1
        fig, axes = plt.subplots(
            len(components), total_cols, figsize=(4 * total_cols, 4 * len(components))
        )
        plt.subplots_adjust(
            wspace=0.1, hspace=0.2, left=0.02, right=0.98, top=0.95, bottom=0.05
        )

        # Plot original image in the first column
        if len(components) == 1:
            axes = [axes]  # Ensure axes is a list for single-band transforms

        # Reconstruct multi-channel original image
        if n_channels == 3:
            # RGB image
            orig_img = img_tensor[0].permute(1, 2, 0).cpu().numpy()
        else:
            # Grayscale or single channel
            orig_img = img_tensor[0, 0].cpu().numpy()

        # Check if image needs rescaling
        if orig_img.max() <= 1.0:
            orig_img = orig_img * 255

        # Ensure image is in the correct range and type
        orig_img = np.clip(orig_img, 0, 255).astype(np.uint8)

        # Plot original image
        axes[0][0].imshow(orig_img, aspect="equal")
        axes[0][0].set_title("Original Image", fontsize=10)
        axes[0][0].axis("off")

        # Iterate through components (or single band for DWT/SWT)
        for comp_idx, component in enumerate(components):
            # Select coefficients for this component
            if component is not None:
                comp_coeffs = coeffs[component]
            else:
                comp_coeffs = coeffs

            # Visualize each level and band
            for level_idx in range(level):
                for band_idx, band in enumerate(bands):
                    # Compute column index (accounting for original image column)
                    col_idx = level_idx * len(bands) + band_idx + 1

                    # Get coefficient data
                    # For multi-channel, use first channel
                    if component is not None:
                        coeff_data = comp_coeffs[band][level_idx][0, 0].cpu().numpy()
                    else:
                        coeff_data = comp_coeffs[band][level_idx][0, 0].cpu().numpy()

                    # Normalize for visualization
                    coeff_norm = (coeff_data - coeff_data.min()) / (
                        coeff_data.max() - coeff_data.min() + 1e-8
                    )

                    # Plot
                    title = f"{band.upper()}{level_idx + 1}"
                    if component is not None:
                        title = f"{component.upper()}-{title}"

                    axes[comp_idx][col_idx].imshow(
                        coeff_norm, cmap="RdBu_r", aspect="equal"
                    )
                    axes[comp_idx][col_idx].set_title(title, fontsize=8)
                    axes[comp_idx][col_idx].axis("off")

        # Add transform type as suptitle
        plt.suptitle(
            f"{transform_name}\n{wavelet} Wavelet, {level} Levels", fontsize=16
        )

        # Determine save path for this transform
        if save_paths:
            print(
                f"Saving {transform_name} visualization to {len(save_paths)} file(s):"
            )
            for save_path in save_paths:
                # Determine file format from extension
                fmt = save_path.stem[1:]

                print(f"  - Saving {save_path}")

                # Save with specified settings
                try:
                    plt.savefig(save_path, dpi=300, bbox_inches="tight")
                    print(f"    ✓ Successfully saved {save_path}")
                except Exception as e:
                    print(f"    ✗ Failed to save {save_path}: {e}")

            # Close the figure to free up memory
            plt.close(fig)
        else:
            plt.show()


def main():
    """
    Main function to parse arguments and visualize wavelet transforms
    """
    parser = argparse.ArgumentParser(description="Visualize Wavelet Transforms")

    # Image input arguments
    parser.add_argument("image", type=str, help="Path to input image")

    # Wavelet transform configuration
    parser.add_argument(
        "--wavelet",
        type=str,
        default="db4",
        help="Wavelet family to use (default: db4)",
    )
    parser.add_argument(
        "--level", type=int, default=3, help="Wavelet decomposition levels (default: 3)"
    )

    # Transform type selection
    parser.add_argument(
        "--transforms",
        nargs="+",
        choices=["dwt", "swt", "qwt"],
        default=["dwt", "swt", "qwt"],
        help="Wavelet transform types to visualize (default: all)",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations",
        help="Directory to save visualization images (default: visualizations)",
    )
    parser.add_argument(
        "--output-formats",
        nargs="+",
        choices=["png", "jpg", "webp", "avif"],
        default=["png"],
        help="Output image formats (default: png)",
    )

    # Additional arguments
    parser.add_argument(
        "--grayscale", action="store_true", help="Convert image to grayscale"
    )
    parser.add_argument(
        "--quality",
        type=int,
        choices=range(1, 101),
        default=95,
        help="Output image quality (1-100, default: 95)",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess image
    img = load_image(args.image, grayscale=args.grayscale)
    img_tensor = preprocess_image(img)

    # Prepare transforms dictionary based on user selection
    available_transforms = {
        "dwt": ("DWT (Discrete Wavelet Transform)", DiscreteWaveletTransform),
        "swt": ("SWT (Stationary Wavelet Transform)", StationaryWaveletTransform),
        "qwt": ("QWT (Quaternion Wavelet Transform)", QuaternionWaveletTransform),
    }

    # Filter selected transforms
    selected_transforms = {
        name: (desc, transform)
        for name, (desc, transform) in available_transforms.items()
        if name in args.transforms
    }

    # Visualize transforms
    for transform_name, (
        transform_desc,
        transform_class,
    ) in selected_transforms.items():
        # Generate output paths for this transform
        # Extract the base transform type (e.g., 'dwt' from 'DWT (Discrete Wavelet Transform)')
        base_transform = transform_name.split()[0].lower()
        base_filename = f"wavelet_transforms_{base_transform}"

        # Generate output paths
        output_paths = [
            Path(output_dir / f"{base_filename}.{fmt}") for fmt in args.output_formats
        ]

        # Create visualization
        visualize_wavelet_transforms(
            img_tensor,
            wavelet=args.wavelet,
            level=args.level,
            save_paths=output_paths,
            quality=args.quality,
            transform_class=transform_class,
            transform_desc=transform_desc,
        )


if __name__ == "__main__":
    main()
