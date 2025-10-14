#!/usr/bin/env python3
"""
Script to analyze and visualize wavelet coefficients between two images.
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import pywt

# Optional image loading libraries
try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None

from wavelet_loss_image import load_image, preprocess_image
from wavelet_loss import WaveletLoss


def normalize_coeffs(coeff_data):
    """
    Normalize coefficient data for visualization.

    Args:
        coeff_data (np.ndarray): Input coefficient data

    Returns:
        np.ndarray: Normalized coefficient data
    """
    coeff_norm = (coeff_data - coeff_data.min()) / (coeff_data.max() - coeff_data.min() + 1e-8)
    return coeff_norm


def visualize_wavelet_coeffs(coeffs_list, title_prefix="", save_path=None):
    """
    Visualize wavelet coefficients.

    Args:
        coeffs_list (list): List of wavelet coefficient lists
        title_prefix (str): Prefix for plot titles
        save_path (str, optional): Path to save visualization
    """
    # Define bands
    bands = ["LL", "LH", "HL", "HH"]

    # Determine number of channels and levels
    n_channels = len(coeffs_list)
    levels = len(coeffs_list[0]) - 1  # Subtract 1 because first element is LL band

    # Create figure
    fig, axes = plt.subplots(n_channels, levels + 1, figsize=(20, 4 * n_channels), squeeze=False)

    # Visualize each channel
    for channel_idx in range(n_channels):
        # LL (approximation) coefficient
        coeff_data = coeffs_list[channel_idx][0]
        coeff_norm = normalize_coeffs(coeff_data)

        axes[channel_idx][0].imshow(coeff_norm, cmap="viridis")
        axes[channel_idx][0].set_title(f"{title_prefix}Channel {channel_idx}: LL\n{coeff_data.shape}")
        axes[channel_idx][0].axis("off")

        # Detail coefficients for each level
        for level_idx in range(levels):
            level_coeffs = coeffs_list[channel_idx][level_idx + 1]

            # Plot LH, HL, HH bands
            for band_idx, band in enumerate(["LH", "HL", "HH"]):
                coeff_data = level_coeffs[band_idx]
                coeff_norm = normalize_coeffs(coeff_data)

                axes[channel_idx][level_idx + 1].imshow(coeff_norm, cmap="coolwarm")
                axes[channel_idx][level_idx + 1].set_title(
                    f"{title_prefix}Ch{channel_idx} L{level_idx + 1} {band}\n{coeff_data.shape}"
                )
                axes[channel_idx][level_idx + 1].axis("off")

    # Adjust layout with minimal whitespace
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)

    # Save or show
    if save_path:
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()
    else:
        plt.show()


def compute_coefficient_difference(coeffs1, coeffs2):
    """
    Compute the difference between two sets of wavelet coefficients.

    Args:
        coeffs1 (list): First set of wavelet coefficients
        coeffs2 (list): Second set of wavelet coefficients

    Returns:
        list: Difference between corresponding coefficients
    """
    # Validate input
    assert len(coeffs1) == len(coeffs2), "Coefficient lists must have the same length"

    diff_coeffs = []

    # Compute difference for LL (approximation) band
    diff_ll = coeffs1[0] - coeffs2[0]
    diff_coeffs.append(diff_ll)

    # Compute difference for detail bands
    levels = len(coeffs1) - 1
    for level_idx in range(levels):
        # Get detail coefficients for this level
        level_coeffs1 = coeffs1[level_idx + 1]
        level_coeffs2 = coeffs2[level_idx + 1]

        # Compute difference for each detail band (LH, HL, HH)
        diff_level_coeffs = [
            level_coeffs1[0] - level_coeffs2[0],  # LH
            level_coeffs1[1] - level_coeffs2[1],  # HL
            level_coeffs1[2] - level_coeffs2[2],  # HH
        ]

        diff_coeffs.append(diff_level_coeffs)

    return diff_coeffs


def visualize_coefficient_differences(diff_coeffs_list, save_path=None):
    """
    Visualize differences in wavelet coefficients.

    Args:
        diff_coeffs_list (list): List of coefficient differences
        save_path (str, optional): Path to save visualization
    """
    # Define bands
    bands = ["LL", "LH", "HL", "HH"]

    # Determine number of channels and levels
    n_channels = len(diff_coeffs_list)
    levels = len(diff_coeffs_list[0]) - 1  # Subtract 1 because first element is LL band

    # Create figure
    fig, axes = plt.subplots(n_channels, levels + 1, figsize=(20, 4 * n_channels), squeeze=False)

    # Visualize each channel's coefficient differences
    for channel_idx in range(n_channels):
        # LL (approximation) difference
        diff_data = diff_coeffs_list[channel_idx][0]
        diff_norm = normalize_coeffs(np.abs(diff_data))

        axes[channel_idx][0].imshow(diff_norm, cmap="RdBu_r")
        axes[channel_idx][0].set_title(f"Channel {channel_idx}: LL Diff\n{diff_data.shape}")
        axes[channel_idx][0].axis("off")

        # Detail coefficient differences for each level
        for level_idx in range(levels):
            level_diff_coeffs = diff_coeffs_list[channel_idx][level_idx + 1]

            # Plot LH, HL, HH band differences
            for band_idx, band in enumerate(["LH", "HL", "HH"]):
                diff_data = level_diff_coeffs[band_idx]
                diff_norm = normalize_coeffs(np.abs(diff_data))

                axes[channel_idx][level_idx + 1].imshow(diff_norm, cmap="RdBu_r")
                axes[channel_idx][level_idx + 1].set_title(
                    f"Ch{channel_idx} L{level_idx + 1} {band} Diff\n{diff_data.shape}"
                )
                axes[channel_idx][level_idx + 1].axis("off")

    # Adjust layout with minimal whitespace
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)

    # Save or show
    if save_path:
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()
    else:
        plt.show()


def preprocess_with_aspect_ratio(img, max_size=None):
    """
    Preprocess image while maintaining aspect ratio.

    Args:
        img (numpy.ndarray): Input image
        max_size (int, optional): Maximum size for resizing

    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Resize while maintaining aspect ratio if max_size is specified
    if max_size:
        h, w = img.shape[:2]
        max_dim = max(h, w)

        # Determine scaling factor to fit within max_size
        if max_dim > max_size:
            scale = max_size / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)

            if cv2 is not None:
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                img = np.array(Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS))

    # Convert to float and normalize
    img_float = img.astype(np.float32)
    img_normalized = (img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-8)

    # Add batch and channel dimensions
    if len(img_normalized.shape) == 2:
        img_normalized = img_normalized[np.newaxis, np.newaxis, :, :]
    elif len(img_normalized.shape) == 3:
        # Convert RGB to (B, C, H, W)
        if img_normalized.shape[2] == 3:
            img_normalized = img_normalized.transpose(2, 0, 1)
        img_normalized = img_normalized[np.newaxis, :, :, :]

    # Convert to PyTorch tensor
    return torch.from_numpy(img_normalized)


def main():
    """
    Main function to analyze wavelet coefficients.
    """
    parser = argparse.ArgumentParser(description="Analyze Wavelet Coefficients")
    parser.add_argument("image1", type=str, help="Path to first input image")
    parser.add_argument("image2", type=str, help="Path to second input image")

    # Wavelet configuration arguments
    parser.add_argument("--wavelet", type=str, default="db4", help="Wavelet family to use (default: db4)")
    parser.add_argument("--level", type=int, default=3, help="Wavelet decomposition level (default: 3)")
    parser.add_argument(
        "--max-size", type=int, default=None, help="Maximum size for image resizing while maintaining aspect ratio"
    )

    # Output configuration
    parser.add_argument(
        "--output-dir", type=str, default="outputs/wavelet_coeffs", help="Directory to save output visualizations"
    )

    args = parser.parse_args()

    # Create output directory
    import os

    os.makedirs(args.output_dir, exist_ok=True)

    # Load images
    img1 = load_image(args.image1)
    img2 = load_image(args.image2)

    # Preprocess images with aspect ratio preservation
    img1_tensor = preprocess_with_aspect_ratio(img1, max_size=args.max_size)
    img2_tensor = preprocess_with_aspect_ratio(img2, max_size=args.max_size)

    # Print detailed size information
    print("\nOriginal Image Sizes:")
    print(f"Image 1: Shape = {img1.shape}, Aspect Ratio = {img1.shape[1] / img1.shape[0]:.4f}")
    print(f"Image 2: Shape = {img2.shape}, Aspect Ratio = {img2.shape[1] / img2.shape[0]:.4f}")

    print("\nPreprocessed Image Tensor Sizes:")
    print(
        f"Image 1 Tensor: Shape = {img1_tensor.shape}, Aspect Ratio = {img1_tensor.shape[3] / img1_tensor.shape[2]:.4f}"
    )
    print(
        f"Image 2 Tensor: Shape = {img2_tensor.shape}, Aspect Ratio = {img2_tensor.shape[3] / img2_tensor.shape[2]:.4f}"
    )

    # Compute wavelet coefficients
    img1_coeffs = []
    img2_coeffs = []

    # Extract wavelet coefficients for each channel
    img1_np = img1_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    img2_np = img2_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)

    print("\nWavelet Coefficient Details:")
    for ch in range(img1_np.shape[2]):
        # Extract coefficients using pywt
        img1_ch_coeffs = pywt.wavedec2(img1_np[:, :, ch], args.wavelet, level=args.level)
        img2_ch_coeffs = pywt.wavedec2(img2_np[:, :, ch], args.wavelet, level=args.level)

        # Print coefficient shape details
        print(f"\nChannel {ch} Wavelet Coefficients:")
        print("Image 1 Coefficient Shapes:")
        print(f"  LL (Approximation): {img1_ch_coeffs[0].shape}")
        for level_idx in range(1, len(img1_ch_coeffs)):
            print(f"  Level {level_idx} Detail Bands:")
            lh, hl, hh = img1_ch_coeffs[level_idx]
            print(f"    LH: {lh.shape}")
            print(f"    HL: {hl.shape}")
            print(f"    HH: {hh.shape}")

        print("Image 2 Coefficient Shapes:")
        print(f"  LL (Approximation): {img2_ch_coeffs[0].shape}")
        for level_idx in range(1, len(img2_ch_coeffs)):
            print(f"  Level {level_idx} Detail Bands:")
            lh, hl, hh = img2_ch_coeffs[level_idx]
            print(f"    LH: {lh.shape}")
            print(f"    HL: {hl.shape}")
            print(f"    HH: {hh.shape}")

        img1_coeffs.append(img1_ch_coeffs)
        img2_coeffs.append(img2_ch_coeffs)

    # Visualize coefficients for each image
    visualize_wavelet_coeffs(
        img1_coeffs, title_prefix="Image 1 ", save_path=os.path.join(args.output_dir, "img1_coeffs.png")
    )
    visualize_wavelet_coeffs(
        img2_coeffs, title_prefix="Image 2 ", save_path=os.path.join(args.output_dir, "img2_coeffs.png")
    )

    # Compute and visualize coefficient differences
    diff_coeffs_list = [
        compute_coefficient_difference(img1_coeffs[ch], img2_coeffs[ch]) for ch in range(len(img1_coeffs))
    ]

    visualize_coefficient_differences(
        diff_coeffs_list, save_path=os.path.join(args.output_dir, "coeffs_differences.png")
    )


if __name__ == "__main__":
    main()
