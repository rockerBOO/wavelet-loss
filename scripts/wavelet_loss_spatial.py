#!/usr/bin/env python3
"""
Script to analyze spatial wavelet loss between two images.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

# Optional image loading libraries
try:
    import cv2
except ImportError:
    cv2 = None

from PIL import Image

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


def plot_spatial_loss(loss_tensor, title, save_path=None):
    if isinstance(loss_tensor, (float, int)):
        print(f"{title}: {loss_tensor}")
        return

    loss_tensor = torch.as_tensor(loss_tensor).detach()
    if loss_tensor.numel() == 1:
        print(f"{title}: {loss_tensor.item()}")
        return

    # 1) If multi-channel, convert to single-channel (mean or take first)
    loss_np = loss_tensor.squeeze().cpu().numpy()
    if loss_np.ndim == 3:  # (C, H, W)
        loss_np = loss_np.mean(axis=0)  # or loss_np[0]

    plt.figure(figsize=(10, 6))

    vmin = np.percentile(loss_np, 1)
    vmax = np.percentile(loss_np, 99)
    im = plt.imshow(loss_np, cmap="coolwarm", vmin=vmin, vmax=vmax)
    plt.colorbar(im)
    plt.title(title)

    if save_path:
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()
    else:
        plt.show()


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
                img = np.array(Image.fromarray(img).resize((new_w, new_h), Image.Resampling.LANCZOS))

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
    Main function to compute and visualize spatial wavelet loss.
    """
    parser = argparse.ArgumentParser(description="Compute Spatial Wavelet Loss")
    parser.add_argument("image1", type=str, help="Path to first input image")
    parser.add_argument("image2", type=str, help="Path to second input image")

    # Wavelet configuration arguments
    parser.add_argument("--transform", type=str, default="dwt", help="Wavelet transform to use (default: dwt). dwt, swt, qwt")
    parser.add_argument("--wavelet", type=str, default="db4", help="Wavelet family to use (default: db4)")
    parser.add_argument("--level", type=int, default=3, help="Wavelet decomposition level (default: 3)")
    parser.add_argument(
        "--max-size", type=int, default=None, help="Maximum size for image resizing while maintaining aspect ratio"
    )

    # Output configuration
    parser.add_argument(
        "--output-dir", type=str, default="outputs/wavelet_loss_spatial", help="Directory to save output visualizations"
    )

    args = parser.parse_args()

    # Create output directory

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load images
    img1 = load_image(args.image1)
    img2 = load_image(args.image2)

    # Preprocess images with aspect ratio preservation
    img1_tensor = preprocess_with_aspect_ratio(img1, max_size=args.max_size).to(device)
    img2_tensor = preprocess_with_aspect_ratio(img2, max_size=args.max_size).to(device)

    # Initialize WaveletLoss
    loss_fn = WaveletLoss(
        wavelet=args.wavelet,
        level=args.level,
        transform_type=args.transform,
        device=device,
        band_weights={"ll": 1.0, "lh": 1.0, "hl": 1.0, "hh": 1.0},
    )

    # Compute losses
    losses, metrics = loss_fn(img1_tensor, img2_tensor)

    bands = ["ll", "lh", "hl", "hh"]

    print("Losses:")
    print(len(losses))

    print("Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # Print loss details
    print()
    print("Total weighted losses per Level:")
    for i, loss in enumerate(losses, 1):
        print(f"Band: {bands[i % 4]}{i // 4} {loss.mean().item()}")


    for i, values in enumerate(losses):
        key = f"{bands[i % 4]}{i // 4}"
        # Plot if it's a multi-dimensional tensor
        if isinstance(values, torch.Tensor) and values.ndim > 2:
            print(f"Plotting {key}")
            # 
            plot_spatial_loss(
                values,
                f"Spatial {key.replace('_', ' ').title()}",
                save_path=os.path.join(args.output_dir, f"{key}_spatial.png"),
            )
        else:
            print(f"{key} is not a multi-dimensional tensor, skipping visualization.")


if __name__ == "__main__":
    main()
