#!/usr/bin/env python3
"""
Unified script to analyze spatial wavelet loss between two inputs (images or VAE latents).
Supports both regular images and VAE-encoded latent representations.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from wavelet_loss import WaveletLoss

# Import utility functions
try:
    from utils.plotting import plot_spatial_loss, normalize_coefficients
except ImportError:
    # Fallback to local implementations if utils are not available
    def normalize_coefficients(coeff_data, percentile_clip=True):
        """Normalize coefficient data for visualization."""
        if percentile_clip:
            low = np.percentile(coeff_data, 1)
            high = np.percentile(coeff_data, 99)
            coeff_norm = np.clip((coeff_data - low) / (high - low + 1e-8), 0, 1)
        else:
            coeff_norm = (coeff_data - coeff_data.min()) / (coeff_data.max() - coeff_data.min() + 1e-8)
        return coeff_norm

    def plot_spatial_loss(loss_tensor, title="Spatial Loss", cmap="coolwarm", percentile_clip=True, save_path=None):
        """Visualize spatial loss with advanced normalization."""
        if isinstance(loss_tensor, (float, int)):
            print(f"{title}: {loss_tensor}")
            return

        loss_tensor = torch.as_tensor(loss_tensor).detach()
        if loss_tensor.numel() == 1:
            print(f"{title}: {loss_tensor.item()}")
            return

        # Convert to numpy and handle multi-channel case
        loss_np = loss_tensor.squeeze().cpu().numpy()
        if loss_np.ndim == 3:
            loss_np = loss_np.mean(axis=0)  # Average across channels

        plt.figure(figsize=(10, 6))
        vmin, vmax = loss_np.min(), loss_np.max()
        im = plt.imshow(loss_np, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
        plt.colorbar(im, label="Raw Loss")
        plt.title(title)
        plt.axis("off")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
            plt.close()
        else:
            plt.show()


from utils.vae_utils import preprocess_tensor


def plot_individual_channel(channel_loss, channel_idx, band_key, input_type, output_dir, tight_layout=True):
    """
    Plot individual channel loss as a separate file.

    Args:
        channel_loss (np.ndarray): Single channel loss data
        channel_idx (int): Channel index
        band_key (str): Band identifier (e.g., 'll0', 'lh1')
        input_type (str): Type of input ('image' or 'latent')
        output_dir (str): Directory to save plots
        tight_layout (bool): Use tight layout for minimal spacing
    """
    # Create tight figure
    if tight_layout:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Plot with colorbar using raw loss values
    vmin, vmax = channel_loss.min(), channel_loss.max()
    im = ax.imshow(channel_loss, cmap="coolwarm", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(f"{input_type.capitalize()} {band_key.upper()} Ch{channel_idx}", fontsize=12, pad=10)
    ax.axis("off")

    # Add colorbar with tight spacing
    if tight_layout:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Raw Loss", fontsize=10)
        cbar.ax.tick_params(labelsize=8)
    else:
        plt.colorbar(im, ax=ax, label="Raw Loss")

    # Save individual channel file
    save_path = os.path.join(output_dir, f"{band_key}_{input_type}_ch{channel_idx}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()

    return save_path


def plot_combined_channels(loss_tensor, band_key, input_type, output_dir, tight_layout=True):
    """
    Plot all channels in a single tight grid.

    Args:
        loss_tensor (torch.Tensor): Multi-channel loss tensor (B, C, H, W)
        band_key (str): Band identifier
        input_type (str): Type of input ('image' or 'latent')
        output_dir (str): Directory to save plots
        tight_layout (bool): Use tight layout for minimal spacing
    """
    n_channels = loss_tensor.shape[1]

    # Calculate grid dimensions for tight layout
    if n_channels <= 4:
        cols = n_channels
        rows = 1
        figsize = (3 * cols, 3)
    elif n_channels <= 8:
        cols = 4
        rows = 2
        figsize = (3 * cols, 3 * rows)
    else:
        cols = 4
        rows = (n_channels + cols - 1) // cols  # Ceiling division
        figsize = (3 * cols, 3 * rows)

    if tight_layout:
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
    else:
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    # Handle single subplot case
    if n_channels == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if hasattr(axes, "__len__") else [axes]
    else:
        axes = axes.flatten()

    # Plot each channel
    for c in range(n_channels):
        channel_loss = loss_tensor[0, c].cpu().numpy()

        vmin, vmax = channel_loss.min(), channel_loss.max()
        im = axes[c].imshow(channel_loss, cmap="coolwarm", interpolation="nearest", vmin=vmin, vmax=vmax)
        axes[c].set_title(f"Ch{c}", fontsize=8, pad=2)
        axes[c].axis("off")

        # Add colorbar to each subplot
        if tight_layout:
            cbar = plt.colorbar(im, ax=axes[c], shrink=0.5, pad=0.01)
            cbar.ax.tick_params(labelsize=5)
        else:
            plt.colorbar(im, ax=axes[c], shrink=0.8)

    # Hide unused subplots
    for c in range(n_channels, len(axes)):
        axes[c].axis("off")

    # Add overall title
    plt.suptitle(f"{input_type.capitalize()} {band_key.upper()} Loss - All Channels", fontsize=12)

    # Apply tight layout
    if tight_layout:
        plt.tight_layout()

    # Save combined file
    save_path = os.path.join(output_dir, f"{band_key}_{input_type}_all_channels.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    return save_path


def plot_channel_losses(losses, bands, output_dir, input_type="image", individual_files=True, tight_layout=True):
    """
    Plot spatial losses for each channel and band with improved layout and individual file options.

    Args:
        losses (list): List of loss tensors from WaveletLoss
        bands (list): List of band names
        output_dir (str): Directory to save plots
        input_type (str): Type of input ('image' or 'latent')
        individual_files (bool): Whether to save individual channel files
        tight_layout (bool): Use tight layout for minimal spacing
    """
    os.makedirs(output_dir, exist_ok=True)
    individual_dir = os.path.join(output_dir, "individual_channels")
    combined_dir = os.path.join(output_dir, "combined_channels")

    if individual_files:
        os.makedirs(individual_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)

    for i, loss_tensor in enumerate(losses):
        band_key = f"{bands[i % 4]}{i // 4}"

        # Plot if it's a multi-dimensional tensor
        if isinstance(loss_tensor, torch.Tensor) and loss_tensor.ndim > 2:
            print(f"Plotting {band_key} loss")

            # Handle multi-channel tensors
            if loss_tensor.ndim == 4:  # (B, C, H, W)
                n_channels = loss_tensor.shape[1]

                # Always create combined visualization
                combined_path = plot_combined_channels(loss_tensor, band_key, input_type, combined_dir, tight_layout)
                print(f"  Saved combined: {combined_path}")

                # Optionally create individual channel files
                if individual_files:
                    print("  Saving individual channels...")
                    for c in range(n_channels):
                        channel_loss = loss_tensor[0, c].cpu().numpy()
                        individual_path = plot_individual_channel(
                            channel_loss, c, band_key, input_type, individual_dir, tight_layout
                        )
                        print(f"    Ch{c}: {individual_path}")
            else:
                # Use existing function for other tensor shapes (3D tensors, etc.)
                plot_spatial_loss(
                    loss_tensor,
                    f"Spatial {band_key.replace('_', ' ').title()} Loss - {input_type.capitalize()}",
                    cmap="coolwarm",
                    save_path=os.path.join(combined_dir, f"{band_key}_{input_type}_spatial.png"),
                )
        else:
            print(
                f"{band_key} is not a multi-dimensional tensor (shape: {loss_tensor.shape if hasattr(loss_tensor, 'shape') else type(loss_tensor)}), skipping visualization."
            )


def main():
    """
    Main function to compute and visualize spatial wavelet loss for images or VAE latents.
    """
    parser = argparse.ArgumentParser(description="Compute Spatial Wavelet Loss - Unified (Images & VAE Latents)")

    # Input arguments
    parser.add_argument("input1", type=str, help="Path to first input (image file)")
    parser.add_argument("input2", type=str, help="Path to second input (image file)")
    parser.add_argument(
        "--input-type",
        type=str,
        choices=["image", "latent"],
        default="image",
        help="Type of input processing (default: image)",
    )

    # VAE-specific arguments (only used when input-type is 'latent')
    parser.add_argument(
        "--vae-model",
        type=str,
        default="stabilityai/sd-vae-ft-mse",
        help="VAE model name or path (used when input-type='latent')",
    )
    parser.add_argument("--subfolder", type=str, help="Subfolder for VAE model")

    # Wavelet configuration arguments
    parser.add_argument(
        "--transform", type=str, default="dwt", help="Wavelet transform to use (default: dwt). dwt, swt, qwt"
    )
    parser.add_argument("--wavelet", type=str, default="db4", help="Wavelet family to use (default: db4)")
    parser.add_argument("--level", type=int, default=3, help="Wavelet decomposition level (default: 3)")

    # Image preprocessing arguments (used for both image and latent modes)
    parser.add_argument(
        "--target-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), help="Target size for resizing (width height)"
    )
    parser.add_argument("--min-size", type=int, default=256, help="Minimum size for image dimensions (default: 256)")
    parser.add_argument("--no-power-of-two", action="store_true", help="Don't force power-of-two dimensions")
    parser.add_argument("--grayscale", action="store_true", help="Convert images to grayscale")

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/wavelet_loss_spatial_unified",
        help="Directory to save output visualizations",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cpu, cuda, auto)")

    # Visualization options
    parser.add_argument(
        "--no-individual-files", action="store_true", help="Don't save individual channel files (only combined)"
    )
    parser.add_argument("--loose-layout", action="store_true", help="Use loose layout instead of tight spacing")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Processing inputs as: {args.input_type}")

    # Preprocess inputs using unified function
    print(f"Processing input 1: {args.input1}")
    tensor1 = preprocess_tensor(
        args.input1,
        args.input_type,
        vae_model=args.vae_model if args.input_type == "latent" else None,
        subfolder=args.subfolder,
        device=device,
        target_size=tuple(args.target_size) if args.target_size else None,
        min_size=args.min_size,
        force_power_of_two=not args.no_power_of_two,
        grayscale=args.grayscale,
    )

    print(f"Processing input 2: {args.input2}")
    tensor2 = preprocess_tensor(
        args.input2,
        args.input_type,
        vae_model=args.vae_model if args.input_type == "latent" else None,
        subfolder=args.subfolder,
        device=device,
        target_size=tuple(args.target_size) if args.target_size else None,
        min_size=args.min_size,
        force_power_of_two=not args.no_power_of_two,
        grayscale=args.grayscale,
    )

    print(f"Tensor 1 shape: {tensor1.shape}")
    print(f"Tensor 2 shape: {tensor2.shape}")

    # Ensure tensors are on the same device
    tensor1 = tensor1.to(device)
    tensor2 = tensor2.to(device)

    # Initialize WaveletLoss
    loss_fn = WaveletLoss(
        wavelet=args.wavelet,
        level=args.level,
        transform_type=args.transform,
        device=device,
        band_weights={"ll": 1.0, "lh": 1.0, "hl": 1.0, "hh": 1.0},
    )

    # Compute losses
    print("\nComputing wavelet losses...")
    losses, metrics = loss_fn(tensor1, tensor2)

    bands = ["ll", "lh", "hl", "hh"]

    # Print overall metrics
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Print loss details
    print("\nTotal weighted losses per band:")
    for i, loss in enumerate(losses):
        band_key = f"{bands[i % 4]}{i // 4}"
        if isinstance(loss, torch.Tensor):
            print(f"  {band_key}: {loss.mean().item():.6f} (shape: {loss.shape})")
        else:
            print(f"  {band_key}: {loss}")

    # Plot spatial losses with channel awareness
    print("\nGenerating visualizations...")
    plot_channel_losses(
        losses,
        bands,
        args.output_dir,
        args.input_type,
        individual_files=not args.no_individual_files,
        tight_layout=not args.loose_layout,
    )

    print("\nVisualization complete! Outputs saved to:")
    print(f"  Combined channels: {args.output_dir}/combined_channels/")
    if not args.no_individual_files:
        print(f"  Individual channels: {args.output_dir}/individual_channels/")


if __name__ == "__main__":
    main()
