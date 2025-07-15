#!/usr/bin/env python3
"""
Visualize different wavelet transform decompositions for VAE latents.
Supports DWT, SWT, and QWT transforms on VAE-encoded representations.
"""

from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from wavelet_transform import (
    DiscreteWaveletTransform,
    StationaryWaveletTransform,
    QuaternionWaveletTransform,
)

from utils.image_processing import load_image, generate_image_hash
from utils.vae_utils import load_vae_model, preprocess_image_with_vae_processor, encode_image_to_latent


def visualize_vae_latent_transforms(
    latent_tensor,
    wavelet="db4",
    level=2,
    save_paths: list[Path] | None = None,
    quality=95,
    transform_class=None,
    transform_desc=None,
    original_image=None,
    reconstructed_image=None,
    original_shape=None,
    **kwargs,
):
    """
    Visualize different wavelet transform decompositions on VAE latents.

    Args:
        latent_tensor (torch.Tensor): Input VAE latent tensor
        wavelet (str): Wavelet family to use
        level (int): Decomposition levels
        save_paths (list, optional): Paths to save the visualization in different formats
        quality (int): Output image quality (1-100)
        transform_class (type, optional): Specific wavelet transform class to use
        transform_desc (str, optional): Description for the transform
        original_image (torch.Tensor, optional): Original input image
        reconstructed_image (torch.Tensor, optional): VAE reconstructed image
    """
    # Prepare transforms
    device = latent_tensor.device

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

    # Determine number of latent channels
    n_channels = latent_tensor.shape[1]

    # Create a figure for each transform type
    for transform_name, transform in transforms.items():
        # Decompose the latent tensor
        if "QWT" in transform_name:
            # For QWT, process all four components
            coeffs = transform.decompose(latent_tensor, level)
            components = ["r", "i", "j", "k"]
            sample_component_coeffs = coeffs["r"]
        else:
            # For other transforms, use single band
            coeffs = transform.decompose(latent_tensor, level)
            components = [None]  # Placeholder for single-band transforms
            sample_component_coeffs = coeffs

        # Create a figure with additional columns for original and reconstructed images
        # Each wavelet coefficient type gets one column per band
        total_cols = level * len(bands) + 3  # +3 for original, latent, reconstructed
        # Each channel gets its own row
        total_rows = n_channels
        
        # Calculate aspect ratios for proper sizing
        if original_image is not None:
            # Use actual image tensor aspect ratio
            if original_image.shape[1] == 3:
                img_shape = original_image[0].permute(1, 2, 0).cpu().numpy().shape
            else:
                img_shape = original_image[0, 0].cpu().numpy().shape
            img_aspect = img_shape[1] / img_shape[0]  # width/height
        else:
            img_aspect = 1.0  # Default to square
        
        # Calculate latent aspect ratio
        latent_aspect = latent_tensor.shape[3] / latent_tensor.shape[2]  # width/height
        
        # Calculate aspect ratios for each level and band (same for all channels)
        coeff_aspects = []
        for level_idx in range(level):
            for band in bands:
                coeff_data = sample_component_coeffs[band][level_idx][0, 0].cpu().numpy()
                coeff_aspect = coeff_data.shape[1] / coeff_data.shape[0]  # width/height
                coeff_aspects.append(coeff_aspect)
        
        # Create figure with variable width for different aspect ratios
        # Use different widths based on aspect ratios
        col_widths = []
        col_widths.append(img_aspect * 4)  # Original image
        col_widths.append(latent_aspect * 4)  # Latent
        col_widths.append(img_aspect * 4)  # Reconstructed
        
        # Add coefficient columns with their actual aspect ratios
        for coeff_aspect in coeff_aspects:
            col_widths.append(coeff_aspect * 4)
        
        # Create subplots with custom widths and heights
        # Each VAE latent channel gets its own row
        row_heights = [1.0] * total_rows
        
        fig, axes = plt.subplots(
            total_rows, total_cols, 
            figsize=(sum(col_widths), 4 * total_rows),
            gridspec_kw={'width_ratios': col_widths, 'height_ratios': row_heights}
        )
        plt.subplots_adjust(
            wspace=0.05, hspace=0.15, left=0.02, right=0.98, top=0.95, bottom=0.05
        )

        # Ensure axes is always 2D for consistency
        if total_rows == 1:
            axes = [axes]

        # Plot original and reconstructed images in all rows, or hide them
        for row_idx in range(total_rows):
            if row_idx == 0:
                # Plot original image if provided (only in first row)
                if original_image is not None:
                    if original_image.shape[1] == 3:
                        # RGB image
                        orig_img = original_image[0].permute(1, 2, 0).cpu().numpy()
                    else:
                        # Grayscale
                        orig_img = original_image[0, 0].cpu().numpy()

                    orig_img = np.clip(orig_img, 0, 1)
                    
                    # Display with correct aspect ratio
                    axes[row_idx][0].imshow(orig_img, aspect='auto')
                    axes[row_idx][0].set_title("Original Image", fontsize=10)
                    axes[row_idx][0].axis("off")

                # Plot reconstructed image if provided (only in first row)
                if reconstructed_image is not None:
                    if reconstructed_image.shape[1] == 3:
                        # RGB image
                        recon_img = reconstructed_image[0].permute(1, 2, 0).cpu().numpy()
                    else:
                        # Grayscale
                        recon_img = reconstructed_image[0, 0].cpu().numpy()

                    recon_img = np.clip(recon_img, 0, 1)
                    
                    # Display with correct aspect ratio
                    axes[row_idx][2].imshow(recon_img, aspect='auto')
                    axes[row_idx][2].set_title("Reconstructed", fontsize=10)
                    axes[row_idx][2].axis("off")
            else:
                # Hide original and reconstructed image columns for other rows
                axes[row_idx][0].axis("off")
                axes[row_idx][2].axis("off")

        # Plot individual latent channels and their wavelet coefficients
        for channel_idx in range(n_channels):
            # Plot individual latent channel
            latent_channel = latent_tensor[0, channel_idx].cpu().numpy()
            latent_norm = (latent_channel - latent_channel.min()) / (
                latent_channel.max() - latent_channel.min() + 1e-8
            )
            axes[channel_idx][1].imshow(latent_norm, cmap="RdBu_r", aspect='auto')
            axes[channel_idx][1].set_title(f"VAE Latent Ch{channel_idx}", fontsize=10)
            axes[channel_idx][1].axis("off")

        # Iterate through components (or single band for DWT/SWT)
        for comp_idx, component in enumerate(components):
            # Select coefficients for this component
            if component is not None:
                comp_coeffs = coeffs[component]
            else:
                comp_coeffs = coeffs

            # Visualize each channel in its own row
            for channel_idx in range(n_channels):
                # Visualize each level and band for this channel
                for level_idx in range(level):
                    for band_idx, band in enumerate(bands):
                        # Compute column index (accounting for original, latent, reconstructed columns)
                        col_idx = level_idx * len(bands) + band_idx + 3

                        # Get coefficient data for specific channel
                        if component is not None:
                            coeff_data = comp_coeffs[band][level_idx][0, channel_idx].cpu().numpy()
                        else:
                            coeff_data = comp_coeffs[band][level_idx][0, channel_idx].cpu().numpy()

                        # Normalize for visualization
                        coeff_norm = (coeff_data - coeff_data.min()) / (
                            coeff_data.max() - coeff_data.min() + 1e-8
                        )

                        # Plot
                        title = f"{band.upper()}{level_idx + 1}"
                        if component is not None:
                            title = f"{component.upper()}-{title}"

                        axes[channel_idx][col_idx].imshow(
                            coeff_norm, cmap="RdBu_r", aspect='auto'
                        )
                        axes[channel_idx][col_idx].set_title(title, fontsize=8)
                        axes[channel_idx][col_idx].axis("off")

        # Add transform type as suptitle
        plt.suptitle(
            f"VAE Latent {transform_name}\\n{wavelet} Wavelet, {level} Levels, {n_channels} Latent Channels",
            fontsize=16,
        )

        # Determine save path for this transform
        if save_paths:
            print(
                f"Saving {transform_name} VAE latent visualization to {len(save_paths)} file(s):"
            )
            for save_path in save_paths:
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
    Main function to parse arguments and visualize VAE latent wavelet transforms
    """
    parser = argparse.ArgumentParser(
        description="Visualize VAE Latent Wavelet Transforms"
    )

    # Image input arguments
    parser.add_argument("image", type=str, help="Path to input image")

    # VAE configuration
    parser.add_argument(
        "--vae-model", 
        type=str, 
        default="stabilityai/sd-vae-ft-mse",
        help="Hugging Face VAE model name or path (default: stabilityai/sd-vae-ft-mse)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use (cpu, cuda, auto) (default: auto)"
    )

    # Wavelet transform configuration
    parser.add_argument(
        "--wavelet",
        type=str,
        default="db4",
        help="Wavelet family to use (default: db4)",
    )
    parser.add_argument(
        "--level", type=int, default=2, help="Wavelet decomposition levels (default: 2)"
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
        default="vae_latent_visualizations",
        help="Directory to save visualization images (default: vae_latent_visualizations)",
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

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load and preprocess image
    img = load_image(args.image, grayscale=args.grayscale)
    
    # Generate hash for the original image
    image_hash = generate_image_hash(args.image)
    print(f"Image hash: {image_hash}")
    
    # Load VAE model and processor
    print(f"Loading VAE model: {args.vae_model}")
    vae, processor = load_vae_model(args.vae_model)
    vae = vae.to(device)
    vae.eval()
    
    # Preprocess image using VaeImageProcessor
    img_tensor = preprocess_image_with_vae_processor(img, processor)
    
    # Store original image shape for aspect ratio calculation
    original_aspect = img.shape[1] / img.shape[0]  # width/height

    # Encode image to latent space
    latent, reconstructed, img_tensor_display = encode_image_to_latent(vae, processor, img_tensor, device)

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
        # Generate improved filename with wavelet type, levels, and image hash
        # Format: vae_latent_transforms_{transform}_{wavelet}_L{levels}_{hash}.{ext}
        base_filename = f"vae_latent_transforms_{transform_name}_{args.wavelet}_L{args.level}_{image_hash}"

        # Generate output paths
        output_paths = [
            Path(output_dir / f"{base_filename}.{fmt}") for fmt in args.output_formats
        ]

        # Create visualization
        visualize_vae_latent_transforms(
            latent,
            wavelet=args.wavelet,
            level=args.level,
            save_paths=output_paths,
            quality=args.quality,
            transform_class=transform_class,
            transform_desc=transform_desc,
            original_image=img_tensor_display,
            reconstructed_image=reconstructed,
        )


if __name__ == "__main__":
    main()
