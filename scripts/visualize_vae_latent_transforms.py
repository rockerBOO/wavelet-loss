#!/usr/bin/env python3
"""
Visualize different wavelet transform decompositions for VAE latents.
Supports DWT, SWT, and QWT transforms on VAE-encoded representations.
"""

from pathlib import Path
from diffusers.models.autoencoders import AutoencoderKLQwenImage

try:
    from diffusers.models.autoencoders import AutoencoderKLQwenImage21
except ImportError:
    AutoencoderKLQwenImage21 = None

QWEN_TEMPORAL_VAE_CLASSES = tuple(
    cls for cls in (AutoencoderKLQwenImage, AutoencoderKLQwenImage21) if cls is not None
)
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
    vae_model=None,
    max_channels_per_file=16,
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
        transforms = {transform_desc or transform_class.__name__: transform_class(wavelet, device)}
    else:
        # Default transforms
        transforms = {
            "DWT (Discrete Wavelet Transform)": DiscreteWaveletTransform(wavelet, device),
            "SWT (Stationary Wavelet Transform)": StationaryWaveletTransform(wavelet, device),
            "QWT (Quaternion Wavelet Transform)": QuaternionWaveletTransform(wavelet, device),
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
            coeffs = transform.decompose_quaternion(latent_tensor, level)
            components = ["r", "i", "j", "k"]
            sample_component_coeffs = coeffs["r"]
        else:
            # For other transforms, use single band
            coeffs = transform.decompose(latent_tensor, level)
            components = [None]  # Placeholder for single-band transforms
            sample_component_coeffs = coeffs

        # Split channels into chunks so each figure/file stays a manageable size
        channel_chunks = [
            list(range(start, min(start + max_channels_per_file, n_channels)))
            for start in range(0, n_channels, max_channels_per_file)
        ]

        for chunk in channel_chunks:
            _render_and_save_chunk(
                chunk=chunk,
                coeffs=coeffs,
                components=components,
                sample_component_coeffs=sample_component_coeffs,
                bands=bands,
                level=level,
                n_channels=n_channels,
                latent_tensor=latent_tensor,
                original_image=original_image,
                reconstructed_image=reconstructed_image,
                transform_name=transform_name,
                wavelet=wavelet,
                vae_model=vae_model,
                save_paths=save_paths,
                multi_chunk=len(channel_chunks) > 1,
            )


def _render_and_save_chunk(
    chunk,
    coeffs,
    components,
    sample_component_coeffs,
    bands,
    level,
    n_channels,
    latent_tensor,
    original_image,
    reconstructed_image,
    transform_name,
    wavelet,
    vae_model,
    save_paths,
    multi_chunk,
):
    """Render one figure covering a subset (chunk) of latent channels and save it."""
    # Create a figure with additional columns for original and reconstructed images
    # Each wavelet coefficient type gets one column per band
    total_cols = level * len(bands) + 3  # +3 for original, latent, reconstructed
    # Each channel in this chunk gets its own row
    total_rows = len(chunk)

    # Calculate aspect ratios for proper sizing
    if original_image is not None:
        # Use actual image tensor aspect ratio
        if original_image.shape[1] >= 3:
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
    # Each VAE latent channel in this chunk gets its own row
    row_heights = [1.0] * total_rows

    fig, axes = plt.subplots(
        total_rows,
        total_cols,
        figsize=(sum(col_widths), 4 * total_rows),
        gridspec_kw={"width_ratios": col_widths, "height_ratios": row_heights},
    )
    plt.subplots_adjust(wspace=0.05, hspace=0.15, left=0.02, right=0.98, top=0.95, bottom=0.05)

    # Ensure axes is always 2D for consistency
    if total_rows == 1:
        axes = [axes]

    # Plot original and reconstructed images in all rows, or hide them
    for row_idx in range(total_rows):
        if row_idx == 0:
            # Plot original image if provided (only in first row)
            if original_image is not None:
                if original_image.shape[1] >= 3:
                    # RGB(A) image; drop alpha/extra channels beyond RGB
                    orig_img = original_image[0, :3].permute(1, 2, 0).cpu().numpy()
                else:
                    # Grayscale
                    orig_img = original_image[0, 0].cpu().numpy()

                orig_img = np.clip(orig_img, 0, 1)

                # Display with correct aspect ratio
                axes[row_idx][0].imshow(orig_img, aspect="auto")
                axes[row_idx][0].set_title("Original Image", fontsize=10)
                axes[row_idx][0].axis("off")

            # Plot reconstructed image if provided (only in first row)
            if reconstructed_image is not None:
                if reconstructed_image.shape[1] >= 3:
                    # RGB(A) image; drop alpha/extra channels beyond RGB
                    recon_img = reconstructed_image[0, :3].permute(1, 2, 0).cpu().numpy()
                else:
                    # Grayscale
                    recon_img = reconstructed_image[0, 0].cpu().numpy()

                recon_img = np.clip(recon_img, 0, 1)

                # Display with correct aspect ratio
                axes[row_idx][2].imshow(recon_img, aspect="auto")
                axes[row_idx][2].set_title("Reconstructed", fontsize=10)
                axes[row_idx][2].axis("off")
        else:
            # Hide original and reconstructed image columns for other rows
            axes[row_idx][0].axis("off")
            axes[row_idx][2].axis("off")

    # Plot individual latent channels and their wavelet coefficients
    for row_idx, channel_idx in enumerate(chunk):
        # Plot individual latent channel
        latent_channel = latent_tensor[0, channel_idx].cpu().numpy()
        latent_norm = (latent_channel - latent_channel.min()) / (latent_channel.max() - latent_channel.min() + 1e-8)
        axes[row_idx][1].imshow(latent_norm, cmap="RdBu_r", aspect="auto")
        axes[row_idx][1].set_title(f"VAE Latent Ch{channel_idx}", fontsize=10)
        axes[row_idx][1].axis("off")

    # Iterate through components (or single band for DWT/SWT)
    for comp_idx, component in enumerate(components):
        # Select coefficients for this component
        if component is not None:
            comp_coeffs = coeffs[component]
        else:
            comp_coeffs = coeffs

        # Visualize each channel in its own row
        for row_idx, channel_idx in enumerate(chunk):
            # Visualize each level and band for this channel
            for level_idx in range(level):
                for band_idx, band in enumerate(bands):
                    # Compute column index (accounting for original, latent, reconstructed columns)
                    col_idx = level_idx * len(bands) + band_idx + 3

                    # Get coefficient data for specific channel
                    coeff_data = comp_coeffs[band][level_idx][0, channel_idx].cpu().numpy()

                    # Normalize for visualization
                    coeff_norm = (coeff_data - coeff_data.min()) / (coeff_data.max() - coeff_data.min() + 1e-8)

                    # Plot
                    title = f"{band.upper()}{level_idx + 1}"
                    if component is not None:
                        title = f"{component.upper()}-{title}"

                    axes[row_idx][col_idx].imshow(coeff_norm, cmap="RdBu_r", aspect="auto")
                    axes[row_idx][col_idx].set_title(title, fontsize=8)
                    axes[row_idx][col_idx].axis("off")

    # Add transform type as suptitle
    vae_line = f"VAE: {vae_model}\n" if vae_model else ""
    chunk_line = f"Channels {chunk[0]}-{chunk[-1]} of {n_channels}\n" if multi_chunk else ""
    plt.suptitle(
        f"{vae_line}VAE Latent {transform_name}\n{chunk_line}{wavelet} Wavelet, {level} Levels",
        fontsize=16,
    )

    # Determine save path for this transform/chunk
    if save_paths:
        chunk_paths = save_paths
        if multi_chunk:
            chunk_suffix = f"_ch{chunk[0]:03d}-{chunk[-1]:03d}"
            chunk_paths = [path.with_stem(f"{path.stem}{chunk_suffix}") for path in save_paths]

        print(f"Saving {transform_name} VAE latent visualization to {len(chunk_paths)} file(s):")
        for save_path in chunk_paths:
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
    parser = argparse.ArgumentParser(description="Visualize VAE Latent Wavelet Transforms")

    # Image input arguments
    parser.add_argument("image", type=str, help="Path to input image")

    # VAE configuration
    parser.add_argument(
        "--vae-model",
        type=str,
        default="stabilityai/sd-vae-ft-mse",
        help="Hugging Face VAE model name or path (default: stabilityai/sd-vae-ft-mse)",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cpu, cuda, auto) (default: auto)")

    # Wavelet transform configuration
    parser.add_argument(
        "--wavelet",
        type=str,
        default="db4",
        help="Wavelet family to use (default: db4)",
    )
    parser.add_argument("--level", type=int, default=2, help="Wavelet decomposition levels (default: 2)")

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

    parser.add_argument("--subfolder", help="Subfolder on hugging face with the VAE")

    parser.add_argument(
        "--max-channels-per-file",
        type=int,
        default=16,
        help="Split visualizations into multiple files with at most this many latent channels (rows) each (default: 16)",
    )

    parser.add_argument(
        "--no-vae-tiling",
        action="store_true",
        help="Disable VAE tiling (tiling can introduce seam artifacts visible in the wavelet coefficients, but uses less VRAM)",
    )

    parser.add_argument(
        "--vae-dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="VAE compute dtype; fp16/bf16 roughly halve VRAM use, useful with --no-vae-tiling (default: fp32)",
    )

    # Additional arguments
    parser.add_argument("--grayscale", action="store_true", help="Convert image to grayscale")
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
    vae_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.vae_dtype]
    vae, processor = load_vae_model(args.vae_model, args.subfolder)
    vae = vae.to(device=device, dtype=vae_dtype)
    vae.eval()
    if not args.no_vae_tiling and hasattr(vae, "enable_tiling"):
        vae.enable_tiling()

    # Store original image shape for aspect ratio calculation
    original_aspect = img.shape[1] / img.shape[0]  # width/height

    # QwenImage-2.1's VAE expects RGBA input (4 channels); pad with an opaque alpha channel
    if AutoencoderKLQwenImage21 is not None and isinstance(vae, AutoencoderKLQwenImage21) and img.shape[-1] == 3:
        alpha = np.full(img.shape[:2] + (1,), 255, dtype=img.dtype)
        img = np.concatenate([img, alpha], axis=-1)

    # Preprocess image using VaeImageProcessor
    img_tensor = preprocess_image_with_vae_processor(img, processor)

    if isinstance(vae, QWEN_TEMPORAL_VAE_CLASSES):
        img_tensor = img_tensor.unsqueeze(2)

    # Encode image to latent space
    latent, reconstructed, img_tensor_display = encode_image_to_latent(vae, processor, img_tensor, device)

    if isinstance(vae, QWEN_TEMPORAL_VAE_CLASSES):
        latent = latent.squeeze(2)
        img_tensor_display = img_tensor_display.squeeze(2)
        reconstructed = reconstructed.squeeze(2)

    # Prepare transforms dictionary based on user selection
    available_transforms = {
        "dwt": ("DWT (Discrete Wavelet Transform)", DiscreteWaveletTransform),
        "swt": ("SWT (Stationary Wavelet Transform)", StationaryWaveletTransform),
        "qwt": ("QWT (Quaternion Wavelet Transform)", QuaternionWaveletTransform),
    }

    print(latent.shape)

    # Filter selected transforms
    selected_transforms = {
        name: (desc, transform) for name, (desc, transform) in available_transforms.items() if name in args.transforms
    }

    # Visualize transforms
    for transform_name, (
        transform_desc,
        transform_class,
    ) in selected_transforms.items():
        # Generate improved filename with vae model, wavelet type, levels, and image hash
        # Format: vae_latent_transforms_{vae_model}_{transform}_{wavelet}_L{levels}_{hash}.{ext}
        vae_model_slug = args.vae_model.replace("/", "-")
        base_filename = f"vae_latent_transforms_{vae_model_slug}_{transform_name}_{args.wavelet}_L{args.level}_{image_hash}"

        # Generate output paths
        output_paths = [Path(output_dir / f"{base_filename}.{fmt}") for fmt in args.output_formats]

        print(img_tensor_display.shape)

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
            vae_model=args.vae_model,
            max_channels_per_file=args.max_channels_per_file,
        )


if __name__ == "__main__":
    main()
