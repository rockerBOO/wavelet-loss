#!/usr/bin/env python3
"""
Script to compute Wavelet Loss between two images using VAE latent representations.
Supports different wavelet transform types and visualization.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from wavelet_loss import WaveletLoss
from utils.image_processing import load_image, generate_image_hash
from utils.vae_utils import load_vae_model, preprocess_image_with_vae_processor, encode_image_to_latent


def compute_wavelet_loss_latent(latent1, latent2, args):
    """
    Compute wavelet loss between two VAE latent representations.

    Args:
        latent1 (torch.Tensor): First latent tensor
        latent2 (torch.Tensor): Second latent tensor
        args (argparse.Namespace): Command line arguments

    Returns:
        tuple: Total losses and component losses
    """
    device = torch.device(args.device)

    loss_fn = WaveletLoss(wavelet=args.wavelet, level=args.level, transform_type=args.transform_type, device=device)

    # Move latents to specified device
    latent1 = latent1.to(device)
    latent2 = latent2.to(device)

    # Compute losses
    losses, component_losses = loss_fn(latent1, latent2)

    return losses, component_losses


def visualize_losses(losses, component_losses, save_path=None, image_hashes=None):
    """
    Visualize and optionally save loss information with detailed breakdown.

    Args:
        losses (list): List of losses for each level
        component_losses (dict): Dictionary of component losses
        save_path (str, optional): Path to save the visualization
        image_hashes (tuple, optional): Tuple of image hashes for title
    """
    fig = plt.figure(figsize=(16, 12))

    # 1. Total losses per level
    plt.subplot(3, 3, 1)
    plt.title("Total Losses per Level", fontweight="bold")
    level_losses = [loss.mean().item() for loss in losses]
    bars = plt.bar(range(1, len(losses) + 1), level_losses, color="steelblue", alpha=0.8)
    plt.xlabel("Wavelet Level")
    plt.ylabel("Loss Value")
    plt.grid(True, alpha=0.3)
    # Add value labels on bars
    for bar, value in zip(bars, level_losses):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(level_losses) * 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 2. Wavelet coefficient types breakdown
    plt.subplot(3, 3, 2)
    plt.title("Loss by Wavelet Coefficient Type", fontweight="bold")
    coeff_types = ["lh", "hl", "hh", "ll"]
    coeff_totals = {}
    for coeff in coeff_types:
        coeff_totals[coeff] = sum(v for k, v in component_losses.items() if k.startswith(coeff))

    bars = plt.bar(coeff_totals.keys(), coeff_totals.values(), color=["orange", "green", "red", "purple"], alpha=0.8)
    plt.xlabel("Coefficient Type")
    plt.ylabel("Total Loss")
    plt.grid(True, alpha=0.3)
    # Add value labels
    for bar, (coeff, value) in zip(bars, coeff_totals.items()):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(coeff_totals.values()) * 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 3. Loss type breakdown (band, gradient, correlation)
    plt.subplot(3, 3, 3)
    plt.title("Loss by Type (Band/Gradient/Correlation)", fontweight="bold")
    loss_types = ["band", "gradient", "correlation"]
    type_totals = {}
    for loss_type in loss_types:
        type_totals[loss_type] = sum(v for k, v in component_losses.items() if f"_{loss_type}_loss" in k)

    bars = plt.bar(type_totals.keys(), type_totals.values(), color=["lightblue", "lightcoral", "lightgreen"], alpha=0.8)
    plt.xlabel("Loss Type")
    plt.ylabel("Total Loss")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    # Add value labels
    for bar, (loss_type, value) in zip(bars, type_totals.items()):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(type_totals.values()) * 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 4. Heatmap of losses by level and coefficient
    plt.subplot(3, 3, 4)
    plt.title("Loss Heatmap: Coefficient × Level", fontweight="bold")

    # Create matrix for heatmap
    levels = list(range(len(losses)))
    coeffs = ["lh", "hl", "hh", "ll"]
    heatmap_data = np.zeros((len(coeffs), len(levels)))

    for i, coeff in enumerate(coeffs):
        for j, level in enumerate(levels):
            # Sum all loss types for this coefficient at this level
            total = sum(v for k, v in component_losses.items() if k.startswith(f"{coeff}{level}_"))
            heatmap_data[i, j] = total

    im = plt.imshow(heatmap_data, cmap="viridis", aspect="auto")
    plt.colorbar(im, shrink=0.8)
    plt.yticks(range(len(coeffs)), coeffs)
    plt.xticks(range(len(levels)), [f"L{i}" for i in levels])
    plt.xlabel("Wavelet Level")
    plt.ylabel("Coefficient Type")

    # Add text annotations
    for i in range(len(coeffs)):
        for j in range(len(levels)):
            plt.text(j, i, f"{heatmap_data[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)

    # 5. Detailed breakdown by level (stacked bar)
    plt.subplot(3, 3, 5)
    plt.title("Detailed Loss Breakdown by Level", fontweight="bold")

    level_data = {}
    for level in range(len(losses)):
        level_data[level] = {}
        for loss_type in loss_types:
            level_data[level][loss_type] = sum(
                v for k, v in component_losses.items() if k.endswith(f"{loss_type}_loss") and f"{level}_" in k
            )

    bottoms = [0] * len(levels)
    colors = ["lightblue", "lightcoral", "lightgreen"]

    for i, loss_type in enumerate(loss_types):
        values = [level_data[level][loss_type] for level in range(len(losses))]
        plt.bar(range(len(losses)), values, bottom=bottoms, label=loss_type.capitalize(), color=colors[i], alpha=0.8)
        bottoms = [b + v for b, v in zip(bottoms, values)]

    plt.xlabel("Wavelet Level")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.xticks(range(len(levels)), [f"L{i}" for i in levels])
    plt.grid(True, alpha=0.3)

    # 6. Top contributing components
    plt.subplot(3, 3, 6)
    plt.title("Top 10 Contributing Components", fontweight="bold")

    # Sort components by loss value
    sorted_components = sorted(component_losses.items(), key=lambda x: x[1], reverse=True)[:10]
    comp_names = [k for k, v in sorted_components]
    comp_values = [v for k, v in sorted_components]

    bars = plt.barh(range(len(comp_names)), comp_values, color="coral", alpha=0.8)
    plt.yticks(range(len(comp_names)), comp_names)
    plt.xlabel("Loss Value")
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, comp_values):
        plt.text(
            bar.get_width() + max(comp_values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            ha="left",
            va="center",
            fontsize=8,
        )

    # 7. Loss distribution histogram
    plt.subplot(3, 3, 7)
    plt.title("Loss Value Distribution", fontweight="bold")
    all_losses = list(component_losses.values())
    plt.hist(all_losses, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
    plt.xlabel("Loss Value")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)

    # 8. Cumulative loss contribution
    plt.subplot(3, 3, 8)
    plt.title("Cumulative Loss Contribution", fontweight="bold")

    sorted_losses = sorted(component_losses.values(), reverse=True)
    cumulative = np.cumsum(sorted_losses)
    percentage = (cumulative / cumulative[-1]) * 100

    plt.plot(range(1, len(sorted_losses) + 1), percentage, "o-", color="green", alpha=0.8)
    plt.xlabel("Component Rank")
    plt.ylabel("Cumulative Contribution (%)")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=80, color="red", linestyle="--", alpha=0.7, label="80% threshold")
    plt.legend()

    # 9. Summary statistics
    plt.subplot(3, 3, 9)
    plt.title("Summary Statistics", fontweight="bold")
    plt.axis("off")

    total_loss = sum(loss.mean().item() for loss in losses)
    max_loss = max(component_losses.values())
    min_loss = min(component_losses.values())
    mean_loss = np.mean(list(component_losses.values()))
    std_loss = np.std(list(component_losses.values()))

    stats_text = f"""
    Total Loss: {total_loss:.6f}
    Max Component: {max_loss:.6f}
    Min Component: {min_loss:.6f}
    Mean Component: {mean_loss:.6f}
    Std Component: {std_loss:.6f}
    
    Num Components: {len(component_losses)}
    Num Levels: {len(losses)}
    """

    plt.text(
        0.1,
        0.9,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontfamily="monospace",
        fontsize=10,
    )

    # Add overall title with image hashes if provided
    if image_hashes:
        plt.suptitle(
            f"VAE Latent Wavelet Loss Analysis\nImage 1: {image_hashes[0]} vs Image 2: {image_hashes[1]}",
            fontsize=16,
            fontweight="bold",
        )
    else:
        plt.suptitle("VAE Latent Wavelet Loss Analysis", fontsize=16, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Loss visualization saved to: {save_path}")
    else:
        plt.show()


def main():
    """
    Main function to parse arguments and compute wavelet loss on VAE latents
    """
    parser = argparse.ArgumentParser(
        description="Compute Wavelet Loss between two images using VAE latent representations"
    )

    # Image input arguments
    parser.add_argument("image1", type=str, help="Path to first input image")
    parser.add_argument("image2", type=str, help="Path to second input image")

    # VAE configuration
    parser.add_argument(
        "--vae-model",
        type=str,
        default="stabilityai/sd-vae-ft-mse",
        help="Hugging Face VAE model name or path (default: stabilityai/sd-vae-ft-mse)",
    )

    # Wavelet loss configuration arguments
    parser.add_argument("--wavelet", type=str, default="db4", help="Wavelet family to use (default: db4)")
    parser.add_argument("--level", type=int, default=3, help="Wavelet decomposition level (default: 3)")
    parser.add_argument(
        "--transform-type",
        type=str,
        default="dwt",
        choices=["dwt", "swt", "qwt"],
        help="Wavelet transform type (default: dwt)",
    )

    # Additional arguments
    parser.add_argument("--grayscale", action="store_true", help="Convert images to grayscale")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cuda, cpu) (default: auto)")
    parser.add_argument("--save-plot", type=str, help="Path to save loss visualization plot")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="wavelet_loss_results",
        help="Directory to save results (default: wavelet_loss_results)",
    )

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    args.device = device

    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess images
    img1 = load_image(args.image1, grayscale=args.grayscale)
    img2 = load_image(args.image2, grayscale=args.grayscale)

    # Generate hashes for the images
    image1_hash = generate_image_hash(args.image1)
    image2_hash = generate_image_hash(args.image2)
    print(f"Image 1 hash: {image1_hash}")
    print(f"Image 2 hash: {image2_hash}")

    # Load VAE model and processor
    print(f"Loading VAE model: {args.vae_model}")
    vae, processor = load_vae_model(args.vae_model)
    vae = vae.to(device)
    vae.eval()

    # Preprocess images using VaeImageProcessor
    img1_tensor = preprocess_image_with_vae_processor(img1, processor)
    img2_tensor = preprocess_image_with_vae_processor(img2, processor)

    # Encode images to latent space
    print("Encoding images to VAE latent space...")
    latent1, _, _ = encode_image_to_latent(vae, processor, img1_tensor, device)
    latent2, _, _ = encode_image_to_latent(vae, processor, img2_tensor, device)

    print(f"Latent 1 shape: {latent1.shape}")
    print(f"Latent 2 shape: {latent2.shape}")

    # Compute wavelet loss on latents
    print("Computing wavelet loss on VAE latents...")
    losses, component_losses = compute_wavelet_loss_latent(latent1, latent2, args)

    # Print losses
    print("\nTotal Losses per Level:")
    for i, loss in enumerate(losses, 1):
        print(f"Level {i}: {loss.mean().item():.6f}")

    print("\nComponent Losses:")
    for key, value in component_losses.items():
        print(f"{key}: {value:.6f}")

    # Calculate total loss
    total_loss = sum(loss.mean().item() for loss in losses)
    print(f"\nTotal Wavelet Loss: {total_loss:.6f}")

    # Generate output filename if not provided
    if not args.save_plot:
        filename = f"wavelet_loss_{args.transform_type}_{args.wavelet}_L{args.level}_{image1_hash}_{image2_hash}.png"
        args.save_plot = output_dir / filename

    # Visualize losses
    visualize_losses(losses, component_losses, args.save_plot, (image1_hash, image2_hash))

    # Save loss results to text file
    results_filename = (
        f"wavelet_loss_results_{args.transform_type}_{args.wavelet}_L{args.level}_{image1_hash}_{image2_hash}.txt"
    )
    results_path = output_dir / results_filename

    with open(results_path, "w") as f:
        f.write("Wavelet Loss Results\n")
        f.write("==================\n\n")
        f.write("Images:\n")
        f.write(f"  Image 1: {args.image1} (hash: {image1_hash})\n")
        f.write(f"  Image 2: {args.image2} (hash: {image2_hash})\n\n")
        f.write(f"VAE Model: {args.vae_model}\n")
        f.write(f"Wavelet: {args.wavelet}\n")
        f.write(f"Transform Type: {args.transform_type}\n")
        f.write(f"Levels: {args.level}\n")
        f.write(f"Device: {device}\n\n")
        f.write("Latent Shapes:\n")
        f.write(f"  Latent 1: {latent1.shape}\n")
        f.write(f"  Latent 2: {latent2.shape}\n\n")
        f.write("Total Losses per Level:\n")
        for i, loss in enumerate(losses, 1):
            f.write(f"  Level {i}: {loss.mean().item():.6f}\n")
        f.write("\nComponent Losses:\n")
        for key, value in component_losses.items():
            f.write(f"  {key}: {value:.6f}\n")
        f.write(f"\nTotal Wavelet Loss: {total_loss:.6f}\n")

    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
