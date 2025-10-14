#!/usr/bin/env python3
"""
Script to investigate wavelet loss landscape by exploring different
parameters and visualizing loss variations.
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from wavelet_loss import WaveletLoss


def generate_interpolated_images(img1, img2, num_steps=10):
    """
    Generate a series of interpolated images between two input images.

    Args:
        img1 (torch.Tensor): First input image
        img2 (torch.Tensor): Second input image
        num_steps (int): Number of interpolation steps

    Returns:
        torch.Tensor: Interpolated images tensor
    """
    # Ensure images are on the same device and have the same shape
    img1 = img1.to(img2.device)

    # Create interpolation weights
    alphas = torch.linspace(0, 1, num_steps)

    # Create interpolated images
    interpolated_images = torch.stack([img1 * (1 - alpha) + img2 * alpha for alpha in alphas])

    return interpolated_images


def compute_loss_landscape(images, wavelet_params=None):
    """
    Compute wavelet loss across a series of interpolated images.

    Args:
        images (torch.Tensor): Series of interpolated images
        wavelet_params (dict): Parameters for WaveletLoss

    Returns:
        list: Total losses for each interpolation step
    """
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Default wavelet loss parameters
    default_params = {"wavelet": "db4", "level": 3, "transform_type": "dwt", "device": device}

    # Update with provided parameters
    if wavelet_params:
        default_params.update(wavelet_params)

    # Initialize loss function
    loss_fn = WaveletLoss(**default_params)

    # Move images to the same device
    images = images.to(device)

    # Compute pairwise losses
    losses = []
    base_image = images[0]
    for img in images[1:]:
        total_loss, _ = loss_fn(base_image, img)
        losses.append(total_loss[0].mean().item())  # Take mean of total loss

    return losses


def visualize_loss_landscape(losses, save_path=None):
    """
    Visualize the loss landscape.

    Args:
        losses (list): Losses across interpolation steps
        save_path (str, optional): Path to save visualization
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.title("Wavelet Loss Landscape")
    plt.xlabel("Interpolation Step")
    plt.ylabel("Wavelet Loss")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def main():
    """
    Main function to compute and visualize wavelet loss landscape.
    """
    parser = argparse.ArgumentParser(description="Investigate Wavelet Loss Landscape")
    parser.add_argument("image1", type=str, help="Path to first input image")
    parser.add_argument("image2", type=str, help="Path to second input image")

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
    parser.add_argument("--num-steps", type=int, default=10, help="Number of interpolation steps (default: 10)")
    parser.add_argument("--save-plot", type=str, help="Path to save loss landscape plot")

    args = parser.parse_args()

    # Reuse image loading and preprocessing from wavelet_loss_image.py
    from wavelet_loss_image import load_image, preprocess_image

    # Load and preprocess images
    img1 = load_image(args.image1)
    img2 = load_image(args.image2)

    # Print image details
    print(f"Image 1 shape: {img1.shape}, dtype: {img1.dtype}")
    print(f"Image 2 shape: {img2.shape}, dtype: {img2.dtype}")

    # Preprocess images
    img1_tensor = preprocess_image(img1)
    img2_tensor = preprocess_image(img2)

    print(f"Preprocessed Image 1 shape: {img1_tensor.shape}, dtype: {img1_tensor.dtype}")
    print(f"Preprocessed Image 2 shape: {img2_tensor.shape}, dtype: {img2_tensor.dtype}")

    # Generate interpolated images
    interpolated_images = generate_interpolated_images(img1_tensor, img2_tensor, num_steps=args.num_steps)

    # Compute loss landscape
    wavelet_params = {"wavelet": args.wavelet, "level": args.level, "transform_type": args.transform_type}
    losses = compute_loss_landscape(interpolated_images, wavelet_params)

    # Visualize loss landscape
    visualize_loss_landscape(losses, args.save_plot)

    # Print losses
    print("Wavelet Loss at Interpolation Steps:")
    for i, loss in enumerate(losses, 1):
        print(f"Step {i}: {loss}")


if __name__ == "__main__":
    main()
