#!/usr/bin/env python3
"""
Script to compute Wavelet Loss between two images.
Supports different wavelet transform types and visualization.
"""

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

from wavelet_loss import WaveletLoss


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


def preprocess_image(img, target_size=None, normalize=True, force_power_of_two=True, min_size=256):
    """
    Preprocess image for wavelet loss calculation.

    Args:
        img (numpy.ndarray): Input image
        target_size (tuple, optional): Resize image to this size
        normalize (bool): Normalize image to [0, 1] range
        force_power_of_two (bool): Ensure image dimensions are powers of two
        min_size (int): Minimum size for image dimensions

    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Resize if target size is specified
    if target_size:
        if cv2 is not None:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        elif Image is not None:
            img = np.array(Image.fromarray(img).resize(target_size, Image.LANCZOS))

    # Ensure power of two dimensions if requested
    if force_power_of_two and len(img.shape) >= 2:
        height, width = img.shape[-2:]

        # Ensure minimum size
        new_height = max(next_power_of_two(height), min_size)
        new_width = max(next_power_of_two(width), min_size)

        # Pad or resize to power of two
        if cv2 is not None:
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        elif Image is not None:
            img = np.array(Image.fromarray(img).resize((new_width, new_height), Image.LANCZOS))

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


def compute_wavelet_loss(img1, img2, args):
    """
    Compute wavelet loss between two images.

    Args:
        img1 (torch.Tensor): First image tensor
        img2 (torch.Tensor): Second image tensor
        args (argparse.Namespace): Command line arguments

    Returns:
        tuple: Total losses and component losses
    """
    device = torch.device(args.device)

    loss_fn = WaveletLoss(wavelet=args.wavelet, level=args.level, transform_type=args.transform_type, device=device)

    # Move images to specified device
    img1 = img1.to(device)
    img2 = img2.to(device)

    # Compute losses
    losses, component_losses = loss_fn(img1, img2)

    return losses, component_losses


def visualize_losses(losses, component_losses, save_path=None):
    """
    Visualize and optionally save loss information.

    Args:
        losses (list): List of losses for each level
        component_losses (dict): Dictionary of component losses
        save_path (str, optional): Path to save the visualization
    """
    plt.figure(figsize=(10, 6))

    # Plot total losses
    plt.subplot(1, 2, 1)
    plt.title("Total Losses per Level")
    plt.bar(range(1, len(losses) + 1), [loss.mean().item() for loss in losses])
    plt.xlabel("Level")
    plt.ylabel("Loss")

    # Plot component losses
    plt.subplot(1, 2, 2)
    plt.title("Component Losses")
    components = list(set(key.split("_")[0] for key in component_losses.keys()))
    loss_values = [
        np.mean([component_losses[key] for key in component_losses.keys() if key.startswith(comp)])
        for comp in components
    ]
    plt.bar(components, loss_values)
    plt.xlabel("Component")
    plt.ylabel("Avg Loss")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def main():
    """
    Main function to parse arguments and compute wavelet loss
    """
    parser = argparse.ArgumentParser(description="Compute Wavelet Loss between two images")

    # Image input arguments
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

    # Additional arguments
    parser.add_argument("--grayscale", action="store_true", help="Convert images to grayscale")
    parser.add_argument("--target-size", type=int, nargs=2, help="Resize images to specified size (width height)")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (default: cuda if available)",
    )
    parser.add_argument("--save-plot", type=str, help="Path to save loss visualization plot")

    args = parser.parse_args()

    # Load and preprocess images
    img1 = load_image(args.image1, grayscale=args.grayscale)
    img2 = load_image(args.image2, grayscale=args.grayscale)

    # Preprocess images
    img1_tensor = preprocess_image(img1, target_size=tuple(args.target_size) if args.target_size else None)
    img2_tensor = preprocess_image(img2, target_size=tuple(args.target_size) if args.target_size else None)

    # Compute wavelet loss
    losses, component_losses = compute_wavelet_loss(img1_tensor, img2_tensor, args)

    # Print losses
    print("Total Losses per Level:")
    for i, loss in enumerate(losses, 1):
        print(f"Level {i}: {loss.mean().item()}")

    print("\nComponent Losses:")
    for key, value in component_losses.items():
        print(f"{key}: {value}")

    # Visualize losses
    visualize_losses(losses, component_losses, args.save_plot)


if __name__ == "__main__":
    main()
