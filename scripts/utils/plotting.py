#!/usr/bin/env python3
"""
Common plotting utilities for wavelet loss visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def set_plot_style(style="seaborn", font_scale=1.1):
    """
    Set global plot styling for consistent visualization.

    Args:
        style (str): Matplotlib/Seaborn style to use
        font_scale (float): Scaling factor for font sizes
    """
    plt.style.use(style)
    sns.set(font_scale=font_scale)


def normalize_coefficients(coeff_data, percentile_clip=True):
    """
    Normalize coefficient data for visualization with optional percentile clipping.

    Args:
        coeff_data (np.ndarray): Input coefficient data
        percentile_clip (bool): Whether to use percentile-based normalization

    Returns:
        np.ndarray: Normalized coefficient data
    """
    if percentile_clip:
        # Use percentile-based normalization to reduce impact of extreme values
        low = np.percentile(coeff_data, 1)
        high = np.percentile(coeff_data, 99)
        coeff_norm = np.clip((coeff_data - low) / (high - low + 1e-8), 0, 1)
    else:
        # Standard min-max normalization
        coeff_norm = (coeff_data - coeff_data.min()) / (coeff_data.max() - coeff_data.min() + 1e-8)
    return coeff_norm


def plot_loss_landscape(
    losses,
    title="Wavelet Loss Landscape",
    xlabel="Interpolation Step",
    ylabel="Wavelet Loss",
    save_path=None,
    show_error_bars=False,
):
    """
    Advanced visualization of loss landscape with optional error bars.

    Args:
        losses (list or np.ndarray): Loss values
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        save_path (str, optional): Path to save figure
        show_error_bars (bool): Whether to estimate and show error bars
    """
    plt.figure(figsize=(12, 7))

    # Use seaborn for more aesthetic plot
    with sns.axes_style("whitegrid"):
        x = range(1, len(losses) + 1)

        if show_error_bars and len(losses) > 2:
            # Compute rolling window standard deviation as error bars
            window_size = min(3, len(losses) // 2)
            errors = [
                np.std(losses[max(0, i - window_size) : min(len(losses), i + window_size)]) for i in range(len(losses))
            ]
            plt.errorbar(
                x,
                losses,
                yerr=errors,
                fmt="o-",
                capsize=5,
                ecolor="red",
                markerfacecolor="blue",
                markeredgecolor="black",
            )
        else:
            plt.plot(x, losses, marker="o", linestyle="-", linewidth=2, markersize=8)

        plt.title(title, fontsize=15)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_spatial_loss(loss_tensor, title="Spatial Loss", cmap="coolwarm", percentile_clip=True, save_path=None):
    """
    Visualize spatial loss with advanced normalization.

    Args:
        loss_tensor (torch.Tensor): Loss tensor to visualize
        title (str): Plot title
        cmap (str): Colormap to use
        percentile_clip (bool): Whether to use percentile-based normalization
        save_path (str, optional): Path to save figure
    """
    # Handle various tensor shapes
    if isinstance(loss_tensor, (float, int)):
        print(f"{title}: {loss_tensor}")
        return

    loss_tensor = torch.as_tensor(loss_tensor).detach()

    # Convert to numpy and handle multi-channel case
    loss_np = loss_tensor.squeeze().cpu().numpy()
    if loss_np.ndim == 3:
        loss_np = loss_np.mean(axis=0)  # Average across channels

    # Enhanced normalization
    loss_norm = normalize_coefficients(loss_np, percentile_clip)

    plt.figure(figsize=(12, 8))
    im = plt.imshow(loss_norm, cmap=cmap, interpolation="nearest")
    plt.colorbar(im, label="Normalized Loss")
    plt.title(title)
    plt.axis("off")

    if save_path:
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()
    else:
        plt.show()


def main():
    """
    Example usage and testing of plotting utilities.
    """
    # Example loss landscape
    example_losses = [0.1, 0.3, 0.6, 0.8, 1.0, 0.9, 0.7, 0.5, 0.3, 0.2]
    plot_loss_landscape(example_losses, show_error_bars=True)


if __name__ == "__main__":
    main()
