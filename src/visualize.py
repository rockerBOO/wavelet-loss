import torch
import numpy as np
import matplotlib.pyplot as plt


def explore_wavelets(coeffs, coeffs_name="Coefficients"):
    """Interactive exploration of wavelet coefficients"""

    bands = list(coeffs.keys())
    levels = list(range(len(coeffs[bands[0]])))
    batch_size, n_channels = coeffs[bands[0]][0].shape[:2]

    print(f"\n=== {coeffs_name} Structure ===")
    print(f"Bands: {bands}")
    print(f"Levels: {levels}")
    print(f"Batch size: {batch_size}")
    print(f"Channels: {n_channels}")

    for band in bands:
        for level in levels:
            shape = coeffs[band][level].shape
            sparsity = (torch.abs(coeffs[band][level]) < 0.01).float().mean().item()
            magnitude = torch.abs(coeffs[band][level]).mean().item()

            print(f"{band.upper()}{level + 1}: shape={shape}, sparsity={sparsity:.1%}, avg_magnitude={magnitude:.4f}")


# During training, visualize specific coefficients
def visualize_training_wavelets(pred_coeffs, target_coeffs, step):
    """Call this during training to save wavelet visualizations"""

    # 1. Visualize predicted coefficients for LH band, level 0
    fig1 = visualize_wavelet_coefficients(
        pred_coeffs,
        band="lh",
        level=0,
        batch_idx=0,
        title_prefix="Predicted",
        save_path=f"wavelets/pred_lh1_step_{step}.png",
    )
    plt.close(fig1)

    # 2. Compare predicted vs target
    fig2 = compare_wavelet_coefficients(
        pred_coeffs,
        target_coeffs,
        band="hl",
        level=1,
        batch_idx=0,
        channel_idx=0,
        save_path=f"wavelets/comparison_hl2_step_{step}.png",
    )
    plt.close(fig2)

    # 3. Overview of all bands
    fig3 = visualize_all_bands_levels(
        pred_coeffs,
        title_prefix="Predicted",
        batch_idx=0,
        channel_idx=0,
        save_path=f"wavelets/overview_step_{step}.png",
    )
    plt.close(fig3)


def visualize_all_bands_levels(coeffs, title_prefix="", batch_idx=0, channel_idx=0, save_path=None):
    """
    Show all wavelet bands and levels in one overview plot
    """

    bands = ["lh", "hl", "hh"]
    n_levels = len(coeffs["lh"])  # Assuming all bands have same levels

    fig, axes = plt.subplots(len(bands), n_levels, figsize=(4 * n_levels, 3 * len(bands)))

    if n_levels == 1:
        axes = axes.reshape(-1, 1)

    for band_idx, band in enumerate(bands):
        for level in range(n_levels):
            ax = axes[band_idx, level]

            # Get coefficient data
            coeff_data = coeffs[band][level][batch_idx, channel_idx].detach().cpu().numpy()

            # Plot
            im = ax.imshow(coeff_data, cmap="RdBu_r", aspect="auto")
            ax.set_title(f"{band.upper()}{level + 1}")

            # Add colorbar for better interpretation
            plt.colorbar(im, ax=ax, shrink=0.6)

            # Add sparsity info
            sparsity = (np.abs(coeff_data) < 0.01).mean()
            ax.text(
                0.02,
                0.02,
                f"Sparse: {sparsity:.1%}",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                fontsize=8,
            )

    fig.suptitle(
        f"{title_prefix} All Wavelet Bands - Sample {batch_idx}, Channel {channel_idx}",
        fontsize=14,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def compare_wavelet_coefficients(pred_coeffs, target_coeffs, band, level, batch_idx=0, channel_idx=0, save_path=None):
    """
    Side-by-side comparison of predicted vs target coefficients
    """

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Get data
    pred_data = pred_coeffs[band][level][batch_idx, channel_idx].detach().cpu().numpy()
    target_data = target_coeffs[band][level][batch_idx, channel_idx].detach().cpu().numpy()

    # Calculate difference
    diff_data = pred_data - target_data

    # Determine common color scale
    vmin = min(pred_data.min(), target_data.min())
    vmax = max(pred_data.max(), target_data.max())

    # Plot predicted
    im1 = ax1.imshow(pred_data, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax1.set_title(f"Predicted {band.upper()}{level + 1} Ch{channel_idx}")
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # Plot target
    im2 = ax2.imshow(target_data, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax2.set_title(f"Target {band.upper()}{level + 1} Ch{channel_idx}")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # Plot difference
    im3 = ax3.imshow(
        diff_data,
        cmap="RdBu_r",
        vmin=-np.abs(diff_data).max(),
        vmax=np.abs(diff_data).max(),
    )
    ax3.set_title("Difference (Pred - Target)")
    plt.colorbar(im3, ax=ax3, shrink=0.8)

    # Add correlation info
    correlation = np.corrcoef(pred_data.flatten(), target_data.flatten())[0, 1]
    mse = np.mean((pred_data - target_data) ** 2)

    fig.suptitle(
        f"Wavelet Comparison - Correlation: {correlation:.3f}, MSE: {mse:.6f}",
        fontsize=14,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_wavelet_coefficients(
    coeffs,
    band,
    level,
    batch_idx=0,
    channel_idx=None,
    title_prefix="",
    save_path=None,
    figsize=(15, 10),
):
    """
    Visualize wavelet coefficients for a specific band and level

    Args:
        coeffs: dict with structure coeffs[band][level] -> [batch, channel, h, w]
        band: str, one of ['lh', 'hl', 'hh']
        level: int, wavelet decomposition level (0-indexed)
        batch_idx: int, which sample in batch to visualize
        channel_idx: int or None, specific channel to show (None = all channels)
        title_prefix: str, prefix for plot titles (e.g., "Predicted" or "Target")
        save_path: str or None, path to save the plot
        figsize: tuple, figure size

    Returns:
        fig: matplotlib figure object
    """

    # Extract the specific coefficients
    coeff_tensor = coeffs[band][level]  # [batch, channel, h, w]

    # Get single sample
    sample_coeffs = coeff_tensor[batch_idx]  # [channel, h, w]

    batch_size, num_channels, height, width = coeff_tensor.shape

    # Determine which channels to visualize
    if channel_idx is not None:
        channels_to_show = [channel_idx]
        sample_coeffs = sample_coeffs[channel_idx : channel_idx + 1]
    else:
        channels_to_show = list(range(num_channels))

    # Create subplot layout
    n_channels = len(channels_to_show)
    cols = min(4, n_channels)  # Max 4 columns
    rows = (n_channels + cols - 1) // cols  # Ceiling division

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Handle single subplot case
    if n_channels == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if n_channels == 1 else axes
    else:
        axes = axes.flatten()

    # Plot each channel
    for i, ch_idx in enumerate(channels_to_show):
        if i >= len(axes):
            break

        ax = axes[i]

        # Get coefficient data for this channel
        coeff_data = sample_coeffs[i].detach().cpu().numpy()

        # Create visualization
        im = ax.imshow(coeff_data, cmap="RdBu_r", aspect="auto")

        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Set title
        ax.set_title(
            f"{title_prefix} {band.upper()}{level + 1} Ch{ch_idx}\n"
            f"Range: [{coeff_data.min():.3f}, {coeff_data.max():.3f}]"
        )

        # Add statistics text
        stats_text = (
            f"Mean: {coeff_data.mean():.3f}\n"
            f"Std: {coeff_data.std():.3f}\n"
            f"Non-zero: {(np.abs(coeff_data) > 0.01).mean():.1%}"
        )

        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=8,
        )

    # Hide unused subplots
    for i in range(n_channels, len(axes)):
        axes[i].axis("off")

    # Add main title
    fig.suptitle(
        f"{title_prefix} Wavelet Coefficients - {band.upper()} Level {level + 1}\n"
        f"Sample {batch_idx}, Shape: {coeff_tensor.shape}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_qwt_results(qwt_transform, lr_image, pred_latent, target_latent, filename):
    """
    Visualize QWT decomposition of input, prediction, and target.

    visualize_qwt_results(
        model.qwt_loss.transform,
        lr_images[0:1],
        pred_latents[0:1],
        target_latents[0:1],
        f"qwt_vis_epoch{epoch}_batch{batch_idx}.png"
    )

    Args:
        qwt_transform: Quaternion Wavelet Transform instance
        lr_image: Low-resolution input image
        pred_latent: Predicted latent
        target_latent: Target latent
        filename: Output filename
    """

    # Apply QWT
    lr_qwt = qwt_transform.decompose(lr_image, level=2)
    pred_qwt = qwt_transform.decompose(pred_latent, level=2)
    target_qwt = qwt_transform.decompose(target_latent, level=2)

    # Set up figure
    fig, axes = plt.subplots(4, 9, figsize=(27, 12))

    # First, show original images/latents
    axes[0, 0].imshow(lr_image[0].permute(1, 2, 0).detach().cpu().numpy())
    axes[0, 0].set_title("LR Input")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(pred_latent[0].permute(1, 2, 0).detach().cpu().numpy())
    axes[0, 1].set_title("Pred Latent")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(target_latent[0].permute(1, 2, 0).detach().cpu().numpy())
    axes[0, 2].set_title("Target Latent")
    axes[0, 2].axis("off")

    # Keep track of current column
    col = 3

    # For each component (r, i, j, k)
    for i, component in enumerate(["r", "i", "j", "k"]):
        # For first level only, display LL band
        if i == 0:  # Only for real component to save space
            # First level LL band
            lr_ll = lr_qwt[component]["ll"][0][0, 0].detach().cpu().numpy()
            pred_ll = pred_qwt[component]["ll"][0][0, 0].detach().cpu().numpy()
            target_ll = target_qwt[component]["ll"][0][0, 0].detach().cpu().numpy()

            # Normalize for visualization
            lr_ll = (lr_ll - lr_ll.min()) / (lr_ll.max() - lr_ll.min() + 1e-8)
            pred_ll = (pred_ll - pred_ll.min()) / (pred_ll.max() - pred_ll.min() + 1e-8)
            target_ll = (target_ll - target_ll.min()) / (target_ll.max() - target_ll.min() + 1e-8)

            axes[0, col].imshow(lr_ll, cmap="viridis")
            axes[0, col].set_title(f"LR {component}_LL")
            axes[0, col].axis("off")

            axes[0, col + 1].imshow(pred_ll, cmap="viridis")
            axes[0, col + 1].set_title(f"Pred {component}_LL")
            axes[0, col + 1].axis("off")

            axes[0, col + 2].imshow(target_ll, cmap="viridis")
            axes[0, col + 2].set_title(f"Target {component}_LL")
            axes[0, col + 2].axis("off")

            col = 0  # Reset column for next row

        # For each component, show detail bands
        for band_idx, band in enumerate(["lh", "hl", "hh"]):
            # Get band coefficients
            lr_band = lr_qwt[component][band][0][0, 0].detach().cpu().numpy()
            pred_band = pred_qwt[component][band][0][0, 0].detach().cpu().numpy()
            target_band = target_qwt[component][band][0][0, 0].detach().cpu().numpy()

            # Normalize for visualization
            lr_band = (lr_band - lr_band.min()) / (lr_band.max() - lr_band.min() + 1e-8)
            pred_band = (pred_band - pred_band.min()) / (pred_band.max() - pred_band.min() + 1e-8)
            target_band = (target_band - target_band.min()) / (target_band.max() - target_band.min() + 1e-8)

            # Plot in the corresponding row
            row = i + 1 if i > 0 else i + 1 + band_idx

            axes[row, col].imshow(lr_band, cmap="viridis")
            axes[row, col].set_title(f"LR {component}_{band}")
            axes[row, col].axis("off")
            axes[row, col + 1].imshow(pred_band, cmap="viridis")
            axes[row, col + 1].set_title(f"Pred {component}_{band}")
            axes[row, col + 1].axis("off")

            axes[row, col + 2].imshow(target_band, cmap="viridis")
            axes[row, col + 2].set_title(f"Target {component}_{band}")
            axes[row, col + 2].axis("off")

            col += 3

            # Reset column for next row
            if col >= 9:
                col = 0

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
