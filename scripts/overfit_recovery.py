"""Overfit / recovery example for WaveletLoss.

Demonstrates that minimizing the wavelet loss is a valid, differentiable training
objective: a randomly initialized tensor is optimized to match a fixed target
using ONLY the wavelet loss. We track the loss and an independent pixel-MSE
(pred vs target) to confirm the loss actually drives pred -> target, plot the
recovered signal against the target, and show the per-band loss breakdown.

Usage:
    uv run python scripts/overfit_recovery.py
    uv run python scripts/overfit_recovery.py --transform-type swt --steps 500
    uv run python scripts/overfit_recovery.py --backend custom --out outputs/recovery.png
"""

import argparse

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import torch

from wavelet_loss import WaveletLoss


def make_target(channels: int, size: int, device: torch.device) -> torch.Tensor:
    """Smooth low-frequency base + fine high-frequency texture so that
    frequency-aware behavior is exercised."""
    yy, xx = torch.meshgrid(
        torch.linspace(0, 6, size, device=device),
        torch.linspace(0, 6, size, device=device),
        indexing="ij",
    )
    base = torch.sin(xx) * torch.cos(yy)
    texture = 0.3 * torch.sin(20 * xx) * torch.sin(20 * yy)
    img = (base + texture).view(1, 1, size, size).repeat(1, channels, 1, 1)
    return img.contiguous()


def per_band_losses(loss_fn: WaveletLoss, pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Mean weighted loss per (level, band), using the list form (reduce=False)."""
    with torch.no_grad():
        losses, _ = loss_fn(pred, target, reduce=False)
    bands = ["ll", "lh", "hl", "hh"]
    out: dict[str, float] = {}
    for idx, value in enumerate(losses):
        level = idx // len(bands) + 1
        band = bands[idx % len(bands)]
        out[f"{band}{level}"] = float(value.mean())
    return out


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    target = make_target(args.channels, args.size, device)
    pred = torch.randn_like(target, requires_grad=True)

    loss_fn = WaveletLoss(
        wavelet=args.wavelet,
        level=args.level,
        transform_type=args.transform_type,
        backend=args.backend,
        device=device,
    )
    opt = torch.optim.Adam([pred], lr=args.lr)

    loss_hist: list[float] = []
    mse_hist: list[float] = []
    for step in range(args.steps):
        opt.zero_grad()
        loss, _ = loss_fn(pred, target)  # scalar (reduce=True)
        loss.backward()
        opt.step()
        loss_hist.append(loss.item())
        mse_hist.append(torch.mean((pred.detach() - target) ** 2).item())
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            print(f"step {step:4d}  loss={loss.item():.6f}  pixel-MSE={mse_hist[-1]:.3e}")

    bands = per_band_losses(loss_fn, pred, target)
    print("\nFinal per-band loss (weighted):")
    for name, value in bands.items():
        print(f"  {name}: {value:.3e}")
    print(f"\nloss {loss_hist[0]:.4f} -> {loss_hist[-1]:.6f} | final pixel-MSE={mse_hist[-1]:.3e}")

    # --- plots ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    title = f"WaveletLoss overfit: {args.transform_type}/{args.backend} {args.wavelet} L{args.level}"
    fig.suptitle(title)

    ax = axes[0, 0]
    ax.plot(loss_hist, label="wavelet loss")
    ax.plot(mse_hist, label="pixel-MSE(pred,target)")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("value (log)")
    ax.set_title("Convergence: minimizing wavelet loss drives pred -> target")
    ax.legend()

    ch = 0
    axes[0, 1].imshow(target[0, ch].detach().cpu(), cmap="viridis")
    axes[0, 1].set_title("target (channel 0)")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(pred[0, ch].detach().cpu(), cmap="viridis")
    axes[1, 0].set_title("recovered (channel 0)")
    axes[1, 0].axis("off")

    ax = axes[1, 1]
    ax.bar(list(bands.keys()), list(bands.values()))
    ax.set_title("Final per-band loss (weighted)")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"\nSaved figure to {args.out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wavelet", default="db4")
    p.add_argument("--level", type=int, default=3)
    p.add_argument("--transform-type", default="dwt", choices=["dwt", "swt", "qwt"])
    p.add_argument("--backend", default="pytorch_wavelets", choices=["pytorch_wavelets", "custom"])
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--channels", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="outputs/overfit_recovery.png")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
