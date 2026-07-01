"""Auxiliary-loss A/B: does adding WaveletLoss improve high-frequency fidelity?

A capacity-limited autoencoder (tight spatial bottleneck -- the VAE-like regime
this library targets) reconstructs synthetic textured images. With limited
capacity, a plain MSE loss spends it on high-energy LOW frequencies and drops
texture; a detail-band-weighted wavelet term reallocates toward high frequencies.

The same model is trained twice from an IDENTICAL initialization, same
data/steps/optimizer:

    A) loss = MSE
    B) loss = MSE + lambda * WaveletLoss   (detail bands weighted; see run())

and evaluated on a held-out set with two metrics:
    - overall MSE / PSNR (pixel fidelity)
    - high-frequency error via a fixed Laplacian high-pass filter -- an INDEPENDENT
      metric (NOT the wavelet loss), so model B isn't graded on its own objective.

Expected honest trade-off: B gives up a little overall MSE for better
high-frequency/texture fidelity. (Note: with an L2 base loss and full model
capacity the wavelet-L2 term is largely redundant by Parseval -- the bottleneck
is what creates the trade-off.)

Usage:
    uv run python scripts/auxiliary_loss_ab.py
    uv run python scripts/auxiliary_loss_ab.py --lam 1.0 --steps 800
"""

import argparse
import copy

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from wavelet_loss import WaveletLoss


def make_batch(n: int, channels: int, size: int, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    """Random superpositions of low- and high-frequency sinusoids (varied texture)."""
    yy, xx = torch.meshgrid(
        torch.linspace(0, 2 * torch.pi, size, device=device),
        torch.linspace(0, 2 * torch.pi, size, device=device),
        indexing="ij",
    )
    imgs = torch.zeros(n, channels, size, size, device=device)
    for i in range(n):
        for c in range(channels):
            acc = torch.zeros(size, size, device=device)
            for _ in range(3):  # low-frequency structure
                fx, fy = torch.randint(1, 4, (2,), generator=generator).tolist()
                ph = torch.rand(1, generator=generator).item() * 6.28
                acc = acc + torch.sin(fx * xx + fy * yy + ph)
            for _ in range(3):  # high-frequency texture
                fx, fy = torch.randint(8, 16, (2,), generator=generator).tolist()
                ph = torch.rand(1, generator=generator).item() * 6.28
                acc = acc + 0.4 * torch.sin(fx * xx + fy * yy + ph)
            imgs[i, c] = acc
    # normalize per-image to ~unit std
    imgs = (imgs - imgs.mean(dim=(2, 3), keepdim=True)) / (imgs.std(dim=(2, 3), keepdim=True) + 1e-6)
    return imgs


def highpass(x: torch.Tensor) -> torch.Tensor:
    """Fixed Laplacian high-pass -- the independent HF metric (not the wavelet loss)."""
    k = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], device=x.device)
    k = k.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), k, groups=x.shape[1])


class BottleneckAE(nn.Module):
    """Lossy autoencoder with a tight spatial bottleneck (VAE-like capacity limit)."""

    def __init__(self, channels: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(channels, 16, 4, stride=2, padding=1),  # /2
            nn.ReLU(),
            nn.Conv2d(16, 4, 4, stride=2, padding=1),  # /4 bottleneck (4 ch @ 16x16)
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(4, 16, 4, stride=2, padding=1),  # x2
            nn.ReLU(),
            nn.ConvTranspose2d(16, channels, 4, stride=2, padding=1),  # x2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))


def train(model, data, lam, wavelet_loss, steps, lr, device):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for step in range(steps):
        target = data[torch.randint(0, data.shape[0], (16,), device=device)]
        opt.zero_grad()
        out = model(target)  # reconstruction through the bottleneck
        mse = F.mse_loss(out, target)
        loss = mse if lam == 0 else mse + lam * wavelet_loss(out, target)[0]
        loss.backward()
        opt.step()
        hist.append(mse.item())
    return hist


@torch.no_grad()
def evaluate(model, target, device):
    out = model(target)
    mse = F.mse_loss(out, target).item()
    psnr = 10.0 * torch.log10(1.0 / (F.mse_loss(out, target) + 1e-12)).item() if mse > 0 else float("inf")
    hf_err = F.l1_loss(highpass(out), highpass(target)).item()
    return out, mse, psnr, hf_err


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    gen = torch.Generator().manual_seed(args.seed)
    train_data = make_batch(64, args.channels, args.size, device, gen)
    test_data = make_batch(16, args.channels, args.size, device, gen)

    # Configure the loss to emphasize DETAIL (high-frequency) bands -- this is the
    # knob that makes it a high-frequency-aware auxiliary term. The library default
    # weights instead emphasize the low-frequency approximation (tuned for a
    # different, super-resolution objective), which does NOT help HF fidelity here.
    wavelet_loss = WaveletLoss(
        wavelet=args.wavelet,
        level=args.level,
        transform_type="dwt",
        device=device,
        band_weights={"ll": 0.0, "lh": 1.0, "hl": 1.0, "hh": 1.0},
        ll_level_threshold=None,
    )

    torch.manual_seed(args.seed)
    base = BottleneckAE(args.channels).to(device)
    model_a = copy.deepcopy(base)  # identical init
    model_b = copy.deepcopy(base)

    print(f"Training A (MSE only) and B (MSE + {args.lam} * wavelet), identical init...")
    hist_a = train(model_a, train_data, 0.0, wavelet_loss, args.steps, args.lr, device)
    hist_b = train(model_b, train_data, args.lam, wavelet_loss, args.steps, args.lr, device)

    out_a, mse_a, psnr_a, hf_a = evaluate(model_a, test_data, device)
    out_b, mse_b, psnr_b, hf_b = evaluate(model_b, test_data, device)

    print("\nHeld-out evaluation:")
    print(f"  A (MSE only)      : pixel-MSE={mse_a:.4e}  PSNR={psnr_a:5.2f}dB  HF-error(Laplacian)={hf_a:.4e}")
    print(f"  B (MSE + wavelet) : pixel-MSE={mse_b:.4e}  PSNR={psnr_b:5.2f}dB  HF-error(Laplacian)={hf_b:.4e}")
    hf_impr = 100.0 * (hf_a - hf_b) / hf_a
    print(f"\n  High-frequency error change (B vs A): {hf_impr:+.1f}%  ({'better' if hf_b < hf_a else 'worse'})")

    # --- plots ---
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(f"Auxiliary wavelet loss A/B (lambda={args.lam}, {args.wavelet} L{args.level})")
    axes[0, 0].plot(hist_a, label="A: MSE only")
    axes[0, 0].plot(hist_b, label="B: MSE + wavelet")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("training pixel-MSE")
    axes[0, 0].set_xlabel("step")
    axes[0, 0].legend()

    metrics = ["pixel-MSE", "HF-error"]
    axes[0, 1].bar([0, 1], [mse_a, hf_a], width=0.35, label="A")
    axes[0, 1].bar([0.4, 1.4], [mse_b, hf_b], width=0.35, label="B")
    axes[0, 1].set_xticks([0.2, 1.2])
    axes[0, 1].set_xticklabels(metrics)
    axes[0, 1].set_title("held-out metrics (lower=better)")
    axes[0, 1].legend()

    ch = 0
    panels = [
        (axes[0, 2], highpass(test_data)[0, ch], "target high-freq (to preserve)"),
        (axes[1, 0], out_a[0, ch], f"A: MSE only (HF={hf_a:.3e})"),
        (axes[1, 1], out_b[0, ch], f"B: MSE+wavelet (HF={hf_b:.3e})"),
        (axes[1, 2], test_data[0, ch], "target"),
    ]
    for ax, img, title in panels:
        ax.imshow(img.detach().cpu(), cmap="viridis")
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"\nSaved figure to {args.out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wavelet", default="db4")
    p.add_argument("--level", type=int, default=3)
    p.add_argument("--lam", type=float, default=2.0, help="weight on the wavelet loss term")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="outputs/auxiliary_loss_ab.png")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
