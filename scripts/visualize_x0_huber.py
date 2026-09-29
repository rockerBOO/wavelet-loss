"""Figures for docs/x0-huber.md.

1. huber_thresholds.png: SNR-Huber vs x0-Huber across timesteps. Shows where each threshold sits in
   velocity units against the velocity residual a fixed x0 error produces, and how much gradient each
   loss gives that error relative to L2.
2. local_energy_window.png: the local-energy mottle check. Shows what the 5x5 window does to the
   per-coefficient HF energy, the optimal estimate's expected energy G*S, and the mottle ratio M on
   an ideal vs a mottled estimate.

All curves and maps come from the library functions (``snr_aware_huber_loss``,
``x0_threshold_huber_loss``, ``local_energy``, ``optimal_local_energy``, ``flat_mask``).

    uv run python scripts/visualize_x0_huber.py [--latent latents.safetensors --key bojack] [--out docs/images]

Without ``--latent`` a synthetic latent is used (flat background with a textured blob). The committed
figures use a real FLUX.2 latent ([1, 128, 36, 64], BN-normalized, from a 576x1024 image).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from wavelet_loss import snr_aware_huber_loss, x0_threshold_huber_loss
from wavelet_loss.local_energy import HF_BANDS, flat_mask, local_energy, optimal_local_energy
from wavelet_transform import StationaryWaveletTransform

# Reference palette (validated: CVD + normal vision). Same slots as flux2-denoiser's loss plots.
SNR, X0 = "#eb6834", "#2a78d6"
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq_blue", [SURFACE, "#cde2fb", "#6da7ec", "#2a78d6", "#104281"])
DIV = LinearSegmentedColormap.from_list("div", ["#104281", "#6da7ec", "#f0efec", "#ef8a89", "#a83232"])


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def grad_ratio(loss_fn, r: torch.Tensor, t: torch.Tensor) -> np.ndarray:
    """d loss / d residual at residual r, relative to L2's (r^2/2 -> gradient r)."""
    pred = r.clone().requires_grad_(True)
    loss_fn(pred, torch.zeros_like(pred), t).sum().backward()
    return (pred.grad / r).numpy()


def huber_figure(out: Path):
    t = torch.linspace(0.01, 0.99, 400, dtype=torch.float64)
    snr_fn = lambda p, q, tt: snr_aware_huber_loss(p, q, tt, 1.0)  # noqa: E731 (defaults cmin .2, cmax 1)
    x0_fn = lambda p, q, tt: x0_threshold_huber_loss(p, q, tt, 1.0, c=1.0)  # noqa: E731

    # a pseudo-Huber's gradient saturates at its threshold, so read each threshold off the library
    # function as the gradient at a huge residual
    huge = torch.full((len(t), 1), 1e9, dtype=torch.float64)
    c_snr = torch.from_numpy(grad_ratio(snr_fn, huge, t).ravel() * 1e9)
    c_x0 = torch.from_numpy(grad_ratio(x0_fn, huge, t).ravel() * 1e9)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.6), facecolor=SURFACE)

    # (a) thresholds in v units vs the v residual of a fixed x0 error
    a = ax[0]
    a.plot(t, c_snr, color=SNR, lw=2)
    a.plot(t, c_x0, color=X0, lw=2)
    for e in (0.1, 0.3):
        a.plot(t, e / t, color=MUTED, lw=1.2, ls="--")
        a.annotate(
            f"v residual of a {e} x0 error (= {e}/t)",
            (0.012, e / 0.012),
            xytext=(4, 2),
            textcoords="offset points",
            fontsize=9,
            color=MUTED,
            va="bottom",
        )
    a.annotate(
        "SNR-Huber threshold c_t",
        (0.99, c_snr[-1]),
        xytext=(-4, 8),
        textcoords="offset points",
        ha="right",
        fontsize=10,
        color=INK,
    )
    a.annotate(
        "x0-Huber threshold c/t (c = 1)",
        (0.35, 1 / 0.35),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=10,
        color=INK,
    )
    a.fill_between(t, c_snr, 0.3 / t, where=(0.3 / t) > c_snr, color=SNR, alpha=0.10, lw=0)
    a.text(0.08, 0.45, "SNR-Huber saturated on a 0.3 x0 error\n(gradient capped at c_t)", fontsize=9, color=INK)
    a.set_yscale("log")
    a.set_ylim(0.1, 40)
    a.set_xlim(0, 1)
    style(a, "(a) Huber threshold in velocity units", "t (noise level)", "threshold / |v residual| (log)")

    # (b) gradient relative to L2 on a fixed x0 error
    b = ax[1]
    ends = []
    for e, ls in ((0.1, "-"), (0.3, "--")):
        r = (e / t).view(-1, 1)
        tt = t.clone()
        for fn, color, name in ((snr_fn, SNR, "SNR-Huber"), (x0_fn, X0, "x0-Huber")):
            y = grad_ratio(fn, r, tt).ravel()
            b.plot(t, y, color=color, lw=2, ls=ls, label=f"{name}, x0 error {e}")
            ends.append((y[0], f"{name}, {e}", color))
    b.axhline(1.0, color=MUTED, lw=1, ls=":")
    b.text(0.99, 1.015, "L2", ha="right", va="bottom", fontsize=9, color=MUTED)
    b.set_ylim(0, 1.08)
    b.set_xlim(0, 1)
    style(b, "(b) Pushback on a fixed x0 error, relative to L2", "t (noise level)", "gradient / L2 gradient")
    b.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")

    fig.suptitle(
        "An x0 error e is a velocity residual e/t. SNR-Huber's threshold tops out at 1 at low t, so low-t texture "
        "errors land in its capped regime; x0-Huber's threshold grows as 1/t and keeps them quadratic.",
        color=INK,
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    print(out)


def load_latent(path: str | None, key: str | None) -> torch.Tensor:
    if path:
        from safetensors.torch import load_file

        d = load_file(path)
        x = d[key or next(iter(d))].float()
        return x if x.ndim == 4 else x.unsqueeze(0)
    g = torch.Generator().manual_seed(0)
    h, w = 36, 64
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    blob = (((xx + 0.2) ** 2 / 0.25 + yy**2 / 0.5) < 1).float()
    x = 0.1 * torch.randn(1, 16, h, w, generator=g) + 1.5 * blob * torch.randn(1, 16, h, w, generator=g)
    return x + xx  # smooth gradient background


def energy_figure(x0: torch.Tensor, out: Path, t_show: float = 0.1, mottle_amp: float = 0.3):
    level = 1
    swt = StationaryWaveletTransform("sym7", torch.device("cpu"))
    c0 = swt.decompose(x0, level)
    t = torch.tensor([t_show])
    n_prime = t_show**2 / (1 - t_show) ** 2
    g = torch.Generator().manual_seed(1)

    # Band-domain simulation of the ideal (posterior-mean) estimate and of a mottled one.
    # SWT sym7 bands of unit white noise have unit variance, so x_t/(1-t) in a band is c0 + sqrt(N') n.
    ideal, mottled = {}, {}
    for b in HF_BANDS:
        c = c0[b][0]
        s = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(c.pow(2), (2, 2, 2, 2), mode="reflect"), 5, stride=1)
        gain = s / (s + n_prime)
        est = gain * (c + n_prime**0.5 * torch.randn(c.shape, generator=g))
        ideal[b] = [est]
        mottled[b] = [est + mottle_amp * torch.randn(c.shape, generator=g)]

    raw = sum(c0[b][0].pow(2).mean(1, keepdim=True) for b in HF_BANDS)[0, 0]
    win = {w: sum(local_energy(c0[b][0], w) for b in HF_BANDS)[0, 0] for w in (3, 5, 9)}
    gs = sum(optimal_local_energy(c0[b][0], t) for b in HF_BANDS)[0, 0]
    m_ideal = (
        sum(local_energy(ideal[b][0]) for b in HF_BANDS) / sum(optimal_local_energy(c0[b][0], t) for b in HF_BANDS)
    )[0, 0]
    m_mott = (
        sum(local_energy(mottled[b][0]) for b in HF_BANDS) / sum(optimal_local_energy(c0[b][0], t) for b in HF_BANDS)
    )[0, 0]
    mask = flat_mask(c0)[0, 0]

    fig, ax = plt.subplots(2, 4, figsize=(18, 6.4), facecolor=SURFACE, layout="constrained")
    lo = float(torch.quantile(win[5], 0.01))
    hi = float(raw.max())
    enorm = LogNorm(vmin=max(lo, 1e-4), vmax=hi)
    panels = [
        (raw, "HF energy per coefficient (no window)\nchannel-averaged, speckled"),
        (win[3], "local energy, 3x3 window"),
        (win[5], "local energy S, 5x5 window (default)\n= 80x80 image px on FLUX.2"),
        (win[9], "local energy, 9x9 window\nbleeds busy energy into flat edges"),
    ]
    for a, (img, title) in zip(ax[0], panels):
        im = a.imshow(img.clamp(min=1e-4).numpy(), cmap=SEQ, norm=enorm)
        a.set_title(title, loc="left", fontsize=10, color=INK)
        a.axis("off")
    fig.colorbar(im, ax=ax[0].tolist(), fraction=0.012, pad=0.01, label="HF energy (log)")

    a = ax[1, 0]
    im = a.imshow(gs.clamp(min=1e-4).numpy(), cmap=SEQ, norm=enorm)
    a.set_title(f"target G*S at t={t_show}\n(optimal estimate's expected energy)", loc="left", fontsize=10, color=INK)
    a.axis("off")

    a = ax[1, 1]
    a.imshow(mask.float().numpy(), cmap=LinearSegmentedColormap.from_list("m", [SURFACE, "#6b6a64"]))
    a.set_title("flat mask: flattest 30% of S", loc="left", fontsize=10, color=INK)
    a.axis("off")

    mnorm = LogNorm(vmin=1 / 3, vmax=3)
    for a, m, title in (
        (ax[1, 2], m_ideal, "M = E_loc(x0_hat) / G*S, ideal estimate"),
        (ax[1, 3], m_mott, f"M, estimate + {mottle_amp} x0 mottle everywhere"),
    ):
        im2 = a.imshow(m.numpy(), cmap=DIV, norm=mnorm)
        fm, bm = float(m[mask].mean()), float(m[~mask].mean())
        a.set_title(f"{title}\nflat {fm:.2f}   busy {bm:.2f}", loc="left", fontsize=10, color=INK)
        a.axis("off")
    cb = fig.colorbar(im2, ax=ax[1].tolist(), fraction=0.012, pad=0.01, label="M (log, 1 = ideal)")
    cb.set_ticks([1 / 3, 0.5, 1, 2, 3], labels=["1/3", "1/2", "1", "2", "3"])
    cb.minorticks_off()

    fig.suptitle(
        "Local energy matching: an ideal estimate sits at M = 1 everywhere; added texture pushes M above 1, "
        "most where the target energy G*S is lowest (flat regions)",
        color=INK,
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.savefig(out, dpi=100, facecolor=SURFACE)
    plt.close(fig)
    print(out, f"(flat M ideal {float(m_ideal[mask].mean()):.2f}, mottled {float(m_mott[mask].mean()):.2f})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latent", help="safetensors file with a [1, C, H, W] or [C, H, W] latent")
    p.add_argument("--key", help="tensor key in --latent (default: first)")
    p.add_argument("--out", default="docs/images")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    huber_figure(out / "huber_thresholds.png")
    energy_figure(load_latent(args.latent, args.key), out / "local_energy_window.png")


if __name__ == "__main__":
    main()
