"""Local high-frequency energy of flow-matching x0 estimates.

Flow matching: ``x_t = (1 - t) x0 + t eps`` and ``x0_hat = x_t - t v_hat``. Under a local
Gaussian model of each wavelet HF band (per channel, local x0 energy ``S``), the ideal
(posterior-mean) estimate has expected local energy::

    E[x0_hat*^2] = G S,   G = S / (S + N'),   N' = t^2 / (1 - t)^2

(verified against real Flux.2 latents to within a few percent). A model that paints extra
HF texture into flat regions -- mottle -- has local energy well above ``G S`` there. A
per-coefficient loss can't see "too much texture here"; these functions compare energy
directly:

- :func:`local_energy_matching_loss`: ``(log E_loc(x0_hat) - log G S)^2``, a training term.
- :func:`mottle_ratio`: ``sum E_loc(x0_hat) / sum G S`` over flat / busy regions, a metric
  (~1 ideal, >1 mottle, <1 over-smooth).

All functions take coefficient dicts as returned by ``WaveletTransform.decompose``
(``{"lh": [level1, level2, ...], ...}``) and use the HF bands only.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

HF_BANDS = ("lh", "hl", "hh")


def local_energy(x: Tensor, window: int = 5) -> Tensor:
    """Channel-averaged local energy ``mean_c(boxblur(x^2))`` -> ``[B, 1, H, W]`` (reflect pad)."""
    return _box(x.pow(2), window).mean(1, keepdim=True)


def optimal_local_energy(x0_band: Tensor, t_frac: Tensor, window: int = 5) -> Tensor:
    """Expected channel-averaged local energy ``G S`` of the ideal x0 estimate -> ``[B, 1, H, W]``.

    ``t_frac`` is the per-sample noise level ``[B]`` in ``[0, 1)``.
    """
    s = _box(x0_band.pow(2), window)
    t = t_frac.view(-1, 1, 1, 1)
    n = t.pow(2) / (1 - t).pow(2)
    return (s * s / (s + n)).mean(1, keepdim=True)


def local_energy_matching_loss(
    x0_hat_coeffs: dict[str, list[Tensor]],
    x0_coeffs: dict[str, list[Tensor]],
    t_frac: Tensor,
    level: int,
    weight: Tensor | None = None,
    window: int = 5,
    eps: float = 1e-3,
    t_max: float = 0.95,
) -> Tensor:
    """Sum over HF bands/levels of ``mean(weight * (log(E_loc(x0_hat)+eps) - log(G S+eps))^2)``.

    Args:
        x0_hat_coeffs: Decomposition of the model's x0 estimate (carries the gradient).
        x0_coeffs: Decomposition of the clean latent.
        t_frac: Per-sample noise level ``[B]`` in ``[0, 1]``; clamped to ``t_max`` because
            ``N'`` diverges at t=1.
        level: Number of decomposition levels to use.
        weight: Optional per-sample weight ``[B]`` (e.g. the wavelet timestep fade).
        window: Box window for local energy.
        eps: Floor inside the logs (latent HF energy is O(1)).
        t_max: Clamp for ``t_frac``.
    """
    t = t_frac.clamp(max=t_max)
    total = x0_coeffs[HF_BANDS[0]][0].new_zeros(())
    for j in range(level):
        for b in HF_BANDS:
            e_hat = local_energy(x0_hat_coeffs[b][j], window)
            with torch.no_grad():
                e_opt = optimal_local_energy(x0_coeffs[b][j], t, window)
            d = (torch.log(e_hat + eps) - torch.log(e_opt + eps)).pow(2)
            if weight is not None:
                d = weight.view(-1, 1, 1, 1) * d
            total = total + d.mean()
    return total


@torch.no_grad()
def flat_mask(x0_coeffs: dict[str, list[Tensor]], quantile: float = 0.3, window: int = 5) -> Tensor:
    """Per-sample mask ``[B, 1, H, W]`` of the flattest ``quantile`` of locations (level-1 HF energy of x0)."""
    r = sum(local_energy(x0_coeffs[b][0], window) for b in HF_BANDS) / len(HF_BANDS)
    thr = torch.quantile(r.flatten(1).float(), quantile, dim=1).view(-1, 1, 1, 1)
    return r <= thr


@torch.no_grad()
def mottle_ratio(
    x0_hat_coeffs: dict[str, list[Tensor]],
    x0_coeffs: dict[str, list[Tensor]],
    t_frac: Tensor,
    level: int,
    window: int = 5,
    quantile: float = 0.3,
    t_max: float = 0.95,
) -> dict[str, float]:
    """M = ``sum E_loc(x0_hat) / sum G S`` over flat and busy regions (whole batch pooled).

    M ~ 1: ideal. M > 1: extra HF texture (mottle). M < 1: over-smoothed. Costs GPU syncs.
    """
    t = t_frac.clamp(max=t_max)
    num = sum(local_energy(x0_hat_coeffs[b][j], window) for j in range(level) for b in HF_BANDS)
    den = sum(optimal_local_energy(x0_coeffs[b][j], t, window) for j in range(level) for b in HF_BANDS)
    mask = flat_mask(x0_coeffs, quantile, window)
    out = {}
    for name, m in (("flat", mask), ("busy", ~mask)):
        out[name] = (num * m).sum().item() / max((den * m).sum().item(), 1e-12)
    return out


def _box(x: Tensor, window: int) -> Tensor:
    p = window // 2
    return F.avg_pool2d(F.pad(x, (p, p, p, p), mode="reflect"), window, stride=1)
