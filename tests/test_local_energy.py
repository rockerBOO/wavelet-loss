import torch

from wavelet_loss.local_energy import local_energy_matching_loss, mottle_ratio, optimal_local_energy
from wavelet_transform import StationaryWaveletTransform

LEVEL = 2


def _latent_with_flat_half(seed=0):
    """Busy left half (unit-variance texture), near-flat right half."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 8, 32, 32, generator=g)
    x[..., 16:] *= 0.05
    return x


def _coeffs(x):
    return StationaryWaveletTransform("sym7", torch.device("cpu")).decompose(x, LEVEL)


def test_optimal_energy_shrinks_with_noise():
    c = _coeffs(_latent_with_flat_half())["hh"][0]
    e = [optimal_local_energy(c, torch.tensor([t])).mean().item() for t in (0.0, 0.3, 0.7)]
    assert e[0] > e[1] > e[2] > 0


def test_perfect_estimate_at_t0_has_zero_energy_loss_and_unit_ratio():
    x0 = _latent_with_flat_half()
    c = _coeffs(x0)
    t = torch.tensor([0.0])
    assert local_energy_matching_loss(c, c, t, LEVEL).item() < 1e-8
    m = mottle_ratio(c, c, t, LEVEL)
    assert abs(m["flat"] - 1) < 1e-5 and abs(m["busy"] - 1) < 1e-5


def test_mottle_in_flat_region_raises_flat_ratio_and_energy_loss():
    x0 = _latent_with_flat_half()
    mottled = x0.clone()
    mottled[..., 16:] += 0.3 * torch.randn(1, 8, 32, 16, generator=torch.Generator().manual_seed(1))
    c, cm = _coeffs(x0), _coeffs(mottled)
    t = torch.tensor([0.0])
    m = mottle_ratio(cm, c, t, LEVEL)
    # level-2 sym7 support (~28px) leaks busy-half energy into the 16px flat half, so the
    # reference G*S there isn't tiny; a truly flat region would give a much larger ratio
    assert m["flat"] > 2 and m["busy"] < 1.2
    assert local_energy_matching_loss(cm, c, t, LEVEL) > local_energy_matching_loss(c, c, t, LEVEL)


def test_energy_loss_gradient_reduces_flat_texture():
    x0 = _latent_with_flat_half()
    x0_hat = (x0 + 0.3 * torch.randn_like(x0)).requires_grad_(True)
    loss = local_energy_matching_loss(_coeffs(x0_hat), _coeffs(x0), torch.tensor([0.0]), LEVEL)
    loss.backward()

    def flat_hf(x):
        return sum(_coeffs(x)[b][0][..., 20:28].pow(2).mean() for b in ("lh", "hl", "hh"))

    # a descent step removes HF energy from the flat half
    stepped = (x0_hat - 0.5 * x0_hat.grad).detach()
    assert flat_hf(stepped) < flat_hf(x0_hat.detach())


def test_per_sample_weight_zero_disables_term():
    x0 = _latent_with_flat_half().repeat(2, 1, 1, 1)
    x0_hat = x0 + 0.3 * torch.randn_like(x0)
    c, ch = _coeffs(x0), _coeffs(x0_hat)
    t = torch.tensor([0.1, 0.1])
    assert local_energy_matching_loss(ch, c, t, LEVEL, weight=torch.zeros(2)).item() == 0.0
