import pytest
import torch

from wavelet_loss import WaveletLoss


def _loss(**kwargs):
    return WaveletLoss(wavelet="db2", level=2, transform_type="dwt", **kwargs)


def _inputs():
    torch.manual_seed(0)
    return torch.randn(2, 4, 32, 32), torch.randn(2, 4, 32, 32)


def test_default_max_timestep_is_flow_matching_native():
    assert _loss().max_timestep == 1.0


def test_weight_varies_across_sigmas_with_defaults():
    """The original bug: [0,1] sigmas produced a constant weight of ~1.0."""
    lf = _loss()
    w_clean = lf.smooth_timestep_weight(torch.tensor([0.0])).item()
    w_cutoff = lf.smooth_timestep_weight(torch.tensor([0.7])).item()
    w_noise = lf.smooth_timestep_weight(torch.tensor([1.0])).item()
    assert w_clean > 0.95, "weight near clean data (t=0) should be ~1"
    assert abs(w_cutoff - 0.5) < 1e-6, "weight should cross 0.5 at the cutoff"
    assert w_noise < 0.05, "weight near pure noise (t=1) should be ~0"


def test_new_formula_matches_legacy_curve():
    """Defaults must exactly reproduce sigmoid(((1 - t/T) - 0.3) * 10)."""
    lf = _loss(max_timestep=1000)
    for t in [0.0, 100.0, 250.0, 500.0, 700.0, 900.0, 1000.0]:
        ts = torch.tensor([t])
        legacy = torch.sigmoid(((1.0 - ts / 1000.0) - 0.3) * 10.0)
        assert torch.allclose(lf.smooth_timestep_weight(ts), legacy, atol=1e-6)


def test_cutoff_and_width_are_configurable():
    lf = _loss(timestep_cutoff=0.5, timestep_transition_width=0.2)
    w_mid = lf.smooth_timestep_weight(torch.tensor([0.5])).item()
    assert abs(w_mid - 0.5) < 1e-6
    # Narrower width => sharper transition than default at the same offset
    sharp = lf.smooth_timestep_weight(torch.tensor([0.6])).item()
    default = _loss(timestep_cutoff=0.5).smooth_timestep_weight(torch.tensor([0.6])).item()
    assert sharp < default


def test_forward_raises_on_timestep_above_max():
    lf = _loss()
    pred, target = _inputs()
    with pytest.raises(ValueError, match="max_timestep"):
        lf(pred, target, timestep=torch.tensor([500.0, 500.0]))


def test_forward_raises_on_negative_timestep():
    lf = _loss()
    pred, target = _inputs()
    with pytest.raises(ValueError, match="max_timestep"):
        lf(pred, target, timestep=torch.tensor([-0.1, 0.5]))


def test_forward_accepts_boundary_timesteps():
    lf = _loss()
    pred, target = _inputs()
    loss, _ = lf(pred, target, timestep=torch.tensor([0.0, 1.0]))
    assert torch.isfinite(loss)


def test_qwt_forward_validates_timestep():
    lf = WaveletLoss(wavelet="db2", level=1, transform_type="qwt")
    pred, target = _inputs()
    with pytest.raises(ValueError, match="max_timestep"):
        lf(pred, target, timestep=torch.tensor([500.0, 500.0]))


def test_loss_actually_varies_with_sigma():
    """End-to-end: different sigmas must produce different loss values."""
    lf = _loss()
    pred, target = _inputs()
    loss_low, _ = lf(pred, target, timestep=torch.tensor([0.1, 0.1]))
    loss_high, _ = lf(pred, target, timestep=torch.tensor([0.9, 0.9]))
    assert abs(loss_low.item() - loss_high.item()) > 1e-6
