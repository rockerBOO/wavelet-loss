import pytest
import torch

from wavelet_loss import WaveletLoss, x0_threshold_huber_loss


def _pseudo_huber(r, c):
    return c**2 * (torch.sqrt(1 + (r / c) ** 2) - 1)


def test_equals_pseudo_huber_with_threshold_c_over_t():
    torch.manual_seed(0)
    r = torch.randn(4, 3, 8, 8, dtype=torch.float64) * 5
    t = torch.tensor([0.05, 0.2, 0.5, 0.9], dtype=torch.float64)
    c = 0.7
    got = x0_threshold_huber_loss(r, torch.zeros_like(r), t * 1000, 1000.0, c=c)
    want = _pseudo_huber(r, (c / t).view(-1, 1, 1, 1))
    assert torch.allclose(got, want, rtol=1e-9)


def test_quadratic_while_x0_error_is_small_at_any_t():
    """A small x0 error e at low t is a large velocity residual e/t; it must stay ~L2 (r^2/2)."""
    e = 0.05  # x0 units, well below c=1
    for t in (0.02, 0.1, 0.5):
        r = torch.full((1, 1, 4, 4), e / t)
        got = x0_threshold_huber_loss(r, torch.zeros_like(r), torch.tensor([t]), 1.0, c=1.0)
        assert torch.allclose(got, r.pow(2) / 2, rtol=2e-3)


def test_gradient_bounded_once_x0_error_exceeds_c():
    r = torch.tensor([[[[1e4]]]], requires_grad=True)
    t, c = 0.5, 1.0
    x0_threshold_huber_loss(r, torch.zeros_like(r), torch.tensor([t]), 1.0, c=c).sum().backward()
    assert r.grad.item() == pytest.approx(c / t, rel=1e-3)


def _swt(**kw):
    return WaveletLoss(
        wavelet="sym7", level=2, transform_type="swt", band_weights={"ll": 0, "lh": 1, "hl": 1, "hh": 1}, **kw
    )


def test_waveletloss_x0_huber_band_loss():
    torch.manual_seed(0)
    p, q = torch.randn(2, 4, 32, 32), torch.randn(2, 4, 32, 32)
    t = torch.tensor([0.1, 0.6])
    loss, _ = _swt(use_x0_huber=True, x0_huber_c=0.5)(p, q, t)
    assert torch.isfinite(loss) and loss > 0
    # c -> inf is plain L2 / 2
    big, _ = _swt(use_x0_huber=True, x0_huber_c=1e6)(p, q, t)
    l2, _ = _swt()(p, q, t)
    assert big.item() == pytest.approx(l2.item() / 2, rel=1e-4)


def test_waveletloss_x0_huber_requires_timestep():
    with pytest.raises(ValueError, match="use_x0_huber"):
        _swt(use_x0_huber=True)(torch.randn(1, 2, 32, 32), torch.randn(1, 2, 32, 32))


def test_waveletloss_huber_modes_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _swt(use_x0_huber=True, use_snr_aware_huber=True)
