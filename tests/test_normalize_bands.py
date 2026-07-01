import torch
from wavelet_loss import WaveletLoss


def _total(lf, p, t):
    # Robust to both return shapes: this task runs before the scalar `reduce`
    # path exists (forward still returns a list), and must also pass afterward.
    out, _ = lf(p, t)
    if isinstance(out, list):
        return float(sum(x.mean() for x in out))
    return float(out)


def test_default_normalize_bands_is_off_and_penalizes_amplitude():
    torch.manual_seed(0)
    target = torch.randn(2, 4, 32, 32)
    lf = WaveletLoss(level=2, transform_type="dwt", ll_level_threshold=None)
    assert lf.normalize_bands is False
    assert _total(lf, target * 5.0, target) > 1e-3  # scaled prediction must cost something
    assert _total(lf, target + 3.0, target) > 1e-3  # offset prediction must cost something
    assert _total(lf, target.clone(), target) < 1e-6  # perfect prediction ~ 0


def test_shared_normalization_is_not_scale_blind():
    torch.manual_seed(0)
    target = torch.randn(2, 4, 32, 32)
    lf = WaveletLoss(level=2, transform_type="dwt", normalize_bands=True, ll_level_threshold=None)
    assert _total(lf, target * 5.0, target) > 1e-3
