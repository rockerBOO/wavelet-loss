import torch
from wavelet_loss import WaveletLoss


def test_wavelet_loss_uses_pytorch_wavelets_backend_by_default():
    lf = WaveletLoss(wavelet="db4", level=2, transform_type="dwt", device=torch.device("cpu"))
    assert lf.transform.__class__.__name__ == "PytorchWaveletsBackend"


def test_wavelet_loss_custom_backend_selectable():
    lf = WaveletLoss(wavelet="db4", level=2, transform_type="dwt", backend="custom", device=torch.device("cpu"))
    assert lf.transform.__class__.__name__ == "CustomDWTBackend"


def test_wavelet_loss_rejects_non_4d_input():
    import pytest

    lf = WaveletLoss(wavelet="db4", level=2, transform_type="dwt")
    with pytest.raises(ValueError, match=r"\[B, C, H, W\]"):
        lf(torch.randn(10, 1), torch.randn(10, 1))
