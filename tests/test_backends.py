# tests/test_backends.py
import numpy as np
import pywt
import torch
from wavelet_transform.backends import PytorchWaveletsBackend


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)


def test_pytorch_wavelets_backend_matches_pywt_dwt2_zero():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, 32, dtype=torch.float64)
    for wavelet in ["haar", "db4", "sym4"]:
        backend = PytorchWaveletsBackend(wavelet=wavelet, mode="zero")
        bands = backend.decompose(x, level=1)
        cA, (cH, cV, cD) = pywt.dwt2(x[0, 0].numpy(), wavelet, mode="zero")
        assert _rel(bands["ll"][0][0, 0].numpy(), cA) < 1e-6
        assert _rel(bands["lh"][0][0, 0].numpy(), cH) < 1e-6
        assert _rel(bands["hl"][0][0, 0].numpy(), cV) < 1e-6
        assert _rel(bands["hh"][0][0, 0].numpy(), cD) < 1e-6


def test_pytorch_wavelets_backend_per_level_ll_shapes():
    x = torch.randn(1, 1, 32, 32)
    bands = PytorchWaveletsBackend(wavelet="db4", mode="zero").decompose(x, level=3)
    assert len(bands["ll"]) == 3
    # each level halves (mode='zero', db4: 32->19->13->10 by floor((N+k-1)/2))
    assert bands["ll"][0].shape[-1] == 19
    assert bands["ll"][1].shape[-1] == 13
    assert bands["ll"][2].shape[-1] == 10
