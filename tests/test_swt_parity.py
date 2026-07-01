import numpy as np
import pywt
import torch
from wavelet_transform import StationaryWaveletTransform


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)


def test_swt_matches_pywt_swt2():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, 32, dtype=torch.float64)
    for wavelet in ["haar", "db2", "db4", "sym4"]:
        swt = StationaryWaveletTransform(wavelet=wavelet)
        bands = swt.decompose(x, level=2)
        for s in range(2):
            for c in range(3):
                # pywt.swt2 returns coarsest-first: ref[0]=level2, ref[1]=level1
                ref = pywt.swt2(x[s, c].numpy(), wavelet, level=2, trim_approx=False, norm=False)
                for our_level, ref_idx in [(0, 1), (1, 0)]:
                    cA, (cH, cV, cD) = ref[ref_idx]
                    assert _rel(bands["ll"][our_level][s, c].numpy(), cA) < 1e-6
                    assert _rel(bands["lh"][our_level][s, c].numpy(), cH) < 1e-6
                    assert _rel(bands["hl"][our_level][s, c].numpy(), cV) < 1e-6
                    assert _rel(bands["hh"][our_level][s, c].numpy(), cD) < 1e-6


def test_swt_preserves_spatial_dims_and_is_differentiable():
    x = torch.randn(1, 2, 32, 32, requires_grad=True)
    bands = StationaryWaveletTransform(wavelet="db4").decompose(x, level=2)
    assert bands["ll"][0].shape == x.shape
    loss = sum((t**2).sum() for v in bands.values() for t in v)
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
