# src/wavelet_transform/backends.py
from typing import Protocol

import numpy as np
import pywt
import torch
from torch import Tensor


class WaveletBackend(Protocol):
    dec_lo: Tensor
    dec_hi: Tensor

    def decompose(self, x: Tensor, level: int) -> dict[str, list[Tensor]]: ...


def _load_filters(wavelet: str, dtype: torch.dtype, device) -> tuple[Tensor, Tensor]:
    w = pywt.Wavelet(wavelet)
    dec_lo = torch.tensor(np.array(w.dec_lo), dtype=dtype, device=device)
    dec_hi = torch.tensor(np.array(w.dec_hi), dtype=dtype, device=device)
    return dec_lo, dec_hi


class PytorchWaveletsBackend:
    """DWT backend wrapping pytorch_wavelets.DWTForward (mode='zero' by default)."""

    def __init__(self, wavelet: str = "db4", mode: str = "zero", device=torch.device("cpu")):
        try:
            from pytorch_wavelets import DWTForward
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pytorch_wavelets is required for backend='pytorch_wavelets'. "
                "Install it (`uv add pytorch_wavelets`) or use backend='custom'."
            ) from exc
        self.wavelet = wavelet
        self.mode = mode
        self.device = device
        self._dwt1 = DWTForward(J=1, wave=wavelet, mode=mode).to(device)
        self.dec_lo, self.dec_hi = _load_filters(wavelet, torch.get_default_dtype(), device)

    def decompose(self, x: Tensor, level: int) -> dict[str, list[Tensor]]:
        bands: dict[str, list[Tensor]] = {"ll": [], "lh": [], "hl": [], "hh": []}
        dwt = self._dwt1.to(device=x.device, dtype=x.dtype)
        ll = x
        for _ in range(level):
            yl, yh = dwt(ll)
            lh, hl, hh = torch.unbind(yh[0], dim=2)  # verified order: (cH, cV, cD) = (lh, hl, hh)
            bands["ll"].append(yl)
            bands["lh"].append(lh)
            bands["hl"].append(hl)
            bands["hh"].append(hh)
            ll = yl
        return bands
