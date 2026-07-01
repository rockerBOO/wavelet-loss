# src/wavelet_transform/backends.py
import warnings
from typing import Protocol

import numpy as np
import pywt
import torch
import torch.nn.functional as F
from torch import Tensor

from .transform import QuaternionWaveletTransform, StationaryWaveletTransform


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


def _dwt1d_zero(x: Tensor, ker: Tensor, dim: int) -> Tensor:
    """Single-axis zero-mode DWT matching pywt: full convolution then downsample 1::2."""
    k = ker.numel()
    ker_flipped = ker.flip(0)
    pad = (0, 0, k - 1, k - 1) if dim == 2 else (k - 1, k - 1, 0, 0)
    xp = F.pad(x, pad)  # zero padding
    weight = ker_flipped.view(1, 1, -1, 1) if dim == 2 else ker_flipped.view(1, 1, 1, -1)
    out = F.conv2d(xp, weight)
    idx = torch.arange(1, out.shape[dim], 2, device=x.device)
    return out.index_select(dim, idx)


class CustomDWTBackend:
    """Hand-rolled DWT matching pywt.dwt2(mode='zero')."""

    def __init__(self, wavelet: str = "db4", mode: str = "zero", device=torch.device("cpu")):
        if mode != "zero":
            raise NotImplementedError(
                f"CustomDWTBackend supports mode='zero' only, got {mode!r}. "
                "Use backend='pytorch_wavelets' for other modes."
            )
        self.wavelet = wavelet
        self.mode = mode
        self.device = device
        self.dec_lo, self.dec_hi = _load_filters(wavelet, torch.get_default_dtype(), device)

    def _single_level(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        lo = self.dec_lo.to(device=x.device, dtype=x.dtype)
        hi = self.dec_hi.to(device=x.device, dtype=x.dtype)
        b, c, h, w = x.shape
        xbc = x.reshape(b * c, 1, h, w)
        lo_h = _dwt1d_zero(xbc, lo, 2)
        hi_h = _dwt1d_zero(xbc, hi, 2)
        ll = _dwt1d_zero(lo_h, lo, 3)
        lh = _dwt1d_zero(hi_h, lo, 3)
        hl = _dwt1d_zero(lo_h, hi, 3)
        hh = _dwt1d_zero(hi_h, hi, 3)
        out = [t.reshape(b, c, *t.shape[2:]) for t in (ll, lh, hl, hh)]
        return out[0], out[1], out[2], out[3]

    def decompose(self, x: Tensor, level: int) -> dict[str, list[Tensor]]:
        bands: dict[str, list[Tensor]] = {"ll": [], "lh": [], "hl": [], "hh": []}
        ll = x
        for _ in range(level):
            ll, lh, hl, hh = self._single_level(ll)
            bands["ll"].append(ll)
            bands["lh"].append(lh)
            bands["hl"].append(hl)
            bands["hh"].append(hh)
        return bands


def make_backend(backend: str, transform_type: str, wavelet: str, mode: str, device):
    """Build a wavelet backend. `backend` selects the DWT implementation; SWT and
    QWT are always custom (no library equivalent in pytorch_wavelets 1.3.0)."""
    if transform_type == "dwt":
        if backend == "pytorch_wavelets":
            return PytorchWaveletsBackend(wavelet=wavelet, mode=mode, device=device)
        if backend == "custom":
            return CustomDWTBackend(wavelet=wavelet, mode=mode, device=device)
        raise ValueError(f"Unknown backend {backend!r}; expected 'pytorch_wavelets' or 'custom'.")
    if transform_type == "swt":
        return StationaryWaveletTransform(wavelet=wavelet, device=device)
    if transform_type == "qwt":
        warnings.warn(
            "qwt is experimental and not validated against a reference transform.",
            UserWarning,
            stacklevel=2,
        )
        return QuaternionWaveletTransform(wavelet=wavelet, device=device)
    raise ValueError(f"Unknown transform_type {transform_type!r}; expected 'dwt', 'swt', or 'qwt'.")
