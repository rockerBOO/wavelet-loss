"""musubi adapter tests, with the one musubi_tuner import it needs stubbed out."""

import sys
import types
from argparse import Namespace
from dataclasses import dataclass, field

import pytest
import torch


@pytest.fixture(scope="module")
def musubi():
    if "musubi_tuner" not in sys.modules:
        stub = types.ModuleType("musubi_tuner.training.timesteps")
        stub.compute_loss_weighting_for_sd3 = lambda *a, **k: None
        for name in ("musubi_tuner", "musubi_tuner.training"):
            sys.modules[name] = types.ModuleType(name)
        sys.modules["musubi_tuner.training.timesteps"] = stub
    import wavelet_loss.musubi as m

    return m


@dataclass
class _Out:
    pred: torch.Tensor
    target: torch.Tensor
    extra: dict = field(default_factory=dict)


def _ctx(ndim=4, stash=True, t=(100.0, 600.0)):
    torch.manual_seed(0)
    shape = (2, 8, 32, 32) if ndim == 4 else (2, 8, 1, 32, 32)
    latents, noise = torch.randn(shape), torch.randn(shape)
    ts = torch.tensor(t)
    s = (ts / 1000).view(-1, *([1] * (len(shape) - 1)))
    noisy = (1 - s) * latents + s * noise
    target = noise - latents
    pred = (target + 0.1 * torch.randn(shape)).requires_grad_(True)
    out = _Out(pred, target, {"noisy_model_input": noisy} if stash else {})
    return Namespace(
        args=Namespace(weighting_scheme="none"),
        output=out,
        timesteps=ts,
        noise_scheduler=None,
        dit_dtype=torch.float32,
        network_dtype=torch.float32,
        global_step=0,
    )


KW = dict(transform_type="swt", wavelet="sym7", level=2, band_weights={"ll": 0.0, "lh": 1.0, "hl": 1.0, "hh": 1.0})


@pytest.mark.parametrize("ndim", [4, 5])
def test_x0_huber_adapter_with_energy_and_mottle_metrics(musubi, ndim):
    fn = musubi.WaveletPlusX0Huber(alpha=1.0, loss_type="x0_huber", energy_beta=0.1, mottle_metrics=True, **KW)
    ctx = _ctx(ndim)
    loss, metrics = fn(ctx)
    loss.backward()
    assert torch.isfinite(loss) and ctx.output.pred.grad is not None
    assert {"loss/x0_huber", "loss/wavelet", "loss/energy"} <= metrics.keys()
    assert any(k.startswith("mottle/M_flat_") for k in metrics)


def test_energy_is_additive_to_existing_adapter(musubi):
    base = musubi.WaveletPlusSNRHuber(alpha=1.0, loss_type="snr_huber", **KW)
    with_e = musubi.WaveletPlusSNRHuber(alpha=1.0, loss_type="snr_huber", energy_beta=0.5, **KW)
    ctx = _ctx()
    l0, _ = base(ctx)
    l1, m1 = with_e(ctx)
    assert (l1 - l0).item() == pytest.approx(0.5 * m1["loss/energy"], rel=1e-4)


def test_energy_without_stash_raises(musubi):
    fn = musubi.WaveletPlusMSE(alpha=1.0, energy_beta=0.1, **KW)
    with pytest.raises(ValueError, match="noisy_model_input"):
        fn(_ctx(stash=False))


def test_defaults_do_not_need_stash(musubi):
    fn = musubi.WaveletPlusSNRHuber(alpha=1.0, loss_type="snr_huber", **KW)
    loss, metrics = fn(_ctx(stash=False))
    assert torch.isfinite(loss) and "loss/energy" not in metrics
