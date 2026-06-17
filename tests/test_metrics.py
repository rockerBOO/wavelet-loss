import torch

from wavelet_loss import WaveletLoss


def _loss(metrics=True):
    return WaveletLoss(wavelet="db2", level=3, transform_type="dwt", metrics=metrics)


def _inputs(seed=0):
    torch.manual_seed(seed)
    return torch.randn(2, 4, 32, 32), torch.randn(2, 4, 32, 32)


def test_total_metric_matches_reduced_loss():
    lf = _loss()
    pred, target = _inputs()
    scalar, metrics = lf(pred, target, reduce=True)
    assert "wavelet_loss/total" in metrics
    assert abs(metrics["wavelet_loss/total"] - scalar.item()) < 1e-5


def test_legacy_loss_keys_removed():
    lf = _loss()
    pred, target = _inputs()
    _, metrics = lf(pred, target)
    assert not any(k.startswith("pattern_loss-") for k in metrics)
    assert not any(k.startswith("total_loss-") for k in metrics)


def test_band_loss_keys_are_1_indexed_and_skip_masked_ll():
    lf = _loss()
    pred, target = _inputs()
    _, metrics = lf(pred, target)
    # active detail band at finest level present, 1-indexed
    assert "wavelet_loss/band_loss/lh1" in metrics
    assert "wavelet_loss/weighted_band_loss/lh1" in metrics
    # default ll_level_threshold=-1 masks LL levels 1 and 2; only LL3 active
    assert "wavelet_loss/band_loss/ll1" not in metrics
    assert "wavelet_loss/band_loss/ll2" not in metrics
    assert "wavelet_loss/band_loss/ll3" in metrics
    # no 0-indexed leftovers
    assert not any("lh0" in k or "hh0" in k for k in metrics)


def test_weighted_band_losses_sum_to_total():
    lf = _loss()
    pred, target = _inputs()
    _, metrics = lf(pred, target)
    weighted = sum(v for k, v in metrics.items() if k.startswith("wavelet_loss/weighted_band_loss/"))
    assert abs(weighted - metrics["wavelet_loss/total"]) < 1e-5


def test_avg_hf_signed_mean_metrics_removed():
    lf = _loss()
    pred, target = _inputs()
    _, metrics = lf(pred, target)
    assert "avg_hf_pred" not in metrics
    assert "avg_hf_target" not in metrics


def test_energy_metrics_present_and_nonnegative():
    lf = _loss()
    pred, target = _inputs()
    _, metrics = lf(pred, target)
    assert "wavelet_loss/energy/lh1_pred" in metrics
    assert "wavelet_loss/energy/lh1_target" in metrics
    assert "wavelet_loss/avg_hf_energy_pred" in metrics
    assert "wavelet_loss/avg_hf_energy_target" in metrics
    assert metrics["wavelet_loss/avg_hf_energy_pred"] >= 0.0
    assert metrics["wavelet_loss/avg_hf_energy_target"] >= 0.0


def test_energy_is_amplitude_sensitive():
    lf = _loss()
    torch.manual_seed(0)
    target = torch.randn(2, 4, 32, 32)
    _, metrics = lf(5 * target, target)
    ratio = metrics["wavelet_loss/avg_hf_energy_pred"] / metrics["wavelet_loss/avg_hf_energy_target"]
    # energy ~ amplitude^2, so a 5x-scaled prediction has ~25x the HF energy
    assert 20.0 < ratio < 30.0
