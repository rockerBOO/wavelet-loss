import torch
from wavelet_loss import WaveletLoss


def test_reduce_true_returns_scalar_and_backprops():
    torch.manual_seed(0)
    pred = torch.randn(2, 4, 32, 32, requires_grad=True)
    target = torch.randn(2, 4, 32, 32)
    lf = WaveletLoss(level=2, transform_type="dwt")
    loss, metrics = lf(pred, target)  # reduce=True default
    assert loss.ndim == 0
    assert loss.requires_grad
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_reduce_false_returns_list():
    pred = torch.randn(2, 4, 32, 32)
    target = torch.randn(2, 4, 32, 32)
    lf = WaveletLoss(level=2, transform_type="dwt")
    losses, _ = lf(pred, target, reduce=False)
    assert isinstance(losses, list)
    assert all(torch.is_tensor(t) for t in losses)
