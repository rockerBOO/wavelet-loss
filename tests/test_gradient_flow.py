import torch
import pytest
import numpy as np
from torch.nn import functional as F
from wavelet_loss import WaveletLoss

@pytest.fixture
def sample_tensors():
    batch_size, channels, height, width = 4, 3, 32, 32
    pred = torch.randn(batch_size, channels, height, width, requires_grad=True)
    target = torch.randn(batch_size, channels, height, width, requires_grad=True)
    return pred, target

def test_gradient_properties(sample_tensors):
    """
    Comprehensive test of gradient properties:
    1. Gradient flow exists
    2. No gradient explosion
    3. Reasonable gradient distribution
    """
    pred, target = sample_tensors

    # Initialize wavelet loss
    wavelet_loss = WaveletLoss()

    # Repeated gradient computations to check statistical properties
    num_iterations = 5
    grad_stats = {
        'pred_grad_norms': [],
        'target_grad_norms': [],
        'pred_grad_means': [],
        'target_grad_means': [],
        'pred_grad_stds': [],
        'target_grad_stds': []
    }

    for _ in range(num_iterations):
        # Reset gradients
        pred.grad = None
        target.grad = None

        # Compute loss
        losses, _ = wavelet_loss(pred, target)
        reduced_losses = [loss.mean() for loss in losses]
        total_loss = torch.sum(torch.stack(reduced_losses))
        total_loss.backward()

        # Collect gradient statistics
        if pred.grad is not None:
            grad_stats['pred_grad_norms'].append(pred.grad.norm().item())
            grad_stats['pred_grad_means'].append(pred.grad.mean().item())
            grad_stats['pred_grad_stds'].append(pred.grad.std().item())

        if target.grad is not None:
            grad_stats['target_grad_norms'].append(target.grad.norm().item())
            grad_stats['target_grad_means'].append(target.grad.mean().item())
            grad_stats['target_grad_stds'].append(target.grad.std().item())

    # Perform checks
    def check_grad_stats(name, stats):
        # Check gradients are not zero
        assert len(stats) > 0, f"No {name} gradients"

        # Gradient norms
        grad_norms = stats['%s_grad_norms' % name]
        assert np.mean(grad_norms) > 0, f"{name} gradient norm is zero"
        assert max(grad_norms) < 100, f"{name} gradient norm explosion"

        # Gradient means
        grad_means = stats['%s_grad_means' % name]
        assert np.std(grad_means) < 1.0, f"High {name} gradient mean variance"

        # Gradient standard deviations (more relaxed check)
        grad_stds = stats['%s_grad_stds' % name]
        assert 0.0001 < np.mean(grad_stds) < 10.0, f"Unusual {name} gradient std"

    # Run checks for both prediction and target
    check_grad_stats('pred', grad_stats)
    check_grad_stats('target', grad_stats)

def test_multiple_loss_functions(sample_tensors):
    """
    Test gradient flow with multiple loss functions
    """
    pred, target = sample_tensors

    # Test different loss functions
    loss_functions = [
        F.mse_loss,    # Mean Squared Error
        F.l1_loss,     # Mean Absolute Error
        F.huber_loss   # Huber Loss
    ]

    for loss_fn in loss_functions:
        # Reset gradients
        pred.grad = None
        target.grad = None

        # Initialize wavelet loss with specific loss function
        wavelet_loss = WaveletLoss(loss_fn=loss_fn)

        # Compute loss
        losses, _ = wavelet_loss(pred, target)
        reduced_losses = [loss.mean() for loss in losses]
        total_loss = torch.sum(torch.stack(reduced_losses))

        # Compute gradients
        total_loss.backward()

        # Check gradient properties
        assert pred.grad is not None, f"No gradient for {loss_fn.__name__}"
        assert target.grad is not None, f"No gradient for target with {loss_fn.__name__}"

        # Basic gradient checks
        assert not torch.all(pred.grad == 0), f"Zero gradient for {loss_fn.__name__}"
        assert not torch.all(target.grad == 0), f"Zero gradient for target with {loss_fn.__name__}"

        # Gradient norm check
        assert pred.grad.norm() < 100, f"Gradient explosion for {loss_fn.__name__}"
        assert target.grad.norm() < 100, f"Target gradient explosion for {loss_fn.__name__}"