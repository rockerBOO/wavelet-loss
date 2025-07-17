import numpy as np
import pytest
import torch
import torch.nn.functional as F

from wavelet_loss import WaveletLoss


class TestQWTForward:
    @pytest.fixture
    def setup_qwt_inputs(self):
        """Create test inputs for Quaternion Wavelet Transform"""
        batch_size = 2
        channels = 3
        height = 64
        width = 64

        # Create input tensors with predictable patterns
        pred = torch.zeros(batch_size, channels, height, width)
        target = torch.zeros(batch_size, channels, height, width)

        # Create different patterns
        for b in range(batch_size):
            for c in range(channels):
                # Sinusoidal pattern with added noise
                pred[b, c] = torch.sin(torch.linspace(0, 4 * np.pi, width)).view(1, -1) * torch.sin(
                    torch.linspace(0, 4 * np.pi, height)
                ).view(-1, 1)
                target[b, c] = torch.sin(torch.linspace(0, 4 * np.pi, width)).view(1, -1) * torch.sin(
                    torch.linspace(0, 4 * np.pi, height)
                ).view(-1, 1)

                # Add some differences
                if b == 1:
                    pred[b, c] += 0.2 * torch.randn(height, width)
                    target[b, c] += 0.1 * torch.randn(height, width)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return pred.to(device), target.to(device), device

    def test_qwt_forward_method_signature(self, setup_qwt_inputs):
        """Test the basic signature and return types of the QWT forward method"""
        pred, target, device = setup_qwt_inputs

        loss_fn = WaveletLoss(wavelet="db4", level=2, transform_type="qwt", device=device)

        # Call forward method
        losses, component_losses = loss_fn(pred, target)

        # Check return types
        assert isinstance(losses, list), "Losses should be a list"
        assert isinstance(component_losses, dict), "Component losses should be a dictionary"

        # Check losses structure
        assert len(losses) == 32, "Should have 32 losses, 2 levels, 4 components, 4 bands"
        for loss in losses:
            assert isinstance(loss, torch.Tensor), "Each loss should be a tensor"
            assert loss.dim() == 4, "Loss should be a 4D tensor"

    def test_qwt_forward_component_loss_keys(self, setup_qwt_inputs):
        """Verify the structure of component losses"""
        pred, target, device = setup_qwt_inputs

        loss_fn = WaveletLoss(wavelet="db4", level=2, transform_type="qwt", device=device)

        # Call forward method
        _, component_losses = loss_fn(pred, target)

        # Check component loss keys
        expected_components = ["r", "i", "j", "k"]
        expected_bands = ["ll", "lh", "hl", "hh"]
        expected_levels = [1, 2]

        for component in expected_components:
            for band in expected_bands:
                for level in expected_levels:
                    key = f"{component}_{band}_{level}"
                    assert key in component_losses, f"Missing key: {key}"
                    assert isinstance(component_losses[key], (float, torch.Tensor)), f"Invalid loss type for {key}"

    def test_qwt_forward_custom_component_weights(self, setup_qwt_inputs):
        """Test QWT forward with custom component weights"""
        pred, target, device = setup_qwt_inputs

        # Different weight configurations
        weight_configs = [
            {"r": 1.0, "i": 0.5, "j": 0.5, "k": 0.2},
            {"r": 0.7, "i": 1.0, "j": 0.3, "k": 0.1},
        ]

        for weights in weight_configs:
            loss_fn = WaveletLoss(
                wavelet="db4",
                level=2,
                transform_type="qwt",
                device=device,
                quaternion_component_weights=weights,
            )

            # Call forward method
            losses, component_losses = loss_fn(pred, target)

            # Verify losses are computed with the custom weights
            for component, weight in weights.items():
                # Check that each component has a loss entry
                component_specific_losses = [
                    loss for key, loss in component_losses.items() if key.startswith(f"{component}_")
                ]
                assert len(component_specific_losses) > 0, f"No losses found for component {component}"

    def test_qwt_forward_identical_inputs(self, setup_qwt_inputs):
        """Test QWT forward method with identical inputs"""
        pred, target, device = setup_qwt_inputs

        loss_fn = WaveletLoss(wavelet="db4", level=2, transform_type="qwt", device=device)

        # Use identical inputs
        losses, component_losses = loss_fn(pred, pred)

        # For identical inputs, losses should be very small
        for loss in losses:
            for item in loss:
                assert item.mean().item() < 1e-5, "Loss for identical inputs should be near zero"

        # Component losses should also be near zero
        for loss_value in component_losses.values():
            assert np.abs(loss_value) < 1e-5, "Component loss for identical inputs should be near zero"

    def test_qwt_forward_default_vs_custom_loss_fn(self, setup_qwt_inputs):
        """Test QWT forward method with different loss functions"""

        pred, target, device = setup_qwt_inputs

        # Default MSE loss
        loss_fn_mse = WaveletLoss(wavelet="db4", level=2, transform_type="qwt", device=device)

        # Custom L1 loss
        loss_fn_l1 = WaveletLoss(wavelet="db4", level=2, transform_type="qwt", device=device)
        loss_fn_l1.set_loss_fn(F.l1_loss)

        # Compute losses
        mse_losses, mse_component_losses = loss_fn_mse(pred, target)
        l1_losses, l1_component_losses = loss_fn_l1(pred, target)

        # Verify that losses are different
        assert len(mse_losses) == len(l1_losses)

        # At least some component losses should be different
        different_loss_found = False
        for (mse_key, mse_loss), (l1_key, l1_loss) in zip(mse_component_losses.items(), l1_component_losses.items()):
            assert mse_key == l1_key, "Component loss keys should match"
            if abs(mse_loss - l1_loss) > 1e-6:
                different_loss_found = True

        assert different_loss_found, "At least some component losses should differ between MSE and L1"
