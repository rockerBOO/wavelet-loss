import pytest
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from wavelet_loss import (
    WaveletLoss,
)

from wavelet_transform import (
    DiscreteWaveletTransform,
    StationaryWaveletTransform,
    QuaternionWaveletTransform,
)


class TestWaveletLoss:
    @pytest.fixture(autouse=True)
    def no_grad_context(self):
        with torch.no_grad():
            yield

    @pytest.fixture
    def setup_inputs(self):
        # Create simple test inputs
        batch_size = 2
        channels = 3
        height = 64
        width = 64

        # Create predictable patterns for testing
        pred = torch.zeros(batch_size, channels, height, width)
        target = torch.zeros(batch_size, channels, height, width)

        # Add some patterns
        for b in range(batch_size):
            for c in range(channels):
                # Create different patterns for pred and target
                pred[b, c] = torch.sin(torch.linspace(0, 4 * np.pi, width)).view(
                    1, -1
                ) * torch.sin(torch.linspace(0, 4 * np.pi, height)).view(-1, 1)
                target[b, c] = torch.sin(torch.linspace(0, 4 * np.pi, width)).view(
                    1, -1
                ) * torch.sin(torch.linspace(0, 4 * np.pi, height)).view(-1, 1)

                # Add some differences
                if b == 1:
                    pred[b, c] += 0.2 * torch.randn(height, width)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return pred.to(device), target.to(device), device

    # ... (previous tests remain the same)

    def test_timestep_weighting(self, setup_inputs):
        """
        Test timestep weighting mechanism
        """
        pred, target, device = setup_inputs

        # Test different timestep values
        loss_fn = WaveletLoss(
            wavelet="db4",
            level=2,
            transform_type="dwt",
            device=device,
            max_timestep=1000,
            timestep_intensity=0.5,
        )

        # Test different timestep scenarios
        timesteps = [0, 500, 1000]
        losses_with_timesteps = []

        for timestep in timesteps:
            timestep_tensor = torch.tensor([timestep] * pred.shape[0], device=device)
            losses, metrics = loss_fn(pred, target, timestep=timestep_tensor)

            # Check timestep-adjusted metrics are present
            assert "wavelet_loss/avg_timestep_adjusted_weight" in metrics

            # Collect losses for comparison
            losses_with_timesteps.append(losses)

        # Verify that losses change with timestep
        # Note: this might not always be true due to the complex weighting mechanism
        # So we'll do a soft comparison
        for i in range(len(timesteps) - 1):
            diff_found = False
            for loss1, loss2 in zip(
                losses_with_timesteps[i], losses_with_timesteps[i + 1]
            ):
                if abs(loss1.mean().item() - loss2.mean().item()) > 1e-6:
                    diff_found = True
                    break

            # At least some losses should be different across timesteps
            assert diff_found, (
                f"No meaningful difference in losses between timesteps {timesteps[i]} and {timesteps[i + 1]}"
            )

    def test_qwt_custom_component_weighting(self, setup_inputs):
        """
        Enhanced test for Quaternion Wavelet Transform with custom component weights
        """
        pred, target, device = setup_inputs

        # Test various component weight configurations
        test_configs = [
            # Default weights
            None,
            # Custom weights with varying intensities
            {"r": 1.0, "i": 0.1, "j": 0.1, "k": 0.05},
            {"r": 0.5, "i": 1.0, "j": 0.7, "k": 0.3},
        ]

        for weights in test_configs:
            loss_fn = WaveletLoss(
                wavelet="db4",
                level=2,
                transform_type="qwt",
                device=device,
                quaternion_component_weights=weights,
            )

            # Perform forward pass
            losses, component_losses = loss_fn(pred, target)

            # Basic checks
            assert len(losses) == 2  # level=2

            # Verify component loss keys
            for level in range(2):
                for component in ["r", "i", "j", "k"]:
                    for band in ["ll", "lh", "hl", "hh"]:
                        key = f"{component}_{band}_{level + 1}"
                        assert key in component_losses, f"Missing key: {key}"

            # Optional: Check if weights are applied correctly
            if weights:
                # Verify each component has different contribution to the loss
                unique_losses = set(component_losses.values())
                assert len(unique_losses) > 1, (
                    "Component losses should have different magnitudes"
                )

    def test_smooth_timestep_weight(self, setup_inputs):
        """
        Test the smooth_timestep_weight method directly
        """
        _, _, device = setup_inputs

        loss_fn = WaveletLoss(
            wavelet="db4",
            level=2,
            transform_type="dwt",
            device=device,
            max_timestep=1000,
        )

        # Test various timestep values
        test_timesteps = [0, 250, 500, 750, 1000]
        weights = []

        for timestep in test_timesteps:
            timestep_tensor = torch.tensor([timestep] * 2, device=device)
            weight = loss_fn.smooth_timestep_weight(timestep_tensor)
            weights.append(weight.mean().item())

        # Verify monotonicity and range
        for i in range(1, len(weights)):
            assert weights[i] <= weights[i - 1], (
                "Weights should be monotonically decreasing"
            )

        # Check weight ranges
        assert all(0 <= w <= 1 for w in weights), "Weights should be between 0 and 1"

        # Check extreme cases
        zero_weight = loss_fn.smooth_timestep_weight(
            torch.tensor([0] * 2, device=device)
        )
        max_weight = loss_fn.smooth_timestep_weight(
            torch.tensor([1000] * 2, device=device)
        )

        assert zero_weight.mean().item() > 0.5, (
            "Weight at early timestep should be high"
        )
        assert max_weight.mean().item() < 0.5, "Weight at max timestep should be low"


# Remaining previous tests...
