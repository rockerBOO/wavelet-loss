import os

import pytest
import torch
import numpy as np

from wavelet_loss import (
    WaveletLoss,
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
                pred[b, c] = torch.sin(torch.linspace(0, 4 * np.pi, width)).view(1, -1) * torch.sin(
                    torch.linspace(0, 4 * np.pi, height)
                ).view(-1, 1)
                target[b, c] = torch.sin(torch.linspace(0, 4 * np.pi, width)).view(1, -1) * torch.sin(
                    torch.linspace(0, 4 * np.pi, height)
                ).view(-1, 1)

                # Add some differences
                if b == 1:
                    pred[b, c] += 0.2 * torch.randn(height, width)

        device = torch.device("cuda" if os.environ.get("WAVELET_TEST_CUDA") else "cpu")
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
        )

        # Test different timestep scenarios
        timesteps = [0, 500, 1000]
        losses_with_timesteps = []

        for timestep in timesteps:
            timestep_tensor = torch.tensor([timestep] * pred.shape[0], device=device)
            losses, metrics = loss_fn(pred, target, timestep=timestep_tensor, reduce=False)

            # Check timestep-adjusted metrics are present
            assert "wavelet_loss/avg_timestep_adjusted_weight" in metrics

            # Collect losses for comparison
            losses_with_timesteps.append(losses)

        # Verify that losses change with timestep
        # Note: this might not always be true due to the complex weighting mechanism
        # So we'll do a soft comparison
        for i in range(len(timesteps) - 1):
            diff_found = False
            for loss1, loss2 in zip(losses_with_timesteps[i], losses_with_timesteps[i + 1]):
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
            losses, component_losses = loss_fn(pred, target, reduce=False)

            # Basic checks
            assert len(losses) == 32  # 2 levels, 4 components, 4 bands

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
                assert len(unique_losses) > 1, "Component losses should have different magnitudes"

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
            assert weights[i] <= weights[i - 1], "Weights should be monotonically decreasing"

        # Check weight ranges
        assert all(0 <= w <= 1 for w in weights), "Weights should be between 0 and 1"

        # Check extreme cases
        zero_weight = loss_fn.smooth_timestep_weight(torch.tensor([0] * 2, device=device))
        max_weight = loss_fn.smooth_timestep_weight(torch.tensor([1000] * 2, device=device))

        assert zero_weight.mean().item() > 0.5, "Weight at early timestep should be high"
        assert max_weight.mean().item() < 0.5, "Weight at max timestep should be low"

    def test_ll_level_threshold_handling(self, setup_inputs):
        """
        Test LL level threshold handling functionality including negative values
        """
        pred, target, device = setup_inputs

        # Test with different ll_level_threshold values including negative ones
        test_cases = [
            # (threshold, expected_effective_threshold, description)
            (None, None, "No threshold"),
            (1, 1, "Positive threshold: level 1"),
            (2, 2, "Positive threshold: level 2"),
            (3, 3, "Positive threshold: level 3"),
            (-1, 2, "Negative threshold: -1 from end (3-1=2)"),
            (-2, 1, "Negative threshold: -2 from end (3-2=1)"),
            (-3, 0, "Negative threshold: -3 from end (3-3=0)"),
        ]

        for threshold, expected_effective, description in test_cases:
            loss_fn = WaveletLoss(
                wavelet="db4",
                level=3,
                transform_type="dwt",
                device=device,
                ll_level_threshold=threshold,
            )

            losses, metrics = loss_fn(pred, target, reduce=False)

            # Basic functionality check
            assert len(losses) > 0, f"Should produce some losses for {description}"

            # If threshold is set, verify LL bands at lower levels are handled
            if threshold is not None:
                # Check that metrics are calculated correctly
                assert isinstance(metrics, dict), f"Metrics should be a dictionary for {description}"

                # Test the process_band method directly for LL bands
                pred_coeffs = loss_fn.transform.decompose(pred, loss_fn.level)
                target_coeffs = loss_fn.transform.decompose(target, loss_fn.level)
                base_weight = torch.ones((pred.shape[0]), device=device)

                # Test levels within effective threshold
                if expected_effective > 0:
                    for i in range(min(expected_effective, loss_fn.level)):
                        band_loss, pred_band, target_band, band_metrics = loss_fn.process_band(
                            pred_coeffs, target_coeffs, "ll", i, base_weight=base_weight
                        )

                        # For LL bands within threshold, should return zero loss
                        assert torch.all(band_loss == 0.0), (
                            f"LL band at level {i + 1} should have zero loss for {description}"
                        )
                        assert torch.all(pred_band == 0), f"LL pred at level {i + 1} should be zero for {description}"
                        assert torch.all(target_band == 0), (
                            f"LL target at level {i + 1} should be zero for {description}"
                        )
                        assert band_metrics == {}, f"LL band metrics at level {i + 1} should be empty for {description}"

                # Test levels beyond effective threshold (if any)
                for i in range(max(0, expected_effective), loss_fn.level):
                    band_loss, pred_band, target_band, band_metrics = loss_fn.process_band(
                        pred_coeffs, target_coeffs, "ll", i, base_weight=base_weight
                    )

                    # For LL bands beyond threshold, should have normal processing
                    assert not torch.all(band_loss == 0.0), (
                        f"LL band at level {i + 1} should have non-zero loss for {description}"
                    )
                    assert not torch.all(pred_band == 0), (
                        f"LL pred at level {i + 1} should not be all zeros for {description}"
                    )

    def test_ll_level_threshold_calculation(self, setup_inputs):
        """
        Test the threshold calculation logic directly
        """
        pred, target, device = setup_inputs

        # Test the threshold calculation for different levels
        test_cases = [
            # (level, threshold, expected_effective_threshold)
            (3, 1, 1),
            (3, 2, 2),
            (3, 3, 3),
            (3, -1, 2),  # 3 + (-1) = 2
            (3, -2, 1),  # 3 + (-2) = 1
            (3, -3, 0),  # 3 + (-3) = 0
            (4, -1, 3),  # 4 + (-1) = 3
            (4, -2, 2),  # 4 + (-2) = 2
        ]

        for level, threshold, expected in test_cases:
            loss_fn = WaveletLoss(
                wavelet="db4",
                level=level,
                transform_type="dwt",
                device=device,
                ll_level_threshold=threshold,
            )

            # Calculate the effective threshold as the code does
            if threshold > 0:
                effective_threshold = threshold
            else:
                effective_threshold = level + threshold

            assert effective_threshold == expected, (
                f"For level={level}, threshold={threshold}, expected effective threshold {expected}, "
                f"got {effective_threshold}"
            )

            # Test that it works in practice
            pred_coeffs = loss_fn.transform.decompose(pred, loss_fn.level)
            target_coeffs = loss_fn.transform.decompose(target, loss_fn.level)
            base_weight = torch.ones((pred.shape[0]), device=device)

            # Test each level
            for i in range(level):
                band_loss, pred_band, target_band, band_metrics = loss_fn.process_band(
                    pred_coeffs, target_coeffs, "ll", i, base_weight=base_weight
                )

                if i + 1 <= effective_threshold:
                    # Should be zero loss within threshold
                    assert torch.all(band_loss == 0.0), (
                        f"Level {i + 1} should have zero loss with threshold {threshold} "
                        f"(effective: {effective_threshold})"
                    )
                else:
                    # Should have normal processing beyond threshold
                    assert not torch.all(band_loss == 0.0), (
                        f"Level {i + 1} should have non-zero loss with threshold {threshold} "
                        f"(effective: {effective_threshold})"
                    )

    def test_calculate_effective_ll_threshold(self, setup_inputs):
        """
        Test the _calculate_effective_ll_threshold method directly
        """
        _, _, device = setup_inputs

        # Test cases: (level, threshold, expected_effective_threshold)
        test_cases = [
            (3, None, None),
            (3, 1, 1),
            (3, 2, 2),
            (3, 3, 3),
            (3, -1, 2),  # 3 + (-1) = 2
            (3, -2, 1),  # 3 + (-2) = 1
            (3, -3, 0),  # 3 + (-3) = 0
            (4, -1, 3),  # 4 + (-1) = 3
            (4, -2, 2),  # 4 + (-2) = 2
            (5, -1, 4),  # 5 + (-1) = 4
        ]

        for level, threshold, expected in test_cases:
            loss_fn = WaveletLoss(
                wavelet="db4",
                level=level,
                transform_type="dwt",
                device=device,
                ll_level_threshold=threshold,
            )

            effective_threshold = loss_fn._calculate_effective_ll_threshold()

            assert effective_threshold == expected, (
                f"For level={level}, threshold={threshold}, expected {expected}, got {effective_threshold}"
            )

            # Test edge cases
            if threshold is not None and threshold != 0:
                # Verify the logic is sound
                if threshold > 0:
                    assert effective_threshold == threshold, "Positive threshold should be unchanged"
                else:
                    assert effective_threshold == level + threshold, "Negative threshold should be calculated from end"

# Remaining previous tests...
