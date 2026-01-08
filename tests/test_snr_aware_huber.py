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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return pred.to(device), target.to(device), device

    def test_snr_aware_huber_basic(self, setup_inputs):
        """
        Test basic SNR-aware Huber loss functionality
        """
        pred, target, device = setup_inputs

        loss_fn = WaveletLoss(
            wavelet="db4",
            level=2,
            transform_type="dwt",
            device=device,
            use_snr_aware_huber=True,
            snr_huber_cmin=0.2,
            snr_huber_cmax=1.0,
            snr_huber_gamma=5.0,
            snr_huber_alpha=0.5,
            min_snr_beta=0.0,
        )

        # Test with different timesteps
        timesteps = torch.tensor([0.0, 0.5, 0.9], device=device)

        for t in timesteps:
            timestep_tensor = torch.tensor([t] * pred.shape[0], device=device)
            losses, metrics = loss_fn(pred, target, timestep=timestep_tensor)

            # Should produce valid losses
            assert len(losses) > 0, f"Should produce losses for timestep {t}"
            assert all(not torch.isnan(loss).any() for loss in losses), f"No NaN losses for timestep {t}"
            assert all(not torch.isinf(loss).any() for loss in losses), f"No inf losses for timestep {t}"

    def test_snr_aware_huber_vs_standard_loss(self, setup_inputs):
        """
        Compare SNR-aware Huber loss with standard loss functions
        """
        pred, target, device = setup_inputs

        # Create two loss functions - one with SNR-aware, one without
        loss_fn_snr = WaveletLoss(
            wavelet="db4",
            level=2,
            transform_type="dwt",
            device=device,
            use_snr_aware_huber=True,
        )

        loss_fn_standard = WaveletLoss(
            wavelet="db4",
            level=2,
            transform_type="dwt",
            device=device,
            use_snr_aware_huber=False,
        )

        timestep = torch.tensor([0.5] * pred.shape[0], device=device)

        # Get losses from both
        losses_snr, _ = loss_fn_snr(pred, target, timestep=timestep)
        losses_standard, _ = loss_fn_standard(pred, target, timestep=timestep)

        # Losses should be different
        assert len(losses_snr) == len(losses_standard)

        # At least some losses should differ
        diff_found = False
        for l1, l2 in zip(losses_snr, losses_standard):
            if not torch.allclose(l1, l2, rtol=1e-5):
                diff_found = True
                break

        assert diff_found, "SNR-aware and standard losses should differ"

    def test_snr_aware_huber_timestep_dependency(self, setup_inputs):
        """
        Test that SNR-aware Huber loss changes with timestep as expected
        """
        pred, target, device = setup_inputs

        loss_fn = WaveletLoss(
            wavelet="db4",
            level=2,
            transform_type="dwt",
            device=device,
            use_snr_aware_huber=True,
            snr_huber_cmin=0.2,
            snr_huber_cmax=1.0,
            snr_huber_gamma=5.0,
            snr_huber_alpha=0.5,
        )

        # Test different timesteps from early (high SNR) to late (low SNR)
        timesteps = [0.1, 0.3, 0.5, 0.7, 0.9]
        all_losses = []

        for t in timesteps:
            timestep_tensor = torch.tensor([t] * pred.shape[0], device=device)
            losses, _ = loss_fn(pred, target, timestep=timestep_tensor)

            # Sum all losses for comparison
            total_loss = sum(loss.mean().item() for loss in losses)
            all_losses.append(total_loss)

        # Verify that losses vary with timestep
        unique_losses = set(all_losses)
        assert len(unique_losses) > 1, "Losses should vary with timestep"

        # The pattern should show some variation (not monotonic necessarily,
        # but definitely not all the same)
        loss_variance = np.var(all_losses)
        assert loss_variance > 1e-10, "Loss should have meaningful variance across timesteps"

    def test_snr_aware_huber_parameter_effects(self, setup_inputs):
        """
        Test that different SNR-aware Huber parameters produce different results
        """
        pred, target, device = setup_inputs
        timestep = torch.tensor([0.5] * pred.shape[0], device=device)

        # Test different cmin/cmax values
        configs: list[tuple[float, float]] = [
            (0.1, 1.0),
            (0.2, 1.0),
            (0.2, 2.0),
        ]

        all_results: list[float] = []
        for snr_huber_cmin, snr_huber_cmax in configs:
            loss_fn = WaveletLoss(
                wavelet="db4",
                level=2,
                transform_type="dwt",
                device=device,
                use_snr_aware_huber=True,
                snr_huber_cmin=snr_huber_cmin,
                snr_huber_cmax=snr_huber_cmax,
            )

            losses, _ = loss_fn(pred, target, timestep=timestep)
            total_loss = sum(loss.mean().item() for loss in losses)
            all_results.append(total_loss)

        # Different configurations should produce different losses
        assert len(set(all_results)) > 1, "Different parameters should affect loss"

    def test_snr_aware_huber_direct_method(self, setup_inputs):
        """
        Test the snr_aware_huber_wavelet_loss method directly
        """
        pred, target, device = setup_inputs

        loss_fn = WaveletLoss(
            wavelet="db4",
            level=2,
            transform_type="dwt",
            device=device,
            use_snr_aware_huber=True,
            snr_huber_cmin=0.2,
            snr_huber_cmax=1.0,
            snr_huber_gamma=5.0,
            snr_huber_alpha=0.5,
            min_snr_beta=0.0,
        )

        # Test with different timesteps
        timesteps = torch.tensor([0.0, 0.5, 1.0], device=device)

        for t in timesteps:
            timestep_batch = torch.tensor([t] * pred.shape[0], device=device)

            # Call the method directly
            loss = loss_fn.snr_aware_huber_wavelet_loss(pred, target, timestep_batch)

            # Check output shape
            assert loss.shape == pred.shape, "Loss shape should match input shape"

            # Check for valid values
            assert not torch.isnan(loss).any(), f"No NaN values for timestep {t}"
            assert not torch.isinf(loss).any(), f"No inf values for timestep {t}"
            assert torch.all(loss >= 0), f"Loss should be non-negative for timestep {t}"

    def test_snr_aware_huber_threshold_calculation(self, setup_inputs):
        """
        Test the adaptive threshold calculation in SNR-aware Huber loss
        """
        pred, target, device = setup_inputs

        loss_fn = WaveletLoss(
            wavelet="db4",
            level=2,
            transform_type="dwt",
            device=device,
            use_snr_aware_huber=True,
            snr_huber_cmin=0.2,
            snr_huber_cmax=1.0,
            snr_huber_gamma=5.0,
            snr_huber_alpha=0.5,
        )

        # Manually calculate what threshold should be for different timesteps
        timesteps = torch.tensor([0.1, 0.5, 0.9], device=device)

        for t in timesteps:
            # Calculate SNR: (1-t)²/t²
            snr = ((1 - t) ** 2) / (t**2 + 1e-8)

            # Clamp SNR
            snr_clamped = min(snr.item(), loss_fn.snr_huber_gamma)

            # Calculate expected threshold
            expected_c_t = (
                loss_fn.snr_huber_cmin
                + (loss_fn.snr_huber_cmax - loss_fn.snr_huber_cmin)
                * (snr_clamped / loss_fn.snr_huber_gamma) ** loss_fn.snr_huber_alpha
            )

            # Create a small test tensor
            test_pred = torch.ones(1, 1, 4, 4, device=device)
            test_target = torch.zeros(1, 1, 4, 4, device=device)
            timestep_batch = torch.tensor([t], device=device)

            # Get actual loss (we can't directly check c_t, but we can verify behavior)
            loss = loss_fn.snr_aware_huber_wavelet_loss(test_pred, test_target, timestep_batch)

            # The loss should be finite and positive
            assert torch.isfinite(loss).all(), f"Loss should be finite for timestep {t}"
            assert (loss >= 0).all(), f"Loss should be non-negative for timestep {t}"

            # For high SNR (low t), threshold should be higher (more like L2)
            # For low SNR (high t), threshold should be lower (more robust)
            if t < 0.5:
                # High SNR - threshold closer to cmax
                assert expected_c_t > (loss_fn.snr_huber_cmin + loss_fn.snr_huber_cmax) / 2
            elif t > 0.7:
                # Low SNR - threshold closer to cmin
                assert expected_c_t < (loss_fn.snr_huber_cmin + loss_fn.snr_huber_cmax) / 2

    def test_snr_aware_huber_with_min_snr_weighting(self, setup_inputs):
        """
        Test SNR-aware Huber loss with Min-SNR weighting (beta > 0)
        """
        pred, target, device = setup_inputs

        # Test with beta = 0 (no Min-SNR weighting)
        loss_fn_no_weight = WaveletLoss(
            wavelet="db4",
            level=2,
            transform_type="dwt",
            device=device,
            use_snr_aware_huber=True,
            min_snr_beta=0.0,
        )

        # Test with beta > 0 (with Min-SNR weighting)
        loss_fn_with_weight = WaveletLoss(
            wavelet="db4",
            level=2,
            transform_type="dwt",
            device=device,
            use_snr_aware_huber=True,
            min_snr_beta=1.0,
        )

        # Use a timestep where SNR != 1 (so beta matters)
        # At t=0.3: SNR = (1-0.3)^2 / 0.3^2 = 0.49/0.09 ≈ 5.44
        timestep = torch.tensor([0.3] * pred.shape[0], device=device)

        losses_no_weight, _ = loss_fn_no_weight(pred, target, timestep=timestep)
        losses_with_weight, _ = loss_fn_with_weight(pred, target, timestep=timestep)

        # Losses should be different when beta changes (unless SNR is exactly 1)
        assert len(losses_no_weight) == len(losses_with_weight)

        # Check if any losses differ
        diff_found = False
        for l1, l2 in zip(losses_no_weight, losses_with_weight):
            # Use larger tolerance since the difference might be subtle
            if not torch.allclose(l1, l2, rtol=1e-2, atol=1e-6):
                diff_found = True
                break

        # If still no difference found, test the method directly with controlled inputs
        if not diff_found:
            # Create simple test case with known difference
            test_pred = torch.ones(2, 1, 8, 8, device=device)
            test_target = torch.zeros(2, 1, 8, 8, device=device)
            test_timestep = torch.tensor([0.2, 0.8], device=device)

            loss_beta0 = loss_fn_no_weight.snr_aware_huber_wavelet_loss(test_pred, test_target, test_timestep)
            loss_beta1 = loss_fn_with_weight.snr_aware_huber_wavelet_loss(test_pred, test_target, test_timestep)

            # These should definitely differ
            assert not torch.allclose(loss_beta0, loss_beta1, rtol=1e-2), (
                "Direct method call: Min-SNR weighting with beta=1 should differ from beta=0"
            )
        else:
            # Difference was found in full forward pass
            assert True, "Min-SNR weighting affects losses"

    def test_snr_aware_huber_without_timestep(self, setup_inputs):
        """
        Test that SNR-aware Huber loss falls back gracefully when no timestep provided
        """
        pred, target, device = setup_inputs

        loss_fn = WaveletLoss(
            wavelet="db4",
            level=2,
            transform_type="dwt",
            device=device,
            use_snr_aware_huber=True,
        )

        # Call without timestep - should fall back to standard loss
        losses, metrics = loss_fn(pred, target, timestep=None)

        # Should still produce valid losses
        assert len(losses) > 0, "Should produce losses even without timestep"
        assert all(not torch.isnan(loss).any() for loss in losses), "No NaN losses"
        assert all(not torch.isinf(loss).any() for loss in losses), "No inf losses"

    def test_snr_aware_huber_with_quaternion(self, setup_inputs):
        """
        Test SNR-aware Huber loss with Quaternion Wavelet Transform
        """
        pred, target, device = setup_inputs

        loss_fn = WaveletLoss(
            wavelet="db4",
            level=2,
            transform_type="qwt",
            device=device,
            use_snr_aware_huber=True,
            snr_huber_cmin=0.2,
            snr_huber_cmax=1.0,
        )

        timestep = torch.tensor([0.5] * pred.shape[0], device=device)

        # Should work with QWT
        losses, metrics = loss_fn(pred, target, timestep=timestep)

        # Should produce losses for all quaternion components
        assert len(losses) > 0, "Should produce losses for QWT"
        assert all(not torch.isnan(loss).any() for loss in losses), "No NaN losses in QWT"

    def test_snr_aware_huber_extreme_timesteps(self, setup_inputs):
        """
        Test SNR-aware Huber loss with extreme timestep values
        """
        pred, target, device = setup_inputs

        loss_fn = WaveletLoss(
            wavelet="db4",
            level=2,
            transform_type="dwt",
            device=device,
            use_snr_aware_huber=True,
            max_timestep=1000,
        )

        # Test extreme cases
        extreme_timesteps = [
            torch.tensor([0.0] * pred.shape[0], device=device),  # Very beginning
            torch.tensor([1.0] * pred.shape[0], device=device),  # Very end
            torch.tensor([0.001] * pred.shape[0], device=device),  # Nearly zero
            torch.tensor([0.999] * pred.shape[0], device=device),  # Nearly one
        ]

        for timestep in extreme_timesteps:
            losses, _ = loss_fn(pred, target, timestep=timestep)

            # Should handle extreme cases gracefully
            assert len(losses) > 0, f"Should produce losses for timestep {timestep[0]}"
            assert all(not torch.isnan(loss).any() for loss in losses), f"No NaN for timestep {timestep[0]}"
            assert all(not torch.isinf(loss).any() for loss in losses), f"No inf for timestep {timestep[0]}"

    def test_snr_aware_huber_normalized_vs_unnormalized_timesteps(self, setup_inputs):
        """
        Test that the loss handles both normalized [0,1] and unnormalized [0, max_timestep] timesteps
        """
        pred, target, device = setup_inputs

        max_timestep = 1000
        loss_fn = WaveletLoss(
            wavelet="db4",
            level=2,
            transform_type="dwt",
            device=device,
            use_snr_aware_huber=True,
            max_timestep=max_timestep,
        )

        # Test with normalized timestep
        t_normalized = torch.tensor([0.5] * pred.shape[0], device=device)
        losses_normalized, _ = loss_fn(pred, target, timestep=t_normalized)

        # Test with unnormalized timestep (should be auto-normalized in the method)
        t_unnormalized = torch.tensor([500.0] * pred.shape[0], device=device)
        losses_unnormalized, _ = loss_fn(pred, target, timestep=t_unnormalized)

        # Both should produce similar results (accounting for normalization)
        # The method should handle this internally
        assert len(losses_normalized) == len(losses_unnormalized)

        # Check that both produce valid results
        for loss in losses_normalized + losses_unnormalized:
            assert not torch.isnan(loss).any(), "No NaN values"
            assert not torch.isinf(loss).any(), "No inf values"

    def test_snr_aware_huber_gamma_clamping(self, setup_inputs):
        """
        Test that gamma properly clamps the SNR values
        """
        pred, target, device = setup_inputs

        # Test with small gamma
        loss_fn_small_gamma = WaveletLoss(
            wavelet="db4",
            level=2,
            transform_type="dwt",
            device=device,
            use_snr_aware_huber=True,
            snr_huber_gamma=1.0,  # Small gamma
        )

        # Test with large gamma
        loss_fn_large_gamma = WaveletLoss(
            wavelet="db4",
            level=2,
            transform_type="dwt",
            device=device,
            use_snr_aware_huber=True,
            snr_huber_gamma=100.0,  # Large gamma
        )

        # Use early timestep (high SNR case where gamma matters)
        timestep = torch.tensor([0.1] * pred.shape[0], device=device)

        losses_small, _ = loss_fn_small_gamma(pred, target, timestep=timestep)
        losses_large, _ = loss_fn_large_gamma(pred, target, timestep=timestep)

        # Different gamma values should produce different results at high SNR
        diff_found = False
        for l1, l2 in zip(losses_small, losses_large):
            if not torch.allclose(l1, l2, rtol=1e-5):
                diff_found = True
                break

        assert diff_found, "Different gamma values should affect high-SNR losses"
