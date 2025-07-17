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
            losses, component_losses = loss_fn(pred, target)

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
        Test LL level threshold handling functionality
        """
        pred, target, device = setup_inputs

        # Test with different ll_level_threshold values
        test_thresholds = [None, 1, 2]
        
        for threshold in test_thresholds:
            loss_fn = WaveletLoss(
                wavelet="db4",
                level=3,
                transform_type="dwt",
                device=device,
                ll_level_threshold=threshold,
            )
            
            losses, metrics = loss_fn(pred, target)
            
            # Basic functionality check
            assert len(losses) > 0, "Should produce some losses"
            
            # If threshold is set, verify LL bands at lower levels are handled
            if threshold is not None:
                # Check that metrics are calculated correctly
                assert isinstance(metrics, dict), "Metrics should be a dictionary"
                
                # Test the process_band method directly for LL bands
                pred_coeffs = loss_fn.transform.decompose(pred, loss_fn.level)
                target_coeffs = loss_fn.transform.decompose(target, loss_fn.level)
                base_weight = torch.ones((pred.shape[0]), device=device)
                
                # Test levels within threshold
                for i in range(min(threshold, loss_fn.level)):
                    band_loss, pred_band, target_band, band_metrics = loss_fn.process_band(
                        pred_coeffs, target_coeffs, "ll", i, base_weight=base_weight
                    )
                    
                    # For LL bands within threshold, should return zero loss
                    assert torch.all(band_loss == 0.0), f"LL band at level {i+1} should have zero loss"
                    assert torch.all(pred_band == 0), f"LL pred at level {i+1} should be zero"
                    assert torch.all(target_band == 0), f"LL target at level {i+1} should be zero"
                    assert band_metrics == {}, f"LL band metrics at level {i+1} should be empty"
                
                # Test levels beyond threshold (if any)
                for i in range(threshold, loss_fn.level):
                    band_loss, pred_band, target_band, band_metrics = loss_fn.process_band(
                        pred_coeffs, target_coeffs, "ll", i, base_weight=base_weight
                    )
                    
                    # For LL bands beyond threshold, should have normal processing
                    assert not torch.all(band_loss == 0.0), f"LL band at level {i+1} should have non-zero loss"
                    assert not torch.all(pred_band == 0), f"LL pred at level {i+1} should not be all zeros"

    def test_pad_tensors_function(self, setup_inputs):
        """
        Test the _pad_tensors function directly
        """
        pred, target, device = setup_inputs
        
        loss_fn = WaveletLoss(
            wavelet="db4",
            level=2,
            transform_type="dwt",
            device=device,
        )
        
        # Create tensors with different sizes to test padding
        tensor1 = torch.randn(2, 3, 32, 32, device=device)
        tensor2 = torch.randn(2, 3, 16, 16, device=device)
        tensor3 = torch.randn(2, 3, 24, 20, device=device)
        
        tensors = [tensor1, tensor2, tensor3]
        
        # Test padding
        padded_tensors = loss_fn._pad_tensors(tensors)
        
        # Check that all tensors have the same size
        expected_h = max(t.shape[2] for t in tensors)  # 32
        expected_w = max(t.shape[3] for t in tensors)  # 32
        
        assert len(padded_tensors) == len(tensors), "Should return same number of tensors"
        
        for i, padded in enumerate(padded_tensors):
            assert padded.shape[0] == tensors[i].shape[0], f"Batch dimension should be preserved for tensor {i}"
            assert padded.shape[1] == tensors[i].shape[1], f"Channel dimension should be preserved for tensor {i}"
            assert padded.shape[2] == expected_h, f"Height should be padded to {expected_h} for tensor {i}"
            assert padded.shape[3] == expected_w, f"Width should be padded to {expected_w} for tensor {i}"
        
        # Test with tensors of same size (no padding needed)
        same_size_tensors = [
            torch.randn(2, 3, 16, 16, device=device),
            torch.randn(2, 3, 16, 16, device=device),
        ]
        
        padded_same = loss_fn._pad_tensors(same_size_tensors)
        
        for i, (original, padded) in enumerate(zip(same_size_tensors, padded_same)):
            assert torch.equal(original, padded), f"Tensor {i} should remain unchanged when no padding needed"
        
        # Test that padding preserves original content
        # The original content should be in the top-left corner
        original_tensor = torch.randn(1, 1, 8, 8, device=device)
        padded_result = loss_fn._pad_tensors([original_tensor, torch.randn(1, 1, 16, 16, device=device)])
        
        # Check that original content is preserved in top-left
        assert torch.equal(
            padded_result[0][:, :, :8, :8], 
            original_tensor
        ), "Original content should be preserved in top-left corner"
        
        # Check that padding areas are zero
        assert torch.all(padded_result[0][:, :, 8:, :] == 0), "Bottom padding should be zero"
        assert torch.all(padded_result[0][:, :, :, 8:] == 0), "Right padding should be zero"

    def test_pad_tensors_with_high_frequency_metrics(self, setup_inputs):
        """
        Test _pad_tensors in the context of high frequency metrics calculation
        """
        pred, target, device = setup_inputs
        
        loss_fn = WaveletLoss(
            wavelet="db4",
            level=2,
            transform_type="dwt",
            device=device,
            metrics=True,  # Enable metrics to trigger pad_tensors usage
        )
        
        # Run forward pass to trigger the pad_tensors usage
        losses, metrics = loss_fn(pred, target)
        
        # Check that avg_hf metrics are calculated (which uses pad_tensors)
        assert "avg_hf_pred" in metrics, "avg_hf_pred metric should be present"
        assert "avg_hf_target" in metrics, "avg_hf_target metric should be present"
        
        # The metrics should be reasonable values
        assert isinstance(metrics["avg_hf_pred"], (int, float)), "avg_hf_pred should be numeric"
        assert isinstance(metrics["avg_hf_target"], (int, float)), "avg_hf_target should be numeric"
        
        # Test that the metrics are computed correctly by comparing with direct calculation
        pred_coeffs = loss_fn.transform.decompose(pred, loss_fn.level)
        target_coeffs = loss_fn.transform.decompose(target, loss_fn.level)
        
        # Manually compute what should happen
        combined_hf_pred = []
        combined_hf_target = []
        
        for band in ["lh", "hl", "hh"]:
            for i in range(loss_fn.level):
                combined_hf_pred.append(pred_coeffs[band][i])
                combined_hf_target.append(target_coeffs[band][i])
        
        # This should match what happens in calculate_avg_high_frequency
        padded_pred = loss_fn._pad_tensors(combined_hf_pred)
        padded_target = loss_fn._pad_tensors(combined_hf_target)
        
        combined_pred = torch.cat(padded_pred, dim=1)
        combined_target = torch.cat(padded_target, dim=1)
        
        expected_avg_pred = combined_pred.mean().item()
        expected_avg_target = combined_target.mean().item()
        
        # Check that the computed metrics match our manual calculation
        assert abs(metrics["avg_hf_pred"] - expected_avg_pred) < 1e-6, "avg_hf_pred should match manual calculation"
        assert abs(metrics["avg_hf_target"] - expected_avg_target) < 1e-6, "avg_hf_target should match manual calculation"


# Remaining previous tests...
