import os

import numpy as np
import pywt
import pytest
import torch

from wavelet_transform import DiscreteWaveletTransform
from wavelet_transform import WaveletTransform
from wavelet_transform.transform import dwt_single_level


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)


class TestDiscreteWaveletTransform:
    @pytest.fixture
    def dwt(self):
        """Fixture to create a DiscreteWaveletTransform instance."""
        return DiscreteWaveletTransform(wavelet="db4", device=torch.device("cpu"))

    @pytest.fixture
    def sample_image(self):
        """Fixture to create a sample image tensor for testing."""
        # Create a 2x2x32x32 sample image (batch x channels x height x width)
        return torch.randn(2, 2, 32, 32)

    def test_initialization(self, dwt):
        """Test proper initialization of DWT with wavelet filters."""
        # Check if the base wavelet filters are initialized
        assert hasattr(dwt, "dec_lo") and dwt.dec_lo is not None
        assert hasattr(dwt, "dec_hi") and dwt.dec_hi is not None

        # Check filter dimensions for db4
        assert dwt.dec_lo.size(0) == 8
        assert dwt.dec_hi.size(0) == 8

    def test_dwt_single_level_pywt_parity(self, dwt: DiscreteWaveletTransform):
        """Test single-level DWT decomposition matches pywt.dwt2(mode='zero')."""
        torch.manual_seed(0)
        x = torch.randn(2, 2, 32, 32, dtype=torch.float64)

        # Use decompose (delegates to CustomDWTBackend) for level=1
        result = dwt.decompose(x, level=1)
        ll = result["ll"][0]
        lh = result["lh"][0]
        hl = result["hl"][0]
        hh = result["hh"][0]

        # All subbands have the same shape
        assert ll.shape == lh.shape == hl.shape == hh.shape

        # Batch and channel dimensions preserved
        assert ll.shape[0] == x.shape[0]
        assert ll.shape[1] == x.shape[1]

        # Correctness: each subband must match pywt.dwt2(mode='zero')
        # lh=cH, hl=cV, hh=cD per CustomDWTBackend convention
        cA, (cH, cV, cD) = pywt.dwt2(x[0, 0].numpy(), "db4", mode="zero")
        assert _rel(ll[0, 0].numpy(), cA) < 1e-6, f"ll relerr={_rel(ll[0, 0].numpy(), cA):.2e}"
        assert _rel(lh[0, 0].numpy(), cH) < 1e-6, f"lh relerr={_rel(lh[0, 0].numpy(), cH):.2e}"
        assert _rel(hl[0, 0].numpy(), cV) < 1e-6, f"hl relerr={_rel(hl[0, 0].numpy(), cV):.2e}"
        assert _rel(hh[0, 0].numpy(), cD) < 1e-6, f"hh relerr={_rel(hh[0, 0].numpy(), cD):.2e}"

    def test_decompose_structure(self, dwt, sample_image):
        """Test structure of decomposition result."""
        x = sample_image
        level = 2

        # Perform decomposition
        result = dwt.decompose(x, level=level)

        # Check structure of result
        bands = ["ll", "lh", "hl", "hh"]

        for band in bands:
            assert band in result
            assert len(result[band]) == level

    def test_decompose_shapes_pywt_parity(self, dwt: DiscreteWaveletTransform):
        """Test shapes of decomposition coefficients match pywt.dwt2 output shapes."""
        torch.manual_seed(1)
        x = torch.randn(2, 2, 32, 32, dtype=torch.float64)
        level = 3

        result = dwt.decompose(x, level=level)

        # Compute expected shapes using pywt for each level
        current_h, current_w = x.shape[2], x.shape[3]
        for lvl in range(level):
            cA, _ = pywt.dwt2(np.zeros((current_h, current_w)), "db4", mode="zero")
            exp_h, exp_w = cA.shape
            expected_shape = (x.shape[0], x.shape[1], exp_h, exp_w)

            for band in ["ll", "lh", "hl", "hh"]:
                assert result[band][lvl].shape == expected_shape, (
                    f"Level {lvl}, {band}: expected {expected_shape}, got {result[band][lvl].shape}"
                )

            # Next level input is the ll output
            current_h, current_w = exp_h, exp_w

        for band in ["ll", "lh", "hl", "hh"]:
            assert len(result[band]) == level, f"Expected {level} levels for {band}, got {len(result[band])}"

    def test_decompose_different_levels(self, dwt, sample_image):
        """Test decomposition with different levels."""
        x = sample_image

        # Test with different levels
        for level in [1, 2, 3]:
            result = dwt.decompose(x, level=level)

            # Check number of coefficients at each level
            for band in ["ll", "lh", "hl", "hh"]:
                assert len(result[band]) == level

    @pytest.mark.parametrize(
        "wavelet",
        [
            "db1",
            "db4",
            "sym4",
            "sym7",
            "haar",
            "coif3",
            "bior3.3",
            "rbio1.3",
            "dmey",
        ],
    )
    def test_different_wavelets(self, sample_image, wavelet):
        """Test DWT with different wavelet families."""
        dwt = DiscreteWaveletTransform(wavelet=wavelet, device=torch.device("cpu"))

        # Simple test that decomposition works with this wavelet
        result = dwt.decompose(sample_image, level=1)

        # Basic structure check
        assert all(band in result for band in ["ll", "lh", "hl", "hh"])

    @pytest.mark.parametrize(
        "wavelet",
        [
            "db1",
            "db4",
            "sym4",
            "sym7",
            "haar",
            "coif3",
            "bior3.3",
            "rbio1.3",
            "dmey",
        ],
    )
    def test_different_wavelets_different_sizes_pywt_parity(self, wavelet):
        """Test DWT with different wavelet families and input sizes matches pywt.dwt2."""
        torch.manual_seed(2)
        dwt = DiscreteWaveletTransform(wavelet=wavelet, device=torch.device("cpu"))

        test_sizes = [(8, 8), (32, 32), (64, 64)]

        for h, w in test_sizes:
            x = torch.randn(2, 2, h, w, dtype=torch.float64)
            result = dwt.decompose(x, level=1)

            # Shape must match pywt
            cA, (cH, cV, cD) = pywt.dwt2(x[0, 0].numpy(), wavelet, mode="zero")
            exp_shape = (2, 2, *cA.shape)
            assert result["ll"][0].shape == exp_shape, (
                f"wavelet={wavelet}, input ({h},{w}): expected {exp_shape}, got {result['ll'][0].shape}"
            )

            # Value parity (check first sample/channel)
            assert _rel(result["ll"][0][0, 0].numpy(), cA) < 1e-6
            assert _rel(result["lh"][0][0, 0].numpy(), cH) < 1e-6
            assert _rel(result["hl"][0][0, 0].numpy(), cV) < 1e-6
            assert _rel(result["hh"][0][0, 0].numpy(), cD) < 1e-6

    @pytest.mark.parametrize("shape", [(2, 3, 64, 64), (1, 1, 128, 128), (4, 3, 120, 160)])
    def test_different_input_shapes_pywt_parity(self, shape):
        """Test DWT with different input shapes matches pywt.dwt2(mode='zero')."""
        torch.manual_seed(3)
        dwt = DiscreteWaveletTransform(wavelet="db4", device=torch.device("cpu"))
        x = torch.randn(*shape, dtype=torch.float64)

        result = dwt.decompose(x, level=1)

        # Expected shapes from pywt
        cA, (cH, cV, cD) = pywt.dwt2(x[0, 0].numpy(), "db4", mode="zero")
        expected_shape = (shape[0], shape[1], *cA.shape)

        for band in ["ll", "lh", "hl", "hh"]:
            assert result[band][0].shape == expected_shape, (
                f"For input {shape}, {band}: expected {expected_shape}, got {result[band][0].shape}"
            )

        # Value parity for first sample/channel
        assert _rel(result["ll"][0][0, 0].numpy(), cA) < 1e-6
        assert _rel(result["lh"][0][0, 0].numpy(), cH) < 1e-6
        assert _rel(result["hl"][0][0, 0].numpy(), cV) < 1e-6
        assert _rel(result["hh"][0][0, 0].numpy(), cD) < 1e-6

    def test_device_support(self):
        """Test that DWT supports CPU and GPU (if available)."""
        # Test CPU
        cpu_device = torch.device("cpu")
        dwt_cpu = DiscreteWaveletTransform(device=cpu_device)
        assert dwt_cpu.dec_lo.device == cpu_device
        assert dwt_cpu.dec_hi.device == cpu_device

        # Test GPU if available and WAVELET_TEST_CUDA is set
        if os.environ.get("WAVELET_TEST_CUDA") and torch.cuda.is_available():
            gpu_device = torch.device("cuda:0")
            dwt_gpu = DiscreteWaveletTransform(device=gpu_device)
            assert dwt_gpu.dec_lo.device == gpu_device
            assert dwt_gpu.dec_hi.device == gpu_device

    def test_legacy_helper_band_convention(self, dwt: DiscreteWaveletTransform):
        """dwt_single_level must follow the pywt convention: lh = high-pass on H, hl = high-pass on W.

        An input that varies only along H (constant along W) has no W-direction
        detail, so its detail energy must land in lh, and hl must be ~0.
        """
        torch.manual_seed(4)
        # Varies along H only: each row is a constant
        row_signal = torch.randn(32, 1)
        x_h = row_signal.expand(32, 32).reshape(1, 1, 32, 32).contiguous()

        _, lh, hl, _ = dwt_single_level(x_h, dwt.dec_lo, dwt.dec_hi)
        assert lh.abs().max() > 1e-2, "lh must carry H-direction detail (high-pass on H)"
        assert hl.abs().max() < 1e-4, "hl must be ~0 for a signal constant along W"

        # Symmetric check: varies along W only
        col_signal = torch.randn(1, 32)
        x_w = col_signal.expand(32, 32).reshape(1, 1, 32, 32).contiguous()

        _, lh, hl, _ = dwt_single_level(x_w, dwt.dec_lo, dwt.dec_hi)
        assert hl.abs().max() > 1e-2, "hl must carry W-direction detail (high-pass on W)"
        assert lh.abs().max() < 1e-4, "lh must be ~0 for a signal constant along H"

    def test_base_class_abstract_method(self):
        """Test that base class requires implementation of decompose."""
        base_transform = WaveletTransform(wavelet="db4", device=torch.device("cpu"))

        with pytest.raises(NotImplementedError):
            base_transform.decompose(torch.randn(2, 2, 32, 32), 2)
