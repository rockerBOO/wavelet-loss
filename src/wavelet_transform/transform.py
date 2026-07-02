from torch import Tensor
import torch
import torch.nn.functional as F
import pywt


def dwt_single_level(x: Tensor, dec_lo: Tensor, dec_hi: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Perform single-level DWT decomposition.

    Args:
        x: Input tensor [B, C, H, W]
        dec_lo: Low-pass decomposition filter
        dec_hi: High-pass decomposition filter

    Returns:
        Tuple of (ll, lh, hl, hh) decomposed tensors
    """
    batch, channels, height, width = x.shape
    x = x.view(batch * channels, 1, height, width)

    # Calculate proper padding for the filter size
    filter_size = dec_lo.size(0)
    pad_size = filter_size // 2

    # Pad for proper convolution
    try:
        x_pad = F.pad(x, (pad_size,) * 4, mode="reflect")
    except RuntimeError:
        # Fallback for very small tensors
        x_pad = F.pad(x, (pad_size,) * 4, mode="constant")

    # Apply filter to rows
    lo = F.conv2d(x_pad, dec_lo.view(1, 1, -1, 1), stride=(2, 1))
    hi = F.conv2d(x_pad, dec_hi.view(1, 1, -1, 1), stride=(2, 1))

    # Apply filter to columns (pywt convention: lh = hi on H / lo on W, hl = lo on H / hi on W)
    ll = F.conv2d(lo, dec_lo.view(1, 1, 1, -1), stride=(1, 2))
    lh = F.conv2d(hi, dec_lo.view(1, 1, 1, -1), stride=(1, 2))
    hl = F.conv2d(lo, dec_hi.view(1, 1, 1, -1), stride=(1, 2))
    hh = F.conv2d(hi, dec_hi.view(1, 1, 1, -1), stride=(1, 2))

    # Reshape back to batch format
    ll = ll.view(batch, channels, ll.shape[2], ll.shape[3]).to(x.device)
    lh = lh.view(batch, channels, lh.shape[2], lh.shape[3]).to(x.device)
    hl = hl.view(batch, channels, hl.shape[2], hl.shape[3]).to(x.device)
    hh = hh.view(batch, channels, hh.shape[2], hh.shape[3]).to(x.device)

    return ll, lh, hl, hh


class WaveletTransform:
    """
    Base class for wavelet transforms.

    Provides common functionality for wavelet decomposition operations
    including filter initialization from PyWavelets.

    Attributes:
        dec_lo: Low-pass decomposition filter tensor
        dec_hi: High-pass decomposition filter tensor
    """

    def __init__(self, wavelet="db4", device=torch.device("cpu")):
        """
        Initialize wavelet filters from PyWavelets.

        Args:
            wavelet: Wavelet name (e.g., 'db4', 'haar', 'sym4')
            device: Device to place tensors on

        Raises:
            AssertionError: If PyWavelets module is not available
        """
        assert pywt.Wavelet is not None, "PyWavelets module not available. Please install `pip install PyWavelets`"

        self.wavelet = wavelet

        # Create filters from wavelet
        wav = pywt.Wavelet(wavelet)
        self.dec_lo = torch.tensor(wav.dec_lo).to(device)
        self.dec_hi = torch.tensor(wav.dec_hi).to(device)

    def decompose(self, x: Tensor, level: int) -> dict[str, list[Tensor]]:
        """
        Abstract method for wavelet decomposition.

        Args:
            x: Input tensor [B, C, H, W]
            level: Number of decomposition levels

        Returns:
            Dictionary containing decomposition coefficients
            Format: {band: [level1, level2, ...]} where band ∈ {ll, lh, hl, hh}

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("WaveletTransform subclasses must implement decompose method")


class DiscreteWaveletTransform(WaveletTransform):
    """
    Discrete Wavelet Transform (DWT) implementation.

    Performs standard separable 2D DWT with downsampling at each level.
    Each level reduces spatial dimensions by factor of 2.

    Uses PyTorch convolutions for efficient GPU computation.
    """

    def decompose(self, x: Tensor, level=1) -> dict[str, list[Tensor]]:
        """
        Perform multi-level DWT decomposition.

        Delegates to CustomDWTBackend (mode='zero') which matches pywt.dwt2(mode='zero').

        Args:
            x: Input tensor [B, C, H, W]
            level: Number of decomposition levels

        Returns:
            Dictionary containing decomposition coefficients
            Format: {band: [level1, level2, ...]} where:
            - 'll': Approximation coefficients (pywt cA)
            - 'lh': Horizontal detail, high-pass on H / low-pass on W (pywt cH)
            - 'hl': Vertical detail, low-pass on H / high-pass on W (pywt cV)
            - 'hh': Diagonal detail, high-pass on both (pywt cD)
        """
        from .backends import CustomDWTBackend

        return CustomDWTBackend(self.wavelet, mode="zero", device=x.device).decompose(x, level)

    def _dwt_single_level(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Perform single-level DWT decomposition.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Tuple of (ll, lh, hl, hh) decomposed tensors with half spatial dimensions
        """
        return dwt_single_level(x, self.dec_lo, self.dec_hi)


class StationaryWaveletTransform(WaveletTransform):
    """Undecimated (à-trous) SWT, vectorized, matching pywt.swt2 (periodic boundary)."""

    def decompose(self, x: Tensor, level: int = 1) -> dict[str, list[Tensor]]:
        bands: dict[str, list[Tensor]] = {"ll": [], "lh": [], "hl": [], "hh": []}
        lo0 = self.dec_lo.to(device=x.device, dtype=x.dtype).flip(0)
        hi0 = self.dec_hi.to(device=x.device, dtype=x.dtype).flip(0)
        k0 = lo0.numel()
        b, c, h, w = x.shape
        ll = x
        for j in range(level):
            lof = self._upsample(lo0, 2**j)
            hif = self._upsample(hi0, 2**j)
            k = lof.numel()
            # Roll to center the à-trous filter: use ORIGINAL filter length k0, scaled by 2**j (NOT the upsampled length).
            shift = -(k0 // 2) * (2**j)
            xbc = ll.reshape(b * c, 1, h, w)
            lo_r = self._cdim(xbc, lof, 2, k, shift)
            hi_r = self._cdim(xbc, hif, 2, k, shift)
            ll = self._cdim(lo_r, lof, 3, k, shift).reshape(b, c, h, w)
            lh = self._cdim(hi_r, lof, 3, k, shift).reshape(b, c, h, w)
            hl = self._cdim(lo_r, hif, 3, k, shift).reshape(b, c, h, w)
            hh = self._cdim(hi_r, hif, 3, k, shift).reshape(b, c, h, w)
            bands["ll"].append(ll)
            bands["lh"].append(lh)
            bands["hl"].append(hl)
            bands["hh"].append(hh)
        return bands

    @staticmethod
    def _upsample(f: Tensor, up: int) -> Tensor:
        if up == 1:
            return f
        out = torch.zeros((f.numel() - 1) * up + 1, dtype=f.dtype, device=f.device)
        out[::up] = f
        return out

    @staticmethod
    def _cdim(t: Tensor, f: Tensor, dim: int, k: int, shift: int) -> Tensor:
        pad = (0, 0, k - 1, 0) if dim == 2 else (k - 1, 0, 0, 0)
        tp = F.pad(t, pad, mode="circular")
        weight = f.view(1, 1, -1, 1) if dim == 2 else f.view(1, 1, 1, -1)
        return torch.roll(F.conv2d(tp, weight), shift, dims=dim)


class QuaternionWaveletTransform(WaveletTransform):
    """
    Quaternion Wavelet Transform implementation.
    Combines real DWT with three Hilbert transforms along x, y, and xy axes.
    """

    def __init__(self, wavelet="db4", device=torch.device("cpu")):
        """
        Initialize wavelet filters and Hilbert transforms.

        Args:
            wavelet: Wavelet name (e.g., 'db4', 'haar', 'sym4')
            device: Device to place tensors on
        """
        super().__init__(wavelet, device)

        # Register Hilbert transform filters
        self.register_hilbert_filters(device)

    def register_hilbert_filters(self, device):
        """
        Create and register Hilbert transform filters for quaternion components.

        Args:
            device: Device to place filter tensors on

        Notes:
            - Creates filters for x, y, and xy (diagonal) directions
            - Filters are approximations suitable for image processing
        """
        # Create x-axis Hilbert filter
        self.hilbert_x = self._create_hilbert_filter("x").to(device)

        # Create y-axis Hilbert filter
        self.hilbert_y = self._create_hilbert_filter("y").to(device)

        # Create xy (diagonal) Hilbert filter
        self.hilbert_xy = self._create_hilbert_filter("xy").to(device)

    def _create_hilbert_filter(self, direction):
        """
        Create a Hilbert transform filter for the specified direction.

        Args:
            direction: Filter direction ('x', 'y', or 'xy')

        Returns:
            2D convolution filter tensor [1, 1, H, W]

        Notes:
            - 'x': Horizontal Hilbert transform (anti-symmetric along x-axis)
            - 'y': Vertical Hilbert transform (anti-symmetric along y-axis)
            - 'xy': Diagonal Hilbert transform (point reflection symmetry)
            - Filters use approximations suitable for discrete images
        """
        if direction == "x":
            # Horizontal Hilbert filter (approximation)
            filt = torch.tensor(
                [
                    [-0.0106, -0.0329, -0.0308, 0.0000, 0.0308, 0.0329, 0.0106],
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                ]
            ).float()
            return filt.unsqueeze(0).unsqueeze(0)

        elif direction == "y":
            # Vertical Hilbert filter (approximation)
            filt = torch.tensor(
                [
                    [-0.0106, 0.0000],
                    [-0.0329, 0.0000],
                    [-0.0308, 0.0000],
                    [0.0000, 0.0000],
                    [0.0308, 0.0000],
                    [0.0329, 0.0000],
                    [0.0106, 0.0000],
                ]
            ).float()
            return filt.unsqueeze(0).unsqueeze(0)

        else:  # 'xy' - diagonal
            # Diagonal Hilbert filter (approximation)
            filt = torch.tensor(
                [
                    [-0.0011, -0.0035, -0.0033, 0.0000, 0.0033, 0.0035, 0.0011],
                    [-0.0035, -0.0108, -0.0102, 0.0000, 0.0102, 0.0108, 0.0035],
                    [-0.0033, -0.0102, -0.0095, 0.0000, 0.0095, 0.0102, 0.0033],
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.0033, 0.0102, 0.0095, 0.0000, -0.0095, -0.0102, -0.0033],
                    [0.0035, 0.0108, 0.0102, 0.0000, -0.0102, -0.0108, -0.0035],
                    [0.0011, 0.0035, 0.0033, 0.0000, -0.0033, -0.0035, -0.0011],
                ]
            ).float()
            return filt.unsqueeze(0).unsqueeze(0)

    def _apply_hilbert(self, x, direction):
        """
        Apply Hilbert transform in specified direction with correct padding.

        Args:
            x: Input tensor [B, C, H, W]
            direction: Transform direction ('x', 'y', or 'xy')

        Returns:
            Hilbert-transformed tensor with same dimensions as input

        Notes:
            - Uses reflective padding to minimize boundary artifacts
            - Automatically handles even/odd filter sizes
            - Crops output to match input dimensions exactly
        """
        batch, channels, height, width = x.shape

        x_flat = x.reshape(batch * channels, 1, height, width)

        # Get the appropriate filter
        if direction == "x":
            h_filter = self.hilbert_x
        elif direction == "y":
            h_filter = self.hilbert_y
        else:  # 'xy'
            h_filter = self.hilbert_xy

        # Calculate correct padding based on filter dimensions
        # For 'same' padding: pad = (filter_size - 1) / 2
        filter_h, filter_w = h_filter.shape[2:]
        pad_h = (filter_h - 1) // 2
        pad_w = (filter_w - 1) // 2

        # For even-sized filters, we need to adjust padding
        pad_h_left, pad_h_right = pad_h, pad_h
        pad_w_left, pad_w_right = pad_w, pad_w

        if filter_h % 2 == 0:  # Even height
            pad_h_right += 1
        if filter_w % 2 == 0:  # Even width
            pad_w_right += 1

        # Apply padding with possibly asymmetric padding
        x_pad = F.pad(x_flat, (pad_w_left, pad_w_right, pad_h_left, pad_h_right), mode="reflect")

        # Apply convolution
        x_hilbert = F.conv2d(x_pad, h_filter)

        # Ensure output dimensions match input dimensions
        if x_hilbert.shape[2:] != (height, width):
            # Need to crop or pad to match original dimensions
            # For this case, center crop is appropriate
            if x_hilbert.shape[2] > height:
                # Crop height
                diff = x_hilbert.shape[2] - height
                start = diff // 2
                x_hilbert = x_hilbert[:, :, start : start + height, :]

            if x_hilbert.shape[3] > width:
                # Crop width
                diff = x_hilbert.shape[3] - width
                start = diff // 2
                x_hilbert = x_hilbert[:, :, :, start : start + width]

        # Reshape back to original format
        return x_hilbert.reshape(batch, channels, height, width)

    def decompose(self, x: Tensor, level=1) -> dict[str, list[Tensor]]:
        """
        Perform multi-level wavelet decomposition on a single tensor.

        Args:
            x: Input tensor [B, C, H, W]
            level: Number of decomposition levels

        Returns:
            Dictionary containing wavelet coefficients
            Format: {band: [level1, level2, ...]}
            where band ∈ {ll, lh, hl, hh}
        """
        # Initialize result dictionary
        coeffs = {"ll": [], "lh": [], "hl": [], "hh": []}

        # Initialize with input
        ll = x

        # Perform wavelet decomposition for each level
        for i in range(level):
            ll, lh, hl, hh = self._dwt_single_level(ll)

            # Store coefficients for this level
            coeffs["ll"].append(ll)
            coeffs["lh"].append(lh)
            coeffs["hl"].append(hl)
            coeffs["hh"].append(hh)

        return coeffs

    def decompose_quaternion(self, x: Tensor, level=1) -> dict[str, dict[str, list[Tensor]]]:
        """
        Perform multi-level QWT decomposition with quaternion components.

        Args:
            x: Input tensor [B, C, H, W]
            level: Number of decomposition levels

        Returns:
            Dictionary containing quaternion wavelet coefficients
            Format: {component: {band: [level1, level2, ...]}}
            where component ∈ {r, i, j, k} and band ∈ {ll, lh, hl, hh}

        Notes:
            - Real component (r) uses standard DWT of input
            - Imaginary components (i, j, k) use DWT of Hilbert transforms
            - Provides phase information useful for texture analysis
        """
        # Generate Hilbert transforms of the input
        x_hilbert_x = self._apply_hilbert(x, "x")
        x_hilbert_y = self._apply_hilbert(x, "y")
        x_hilbert_xy = self._apply_hilbert(x, "xy")

        # Perform decomposition for each quaternion component
        qwt_coeffs = {
            "r": self.decompose(x, level),  # Real part
            "i": self.decompose(x_hilbert_x, level),  # Imaginary part (x-Hilbert)
            "j": self.decompose(x_hilbert_y, level),  # Imaginary part (y-Hilbert)
            "k": self.decompose(x_hilbert_xy, level),  # Imaginary part (xy-Hilbert)
        }

        return qwt_coeffs

    def _dwt_single_level(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Perform single-level DWT decomposition for quaternion component.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Tuple of (ll, lh, hl, hh) decomposed tensors with half spatial dimensions
        """
        return dwt_single_level(x, self.dec_lo, self.dec_hi)
