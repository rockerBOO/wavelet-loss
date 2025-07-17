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

    # Apply filter to columns
    ll = F.conv2d(lo, dec_lo.view(1, 1, 1, -1), stride=(1, 2))
    lh = F.conv2d(lo, dec_hi.view(1, 1, 1, -1), stride=(1, 2))
    hl = F.conv2d(hi, dec_lo.view(1, 1, 1, -1), stride=(1, 2))
    hh = F.conv2d(hi, dec_hi.view(1, 1, 1, -1), stride=(1, 2))

    # Reshape back to batch format
    ll = ll.view(batch, channels, ll.shape[2], ll.shape[3]).to(x.device)
    lh = lh.view(batch, channels, lh.shape[2], lh.shape[3]).to(x.device)
    hl = hl.view(batch, channels, hl.shape[2], hl.shape[3]).to(x.device)
    hh = hh.view(batch, channels, hh.shape[2], hh.shape[3]).to(x.device)

    return ll, lh, hl, hh


class WaveletTransform:
    """Base class for wavelet transforms."""

    def __init__(self, wavelet="db4", device=torch.device("cpu")):
        """Initialize wavelet filters."""
        assert pywt.Wavelet is not None, "PyWavelets module not available. Please install `pip install PyWavelets`"

        # Create filters from wavelet
        wav = pywt.Wavelet(wavelet)
        self.dec_lo = torch.tensor(wav.dec_lo).to(device)
        self.dec_hi = torch.tensor(wav.dec_hi).to(device)

    def decompose(self, x: Tensor, level: int) -> dict[str, list[Tensor]]:
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("WaveletTransform subclasses must implement decompose method")


class DiscreteWaveletTransform(WaveletTransform):
    """Discrete Wavelet Transform (DWT) implementation."""

    def decompose(self, x: Tensor, level=1) -> dict[str, list[Tensor]]:
        """
        Perform multi-level DWT decomposition.

        Args:
            x: Input tensor [B, C, H, W]
            level: Number of decomposition levels

        Returns:
            Dictionary containing decomposition coefficients
        """
        bands: dict[str, list[Tensor]] = {
            "ll": [],
            "lh": [],
            "hl": [],
            "hh": [],
        }

        # Start low frequency with input
        ll = x

        for _ in range(level):
            ll, lh, hl, hh = self._dwt_single_level(ll)

            bands["lh"].append(lh)
            bands["hl"].append(hl)
            bands["hh"].append(hh)
            bands["ll"].append(ll)

        return bands

    def _dwt_single_level(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Perform single-level DWT decomposition."""
        return dwt_single_level(x, self.dec_lo, self.dec_hi)


class StationaryWaveletTransform(WaveletTransform):
    """Stationary Wavelet Transform (SWT) implementation."""

    def __init__(self, wavelet="db4", device=torch.device("cpu")):
        """Initialize wavelet filters."""
        super().__init__(wavelet, device)

        # Store original filters
        self.orig_dec_lo = self.dec_lo.clone()
        self.orig_dec_hi = self.dec_hi.clone()

    def decompose(self, x: Tensor, level=1) -> dict[str, list[Tensor]]:
        """Perform multi-level SWT decomposition."""
        bands = {
            "ll": [],
            "lh": [],
            "hl": [],
            "hh": [],
        }

        # Start with input as low frequency
        ll = x

        for j in range(level):
            # Get upsampled filters for current level
            dec_lo, dec_hi = self._get_filters_for_level(j)

            # Decompose current approximation
            ll, lh, hl, hh = self._swt_single_level(ll, dec_lo, dec_hi)

            # Store results in bands
            bands["ll"].append(ll)
            bands["lh"].append(lh)
            bands["hl"].append(hl)
            bands["hh"].append(hh)

            # No need to update ll explicitly as it's already the next approximation

        return bands

    def _get_filters_for_level(self, level: int) -> tuple[Tensor, Tensor]:
        """Get upsampled filters for the specified level."""
        if level == 0:
            return self.orig_dec_lo, self.orig_dec_hi

        # Calculate number of zeros to insert
        zeros = 2**level - 1

        # Create upsampled filters
        upsampled_dec_lo = torch.zeros(
            len(self.orig_dec_lo) + (len(self.orig_dec_lo) - 1) * zeros,
            device=self.orig_dec_lo.device,
        )
        upsampled_dec_hi = torch.zeros(
            len(self.orig_dec_hi) + (len(self.orig_dec_hi) - 1) * zeros,
            device=self.orig_dec_hi.device,
        )

        # Insert original coefficients with zeros in between
        upsampled_dec_lo[:: zeros + 1] = self.orig_dec_lo
        upsampled_dec_hi[:: zeros + 1] = self.orig_dec_hi

        return upsampled_dec_lo, upsampled_dec_hi

    def _swt_single_level(self, x: Tensor, dec_lo: Tensor, dec_hi: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Perform single-level SWT decomposition with 1D convolutions."""
        batch, channels, height, width = x.shape

        # Prepare output tensors
        ll = torch.zeros((batch, channels, height, width), device=x.device)
        lh = torch.zeros((batch, channels, height, width), device=x.device)
        hl = torch.zeros((batch, channels, height, width), device=x.device)
        hh = torch.zeros((batch, channels, height, width), device=x.device)

        # Prepare 1D filter kernels
        dec_lo_1d = dec_lo.view(1, 1, -1)
        dec_hi_1d = dec_hi.view(1, 1, -1)
        pad_len = dec_lo.size(0) - 1

        for b in range(batch):
            for c in range(channels):
                # Extract single channel/batch and reshape for 1D convolution
                x_bc = x[b, c]  # Shape: [height, width]

                # Process rows with 1D convolution
                # Reshape to [width, 1, height] for treating each row as a batch
                x_rows = x_bc.transpose(0, 1).unsqueeze(1)  # Shape: [width, 1, height]

                # Pad for circular convolution
                x_rows_padded = F.pad(x_rows, (pad_len, 0), mode="circular")

                # Apply filters to rows
                x_lo_rows = F.conv1d(x_rows_padded, dec_lo_1d)  # [width, 1, height]
                x_hi_rows = F.conv1d(x_rows_padded, dec_hi_1d)  # [width, 1, height]

                # Reshape and transpose back
                x_lo_rows = x_lo_rows.squeeze(1).transpose(0, 1)  # [height, width]
                x_hi_rows = x_hi_rows.squeeze(1).transpose(0, 1)  # [height, width]

                # Process columns with 1D convolution
                # Reshape for column filtering (no transpose needed)
                x_lo_cols = x_lo_rows.unsqueeze(1)  # [height, 1, width]
                x_hi_cols = x_hi_rows.unsqueeze(1)  # [height, 1, width]

                # Pad for circular convolution
                x_lo_cols_padded = F.pad(x_lo_cols, (pad_len, 0), mode="circular")
                x_hi_cols_padded = F.pad(x_hi_cols, (pad_len, 0), mode="circular")

                # Apply filters to columns
                ll[b, c] = F.conv1d(x_lo_cols_padded, dec_lo_1d).squeeze(1)  # [height, width]
                lh[b, c] = F.conv1d(x_lo_cols_padded, dec_hi_1d).squeeze(1)  # [height, width]
                hl[b, c] = F.conv1d(x_hi_cols_padded, dec_lo_1d).squeeze(1)  # [height, width]
                hh[b, c] = F.conv1d(x_hi_cols_padded, dec_hi_1d).squeeze(1)  # [height, width]

        return ll, lh, hl, hh


class QuaternionWaveletTransform(WaveletTransform):
    """
    Quaternion Wavelet Transform implementation.
    Combines real DWT with three Hilbert transforms along x, y, and xy axes.
    """

    def __init__(self, wavelet="db4", device=torch.device("cpu")):
        """Initialize wavelet filters and Hilbert transforms."""
        super().__init__(wavelet, device)

        # Register Hilbert transform filters
        self.register_hilbert_filters(device)

    def register_hilbert_filters(self, device):
        """Create and register Hilbert transform filters."""
        # Create x-axis Hilbert filter
        self.hilbert_x = self._create_hilbert_filter("x").to(device)

        # Create y-axis Hilbert filter
        self.hilbert_y = self._create_hilbert_filter("y").to(device)

        # Create xy (diagonal) Hilbert filter
        self.hilbert_xy = self._create_hilbert_filter("xy").to(device)

    def _create_hilbert_filter(self, direction):
        """Create a Hilbert transform filter for the specified direction."""
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
        """Apply Hilbert transform in specified direction with correct padding."""
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
        """Perform single-level DWT decomposition."""
        return dwt_single_level(x, self.dec_lo, self.dec_hi)
