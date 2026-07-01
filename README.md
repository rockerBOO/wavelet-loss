# Wavelet Loss

A Python library for wavelet-based loss calculations in machine learning.

## Installation

```bash
pip install git+https://github.com/rockerBOO/wavelet-loss.git
```

## Usage

## Quick Start

```python
import torch
from wavelet_loss import WaveletLoss

# Frequency-aware loss for VAE latents [B, C, H, W]
loss_fn = WaveletLoss(wavelet="db4", level=2, transform_type="dwt")

prediction = torch.randn(2, 4, 32, 32, requires_grad=True)
target = torch.randn(2, 4, 32, 32)

loss, metrics = loss_fn(prediction, target)  # scalar loss (reduce=True default)
loss.backward()
```

## Features

- Discrete Wavelet Transform (DWT)
- Quadrature Wavelet Transform (QWT)
- Stationary Wavelet Transform (SWT)
- Wavelet-based loss calculations

## Development

- Run tests: `uv run pytest`
- Python 3.10+ required
