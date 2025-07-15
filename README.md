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

# Create a loss function
loss_fn = WaveletLoss()

# Example usage in a training loop
prediction = torch.randn(10, 1)
target = torch.randn(10, 1)

loss = loss_fn(prediction, target)
```

## Features

- Discrete Wavelet Transform (DWT)
- Quadrature Wavelet Transform (QWT)
- Stationary Wavelet Transform (SWT)
- Wavelet-based loss calculations

## Development

- Run tests: `uv run pytest`
- Python 3.10+ required
