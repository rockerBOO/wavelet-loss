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

### Diffusion / flow-matching training

Pass the current timestep to fade out high-frequency loss at high noise levels:

```python
# Flow-matching sigmas in [0, 1] (default convention)
loss_fn = WaveletLoss(wavelet="db4", level=2)
loss, metrics = loss_fn(prediction, target, timestep=timesteps)

# DDPM-style integer timesteps require max_timestep=1000
loss_fn = WaveletLoss(wavelet="db4", level=2, max_timestep=1000)
```

See [docs/configurations.md](docs/configurations.md) for `timestep_cutoff`,
`timestep_transition_width`, and other options.

## Features

- Discrete Wavelet Transform (DWT)
- Quadrature Wavelet Transform (QWT)
- Stationary Wavelet Transform (SWT)
- Wavelet-based loss calculations
- Timestep-aware loss weighting for diffusion / flow-matching training

## Upgrading to 2.0

See [CHANGELOG.md](CHANGELOG.md) for breaking changes: `normalize_bands` now
defaults to `False`, `forward` returns a scalar by default (`reduce=True`),
metrics are opt-in (`metrics=True`), and timesteps are validated against
`max_timestep` (default `1.0`, flow-matching convention).

## Development

- Run tests: `uv run pytest`
- Python 3.10+ required
