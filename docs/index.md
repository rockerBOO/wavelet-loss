# Wavelet Loss Library

## Overview

The Wavelet Loss library provides advanced wavelet-based loss calculations for machine learning tasks, supporting multiple wavelet transforms and customizable loss configurations.

## Key Features

- Multiple Wavelet Transform Types
  - Discrete Wavelet Transform (DWT)
  - Stationary Wavelet Transform (SWT)
  - Quaternion Wavelet Transform (QWT)

- Flexible Loss Configuration
  - Custom band weights
  - Level-specific weighting
  - Selectable base loss functions
  - Timestep-aware weighting for diffusion / flow-matching training

## Quick Start

```python
import torch
from wavelet_loss import WaveletLoss

# Inputs must be 4D [B, C, H, W]
prediction = torch.randn(2, 4, 32, 32, requires_grad=True)
target = torch.randn(2, 4, 32, 32)

loss_fn = WaveletLoss(wavelet="db4", level=3)
loss, metrics = loss_fn(prediction, target)  # scalar loss, empty metrics dict by default
loss.backward()
```

For diffusion/flow-matching training, pass the current timestep to fade out
high-frequency loss at high noise levels — see
[Loss Configurations](configurations.md#timestep-weighting-diffusion--flow-matching-training).

## Documentation Sections

- [Wavelet Transforms](transforms.md)
- [Loss Configurations](configurations.md)
- [Advanced Usage](advanced.md)
- [PyWavelets Integration](pywavelets.md)