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

## Quick Start

```python
import torch
from wavelet_loss import WaveletLoss

# Basic usage
loss_fn = WaveletLoss(wavelet="db4", level=3)
loss = loss_fn(prediction, target)
```

## Documentation Sections

- [Wavelet Transforms](transforms.md)
- [Loss Configurations](configurations.md)
- [Advanced Usage](advanced.md)
- [PyWavelets Integration](pywavelets.md)