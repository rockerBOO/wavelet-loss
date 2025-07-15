# Loss Configurations

## Custom Band Weights

You can customize the importance of different wavelet bands:

```python
# Custom band weights
band_weights = {"ll": 0.5, "lh": 0.2, "hl": 0.2, "hh": 0.1}
loss_fn = WaveletLoss(
    wavelet="db4",
    level=2,
    band_weights=band_weights
)
```

## Level-Specific Weights

Configure weights for specific levels and bands:

```python
# Level-specific band weights
band_level_weights = {
    "ll1": 0.3, "lh1": 0.1, "hl1": 0.1, "hh1": 0.1,
    "ll2": 0.2, "lh2": 0.05, "hl2": 0.05, "hh2": 0.1
}
loss_fn = WaveletLoss(
    wavelet="db4",
    level=2,
    band_level_weights=band_level_weights
)
```

## Loss Function Selection

Change the underlying loss calculation method:

```python
import torch.nn.functional as F

# Default is MSE loss
loss_fn = WaveletLoss(wavelet="db4", level=2)

# Change to L1 loss
loss_fn.set_loss_fn(F.l1_loss)
```

## LL Level Thresholding

Control the lowest-frequency band inclusion:

```python
# Adjust LL level threshold
loss_fn = WaveletLoss(
    wavelet="db4",
    level=3,
    ll_level_threshold=2  # Only include LL bands up to level 2
)
```