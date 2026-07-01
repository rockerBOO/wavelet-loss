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

## Timestep Weighting (Diffusion / Flow-Matching Training)

When training diffusion or flow-matching models, high-frequency wavelet detail
is only meaningful at low noise levels. Passing the current `timestep` to
`forward()` applies a smooth sigmoid fade: the loss weight is ~1 for timesteps
below the cutoff and fades toward 0 as the timestep approaches `max_timestep`
(pure noise).

```python
loss_fn = WaveletLoss(
    wavelet="db4",
    level=2,
    max_timestep=1.0,               # flow-matching sigmas in [0, 1] (default)
    timestep_cutoff=0.7,            # fraction of max_timestep where weight crosses 0.5
    timestep_transition_width=0.4,  # fraction of range the fade spans; smaller = harder cutoff
)

# In the training loop
loss, metrics = loss_fn(prediction, target, timestep=timesteps)  # [B] tensor
```

### Timestep conventions

`max_timestep` declares your trainer's timestep convention:

- **Flow-matching sigmas** (e.g. Flux2), timesteps in `[0, 1]`: use the
  default `max_timestep=1.0`.
- **DDPM-style integer timesteps** in `[0, 1000]`: you must pass
  `max_timestep=1000` explicitly.

Validation is strict by design: timesteps outside `[0, max_timestep]` raise
`ValueError` rather than silently saturating the weight. If you see this error
with integer timesteps, you forgot to set `max_timestep=1000`.

Omitting `timestep` disables the weighting entirely.

## Metrics

By default `forward` returns an empty metrics dict and performs no GPU syncs.
Pass `metrics=True` at construction to collect per-band, timestep-weight, and
QWT-component metrics (costs `.item()` syncs in the hot path):

```python
loss_fn = WaveletLoss(wavelet="db4", level=2, metrics=True)
loss, metrics = loss_fn(prediction, target)
```

## Band Normalization

`normalize_bands` defaults to `False`. When enabled, both prediction and
target coefficients are normalized with **shared** statistics (target-derived
mean/std), which preserves amplitude error:

```python
loss_fn = WaveletLoss(wavelet="db4", level=2, normalize_bands=True)
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