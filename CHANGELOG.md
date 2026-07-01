# Changelog

## 2.0.0 (2026-07-01)

Correctness-focused major release. Several defaults changed because the old
behavior was silently wrong for training; upgrading requires reviewing the
breaking changes below.

### Breaking changes

- **`normalize_bands` now defaults to `False`.** The old default (`True`)
  standardized prediction and target bands independently, making the loss
  blind to amplitude and offset errors — a 5×-scaled prediction returned ≈0
  loss. The opt-in `True` path now uses shared normalization (target-derived
  mean/std applied to both), which preserves amplitude error. If you relied on
  the old per-band standardization, there is no equivalent; it was a bug.
- **`forward` returns a scalar by default (`reduce=True`).** It returns
  `(loss, metrics)` where `loss` is a scalar ready for `backward()`. Callers
  that indexed or took `len()` of the result must pass `reduce=False` to get
  the per-band list.
- **Metrics are opt-in.** `WaveletLoss(metrics=True)` is required to collect
  metrics; the default returns an empty dict and performs zero `.item()` GPU
  syncs in the hot path. Metric keys were also renamed into a unified
  `wavelet_loss/` namespace, and misleading metrics were replaced (signed-mean
  `avg_hf` → coefficient energy) or dropped (scale-confounded sparsity).
- **Timesteps are validated against `max_timestep` (default `1.0`).** The
  timestep weighting is now flow-matching-native: sigmas in `[0, 1]` work out
  of the box, and DDPM-style integer timesteps require
  `WaveletLoss(..., max_timestep=1000)`. Timesteps outside
  `[0, max_timestep]` raise `ValueError` instead of silently saturating the
  weight. The fade is configurable via `timestep_cutoff` (default `0.7`) and
  `timestep_transition_width` (default `0.4`).
- **Inputs must be 4D `[B, C, H, W]`.** `forward` raises `ValueError`
  otherwise.

### Fixed

- SWT rewritten to match `pywt.swt2` exactly (vectorized and differentiable);
  previously the à-trous roll shift produced coefficients that disagreed with
  PyWavelets.
- DWT correctness is now asserted via parity tests against PyWavelets,
  including band mapping and boundary modes.
- Loss-reporting metrics corrected.

### Added

- Backend factory (`make_backend`) selecting between `pytorch_wavelets` and a
  custom zero-mode DWT backend with PyWavelets parity; QWT is flagged
  experimental.
- Timestep-aware loss weighting for diffusion / flow-matching training (see
  `docs/configurations.md`).
- Backprop tests for SWT, QWT, and the custom DWT backend; CPU-default test
  suite (`WAVELET_TEST_CUDA=1` for GPU paths).
- Transform conventions reference (`docs/transform-conventions.md`).

## 0.1.0

Initial release.
