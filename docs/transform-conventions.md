# Wavelet Transform Conventions (read before touching transforms)

Reference for contributors working on `src/wavelet_transform/`. These conventions
are non-obvious and were verified empirically against PyWavelets. Getting them
wrong silently produces a transform that "runs" but is incorrect (the kind of bug
that passes a loose energy check yet fails coefficient-level parity).

## Hybrid backend

- DWT has two backends: **`pytorch_wavelets`** (default, GPU-native) and a
  hand-rolled **`custom`** conv. `WaveletLoss(backend="custom")` selects the latter.
- **SWT is custom-only** — `pytorch_wavelets` 1.3.0 ships no SWT.
- **QWT is experimental/unvalidated** — its Hilbert filters are weak
  approximations (imaginary components ~200× smaller than the real part); it
  emits a `UserWarning`. Not validated against a reference transform.
- The `backend` flag only affects DWT. `make_backend` in
  `src/wavelet_transform/backends.py` routes the choice.

## Band naming

Matches pywt / `pytorch_wavelets`. Apply the 1D filters along **H (dim 2) first,
then W (dim 3)**:

| Band | Filters | pywt name |
|------|---------|-----------|
| `ll` | lo·H, lo·W | `cA` (approximation) |
| `lh` | **hi·H, lo·W** | `cH` (horizontal) |
| `hl` | **lo·H, hi·W** | `cV` (vertical) |
| `hh` | hi·H, hi·W | `cD` (diagonal) |

> ⚠️ `lh`/`hl` are **not** the naive "low-high / high-low along the first axis"
> order — `lh` is high-pass on H. The original code (and `docs/transforms.md`'s
> prose) had this swapped, which was one reason it didn't match pywt.
> `pytorch_wavelets`: `torch.unbind(yh[0], dim=2)` → `(lh, hl, hh)`.

## Boundary mode

- **DWT** default is `mode="zero"` (matches `pywt.dwt2(mode='zero')`). The custom
  DWT supports **only** `zero` (others raise `NotImplementedError`).
  `periodization` and other modes are deferred.
- **SWT** is inherently periodic — `pywt.swt2` takes no `mode` argument; the
  implementation uses circular padding.

## SWT à-trous roll-shift gotcha

Per level `j`, the centering roll is:

```
shift = -(k0 // 2) * (2**j)
```

where **`k0` is the original (un-upsampled) filter length**. Do NOT use
`-(k // 2)` with the upsampled length `k = (k0 - 1) * 2**j + 1` — that undershoots
for `j > 0` (~0.65 relerr) and is only correct at `j = 0`.

The verified SWT recipe, per axis: flip the filter; circular left-pad `(k-1)`;
`F.conv2d`; then `torch.roll(out, shift, dim)`; à-trous upsample filters by
`2**j` at level `j`.

## Verifying transform correctness

- **Validate against PyWavelets parity, not energy heuristics.** Target
  `relerr < 1e-6` vs `pywt.dwt2(mode='zero')` (DWT) and `pywt.swt2` (SWT). Note
  `pywt.swt2` returns levels **coarsest-first** (our level 0 ↔ ref index
  `level-1`).
- The DWT **custom-vs-library `allclose`** test is the contract that keeps the
  hybrid safe — both backends must agree, and both must match pywt.
- `pywt.Wavelet(w).dec_lo/dec_hi` return **float32** lists — wrap with
  `np.array(...)` before `torch.tensor(...)`.
- Precision ceiling: the library/SWT paths run at float32-filter precision
  (~3e-8–5e-8), comfortably under the `1e-6` gate; the custom DWT matches to
  ~2e-16 in float64.

## Deferred (future passes)

- `periodization` / other boundary modes.
- Real QWT — a proper Hilbert pair or dual-tree complex wavelet (DTCWT) instead
  of the current weak Hilbert approximations.
- AWWL-style timestep-adaptive band weighting (σ-driven LL-vs-detail balance).
