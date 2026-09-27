# Flat-region mottle and the x0-unit Huber threshold

Status as of 2026-09-27.

## Problem

DiT LoRA training with `WaveletPlusSNRHuber` (Krea 2, FLUX.2 Klein) leaves low-amplitude high-frequency
texture ("mottle") in flat regions such as plain backgrounds and skies. It shows up after roughly 1000 steps.

## Why SNR-Huber lets it through

Flow matching: `x_t = (1 - t) x0 + t ε`, `v = ε - x0`, `x0_hat = x_t - t v_hat`. So an x0 error `e` is a
velocity residual `r = e / t`. Mottle is formed at low `t`, where a small `e` becomes a large `r`. A
v-space Huber with a fixed threshold treats that residual as an outlier and caps the gradient pushing it
back. Reweighting timesteps or setting a local `c` can't undo that cap.

## Fix A: x0-unit Huber threshold (implemented, in use)

```
rho(r) = r^2 / (sqrt(1 + (r t / c)^2) + 1)
```

This is pseudo-Huber with threshold `c / t` in v, i.e. threshold `c` in x0 units. It is ~`r^2 / 2` (plain
L2) while the x0 error is below `c` at any `t`. The gradient is bounded at `c / t` once the error exceeds
`c`. Tests: `tests/test_x0_huber.py`.

- `wavelet_loss.x0_threshold_huber_loss(pred, target, timestep, max_timestep, c=1.0)`
- `WaveletLoss(use_x0_huber=True, x0_huber_c=...)`: per-band. Requires `timestep`, and can't be combined
  with `use_snr_aware_huber`.
- `wavelet_loss.musubi.WaveletPlusX0Huber`: x0-Huber main term plus the wavelet term. Set
  `loss_type='x0_huber'` so the bands use it too.

## Fix B: local energy matching (implemented, not yet trained)

`wavelet_loss/local_energy.py`. Target band energy is the MMSE-optimal energy at noise level `t`:
`G S` with `G = S / (S + t^2 / (1 - t)^2)`. `S` is the local (5x5 box) high-frequency SWT energy of x0.

```
(log(E_loc(x0_hat) + eps) - log(G S + eps))^2
```

The loss above is enabled by `energy_beta > 0` on any musubi adapter. `mottle_metrics=True` logs
`mottle/M_{flat,busy}_{t00-20,t20-50,t50-100}`. `M` = predicted HF energy / optimal HF energy, measured over
the flattest 30% of the image (level-1 HF energy of x0) or the busy rest. 1.0 is ideal.

Both features need `noisy_model_input`. They require the boo-musubi-tuner `loss_context` trainers
(`src/boo_musubi_tuner/loss_context/`), which stash it in `DiTOutput.extra`.

## Experiment log (FLUX.2 Klein 4B, retro-anime set, `~/code/training_runs/retro-anime-mottle-*`)

The runs share: lr 1e-4, dim 8, batch 1, max_grad_norm 1, flux2_shift 3, SWT sym7 level 2 with LL
weight 0, XM off.

| run | optimizer / scale | loss | result |
|---|---|---|---|
| `4b57be73` | RAdam, alpha 16 (x2), loraplus 8 | SNR-Huber | collapsed by step 1000; LoRA had no outlier tensors |
| `b3ebab42` (wandb `xzxa7ydz`) | AdamW, x2, loraplus 8 | SNR-Huber | wobble at 1000, good at 1500; clay background mottle 0.31 → 1.08 → 1.26 (steps 500/1000/1500). Stopped at 1513. |
| `b92c40f0` (wandb `r7ntkvri`) | AdamW, x2, loraplus 8 | x0-Huber (`c=1`, alpha 1.0) | ran to 2000; clay background 1.01 → 1.45 → 0.61 → 0.38; quality comparable or better, judged "surprisingly good" |

The mottle number is the std (gray levels) of `image - 5x5 box blur` on the clay prompt's background crop
`(20, 20, 220, 220)`. Composition changes between checkpoints make per-sample numbers noisy. An
all-prompt automatic flat mask was inconclusive for the same reason.

In-training M for `r7ntkvri` doesn't drift up over training:

- t < 0.2: M_flat ≈ 1.00–1.02, but only 11 samples.
- t 0.2–0.5: M_flat ≈ 1.3, M_busy ≈ 1.25.

shift 3 rarely samples t < 0.2, which is exactly where mottle forms.

## Next

1. Offline M on fixed images, noise and t ∈ [0.05, 0.3] for both runs' step 500/1000/1500 checkpoints. This
   gives a like-for-like comparison, including the control, which never logged M.
2. "Work backwards" from x0-Huber:
   - gentler scale (`retro-anime-mottle-control` config now holds alpha 8 = x1, loraplus 4);
   - then `energy_beta` (Fix B) if mottle remains.
