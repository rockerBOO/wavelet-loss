"""musubi-tuner adapter: loss classes for the pluggable ``--loss_fn`` seam.

musubi-tuner resolves ``--loss_fn wavelet_loss.musubi.WaveletPlusMSE`` via a
dotted import and instantiates the class with the parsed ``--loss_fn_args``
key=value pairs. The instance is called with a single ``LossContext`` (``ctx.args``,
``ctx.output``, ``ctx.timesteps``, ``ctx.noise_scheduler``, ``ctx.dit_dtype``,
``ctx.network_dtype``, ``ctx.global_step``) and returns ``(scalar_loss, metrics_dict)``.

Example::

    --loss_fn wavelet_loss.musubi.WaveletPlusMSE \
    --loss_fn_args alpha=1.0 "transform_type='swt'" "wavelet='sym7'" level=2

    --loss_fn wavelet_loss.musubi.WaveletPlusSNRHuber \
    --loss_fn_args alpha=1.0 loss_type='snr_huber'

    --loss_fn wavelet_loss.musubi.WaveletOnly \
    --loss_fn_args loss_type='l1' "transform_type='swt'"

    --loss_fn wavelet_loss.musubi.WaveletPlusX0Huber \
    --loss_fn_args alpha=1.0 loss_type='x0_huber' energy_beta=0.1 mottle_metrics=True

This module imports musubi_tuner and is only importable inside a
musubi-tuner environment.
"""

import torch
import torch.nn.functional as F

from musubi_tuner.training.timesteps import compute_loss_weighting_for_sd3

from wavelet_loss import WaveletLoss
from wavelet_loss.local_energy import local_energy_matching_loss, mottle_ratio
from wavelet_loss.loss import snr_aware_huber_loss, x0_threshold_huber_loss


class _WaveletPlusBase(torch.nn.Module):
    """Shared plumbing for the ``WaveletPlus*`` musubi-tuner adapters.

    Builds the inner :class:`WaveletLoss` (the wavelet auxiliary term) and
    computes the tensors it's called on. Subclasses only need to implement
    the outer flow-matching term in ``forward``.

    ``**wavelet_kwargs`` are forwarded to :class:`WaveletLoss`
    (``transform_type``, ``wavelet``, ``level``, ``band_weights``,
    ``ll_level_threshold``, ``metrics``, ...).

    ``max_timestep`` defaults to 1000.0 here (not the WaveletLoss default of
    1.0) because musubi-tuner trainers pass timesteps on the scheduler's
    1..1000 footing.

    ``loss_type`` selects the wavelet *band* residual penalty: ``'l1'``/``'mae'``,
    ``'huber'``/``'smooth_l1'``, ``'snr_huber'`` (SNR-aware adaptive-threshold
    pseudo-Huber, requires timesteps), ``'x0_huber'`` (pseudo-Huber with its
    threshold ``x0_huber_c`` in x0 units, see :func:`x0_threshold_huber_loss`),
    else MSE. This is independent of which outer flow-matching term a subclass uses.

    ``energy_beta > 0`` adds ``energy_beta *`` :func:`local_energy_matching_loss`:
    the local HF energy of the model's x0 estimate is pulled toward the ideal
    denoiser's expected energy, penalising texture painted into flat regions
    (mottle). ``mottle_metrics=True`` logs the mottle ratio M (flat/busy, ~1 ideal,
    >1 mottle) bucketed by noise level. Both need ``noisy_model_input`` in
    ``output.extra``.

    ``rectified_flow=True`` runs the wavelet term on reconstructed clean
    latents ``x0 = noisy - sigma * v`` instead of raw velocity. This requires
    the trainer to stash ``noisy_model_input`` into ``output.extra`` (e.g. via
    a ``call_dit`` override) — the base trainers do not, so the default is
    False (raw velocity space, AWWL-style).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        rectified_flow: bool = False,
        loss_type: str = "l2",
        energy_beta: float = 0.0,
        energy_window: int = 5,
        mottle_metrics: bool = False,
        **wavelet_kwargs,
    ):
        super().__init__()
        wavelet_kwargs.setdefault("max_timestep", 1000.0)
        self.alpha = alpha
        self.rectified_flow = rectified_flow
        self.loss_type = loss_type
        self.max_timestep = wavelet_kwargs["max_timestep"]
        self.energy_beta = energy_beta
        self.energy_window = energy_window
        self.mottle_metrics = mottle_metrics

        if loss_type == "snr_huber":
            wavelet_kwargs.setdefault("use_snr_aware_huber", True)
            self.wavelet = WaveletLoss(**wavelet_kwargs)
        elif loss_type == "x0_huber":
            wavelet_kwargs.setdefault("use_x0_huber", True)
            self.wavelet = WaveletLoss(**wavelet_kwargs)
        else:
            # band loss: matches the old musubi wavelet entrypoint's --loss_type fallback
            # (velocity-space residuals often exceed 1, where huber/smooth_l1 grows
            # linearly instead of quadratically -- expect much smaller band values than l2)
            def _band_loss(input: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
                if loss_type in ("l1", "mae"):
                    return F.l1_loss(input, target, reduction=reduction)
                if loss_type in ("huber", "smooth_l1"):
                    return F.smooth_l1_loss(input, target, reduction=reduction)
                return F.mse_loss(input, target, reduction=reduction)

            self.wavelet = WaveletLoss(loss_fn=_band_loss, **wavelet_kwargs)

    def _x0_pair(self, output, timesteps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(x0_hat, x0) recovered from ``noisy_model_input``, as 4D float tensors."""
        if "noisy_model_input" not in output.extra:
            raise ValueError(
                "this loss needs output.extra['noisy_model_input'] (rectified_flow / energy_beta / "
                "mottle_metrics); use a trainer that stashes it, e.g. boo-musubi-tuner's loss_context trainers"
            )
        noisy_model_input = output.extra["noisy_model_input"]
        # sigma == timesteps/1000, the value the DiT is conditioned on; the same
        # sigma scales both terms so it only weights the residual.
        sigmas = (timesteps.to(noisy_model_input.device, dtype=output.pred.dtype) / 1000.0).view(
            -1, *([1] * (output.pred.ndim - 1))
        )
        x0_hat = noisy_model_input - sigmas * output.pred.to(noisy_model_input.dtype)
        x0 = noisy_model_input - sigmas * output.target.to(noisy_model_input.dtype)
        return _to_4d(x0_hat).float(), _to_4d(x0).float()

    def _wavelet_term(self, output, timesteps: torch.Tensor, x0_pair=None) -> tuple[torch.Tensor, dict[str, float]]:
        if self.rectified_flow:
            wav_pred, wav_target = x0_pair if x0_pair is not None else self._x0_pair(output, timesteps)
        else:
            wav_pred, wav_target = _to_4d(output.pred).float(), _to_4d(output.target).float()

        return self.wavelet(wav_pred, wav_target, timesteps)

    def _aux_terms(self, output, timesteps: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """``alpha * wavelet + energy_beta * energy`` and their metrics."""
        needs_x0 = self.rectified_flow or self.energy_beta > 0 or self.mottle_metrics
        x0_pair = self._x0_pair(output, timesteps) if needs_x0 else None

        wav_loss, wav_metrics = self._wavelet_term(output, timesteps, x0_pair)
        metrics = dict(wav_metrics)
        metrics["loss/wavelet"] = float(wav_loss.detach())
        loss = self.alpha * wav_loss

        if self.energy_beta > 0 or self.mottle_metrics:
            x0_hat, x0 = x0_pair
            t_frac = (timesteps / self.max_timestep).to(x0.device, torch.float32).clamp(0, 1)
            level = self.wavelet.level
            x0_hat_c = self.wavelet.transform.decompose(x0_hat, level)
            x0_c = self.wavelet.transform.decompose(x0, level)
            if self.energy_beta > 0:
                fade = self.wavelet.smooth_timestep_weight(timesteps).to(x0.device, torch.float32)
                energy = local_energy_matching_loss(
                    x0_hat_c, x0_c, t_frac, level, weight=fade, window=self.energy_window
                )
                metrics["loss/energy"] = float(energy.detach())
                loss = loss + self.energy_beta * energy
            if self.mottle_metrics:
                # M depends strongly on t, so log it per noise-level bucket (one bucket per step at batch size 1)
                bucket = _t_bucket(float(t_frac.mean()))
                m = mottle_ratio(x0_hat_c, x0_c, t_frac, level, window=self.energy_window)
                metrics[f"mottle/M_flat_{bucket}"] = m["flat"]
                metrics[f"mottle/M_busy_{bucket}"] = m["busy"]

        return loss, metrics


def _to_4d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 5:
        # single-frame video-style latents (B, C, 1, H, W), e.g. Krea2
        if x.shape[2] != 1:
            raise ValueError(f"wavelet term only supports single-frame 5D latents, got shape {tuple(x.shape)}")
        return x.squeeze(2)
    return x


def _t_bucket(t: float) -> str:
    if t < 0.2:
        return "t00-20"
    if t < 0.5:
        return "t20-50"
    return "t50-100"


class WaveletOnly(_WaveletPlusBase):
    """Pure wavelet loss -- no outer flow-matching MSE/Huber term.

    Trains solely on the wavelet-band residual (``loss_type`` selects the
    per-band penalty; see :class:`_WaveletPlusBase`). ``alpha`` still scales
    the wavelet term so it composes with the same ``--loss_fn_args`` shape as
    the other adapters, but defaults to ``1.0`` since there's no other term
    to balance against.
    """

    def __init__(self, alpha: float = 1.0, rectified_flow: bool = False, loss_type: str = "l2", **kwargs):
        super().__init__(alpha, rectified_flow, loss_type, **kwargs)

    def forward(self, ctx) -> tuple[torch.Tensor, dict[str, float]]:
        return self._aux_terms(ctx.output, ctx.timesteps)


class WaveletPlusMSE(_WaveletPlusBase):
    """Weighted flow-matching MSE + ``alpha`` * wavelet auxiliary loss.

    The MSE term mirrors musubi-tuner's default loss exactly (SD3-style
    ``args.weighting_scheme``, then mean), so ``alpha=0`` reproduces baseline
    training. See :class:`_WaveletPlusBase` for the shared wavelet-term kwargs.
    """

    def forward(self, ctx) -> tuple[torch.Tensor, dict[str, float]]:
        args, output, timesteps = ctx.args, ctx.output, ctx.timesteps
        weighting = compute_loss_weighting_for_sd3(
            args.weighting_scheme, ctx.noise_scheduler, timesteps, timesteps.device, ctx.dit_dtype
        )
        mse = F.mse_loss(output.pred.to(ctx.network_dtype), output.target, reduction="none")
        if weighting is not None:
            mse = mse * weighting.reshape(-1, *([1] * (mse.ndim - 1)))
        mse = mse.mean()

        aux, metrics = self._aux_terms(output, timesteps)
        metrics["loss/mse"] = float(mse.detach())

        return mse + aux, metrics


class WaveletPlusSNRHuber(_WaveletPlusBase):
    """Weighted flow-matching SNR-aware Huber + ``alpha`` * wavelet auxiliary loss.

    Same structure as :class:`WaveletPlusMSE`, but the outer flow-matching
    term uses the SNR-aware adaptive-threshold pseudo-Huber loss (UltraFlux
    paper; ported from rockerBOO/sd-scripts#1991) instead of plain MSE. This
    is independent of ``loss_type`` (the wavelet band residual penalty) --
    e.g. outer=snr_huber + bands=l1 is valid, as is outer=snr_huber +
    bands=snr_huber (``loss_type='snr_huber'``).

    The ``main_snr_huber_*`` args tune the outer term; the wavelet band term
    (when ``loss_type='snr_huber'``) is tuned independently via the
    ``snr_huber_*``/``min_snr_beta`` ``**wavelet_kwargs`` forwarded to
    :class:`WaveletLoss`.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        rectified_flow: bool = False,
        loss_type: str = "l2",
        main_snr_huber_cmin: float = 0.2,
        main_snr_huber_cmax: float = 1.0,
        main_snr_huber_gamma: float = 5.0,
        main_snr_huber_alpha: float = 0.5,
        main_min_snr_beta: float = 0.0,
        **kwargs,
    ):
        super().__init__(alpha, rectified_flow, loss_type, **kwargs)
        self.main_snr_huber_cmin = main_snr_huber_cmin
        self.main_snr_huber_cmax = main_snr_huber_cmax
        self.main_snr_huber_gamma = main_snr_huber_gamma
        self.main_snr_huber_alpha = main_snr_huber_alpha
        self.main_min_snr_beta = main_min_snr_beta

    def forward(self, ctx) -> tuple[torch.Tensor, dict[str, float]]:
        args, output, timesteps = ctx.args, ctx.output, ctx.timesteps
        weighting = compute_loss_weighting_for_sd3(
            args.weighting_scheme, ctx.noise_scheduler, timesteps, timesteps.device, ctx.dit_dtype
        )
        main = snr_aware_huber_loss(
            output.pred.to(ctx.network_dtype),
            output.target,
            timesteps,
            self.max_timestep,
            cmin=self.main_snr_huber_cmin,
            cmax=self.main_snr_huber_cmax,
            gamma=self.main_snr_huber_gamma,
            alpha=self.main_snr_huber_alpha,
            min_snr_beta=self.main_min_snr_beta,
        )
        if weighting is not None:
            main = main * weighting.reshape(-1, *([1] * (main.ndim - 1)))
        main = main.mean()

        aux, metrics = self._aux_terms(output, timesteps)
        metrics["loss/snr_huber"] = float(main.detach())

        return main + aux, metrics


class WaveletPlusX0Huber(_WaveletPlusBase):
    """Weighted flow-matching x0-threshold pseudo-Huber + ``alpha`` * wavelet auxiliary loss.

    Same structure as :class:`WaveletPlusSNRHuber`, but the outer term is
    :func:`x0_threshold_huber_loss` with threshold ``main_x0_huber_c`` in x0 units:
    near-L2 pushback on small errors at every noise level (the velocity-space
    SNR-Huber saturates on small low-t x0 errors, since they appear in velocity
    divided by t). Pair with ``loss_type='x0_huber'`` for the same threshold on
    the bands (tuned by ``x0_huber_c``).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        rectified_flow: bool = False,
        loss_type: str = "l2",
        main_x0_huber_c: float = 1.0,
        **kwargs,
    ):
        super().__init__(alpha, rectified_flow, loss_type, **kwargs)
        self.main_x0_huber_c = main_x0_huber_c

    def forward(self, ctx) -> tuple[torch.Tensor, dict[str, float]]:
        args, output, timesteps = ctx.args, ctx.output, ctx.timesteps
        weighting = compute_loss_weighting_for_sd3(
            args.weighting_scheme, ctx.noise_scheduler, timesteps, timesteps.device, ctx.dit_dtype
        )
        main = x0_threshold_huber_loss(
            output.pred.to(ctx.network_dtype), output.target, timesteps, self.max_timestep, c=self.main_x0_huber_c
        )
        if weighting is not None:
            main = main * weighting.reshape(-1, *([1] * (main.ndim - 1)))
        main = main.mean()

        aux, metrics = self._aux_terms(output, timesteps)
        metrics["loss/x0_huber"] = float(main.detach())

        return main + aux, metrics
