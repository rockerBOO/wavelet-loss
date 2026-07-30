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

This module imports musubi_tuner and is only importable inside a
musubi-tuner environment.
"""

import torch
import torch.nn.functional as F

from musubi_tuner.training.timesteps import compute_loss_weighting_for_sd3

from wavelet_loss import WaveletLoss
from wavelet_loss.loss import snr_aware_huber_loss


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
    pseudo-Huber, requires timesteps), else MSE. This is independent of which
    outer flow-matching term a subclass uses.

    ``rectified_flow=True`` runs the wavelet term on reconstructed clean
    latents ``x0 = noisy - sigma * v`` instead of raw velocity. This requires
    the trainer to stash ``noisy_model_input`` into ``output.extra`` (e.g. via
    a ``call_dit`` override) — the base trainers do not, so the default is
    False (raw velocity space, AWWL-style).
    """

    def __init__(self, alpha: float = 0.1, rectified_flow: bool = False, loss_type: str = "l2", **wavelet_kwargs):
        super().__init__()
        wavelet_kwargs.setdefault("max_timestep", 1000.0)
        self.alpha = alpha
        self.rectified_flow = rectified_flow
        self.loss_type = loss_type
        self.max_timestep = wavelet_kwargs["max_timestep"]

        if loss_type == "snr_huber":
            wavelet_kwargs.setdefault("use_snr_aware_huber", True)
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

    def _wavelet_term(self, output, timesteps: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        if self.rectified_flow:
            noisy_model_input = output.extra["noisy_model_input"]
            # sigma == timesteps/1000, the value the DiT is conditioned on; the same
            # sigma scales both terms so it only weights the residual.
            sigmas = (timesteps.to(noisy_model_input.device, dtype=output.pred.dtype) / 1000.0).view(
                -1, *([1] * (output.pred.ndim - 1))
            )
            wav_pred = noisy_model_input - sigmas * output.pred.to(noisy_model_input.dtype)
            wav_target = noisy_model_input - sigmas * output.target.to(noisy_model_input.dtype)
        else:
            wav_pred = output.pred
            wav_target = output.target

        if wav_pred.ndim == 5:
            # single-frame video-style latents (B, C, 1, H, W), e.g. Krea2
            if wav_pred.shape[2] != 1:
                raise ValueError(f"wavelet term only supports single-frame 5D latents, got shape {tuple(wav_pred.shape)}")
            wav_pred = wav_pred.squeeze(2)
            wav_target = wav_target.squeeze(2)

        return self.wavelet(wav_pred.float(), wav_target.float(), timesteps)


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
            mse = mse * weighting
        mse = mse.mean()

        wav_loss, wav_metrics = self._wavelet_term(output, timesteps)

        metrics = dict(wav_metrics)
        metrics["loss/mse"] = float(mse.detach())
        metrics["loss/wavelet"] = float(wav_loss.detach())

        return mse + self.alpha * wav_loss, metrics


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
        **wavelet_kwargs,
    ):
        super().__init__(alpha, rectified_flow, loss_type, **wavelet_kwargs)
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
            main = main * weighting
        main = main.mean()

        wav_loss, wav_metrics = self._wavelet_term(output, timesteps)

        metrics = dict(wav_metrics)
        metrics["loss/snr_huber"] = float(main.detach())
        metrics["loss/wavelet"] = float(wav_loss.detach())

        return main + self.alpha * wav_loss, metrics
