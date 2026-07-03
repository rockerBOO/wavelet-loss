"""musubi-tuner adapter: loss classes for the pluggable ``--loss_fn`` seam.

musubi-tuner resolves ``--loss_fn wavelet_loss.musubi.WaveletPlusMSE`` via a
dotted import and instantiates the class with the parsed ``--loss_fn_args``
key=value pairs. The instance is called with the exact signature of
``NetworkTrainer.compute_loss`` (minus ``self``) and returns
``(scalar_loss, metrics_dict)``.

Example::

    --loss_fn wavelet_loss.musubi.WaveletPlusMSE \
    --loss_fn_args alpha=1.0 transform_type=swt wavelet=sym7 level=2

This module imports musubi_tuner and is only importable inside a
musubi-tuner environment.
"""

import torch
import torch.nn.functional as F

from musubi_tuner.training.timesteps import compute_loss_weighting_for_sd3

from wavelet_loss import WaveletLoss


class WaveletPlusMSE(torch.nn.Module):
    """Weighted flow-matching MSE + ``alpha`` * wavelet auxiliary loss.

    The MSE term mirrors musubi-tuner's default loss exactly (SD3-style
    ``args.weighting_scheme``, then mean), so ``alpha=0`` reproduces baseline
    training. ``**wavelet_kwargs`` are forwarded to :class:`WaveletLoss`
    (``transform_type``, ``wavelet``, ``level``, ``band_weights``,
    ``ll_level_threshold``, ``metrics``, ...).

    ``max_timestep`` defaults to 1000.0 here (not the WaveletLoss default of
    1.0) because musubi-tuner trainers pass timesteps on the scheduler's
    1..1000 footing.

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

    def forward(
        self,
        args,
        output,
        timesteps: torch.Tensor,
        noise_scheduler,
        dit_dtype: torch.dtype,
        network_dtype: torch.dtype,
        global_step: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        weighting = compute_loss_weighting_for_sd3(
            args.weighting_scheme, noise_scheduler, timesteps, timesteps.device, dit_dtype
        )
        mse = F.mse_loss(output.pred.to(network_dtype), output.target, reduction="none")
        if weighting is not None:
            mse = mse * weighting
        mse = mse.mean()

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

        wav_loss, wav_metrics = self.wavelet(wav_pred.float(), wav_target.float(), timesteps)

        metrics = dict(wav_metrics)
        metrics["loss/mse"] = float(mse.detach())
        metrics["loss/wavelet"] = float(wav_loss.detach())

        return mse + self.alpha * wav_loss, metrics
