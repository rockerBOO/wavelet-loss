import math
import numpy as np

from torch import Tensor
from typing import Protocol
from collections.abc import Mapping

import torch
import torch.nn as nn
from torch.nn import functional as F

from wavelet_transform import QuaternionWaveletTransform


class LossCallableMSE(Protocol):
    def __call__(
        self,
        input: Tensor,
        target: Tensor,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
    ) -> Tensor: ...


class LossCallableReduction(Protocol):
    def __call__(self, input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor: ...


LossCallable = LossCallableReduction | LossCallableMSE
Metrics = dict[str, int | float | None]


class WaveletLoss(nn.Module):
    """Wavelet-based loss calculation module."""

    def __init__(
        self,
        wavelet="db4",
        level=3,
        transform_type="dwt",
        backend: str = "pytorch_wavelets",
        mode: str = "zero",
        loss_fn: LossCallable = F.mse_loss,
        device=torch.device("cpu"),
        band_level_weights: dict[str, float] | None = None,
        band_weights: dict[str, float] | None = None,
        quaternion_component_weights: dict[str, float] | None = None,
        ll_level_threshold: int | None = -1,
        metrics: bool = False,
        normalize_bands: bool = False,
        max_timestep: float = 1000,
    ):
        """

        Args:
            wavelet: Wavelet family (e.g., 'db4', 'sym7')
            level: Decomposition level
            transform_type: Type of wavelet transform ('dwt' or 'swt')
            loss_fn: Loss function to apply to wavelet coefficients
            device: Computation device
            band_level_weights: Optional custom weights for different bands on different levels
            band_weights: Optional custom weights for different bands
            component_weights: Weights for quaternion components
            ll_level_threshold: Level when applying loss for ll. Default -1 or last level.
        """
        super().__init__()
        self.level = level
        self.wavelet = wavelet
        self.transform_type = transform_type
        self.loss_fn = loss_fn
        self.device = device
        self.ll_level_threshold: int | None = ll_level_threshold
        self.metrics = metrics
        self.max_timestep = max_timestep
        self.normalize_bands = normalize_bands

        # Initialize transform via backend factory
        from wavelet_transform import make_backend

        self.backend = backend
        self.mode = mode
        self.transform = make_backend(backend, transform_type, wavelet, mode, device)

        if transform_type == "qwt":
            self.register_buffer("hilbert_x", self.transform.hilbert_x)
            self.register_buffer("hilbert_y", self.transform.hilbert_y)
            self.register_buffer("hilbert_xy", self.transform.hilbert_xy)
            self.component_weights = quaternion_component_weights or {
                "r": 1.0,
                "i": 0.7,
                "j": 0.7,
                "k": 0.5,
            }

        # Register wavelet filters as module buffers
        self.register_buffer("dec_lo", self.transform.dec_lo.to(device))
        self.register_buffer("dec_hi", self.transform.dec_hi.to(device))

        # Default weights from paper:
        # "Training Generative Image Super-Resolution Models by Wavelet-Domain Losses"
        self.band_level_weights = band_level_weights or {}
        self.band_weights = band_weights or {
            "ll": 0.1,
            "lh": 0.01,
            "hl": 0.01,
            "hh": 0.05,
        }

    def forward(
        self,
        pred_latent: Tensor,
        target_latent: Tensor,
        timestep: torch.Tensor | None = None,
        reduce: bool = True,
    ) -> tuple[Tensor | list[Tensor], Mapping[str, int | float | None]]:
        """
        Calculate wavelet loss between prediction and target.

        Returns:
            loss: Total wavelet loss (scalar if reduce=True, list of tensors if reduce=False)
            metrics: Wavelet metrics if requested in WaveletLoss(metrics=True)
        """
        if pred_latent.ndim != 4 or target_latent.ndim != 4:
            raise ValueError(
                f"WaveletLoss expects 4D [B, C, H, W] tensors, got "
                f"pred.ndim={pred_latent.ndim}, target.ndim={target_latent.ndim}."
            )

        if isinstance(self.transform, QuaternionWaveletTransform):
            return self.quaternion_forward(pred_latent, target_latent, timestep, reduce)

        batch_size = pred_latent.shape[0]
        device = pred_latent.device

        # Decompose inputs
        pred_coeffs = self.transform.decompose(pred_latent, self.level)
        target_coeffs = self.transform.decompose(target_latent, self.level)

        # Calculate weighted loss
        pattern_losses = []
        metrics: Metrics = {}

        base_weight = torch.ones((batch_size), device=device)
        if timestep is not None:
            base_weight *= self.smooth_timestep_weight(timestep)
            metrics["wavelet_loss/avg_timestep_adjusted_weight"] = base_weight.detach().mean().item()

        for i in range(self.level):
            # High frequency bands
            for band in ["ll", "lh", "hl", "hh"]:
                band_loss, pred, target, band_metrics = self.process_band(
                    pred_coeffs, target_coeffs, band, i, base_weight=base_weight
                )
                metrics.update(band_metrics)

                pattern_losses.append(band_loss)

        losses = pattern_losses

        # METRICS: Calculate all additional metrics (no gradients needed)
        if self.metrics:
            with torch.no_grad():
                metrics.update(self.process_coeff_metrics(pred_coeffs, target_coeffs))
                metrics.update(self.process_loss_metrics(pattern_losses))
                metrics.update(self.process_latent_metrics(pred_latent))

        if reduce:
            total = sum(loss_item.mean() for loss_item in losses)
            return total, metrics
        return losses, metrics

    def process_coeff_metrics(
        self,
        pred_coeffs: dict[str, list[Tensor]],
        target_coeffs: dict[str, list[Tensor]],
    ) -> Metrics:
        metrics: Metrics = {}
        metrics.update(self.calculate_correlation_metrics(pred_coeffs, target_coeffs))
        metrics.update(self.calculate_energy_metrics(pred_coeffs, target_coeffs))
        metrics.update(self.calculate_cross_scale_consistency_metrics(pred_coeffs, target_coeffs))
        metrics.update(self.calculate_directional_consistency_metrics(pred_coeffs, target_coeffs))

        return metrics

    @torch.no_grad()
    def calculate_energy_metrics(
        self,
        pred_coeffs: dict[str, list[Tensor]],
        target_coeffs: dict[str, list[Tensor]],
    ) -> Metrics:
        """Per-band coefficient energy (mean of squares) for pred and target.

        Energy is non-negative and amplitude-sensitive, so it surfaces the
        scale/amplitude errors that correlation and ratio metrics are blind to
        (e.g. a uniformly scaled prediction). Replaces the old signed-mean
        ``avg_hf_*`` metrics, which hovered near zero by construction.
        """
        metrics: Metrics = {}
        hf_pred: list[float] = []
        hf_target: list[float] = []

        for band in ["ll", "lh", "hl", "hh"]:
            for i in range(self.level):
                pred_e = torch.mean(pred_coeffs[band][i] ** 2).item()
                target_e = torch.mean(target_coeffs[band][i] ** 2).item()
                metrics[f"wavelet_loss/energy/{band}{i + 1}_pred"] = pred_e
                metrics[f"wavelet_loss/energy/{band}{i + 1}_target"] = target_e
                if band != "ll":
                    hf_pred.append(pred_e)
                    hf_target.append(target_e)

        if hf_pred:
            metrics["wavelet_loss/avg_hf_energy_pred"] = sum(hf_pred) / len(hf_pred)
            metrics["wavelet_loss/avg_hf_energy_target"] = sum(hf_target) / len(hf_target)

        return metrics

    def process_latent_metrics(self, pred_latent: Tensor) -> dict[str, int | float | None]:
        """
        Calculate metrics for the latent space.

        Args:
            pred_latent: The predicted latent tensor
            target_latent: The target latent tensor

        Returns:
            metrics: The metrics dictionary
        """
        metrics: dict[str, int | float | None] = {}
        metrics.update(self.calculate_latent_regularity_metrics(pred_latent))

        return metrics

    def process_loss_metrics(self, losses: list[Tensor]) -> Metrics:
        """Aggregate the weighted per-band losses into a scalar total metric.

        ``losses`` are the weighted per-band loss tensors — the same ones summed
        for the ``reduce=True`` return — so this mirrors the optimized objective
        exactly. Per-band breakdowns are emitted by ``process_band``.
        """
        metrics: Metrics = {}
        total = sum(loss_item.detach().mean() for loss_item in losses)
        metrics["wavelet_loss/total"] = float(total)
        return metrics

    def process_band(
        self,
        pred_coeffs: dict[str, list[Tensor]],
        target_coeffs: dict[str, list[Tensor]],
        band: str,
        i: int,
        base_weight: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Metrics]:
        """
        Process a single band and calculate the loss.

        Args:
            pred_coeffs: The predicted coefficients
            target_coeffs: The target coefficients
            band: The band to process (e.g. "lh", "hl", etc.)
            i: The level index
            base_weight: The base weight for the band

        Returns:
            loss: The band loss
            pred: The predicted wavelet component
            target: The target wavelet component
            metrics: The metrics for this band

        """
        # # If negative it's from the end of the levels else it's the level.
        # ll_threshold = None
        ll_threshold = self._calculate_effective_ll_threshold()
        if ll_threshold is not None and band == "ll" and i + 1 <= ll_threshold:
            return (
                torch.zeros_like(pred_coeffs[band][i]),
                torch.zeros_like(pred_coeffs[band][i]),
                torch.zeros_like(target_coeffs[band][i]),
                {},
            )

        weight_key = f"{band}{i + 1}"
        pred = pred_coeffs[band][i]
        target = target_coeffs[band][i]

        if self.normalize_bands:
            # Shared normalization: use the TARGET band statistics for BOTH tensors
            # so relative amplitude/offset errors are preserved (not zeroed out).
            mean = target.mean()
            std = target.std() + 1e-8
            pred = (pred - mean) / std
            target = (target - mean) / std

        band_loss = self.loss_fn(pred, target, reduction="none")

        weight = base_weight * self.band_level_weights.get(weight_key, self.band_weights[band])
        loss = weight.view(-1, 1, 1, 1) * band_loss

        metrics: Metrics = {
            f"wavelet_loss/band_loss/{band}{i + 1}": band_loss.detach().mean().item(),
            f"wavelet_loss/weighted_band_loss/{band}{i + 1}": loss.detach().mean().item(),
        }

        return loss, pred, target, metrics

    def quaternion_forward(
        self, pred: Tensor, target: Tensor, timestep: Tensor | None, reduce: bool = True
    ) -> tuple[Tensor | list[Tensor], Mapping[str, int | float | None]]:
        """
        Calculate QWT loss between prediction and target.

        Args:
            pred: Predicted tensor [B, C, H, W]
            target: Target tensor [B, C, H, W]

        Returns:
            Tuple of (total loss, detailed component losses)
        """
        batch_size = pred.shape[0]
        device = pred.device

        assert isinstance(self.transform, QuaternionWaveletTransform), "Not a quaternion wavelet transform"
        # Apply QWT to both inputs
        pred_qwt = self.transform.decompose_quaternion(pred, self.level)
        target_qwt = self.transform.decompose_quaternion(target, self.level)

        # Initialize total loss and component losses
        pattern_losses = []
        component_losses = {
            f"{component}_{band}_{level + 1}": torch.zeros_like(pred_qwt[component][band][level], device=pred.device)
            for level in range(self.level)
            for component in ["r", "i", "j", "k"]
            for band in ["ll", "lh", "hl", "hh"]
        }
        metrics: dict[str, float | int | None] = {}

        # Calculate the weighted loss based on the timestep
        base_weight = torch.ones((batch_size), device=device)
        if timestep is not None:
            base_weight *= self.smooth_timestep_weight(timestep)
            metrics["wavelet_loss/avg_timestep_adjusted_weight"] = base_weight.detach().mean().item()

        # Calculate loss for each quaternion component, band and level

        for component in ["r", "i", "j", "k"]:
            component_weight = self.component_weights[component]
            for band in ["ll", "lh", "hl", "hh"]:
                for level_idx in range(self.level):
                    band_loss, pred_coeffs, target_coeffs, band_metrics = self.process_band(
                        pred_qwt[component], target_qwt[component], band, level_idx, base_weight=base_weight
                    )
                    component_losses[f"{component}_{band}_{level_idx + 1}"] = band_loss
                    metrics[f"{component}_{band}_{level_idx + 1}"] = band_loss.detach().mean().item()
                    metrics.update(band_metrics)

                    pattern_losses.append(component_weight * band_loss)

            if self.metrics:
                component_metrics = self.process_coeff_metrics(pred_qwt[component], target_qwt[component])
                for k, v in component_metrics.items():
                    metrics[f"{component}_{k}"] = v

        # METRICS: Calculate all additional metrics
        if self.metrics:
            metrics.update(self.process_loss_metrics(pattern_losses))
            metrics.update(self.process_latent_metrics(pred))

        if reduce:
            total = sum(loss_item.mean() for loss_item in pattern_losses)
            return total, metrics
        return pattern_losses, metrics

    @torch.no_grad()
    def calculate_cross_scale_consistency_metrics(
        self,
        pred_coeffs: dict[str, list[Tensor]],
        target_coeffs: dict[str, list[Tensor]],
    ) -> dict:
        """
        Calculate metrics for cross-scale consistency between adjacent wavelet levels.

        Args:
            pred_coeffs: Dictionary of predicted wavelet coefficients
            target_coeffs: Dictionary of target wavelet coefficients

        Returns:
            Dictionary containing cross-scale consistency metrics

        Notes:
            - Compares energy ratios between adjacent scales
            - Uses log-scale differences for stability
            - Provides per-level and averaged metrics
        """
        metrics = {}

        for band in ["lh", "hl", "hh"]:
            for i in range(1, self.level):
                # Compare ratio of energies between adjacent scales
                pred_energy_fine = torch.mean(pred_coeffs[band][i - 1] ** 2).item()
                pred_energy_coarse = torch.mean(pred_coeffs[band][i] ** 2).item()
                target_energy_fine = torch.mean(target_coeffs[band][i - 1] ** 2).item()
                target_energy_coarse = torch.mean(target_coeffs[band][i] ** 2).item()

                # Calculate ratios and log differences
                pred_ratio = pred_energy_coarse / (pred_energy_fine + 1e-8)
                target_ratio = target_energy_coarse / (target_energy_fine + 1e-8)
                log_ratio_diff = abs(math.log(pred_ratio + 1e-8) - math.log(target_ratio + 1e-8))

                # Store individual metrics
                metrics[f"{band}{i}_to_{i + 1}_pred_scale_ratio"] = pred_ratio
                metrics[f"{band}{i}_to_{i + 1}_target_scale_ratio"] = target_ratio
                metrics[f"{band}{i}_to_{i + 1}_scale_log_diff"] = log_ratio_diff

        # Calculate average difference across all bands and scales
        if metrics:  # Check if dictionary is not empty
            metrics["avg_cross_scale_difference"] = sum(
                v for k, v in metrics.items() if k.endswith("scale_log_diff")
            ) / len([k for k in metrics if k.endswith("scale_log_diff")])

        return metrics

    @torch.no_grad()
    def calculate_correlation_metrics(
        self,
        pred_coeffs: dict[str, list[Tensor]],
        target_coeffs: dict[str, list[Tensor]],
    ) -> dict:
        """
        Calculate spatial correlation metrics between predicted and target wavelet coefficients.

        Args:
            pred_coeffs: Dictionary of predicted wavelet coefficients
            target_coeffs: Dictionary of target wavelet coefficients

        Returns:
            Dictionary containing correlation metrics for each band and level

        Notes:
            - Calculates correlation across spatial dimensions
            - Provides per-level and per-band averaged correlations
            - Uses centered coefficients for accurate correlation measurement
        """
        metrics = {}

        for band in ["lh", "hl", "hh"]:
            band_correlations = []
            for i in range(self.level):
                pred = pred_coeffs[band][i]  # [B, C, H, W]
                target = target_coeffs[band][i]

                # Flatten spatial dims but keep batch/channel separate
                pred_flat = pred.flatten(start_dim=2)  # [B, C, H*W]
                target_flat = target.flatten(start_dim=2)

                # Calculate correlation across spatial dimension
                pred_centered = pred_flat - pred_flat.mean(dim=2, keepdim=True)
                target_centered = target_flat - target_flat.mean(dim=2, keepdim=True)

                numerator = torch.sum(pred_centered * target_centered, dim=2)
                denom = torch.sqrt(torch.sum(pred_centered**2, dim=2) * torch.sum(target_centered**2, dim=2) + 1e-8)

                correlation = numerator / denom  # [B, C]
                avg_corr = correlation.mean().item()

                metrics[f"{band}{i + 1}_spatial_correlation"] = avg_corr
                band_correlations.append(avg_corr)

            metrics[f"{band}_avg_correlation"] = np.mean(band_correlations)

        return metrics

    @torch.no_grad()
    def calculate_directional_consistency_metrics(
        self,
        pred_coeffs: dict[str, list[Tensor]],
        target_coeffs: dict[str, list[Tensor]],
    ) -> dict:
        """
        Calculate metrics for directional consistency between wavelet bands.

        Args:
            pred_coeffs: Dictionary of predicted wavelet coefficients
            target_coeffs: Dictionary of target wavelet coefficients

        Returns:
            Dictionary containing directional consistency metrics

        Notes:
            - Analyzes horizontal vs vertical energy ratios (hl/lh)
            - Analyzes diagonal vs horizontal+vertical energy ratios (hh/(hl+lh))
            - Uses log-scale differences for stability
            - Provides per-level and averaged metrics
        """
        metrics = {}
        hv_diffs = []
        diag_diffs = []

        for i in range(1, self.level + 1):
            # Horizontal to vertical energy ratio
            pred_hl_energy = torch.mean(pred_coeffs["hl"][i - 1] ** 2).item()
            pred_lh_energy = torch.mean(pred_coeffs["lh"][i - 1] ** 2).item()
            target_hl_energy = torch.mean(target_coeffs["hl"][i - 1] ** 2).item()
            target_lh_energy = torch.mean(target_coeffs["lh"][i - 1] ** 2).item()

            pred_hv_ratio = pred_hl_energy / (pred_lh_energy + 1e-8)
            target_hv_ratio = target_hl_energy / (target_lh_energy + 1e-8)
            hv_log_diff = abs(math.log(pred_hv_ratio + 1e-8) - math.log(target_hv_ratio + 1e-8))

            # Diagonal to (horizontal+vertical) energy ratio
            pred_hh_energy = torch.mean(pred_coeffs["hh"][i - 1] ** 2).item()
            target_hh_energy = torch.mean(target_coeffs["hh"][i - 1] ** 2).item()

            pred_d_ratio = pred_hh_energy / (pred_hl_energy + pred_lh_energy + 1e-8)
            target_d_ratio = target_hh_energy / (target_hl_energy + target_lh_energy + 1e-8)
            diag_log_diff = abs(math.log(pred_d_ratio + 1e-8) - math.log(target_d_ratio + 1e-8))

            # Store metrics
            metrics[f"level{i}_horiz_vert_pred_ratio"] = pred_hv_ratio
            metrics[f"level{i}_horiz_vert_target_ratio"] = target_hv_ratio
            metrics[f"level{i}_horiz_vert_log_diff"] = hv_log_diff

            metrics[f"level{i}_diag_ratio_pred"] = pred_d_ratio
            metrics[f"level{i}_diag_ratio_target"] = target_d_ratio
            metrics[f"level{i}_diag_ratio_log_diff"] = diag_log_diff

            hv_diffs.append(hv_log_diff)
            diag_diffs.append(diag_log_diff)

        # Average metrics
        if hv_diffs:
            metrics["avg_horiz_vert_diff"] = sum(hv_diffs) / len(hv_diffs)
        if diag_diffs:
            metrics["avg_diag_ratio_diff"] = sum(diag_diffs) / len(diag_diffs)

        return metrics

    @torch.no_grad()
    def calculate_latent_regularity_metrics(self, pred_latents: Tensor) -> dict:
        """
        Calculate metrics for latent space regularity and smoothness.

        Args:
            pred_latents: Predicted latent tensor

        Returns:
            Dictionary containing latent regularity metrics

        Notes:
            - Calculates total variation (TV) for smoothness measurement
            - Provides statistical metrics (mean, std)
            - Measures deviation from normal distribution (std from 1.0)
        """
        metrics = {}

        # Calculate gradient magnitude of latent representation
        grad_x = pred_latents[:, :, 1:, :] - pred_latents[:, :, :-1, :]
        grad_y = pred_latents[:, :, :, 1:] - pred_latents[:, :, :, :-1]

        # Total variation
        tv_x = torch.mean(torch.abs(grad_x)).item()
        tv_y = torch.mean(torch.abs(grad_y)).item()
        tv_total = tv_x + tv_y

        # Statistical metrics
        std_value = torch.std(pred_latents).item()
        mean_value = torch.mean(pred_latents).item()
        std_diff = abs(std_value - 1.0)

        # Store metrics
        metrics["latent_tv_x"] = tv_x
        metrics["latent_tv_y"] = tv_y
        metrics["latent_tv_total"] = tv_total
        metrics["latent_std"] = std_value
        metrics["latent_mean"] = mean_value
        metrics["latent_std_from_normal"] = std_diff

        return metrics

    def smooth_timestep_weight(self, timestep):
        """
        Calculate smooth timestep-based weight using sigmoid transition.

        Args:
            timestep: Current diffusion timestep tensor

        Returns:
            Smooth weight tensor with sigmoid transition instead of hard cutoff

        Notes:
            - Weight decreases as timestep increases (later in denoising process)
            - Uses sigmoid for smooth transition around progress=0.3
            - Higher weights early in denoising, lower weights near completion
        """
        progress = 1.0 - (timestep / self.max_timestep)
        weight = torch.sigmoid((progress - 0.3) * 10)
        return weight

    def _calculate_effective_ll_threshold(self) -> int | None:
        """
        Calculate the effective LL level threshold.

        For positive values, returns the value as-is.
        For negative values, calculates from the end: level + threshold

        Returns:
            Effective threshold level, or None if no threshold is set

        Examples:
            level=3, threshold=1  -> 1
            level=3, threshold=2  -> 2
            level=3, threshold=-1 -> 2 (3 + (-1) = 2)
            level=3, threshold=-2 -> 1 (3 + (-2) = 1)
        """
        if self.ll_level_threshold is None:
            return None

        if self.ll_level_threshold > 0:
            return self.ll_level_threshold
        else:
            return self.level + self.ll_level_threshold

    def set_loss_fn(self, loss_fn: LossCallable):
        """
        Set loss function to use. Wavelet loss wants l1 or huber loss.
        """
        self.loss_fn = loss_fn
