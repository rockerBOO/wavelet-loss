import math
import numpy as np

from torch import Tensor
from typing import Protocol
from collections.abc import Mapping

import torch
import torch.nn as nn
from torch.nn import functional as F

from wavelet_transform import (
    DiscreteWaveletTransform,
    StationaryWaveletTransform,
    QuaternionWaveletTransform,
)


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
    def __call__(
        self, input: Tensor, target: Tensor, reduction: str = "mean"
    ) -> Tensor: ...


LossCallable = LossCallableReduction | LossCallableMSE


class WaveletLoss(nn.Module):
    """Wavelet-based loss calculation module."""

    def __init__(
        self,
        wavelet="db4",
        level=3,
        transform_type="dwt",
        loss_fn: LossCallable = F.mse_loss,
        device=torch.device("cpu"),
        band_level_weights: dict[str, float] | None = None,
        band_weights: dict[str, float] | None = None,
        quaternion_component_weights: dict[str, float] | None = None,
        ll_level_threshold: int | None = -1,
        metrics: bool = False,
        energy_ratio: float = 0.0,
        energy_scale_factor: float = 0.01,
        normalize_bands: bool = True,
        max_timestep: float = 1000,
        timestep_intensity: float = 0.5,
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
        self.ll_level_threshold = (
            ll_level_threshold if ll_level_threshold is not None else None
        )
        self.metrics = metrics
        self.energy_ratio = energy_ratio
        self.energy_scale_factor = energy_scale_factor
        self.max_timestep = max_timestep
        self.timestep_intensity = timestep_intensity
        self.normalize_bands = normalize_bands

        # Initialize transform based on type
        if transform_type == "dwt":
            self.transform = DiscreteWaveletTransform(wavelet, device)
        elif transform_type == "swt":  # swt
            self.transform = StationaryWaveletTransform(wavelet, device)
        elif transform_type == "qwt":
            self.transform = QuaternionWaveletTransform(wavelet, device)

            # Register Hilbert filters as buffers
            self.register_buffer("hilbert_x", self.transform.hilbert_x)
            self.register_buffer("hilbert_y", self.transform.hilbert_y)
            self.register_buffer("hilbert_xy", self.transform.hilbert_xy)

            # Default weights
            self.component_weights = quaternion_component_weights or {
                "r": 1.0,  # Real part (standard wavelet)
                "i": 0.7,  # x-Hilbert (imaginary part)
                "j": 0.7,  # y-Hilbert (imaginary part)
                "k": 0.5,  # xy-Hilbert (imaginary part)
            }
        else:
            raise RuntimeError(f"Invalid transform type {transform_type}")

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
    ) -> tuple[list[Tensor], Mapping[str, int | float | None]]:
        """
        Calculate wavelet loss between prediction and target.

        Returns:
            loss: Total wavelet loss
            metrics: Wavelet metrics if requested in WaveletLoss(metrics=True)
        """
        if isinstance(self.transform, QuaternionWaveletTransform):
            return self.quaternion_forward(pred_latent, target_latent)

        batch_size = pred_latent.shape[0]
        device = pred_latent.device

        # Decompose inputs
        pred_coeffs = self.transform.decompose(pred_latent, self.level)
        target_coeffs = self.transform.decompose(target_latent, self.level)

        # Calculate weighted loss
        pattern_losses = []
        combined_hf_pred = []
        combined_hf_target = []
        metrics = {}

        # Use original weights by default
        band_weights = self.band_weights
        band_level_weights = self.band_level_weights

        base_weight = torch.ones((batch_size), device=device)
        if timestep is not None:
            base_weight *= self.smooth_timestep_weight(timestep)
            metrics["wavelet_loss/avg_timestep_adjusted_weight"] = (
                base_weight.detach().mean().item()
            )

        # If negative it's from the end of the levels else it's the level.
        ll_threshold = None
        if self.ll_level_threshold is not None:
            ll_threshold = (
                self.ll_level_threshold
                if self.ll_level_threshold > 0
                else self.level + self.ll_level_threshold
            )

        # 1. Pattern Loss (using normalization)
        for i in range(self.level):
            pattern_level_losses = torch.zeros_like(pred_coeffs["lh"][i])

            # High frequency bands
            for band in ["ll", "lh", "hl", "hh"]:
                # Skip LL bands except for ones at or beyond the threshold
                if ll_threshold is not None and band == "ll" and i + 1 <= ll_threshold:
                    continue

                weight_key = f"{band}{i + 1}"
                pred = pred_coeffs[band][i]
                target = target_coeffs[band][i]

                if band in pred_coeffs and band in target_coeffs:
                    if self.normalize_bands:
                        # Normalize wavelet components
                        pred_coeffs[band][i] = (
                            pred_coeffs[band][i] - pred_coeffs[band][i].mean()
                        ) / (pred_coeffs[band][i].std() + 1e-8)
                        target_coeffs[band][i] = (
                            target_coeffs[band][i] - target_coeffs[band][i].mean()
                        ) / (target_coeffs[band][i].std() + 1e-8)

                    # 1. Magnitude loss
                    band_loss = self.loss_fn(pred, target)

                    # 2. Local structure loss
                    pred_grad_x = torch.diff(pred, dim=-1)
                    pred_grad_y = torch.diff(pred, dim=-2)
                    target_grad_x = torch.diff(target, dim=-1)
                    target_grad_y = torch.diff(target, dim=-2)

                    gradient_loss = F.mse_loss(pred_grad_x, target_grad_x) + F.mse_loss(
                        pred_grad_y, target_grad_y
                    )

                    # 3. Global correlation per channel
                    B, C = pred.shape[:2]
                    pred_flat = pred.view(B, C, -1)
                    target_flat = target.view(B, C, -1)

                    cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=2)
                    correlation_loss = (1 - cos_sim).mean()

                    weight = base_weight * band_level_weights.get(
                        weight_key, band_weights[band]
                    )
                    pattern_level_losses += weight.view(-1, 1, 1, 1) * (
                        band_loss + 0.05 * gradient_loss + 0.1 * correlation_loss
                    )  # mean stack dim

                    metrics[f"{band}{i}_band_loss"] = band_loss.detach().mean().item()
                    metrics[f"{band}{i}_gradient_loss"] = (
                        gradient_loss.detach().mean().item()
                    )
                    metrics[f"{band}{i}_correlation_loss"] = (
                        correlation_loss.detach().mean().item()
                    )

                    # Collect high frequency bands for visualization
                    combined_hf_pred.append(pred_coeffs[band][i])
                    combined_hf_target.append(target_coeffs[band][i])

            pattern_losses.append(pattern_level_losses)

        # TODO: need to update this to work with a list of losses
        # If we are balancing the energy loss with the pattern loss
        # if self.energy_ratio > 0.0:
        #     energy_loss = self.energy_matching_loss(batch_size, pred_coeffs, target_coeffs, device)
        #
        #     loss = (
        #         (1 - self.energy_ratio) * pattern_loss  # Core spatial patterns
        #         + self.energy_ratio * (self.energy_scale_factor * energy_loss)  # Fixes energy disparity
        #     )
        # else:
        energy_loss = None
        losses = pattern_losses

        # METRICS: Calculate all additional metrics (no gradients needed)
        if self.metrics:
            with torch.no_grad():
                # Raw energy metrics
                for band in ["lh", "hl", "hh"]:
                    for i in range(1, self.level + 1):
                        pred_stack = pred_coeffs[band][i - 1]
                        target_stack = target_coeffs[band][i - 1]

                        metrics[f"{band}{i}_raw_pred_energy"] = torch.mean(
                            pred_stack**2
                        ).item()
                        metrics[f"{band}{i}_raw_target_energy"] = torch.mean(
                            target_stack**2
                        ).item()
                        metrics[f"{band}{i}_energy_ratio"] = (
                            torch.mean(pred_stack**2)
                            / (torch.mean(target_stack**2) + 1e-8)
                        ).item()

                metrics.update(
                    self.calculate_correlation_metrics(pred_coeffs, target_coeffs)
                )
                metrics.update(
                    self.calculate_cross_scale_consistency_metrics(
                        pred_coeffs, target_coeffs
                    )
                )
                metrics.update(
                    self.calculate_directional_consistency_metrics(
                        pred_coeffs, target_coeffs
                    )
                )
                metrics.update(
                    self.calculate_sparsity_metrics(pred_coeffs, target_coeffs)
                )
                metrics.update(self.calculate_latent_regularity_metrics(pred_latent))

                # Add loss components to metrics
                for i, pattern_loss in enumerate(pattern_losses):
                    metrics[f"pattern_loss-{i + 1}"] = (
                        pattern_loss.detach().mean().item()
                    )

                for i, total_loss in enumerate(losses):
                    metrics[f"total_loss-{i + 1}"] = total_loss.detach().mean().item()

                if energy_loss is not None:
                    metrics["energy_loss"] = energy_loss.detach().mean().item()

        # Combine high frequency bands for visualization
        if combined_hf_pred and combined_hf_target:
            combined_hf_pred = self._pad_tensors(combined_hf_pred)
            combined_hf_target = self._pad_tensors(combined_hf_target)

            combined_hf_pred = torch.cat(combined_hf_pred, dim=1)
            combined_hf_target = torch.cat(combined_hf_target, dim=1)

            metrics["combined_hf_pred"] = combined_hf_pred.detach().mean().item()
            metrics["combined_hf_target"] = combined_hf_target.detach().mean().item()
        else:
            combined_hf_pred = None
            combined_hf_target = None

        return losses, metrics

    def quaternion_forward(
        self, pred: Tensor, target: Tensor
    ) -> tuple[list[Tensor], Mapping[str, int | float | None]]:
        """
        Calculate QWT loss between prediction and target.

        Args:
            pred: Predicted tensor [B, C, H, W]
            target: Target tensor [B, C, H, W]

        Returns:
            Tuple of (total loss, detailed component losses)
        """
        assert isinstance(self.transform, QuaternionWaveletTransform), (
            "Not a quaternion wavelet transform"
        )
        # Apply QWT to both inputs
        pred_qwt = self.transform.decompose(pred, self.level)
        target_qwt = self.transform.decompose(target, self.level)

        # Initialize total loss and component losses
        total_losses = []
        component_losses = {
            f"{component}_{band}_{level + 1}": torch.zeros_like(
                pred_qwt[component][band][level], device=pred.device
            )
            for level in range(self.level)
            for component in ["r", "i", "j", "k"]
            for band in ["ll", "lh", "hl", "hh"]
        }

        # Calculate loss for each quaternion component, band and level
        for level_idx in range(self.level):
            pattern_level_losses = torch.zeros_like(pred_qwt["r"]["lh"][level_idx])
            for band in ["ll", "lh", "hl", "hh"]:
                band_weight = self.band_weights[band]
                for component in ["r", "i", "j", "k"]:
                    component_weight = self.component_weights[component]

                    band_level_key = f"{band}{level_idx + 1}"
                    # band_level_weights take priority over band_weight if exists
                    if band_level_key in self.band_level_weights:
                        level_weight = self.band_level_weights[band_level_key]
                    else:
                        level_weight = band_weight

                    # Get coefficients at this level
                    pred_coeff = pred_qwt[component][band][level_idx]
                    target_coeff = target_qwt[component][band][level_idx]

                    # Calculate loss
                    level_loss = self.loss_fn(pred_coeff, target_coeff)

                    # Apply weights
                    weighted_loss = component_weight * level_weight * level_loss

                    # Add to total loss
                    pattern_level_losses += weighted_loss

                    # Add to component loss
                    component_losses[f"{component}_{band}_{level_idx + 1}"] += (
                        weighted_loss
                    )

            total_losses.append(pattern_level_losses)

        metrics = {k: v.detach().mean().item() for k, v in component_losses.items()}
        return total_losses, metrics

    def _pad_tensors(self, tensors: list[Tensor]) -> list[Tensor]:
        """Pad tensors to match the largest size."""
        # Find max dimensions
        max_h = max(t.shape[2] for t in tensors)
        max_w = max(t.shape[3] for t in tensors)

        padded_tensors = []
        for tensor in tensors:
            h_pad = max_h - tensor.shape[2]
            w_pad = max_w - tensor.shape[3]

            if h_pad > 0 or w_pad > 0:
                # Pad bottom and right to match max dimensions
                padded = F.pad(tensor, (0, w_pad, 0, h_pad))
                padded_tensors.append(padded)
            else:
                padded_tensors.append(tensor)

        return padded_tensors

    def energy_matching_loss(
        self,
        batch_size: int,
        pred_coeffs: dict[str, list[Tensor]],
        target_coeffs: dict[str, list[Tensor]],
        device: torch.device,
    ) -> Tensor:
        energy_loss = torch.zeros(batch_size, device=device)
        for band in ["lh", "hl", "hh"]:
            for i in range(1, self.level + 1):
                weight_key = f"{band}{i}"
                # Calculate band energies
                pred_energy = torch.mean(pred_coeffs[band][i - 1] ** 2)
                target_energy = torch.mean(target_coeffs[band][i - 1] ** 2)

                # Log-scale energy ratio loss (more stable than direct ratio)
                ratio_loss = torch.abs(
                    torch.log(pred_energy + 1e-8) - torch.log(target_energy + 1e-8)
                )

                weight = self.band_level_weights.get(
                    weight_key, self.band_weights[band]
                )
                energy_loss += weight * ratio_loss

        return energy_loss

    @torch.no_grad()
    def calculate_raw_energy_metrics(
        self, pred_stack: Tensor, target_stack: Tensor, band: str, level: int
    ):
        metrics: dict[str, float | int] = {}
        metrics[f"{band}{level}_raw_pred_energy"] = (
            torch.mean(pred_stack**2).detach().item()
        )
        metrics[f"{band}{level}_raw_target_energy"] = (
            torch.mean(target_stack**2).detach().item()
        )

        metrics[f"{band}{level}_raw_error"] = (
            self.loss_fn(pred_stack.float(), target_stack.float()).detach().item()
        )

        return metrics

    @torch.no_grad()
    def calculate_cross_scale_consistency_metrics(
        self,
        pred_coeffs: dict[str, list[Tensor]],
        target_coeffs: dict[str, list[Tensor]],
    ) -> dict:
        """Calculate metrics for cross-scale consistency"""
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
                log_ratio_diff = abs(
                    math.log(pred_ratio + 1e-8) - math.log(target_ratio + 1e-8)
                )

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
        """Calculate correlation metrics between prediction and target wavelet coefficients"""
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
                denom = torch.sqrt(
                    torch.sum(pred_centered**2, dim=2)
                    * torch.sum(target_centered**2, dim=2)
                    + 1e-8
                )

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
        """Calculate metrics for directional consistency between bands"""
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
            hv_log_diff = abs(
                math.log(pred_hv_ratio + 1e-8) - math.log(target_hv_ratio + 1e-8)
            )

            # Diagonal to (horizontal+vertical) energy ratio
            pred_hh_energy = torch.mean(pred_coeffs["hh"][i - 1] ** 2).item()
            target_hh_energy = torch.mean(target_coeffs["hh"][i - 1] ** 2).item()

            pred_d_ratio = pred_hh_energy / (pred_hl_energy + pred_lh_energy + 1e-8)
            target_d_ratio = target_hh_energy / (
                target_hl_energy + target_lh_energy + 1e-8
            )
            diag_log_diff = abs(
                math.log(pred_d_ratio + 1e-8) - math.log(target_d_ratio + 1e-8)
            )

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
        """Calculate metrics for latent space regularity"""
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

    @torch.no_grad()
    def calculate_sparsity_metrics(
        self,
        coeffs: dict[str, list[Tensor]],
        reference_coeffs: dict[str, list[Tensor]] | None = None,
    ) -> dict:
        """Calculate sparsity metrics for wavelet coefficients"""
        metrics = {}
        band_sparsities = []
        band_non_zero_ratios = []

        for band in ["lh", "hl", "hh"]:
            for i in range(1, self.level + 1):
                coef = coeffs[band][i - 1]

                # L1 norm (sparsity measure)
                l1_norm = torch.mean(torch.abs(coef)).item()
                metrics[f"{band}{i}_l1_norm"] = l1_norm
                band_sparsities.append(l1_norm)

                # Additional sparsity metrics
                non_zero_ratio = torch.mean((torch.abs(coef) > 0.01).float()).item()
                metrics[f"{band}{i}_non_zero_ratio"] = non_zero_ratio
                band_non_zero_ratios.append(non_zero_ratio)

                # If reference coefficients provided, calculate relative sparsity
                if reference_coeffs is not None:
                    ref_coef = reference_coeffs[band][i - 1]
                    ref_l1_norm = torch.mean(torch.abs(ref_coef)).item()
                    rel_sparsity = l1_norm / (ref_l1_norm + 1e-8)
                    metrics[f"{band}{i}_relative_sparsity"] = rel_sparsity

        # Average sparsity across bands
        if band_sparsities:
            metrics["avg_sparsity_score"] = 1.0 / (
                sum(band_sparsities) / len(band_sparsities) + 1e-8
            )

        return metrics

    def smooth_timestep_weight(self, timestep):
        """Smooth weight transition instead of hard cutoff"""

        progress = 1.0 - (timestep / self.max_timestep)

        weight = torch.sigmoid((progress - 0.3) * 10)

        return weight

    # TODO: does not work right in terms of weighting in an appropriate range
    def noise_aware_weighting(
        self, timestep: Tensor, max_timestep: float, intensity=1.0
    ):
        """
        Adjust band weights based on diffusion timestep, maintaining reasonable magnitudes

        Args:
            timestep: Current diffusion timestep
            max_timestep: Maximum diffusion timestep
            intensity: Controls how strongly timestep affects weights (0.0-1.0)

        Returns:
            Dictionary of adjusted weights with reasonable magnitudes
        """
        # Calculate denoising progress (0.0 = noisy start, 1.0 = clean end)
        progress = 1.0 - (timestep / max_timestep)

        # Initialize adjusted weights dictionaries
        band_weights_adjusted = {}
        band_level_weights_adjusted = {}

        # Define target ranges for weights
        # These ensure weights stay within reasonable bounds regardless of input
        ll_range = (0.5, 2.0)  # Low-frequency weights
        hf_range = (0.01, 0.2)  # High-frequency weights (lh, hl)
        hh_range = (0.005, 0.1)  # Diagonal details weight (hh)

        # Determine sign for each weight - properly handling different types
        def get_sign(w):
            if isinstance(w, torch.Tensor):
                # For tensor weights: check if all values are positive
                if w.numel() > 1:
                    return 1 if (w > 0).all().item() else -1
                else:
                    return 1 if w.item() > 0 else -1
            else:
                # For float or int weights
                return 1 if w > 0 else -1

        # Get sign of each band weight (to preserve positive/negative direction)
        signs = {band: get_sign(weight) for band, weight in self.band_weights.items()}

        # Apply modulated weighting based on progress
        for band, weight in self.band_weights.items():
            if band == "ll":
                # For low frequency: high at start, decreases toward end
                # Map from progress to target range
                target_value = (
                    ll_range[0]
                    + (1.0 - progress) * (ll_range[1] - ll_range[0]) * intensity
                )
            elif band == "hh":
                # For diagonal details: low at start, increases toward end
                target_value = (
                    hh_range[0] + progress * (hh_range[1] - hh_range[0]) * intensity
                )
            else:  # "lh", "hl"
                # For horizontal/vertical details: low at start, increases toward end
                target_value = (
                    hf_range[0] + progress * (hf_range[1] - hf_range[0]) * intensity
                )

            # Apply sign to preserve direction
            target_value = target_value * signs[band]

            # Calculate blend factor - how much of original vs. target weight to use
            # Higher intensity means more influence from the target values
            blend_factor = min(
                intensity, 0.8
            )  # Cap at 0.8 to preserve some original weight

            # Create tamed weight by blending original (normalized) and target values
            if isinstance(weight, torch.Tensor) and weight.numel() > 1:
                # Handle tensor weights (multiple values)
                weight_mean = torch.abs(weight).mean()
                normalized_weight = weight / (weight_mean + 1e-8)
                # Blend between normalized weight and target
                blended_weight = (
                    1 - blend_factor
                ) * normalized_weight + blend_factor * target_value
                band_weights_adjusted[band] = blended_weight
            else:
                # Handle scalar weights
                weight_abs = (
                    abs(weight)
                    if isinstance(weight, (int, float))
                    else abs(weight.item())
                )
                normalized_weight = weight / (weight_abs + 1e-8)
                # Blend between normalized weight and target
                blended_weight = (
                    1 - blend_factor
                ) * normalized_weight + blend_factor * target_value
                band_weights_adjusted[band] = blended_weight

        # Similar approach for band_level_weights
        for key, weight in self.band_level_weights.items():
            band = key[:2]  # Extract band name (e.g., "ll" from "ll1")
            level = int(key[2:])  # Extract level number

            # Determine appropriate target range based on band and level
            if band == "ll":
                # Low frequency bands: higher weight early
                level_factor = level / self.level  # Lower levels have lower factor
                target_range = (
                    ll_range[0] * (1 - level_factor),
                    ll_range[1] * (1 - 0.3 * level_factor),
                )
                target_value = (
                    target_range[0]
                    + (1.0 - progress) * (target_range[1] - target_range[0]) * intensity
                )
            elif band == "hh":
                # Diagonal details: lower weight early
                level_factor = (
                    self.level - level + 1
                ) / self.level  # Higher levels have higher factor
                target_range = (hh_range[0] * level_factor, hh_range[1] * level_factor)
                target_value = (
                    target_range[0]
                    + progress * (target_range[1] - target_range[0]) * intensity
                )
            else:  # "lh", "hl"
                # Horizontal/vertical details: lower weight early
                level_factor = (
                    self.level - level + 1
                ) / self.level  # Higher levels have higher factor
                target_range = (hf_range[0] * level_factor, hf_range[1] * level_factor)
                target_value = (
                    target_range[0]
                    + progress * (target_range[1] - target_range[0]) * intensity
                )

            # Apply sign to preserve direction
            sign = 1 if weight > 0 else -1
            target_value = target_value * sign

            # Calculate blend factor
            blend_factor = min(intensity, 0.8)

            # Create tamed weight
            if isinstance(weight, torch.Tensor) and weight.numel() > 1:
                weight_mean = torch.abs(weight).mean()
                normalized_weight = weight / (weight_mean + 1e-8)
                blended_weight = (
                    1 - blend_factor
                ) * normalized_weight + blend_factor * target_value
            else:
                weight_abs = (
                    abs(weight)
                    if isinstance(weight, (int, float))
                    else abs(weight.item())
                )
                normalized_weight = weight / (weight_abs + 1e-8)
                blended_weight = (
                    1 - blend_factor
                ) * normalized_weight + blend_factor * target_value

            band_level_weights_adjusted[key] = blended_weight

        return band_weights_adjusted, band_level_weights_adjusted

    def set_loss_fn(self, loss_fn: LossCallable):
        """
        Set loss function to use. Wavelet loss wants l1 or huber loss.
        """
        self.loss_fn = loss_fn
