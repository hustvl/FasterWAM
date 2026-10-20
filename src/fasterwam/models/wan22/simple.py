"""Gradient-free clean-action mixing in FasterWAM's sigma: 1 -> 0 convention."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .rtc import get_prefix_weights


@dataclass(frozen=True)
class SimpleConfig:
    enabled: bool = False
    prefix_attention_schedule: str = "exp"

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("Simple enabled must be a boolean")
        if self.prefix_attention_schedule not in ("linear", "exp", "ones", "zeros"):
            raise ValueError("Unknown simple prefix_attention_schedule")


@dataclass(frozen=True)
class SimpleGuidance:
    # [B,L,D], rebased and normalized against the current observation.
    previous_actions: torch.Tensor
    inference_delay_steps: int
    config: SimpleConfig

    def __post_init__(self):
        if type(self.inference_delay_steps) is not int or self.inference_delay_steps < 0:
            raise ValueError("inference_delay_steps must be a nonnegative integer")
        if (
            self.previous_actions.ndim != 3
            or not torch.is_floating_point(self.previous_actions)
            or not torch.isfinite(self.previous_actions).all()
        ):
            raise ValueError("Simple prefix must be a finite floating-point [B,L,D] tensor")


@torch.no_grad()
def simple_denoise_step(
    latent: torch.Tensor,
    flow: torch.Tensor,
    next_latent: torch.Tensor,
    sigma: torch.Tensor | float,
    delta_sigma: torch.Tensor | float,
    guidance: SimpleGuidance,
) -> torch.Tensor:
    """Mix clean predictions and use the mixture for the next denoise input.

    The caller supplies the ordinary scheduler update as next_latent, preserving
    its exact values outside the mask and when guidance is disabled/empty.
    With tau = 1-sigma and v = -flow:
        estimate = latent - sigma * flow
        target = weights * previous + (1-weights) * estimate
        corrected_flow = (latent - target) / sigma
    The equivalent correction below avoids constructing a large flow near zero
    sigma. Evaluate only at positive sigma; the scheduler's terminal zero is an
    output, never another model evaluation. No VJP or input gradients are used.
    """
    previous = guidance.previous_actions.detach().to(
        device=latent.device, dtype=torch.float32
    )
    if (
        latent.ndim != 3
        or flow.shape != latent.shape
        or next_latent.shape != latent.shape
        or previous.shape[0] != latent.shape[0]
        or previous.shape[2] != latent.shape[2]
        or previous.shape[1] > latent.shape[1]
    ):
        raise ValueError("Simple prefix/flow/next latent dimensions must match [B,H,D]")
    sigma = torch.as_tensor(sigma, device=latent.device, dtype=torch.float32)
    delta = torch.as_tensor(delta_sigma, device=latent.device, dtype=torch.float32)
    if sigma.numel() != 1 or not torch.isfinite(sigma).all() or not (0 < sigma <= 1):
        raise ValueError("Simple sigma must be a finite scalar in (0,1]")
    if delta.numel() != 1 or not torch.isfinite(delta).all() or delta > 0:
        raise ValueError("Simple delta_sigma must be a finite nonpositive scalar")
    length = previous.shape[1]
    weights = get_prefix_weights(
        guidance.inference_delay_steps, length, latent.shape[1],
        guidance.config.prefix_attention_schedule, device=latent.device,
    )[:length, None]
    if not guidance.config.enabled or not weights.any():
        return next_latent.detach()

    estimate = latent[:, :length].float() - sigma * flow[:, :length].float()
    correction = (-delta / sigma) * weights * (previous - estimate)
    result = next_latent.detach().clone()
    result[:, :length] = torch.where(
        weights > 0,
        (next_latent[:, :length].float() + correction).to(next_latent.dtype),
        next_latent[:, :length],
    )
    if not torch.isfinite(result).all():
        raise ValueError("Simple mixing produced nonfinite actions")
    return result
