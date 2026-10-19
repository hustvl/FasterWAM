"""Inference-only RTC in FasterWAM's sigma: 1 -> 0 convention.

Reference: real-time-chunking/src/model.py, realtime_action/get_prefix_weights.
Reference t = 1 - sigma and v = -u, hence u_corrected = u - weight * VJP.
The scheduler already supplies negative sigma deltas; do not reverse them again.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class RTCConfig:
    enabled: bool = False
    prefix_attention_schedule: str = "exp"
    max_guidance_weight: float = 5.0

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("RTC enabled must be a boolean")
        if self.prefix_attention_schedule not in ("linear", "exp", "ones", "zeros"):
            raise ValueError("Unknown prefix_attention_schedule")
        if not math.isfinite(self.max_guidance_weight) or self.max_guidance_weight < 0:
            raise ValueError("max_guidance_weight must be finite and nonnegative")


@dataclass(frozen=True)
class RTCGuidance:
    # [B,L,D], freshly rebased/normalized for this observation.
    previous_actions: torch.Tensor
    inference_delay_steps: int
    config: RTCConfig

    def __post_init__(self):
        if (
            type(self.inference_delay_steps) is not int
            or self.inference_delay_steps < 0
        ):
            raise ValueError("inference_delay_steps must be a nonnegative integer")
        if (
            self.previous_actions.ndim != 3
            or not torch.isfinite(self.previous_actions).all()
        ):
            raise ValueError("RTC prefix must be a finite [B,L,D] tensor")


def get_prefix_weights(
    start: int, end: int, total: int, schedule: str = "exp", *, device=None
):
    if min(start, end, total) < 0 or end > total:
        raise ValueError("Require start >= 0 and 0 <= end <= total")
    start = min(start, end)
    index = torch.arange(total, device=device)
    if schedule == "ones":
        weights = torch.ones(total, device=device)
    elif schedule == "zeros":
        weights = (index < start).float()
    elif schedule in ("linear", "exp"):
        weights = ((start - 1 - index.float()) / (end - start + 1) + 1).clamp(0, 1)
        if schedule == "exp":
            weights = weights * torch.expm1(weights) / math.expm1(1)
    else:
        raise ValueError(f"Invalid prefix schedule: {schedule}")
    return torch.where(index < end, weights, 0)


def guidance_weight(sigma: torch.Tensor, maximum: float):
    # c * inv_r2 = ((1-sigma)^2 + sigma^2) / (sigma*(1-sigma)).
    # Both endpoints have capped limits. Avoid 0/0 and fp16/bf16 overflow.
    sigma = sigma.float()
    t = 1 - sigma
    denominator = (sigma * t).clamp_min(torch.finfo(torch.float32).tiny)
    return ((t.square() + sigma.square()) / denominator).clamp(max=maximum)


def rtc_corrected_flow(
    predict: Callable[[torch.Tensor], tuple[torch.Tensor, Any]],
    latent: torch.Tensor,
    sigma: torch.Tensor | float,
    guidance: RTCGuidance,
) -> tuple[torch.Tensor, Any]:
    """One VJP, with the prefix weight applied exactly once to its cotangent.

    predict returns (u, auxiliary output). No parameter gradients are accumulated;
    callers freeze model parameters. Inputs and outputs detach at every step.
    Auxiliary video predictions retain their original values, without a graph.
    """
    previous = guidance.previous_actions.detach().to(
        device=latent.device, dtype=torch.float32
    )
    if (
        latent.ndim != 3
        or previous.shape[0] != latent.shape[0]
        or previous.shape[2] != latent.shape[2]
    ):
        raise ValueError("RTC prefix and latent batch/action dimensions must match")
    length, total = previous.shape[1], latent.shape[1]
    if length > total:
        raise ValueError("RTC prefix exceeds action horizon")
    sigma = torch.as_tensor(sigma, device=latent.device, dtype=torch.float32)
    if sigma.numel() != 1 or not torch.isfinite(sigma).all() or not (0 <= sigma <= 1):
        raise ValueError("RTC sigma must be a finite scalar in [0,1]")
    weights = get_prefix_weights(
        guidance.inference_delay_steps,
        length,
        total,
        guidance.config.prefix_attention_schedule,
        device=latent.device,
    )
    if (
        not guidance.config.enabled
        or guidance.config.max_guidance_weight == 0
        or not weights.any()
    ):
        with torch.no_grad():
            return predict(latent.detach())
    if torch.is_inference_mode_enabled():
        raise RuntimeError("RTC requires no_grad, not inference_mode, around inference")
    with torch.enable_grad():
        x = latent.detach().requires_grad_(True)
        flow, auxiliary = predict(x)
        estimate = x.float() - sigma * flow.float()
        # Padding is a zero cotangent, including when the model mixes horizon tokens.
        error = torch.zeros_like(estimate)
        error[:, :length] = (previous - estimate[:, :length]) * weights[:length, None]
        (correction,) = torch.autograd.grad(
            estimate,
            x,
            grad_outputs=error.detach(),
            create_graph=False,
            retain_graph=False,
        )
        corrected = (
            flow.float()
            - guidance_weight(sigma, guidance.config.max_guidance_weight)
            * correction.float()
        )
    corrected = corrected.to(latent.dtype).detach()
    if not torch.isfinite(corrected).all():
        raise ValueError("RTC produced nonfinite flow")
    if isinstance(auxiliary, torch.Tensor):
        auxiliary = auxiliary.detach()
    return corrected, auxiliary
