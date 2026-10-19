"""CPU RTC numerics and both real inference loops with tiny, analytic experts.

PYTHONPATH=src python scripts/test_rtc.py
Seed=42. Synthetic tensors/statistics only; no checkpoints/datasets/downloads.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from fasterwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fasterwam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign
from fasterwam.datasets.lerobot.transforms.relative_action import (
    MaskedRelativeJointTransform,
    RelativeJointTransform,
)
from fasterwam.datasets.lerobot.utils.normalizer import LinearNormalizer
from fasterwam.models.wan22.jointwam import JointWAM
from fasterwam.models.wan22.rtc import (
    RTCConfig,
    RTCGuidance,
    get_prefix_weights,
    rtc_corrected_flow,
)
from fasterwam.serve.widowxai_policy import (
    WidowXAIFasterWAMPolicy,
    WidowXAIOnlinePreprocessor,
)


class RTCTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_serving_config_precedence(self):
        from serve_widowxai_policy import parse_args

        config = Path(__file__).resolve().parents[1] / "configs/serve/rtc.yaml"
        with patch.object(sys, "argv", ["serve"]):
            defaults = parse_args()
        self.assertFalse(defaults.rtc_config.enabled)
        self.assertEqual(defaults.seed, 42)
        with patch.object(sys, "argv", ["serve", "--serving-config", str(config)]):
            configured = parse_args()
        self.assertTrue(configured.rtc_config.enabled)
        with patch.object(
            sys,
            "argv",
            [
                "serve",
                "--serving-config",
                str(config),
                "--no-rtc",
                "--rtc-max-guidance-weight",
                "0",
                "--action-infer-mode",
                "joint_rollout",
                "--seed",
                "7",
            ],
        ):
            overridden = parse_args()
        self.assertFalse(overridden.rtc_config.enabled)
        self.assertEqual(overridden.rtc_config.max_guidance_weight, 0)
        self.assertEqual(overridden.action_infer_mode, "joint_rollout")
        self.assertEqual(overridden.seed, 7)

    def test_prefix_known_values(self):
        linear = torch.tensor([1, 1, 0.8, 0.6, 0.4, 0.2, 0, 0, 0, 0])
        torch.testing.assert_close(get_prefix_weights(2, 6, 10, "linear"), linear)
        torch.testing.assert_close(
            get_prefix_weights(2, 6, 10, "exp"), linear * linear.expm1() / math.expm1(1)
        )
        torch.testing.assert_close(
            get_prefix_weights(2, 6, 10, "zeros"),
            torch.tensor([1.0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        )
        torch.testing.assert_close(
            get_prefix_weights(2, 6, 10, "ones"),
            torch.tensor([1.0, 1, 1, 1, 1, 1, 0, 0, 0, 0]),
        )
        for schedule in ("linear", "exp", "ones", "zeros"):
            self.assertEqual(get_prefix_weights(99, 0, 8, schedule).count_nonzero(), 0)
            torch.testing.assert_close(
                get_prefix_weights(99, 3, 5, schedule), torch.tensor([1.0, 1, 1, 0, 0])
            )

    def test_analytic_vjp_sign_and_time_conversion(self):
        # Dense horizon mixing makes a weighted-loss-squared implementation fail.
        matrix = nn.Parameter(
            torch.tensor([[0.2, 0.1, 0.0], [-0.3, 0.4, 0.2], [0.0, 0.1, -0.2]]),
            requires_grad=False,
        )
        x = torch.tensor([[[0.3], [-0.4], [0.2]]])
        prefix = torch.tensor([[[0.6], [0.1]]])
        guide = RTCGuidance(prefix, 0, RTCConfig(True, "linear", 5.0))
        sigma = 0.4

        def predict(value):
            return torch.einsum("ij,bjd->bid", matrix, value), None

        flow, _ = rtc_corrected_flow(predict, x, sigma, guide)
        u = predict(x)[0]
        estimate = x - sigma * u
        error = torch.zeros_like(x)
        error[:, :2] = (prefix - estimate[:, :2]) * torch.tensor([2 / 3, 1 / 3])[
            None, :, None
        ]
        correction = torch.einsum("ji,bjd->bid", torch.eye(3) - sigma * matrix, error)
        weight = ((1 - sigma) ** 2 + sigma**2) / (sigma * (1 - sigma))
        torch.testing.assert_close(flow, u - weight * correction)
        # Reference v=-u and dt=-delta_sigma gives exactly the same update.
        delta_sigma = -0.1
        torch.testing.assert_close(
            x + delta_sigma * flow, x + 0.1 * (-u + weight * correction)
        )
        self.assertIsNone(matrix.grad)
        self.assertFalse(flow.requires_grad)
        self.assertIsNone(x.grad)

    def test_zero_finite_endpoints_and_masked_prefix(self):
        x = torch.zeros((1, 4, 2))
        prefix = torch.ones((1, 2, 2))

        def predict(value):
            return value * 0.1, value * 2

        for sigma in (0.0, 0.000001, 0.5, 0.999999, 1.0):
            result, aux = rtc_corrected_flow(
                predict, x, sigma, RTCGuidance(prefix, 1, RTCConfig(True))
            )
            self.assertTrue(torch.isfinite(result).all())
            self.assertFalse(aux.requires_grad)
            self.assertTrue((result[:, :2] < 0).all())
            self.assertEqual(result[:, 2:].count_nonzero(), 0)
        result, _ = rtc_corrected_flow(
            predict, x, 0.7, RTCGuidance(prefix, 1, RTCConfig(True, "exp", 0))
        )
        torch.testing.assert_close(result, predict(x)[0], rtol=0, atol=0)
        with torch.no_grad():
            guided, _ = rtc_corrected_flow(
                predict, x, 0.7, RTCGuidance(prefix, 1, RTCConfig(True))
            )
        self.assertFalse(guided.requires_grad)

    def test_rebase_normalization_roundtrip_and_stepwise(self):
        shape = {
            "action": [{"key": "default", "raw_shape": 14, "shape": 14}],
            "state": [{"key": "default", "raw_shape": 14, "shape": 14}],
        }
        mask = [True] * 14
        mask[6] = mask[13] = False
        for transform in (
            MaskedRelativeJointTransform({"default": mask}),
            RelativeJointTransform(["default"]),
        ):
            for stepwise in (False, True):
                with self.subTest(
                    transform=type(transform).__name__, stepwise=stepwise
                ):
                    processor = FastWAMProcessor(
                        shape,
                        1,
                        2,
                        14,
                        14,
                        [transform],
                        stepwise,
                        "z-score",
                        None,
                        ConcatLeftAlign(),
                        None,
                        None,
                    )
                    stats = {}
                    for field in ("action", "state"):
                        stats[field] = {
                            "default": {
                                "global_mean": torch.zeros(14),
                                "global_std": torch.full((14,), 2.0),
                            }
                        }
                    stats["action"]["default"].update(
                        stepwise_mean=torch.zeros(4, 14),
                        stepwise_std=torch.arange(1.0, 5.0)[:, None].expand(4, 14),
                    )
                    processor._normalizer = LinearNormalizer(
                        shape, stepwise, "z-score", None, stats
                    )
                    pre = WidowXAIOnlinePreprocessor.__new__(WidowXAIOnlinePreprocessor)
                    pre.processor, pre.action_dim, pre.action_horizon = processor, 14, 4
                    qpos = np.linspace(0.1, 0.7, 14, dtype=np.float32)
                    action = np.tile(qpos + 0.2, (4, 1))
                    original = action.copy()
                    normalized = pre.normalize_previous_actions(action, qpos)
                    np.testing.assert_allclose(
                        pre.denormalize_action(normalized, pre.prepare_proprio(qpos)),
                        action,
                        atol=1e-6,
                    )
                    short = pre.normalize_previous_actions(action[:2], qpos)
                    torch.testing.assert_close(short, normalized[:2])
                    rebased = pre.normalize_previous_actions(action, qpos + 0.1)
                    if isinstance(transform, MaskedRelativeJointTransform):
                        torch.testing.assert_close(
                            normalized[:, [6, 13]], rebased[:, [6, 13]]
                        )
                    self.assertFalse(torch.equal(normalized[:, 0], rebased[:, 0]))
                    np.testing.assert_array_equal(action, original)

    def test_policy_passes_rebased_context_and_rejects_missing_signature(self):
        policy = WidowXAIFasterWAMPolicy.__new__(WidowXAIFasterWAMPolicy)
        policy.supports_rtc = True
        policy.rtc_config = RTCConfig(True)
        policy.action_horizon, policy.num_video_frames = 4, 5
        policy.context, policy.context_mask = (
            torch.ones(1, 2),
            torch.ones(1, dtype=torch.bool),
        )
        policy.num_inference_steps, policy.seed = 2, 42
        calls = []
        policy.preprocessor = SimpleNamespace(
            prepare_observation=lambda obs: (torch.zeros(3, 16, 16), torch.zeros(14)),
            normalize_previous_actions=lambda actions, qpos: torch.from_numpy(
                actions - qpos
            ),
            denormalize_action=lambda action, proprio: action.numpy(),
        )

        def infer(rtc_guidance=None):
            calls.append(rtc_guidance)
            return {"action": torch.zeros(4, 14)}

        policy._infer_fn = infer
        context = {
            "guidance_method": "rtc",
            "previous_actions": np.ones((2, 14), dtype=np.float32),
            "inference_delay_steps": 2,
        }
        policy.select_action(
            {"qpos": np.full(14, 0.25, dtype=np.float32)}, chunk_context=context
        )
        torch.testing.assert_close(
            calls[0].previous_actions, torch.full((1, 2, 14), 0.75)
        )
        policy._infer_fn = lambda: {"action": torch.zeros(4, 14)}
        with self.assertRaises(TypeError):
            policy.select_action({"qpos": np.zeros(14)}, chunk_context=context)


class TinyVideo(nn.Module):
    def pre_dit(self, x, timestep, context, context_mask, **kwargs):
        return {
            "tokens": x.flatten(2).transpose(1, 2),
            "freqs": None,
            "t_mod": None,
            "context": context,
            "context_mask": context_mask,
            "meta": {"tokens_per_frame": x.shape[-1] * x.shape[-2], "shape": x.shape},
        }

    def post_dit(self, tokens, pre):
        return tokens.transpose(1, 2).reshape(pre["meta"]["shape"])

    def build_video_to_video_mask(self, video_seq_len, video_tokens_per_frame, device):
        return torch.ones(video_seq_len, video_seq_len, dtype=torch.bool, device=device)


class TinyAction(nn.Module):
    action_dim = 2

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.2), requires_grad=False)

    def pre_dit(self, action_tokens, timestep, context, context_mask):
        return {
            "tokens": self.weight * action_tokens,
            "freqs": None,
            "t_mod": None,
            "context": context,
            "context_mask": context_mask,
        }

    def post_dit(self, tokens, pre):
        return tokens


class TinyMoT(nn.Module):
    num_layers = 1

    def forward(self, embeds_all, **kwargs):
        return {
            "video": 0.1 * embeds_all["video"],
            "action": embeds_all["action"] + 0.01 * embeds_all["video"].mean(),
        }

    def prefill_video_cache(self, video_tokens, **kwargs):
        assert not video_tokens.requires_grad and not torch.is_grad_enabled()
        return [{"condition": video_tokens.mean()}]

    def forward_action_with_video_cache(self, action_tokens, video_kv_cache, **kwargs):
        assert all(
            not value.requires_grad
            for entry in video_kv_cache
            for value in entry.values()
        )
        return action_tokens + 0.01 * video_kv_cache[0]["condition"]


class TinyVAE(nn.Module):
    temporal_downsample_factor = 4
    upsampling_factor = 8

    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(z_dim=1)


class LoopTests(unittest.TestCase):
    def test_both_production_loops_use_common_rtc(self):
        for method in ("infer_action", "infer_action_one_pass_future_cache"):
            with self.subTest(method=method):
                model = JointWAM(
                    TinyVideo(), TinyAction(), TinyMoT(), TinyVAE(), text_dim=2
                )
                model._encode_input_image_latents_tensor = lambda **kwargs: torch.zeros(
                    1, 1, 1, 2, 2
                )
                model.requires_grad_(False)
                args = {
                    "prompt": None,
                    "input_image": torch.zeros(1, 3, 16, 16),
                    "action_horizon": 4,
                    "num_video_frames": 5,
                    "context": torch.ones(1, 2),
                    "context_mask": torch.ones(1, dtype=torch.bool),
                    "num_inference_steps": 3,
                    "seed": 42,
                }
                infer = getattr(model, method)
                original = infer(**args)["action"]
                prefix = torch.ones(1, 2, 2)
                zero = RTCGuidance(prefix, 1, RTCConfig(True, "exp", 0))
                torch.testing.assert_close(
                    infer(**args, rtc_guidance=zero)["action"], original, rtol=0, atol=0
                )
                with patch(
                    "fasterwam.models.wan22.jointwam.rtc_corrected_flow",
                    wraps=rtc_corrected_flow,
                ) as common:
                    guided = infer(
                        **args, rtc_guidance=RTCGuidance(prefix, 1, RTCConfig(True))
                    )["action"]
                    self.assertEqual(common.call_count, 3)
                    for call in common.call_args_list:
                        self.assertFalse(call.args[1].requires_grad)
                self.assertTrue(torch.isfinite(guided).all())
                self.assertFalse(torch.equal(original, guided))
                self.assertTrue(
                    all(
                        parameter.grad is None and not parameter.requires_grad
                        for parameter in model.parameters()
                    )
                )


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
