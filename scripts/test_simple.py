"""CPU checks for simple clean-action mixing. Seed=42; no data or checkpoints.

PYTHONPATH=src python scripts/test_simple.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from test_rtc import TinyAction, TinyMoT, TinyVAE, TinyVideo

from fasterwam.models.wan22.jointwam import JointWAM
from fasterwam.models.wan22.rtc import RTCConfig, RTCGuidance, get_prefix_weights
from fasterwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)
from fasterwam.models.wan22.simple import (
    SimpleConfig,
    SimpleGuidance,
    simple_denoise_step,
)
from fasterwam.serve.widowxai_policy import WidowXAIFasterWAMPolicy


class SimpleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_matches_reference_for_uniform_and_shifted_schedules(self):
        # Evaluate the user's tau/velocity equations independently of the sigma
        # correction, including horizon mixing and a short valid prefix.
        previous = torch.randn(1, 2, 2)
        guide = SimpleGuidance(previous, 1, SimpleConfig(True))
        matrix = torch.tensor([
            [0.2, 0.1, 0.0, -0.1], [0.0, 0.4, -0.2, 0.1],
            [0.1, 0.0, -0.3, 0.2], [0.0, 0.1, 0.1, 0.3],
        ])
        weights = get_prefix_weights(1, 2, 4, "exp")[None, :, None]
        old = torch.zeros(1, 4, 2)
        old[:, :2] = previous
        for shift in (1.0, 5.0):
            with self.subTest(shift=shift):
                scheduler = WanContinuousFlowMatchScheduler(shift=shift)
                times, deltas = scheduler.build_inference_schedule(
                    6, torch.device("cpu"), torch.float32
                )
                actual = torch.randn_like(old)
                expected = actual.clone()
                for timestep, delta in zip(times, deltas):
                    sigma = timestep / scheduler.num_train_timesteps
                    tau = 1 - sigma
                    velocity = -torch.einsum("ij,bjd->bid", matrix, expected)
                    predicted = expected + (1 - tau) * velocity
                    target = weights * old + (1 - weights) * predicted
                    expected = expected + (-delta) * (target - expected) / (1 - tau)

                    flow = torch.einsum("ij,bjd->bid", matrix, actual)
                    baseline = scheduler.step(flow, delta, actual)
                    actual = simple_denoise_step(
                        actual, flow, baseline, sigma, delta, guide
                    )
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_zero_empty_disabled_and_valid_length(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            x = torch.randn(2, 4, 3, dtype=dtype, requires_grad=True)
            flow = 0.2 * x
            baseline = WanContinuousFlowMatchScheduler.step(flow, torch.tensor(-0.2), x)
            previous = torch.ones(2, 2, 3)
            for guide in (
                SimpleGuidance(previous, 0, SimpleConfig(True, "zeros")),
                SimpleGuidance(previous[:, :0], 1, SimpleConfig(True)),
                SimpleGuidance(previous, 1, SimpleConfig(False)),
            ):
                with self.subTest(dtype=dtype, config=guide.config):
                    result = simple_denoise_step(x, flow, baseline, 0.7, -0.2, guide)
                    torch.testing.assert_close(result, baseline.detach(), rtol=0, atol=0)
                    self.assertFalse(result.requires_grad)
            originals = [value.detach().clone() for value in (x, flow, baseline, previous)]
            result = simple_denoise_step(
                x, flow, baseline, 0.7, -0.2,
                SimpleGuidance(previous, 1, SimpleConfig(True)),
            )
            torch.testing.assert_close(result[:, 2:], baseline[:, 2:], rtol=0, atol=0)
            self.assertFalse(result.requires_grad)
            for value, original in zip((x, flow, baseline, previous), originals):
                torch.testing.assert_close(value, original, rtol=0, atol=0)

    def test_final_step_full_weight_and_small_sigma(self):
        x = torch.randn(1, 4, 2)
        flow = torch.randn_like(x)
        previous = torch.randn(1, 3, 2)
        guide = SimpleGuidance(previous, 1, SimpleConfig(True, "ones"))
        for sigma in (1.0, 0.2, 1e-8, 1e-30):
            with self.subTest(sigma=sigma):
                baseline = WanContinuousFlowMatchScheduler.step(
                    flow, torch.tensor(-sigma), x
                )
                result = simple_denoise_step(x, flow, baseline, sigma, -sigma, guide)
                torch.testing.assert_close(result[:, :3], previous, atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(result[:, 3:], baseline[:, 3:], atol=0, rtol=0)
                self.assertTrue(torch.isfinite(result).all())

    def test_invalid_conditions_and_no_terminal_evaluation(self):
        x = torch.zeros(1, 4, 2)
        guide = SimpleGuidance(x[:, :2], 0, SimpleConfig(True))
        for sigma, delta in ((0, 0), (-0.1, -0.1), (1.1, -0.1),
                             (float("nan"), -0.1), (0.5, float("inf")), (0.5, 0.1)):
            with self.subTest(sigma=sigma, delta=delta), self.assertRaises(ValueError):
                simple_denoise_step(x, x, x, sigma, delta, guide)
        for previous in (torch.zeros(1, 2), torch.full((1, 2, 2), float("nan"))):
            with self.assertRaises(ValueError):
                SimpleGuidance(previous, 0, SimpleConfig(True))
        for delay in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                SimpleGuidance(x, delay, SimpleConfig(True))
        for previous in (torch.zeros(2, 2, 2), torch.zeros(1, 2, 3), torch.zeros(1, 5, 2)):
            with self.assertRaises(ValueError):
                simple_denoise_step(
                    x, x, x, 0.5, -0.1, SimpleGuidance(previous, 0, SimpleConfig(True))
                )
        with self.assertRaises(ValueError):
            SimpleConfig(enabled="true")
        with self.assertRaises(ValueError):
            SimpleConfig(prefix_attention_schedule="unknown")

    def test_serving_config_precedence_and_independent_enablement(self):
        from serve_widowxai_policy import parse_args

        config = Path(__file__).resolve().parents[1] / "configs/serve/simple.yaml"
        with patch.object(sys, "argv", ["serve"]):
            defaults = parse_args()
        self.assertFalse(defaults.simple_config.enabled)
        self.assertFalse(defaults.rtc_config.enabled)
        with patch.object(sys, "argv", ["serve", "--serving-config", str(config)]):
            configured = parse_args()
        self.assertEqual(configured.simple_config, SimpleConfig(True, "exp"))
        self.assertFalse(configured.rtc_config.enabled)
        with patch.object(sys, "argv", [
            "serve", "--serving-config", str(config), "--no-simple",
            "--simple-prefix-schedule", "linear", "--rtc", "--seed", "7",
            "--action-infer-mode", "joint_rollout",
        ]):
            overridden = parse_args()
        self.assertEqual(overridden.simple_config, SimpleConfig(False, "linear"))
        self.assertTrue(overridden.rtc_config.enabled)
        self.assertEqual(overridden.seed, 7)
        self.assertEqual(overridden.action_infer_mode, "joint_rollout")
        with patch.object(sys, "argv", ["serve", "--rtc", "--simple"]):
            both = parse_args()
        self.assertTrue(both.simple_config.enabled and both.rtc_config.enabled)

    def test_policy_rebases_and_never_drops_simple_argument(self):
        policy = WidowXAIFasterWAMPolicy.__new__(WidowXAIFasterWAMPolicy)
        policy.simple_config, policy.supports_simple = SimpleConfig(True), True
        policy.supports_rtc = False
        policy.action_horizon, policy.num_video_frames = 4, 5
        policy.context, policy.context_mask = torch.ones(1, 2), torch.ones(1, dtype=torch.bool)
        policy.num_inference_steps, policy.seed = 2, 42
        policy.preprocessor = SimpleNamespace(
            prepare_observation=lambda obs: (torch.zeros(3, 16, 16), torch.zeros(14)),
            normalize_previous_actions=lambda actions, qpos: torch.from_numpy(actions - qpos),
            denormalize_action=lambda action, proprio: action.numpy(),
        )
        calls = []

        def infer(simple_guidance=None):
            calls.append(simple_guidance)
            return {"action": torch.zeros(4, 14)}

        def unguided_infer():
            return {"action": torch.zeros(4, 14)}

        policy._infer_fn = infer
        context = {
            "guidance_method": "simple", "previous_actions": np.ones((2, 14), dtype=np.float32),
            "inference_delay_steps": 2,
        }
        for qpos in (0.25, 0.5):
            policy.select_action(
                {"qpos": np.full(14, qpos, dtype=np.float32)}, chunk_context=context
            )
            self.assertIsInstance(calls[-1], SimpleGuidance)
            torch.testing.assert_close(calls[-1].previous_actions, torch.full((1, 2, 14), 1 - qpos))
        for mode, name in (
            ("joint_rollout", "infer_action"),
            ("one_pass_future_cache", "infer_action_one_pass_future_cache"),
        ):
            policy.action_infer_mode = mode
            policy.model = SimpleNamespace(**{name: infer})
            self.assertIs(policy._resolve_infer_fn(), infer)
            setattr(policy.model, name, unguided_infer)
            with self.assertRaisesRegex(TypeError, "simple_guidance"):
                policy._resolve_infer_fn()
        policy._infer_fn = unguided_infer
        with self.assertRaises(TypeError):
            policy.select_action({"qpos": np.zeros(14)}, chunk_context=context)
        policy._infer_fn, policy.supports_simple = infer, False
        with self.assertRaises(ValueError):
            policy.select_action({"qpos": np.zeros(14)}, chunk_context=context)


class SimpleLoopTests(unittest.TestCase):
    def test_both_loops_feed_mixed_latent_into_next_prediction_without_grad(self):
        for method, predictor in (
            ("infer_action", "_predict_joint_noise"),
            ("infer_action_one_pass_future_cache", "_predict_action_noise_with_cache"),
        ):
            with self.subTest(method=method):
                model = JointWAM(TinyVideo(), TinyAction(), TinyMoT(), TinyVAE(), text_dim=2)
                model._encode_input_image_latents_tensor = lambda **kwargs: torch.zeros(1, 1, 1, 2, 2)
                model.requires_grad_(False)
                args = {
                    "prompt": None, "input_image": torch.zeros(1, 3, 16, 16),
                    "action_horizon": 4, "num_video_frames": 5, "context": torch.ones(1, 2),
                    "context_mask": torch.ones(1, dtype=torch.bool),
                    "num_inference_steps": 3, "seed": 42,
                }
                infer = getattr(model, method)
                prefix = torch.ones(1, 2, 2)
                original = infer(**args)["action"]
                for guide in (
                    SimpleGuidance(prefix, 0, SimpleConfig(True, "zeros")),
                    SimpleGuidance(prefix[:, :0], 1, SimpleConfig(True)),
                    SimpleGuidance(prefix, 1, SimpleConfig(False)),
                ):
                    torch.testing.assert_close(
                        infer(**args, simple_guidance=guide)["action"], original, rtol=0, atol=0
                    )
                model_inputs, mixed_outputs, video_outputs = [], [], []
                predict = getattr(model, predictor)

                def record_prediction(
                    model_inputs=model_inputs, predict=predict, video_outputs=video_outputs, **kwargs,
                ):
                    self.assertFalse(torch.is_grad_enabled())
                    self.assertFalse(kwargs["latents_action"].requires_grad)
                    model_inputs.append(kwargs["latents_action"].clone())
                    result = predict(**kwargs)
                    if isinstance(result, tuple):
                        video_outputs.append(result[0].clone())
                    return result

                def record_mix(*values, mixed_outputs=mixed_outputs):
                    self.assertFalse(torch.is_grad_enabled())
                    self.assertGreater(float(values[3]), 0)
                    result = simple_denoise_step(*values)
                    mixed_outputs.append(result.clone())
                    return result

                with (
                    patch.object(model, predictor, side_effect=record_prediction) as prediction,
                    patch("fasterwam.models.wan22.jointwam.simple_denoise_step", side_effect=record_mix),
                    patch.object(model.mot, "prefill_video_cache", wraps=model.mot.prefill_video_cache) as cache,
                    patch("torch.autograd.grad", side_effect=AssertionError("simple must not use VJP")),
                ):
                    guided = infer(
                        **args, simple_guidance=SimpleGuidance(prefix, 1, SimpleConfig(True))
                    )["action"]
                self.assertEqual(prediction.call_count, 3)
                self.assertEqual(len(mixed_outputs), 3)
                self.assertEqual(cache.call_count, int(method.endswith("future_cache")))
                for index in (1, 2):
                    torch.testing.assert_close(model_inputs[index], mixed_outputs[index - 1], rtol=0, atol=0)
                torch.testing.assert_close(guided, mixed_outputs[-1][0], rtol=0, atol=0)
                self.assertFalse(torch.equal(guided, original))
                self.assertTrue(torch.isfinite(guided).all())
                self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))
                if video_outputs:
                    # This tiny MoT respects JointWAM's video-independent-of-action mask.
                    guided_video = list(video_outputs)
                    video_outputs.clear()
                    with patch.object(model, predictor, side_effect=record_prediction):
                        infer(**args)
                    for actual, expected in zip(video_outputs, guided_video):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                    infer(
                        **args, rtc_guidance=RTCGuidance(prefix, 1, RTCConfig(True)),
                        simple_guidance=SimpleGuidance(prefix, 1, SimpleConfig(True)),
                    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
