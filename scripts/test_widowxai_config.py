"""CPU checks for adapting newer training configs to the serving runtime."""

import inspect
import unittest

from omegaconf import OmegaConf

from fasterwam.runtime import create_fasterwam
from fasterwam.serve.widowxai_policy import build_inference_model_config


def training_config():
    return OmegaConf.create({"model": {
        "_target_": "fasterwam.runtime.create_fasterwam",
        "mot_interaction_mode": "v_to_a",
        "action_to_video_layers": [],
        "a2v_temporal_mask_mode": "group_diagonal",
        "a2v_gate_bias_init": -4.0,
        "action_loss_excluded_task_ids": [1, 2],
        "action_loss_excluded_task_names": ["failed attempt", "random policy"],
        "random_future_num_blocks": None,
        "vae_encode_microbatch_size": 1,
        "video_kv_fusion": "interval_weighted_sum",
        "condition_layers": [0, 4, 8, 12, 16, 20, 24, 28],
        "skip_dit_load_from_pretrain": False,
        "action_dit_pretrained_path": "/training-only/backbone.pt",
        "mot_checkpoint_mixed_attn": True,
        "video_dit_config": {"use_gradient_checkpointing": True},
        "action_dit_config": {"use_gradient_checkpointing": True},
    }})


class ConfigTests(unittest.TestCase):
    def test_training_options_adapt_without_changing_architecture_or_source(self):
        cfg = training_config()
        before = OmegaConf.to_container(cfg)
        adapted = build_inference_model_config(cfg)
        self.assertEqual(OmegaConf.to_container(cfg), before)
        self.assertEqual(adapted.video_kv_fusion, cfg.model.video_kv_fusion)
        self.assertEqual(adapted.condition_layers, cfg.model.condition_layers)
        unsupported = set(adapted) - set(inspect.signature(create_fasterwam).parameters) - {"_target_"}
        self.assertEqual(unsupported, set())
        self.assertTrue(adapted.skip_dit_load_from_pretrain)
        self.assertIsNone(adapted.action_dit_pretrained_path)
        self.assertFalse(adapted.video_dit_config.use_gradient_checkpointing)
        self.assertFalse(adapted.action_dit_config.use_gradient_checkpointing)

    def test_incompatible_attention_and_sampling_are_rejected(self):
        for key, value in (("mot_interaction_mode", "bidirectional"),
                           ("action_to_video_layers", [0]),
                           ("random_future_num_blocks", 4)):
            with self.subTest(key=key):
                cfg = training_config()
                cfg.model[key] = value
                with self.assertRaises(ValueError):
                    build_inference_model_config(cfg)


if __name__ == "__main__":
    unittest.main()
