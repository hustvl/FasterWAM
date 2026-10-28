#!/usr/bin/env python3
"""Compare online two-wrist preprocessing against a training-pipeline sample."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fasterwam.serve.widowxai_policy import (
    DEFAULT_RUN_DIR,
    WidowXAIOnlinePreprocessor,
    resolve_run_artifacts,
)
from fasterwam.utils import misc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--norm-stats", type=Path, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = resolve_run_artifacts(
        run_dir=args.run_dir,
        config_path=args.config,
        norm_stats_path=args.norm_stats,
    )
    cfg = OmegaConf.load(artifacts.config_path)

    with tempfile.TemporaryDirectory(prefix="fasterwam-online-validation-") as work_dir:
        misc.register_work_dir(work_dir)
        dataset = instantiate(
            cfg.data.val,
            pretrained_norm_stats=str(artifacts.norm_stats_path),
        )
        base_dataset = dataset.lerobot_dataset
        raw_sample = base_dataset.multi_dataset[args.sample_index]
        images = {
            meta["key"]: base_dataset._get_image(meta, raw_sample)[0].permute(1, 2, 0).numpy()
            for meta in base_dataset.image_meta
        }
        qpos = base_dataset._get_state(base_dataset.state_meta[0], raw_sample)[0].numpy()
        expected_action = base_dataset._get_action(base_dataset.action_meta[0], raw_sample).float()
        reference = dataset._get(args.sample_index)

        online = WidowXAIOnlinePreprocessor(cfg, artifacts.norm_stats_path)
        actual_image, actual_proprio = online.prepare_observation({"images": images, "qpos": qpos})
        actual_action = torch.from_numpy(
            online.denormalize_action(reference["action"], reference["proprio"][0])
        )

        reference_image = reference["video"][:, 0]
        reference_proprio = reference["proprio"][0].float()
        image_max_abs_diff = float((actual_image - reference_image).abs().max().item())
        proprio_max_abs_diff = float((actual_proprio - reference_proprio).abs().max().item())
        action_max_abs_diff = float((actual_action - expected_action).abs().max().item())
        result = {
            "sample_index": args.sample_index,
            "drop_top_view": online.drop_top_view,
            "camera_names": list(online.camera_names),
            "raw_images": {
                name: {"shape": list(image.shape), "dtype": str(image.dtype)}
                for name, image in images.items()
            },
            "qpos_shape": list(qpos.shape),
            "model_image_shape": list(actual_image.shape),
            "proprio_shape": list(actual_proprio.shape),
            "action_shape": list(actual_action.shape),
            "num_video_frames": online.num_video_frames,
            "image_max_abs_diff": image_max_abs_diff,
            "proprio_max_abs_diff": proprio_max_abs_diff,
            "action_max_abs_diff": action_max_abs_diff,
        }
        print(json.dumps(result, indent=2))

        if not torch.equal(actual_image, reference_image):
            raise AssertionError(f"Online image mismatch: max_abs_diff={image_max_abs_diff}")
        if not torch.equal(actual_proprio, reference_proprio):
            raise AssertionError(f"Online proprio mismatch: max_abs_diff={proprio_max_abs_diff}")
        if not torch.allclose(actual_action, expected_action, atol=1e-6, rtol=1e-6):
            raise AssertionError(f"Online action inverse mismatch: max_abs_diff={action_max_abs_diff}")
        print("[validate-online] PASS")


if __name__ == "__main__":
    main()
