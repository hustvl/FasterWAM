"""Short real-checkpoint smoke for both paths, RTC off/on; no robot or downloads.

Use the serving CLI arguments plus --output-dir (must not already exist).
Writes configuration/artifact identities, timings and finite-output checks.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from serve_widowxai_policy import parse_args

from fasterwam.serve.widowxai_policy import ACTION_INFER_MODES, WidowXAIFasterWAMPolicy


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    result = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if stat.st_size < 10_000_000:
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    output, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Real-checkpoint smoke requires an available GPU")
    if args.allow_download:
        raise ValueError("This smoke test only uses local artifacts")
    output.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(args.model_base_path.resolve())
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    rtc = dataclasses.replace(args.rtc_config, enabled=True)
    policy = WidowXAIFasterWAMPolicy(
        run_dir=args.run_dir,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        norm_stats_path=args.norm_stats,
        task=args.task,
        text_embedding_cache_dir=args.text_embedding_cache_dir,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        action_infer_mode=args.action_infer_mode,
        seed=args.seed,
        control_fps=args.control_fps,
        rtc_config=rtc,
    )
    report = {
        "seed": args.seed,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "model": type(policy.model).__name__,
        "rtc": dataclasses.asdict(rtc),
        "config": identity(policy.artifacts.config_path),
        "checkpoint": identity(policy.artifacts.checkpoint_path),
        "normalization": identity(policy.artifacts.norm_stats_path),
        "data": "synthetic zero RGB/qpos; no dataset read",
        "results": [],
    }
    (output.output_dir / "manifest.json").write_text(json.dumps(report, indent=2))
    obs = {
        "qpos": np.zeros(14, dtype=np.float32),
        "images": {
            name: np.zeros(policy.raw_image_shape, dtype=np.uint8)
            for name in policy.camera_names
        },
    }
    for mode in ACTION_INFER_MODES:
        policy.action_infer_mode = mode
        policy._infer_fn = policy._resolve_infer_fn()
        previous = np.empty((0, 14), dtype=np.float32)
        for method in ("none", "rtc"):
            context = {
                "episode_id": f"smoke-{mode}",
                "request_id": len(report["results"]),
                "observation_step": 0,
                "control_dt": policy.control_dt,
                "previous_actions": previous,
                "inference_delay_steps": 2,
                "guidance_method": method,
            }
            torch.cuda.synchronize()
            start = time.monotonic()
            actions, _, terminate = policy.select_action(obs, chunk_context=context)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            assert (
                actions.shape == (policy.action_horizon, 14)
                and np.isfinite(actions).all()
                and not terminate
            )
            assert all(
                p.grad is None and not p.requires_grad
                for p in policy.model.parameters()
            )
            np.save(output.output_dir / f"{mode}-{method}.npy", actions)
            previous = actions[policy.action_horizon // 2 :].copy()
            report["results"].append(
                {
                    "mode": mode,
                    "guidance": method,
                    "seconds": elapsed,
                    "denoise_steps": policy.num_inference_steps,
                    "finite": True,
                }
            )
    # The directory is new for this run; never reuse an existing experiment path.
    (output.output_dir / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
