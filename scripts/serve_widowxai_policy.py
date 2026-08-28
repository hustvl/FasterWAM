#!/usr/bin/env python3
"""Serve a trained WidowXAI FasterWAM policy over WebSocket."""

from __future__ import annotations

import argparse
import os
import socket
from pathlib import Path

from fasterwam.serve import WebsocketPolicyServer, WidowXAIFasterWAMPolicy
from fasterwam.serve.widowxai_policy import (
    ACTION_INFER_MODES,
    DEFAULT_RUN_DIR,
    resolve_run_artifacts,
)

DEFAULT_MODEL_ROOT = Path("/lustre/data/fastwam_nakamura/checkpoints")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--norm-stats", type=Path, default=None)
    parser.add_argument("--task", default=None, help="Defaults to the run's single training task.")
    parser.add_argument("--text-embedding-cache-dir", type=Path, default=None)
    parser.add_argument("--model-base-path", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--action-infer-mode", choices=ACTION_INFER_MODES, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--control-fps", type=float, default=None)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow missing Wan base weights to be downloaded instead of failing locally.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(args.model_base_path.expanduser().resolve())
    if not args.allow_download:
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"

    artifacts = resolve_run_artifacts(
        run_dir=args.run_dir,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        norm_stats_path=args.norm_stats,
    )
    hostname = socket.gethostname()
    print(f"[serve-widowxai] node={hostname}", flush=True)
    print(f"[serve-widowxai] run_dir={artifacts.run_dir}", flush=True)
    print(f"[serve-widowxai] checkpoint={artifacts.checkpoint_path}", flush=True)
    print(f"[serve-widowxai] norm_stats={artifacts.norm_stats_path}", flush=True)
    print(
        f"[serve-widowxai] ssh tunnel: ssh -N -L {args.port}:{hostname}:{args.port} "
        "<user>@<login-node>",
        flush=True,
    )

    policy = WidowXAIFasterWAMPolicy(
        run_dir=artifacts.run_dir,
        config_path=artifacts.config_path,
        checkpoint_path=artifacts.checkpoint_path,
        norm_stats_path=artifacts.norm_stats_path,
        task=args.task,
        text_embedding_cache_dir=args.text_embedding_cache_dir,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        action_infer_mode=args.action_infer_mode,
        seed=args.seed,
        control_fps=args.control_fps,
    )
    print(
        "[serve-widowxai] visual_input="
        + ("2wrist_vertical" if policy.drop_top_view else "3camera_robotwin")
        + f" drop_top_view={str(policy.drop_top_view).lower()}",
        flush=True,
    )
    WebsocketPolicyServer(policy, host=args.host, port=args.port).serve_forever()


if __name__ == "__main__":
    main()
