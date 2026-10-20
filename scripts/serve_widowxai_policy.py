#!/usr/bin/env python3
"""Serve a trained WidowXAI FasterWAM policy over WebSocket."""

from __future__ import annotations

import argparse
import os
import socket
from pathlib import Path

import yaml

from fasterwam.models.wan22.rtc import RTCConfig
from fasterwam.models.wan22.simple import SimpleConfig
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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--control-fps", type=float, default=None)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow missing Wan base weights to be downloaded instead of failing locally.",
    )
    parser.add_argument("--serving-config", type=Path, default=None,
                        help="Serving overrides, independent of the resolved training --config.")
    parser.add_argument("--rtc", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rtc-prefix-schedule", choices=["exp", "linear", "ones", "zeros"], default=None)
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=None)
    parser.add_argument("--simple", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--simple-prefix-schedule", choices=["exp", "linear", "ones", "zeros"], default=None)
    args = parser.parse_args()
    settings = {}
    if args.serving_config is not None:
        with args.serving_config.open(encoding="utf-8") as stream:
            settings = yaml.safe_load(stream) or {}
        if not isinstance(settings, dict) or set(settings) - {
            "rtc", "simple", "action_infer_mode", "num_inference_steps", "seed", "control_fps"
        }:
            parser.error("Unknown serving configuration fields")
    rtc = settings.get("rtc", {})
    if not isinstance(rtc, dict) or set(rtc) - {"enabled", "prefix_attention_schedule", "max_guidance_weight"}:
        parser.error("Unknown RTC configuration fields")
    args.rtc_config = RTCConfig(
        enabled=args.rtc if args.rtc is not None else rtc.get("enabled", False),
        prefix_attention_schedule=args.rtc_prefix_schedule or rtc.get("prefix_attention_schedule", "exp"),
        max_guidance_weight=(args.rtc_max_guidance_weight if args.rtc_max_guidance_weight is not None
                             else rtc.get("max_guidance_weight", 5.0)),
    )
    simple = settings.get("simple", {})
    if not isinstance(simple, dict) or set(simple) - {"enabled", "prefix_attention_schedule"}:
        parser.error("Unknown simple configuration fields")
    args.simple_config = SimpleConfig(
        enabled=args.simple if args.simple is not None else simple.get("enabled", False),
        prefix_attention_schedule=args.simple_prefix_schedule or simple.get("prefix_attention_schedule", "exp"),
    )
    for key in ("action_infer_mode", "num_inference_steps", "seed", "control_fps"):
        if getattr(args, key) is None:
            setattr(args, key, settings.get(key, 42 if key == "seed" else None))
    return args


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
    print(
        f"[serve-widowxai] rtc={args.rtc_config} simple={args.simple_config} seed={args.seed}",
        flush=True,
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
        rtc_config=args.rtc_config,
        simple_config=args.simple_config,
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
