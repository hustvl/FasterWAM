#!/usr/bin/env python
"""Evaluate a chunk of LIBERO tasks with one FasterWAM model load."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from accelerate import PartialState
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT, PROJECT_ROOT / "experiments" / "libero"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def _parse_task_chunk(path: Path) -> list[tuple[str, int]]:
    choices: list[tuple[str, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if line == "" or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 2 or parts[0] == "" or parts[1] == "":
                raise ValueError(
                    f"Invalid task chunk line {line_no} in {path}: {raw_line.rstrip()!r}. "
                    "Expected format: suite_name,task_id"
                )
            choices.append((parts[0], int(parts[1])))
    if not choices:
        raise ValueError(f"Task chunk file is empty: {path}")
    return choices


def _compose_cfg(args: argparse.Namespace, overrides: list[str]) -> DictConfig:
    base_overrides = [
        f"task={args.task}",
        f"ckpt={args.ckpt}",
        f"gpu_id={args.gpu_id}",
        f"EVALUATION.task_suite_name={args.first_suite}",
        f"EVALUATION.task_id={args.first_task_id}",
        f"EVALUATION.num_trials={args.num_trials}",
        f"EVALUATION.output_dir={args.output_dir}",
    ]
    with initialize_config_dir(
        config_dir=str(Path(args.config_dir).resolve()),
        version_base="1.3",
    ):
        return compose(config_name=args.config_name, overrides=base_overrides + overrides)


def _prepare_initial_states(task_suite, task_id: int, num_trials: int):
    initial_states = task_suite.get_task_init_states(task_id)
    while len(initial_states) < num_trials:
        initial_states.extend(initial_states[: (num_trials - len(initial_states))])
    return initial_states


def _write_eval_config(cfg: DictConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")


def run_chunk(cfg: DictConfig, task_choices: list[tuple[str, int]]) -> list[dict]:
    from experiments.libero import eval_libero_single as single
    from fasterwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
    from fasterwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from fasterwam.utils.pytorch_utils import set_global_seed
    from libero.libero import benchmark

    chunk_start_time = time.time()
    partial_state = PartialState()
    partial_state.config = cfg

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)
    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")

    single._validate_visualize_future_video_cfg(cfg)
    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError("Chunk evaluation supports only EVALUATION.env_num=1.")

    model_device = single._resolve_eval_device(cfg)
    model_dtype = single._mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    single._load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset_stats_path = single._resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    action_horizon = (
        int(cfg.data.train.num_frames) - 1
        if action_horizon_cfg is None
        else int(action_horizon_cfg)
    )
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h, input_w = int(video_size[0]), int(video_size[1])

    output_root = Path(cfg.EVALUATION.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    benchmark_name = str(cfg.get("benchmark_name", "libero"))
    benchmark_dict = benchmark.get_benchmark_dict()
    all_results: list[dict] = []

    for chunk_idx, (suite_name, task_id) in enumerate(task_choices):
        if suite_name not in benchmark_dict:
            raise KeyError(f"Unknown benchmark suite {suite_name!r}; available={sorted(benchmark_dict)}")

        task_start_time = time.time()
        cfg.EVALUATION.task_suite_name = suite_name
        cfg.EVALUATION.task_id = int(task_id)
        video_dir = output_root / suite_name / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        predicted_video_dir = output_root / suite_name / "predicted_videos"
        if bool(cfg.EVALUATION.get("visualize_future_video", False)):
            predicted_video_dir.mkdir(parents=True, exist_ok=True)

        _write_eval_config(
            cfg,
            output_root / suite_name / f"eval_config_gpu{cfg.gpu_id}_task{task_id}.yaml",
        )

        task_suite = benchmark_dict[suite_name]()
        task = task_suite.get_task(task_id)
        initial_states = _prepare_initial_states(
            task_suite,
            task_id=task_id,
            num_trials=int(cfg.EVALUATION.num_trials),
        )
        results = {
            "benchmark": benchmark_name,
            "task_suite": suite_name,
            "task_id": int(task_id),
            "task_description": None,
            "successes": 0,
            "total_episodes": int(cfg.EVALUATION.num_trials),
            "gpu_id": int(cfg.gpu_id),
            "success_episodes": [],
            "failure_episodes": [],
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": 0,
        }

        logging.info(
            "Running %s evaluation suite=%s task_id=%s chunk_item=%s/%s",
            benchmark_name,
            suite_name,
            task_id,
            chunk_idx + 1,
            len(task_choices),
        )
        task_results = single.run_single_task(
            task=task,
            initial_states=initial_states,
            model=model,
            processor=processor,
            cfg=cfg,
            video_dir=video_dir,
            predicted_video_dir=predicted_video_dir,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
        )
        results.update(task_results)
        results["duration"] = time.time() - task_start_time

        output_dir = output_root / suite_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"gpu{cfg.gpu_id}_task{task_id}_results.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, cls=single.NumpyEncoder)

        print(
            f"Task {suite_name}/{task_id} completed: "
            f"{results['successes']}/{cfg.EVALUATION.num_trials} successes"
        )
        all_results.append(results)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logging.info(
        "Chunk completed: tasks=%s duration=%.2fs",
        len(task_choices),
        time.time() - chunk_start_time,
    )
    return all_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default=str(PROJECT_ROOT / "configs"))
    parser.add_argument("--config-name", default="sim_libero")
    parser.add_argument("--task", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--task-chunk-file", required=True)
    parser.add_argument("--first-suite", required=True)
    parser.add_argument("--first-task-id", type=int, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--num-trials", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    overrides = list(args.overrides)
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]

    task_chunk_file = Path(args.task_chunk_file).resolve()
    task_choices = _parse_task_chunk(task_chunk_file)
    cfg = _compose_cfg(args, overrides)

    print(f"TASK_CHUNK_FILE={task_chunk_file}")
    print(f"TASKS_IN_CHUNK={len(task_choices)}")
    print(f"GPU_ID={args.gpu_id}")
    print(f"OUTPUT_DIR={args.output_dir}")
    run_chunk(cfg, task_choices)


if __name__ == "__main__":
    main()
