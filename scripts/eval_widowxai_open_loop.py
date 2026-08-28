#!/usr/bin/env python3
"""Distributed WidowXAI offline open-loop action evaluation.

The default inference path is FasterWAM's one-pass future-cache path. Metrics
match the action portions of the existing FastWAM evaluation: MAE in model
z-score space, absolute physical space, and DreamZero q01/q99 space.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fastwam.utils import misc as fastwam_misc
from fasterwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fasterwam.utils import misc
from fasterwam.utils.logging_config import setup_logging


DEFAULT_Q01_Q99_ACTION_KEYS = (
    "left_joint_pos",
    "left_gripper_pos",
    "right_joint_pos",
    "right_gripper_pos",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Resolved FasterWAM config YAML.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--q01-q99-stats", default=None)
    parser.add_argument(
        "--q01-q99-action-keys",
        default=",".join(DEFAULT_Q01_Q99_ACTION_KEYS),
    )
    parser.add_argument("--comparison-action-horizon", type=int, default=24)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument(
        "--action-infer-mode",
        choices=("one_pass_future_cache", "joint_rollout"),
        default="one_pass_future_cache",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-name", default=None)
    return parser.parse_args()


def init_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Full WidowXAI open-loop evaluation requires CUDA.")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=device)
    return rank, local_rank, world_size, device


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def all_reduce_sum(value: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def denormalize_action_and_state(
    action: torch.Tensor,
    proprio: torch.Tensor,
    processor: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if action.ndim != 2 or proprio.ndim != 2:
        raise ValueError(f"Expected action/proprio [T,D], got {action.shape}/{proprio.shape}.")
    batch = {
        "action": action.detach().to(device="cpu", dtype=torch.float32).unsqueeze(0),
        "state": proprio.detach().to(device="cpu", dtype=torch.float32).unsqueeze(0),
    }
    batch = processor.action_state_merger.backward(batch)
    batch = processor.normalizer.backward(batch)
    if processor.action_state_transforms is not None:
        for transform in reversed(processor.action_state_transforms):
            batch = transform.backward(batch)
    action_parts = [batch["action"][meta["key"]] for meta in processor.shape_meta["action"]]
    state_parts = [batch["state"][meta["key"]] for meta in processor.shape_meta["state"]]
    return torch.cat(action_parts, dim=-1).squeeze(0), torch.cat(state_parts, dim=-1).squeeze(0)


def resolve_q01_q99_stats_path(path: str) -> Path:
    stats_path = Path(path).resolve()
    if stats_path.is_dir():
        stats_path = stats_path / "meta" / "relative_stats_dreamzero.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"q01/q99 statistics file not found: {stats_path}")
    return stats_path


def load_q01_q99_stats(
    path: str,
    action_keys: list[str],
    expected_dim: int,
) -> tuple[Path, torch.Tensor, torch.Tensor]:
    stats_path = resolve_q01_q99_stats_path(path)
    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    if "q01" in stats and "q99" in stats:
        q01_parts = [torch.as_tensor(stats["q01"], dtype=torch.float32).flatten()]
        q99_parts = [torch.as_tensor(stats["q99"], dtype=torch.float32).flatten()]
    else:
        missing = [key for key in action_keys if key not in stats]
        if missing:
            raise KeyError(f"q01/q99 statistics missing action keys {missing}: {stats_path}")
        q01_parts = [torch.as_tensor(stats[key]["q01"], dtype=torch.float32).flatten() for key in action_keys]
        q99_parts = [torch.as_tensor(stats[key]["q99"], dtype=torch.float32).flatten() for key in action_keys]
    q01 = torch.cat(q01_parts)
    q99 = torch.cat(q99_parts)
    if q01.numel() != expected_dim or q99.numel() != expected_dim:
        raise ValueError(
            f"q01/q99 statistics have {q01.numel()}/{q99.numel()} dimensions, expected {expected_dim}."
        )
    if bool((q99 <= q01).any()):
        raise ValueError("q99 must be greater than q01 in every action dimension.")
    return stats_path, q01, q99


def normalize_q01_q99(action: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    q01 = q01.to(device=action.device, dtype=action.dtype)
    q99 = q99.to(device=action.device, dtype=action.dtype)
    return (2.0 * (action - q01) / (q99 - q01) - 1.0).clamp(-1.0, 1.0)


def episode_and_frame(dataset: Any, index: int) -> tuple[int, int]:
    episode_index = dataset.lerobot_dataset.episode_data_index
    starts = episode_index["from"].tolist()
    ends = episode_index["to"].tolist()
    for episode, (start, end) in enumerate(zip(starts, ends)):
        if int(start) <= index < int(end):
            return episode, index - int(start)
    raise IndexError(f"Dataset index {index} does not map to an episode.")


def infer_open_loop(
    model: Any,
    sample: dict[str, Any],
    mode: str,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    video = sample["video"]
    kwargs = {
        "prompt": None,
        "input_image": video[:, 0].unsqueeze(0),
        "action_horizon": int(sample["action"].shape[0]),
        "proprio": sample["proprio"][0],
        "context": sample["context"],
        "context_mask": sample["context_mask"],
        "negative_prompt": None,
        "text_cfg_scale": 1.0,
        "num_inference_steps": steps,
        "sigma_shift": None,
        "seed": seed,
        "rand_device": "cpu",
        "tiled": False,
    }
    if mode == "one_pass_future_cache":
        infer_fn = getattr(model, "infer_action_one_pass_future_cache", None)
        if infer_fn is None:
            raise ValueError("Model does not provide infer_action_one_pass_future_cache().")
    else:
        infer_fn = model.infer_action
    if "num_video_frames" in inspect.signature(infer_fn).parameters:
        kwargs["num_video_frames"] = int(video.shape[1])
    return infer_fn(**kwargs)


def checkpoint_step(path: str) -> int | None:
    match = re.search(r"step_(\d+)", Path(path).stem)
    return int(match.group(1)) if match else None


def write_wandb(summary: dict[str, Any], project: str | None, name: str | None) -> None:
    if not project:
        return
    import wandb

    run = wandb.init(
        project=project,
        name=name,
        job_type="open_loop_eval",
        config=summary["evaluation"],
        dir=summary["evaluation"]["output_dir"],
    )
    step = summary["evaluation"].get("checkpoint_step")
    payload = {
        "open_loop/normalized_action_mae": summary["normalized_action_mae"],
        "open_loop/action_mae": summary["action_mae"],
        "open_loop/num_samples": summary["num_samples"],
    }
    if summary.get("q01_q99_action_mae") is not None:
        payload["open_loop/q01_q99_action_mae"] = summary["q01_q99_action_mae"]
        payload["open_loop/q01_q99_num_samples"] = summary["q01_q99_num_samples"]
    wandb.log(payload, step=step)
    for key, value in payload.items():
        run.summary[key] = value
    run.finish()


def main() -> None:
    args = parse_args()
    if args.num_inference_steps <= 0:
        raise ValueError("--num-inference-steps must be positive.")
    if args.comparison_action_horizon <= 0:
        raise ValueError("--comparison-action-horizon must be positive.")
    rank, _local_rank, world_size, device = init_distributed()
    setup_logging(log_level=logging.INFO, is_main_process=(rank == 0))

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier(world_size)
    misc.register_work_dir(str(output_dir))
    fastwam_misc.register_work_dir(str(output_dir))

    cfg = OmegaConf.load(args.config)
    mixed_precision = _normalize_mixed_precision(str(cfg.get("mixed_precision", "bf16")))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    action_dim = int(cfg.data.val.processor.action_output_dim)
    action_keys = [key.strip() for key in args.q01_q99_action_keys.split(",") if key.strip()]
    q_stats_path: Path | None = None
    q01: torch.Tensor | None = None
    q99: torch.Tensor | None = None
    if args.q01_q99_stats:
        q_stats_path, q01, q99 = load_q01_q99_stats(args.q01_q99_stats, action_keys, action_dim)

    model = instantiate(cfg.model, model_dtype=model_dtype, device=str(device))
    model.load_checkpoint(args.checkpoint)
    model.eval()
    dataset = instantiate(cfg.data.val, pretrained_norm_stats=args.norm_stats)
    processor = dataset.lerobot_dataset.processor

    total_samples = len(dataset)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive.")
        total_samples = min(total_samples, args.max_samples)
    local_indices = list(range(rank, total_samples, world_size))

    # normalized sum/count, physical sum/count, q01/q99 sum/count,
    # processed samples, q01/q99-eligible samples
    sums = torch.zeros(8, device=device, dtype=torch.float64)
    samples_path = output_dir / f"samples_rank_{rank:03d}.jsonl"
    start_time = time.time()
    with samples_path.open("w", encoding="utf-8", buffering=1) as samples_file, torch.inference_mode():
        for local_position, index in enumerate(local_indices, start=1):
            sample = dataset._get(index)
            prediction = infer_open_loop(
                model=model,
                sample=sample,
                mode=args.action_infer_mode,
                steps=args.num_inference_steps,
                seed=args.seed,
            )
            pred_action = prediction["action"].float()
            gt_action = sample["action"].detach().to(device="cpu", dtype=torch.float32)
            if pred_action.shape != gt_action.shape:
                raise ValueError(f"Action shape mismatch at {index}: {pred_action.shape} vs {gt_action.shape}")

            valid_steps = ~sample["action_is_pad"].detach().to(device="cpu", dtype=torch.bool)
            dim_is_pad = sample.get("action_dim_is_pad")
            valid_dims = (
                torch.ones(pred_action.shape[-1], dtype=torch.bool)
                if dim_is_pad is None
                else ~dim_is_pad.detach().to(device="cpu", dtype=torch.bool)
            )
            action_mask = valid_steps[:, None] & valid_dims[None, :]
            if not bool(action_mask.any()):
                raise ValueError(f"No valid action values at dataset index {index}.")

            norm_abs = (pred_action - gt_action).abs()
            pred_real, proprio_real = denormalize_action_and_state(pred_action, sample["proprio"], processor)
            gt_real, _ = denormalize_action_and_state(gt_action, sample["proprio"], processor)
            real_abs = (pred_real - gt_real).abs()
            count = torch.tensor(float(action_mask.sum().item()), device=device, dtype=torch.float64)

            q_abs_sum = torch.tensor(0.0, device=device, dtype=torch.float64)
            q_count = torch.tensor(0.0, device=device, dtype=torch.float64)
            q_sample_count = torch.tensor(0.0, device=device, dtype=torch.float64)
            sample_q_mae = None
            if q01 is not None and q99 is not None:
                horizon = args.comparison_action_horizon
                if pred_real.shape[0] < horizon:
                    raise ValueError(f"Comparison horizon {horizon} exceeds action horizon {pred_real.shape[0]}.")
                comparison_mask = action_mask[:horizon]
                if bool(valid_steps[:horizon].all()):
                    current_state = proprio_real[0]
                    pred_q = normalize_q01_q99(pred_real[:horizon] - current_state, q01, q99)
                    gt_q = normalize_q01_q99(gt_real[:horizon] - current_state, q01, q99)
                    q_abs = (pred_q - gt_q).abs()
                    q_abs_sum = q_abs[comparison_mask].double().sum().to(device)
                    q_count = torch.tensor(float(comparison_mask.sum().item()), device=device, dtype=torch.float64)
                    q_sample_count.fill_(1.0)
                    sample_q_mae = float(q_abs[comparison_mask].mean().item())

            sums += torch.stack(
                [
                    norm_abs[action_mask].double().sum().to(device),
                    count,
                    real_abs[action_mask].double().sum().to(device),
                    count,
                    q_abs_sum,
                    q_count,
                    torch.tensor(1.0, device=device, dtype=torch.float64),
                    q_sample_count,
                ]
            )
            episode, frame = episode_and_frame(dataset, index)
            row = {
                "index": index,
                "episode_index": episode,
                "frame_index": frame,
                "valid_action_steps": int(valid_steps.sum().item()),
                "normalized_action_mae": float(norm_abs[action_mask].mean().item()),
                "action_mae": float(real_abs[action_mask].mean().item()),
                "q01_q99_action_mae": sample_q_mae,
            }
            samples_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            if local_position % args.log_every == 0 or local_position == len(local_indices):
                print(
                    f"[rank {rank}] {local_position}/{len(local_indices)} index={index} "
                    f"elapsed={time.time() - start_time:.1f}s",
                    flush=True,
                )

    sums = all_reduce_sum(sums, world_size)
    barrier(world_size)
    if rank == 0:
        if int(sums[6].item()) != total_samples:
            raise RuntimeError(f"Coverage mismatch: reduced {int(sums[6].item())}, expected {total_samples}.")
        if sums[1] <= 0 or sums[3] <= 0:
            raise RuntimeError(f"Action metric denominator is zero: {sums.tolist()}")
        if q01 is not None and (sums[5] <= 0 or sums[7] <= 0):
            raise RuntimeError(f"No q01/q99-eligible windows: {sums.tolist()}")

        merged_rows = []
        for rank_id in range(world_size):
            rank_path = output_dir / f"samples_rank_{rank_id:03d}.jsonl"
            with rank_path.open("r", encoding="utf-8") as f:
                merged_rows.extend(json.loads(line) for line in f if line.strip())
        merged_rows.sort(key=lambda row: row["index"])
        with (output_dir / "per_sample.jsonl").open("w", encoding="utf-8") as f:
            for row in merged_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary = {
            "normalized_action_mae": float((sums[0] / sums[1]).item()),
            "action_mae": float((sums[2] / sums[3]).item()),
            "q01_q99_action_mae": float((sums[4] / sums[5]).item()) if q01 is not None else None,
            "num_samples": int(sums[6].item()),
            "num_valid_action_values": int(sums[1].item()),
            "q01_q99_num_samples": int(sums[7].item()),
            "q01_q99_num_valid_action_values": int(sums[5].item()),
            "duration_seconds": time.time() - start_time,
            "evaluation": {
                "output_dir": str(output_dir.resolve()),
                "config": str(Path(args.config).resolve()),
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "checkpoint_step": checkpoint_step(args.checkpoint),
                "dataset_dirs": OmegaConf.to_container(cfg.data.val.dataset_dirs, resolve=True),
                "dataset_size": len(dataset),
                "evaluated_samples": total_samples,
                "num_inference_steps": args.num_inference_steps,
                "action_infer_mode": args.action_infer_mode,
                "seed": args.seed,
                "world_size": world_size,
                "action_dim": action_dim,
                "model_action_normalization": "z-score",
                "action_padding_excluded": True,
                "aggregation": {
                    "normalized_action_mae": "mean over all valid action scalars in model z-score space",
                    "action_mae": "mean over all valid absolute physical action scalars",
                    "q01_q99_action_mae": "mean over complete leading comparison horizons",
                },
            },
        }
        if q_stats_path is not None:
            summary["evaluation"]["q01_q99_comparison"] = {
                "stats": str(q_stats_path),
                "action_keys": action_keys,
                "action_horizon": args.comparison_action_horizon,
                "complete_horizon_required": True,
                "physical_representation": "absolute action minus current physical state for all dimensions",
                "formula": "clip(2 * (x - q01) / (q99 - q01) - 1, -1, 1)",
                "stats_source": "training dataset, not evaluation dataset",
            }
        with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        write_wandb(summary, args.wandb_project, args.wandb_name)

    barrier(world_size)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
