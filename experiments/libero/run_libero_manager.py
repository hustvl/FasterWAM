from __future__ import annotations

import json
import math
import os
import random
import contextlib
import io
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import hydra
from hydra.core.hydra_config import HydraConfig
from libero.libero import benchmark
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNK_ENTRY = PROJECT_ROOT / "experiments" / "libero" / "eval_libero_chunk.py"
SUMMARY_ENTRY = PROJECT_ROOT / "experiments" / "libero" / "summarize_results.py"
POLL_INTERVAL_SECONDS = 2.0
TERMINATE_TIMEOUT_SECONDS = 10.0


def _resolve_path(path_str: str, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _resolve_sample_ratio(raw_ratio: Any) -> float | None:
    if raw_ratio is None:
        return None
    if isinstance(raw_ratio, str):
        text = raw_ratio.strip()
        if text == "" or text.lower() in {"none", "null"}:
            return None
        ratio = float(text)
    else:
        ratio = float(raw_ratio)
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError(f"MULTIRUN.task_sample_ratio must be in (0, 1], got {ratio}")
    return None if ratio >= 1.0 else ratio


def _validate_benchmark_suite(benchmark_name: str, suite_name: str, n_tasks: int) -> None:
    normalized = benchmark_name.strip().lower().replace("_", "-")
    if normalized == "libero-plus" and n_tasks <= 100:
        raise RuntimeError(
            f"Expected LIBERO-Plus for suite {suite_name!r}, but only found {n_tasks} tasks. "
            "Run this command with the LIBERO-Plus evaluation environment."
        )
    if normalized == "libero" and n_tasks > 100:
        raise RuntimeError(
            f"Expected standard LIBERO for suite {suite_name!r}, but found {n_tasks} tasks. "
            "Run this command with the standard LIBERO evaluation environment."
        )


def _instantiate_suite(benchmark_dict: dict, suite_name: str):
    # LIBERO-Plus prints every task ID while constructing a suite. Suppress
    # that third-party diagnostic so release logs remain usable.
    with contextlib.redirect_stdout(io.StringIO()):
        return benchmark_dict[suite_name]()


def create_task_file(
    output_file: Path,
    task_suite_names: list[str],
    *,
    benchmark_name: str,
    sample_ratio: float | None,
    sample_seed: int,
) -> tuple[Path, dict[str, Any]]:
    benchmark_dict = benchmark.get_benchmark_dict()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[str, int]] = []
    suite_counts: dict[str, int] = {}
    suite_totals: dict[str, int] = {}
    for suite_name in task_suite_names:
        if suite_name not in benchmark_dict:
            raise KeyError(
                f"Unknown benchmark suite {suite_name!r}; available={sorted(benchmark_dict)}"
            )
        task_suite = _instantiate_suite(benchmark_dict, suite_name)
        n_tasks = int(task_suite.n_tasks)
        _validate_benchmark_suite(benchmark_name, suite_name, n_tasks)
        task_ids = list(range(n_tasks))
        if sample_ratio is not None:
            n_sample = max(1, int(math.ceil(n_tasks * sample_ratio)))
            rng = random.Random(f"{sample_seed}:{suite_name}")
            task_ids = sorted(rng.sample(task_ids, n_sample))
        tasks.extend((suite_name, task_id) for task_id in task_ids)
        suite_counts[suite_name] = len(task_ids)
        suite_totals[suite_name] = n_tasks
        print(
            f"{suite_name}: selected={len(task_ids)} total={n_tasks}"
            + (
                f" ratio={sample_ratio:g} seed={sample_seed}"
                if sample_ratio is not None
                else ""
            )
        )

    with output_file.open("w", encoding="utf-8") as f:
        for suite_name, task_id in tasks:
            f.write(f"{suite_name},{task_id}\n")

    metadata = {
        "benchmark": benchmark_name,
        "output": str(output_file),
        "total_tasks": len(tasks),
        "suite_counts": suite_counts,
        "suite_totals": suite_totals,
        "sample_ratio": sample_ratio,
        "sample_seed": sample_seed,
    }
    metadata_path = output_file.with_suffix(output_file.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return output_file, metadata


def _read_task_file(path: Path) -> list[tuple[str, int]]:
    tasks: list[tuple[str, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if line == "" or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 2 or parts[0] == "" or parts[1] == "":
                raise ValueError(
                    f"Invalid task line {line_no} in {path}: {raw_line.rstrip()!r}. "
                    "Expected suite_name,task_id"
                )
            tasks.append((parts[0], int(parts[1])))
    if not tasks:
        raise ValueError(f"Task file is empty: {path}")
    return tasks


def _write_chunks(
    tasks: list[tuple[str, int]],
    *,
    output_dir: Path,
    chunk_size: int,
) -> list[tuple[str, Path, str, int]]:
    if chunk_size <= 0:
        raise ValueError(f"MULTIRUN.chunk_size must be positive, got {chunk_size}")
    chunk_dir = output_dir / "task_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    records: list[tuple[str, Path, str, int]] = []
    for start in range(0, len(tasks), chunk_size):
        chunk_tasks = tasks[start : start + chunk_size]
        chunk_id = f"chunk_{len(records):06d}"
        chunk_file = chunk_dir / f"{chunk_id}.txt"
        with chunk_file.open("w", encoding="utf-8") as f:
            for suite_name, task_id in chunk_tasks:
                f.write(f"{suite_name},{task_id}\n")
        first_suite, first_task_id = chunk_tasks[0]
        records.append((chunk_id, chunk_file, first_suite, first_task_id))
    (output_dir / "task_chunks.txt").write_text(
        "".join(
            f"{chunk_id}|{chunk_file}|{suite}|{task_id}\n"
            for chunk_id, chunk_file, suite, task_id in records
        ),
        encoding="utf-8",
    )
    return records


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    blocked_exact = {
        "task",
        "ckpt",
        "gpu_id",
        "EVALUATION.task_suite_name",
        "EVALUATION.task_id",
        "EVALUATION.output_dir",
    }
    return key in blocked_exact or key.startswith("MULTIRUN.") or key.startswith("hydra.")


def collect_worker_overrides() -> list[str]:
    return [
        override
        for override in HydraConfig.get().overrides.task
        if not _is_blocked_override(override)
    ]


def _resolve_worker_task_choice() -> str:
    task_choice = HydraConfig.get().runtime.choices.get("task")
    if task_choice is None or str(task_choice).strip() == "":
        raise ValueError("Hydra task choice is empty. Please pass task=...")
    return str(task_choice)


def _parse_gpu_ids(manager: DictConfig) -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        gpu_ids = [item.strip() for item in visible.split(",") if item.strip()]
    else:
        configured = manager.get("gpu_ids")
        if configured is not None and str(configured).lower() not in {"none", "null"}:
            gpu_ids = [str(int(item)) for item in configured]
        else:
            num_gpus = int(manager.num_gpus)
            if num_gpus <= 0:
                raise ValueError(f"MULTIRUN.num_gpus must be positive, got {num_gpus}")
            gpu_ids = [str(index) for index in range(num_gpus)]
    if not gpu_ids:
        raise ValueError("No GPU IDs are available for LIBERO evaluation.")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"Duplicate GPU IDs: {gpu_ids}")
    return gpu_ids


@dataclass
class RunningChunk:
    chunk_id: str
    gpu_id: str
    process: subprocess.Popen[str]
    log_path: Path
    log_handle: TextIO


def _terminate_running(running: list[RunningChunk]) -> None:
    for state in running:
        if state.process.poll() is None:
            state.process.terminate()
    deadline = time.time() + TERMINATE_TIMEOUT_SECONDS
    for state in running:
        if state.process.poll() is not None:
            continue
        try:
            state.process.wait(timeout=max(0.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            state.process.kill()
            state.process.wait()
    for state in running:
        state.log_handle.close()


def run_evaluation(
    *,
    task_file: Path,
    config_name: str,
    task_choice: str,
    ckpt: Path,
    num_trials: int,
    max_chunks_per_gpu: int,
    chunk_size: int,
    output_dir: Path,
    extra_overrides: list[str],
    gpu_ids: list[str],
    dry_run: bool,
) -> None:
    if not CHUNK_ENTRY.exists():
        raise FileNotFoundError(f"Chunk worker not found: {CHUNK_ENTRY}")
    if max_chunks_per_gpu <= 0:
        raise ValueError(
            f"MULTIRUN.max_tasks_per_gpu must be positive, got {max_chunks_per_gpu}"
        )

    tasks = _read_task_file(task_file)
    chunks = _write_chunks(tasks, output_dir=output_dir, chunk_size=chunk_size)
    log_dir = output_dir / "task_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    failed_file = output_dir / "failed_tasks.txt"
    failed_file.write_text("", encoding="utf-8")

    def build_command(record: tuple[str, Path, str, int], gpu_id: str) -> list[str]:
        _, chunk_file, first_suite, first_task_id = record
        return [
            sys.executable,
            str(CHUNK_ENTRY),
            "--config-dir",
            str(PROJECT_ROOT / "configs"),
            "--config-name",
            config_name,
            "--task",
            task_choice,
            "--ckpt",
            str(ckpt),
            "--task-chunk-file",
            str(chunk_file),
            "--first-suite",
            first_suite,
            "--first-task-id",
            str(first_task_id),
            "--gpu-id",
            gpu_id,
            "--num-trials",
            str(num_trials),
            "--output-dir",
            str(output_dir),
            "--",
            *extra_overrides,
        ]

    print(
        f"Starting evaluation: tasks={len(tasks)} chunks={len(chunks)} "
        f"chunk_size={chunk_size} gpu_ids={gpu_ids}"
    )
    if dry_run:
        first_command = build_command(chunks[0], gpu_ids[0])
        print("DRY_RUN=true; first worker command:")
        print(subprocess.list2cmdline(first_command))
        return

    pending = deque(chunks)
    running: list[RunningChunk] = []
    try:
        while pending or running:
            failures: list[RunningChunk] = []
            still_running: list[RunningChunk] = []
            for state in running:
                return_code = state.process.poll()
                if return_code is None:
                    still_running.append(state)
                    continue
                state.log_handle.close()
                if return_code != 0:
                    failures.append(state)
                else:
                    print(f"Completed {state.chunk_id} on GPU {state.gpu_id}")
            running = still_running

            if failures:
                with failed_file.open("a", encoding="utf-8") as f:
                    for state in failures:
                        f.write(
                            f"{state.chunk_id},gpu={state.gpu_id},"
                            f"returncode={state.process.returncode},log={state.log_path}\n"
                        )
                _terminate_running(running)
                details = failed_file.read_text(encoding="utf-8")
                raise RuntimeError(f"LIBERO chunk evaluation failed:\n{details}")

            for gpu_id in gpu_ids:
                gpu_load = sum(state.gpu_id == gpu_id for state in running)
                while pending and gpu_load < max_chunks_per_gpu:
                    record = pending.popleft()
                    chunk_id = record[0]
                    log_path = log_dir / f"{chunk_id}_gpu{gpu_id}.log"
                    log_handle = log_path.open("w", encoding="utf-8")
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = gpu_id
                    command = build_command(record, gpu_id)
                    process = subprocess.Popen(
                        command,
                        cwd=str(PROJECT_ROOT),
                        env=env,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    running.append(
                        RunningChunk(
                            chunk_id=chunk_id,
                            gpu_id=gpu_id,
                            process=process,
                            log_path=log_path,
                            log_handle=log_handle,
                        )
                    )
                    gpu_load += 1
                    print(
                        f"Launched {chunk_id} on GPU {gpu_id} "
                        f"({gpu_load}/{max_chunks_per_gpu})"
                    )

            if pending or running:
                time.sleep(POLL_INTERVAL_SECONDS)
    except BaseException:
        _terminate_running(running)
        raise

    if SUMMARY_ENTRY.exists():
        subprocess.run(
            [sys.executable, str(SUMMARY_ENTRY), f"--output_dir={output_dir}"],
            cwd=str(PROJECT_ROOT),
            check=True,
        )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def main(cfg: DictConfig) -> None:
    if cfg.ckpt is None:
        raise ValueError("ckpt must not be None.")
    if cfg.EVALUATION.output_dir is None:
        raise ValueError("EVALUATION.output_dir must not be None.")

    task_choice = _resolve_worker_task_choice()
    manager = cfg.MULTIRUN
    benchmark_name = str(cfg.get("benchmark_name", "libero"))
    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = _resolve_path(str(cfg.ckpt))
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    task_file_cfg = manager.get("task_file")
    if task_file_cfg:
        task_file = _resolve_path(str(task_file_cfg))
        if not task_file.exists():
            raise FileNotFoundError(f"Task file not found: {task_file}")
    else:
        task_file, _ = create_task_file(
            output_dir / "tasks.txt",
            list(manager.task_suite_names),
            benchmark_name=benchmark_name,
            sample_ratio=_resolve_sample_ratio(manager.get("task_sample_ratio")),
            sample_seed=int(manager.get("task_sample_seed", 42)),
        )

    OmegaConf.save(config=cfg, f=str(output_dir / "manager_config.yaml"))
    if bool(manager.get("create_only", False)):
        print("create_only=true; generated task metadata without launching workers.")
        return

    config_name = str(HydraConfig.get().job.config_name)
    run_evaluation(
        task_file=task_file,
        config_name=config_name,
        task_choice=task_choice,
        ckpt=ckpt,
        num_trials=int(cfg.EVALUATION.num_trials),
        max_chunks_per_gpu=int(manager.max_tasks_per_gpu),
        chunk_size=int(manager.get("chunk_size", 1)),
        output_dir=output_dir,
        extra_overrides=collect_worker_overrides(),
        gpu_ids=_parse_gpu_ids(manager),
        dry_run=bool(manager.get("dry_run", False)),
    )


if __name__ == "__main__":
    main()
