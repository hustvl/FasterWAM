#!/usr/bin/env python3
"""Compare two real WebSocket servers on one Slurm GPU, with exact equality.

Uses existing artifacts only. Saves identical synthetic observations, CPU
preprocessing, server logs/metadata and physical action chunks for review.
Both servers are stopped on completion or failure.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--candidate-repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-base-path", type=Path,
                        default=Path("/lustre/data/fastwam_nakamura/checkpoints"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--candidate-serving-config", type=Path, default=None,
                        help="Serving config passed only to the candidate server, e.g. configs/serve/rtc.yaml")
    parser.add_argument("--exercise-rtc", action="store_true",
                        help="Also send empty/nonempty RTC prefixes through synchronous requests")
    parser.add_argument("--modes", nargs="+", default=["one_pass_future_cache", "joint_rollout"],
                        choices=["one_pass_future_cache", "joint_rollout"])
    parser.add_argument("--preprocess-worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    result = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if stat.st_size < 10_000_000:
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def observations(path):
    with np.load(path) as data:
        return [
            {"qpos": qpos, "images": dict(zip(
                ("cam_high", "cam_left_wrist", "cam_right_wrist"), images
            ))}
            for qpos, images in zip(data["qpos"], data["images"])
        ]


def preprocess_worker(args):
    import torch
    from omegaconf import OmegaConf
    from fasterwam.serve.widowxai_policy import (
        WidowXAIOnlinePreprocessor, WidowXAIFasterWAMPolicy,
    )

    cfg = OmegaConf.load(args.run_dir / "config.yaml")
    preprocessor = WidowXAIOnlinePreprocessor(cfg, args.run_dir / "dataset_stats.json")
    policy = WidowXAIFasterWAMPolicy.__new__(WidowXAIFasterWAMPolicy)
    policy.cfg, policy.task = cfg, args.task
    context, mask = policy._load_text_context(None)
    result = {"context": context.float().numpy(), "mask": mask.numpy()}
    for index, obs in enumerate(observations(args.output_dir.parent / "inputs.npz")):
        image, proprio = preprocessor.prepare_observation(obs)
        result[f"image_{index}"] = image.numpy()
        result[f"proprio_{index}"] = proprio.numpy()
        # Exercise the entire inverse action transform as well.
        result[f"physical_{index}"] = preprocessor.denormalize_action(
            torch.linspace(-1, 1, preprocessor.action_horizon * 14).reshape(-1, 14), proprio
        )
    np.savez(args.output_dir / "preprocessing.npz", **result)
    print(json.dumps({"module": sys.modules[WidowXAIFasterWAMPolicy.__module__].__file__}))


def repo_environment(repo):
    env = os.environ.copy()
    # Do not allow the other repository's src or an installed editable package
    # to override the repository being measured.
    env["PYTHONPATH"] = f"{repo / 'src'}:{repo}"
    env["PYTHONUNBUFFERED"] = "1"
    env["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    return env


def compare_arrays(reference, candidate):
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise AssertionError(f"shape/dtype differ: {reference.shape}/{reference.dtype}, "
                             f"{candidate.shape}/{candidate.dtype}")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise AssertionError("Non-finite output")
    delta = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    return {"exact_equal": bool(np.array_equal(reference, candidate)),
            "shape": list(reference.shape), "max_abs_diff": float(delta.max()),
            "mean_abs_diff": float(delta.mean())}


@contextlib.contextmanager
def server(args, repo, directory, mode, *, serving_config=None):
    from websockets.sync.client import connect
    from fasterwam.serve.serialization import unpackb

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [sys.executable, str(repo / "scripts/serve_widowxai_policy.py"),
               "--run-dir", str(args.run_dir), "--checkpoint", str(args.checkpoint),
               "--task", args.task, "--model-base-path", str(args.model_base_path),
               "--host", "127.0.0.1", "--port", str(port), "--device", "cuda:0",
               "--seed", str(args.seed), "--num-inference-steps", str(args.num_inference_steps),
               "--action-infer-mode", mode]
    if serving_config is not None:
        command.extend(["--serving-config", str(serving_config)])
    (directory / "command.json").write_text(json.dumps(command, indent=2))
    with (directory / "server.log").open("w") as log:
        process = subprocess.Popen(command, cwd=repo, env=repo_environment(repo),
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited ({process.returncode}): {directory / 'server.log'}")
                try:
                    client = connect(f"ws://127.0.0.1:{port}", open_timeout=2,
                                     max_size=None, ping_timeout=None)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"Server readiness: {directory / 'server.log'}")
                    time.sleep(1)
            with client:
                metadata = unpackb(client.recv(timeout=30))
                (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
                yield client, metadata
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def exercise_rtc(args, client, metadata, directory, mode, records):
    """Wait for each RTC response; this does not use an asynchronous executor."""
    from fasterwam.serve.serialization import packb, unpackb

    if not metadata.get("supports_rtc"):
        raise AssertionError("Candidate server must advertise supports_rtc=true")
    obs = observations(args.output_dir / "inputs.npz")[2]
    baseline = np.load(directory / "action_2.npy")
    # A previous chunk's unexecuted tail, rebased by the production server
    # against a different current qpos. Delay is zero for these sync requests.
    previous = np.load(directory / "action_1.npy")[16:].copy()
    np.save(directory / "rtc_previous_actions.npy", previous)
    guided = None
    for request_id, (name, prefix) in enumerate((
        ("empty_prefix", np.empty((0, 14), dtype=np.float32)),
        ("nonempty_prefix", previous),
        ("nonempty_prefix_repeat", previous),
    )):
        context = {"episode_id": f"sync-rtc-{mode}", "request_id": request_id,
                   "observation_step": 16, "control_dt": metadata["control_dt"],
                   "previous_actions": prefix, "inference_delay_steps": 0,
                   "guidance_method": "rtc"}
        client.send(packb({"type": "select_action", "observation": obs, "chunk_context": context}))
        response = unpackb(client.recv(timeout=300))
        if response["type"] != "action":
            raise RuntimeError(f"RTC request {name}: {response}")
        expected_ids = {key: context[key] for key in ("episode_id", "request_id", "observation_step")}
        if response.get("chunk_context") != expected_ids:
            raise AssertionError("RTC response identifiers differ")
        action = response["action"]
        np.save(directory / f"rtc_{name}.npy", action)
        result = compare_arrays(baseline, action)
        result.update(mode=mode, case=name, prefix_length=len(prefix), inference_delay_steps=0,
                      seconds=response["infer_seconds"])
        records.append(result)
        print("RTC synchronous request: " + json.dumps(result), flush=True)
        if name == "empty_prefix" and not result["exact_equal"]:
            raise AssertionError("Empty RTC prefix must preserve ordinary synchronous inference")
        if name == "nonempty_prefix":
            if result["exact_equal"]:
                raise AssertionError("Nonempty RTC prefix did not change this synthetic prediction")
            guided = action
        elif name == "nonempty_prefix_repeat":
            # GPU attention backward may be nondeterministic. Report this
            # separately; ordinary sync and empty-prefix parity remain exact
            # requirements, without an added tolerance.
            result["repeat_comparison"] = compare_arrays(guided, action)
            print("RTC repeat comparison: " + json.dumps(result["repeat_comparison"]), flush=True)


def main(args):
    import torch
    from fasterwam.serve.serialization import packb, unpackb

    args.output_dir = args.output_dir.resolve()
    args.run_dir = args.run_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if args.candidate_serving_config is not None:
        args.candidate_serving_config = args.candidate_serving_config.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    repos = {"reference": args.reference_repo.resolve(), "candidate": args.candidate_repo.resolve()}
    rng = np.random.default_rng(args.seed)
    rows, cols = np.indices((480, 640))
    pattern = np.stack([rows % 256, cols % 256, (rows + cols) % 256], axis=-1).astype(np.uint8)
    images = np.stack([
        np.zeros((3, 480, 640, 3), dtype=np.uint8),
        np.stack([pattern, np.roll(pattern, 73, axis=1), np.flip(pattern, axis=0)]),
        rng.integers(0, 256, (3, 480, 640, 3), dtype=np.uint8),
    ])
    qpos = np.stack([np.zeros(14), np.linspace(-0.7, 0.7, 14), rng.uniform(-0.5, 0.5, 14)]).astype(np.float32)
    np.savez(args.output_dir / "inputs.npz", images=images, qpos=qpos)
    report = {
        "status": "running", "seed": args.seed, "denoise_steps": args.num_inference_steps,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(),
        "python": sys.executable, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "config": identity(args.run_dir / "config.yaml"),
        "checkpoint": identity(args.checkpoint), "stats": identity(args.run_dir / "dataset_stats.json"),
        "inputs": identity(args.output_dir / "inputs.npz"),
        "data": "3 synthetic observations: zeros, camera-specific spatial patterns, seeded random RGB/qpos",
        "candidate_serving_config": (identity(args.candidate_serving_config)
                                     if args.candidate_serving_config else None),
        "exercise_rtc": args.exercise_rtc,
        "repositories": {}, "preprocessing": {}, "actions": [], "rtc_requests": [],
    }
    if args.candidate_serving_config:
        import yaml
        report["candidate_serving_settings"] = yaml.safe_load(args.candidate_serving_config.read_text())
    try:
        for label, repo in repos.items():
            directory = args.output_dir / label
            directory.mkdir()
            report["repositories"][label] = {
                "path": str(repo),
                "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                "sources": [identity(repo / name) for name in (
                    "src/fasterwam/serve/widowxai_policy.py", "src/fasterwam/models/wan22/jointwam.py",
                    "src/fasterwam/models/wan22/sparse_mot.py", "src/fasterwam/runtime.py",
                )],
            }
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--reference-repo", str(repos["reference"]),
                       "--run-dir", str(args.run_dir), "--checkpoint", str(args.checkpoint),
                       "--task", args.task, "--output-dir", str(directory), "--preprocess-worker"]
            with (directory / "preprocessing.log").open("w") as log:
                subprocess.run(command, cwd=repo, env=repo_environment(repo),
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        with np.load(args.output_dir / "reference/preprocessing.npz") as ref, np.load(
            args.output_dir / "candidate/preprocessing.npz"
        ) as candidate:
            for key in ref.files:
                result = compare_arrays(ref[key], candidate[key])
                report["preprocessing"][key] = result
                if not result["exact_equal"]:
                    raise AssertionError(f"CPU preprocessing differs: {key}: {result}")
        print("CPU preprocessing/context/denormalization: exact equality", flush=True)
        for mode in args.modes:
            directories = {}
            for label in repos:
                directories[label] = args.output_dir / label / mode
                directories[label].mkdir()
            with server(args, repos["reference"], directories["reference"], mode) as (ref, ref_meta), server(
                args, repos["candidate"], directories["candidate"], mode,
                serving_config=args.candidate_serving_config,
            ) as (candidate, candidate_meta):
                if args.exercise_rtc and not candidate_meta.get("supports_rtc"):
                    raise AssertionError("Candidate did not enable RTC")
                for field in ("task", "action_horizon", "action_dim", "model_image_shape",
                              "num_video_frames", "num_inference_steps", "control_dt", "action_infer_mode"):
                    if ref_meta[field] != candidate_meta[field]:
                        raise AssertionError(f"Metadata differs: {field}")
                # A repeated request after reset also checks determinism/state.
                for request_index, observation_index in enumerate((0, 1, 2, 1)):
                    if request_index == 3:
                        for client in (ref, candidate):
                            client.send(packb({"type": "reset"}))
                            assert unpackb(client.recv(timeout=30))["type"] == "reset_ok"
                    request = packb({"type": "select_action", "observation": observations(
                        args.output_dir / "inputs.npz"
                    )[observation_index]})
                    responses = []
                    for label, client in (("reference", ref), ("candidate", candidate)):
                        client.send(request)
                        response = unpackb(client.recv(timeout=300))
                        if response["type"] != "action":
                            raise RuntimeError(f"{label}: {response}")
                        np.save(directories[label] / f"action_{request_index}.npy", response["action"])
                        responses.append(response)
                    result = compare_arrays(responses[0]["action"], responses[1]["action"])
                    result.update(mode=mode, request_index=request_index, observation_index=observation_index,
                                  reference_seconds=responses[0]["infer_seconds"],
                                  candidate_seconds=responses[1]["infer_seconds"])
                    report["actions"].append(result)
                    print(json.dumps(result), flush=True)
                    if not result["exact_equal"]:
                        raise AssertionError(f"Synchronous action mismatch: {result}")
                for label in repos:
                    np.testing.assert_array_equal(
                        np.load(directories[label] / "action_1.npy"),
                        np.load(directories[label] / "action_3.npy"),
                    )
                if args.exercise_rtc:
                    exercise_rtc(args, candidate, candidate_meta, directories["candidate"], mode,
                                 report["rtc_requests"])
        report["status"] = "pass"
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = repr(error)
        raise
    finally:
        (args.output_dir / "results.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.preprocess_worker:
        preprocess_worker(arguments)
    else:
        main(arguments)
