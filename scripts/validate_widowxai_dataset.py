"""Build WidowXAI statistics and validate full FasterWAM train/eval samples."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from fasterwam.utils import misc
from fasterwam.utils.config_resolvers import register_default_resolvers


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_nested_close(actual, reference, path: str = "root", max_abs=None) -> float:
    """Compare JSON-compatible stats while tolerating reduction-order noise."""
    if max_abs is None:
        max_abs = [0.0]
    if isinstance(reference, dict):
        if not isinstance(actual, dict) or set(actual) != set(reference):
            raise ValueError(f"Stats structure mismatch at {path}")
        for key in reference:
            _assert_nested_close(actual[key], reference[key], f"{path}.{key}", max_abs)
    elif isinstance(reference, list):
        if not isinstance(actual, list) or len(actual) != len(reference):
            raise ValueError(f"Stats list mismatch at {path}")
        for index, (actual_value, reference_value) in enumerate(zip(actual, reference)):
            _assert_nested_close(
                actual_value,
                reference_value,
                f"{path}[{index}]",
                max_abs,
            )
    elif isinstance(reference, (int, float)) and not isinstance(reference, bool):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            raise ValueError(f"Stats numeric type mismatch at {path}")
        difference = abs(float(actual) - float(reference))
        max_abs[0] = max(max_abs[0], difference)
        if not math.isclose(float(actual), float(reference), rel_tol=1e-5, abs_tol=1e-7):
            raise ValueError(
                f"Stats value mismatch at {path}: actual={actual} reference={reference}"
            )
    elif actual != reference:
        raise ValueError(
            f"Stats value mismatch at {path}: actual={actual!r} reference={reference!r}"
        )
    return max_abs[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-stats",
        type=Path,
        default=Path("/groups/gcg51472/acf16376hu/fasterwam/dataset_stats.json"),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/groups/gcg51472/acf16376hu/fasterwam/preparation/widowxai"),
    )
    parser.add_argument(
        "--reference-stats",
        type=Path,
        default=None,
        help=(
            "Optional FastWAM stats file. Independently generated values must "
            "match numerically; the reference is then copied verbatim so both "
            "models train with exactly the same constants."
        ),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_stats.parent.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(args.work_dir)
    register_default_resolvers()

    with initialize_config_dir(config_dir=str(repo_root / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=["task=widowxai_fasterwam_3cam_384_1e-4"],
        )

    cfg.data.train.pretrained_norm_stats = None
    train_dataset = instantiate(cfg.data.train)
    generated_stats = args.work_dir / "dataset_stats.json"
    if not generated_stats.is_file():
        raise FileNotFoundError(f"Stats computation did not create {generated_stats}")
    shutil.copy2(generated_stats, args.output_stats)

    if args.reference_stats is not None:
        if not args.reference_stats.is_file():
            raise FileNotFoundError(f"Reference stats do not exist: {args.reference_stats}")
        generated_hash = _sha256(args.output_stats)
        reference_hash = _sha256(args.reference_stats)
        with args.output_stats.open("r", encoding="utf-8") as handle:
            generated_payload = json.load(handle)
        with args.reference_stats.open("r", encoding="utf-8") as handle:
            reference_payload = json.load(handle)
        max_abs = _assert_nested_close(generated_payload, reference_payload)
        print(
            "stats numerically match FastWAM reference: "
            f"max_abs={max_abs:.3e} generated_sha256={generated_hash} "
            f"reference_sha256={reference_hash}"
        )
        # Reduction order can change the last float32 bit. Use the exact same
        # frozen file for both models after confirming numerical equivalence.
        shutil.copy2(args.reference_stats, args.output_stats)
        print(f"installed exact FastWAM stats: sha256={_sha256(args.output_stats)}")

    base = train_dataset.lerobot_dataset
    raw = base._get_episode_data(0)
    transformed = base.processor.action_state_transform(copy.deepcopy(raw))
    raw_action = raw["action"]["default"]
    state = raw["state"]["default"]
    rel_action = transformed["action"]["default"]
    joint_mask = torch.tensor(
        [
            True, True, True, True, True, True, False,
            True, True, True, True, True, True, False,
        ]
    )
    expected = torch.where(joint_mask, raw_action - state[..., :1, :], raw_action)
    torch.testing.assert_close(rel_action, expected)

    expected_shapes = {
        "video": (3, 9, 384, 320),
        "action": (32, 14),
        "proprio": (32, 14),
        "context": (128, 4096),
        "context_mask": (128,),
    }

    def validate_sample(dataset, index: int, label: str) -> None:
        sample = dataset[index]
        for key, shape in expected_shapes.items():
            actual = tuple(sample[key].shape)
            if actual != shape:
                raise ValueError(
                    f"{label}: {key} shape mismatch: expected {shape}, got {actual}"
                )
        if not torch.isfinite(sample["video"]).all():
            raise ValueError(f"{label}: video contains non-finite values.")
        if not torch.isfinite(sample["action"]).all():
            raise ValueError(f"{label}: action contains non-finite values.")

    train_dirs = [str(path) for path in cfg.data.train.dataset_dirs]
    expected_train_dir = "/home/acf16376hu/data/real_widowxai_success_only"
    if train_dirs != [expected_train_dir]:
        raise ValueError(f"Expected only {expected_train_dir}, got {train_dirs}")
    train_base = train_dataset.lerobot_dataset.multi_dataset
    if len(train_dataset) != 58_600 or train_base.num_episodes != 98:
        raise ValueError(
            f"Unexpected train size: frames={len(train_dataset)} "
            f"episodes={train_base.num_episodes}"
        )
    validate_sample(train_dataset, 0, "train")

    cfg.data.val.pretrained_norm_stats = str(args.output_stats)
    val_dataset = instantiate(cfg.data.val)
    eval_dirs = [str(path) for path in cfg.data.val.dataset_dirs]
    expected_eval_dir = "/home/acf16376hu/data/real_widowxai_test"
    if eval_dirs != [expected_eval_dir]:
        raise ValueError(f"Expected only {expected_eval_dir}, got {eval_dirs}")
    val_base = val_dataset.lerobot_dataset.multi_dataset
    if len(val_dataset) != 1_000 or val_base.num_episodes != 2:
        raise ValueError(
            f"Unexpected eval size: frames={len(val_dataset)} "
            f"episodes={val_base.num_episodes}"
        )
    validate_sample(val_dataset, 0, "eval")

    print(f"train frames/episodes: {len(train_dataset)}/{train_base.num_episodes}")
    print(f"eval frames/episodes: {len(val_dataset)}/{val_base.num_episodes}")
    print(f"sample shapes: {expected_shapes}")
    print(f"dataset stats: {args.output_stats}")
    print("WidowXAI FasterWAM dataset validation ok")


if __name__ == "__main__":
    main()
