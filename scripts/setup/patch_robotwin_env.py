#!/usr/bin/env python
"""Apply the idempotent compatibility fixes required by RoboTwin."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def package_root(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.submodule_search_locations is None:
        raise ModuleNotFoundError(name)
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"Expected compatibility target was not found in {path}")
    path.write_text(text.replace(old, new), encoding="utf-8")
    print(f"patched {path}")


def main() -> None:
    urdf_loader = package_root("sapien") / "wrapper" / "urdf_loader.py"
    replace_once(
        urdf_loader,
        'with open(urdf_file, "r") as f:',
        'with open(urdf_file, "r", encoding="utf-8") as f:',
    )
    replace_once(
        urdf_loader,
        'with open(srdf_file, "r") as f:',
        'with open(srdf_file, "r", encoding="utf-8") as f:',
    )
    planner = package_root("mplib") / "planner.py"
    replace_once(
        planner,
        "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:",
        "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:",
    )


if __name__ == "__main__":
    main()
