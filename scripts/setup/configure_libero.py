#!/usr/bin/env python
"""Write a non-interactive LIBERO config for one isolated benchmark runtime."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--config-dir", required=True)
    args = parser.parse_args()

    source_root = Path(args.source_root).expanduser().resolve()
    benchmark_root = source_root / "libero" / "libero"
    if not benchmark_root.is_dir():
        raise FileNotFoundError(
            f"Expected LIBERO package at {benchmark_root}; got source root {source_root}"
        )
    config_dir = Path(args.config_dir).expanduser().resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    values = {
        "benchmark_root": benchmark_root,
        "bddl_files": benchmark_root / "bddl_files",
        "init_states": benchmark_root / "init_files",
        "datasets": benchmark_root.parent / "datasets",
        "assets": benchmark_root / "assets",
    }
    config_file = config_dir / "config.yaml"
    config_file.write_text(
        "".join(f"{key}: {value}\n" for key, value in values.items()),
        encoding="utf-8",
    )
    print(config_file)


if __name__ == "__main__":
    main()
