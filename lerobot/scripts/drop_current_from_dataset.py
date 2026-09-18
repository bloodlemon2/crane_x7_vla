#!/usr/bin/env python3
"""Create a LeRobot dataset whose observation.state excludes motor currents.

The current-enabled recorder appends motor currents to the end of
``observation.state``. This script keeps the first ``--keep-dims`` values and
copies all other dataset files unchanged, including videos.

Example:
    .venv/bin/python scripts/drop_current_from_dataset.py \
        --input /path/to/recrd-separate-transparent-bags-current \
        --output /path/to/recrd-separate-transparent-bags-no-current \
        --keep-dims 8

The output directory must not already exist. The source dataset is never
modified. To upload the converted directory, pass ``--push-to-hub`` together
with ``--repo-id`` after configuring Hugging Face authentication.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
STATE_KEY = "observation.state"


def _slice_value(value: Any, keep_dims: int) -> Any:
    """Slice one vector while preserving pandas/numpy/list scalar types."""
    if isinstance(value, np.ndarray):
        return value[:keep_dims]
    if isinstance(value, list):
        return value[:keep_dims]
    if isinstance(value, tuple):
        return value[:keep_dims]
    return value


def _slice_series(series: pd.Series, keep_dims: int) -> pd.Series:
    return series.map(lambda value: _slice_value(value, keep_dims))


def _rewrite_data_parquet(path: Path, keep_dims: int) -> None:
    frame = pd.read_parquet(path)
    if STATE_KEY not in frame.columns:
        return
    frame[STATE_KEY] = _slice_series(frame[STATE_KEY], keep_dims)
    frame.to_parquet(path, index=False)


def _rewrite_episode_stats(path: Path, keep_dims: int) -> None:
    frame = pd.read_parquet(path)
    prefix = f"stats/{STATE_KEY}/"
    columns = [column for column in frame.columns if column.startswith(prefix)]
    if not columns:
        return
    for column in columns:
        frame[column] = _slice_series(frame[column], keep_dims)
    frame.to_parquet(path, index=False)


def _rewrite_json_stats(path: Path, keep_dims: int) -> None:
    stats = json.loads(path.read_text())
    state_stats = stats.get(STATE_KEY)
    if not isinstance(state_stats, dict):
        raise ValueError(f"{path} does not contain {STATE_KEY!r} statistics")
    for name, values in state_stats.items():
        if name != "count":
            state_stats[name] = _slice_value(values, keep_dims)
    path.write_text(json.dumps(stats, indent=4, ensure_ascii=False) + "\n")


def _rewrite_info(path: Path, keep_dims: int) -> None:
    info = json.loads(path.read_text())
    feature = info.get("features", {}).get(STATE_KEY)
    if not isinstance(feature, dict):
        raise ValueError(f"{path} does not contain the {STATE_KEY!r} feature")

    old_shape = feature.get("shape")
    if not old_shape or old_shape[0] < keep_dims:
        raise ValueError(f"Cannot keep {keep_dims} values from shape {old_shape}")
    feature["shape"] = [keep_dims, *old_shape[1:]]
    if isinstance(feature.get("names"), list):
        feature["names"] = feature["names"][:keep_dims]
    path.write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n")


def convert_dataset(input_dir: Path, output_dir: Path, keep_dims: int) -> None:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input dataset does not exist: {input_dir}")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    info_path = input_dir / "meta" / "info.json"
    stats_path = input_dir / "meta" / "stats.json"
    if not info_path.is_file() or not stats_path.is_file():
        raise ValueError(f"Not a LeRobot dataset: {input_dir}")

    info = json.loads(info_path.read_text())
    old_shape = info.get("features", {}).get(STATE_KEY, {}).get("shape")
    if not old_shape or len(old_shape) != 1 or keep_dims > old_shape[0]:
        raise ValueError(f"Invalid state shape {old_shape}; cannot keep {keep_dims} values")
    if keep_dims == old_shape[0]:
        raise ValueError("The requested dimension is already the source dimension; nothing to convert")

    LOGGER.info("Copying dataset files: %s -> %s", input_dir, output_dir)
    shutil.copytree(input_dir, output_dir)

    data_files = sorted((output_dir / "data").rglob("*.parquet"))
    for path in data_files:
        _rewrite_data_parquet(path, keep_dims)

    episode_files = sorted((output_dir / "meta" / "episodes").rglob("*.parquet"))
    for path in episode_files:
        _rewrite_episode_stats(path, keep_dims)

    _rewrite_info(output_dir / "meta" / "info.json", keep_dims)
    _rewrite_json_stats(output_dir / "meta" / "stats.json", keep_dims)
    LOGGER.info("Converted %d data files and %d episode metadata files", len(data_files), len(episode_files))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source LeRobot dataset directory")
    parser.add_argument("--output", type=Path, required=True, help="New dataset directory")
    parser.add_argument(
        "--keep-dims",
        type=int,
        default=8,
        help="Number of leading observation.state values to keep (default: 8)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload the converted output directory to the Hugging Face Hub",
    )
    parser.add_argument("--repo-id", help="Hugging Face dataset repo ID, e.g. user/dataset-no-current")
    parser.add_argument("--private", action="store_true", help="Create the Hub repository as private")
    args = parser.parse_args()

    if args.keep_dims <= 0:
        parser.error("--keep-dims must be positive")
    if args.push_to_hub and not args.repo_id:
        parser.error("--repo-id is required with --push-to-hub")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    convert_dataset(args.input, args.output, args.keep_dims)

    if args.push_to_hub:
        from huggingface_hub import HfApi

        LOGGER.info("Uploading converted dataset to %s", args.repo_id)
        api = HfApi()
        api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
        api.upload_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=args.output)
        info = json.loads((args.output / "meta" / "info.json").read_text())
        codebase_version = info.get("codebase_version")
        if not codebase_version:
            raise ValueError("Converted dataset meta/info.json has no codebase_version")
        api.create_tag(repo_id=args.repo_id, tag=codebase_version, repo_type="dataset")
        LOGGER.info("Created Hub dataset tag %s", codebase_version)
        LOGGER.info("Uploaded dataset to https://huggingface.co/datasets/%s", args.repo_id)


if __name__ == "__main__":
    main()
