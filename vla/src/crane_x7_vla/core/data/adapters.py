# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Data adapter for loading CRANE-X7 TFRecord data.

Provides a unified interface for loading data and converting to different formats.
"""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from crane_x7_vla.core.data.image_utils import decode_jpeg, decode_raw
from crane_x7_vla.core.data.tfrecord_reader import TFRecordReader, find_tfrecord_files


logger = logging.getLogger(__name__)


class CraneX7DataAdapter:
    """
    Adapter for loading CRANE-X7 TFRecord data.

    Handles loading from TFRecord files and provides data in various formats
    for different VLA backends.
    """

    # TFRecord feature description (tfrecord library format)
    FEATURE_DESCRIPTION: ClassVar[dict[str, str]] = {
        "observation/state": "float",
        "observation/image": "byte",
        "observation/depth": "byte",
        "observation/timestamp": "float",
        "action": "float",
        "prompt": "byte",
        "task": "byte",
    }

    # Alternative key names (for compatibility)
    KEY_ALIASES: ClassVar[dict[str, str]] = {
        "observation/proprio": "observation/state",
        "observation/image_primary": "observation/image",
        "observation/depth_primary": "observation/depth",
        "task/language_instruction": "prompt",
        "dataset_name": "task",
    }

    # Minimal feature description for backward compatibility
    FEATURE_DESCRIPTION_MINIMAL: ClassVar[dict[str, str]] = {
        "observation/state": "float",
        "observation/image": "byte",
        "action": "float",
    }

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        shuffle: bool = True,
        buffer_size: int = 1000,
        include_depth: bool = False,
    ):
        """
        Initialize data adapter.

        Args:
            data_root: Root directory containing TFRecord files
            split: Data split ('train' or 'val')
            shuffle: Whether to shuffle the dataset
            buffer_size: Buffer size for shuffling
            include_depth: Whether to include depth images
        """
        self.data_root = Path(data_root)
        self.split = split
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.include_depth = include_depth

        # Find TFRecord files
        self.tfrecord_files = self._find_tfrecord_files()
        logger.info(f"Found {len(self.tfrecord_files)} TFRecord files in {self.data_root}")

    def _find_tfrecord_files(self) -> list[Path]:
        """Find all TFRecord files in the data directory."""
        # First try with split filter
        pattern = f"**/*{self.split}*.tfrecord*"
        files = list(self.data_root.glob(pattern))

        if not files:
            # Try without split filter using common function
            files = find_tfrecord_files(self.data_root)

        return sorted(files)

    def _normalize_keys(self, example: dict[str, Any]) -> dict[str, Any]:
        """Normalize key names (convert aliases to standard names)."""
        normalized = {}
        for key, value in example.items():
            if key in self.KEY_ALIASES:
                normalized[self.KEY_ALIASES[key]] = value
            else:
                normalized[key] = value
        return normalized

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        """Decode JPEG-encoded image."""
        return decode_jpeg(image_bytes, channels=3)

    def _decode_depth(self, depth_bytes: bytes) -> np.ndarray:
        """Decode depth image from bytes."""
        # Depth is stored as float32 bytes
        return decode_raw(depth_bytes, dtype=np.float32)

    def iterate_episodes(self) -> Iterator[dict[str, np.ndarray]]:
        """
        Iterate over episodes (unbatched).

        Yields:
            Dictionary containing episode data
        """
        reader = TFRecordReader(
            self.tfrecord_files,
            feature_spec=self.FEATURE_DESCRIPTION,
            use_alternative_keys=True,
        )

        for example in reader:
            episode = self._normalize_keys(example)

            # Convert arrays to proper shapes
            if "observation/state" in episode:
                state = episode["observation/state"]
                if state is not None:
                    episode["observation/state"] = np.array(state, dtype=np.float32).reshape(8)

            if "action" in episode:
                action = episode["action"]
                if action is not None:
                    episode["action"] = np.array(action, dtype=np.float32).reshape(8)

            # Decode image
            if episode.get("observation/image"):
                try:
                    episode["observation/image"] = self._decode_image(episode["observation/image"])
                except Exception:
                    episode["observation/image"] = None

            # Decode depth if included
            if self.include_depth and "observation/depth" in episode and episode["observation/depth"]:
                try:
                    episode["observation/depth"] = self._decode_depth(episode["observation/depth"])
                except Exception:
                    episode["observation/depth"] = None

            yield episode

    def compute_statistics(self, keys: list[str] | None = None) -> dict[str, dict[str, np.ndarray]]:
        """
        Compute statistics over the dataset for normalization.

        Args:
            keys: Keys to compute statistics for

        Returns:
            Dictionary mapping keys to statistics dictionaries
        """
        if keys is None:
            keys = ["observation/state", "action"]
        logger.info("Computing dataset statistics...")

        # Collect all data
        data = {key: [] for key in keys}

        for episode in self.iterate_episodes():
            for key in keys:
                if key in episode and episode[key] is not None:
                    data[key].append(episode[key])

        # Compute statistics
        stats = {}
        for key in keys:
            if data[key]:
                arr = np.array(data[key])
                stats[key] = {
                    "mean": np.mean(arr, axis=0),
                    "std": np.std(arr, axis=0),
                    "min": np.min(arr, axis=0),
                    "max": np.max(arr, axis=0),
                    "q_low": np.quantile(arr, 0.01, axis=0),
                    "q_high": np.quantile(arr, 0.99, axis=0),
                }

        logger.info(f"Computed statistics for keys: {list(stats.keys())}")
        return stats

    def __len__(self) -> int:
        """Get approximate dataset size (number of steps across all files)."""
        # This is approximate - iterating once to count
        count = sum(1 for _ in self.iterate_episodes())
        return count
