# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
MiniVLA Dataset with Multi-Image and Action Chunking Support.

Extends CRANE-X7 dataset with:
- Multi-image input (history frames + wrist camera)
- Action chunking for VQ-based training
- Flexible camera configuration
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

from crane_x7_vla.backends.minivla.action_tokenizer.vq_tokenizer import (
    VQActionTokenizer,
)
from crane_x7_vla.core.data.image_utils import decode_jpeg, resize_image
from crane_x7_vla.core.data.tfrecord_reader import TFRecordReader, find_tfrecord_files


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


@dataclass
class MiniVLADatasetConfig:
    """Configuration for MiniVLA dataset."""

    data_root: Path
    """Root directory containing TFRecord episode files"""

    action_dim: int = 8
    """Action dimension (7 joint angles + 1 gripper)"""

    state_dim: int = 8
    """State dimension (7 joint angles + 1 gripper)"""

    # Multi-image settings
    image_history: int = 2
    """Number of historical frames to include (1 = current only)"""

    use_wrist_camera: bool = True
    """Whether to use wrist camera image"""

    camera_keys: list[str] = field(
        default_factory=lambda: [
            "image_primary",
            "image_wrist",
        ]
    )
    """Camera keys in TFRecord"""

    # Action chunking settings
    action_horizon: int = 8
    """Number of future actions to predict"""

    # Normalization
    normalize_actions: bool = True
    """Whether to normalize actions to [-1, 1]"""

    normalize_states: bool = True
    """Whether to normalize states"""

    # Image settings
    image_size: tuple[int, int] = (224, 224)
    """Target image size (height, width)"""

    image_aug: bool = True
    """Whether to apply image augmentation"""

    # Language
    use_language_instruction: bool = True
    """Whether to use language instructions"""

    default_instruction: str = "manipulate the object"
    """Default language instruction if not present"""


@dataclass
class MiniVLABatchTransform:
    """
    Batch transform for MiniVLA data with multi-image support.

    Converts RLDS batch format to MiniVLA expected format,
    handling multiple images and action chunks.
    """

    action_tokenizer: Any  # VQActionTokenizer or BinActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: Callable
    prompt_builder_fn: Any
    image_history: int = 2
    use_wrist_camera: bool = True
    action_horizon: int = 8
    predict_stop_token: bool = True

    def __call__(self, rlds_batch: dict[str, Any]) -> dict[str, Any]:
        """
        Convert RLDS batch to MiniVLA format.

        Args:
            rlds_batch: Batch from RLDS dataset containing:
                - observation/image_primary: Primary camera images
                - observation/image_wrist: Wrist camera images (optional)
                - action: Action sequence
                - task/language_instruction: Language instruction

        Returns:
            Dictionary with:
                - pixel_values: Tensor of shape (N, C, H, W) where N is num_images
                - input_ids: Token IDs
                - labels: Target token IDs
                - dataset_name: Dataset identifier
        """
        dataset_name = rlds_batch["dataset_name"]

        # Get images (handle history)
        images = []

        # Primary camera with history
        if "observation" in rlds_batch:
            obs = rlds_batch["observation"]
            primary_images = obs.get("image_primary", obs.get("image", None))

            if primary_images is not None:
                # Get historical frames
                for i in range(min(self.image_history, len(primary_images))):
                    img_data = primary_images[i]
                    if isinstance(img_data, np.ndarray):
                        img = Image.fromarray(img_data)
                    else:
                        img = Image.fromarray(np.array(img_data))
                    images.append(img)

            # Wrist camera (current frame only)
            if self.use_wrist_camera:
                wrist_images = obs.get("image_wrist", None)
                if wrist_images is not None and len(wrist_images) > 0:
                    wrist_data = wrist_images[0]
                    if isinstance(wrist_data, np.ndarray):
                        wrist_img = Image.fromarray(wrist_data)
                    else:
                        wrist_img = Image.fromarray(np.array(wrist_data))
                    images.append(wrist_img)

        # Ensure we have at least one image
        if len(images) == 0:
            # Fallback: create dummy image
            images.append(Image.new("RGB", (224, 224), color="black"))

        # Get actions (handle chunking)
        actions = rlds_batch["action"]
        if isinstance(actions, list | np.ndarray):
            actions = np.array(actions)

        # Get action chunk
        # Single action (1D) -> add batch dim; Multiple actions (2D) -> take first H
        action_chunk = actions[np.newaxis, :] if len(actions.shape) == 1 else actions[: self.action_horizon]

        # Pad if needed
        if len(action_chunk) < self.action_horizon:
            pad_size = self.action_horizon - len(action_chunk)
            padding = np.repeat(action_chunk[-1:], pad_size, axis=0)
            action_chunk = np.concatenate([action_chunk, padding], axis=0)

        # Get language instruction
        task = rlds_batch.get("task", {})
        lang = task.get("language_instruction", b"manipulate the object")
        if isinstance(lang, bytes):
            lang = lang.decode()
        lang = lang.lower()

        # Tokenize actions
        if isinstance(self.action_tokenizer, VQActionTokenizer):
            # VQ tokenization -> sequence of codebook indices
            action_tokens = self.action_tokenizer.encode(action_chunk)
            action_str = " ".join([f"<action_{i}>" for i in range(len(action_tokens))])
        else:
            # Bin tokenization -> one token per dimension
            action_tokens = self.action_tokenizer.encode(action_chunk[0])
            action_str = " ".join([f"<action_{i}>" for i in range(len(action_tokens))])

        # Construct prompt
        prompt_builder = self.prompt_builder_fn("minivla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_str},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize text
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Replace action placeholder tokens with actual action tokens
        # Find action token positions and replace
        # For now, append action tokens directly
        input_ids = torch.tensor(input_ids[: -len(action_tokens)] + action_tokens)
        labels = torch.tensor(labels[: -len(action_tokens)] + action_tokens)

        # Transform images
        pixel_values_list = []
        for img in images:
            pv = self.image_transform(img)
            pixel_values_list.append(pv)

        # Stack images: (N, C, H, W)
        pixel_values = torch.stack(pixel_values_list, dim=0)

        # Mask non-action tokens in labels
        labels[: -(len(action_tokens) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
            "dataset_name": dataset_name,
            "action_chunk": torch.tensor(action_chunk, dtype=torch.float32),
        }


class MiniVLADataset(IterableDataset):
    """
    PyTorch IterableDataset for MiniVLA with multi-image and action chunking.

    Features:
    - Multi-image input (history + wrist camera)
    - Action chunking for VQ-based training
    - Step-level train/overfit splitting for overfitting detection
    """

    # TFRecord feature description for CRANE-X7 format with multi-camera (tfrecord library format)
    TFRECORD_FEATURES: ClassVar[dict[str, str]] = {
        "observation/image_primary": "byte",
        "observation/image_wrist": "byte",
        "observation/state": "float",
        "action": "float",
        "task/language_instruction": "byte",
        "episode_id": "int",
        "step_id": "int",
    }

    # Alternative feature names for backward compatibility
    TFRECORD_FEATURES_ALT: ClassVar[dict[str, str]] = {
        "observation/proprio": "float",
        "observation/image_primary": "byte",
        "action": "float",
    }

    def __init__(
        self,
        data_root: Path,
        dataset_name: str,
        batch_transform: MiniVLABatchTransform,
        config: MiniVLADatasetConfig,
        resize_resolution: tuple[int, int] = (224, 224),
        shuffle_buffer_size: int = 100_000,
        image_aug: bool = True,
        overfit_split_ratio: float = 0.0,
        split: str = "train",
        train: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        """
        Initialize MiniVLA dataset.

        Args:
            data_root: Root directory containing TFRecord files
            dataset_name: Name of dataset subdirectory
            batch_transform: Transform to apply to batches
            config: Dataset configuration
            resize_resolution: Target image size
            shuffle_buffer_size: Size of shuffle buffer
            image_aug: Whether to apply image augmentation
            overfit_split_ratio: Ratio of steps for overfitting detection
            split: 'train' or 'overfit'
            train: Whether in training mode (enables repeat)
            rank: Process rank for distributed training
            world_size: Total number of processes
        """
        self.data_root = Path(data_root)
        self.dataset_name = dataset_name
        self.batch_transform = batch_transform
        self.config = config
        self.resize_resolution = resize_resolution
        self.shuffle_buffer_size = shuffle_buffer_size
        self.image_aug = image_aug
        self.overfit_split_ratio = overfit_split_ratio
        self.split = split
        self.train = train
        self.rank = rank
        self.world_size = world_size

        # Find TFRecord files
        self.tfrecord_files = self._find_tfrecord_files()

        # Dataset statistics (computed on first iteration)
        self._dataset_statistics = None

    def _find_tfrecord_files(self) -> list[Path]:
        """Find all TFRecord files in data directory."""
        data_dir = self.data_root / self.dataset_name
        if not data_dir.exists():
            data_dir = self.data_root

        return find_tfrecord_files(data_dir)

    @property
    def dataset_statistics(self) -> dict[str, Any]:
        """Get dataset statistics (computed lazily)."""
        if self._dataset_statistics is None:
            self._dataset_statistics = self._compute_statistics()
        return self._dataset_statistics

    def _compute_statistics(self) -> dict[str, Any]:
        """Compute dataset statistics for normalization."""
        # Placeholder - would compute mean/std from data
        return {
            "action": {
                "mean": np.zeros(self.config.action_dim),
                "std": np.ones(self.config.action_dim),
                "min": -np.ones(self.config.action_dim),
                "max": np.ones(self.config.action_dim),
            },
            "state": {
                "mean": np.zeros(self.config.state_dim),
                "std": np.ones(self.config.state_dim),
            },
        }

    def _parse_tfrecord(self, example: dict[str, Any]) -> dict[str, Any]:
        """Parse a single TFRecord example."""
        # Decode images
        obs = {}
        for key in ["observation/image_primary", "observation/image_wrist"]:
            img_bytes = example.get(key)
            if img_bytes and img_bytes != b"":
                try:
                    img = decode_jpeg(img_bytes, channels=3)
                    img = resize_image(img, self.resize_resolution)
                    obs[key.split("/")[1]] = img
                except Exception:
                    pass

        # Get action
        action = example.get("action")
        action = np.array(action, dtype=np.float32).reshape(8) if action is not None else np.zeros(8, dtype=np.float32)

        # Get state (for alternative key name)
        state = example.get("observation/state") or example.get("observation/proprio")
        if state is not None:
            state = np.array(state, dtype=np.float32).reshape(8)

        # Get language instruction
        lang = example.get("task/language_instruction", b"manipulate the object")
        if isinstance(lang, bytes):
            lang = lang

        # Get episode and step IDs
        episode_id = example.get("episode_id", 0)
        if isinstance(episode_id, np.ndarray):
            episode_id = int(episode_id.flat[0]) if episode_id.size > 0 else 0
        step_id = example.get("step_id", 0)
        if isinstance(step_id, np.ndarray):
            step_id = int(step_id.flat[0]) if step_id.size > 0 else 0

        return {
            "observation": obs,
            "action": action,
            "task": {
                "language_instruction": lang,
            },
            "episode_id": episode_id,
            "step_id": step_id,
            "dataset_name": self.dataset_name,
        }

    def _should_include_step(self, episode_id: int, step_id: int) -> bool:
        """
        Determine if a step should be included based on split.

        Uses deterministic hash for reproducible train/overfit splitting.
        """
        if self.overfit_split_ratio <= 0:
            return True

        # Hash episode and step for deterministic splitting
        hash_val = hash((episode_id, step_id)) % 1000
        threshold = int(self.overfit_split_ratio * 1000)

        if self.split == "overfit":
            return hash_val < threshold
        else:  # train
            return hash_val >= threshold

    def __iter__(self):
        """Iterate over dataset."""
        # Create TFRecord reader
        if not self.tfrecord_files:
            return

        # Shard files for distributed training
        files_to_use = self.tfrecord_files
        if self.world_size > 1:
            files_to_use = [f for i, f in enumerate(self.tfrecord_files) if i % self.world_size == self.rank]

        while True:
            # Read all examples
            all_examples = []
            reader = TFRecordReader(
                files_to_use,
                feature_spec=self.TFRECORD_FEATURES,
                use_alternative_keys=True,
            )

            for example in reader:
                parsed = self._parse_tfrecord(example)
                episode_id = parsed["episode_id"]
                step_id = parsed["step_id"]

                if self._should_include_step(episode_id, step_id):
                    all_examples.append(parsed)

            # Shuffle if training
            if self.train and self.shuffle_buffer_size > 0:
                np.random.shuffle(all_examples)

            # Buffer for action chunking (group by episode)
            episode_buffer = []
            current_episode_id = None

            for parsed in all_examples:
                episode_id = parsed["episode_id"]

                # Handle episode boundaries for action chunking
                if current_episode_id != episode_id:
                    # Process previous episode buffer
                    if episode_buffer:
                        yield from self._process_episode_buffer(episode_buffer)
                    episode_buffer = []
                    current_episode_id = episode_id

                episode_buffer.append(parsed)

            # Process remaining buffer
            if episode_buffer:
                yield from self._process_episode_buffer(episode_buffer)

            # If not training, don't repeat
            if not self.train:
                break

    def _process_episode_buffer(self, buffer: list[dict[str, Any]]):
        """
        Process episode buffer to create action chunks.

        Args:
            buffer: List of parsed steps from one episode

        Yields:
            Transformed batches with action chunks
        """
        horizon = self.config.action_horizon
        history = self.config.image_history

        for i in range(len(buffer)):
            # Gather historical images
            images_history = []
            for j in range(max(0, i - history + 1), i + 1):
                if "image_primary" in buffer[j]["observation"]:
                    images_history.append(buffer[j]["observation"]["image_primary"])

            # Pad history if needed
            while len(images_history) < history:
                images_history.insert(0, images_history[0] if images_history else None)

            # Gather action chunk
            actions = []
            for j in range(i, min(i + horizon, len(buffer))):
                actions.append(buffer[j]["action"])

            # Create batch with multi-image and action chunk
            batch = {
                "observation": {
                    "image_primary": np.array(images_history),
                },
                "action": np.array(actions),
                "task": buffer[i]["task"],
                "dataset_name": buffer[i]["dataset_name"],
            }

            # Add wrist camera if available
            if self.config.use_wrist_camera and "image_wrist" in buffer[i]["observation"]:
                batch["observation"]["image_wrist"] = np.array([buffer[i]["observation"]["image_wrist"]])

            # Apply transform
            try:
                yield self.batch_transform(batch)
            except Exception:
                # Skip problematic samples
                continue
