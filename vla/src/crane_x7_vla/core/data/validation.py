# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Dataset validation utilities."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from crane_x7_vla.core.data.tfrecord_reader import TFRecordReader, find_tfrecord_files
from crane_x7_vla.core.utils.logging import get_logger
from crane_x7_vla.data_types import DatasetInfo


logger = get_logger(__name__)


@dataclass
class DatasetValidationResult:
    """Result of dataset validation."""

    is_valid: bool
    """Whether the dataset passed all validation checks"""

    num_episodes: int
    """Number of episodes found"""

    num_transitions: int
    """Total number of transitions across all episodes"""

    errors: list[str] = field(default_factory=list)
    """List of error messages"""

    warnings: list[str] = field(default_factory=list)
    """List of warning messages"""

    info: DatasetInfo | None = None
    """Dataset information (if successfully extracted)"""


def validate_tfrecord_dataset(data_root: Path | str) -> DatasetValidationResult:
    """
    Validate a TFRecord dataset for VLA training.

    Checks:
    - Directory exists and contains TFRecord files
    - Each TFRecord can be parsed
    - Required fields are present (observation, action)
    - Data shapes are consistent

    Args:
        data_root: Path to dataset root directory

    Returns:
        DatasetValidationResult with validation status and info
    """
    data_root = Path(data_root)
    errors: list[str] = []
    warnings: list[str] = []

    # Check directory exists
    if not data_root.exists():
        return DatasetValidationResult(
            is_valid=False,
            num_episodes=0,
            num_transitions=0,
            errors=[f"Dataset directory not found: {data_root}"],
        )

    # Find TFRecord files
    tfrecord_files = find_tfrecord_files(data_root)

    if not tfrecord_files:
        return DatasetValidationResult(
            is_valid=False,
            num_episodes=0,
            num_transitions=0,
            errors=[f"No TFRecord files found in {data_root}"],
        )

    num_episodes = len(tfrecord_files)
    total_transitions = 0
    action_dim = None
    state_dim = None
    has_images = False
    has_depth = False
    image_size = None

    # Validate each TFRecord file
    for tfrecord_path in tfrecord_files:
        try:
            episode_errors, episode_info = _validate_single_tfrecord(tfrecord_path)
            errors.extend(episode_errors)

            if episode_info:
                total_transitions += episode_info.get("num_transitions", 0)

                # Check consistency of dimensions
                if action_dim is None:
                    action_dim = episode_info.get("action_dim")
                elif action_dim != episode_info.get("action_dim"):
                    errors.append(
                        f"Inconsistent action dimension in {tfrecord_path.name}: "
                        f"expected {action_dim}, got {episode_info.get('action_dim')}"
                    )

                if state_dim is None:
                    state_dim = episode_info.get("state_dim")

                if episode_info.get("has_images"):
                    has_images = True
                    if image_size is None:
                        image_size = episode_info.get("image_size")

                if episode_info.get("has_depth"):
                    has_depth = True

        except Exception as e:
            errors.append(f"Error validating {tfrecord_path.name}: {e}")

    # Create dataset info
    dataset_info = None
    if action_dim is not None and state_dim is not None:
        dataset_info = DatasetInfo(
            num_episodes=num_episodes,
            num_transitions=total_transitions,
            action_dim=action_dim,
            state_dim=state_dim,
            has_images=has_images,
            has_depth=has_depth,
            image_size=image_size,
        )

    return DatasetValidationResult(
        is_valid=len(errors) == 0,
        num_episodes=num_episodes,
        num_transitions=total_transitions,
        errors=errors,
        warnings=warnings,
        info=dataset_info,
    )


def _validate_single_tfrecord(tfrecord_path: Path) -> tuple[list[str], dict[str, Any] | None]:
    """
    Validate a single TFRecord file.

    Args:
        tfrecord_path: Path to TFRecord file

    Returns:
        Tuple of (errors list, info dict or None)
    """
    errors: list[str] = []
    info: dict[str, Any] = {}

    # Feature spec for validation (accept all common fields)
    feature_spec = {
        "observation/state": "float",
        "observation/proprio": "float",
        "observation/image": "byte",
        "observation/image_primary": "byte",
        "observation/depth": "byte",
        "action": "float",
        "task/language_instruction": "byte",
        "prompt": "byte",
    }

    try:
        reader = TFRecordReader(
            [tfrecord_path],
            feature_spec=feature_spec,
            use_alternative_keys=False,
        )

        num_records = 0
        action_dim = None
        state_dim = None
        has_images = False
        has_depth = False
        validated_records = 0

        for example in reader:
            num_records += 1

            # Only validate first 10 records for quick validation
            if validated_records < 10:
                validated_records += 1

                # Check for required fields
                action = example.get("action")
                if action is None:
                    errors.append(f"{tfrecord_path.name}: Missing 'action' field")
                    continue

                # Extract action dimension
                if isinstance(action, list | np.ndarray):
                    action_arr = np.array(action).flatten()
                    current_action_dim = len(action_arr)
                    if action_dim is None:
                        action_dim = current_action_dim
                    elif action_dim != current_action_dim:
                        errors.append(f"{tfrecord_path.name}: Inconsistent action dimension")

                # Check for state (try both key names)
                state = example.get("observation/state") or example.get("observation/proprio")
                if state is not None and isinstance(state, list | np.ndarray):
                    state_arr = np.array(state).flatten()
                    state_dim = len(state_arr)

                # Check for images (try both key names)
                image = example.get("observation/image") or example.get("observation/image_primary")
                if image is not None and image != b"":
                    has_images = True

                # Check for depth
                depth = example.get("observation/depth")
                if depth is not None and depth != b"":
                    has_depth = True

        info = {
            "num_transitions": num_records,
            "action_dim": action_dim,
            "state_dim": state_dim,
            "has_images": has_images,
            "has_depth": has_depth,
            "image_size": None,  # Image size requires decoding, skip for performance
        }

    except Exception as e:
        errors.append(f"{tfrecord_path.name}: Failed to read: {e}")
        return errors, None

    return errors, info


def validate_npz_dataset(data_root: Path | str) -> DatasetValidationResult:
    """
    Validate an NPZ dataset for VLA training.

    Args:
        data_root: Path to dataset root directory

    Returns:
        DatasetValidationResult with validation status and info
    """
    data_root = Path(data_root)
    errors: list[str] = []
    warnings: list[str] = []

    if not data_root.exists():
        return DatasetValidationResult(
            is_valid=False,
            num_episodes=0,
            num_transitions=0,
            errors=[f"Dataset directory not found: {data_root}"],
        )

    # Find NPZ files
    npz_files = list(data_root.glob("episode_*/episode_data.npz"))
    if not npz_files:
        npz_files = list(data_root.glob("**/*.npz"))

    if not npz_files:
        return DatasetValidationResult(
            is_valid=False,
            num_episodes=0,
            num_transitions=0,
            errors=[f"No NPZ files found in {data_root}"],
        )

    try:
        import numpy as np
    except ImportError:
        return DatasetValidationResult(
            is_valid=False,
            num_episodes=len(npz_files),
            num_transitions=0,
            errors=["NumPy not installed"],
        )

    num_episodes = len(npz_files)
    total_transitions = 0
    action_dim = None
    state_dim = None
    has_images = False
    has_depth = False
    image_size = None

    for npz_path in npz_files:
        try:
            data = np.load(npz_path)

            # Check required fields
            if "actions" not in data:
                errors.append(f"{npz_path.name}: Missing 'actions' field")
                continue

            actions = data["actions"]
            total_transitions += len(actions)

            if action_dim is None:
                action_dim = actions.shape[-1] if len(actions.shape) > 1 else 1

            if "states" in data:
                states = data["states"]
                if state_dim is None:
                    state_dim = states.shape[-1] if len(states.shape) > 1 else 1

            if "images" in data:
                has_images = True
                images = data["images"]
                if len(images.shape) >= 3 and image_size is None:
                    image_size = (images.shape[1], images.shape[2])

            if "depths" in data:
                has_depth = True

        except Exception as e:
            errors.append(f"{npz_path.name}: Failed to read: {e}")

    dataset_info = None
    if action_dim is not None:
        dataset_info = DatasetInfo(
            num_episodes=num_episodes,
            num_transitions=total_transitions,
            action_dim=action_dim,
            state_dim=state_dim or 0,
            has_images=has_images,
            has_depth=has_depth,
            image_size=image_size,
        )

    return DatasetValidationResult(
        is_valid=len(errors) == 0,
        num_episodes=num_episodes,
        num_transitions=total_transitions,
        errors=errors,
        warnings=warnings,
        info=dataset_info,
    )
