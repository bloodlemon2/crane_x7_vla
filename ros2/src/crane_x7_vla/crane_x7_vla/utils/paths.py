#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Path resolution utilities with environment variable support."""

import os
import sys
from pathlib import Path
from typing import Optional

# Environment variable names
VLA_PATH_ENV = "VLA_PATH"
VLA_MODEL_PATH_ENV = "VLA_MODEL_PATH"


def get_vla_path() -> Path:
    """Get VLA package path from environment or default location.

    Priority:
        1. VLA_PATH environment variable
        2. /workspace/vla (Docker container)
        3. Relative path from this file (development)

    Returns:
        Path to VLA package directory
    """
    # Check environment variable first
    if VLA_PATH_ENV in os.environ:
        return Path(os.environ[VLA_PATH_ENV])

    # Docker container path
    workspace_vla = Path("/workspace/vla")
    if workspace_vla.exists():
        return workspace_vla

    # Development fallback: relative to this file
    # ros2/src/crane_x7_vla/crane_x7_vla/utils/paths.py -> vla/
    dev_path = Path(__file__).parent.parent.parent.parent.parent.parent / "vla"
    return dev_path


def resolve_model_path(path: str) -> Optional[Path]:
    """Resolve model path, handling relative paths and environment variables.

    Args:
        path: Model path string (can be relative, absolute, or HF Hub ID)

    Returns:
        Resolved Path object, or None if path is a HuggingFace Hub ID
    """
    if not path:
        return None

    # Check if it's a HuggingFace Hub ID
    if is_huggingface_hub_id(path):
        return None

    resolved = Path(path)

    # If relative, try to resolve against VLA outputs directory
    if not resolved.is_absolute():
        vla_path = get_vla_path()
        outputs_path = vla_path / "outputs" / path
        if outputs_path.exists():
            return outputs_path

    return resolved


def is_huggingface_hub_id(path: str) -> bool:
    """Check if path looks like a HuggingFace Hub model ID.

    Args:
        path: Path string to check

    Returns:
        True if path looks like a HF Hub ID (e.g., 'username/model-name')
    """
    if not path:
        return False
    if path.startswith('/') or path.startswith('./') or path.startswith('..'):
        return False
    if '\\' in path:  # Windows path
        return False
    parts = path.split('/')
    return len(parts) == 2 and all(p for p in parts)


def setup_vla_path() -> None:
    """Add VLA directory to Python path if not already present."""
    vla_path = get_vla_path()
    vla_path_str = str(vla_path)
    if vla_path_str not in sys.path:
        sys.path.insert(0, vla_path_str)
