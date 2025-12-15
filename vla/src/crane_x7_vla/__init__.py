# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
CRANE-X7 VLA Training Framework

A unified framework for training Vision-Language-Action models on CRANE-X7 robot data.
Supports both OpenVLA and OpenPI backends with automatic data conversion.

Copyright (c) 2025 nop
Licensed under the MIT License
"""

from crane_x7_vla.core.config.base import UnifiedVLAConfig


__version__ = "0.1.0"

__all__ = [
    "UnifiedVLAConfig",
]


def __getattr__(name: str):
    """Lazy import for VLATrainer to avoid loading all backends."""
    if name == "VLATrainer":
        from crane_x7_vla.training.trainer import VLATrainer

        return VLATrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
