# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Training scripts and utilities."""

__all__ = ["VLATrainer"]


def __getattr__(name: str):
    """Lazy import for VLATrainer to avoid loading all backends."""
    if name == "VLATrainer":
        from crane_x7_vla.training.trainer import VLATrainer

        return VLATrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
