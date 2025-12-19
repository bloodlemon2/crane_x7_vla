#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Type definitions for VLA inference core."""

from typing import Protocol, TypedDict, Literal, Callable, Optional, List

import numpy as np


class LoggerProtocol(Protocol):
    """Protocol for logger-like objects (ROS 2 logger adapter, Python logger)."""

    def info(self, msg: str) -> None:
        """Log info message."""
        ...

    def warning(self, msg: str) -> None:
        """Log warning message."""
        ...

    def error(self, msg: str) -> None:
        """Log error message."""
        ...

    def debug(self, msg: str) -> None:
        """Log debug message."""
        ...


class NormStats(TypedDict, total=False):
    """Type for normalization statistics."""

    q01: List[float]
    q99: List[float]
    mean: List[float]
    std: List[float]
    mask: List[bool]


class ActionStats(TypedDict):
    """Type for action statistics."""

    action: NormStats


# Type aliases
ModelType = Literal['openvla', 'pi0', 'pi0.5']
LogCallback = Callable[[str], None]
ImageArray = np.ndarray  # Shape: (H, W, 3), dtype: uint8
ActionArray = np.ndarray  # Shape: (action_dim,), dtype: float32
