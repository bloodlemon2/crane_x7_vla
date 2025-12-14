# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Data loading and conversion utilities."""

# OpenVLA dataset (requires prismatic - optional)
try:
    from crane_x7_vla.backends.openvla.dataset import (
        CraneX7BatchTransform,
        CraneX7Dataset,
        CraneX7DatasetConfig,
    )

    _OPENVLA_DATA_AVAILABLE = True
except ImportError:
    _OPENVLA_DATA_AVAILABLE = False

# Adapters (optional)
try:
    from crane_x7_vla.core.data.adapters import CraneX7DataAdapter

    _ADAPTERS_AVAILABLE = True
except ImportError:
    _ADAPTERS_AVAILABLE = False

# LeRobot converters (optional)
try:
    from crane_x7_vla.core.data.converters import (
        LeRobotDataset,
        TFRecordToLeRobotConverter,
    )

    _CONVERTERS_AVAILABLE = True
except ImportError:
    _CONVERTERS_AVAILABLE = False

# OpenPI-specific data config (optional import)
try:
    from crane_x7_vla.backends.openpi.data_config import (
        CraneX7DataConfigFactory,
        CraneX7Inputs,
        CraneX7LeRobotDataConfig,
        CraneX7Outputs,
    )

    _OPENPI_DATA_AVAILABLE = True
except ImportError:
    _OPENPI_DATA_AVAILABLE = False

__all__ = []

if _OPENVLA_DATA_AVAILABLE:
    __all__.extend(
        [
            "CraneX7BatchTransform",
            "CraneX7Dataset",
            "CraneX7DatasetConfig",
        ]
    )

if _ADAPTERS_AVAILABLE:
    __all__.append("CraneX7DataAdapter")

if _CONVERTERS_AVAILABLE:
    __all__.extend(
        [
            "LeRobotDataset",
            "TFRecordToLeRobotConverter",
        ]
    )

if _OPENPI_DATA_AVAILABLE:
    __all__.extend(
        [
            "CraneX7DataConfigFactory",
            "CraneX7Inputs",
            "CraneX7LeRobotDataConfig",
            "CraneX7Outputs",
        ]
    )
