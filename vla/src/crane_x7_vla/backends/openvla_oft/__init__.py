# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""OpenVLA-OFT (Optimized Fine-Tuning) backend for CRANE-X7 VLA training."""

from crane_x7_vla.backends.openvla_oft.backend import OpenVLAOFTBackend
from crane_x7_vla.backends.openvla_oft.components import (
    FiLMedVisionBackbone,
    L1RegressionActionHead,
    MLPResNet,
    MLPResNetBlock,
    ProprioProjector,
)
from crane_x7_vla.backends.openvla_oft.config import (
    ActionHeadConfig,
    FiLMConfig,
    MultiImageConfig,
    OpenVLAOFTConfig,
    OpenVLAOFTSpecificConfig,
    ProprioConfig,
)
from crane_x7_vla.backends.openvla_oft.dataset import (
    CraneX7OFTDataset,
    OpenVLAOFTBatchTransform,
    PaddedCollatorForOFT,
)


__all__ = [
    "ActionHeadConfig",
    "CraneX7OFTDataset",
    "FiLMConfig",
    "FiLMedVisionBackbone",
    "L1RegressionActionHead",
    "MLPResNet",
    "MLPResNetBlock",
    "MultiImageConfig",
    "OpenVLAOFTBackend",
    "OpenVLAOFTBatchTransform",
    "OpenVLAOFTConfig",
    "OpenVLAOFTSpecificConfig",
    "PaddedCollatorForOFT",
    "ProprioConfig",
    "ProprioProjector",
]
