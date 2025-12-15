# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""OpenVLA-OFT (Optimized Fine-Tuning) backend for CRANE-X7 VLA training."""

# NOTE: backend.py imports are deferred to avoid circular imports with common.hf
# Use: from crane_x7_vla.backends.openvla_oft.backend import OpenVLAOFTBackend
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
from crane_x7_vla.backends.openvla_oft.constants import (
    ACTION_DIM,
    ACTION_TOKEN_BEGIN_IDX,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    STOP_INDEX,
    NormalizationType,
)
from crane_x7_vla.backends.openvla_oft.dataset import (
    CraneX7OFTDataset,
    OpenVLAOFTBatchTransform,
    PaddedCollatorForOFT,
)
from crane_x7_vla.backends.openvla_oft.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask,
)


def __getattr__(name: str):
    """Lazy import for OpenVLAOFTBackend to avoid circular imports."""
    if name == "OpenVLAOFTBackend":
        from crane_x7_vla.backends.openvla_oft.backend import OpenVLAOFTBackend

        return OpenVLAOFTBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ACTION_DIM",
    "ACTION_TOKEN_BEGIN_IDX",
    "IGNORE_INDEX",
    "NUM_ACTIONS_CHUNK",
    "PROPRIO_DIM",
    "STOP_INDEX",
    "ActionHeadConfig",
    "CraneX7OFTDataset",
    "FiLMConfig",
    "FiLMedVisionBackbone",
    "L1RegressionActionHead",
    "MLPResNet",
    "MLPResNetBlock",
    "MultiImageConfig",
    "NormalizationType",
    "OpenVLAOFTBackend",
    "OpenVLAOFTBatchTransform",
    "OpenVLAOFTConfig",
    "OpenVLAOFTSpecificConfig",
    "PaddedCollatorForOFT",
    "ProprioConfig",
    "ProprioProjector",
    "compute_actions_l1_loss",
    "compute_token_accuracy",
    "get_current_action_mask",
    "get_next_actions_mask",
]
