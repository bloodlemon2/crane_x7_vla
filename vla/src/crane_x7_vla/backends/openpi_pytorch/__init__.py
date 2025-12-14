# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""OpenPI PyTorch backend for CRANE-X7 VLA training.

OpenPI PyTorch features:
- HuggingFace Pi0 model with flow matching
- Action chunking for multi-step prediction
- PyTorch-native training (no JAX dependency)
"""

from crane_x7_vla.backends.openpi_pytorch.backend import (
    CraneX7Pi0FinetuneConfig,
    CraneX7Pi0Trainer,
    FlowMatchingModule,
    OpenPIPytorchBackend,
)
from crane_x7_vla.backends.openpi_pytorch.config import (
    OpenPIPytorchConfig,
    OpenPIPytorchSpecificConfig,
)
from crane_x7_vla.backends.openpi_pytorch.dataset import (
    CraneX7ActionChunkDataset,
    collate_action_chunk_batch,
)


__all__ = [
    "CraneX7ActionChunkDataset",
    "CraneX7Pi0FinetuneConfig",
    "CraneX7Pi0Trainer",
    "FlowMatchingModule",
    "OpenPIPytorchBackend",
    "OpenPIPytorchConfig",
    "OpenPIPytorchSpecificConfig",
    "collate_action_chunk_batch",
]
