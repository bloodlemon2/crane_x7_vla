# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""MiniVLA backend for CRANE-X7 VLA training.

MiniVLA features:
- Qwen 2.5 0.5B LLM backbone (~1B total params, 7x smaller than OpenVLA)
- VQ Action Chunking for efficient multi-step prediction
- Multi-image input support (history + wrist camera)
- ~12.5Hz inference (2.5x faster than OpenVLA)
"""

from crane_x7_vla.backends.minivla.action_tokenizer import (
    BinActionTokenizer,
    ResidualVQ,
    VectorQuantize,
    VQActionTokenizer,
)
from crane_x7_vla.backends.minivla.backend import (
    MiniVLABackend,
    MiniVLAFinetuneConfig,
    MiniVLAModel,
)
from crane_x7_vla.backends.minivla.config import (
    MiniVLAConfig,
    MiniVLASpecificConfig,
    MultiImageConfig,
    VQConfig,
)
from crane_x7_vla.backends.minivla.dataset import (
    MiniVLABatchTransform,
    MiniVLADataset,
    MiniVLADatasetConfig,
)


__all__ = [
    "BinActionTokenizer",
    "MiniVLABackend",
    "MiniVLABatchTransform",
    "MiniVLAConfig",
    "MiniVLADataset",
    "MiniVLADatasetConfig",
    "MiniVLAFinetuneConfig",
    "MiniVLAModel",
    "MiniVLASpecificConfig",
    "MultiImageConfig",
    "ResidualVQ",
    "VQActionTokenizer",
    "VQConfig",
    "VectorQuantize",
]
