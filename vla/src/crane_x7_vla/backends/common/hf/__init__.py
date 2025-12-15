# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""
HuggingFace互換モジュール.

openvla_oft/hf/からの共通化。OpenVLAとOpenVLA-OFTバックエンドで共有。
"""

from crane_x7_vla.backends.common.hf.configuration_prismatic import OpenVLAConfig, PrismaticConfig
from crane_x7_vla.backends.common.hf.modeling_prismatic import (
    OpenVLAForActionPrediction,
    PrismaticForConditionalGeneration,
    PrismaticPreTrainedModel,
)
from crane_x7_vla.backends.common.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


__all__ = [
    "OpenVLAConfig",
    "OpenVLAForActionPrediction",
    "PrismaticConfig",
    "PrismaticForConditionalGeneration",
    "PrismaticImageProcessor",
    "PrismaticPreTrainedModel",
    "PrismaticProcessor",
]
