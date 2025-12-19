#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Backwards-compatible facade for VLA inference core.

This module re-exports the refactored components to maintain
backwards compatibility with existing code that imports from here.

Supports multiple VLA backends:
- OpenVLA: Autoregressive token-based action prediction
- Pi0/Pi0.5: Flow matching based action chunk prediction

Example usage:
    from crane_x7_vla.vla_inference_core import create_inference_core

    core = create_inference_core(
        model_path="/path/to/model",
        device="cuda",
    )
    core.load_model()
    action = core.predict_action(image, "pick up the object")

For new code, prefer importing directly from submodules:
    from crane_x7_vla.core import create_inference_core, BaseVLAInferenceCore
    from crane_x7_vla.backends import OpenVLAInferenceCore, Pi0InferenceCore
"""

# Re-export core abstractions
from crane_x7_vla.core.base import BaseVLAInferenceCore
from crane_x7_vla.core.factory import create_inference_core, detect_model_type

# Re-export backend implementations
from crane_x7_vla.backends.openvla import OpenVLAInferenceCore
from crane_x7_vla.backends.pi0 import Pi0InferenceCore

# Legacy alias for rosbridge client compatibility
VLAInferenceCore = OpenVLAInferenceCore

__all__ = [
    # Core abstractions
    'BaseVLAInferenceCore',
    'create_inference_core',
    'detect_model_type',
    # Backend implementations
    'OpenVLAInferenceCore',
    'Pi0InferenceCore',
    # Legacy alias
    'VLAInferenceCore',
]
