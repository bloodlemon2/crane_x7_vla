#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""CRANE-X7 VLA inference package.

This package provides VLA (Vision-Language-Action) model inference
for controlling the CRANE-X7 robotic arm.

Submodules:
    core: Core abstractions (BaseVLAInferenceCore, factory functions)
    backends: Backend implementations (OpenVLA, Pi0/Pi0.5)
    utils: Shared utilities (image processing, normalization, paths)

Example:
    from crane_x7_vla.vla_inference_core import create_inference_core

    core = create_inference_core(model_path="/path/to/model")
    core.load_model()
    action = core.predict_action(image, "pick up the object")
"""

__version__ = "0.1.0"
