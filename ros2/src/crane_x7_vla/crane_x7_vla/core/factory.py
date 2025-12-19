#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Factory functions for creating VLA inference cores."""

import json
import logging
from pathlib import Path
from typing import Optional

import torch

from crane_x7_vla.core.base import BaseVLAInferenceCore
from crane_x7_vla.core.types import ModelType


def detect_model_type(model_path: str, logger: Optional[logging.Logger] = None) -> ModelType:
    """Detect VLA model type from checkpoint.

    Args:
        model_path: Path to model directory or HuggingFace Hub ID
        logger: Optional logger instance

    Returns:
        Model type string: 'openvla', 'pi0', or 'pi0.5'
    """
    log = logger or logging.getLogger(__name__)

    # Check for HuggingFace Hub ID
    if '/' in model_path and not model_path.startswith('/'):
        parts = model_path.split('/')
        if len(parts) == 2 and all(p for p in parts):
            log.info(f'HuggingFace Hub model detected: {model_path}, assuming OpenVLA')
            return 'openvla'

    path = Path(model_path)
    if not path.exists():
        log.warning(f'Model path does not exist: {path}, defaulting to openvla')
        return 'openvla'

    # Check config.json for model_type
    config_path = path / "config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            model_type = config.get('model_type')
            if model_type in ('pi0', 'pi0.5'):
                log.info(f'Detected model type from config.json: {model_type}')
                return model_type
            # OpenVLA has different config format (model_type not present)
            if 'text_config' in config or 'vision_config' in config:
                log.info('Detected OpenVLA model format')
                return 'openvla'
        except json.JSONDecodeError:
            log.warning(f'Invalid JSON in {config_path}')

    # Check checkpoint.pt for config
    checkpoint_path = path / "checkpoint.pt"
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
                model_type = checkpoint['config'].get('model_type')
                if model_type in ('pi0', 'pi0.5'):
                    log.info(f'Detected model type from checkpoint.pt: {model_type}')
                    return model_type
        except Exception as e:
            log.warning(f'Failed to load checkpoint.pt: {e}')

    # Check for OpenVLA-specific files
    if (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists():
        log.info('Detected OpenVLA format (HuggingFace model files)')
        return 'openvla'

    log.warning(f'Could not determine model type, defaulting to openvla')
    return 'openvla'


def create_inference_core(
    model_path: str,
    device: str = 'cuda',
    unnorm_key: str = 'crane_x7',
    model_base_name: str = 'openvla',
    use_flash_attention: bool = False,
    logger: Optional[logging.Logger] = None,
) -> BaseVLAInferenceCore:
    """Factory function to create appropriate inference core based on model type.

    Args:
        model_path: Path to model (local path or HuggingFace Hub ID)
        device: Device for inference ('cuda' or 'cpu')
        unnorm_key: Key for action normalization statistics
        model_base_name: Base model name for prompt formatting (OpenVLA only)
        use_flash_attention: Whether to use Flash Attention 2 (OpenVLA only)
        logger: Optional logger instance

    Returns:
        Appropriate inference core instance (OpenVLAInferenceCore or Pi0InferenceCore)
    """
    log = logger or logging.getLogger(__name__)
    model_type = detect_model_type(model_path, log)

    if model_type in ('pi0', 'pi0.5'):
        log.info(f'Creating Pi0InferenceCore for {model_type} model')
        # Lazy import to avoid circular dependencies
        from crane_x7_vla.backends.pi0 import Pi0InferenceCore
        return Pi0InferenceCore(
            model_path=model_path,
            device=device,
            logger=logger,
        )
    else:
        log.info('Creating OpenVLAInferenceCore')
        # Lazy import to avoid circular dependencies
        from crane_x7_vla.backends.openvla import OpenVLAInferenceCore
        return OpenVLAInferenceCore(
            model_path=model_path,
            device=device,
            unnorm_key=unnorm_key,
            model_base_name=model_base_name,
            use_flash_attention=use_flash_attention,
            logger=logger,
        )
