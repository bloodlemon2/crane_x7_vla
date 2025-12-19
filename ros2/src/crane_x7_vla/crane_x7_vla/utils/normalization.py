#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Action normalization and denormalization utilities."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Literal, Union

import numpy as np


def load_norm_stats(
    model_path: Path,
    is_hf_hub: bool = False,
    model_path_str: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[Dict[str, Any]]:
    """Load normalization statistics from checkpoint or dataset.

    Args:
        model_path: Path to model directory
        is_hf_hub: Whether model is from HuggingFace Hub
        model_path_str: String representation of model path (for HF Hub)
        logger: Optional logger instance

    Returns:
        Dictionary of normalization statistics, or None if not found
    """
    log = logger or logging.getLogger(__name__)

    if is_hf_hub and model_path_str:
        try:
            from huggingface_hub import hf_hub_download
            stats_file = hf_hub_download(
                model_path_str,
                filename="dataset_statistics.json"
            )
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            log.info(f'Loaded normalization stats from HF Hub: {list(stats.keys())}')
            return stats
        except Exception as e:
            log.debug(f'Could not download dataset_statistics.json: {e}')
            return None

    # Try local paths
    stats_paths = [
        model_path / "dataset_statistics.json",
        model_path.parent / "dataset_statistics.json",
    ]

    for stats_path in stats_paths:
        if stats_path.exists():
            try:
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                log.info(f'Loaded normalization stats from {stats_path}')
                return stats
            except Exception as e:
                log.warning(f'Failed to load stats from {stats_path}: {e}')

    log.warning('No dataset_statistics.json found')
    return None


def load_norm_stats_from_config(
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Optional[Dict[str, Any]]:
    """Load normalization statistics from config's data_root_dir.

    Args:
        config: Model configuration dictionary
        logger: Optional logger instance

    Returns:
        Dictionary of normalization statistics, or None if not found
    """
    log = logger or logging.getLogger(__name__)

    data_root = config.get('data_root_dir')
    if not data_root:
        return None

    data_stats_path = Path(data_root) / "dataset_statistics.json"
    if data_stats_path.exists():
        try:
            with open(data_stats_path, 'r') as f:
                stats = json.load(f)
            log.info(f'Loaded normalization stats from {data_stats_path}')
            return stats
        except Exception as e:
            log.warning(f'Failed to load stats from data root: {e}')

    return None


def denormalize_action(
    action: np.ndarray,
    stats: Dict[str, Any],
    mode: Literal['quantile', 'normal'] = 'quantile',
    stats_key: str = 'crane_x7',
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """Denormalize action using stored statistics.

    Args:
        action: Normalized action array
        stats: Statistics dictionary with nested structure
        mode: Normalization mode ('quantile' or 'normal')
        stats_key: Key for robot-specific stats (e.g., 'crane_x7')
        logger: Optional logger instance

    Returns:
        Denormalized action array
    """
    log = logger or logging.getLogger(__name__)

    if not stats:
        return action

    # Find appropriate stats key
    if stats_key not in stats:
        available_keys = list(stats.keys())
        if not available_keys:
            return action
        # Fallback to first available key
        stats_key = available_keys[0]
        log.debug(f'Using fallback stats key: {stats_key}')

    key_stats = stats[stats_key]

    try:
        if mode == 'quantile':
            return _denormalize_quantile(action, key_stats)
        elif mode == 'normal':
            return _denormalize_normal(action, key_stats)
        else:
            log.warning(f'Unknown normalization mode: {mode}')
            return action
    except Exception as e:
        if log:
            log.warning(f'Denormalization failed: {e}')
        return action


def _denormalize_quantile(action: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    """Denormalize using quantile (q01, q99) statistics.

    Pi0/Pi0.5 normalizes actions to [-1, 1] range during training:
        normalized = 2 * (action - q01) / (q99 - q01) - 1

    This function reverses that transformation:
        denormalized = (normalized + 1) / 2 * (q99 - q01) + q01

    Args:
        action: Normalized action array in [-1, 1] range
        stats: Statistics dictionary with q01, q99 fields

    Returns:
        Denormalized action array in original range
    """
    # Handle nested 'action' key
    if 'action' in stats:
        stats = stats['action']

    q01 = np.array(stats.get('q01', []))
    q99 = np.array(stats.get('q99', []))

    if len(q01) < len(action) or len(q99) < len(action):
        return action

    q01 = q01[:len(action)]
    q99 = q99[:len(action)]

    # Denormalize from [-1, 1] to original range
    # Formula: denorm = (action + 1) / 2 * (q99 - q01) + q01
    denorm = (action + 1) / 2 * (q99 - q01) + q01

    return denorm


def _denormalize_normal(action: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    """Denormalize using normal (mean, std) statistics.

    Args:
        action: Normalized action array (zero mean, unit variance)
        stats: Statistics dictionary with mean, std fields

    Returns:
        Denormalized action array
    """
    # Handle nested 'action' key
    if 'action' in stats:
        stats = stats['action']

    mean = np.array(stats.get('mean', []))
    std = np.array(stats.get('std', []))

    if len(mean) < len(action) or len(std) < len(action):
        return action

    mean = mean[:len(action)]
    std = std[:len(action)]

    # Denormalize: action * std + mean
    denorm = action * std + mean

    return denorm


def denormalize_openvla_action(
    normalized_actions: np.ndarray,
    action_norm_stats: Dict[str, Any],
) -> np.ndarray:
    """Denormalize OpenVLA action using q01/q99 statistics.

    OpenVLA uses a specific denormalization formula:
        action = 0.5 * (normalized + 1) * (q99 - q01) + q01

    Args:
        normalized_actions: Normalized actions in [-1, 1]
        action_norm_stats: Stats dict with q01, q99, and optional mask

    Returns:
        Denormalized action array
    """
    mask = action_norm_stats.get("mask", np.ones(len(normalized_actions), dtype=bool))
    action_high = np.array(action_norm_stats["q99"])
    action_low = np.array(action_norm_stats["q01"])

    action = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )

    return np.asarray(action).flatten()
