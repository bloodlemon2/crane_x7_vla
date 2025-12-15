# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Utilities for automatic CLI argument generation from dataclass configs.

This module provides functions to automatically generate argparse arguments
from dataclass configurations and apply parsed arguments back to configs.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from typing import Any, get_args, get_origin


logger = logging.getLogger(__name__)


def _python_type_to_argparse(field_type: type) -> dict[str, Any]:
    """
    Convert Python type to argparse argument kwargs.

    Args:
        field_type: Python type annotation

    Returns:
        Dictionary of argparse argument kwargs
    """
    # Handle Optional types (Union[X, None])
    origin = get_origin(field_type)
    if origin is type(None):
        return {"type": str}

    # Handle Union types (including Optional)
    if origin is type(None) or str(origin) == "typing.Union":
        args = get_args(field_type)
        # Filter out NoneType
        non_none_types = [t for t in args if t is not type(None)]
        if non_none_types:
            return _python_type_to_argparse(non_none_types[0])
        return {"type": str}

    # Handle Literal types
    if str(origin) == "typing.Literal":
        choices = get_args(field_type)
        return {"type": str, "choices": list(choices)}

    # Handle list types
    if origin is list:
        inner_args = get_args(field_type)
        if inner_args:
            inner_kwargs = _python_type_to_argparse(inner_args[0])
            return {"type": inner_kwargs.get("type", str), "nargs": "+"}
        return {"type": str, "nargs": "+"}

    # Handle tuple types
    if origin is tuple:
        inner_args = get_args(field_type)
        if inner_args:
            inner_kwargs = _python_type_to_argparse(inner_args[0])
            return {"type": inner_kwargs.get("type", str), "nargs": len(inner_args)}
        return {"type": str, "nargs": "+"}

    # Basic types
    if field_type is bool:
        return {"action": "store_true"}
    elif field_type is int:
        return {"type": int}
    elif field_type is float:
        return {"type": float}
    elif field_type is str:
        return {"type": str}

    # Default to string
    return {"type": str}


def _field_name_to_arg_name(field_name: str, prefix: str = "") -> str:
    """
    Convert field name to CLI argument name.

    Args:
        field_name: Dataclass field name (e.g., 'learning_rate')
        prefix: Optional prefix (e.g., 'training')

    Returns:
        CLI argument name (e.g., '--training-learning-rate')
    """
    # Convert underscores to hyphens
    arg_name = field_name.replace("_", "-")
    if prefix:
        return f"--{prefix}-{arg_name}"
    return f"--{arg_name}"


def _arg_name_to_field_name(arg_name: str) -> tuple[str, str]:
    """
    Convert CLI argument name back to field name and prefix.

    Args:
        arg_name: CLI argument name (e.g., 'training_learning_rate')

    Returns:
        Tuple of (prefix, field_name)
    """
    parts = arg_name.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "", arg_name


def add_dataclass_arguments(
    parser: argparse.ArgumentParser,
    dataclass_type: type,
    prefix: str = "",
    exclude_fields: set[str] | None = None,
    include_fields: set[str] | None = None,
) -> None:
    """
    Add CLI arguments from a dataclass to an argparse parser.

    Args:
        parser: ArgumentParser to add arguments to
        dataclass_type: Dataclass type to generate arguments from
        prefix: Prefix for argument names (e.g., 'training' -> '--training-batch-size')
        exclude_fields: Set of field names to exclude
        include_fields: Set of field names to include (if None, include all)
    """
    if not dataclasses.is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} is not a dataclass")

    exclude_fields = exclude_fields or set()

    for field in dataclasses.fields(dataclass_type):
        # Skip excluded fields
        if field.name in exclude_fields:
            continue

        # Skip if include_fields specified and field not in it
        if include_fields is not None and field.name not in include_fields:
            continue

        # Skip nested dataclasses (handle separately)
        if dataclasses.is_dataclass(field.type):
            continue

        # Skip complex types that can't be easily parsed from CLI
        origin = get_origin(field.type)
        if origin is dict:
            continue

        # Generate argument name
        arg_name = _field_name_to_arg_name(field.name, prefix)

        # Get argparse kwargs from type
        kwargs = _python_type_to_argparse(field.type)

        # Add help text from field metadata or docstring
        if field.metadata and "help" in field.metadata:
            kwargs["help"] = field.metadata["help"]
        else:
            # Use field name as basic help
            kwargs["help"] = f"{field.name.replace('_', ' ').title()}"

        # Set default to None (so we can detect if user provided a value)
        if "action" not in kwargs:
            kwargs["default"] = None

        try:
            parser.add_argument(arg_name, **kwargs)
        except argparse.ArgumentError:
            # Argument already exists, skip
            logger.debug(f"Argument {arg_name} already exists, skipping")


def apply_args_to_config(args: argparse.Namespace, config: Any, prefix: str = "") -> None:
    """
    Apply parsed CLI arguments to a config object.

    Args:
        args: Parsed argparse namespace
        config: Config object to apply values to
        prefix: Prefix that was used for argument names
    """
    if not dataclasses.is_dataclass(config):
        return

    args_dict = vars(args)

    for field in dataclasses.fields(config):
        # Build the argument name as it appears in args
        arg_key = f"{prefix}_{field.name}" if prefix else field.name

        # Check if this argument was provided
        if arg_key in args_dict and args_dict[arg_key] is not None:
            value = args_dict[arg_key]

            # Handle tuple conversion for fields like image_size
            if get_origin(field.type) is tuple and isinstance(value, list):
                value = tuple(value)

            setattr(config, field.name, value)


def apply_sweep_config_to_config(sweep_config: dict[str, Any], config: Any) -> None:
    """
    Apply W&B sweep config parameters to a unified config object.

    Automatically maps sweep parameters to the appropriate config fields
    based on field names. Handles nested configs (training, overfitting, backend-specific).

    Args:
        sweep_config: Dictionary from wandb.config
        config: UnifiedVLAConfig or subclass instance
    """
    if not dataclasses.is_dataclass(config):
        return

    # Map of sweep parameter names to config paths
    # Format: sweep_param -> (config_attr, field_name) or just field_name for top-level
    training_fields = {
        "learning_rate",
        "batch_size",
        "grad_accumulation_steps",
        "gradient_accumulation_steps",
        "weight_decay",
        "warmup_ratio",
        "warmup_steps",
        "max_grad_norm",
        "max_steps",
        "num_epochs",
        "save_interval",
        "eval_interval",
        "log_interval",
        "mixed_precision",
        "gradient_checkpointing",
    }

    overfitting_fields = {
        "overfit_split_ratio",
        "overfit_check_interval",
        "overfit_check_steps",
    }

    # Apply training config parameters
    if hasattr(config, "training"):
        for param, value in sweep_config.items():
            if param in training_fields:
                # Handle alias
                field_name = "gradient_accumulation_steps" if param == "grad_accumulation_steps" else param
                if hasattr(config.training, field_name):
                    setattr(config.training, field_name, value)
                    logger.debug(f"Set training.{field_name} = {value}")

    # Apply overfitting config parameters
    if hasattr(config, "overfitting"):
        for param, value in sweep_config.items():
            if param in overfitting_fields and hasattr(config.overfitting, param):
                setattr(config.overfitting, param, value)
                logger.debug(f"Set overfitting.{param} = {value}")

    # Apply backend-specific parameters
    backend = getattr(config, "backend", None)

    if backend == "openvla" and hasattr(config, "openvla"):
        _apply_to_nested_config(sweep_config, config.openvla)
    elif backend == "openvla-oft" and hasattr(config, "openvla_oft"):
        _apply_to_nested_config(sweep_config, config.openvla_oft)
        # Handle nested configs within openvla_oft
        if hasattr(config.openvla_oft, "film"):
            _apply_to_nested_config(sweep_config, config.openvla_oft.film, prefix="film")
        if hasattr(config.openvla_oft, "proprio"):
            _apply_to_nested_config(sweep_config, config.openvla_oft.proprio, prefix="proprio")
        if hasattr(config.openvla_oft, "multi_image"):
            _apply_to_nested_config(sweep_config, config.openvla_oft.multi_image, prefix="multi_image")
        if hasattr(config.openvla_oft, "action_head"):
            _apply_to_nested_config(sweep_config, config.openvla_oft.action_head, prefix="action_head")
    elif backend == "minivla" and hasattr(config, "minivla"):
        _apply_to_nested_config(sweep_config, config.minivla)
    elif backend in ("pi0", "pi0.5") and hasattr(config, "pi0"):
        _apply_to_nested_config(sweep_config, config.pi0)


def _apply_to_nested_config(sweep_config: dict[str, Any], nested_config: Any, prefix: str = "") -> None:
    """
    Apply sweep config to a nested dataclass config.

    Args:
        sweep_config: Dictionary from wandb.config
        nested_config: Nested config dataclass instance
        prefix: Optional prefix for parameter names (e.g., 'film' for 'film_enabled')
    """
    if not dataclasses.is_dataclass(nested_config):
        return

    for field in dataclasses.fields(nested_config):
        # Skip nested dataclasses
        if dataclasses.is_dataclass(field.type):
            continue

        # Try different parameter name formats
        param_names = [
            field.name,  # Direct match
            f"{prefix}_{field.name}" if prefix else None,  # With prefix
        ]
        param_names = [p for p in param_names if p]

        for param_name in param_names:
            if param_name in sweep_config:
                value = sweep_config[param_name]
                setattr(nested_config, field.name, value)
                logger.debug(f"Set {type(nested_config).__name__}.{field.name} = {value}")
                break


def get_config_field_names(dataclass_type: type, prefix: str = "") -> list[str]:
    """
    Get all CLI argument names that would be generated for a dataclass.

    Args:
        dataclass_type: Dataclass type
        prefix: Prefix for argument names

    Returns:
        List of argument names (without leading --)
    """
    if not dataclasses.is_dataclass(dataclass_type):
        return []

    names = []
    for field in dataclasses.fields(dataclass_type):
        if dataclasses.is_dataclass(field.type):
            continue
        origin = get_origin(field.type)
        if origin is dict:
            continue

        if prefix:
            names.append(f"{prefix}_{field.name}")
        else:
            names.append(field.name)

    return names
