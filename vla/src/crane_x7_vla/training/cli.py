#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Command-line interface for CRANE-X7 VLA training.

Provides a unified CLI for training different VLA backends (OpenVLA, OpenPI).
Supports both simplified mode (--config file) and full mode with backend-specific arguments.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Only import core config at module level - other configs imported lazily
from crane_x7_vla.core.config.base import CameraConfig, DataConfig, TrainingConfig, UnifiedVLAConfig


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def _get_backend_configs(backend: str) -> tuple[Any, ...]:
    """
    Lazy import backend-specific configs to avoid loading unnecessary dependencies.

    Returns tuple of (BackendConfig, BackendSpecificConfig, extra_configs...)
    """
    if backend == "openvla":
        from crane_x7_vla.backends.openvla.config import OpenVLAConfig, OpenVLASpecificConfig

        return (OpenVLAConfig, OpenVLASpecificConfig)
    elif backend == "openpi-pytorch":
        from crane_x7_vla.backends.openpi_pytorch.config import OpenPIPytorchConfig, OpenPIPytorchSpecificConfig

        return (OpenPIPytorchConfig, OpenPIPytorchSpecificConfig)
    elif backend == "minivla":
        from crane_x7_vla.backends.minivla.config import MiniVLAConfig, MiniVLASpecificConfig

        return (MiniVLAConfig, MiniVLASpecificConfig)
    elif backend == "openvla-oft":
        from crane_x7_vla.backends.openvla_oft.config import OpenVLAOFTConfig, OpenVLAOFTSpecificConfig

        return (OpenVLAOFTConfig, OpenVLAOFTSpecificConfig)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def create_default_config(backend: str, data_root: Path, output_dir: Path, experiment_name: str) -> UnifiedVLAConfig:
    """
    Create default configuration for the specified backend.

    Args:
        backend: Backend type ('openvla', 'openpi', etc.)
        data_root: Path to training data
        output_dir: Path to output directory
        experiment_name: Experiment name

    Returns:
        Backend-specific config instance
    """
    # Default camera configuration
    cameras = [CameraConfig(name="primary", topic="/camera/color/image_raw", width=640, height=480)]

    # Data configuration
    data_config = DataConfig(data_root=data_root, cameras=cameras)

    # Training configuration
    training_config = TrainingConfig(
        batch_size=16,
        num_epochs=100,
        learning_rate=5e-4,
    )

    # Lazy import backend configs
    configs = _get_backend_configs(backend)
    BackendConfig, BackendSpecificConfig = configs[0], configs[1]

    # Create backend-specific config
    if backend == "openvla":
        return BackendConfig(
            backend=backend,
            data=data_config,
            training=training_config,
            output_dir=output_dir,
            experiment_name=experiment_name,
            openvla=BackendSpecificConfig(),
        )
    elif backend == "openpi-pytorch":
        return BackendConfig(
            backend=backend,
            data=data_config,
            training=training_config,
            output_dir=output_dir,
            experiment_name=experiment_name,
            openpi_pytorch=BackendSpecificConfig(),
        )
    elif backend == "minivla":
        return BackendConfig(
            backend=backend,
            data=data_config,
            training=training_config,
            output_dir=output_dir,
            experiment_name=experiment_name,
            minivla=BackendSpecificConfig(),
        )
    elif backend == "openvla-oft":
        return BackendConfig(
            backend=backend,
            data=data_config,
            training=training_config,
            output_dir=output_dir,
            experiment_name=experiment_name,
            openvla_oft=BackendSpecificConfig(),
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def train_command(args: argparse.Namespace) -> None:
    """Execute training command."""
    backend = args.backend

    # Load or create configuration
    if args.config:
        # Load from YAML and determine backend
        config = UnifiedVLAConfig.from_yaml(args.config)
        # Re-create as proper backend-specific config
        configs = _get_backend_configs(config.backend)
        BackendConfig = configs[0]
        config = BackendConfig.from_yaml(args.config)

        # Warn if config backend doesn't match CLI
        if config.backend != backend:
            logging.warning(
                f"Config file specifies backend '{config.backend}' but CLI subcommand is '{backend}'. "
                f"Using config file backend."
            )
            backend = config.backend
    else:
        if not args.data_root:
            raise ValueError("--data-root is required when not using --config")

        config = create_default_config(
            backend=backend,
            data_root=Path(args.data_root),
            output_dir=Path(args.output_dir),
            experiment_name=args.experiment_name,
        )

    # Apply CLI overrides
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.num_epochs is not None:
        config.training.num_epochs = args.num_epochs
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.grad_accumulation_steps is not None:
        config.training.grad_accumulation_steps = args.grad_accumulation_steps

    # Save configuration
    config_save_path = Path(config.output_dir) / "config.yaml"
    config_save_path.parent.mkdir(parents=True, exist_ok=True)
    config.to_yaml(config_save_path)
    logging.info(f"Configuration saved to {config_save_path}")

    # Lazy import trainer to avoid loading all backends
    from crane_x7_vla.training.trainer import VLATrainer

    # Create trainer and start training
    trainer = VLATrainer(config)
    results = trainer.train()

    logging.info(f"Training completed: {results}")


def evaluate_command(args: argparse.Namespace) -> None:
    """Execute evaluation command."""
    if not args.config:
        raise ValueError("--config is required for evaluation")

    config = UnifiedVLAConfig.from_yaml(args.config)
    configs = _get_backend_configs(config.backend)
    BackendConfig = configs[0]
    config = BackendConfig.from_yaml(args.config)

    from crane_x7_vla.training.trainer import VLATrainer

    trainer = VLATrainer(config)
    metrics = trainer.evaluate(checkpoint_path=args.checkpoint, test_data_path=args.test_data)

    logging.info(f"Evaluation metrics: {metrics}")


def config_command(args: argparse.Namespace) -> None:
    """Generate default configuration file."""
    config = create_default_config(
        backend=args.backend,
        data_root=Path(args.data_root) if args.data_root else Path("./data"),
        output_dir=Path(args.output_dir) if args.output_dir else Path("./outputs"),
        experiment_name=args.experiment_name,
    )

    output_path = Path(args.output)
    config.to_yaml(output_path)

    logging.info(f"Default configuration saved to {output_path}")
    logging.info(f"Backend: {args.backend}")
    logging.info("Edit this file to customize your training configuration.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CRANE-X7 VLA Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with OpenVLA using default settings
  python -m crane_x7_vla.training.cli train openvla --data-root ./data

  # Train with OpenPI
  python -m crane_x7_vla.training.cli train openpi --data-root ./data

  # Train with configuration file
  python -m crane_x7_vla.training.cli train openvla --config my_config.yaml

  # Train with overrides
  python -m crane_x7_vla.training.cli train openpi --data-root ./data --batch-size 8 --max-steps 1000

  # Generate default configuration
  python -m crane_x7_vla.training.cli config --backend openvla --output openvla_config.yaml

  # Evaluate trained model
  python -m crane_x7_vla.training.cli evaluate --config my_config.yaml --checkpoint ./outputs/checkpoint
        """,
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # =====================
    # Train command
    # =====================
    train_parser = subparsers.add_parser(
        "train",
        help="Train a VLA model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    train_subparsers = train_parser.add_subparsers(dest="backend", help="VLA backend to use")

    # Define available backends
    backends = ["openvla", "openpi-pytorch", "minivla", "openvla-oft"]

    for backend in backends:
        backend_parser = train_subparsers.add_parser(
            backend,
            help=f"Train with {backend} backend",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        # Common arguments
        backend_parser.add_argument("--config", type=str, help="Path to configuration file (YAML)")
        backend_parser.add_argument("--data-root", type=str, help="Path to training data directory")
        backend_parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
        backend_parser.add_argument("--experiment-name", type=str, default="crane_x7_vla", help="Experiment name")

        # Training overrides
        backend_parser.add_argument("--batch-size", type=int, help="Batch size")
        backend_parser.add_argument("--learning-rate", type=float, help="Learning rate")
        backend_parser.add_argument("--num-epochs", type=int, help="Number of epochs")
        backend_parser.add_argument("--max-steps", type=int, help="Maximum training steps")
        backend_parser.add_argument("--grad-accumulation-steps", type=int, help="Gradient accumulation steps")

    # =====================
    # Evaluate command
    # =====================
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--config", type=str, required=True, help="Path to configuration file (YAML)")
    eval_parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    eval_parser.add_argument("--test-data", type=str, help="Path to test dataset")

    # =====================
    # Config command
    # =====================
    config_parser = subparsers.add_parser("config", help="Generate default configuration file")
    config_parser.add_argument(
        "--backend",
        type=str,
        choices=backends,
        required=True,
        help="VLA backend",
    )
    config_parser.add_argument("--output", type=str, default="config.yaml", help="Output configuration file path")
    config_parser.add_argument("--data-root", type=str, help="Path to training data directory")
    config_parser.add_argument("--output-dir", type=str, help="Output directory for checkpoints and logs")
    config_parser.add_argument("--experiment-name", type=str, default="crane_x7_vla", help="Experiment name")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Execute command
    if args.command == "train":
        if args.backend is None:
            train_parser.print_help()
            sys.exit(1)
        train_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "config":
        config_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
