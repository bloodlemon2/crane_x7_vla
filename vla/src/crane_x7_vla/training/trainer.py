# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Unified VLA trainer.

Provides a single interface for training different VLA backends.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from crane_x7_vla.backends import get_backend
from crane_x7_vla.core.config.base import UnifiedVLAConfig
from crane_x7_vla.core.utils.logging import get_logger


if TYPE_CHECKING:
    from crane_x7_vla.backends.openpi.config import OpenPIConfig
    from crane_x7_vla.backends.openvla.config import OpenVLAConfig
    from crane_x7_vla.backends.openvla_oft.config import OpenVLAOFTConfig
    from crane_x7_vla.core.base import VLABackend


logger = get_logger(__name__)


class VLATrainer:
    """
    Unified trainer for VLA models.

    Automatically selects and configures the appropriate backend based on configuration.
    """

    def __init__(self, config: UnifiedVLAConfig | OpenVLAConfig | OpenVLAOFTConfig | OpenPIConfig):
        """
        Initialize VLA trainer.

        Args:
            config: Unified VLA configuration or backend-specific configuration
        """
        self.config = config
        self.backend: VLABackend | None = None

        # Create backend
        self._create_backend()

    def _create_backend(self) -> None:
        """Create the appropriate backend based on configuration."""
        backend_type = self.config.backend

        logger.info(f"Creating {backend_type} backend...")

        # Get backend class using lazy loading
        BackendClass = get_backend(backend_type)

        # Convert config to backend-specific config if needed
        if backend_type == "openvla":
            if hasattr(self.config, "openvla") and self.config.openvla is not None:
                backend_config = self.config
            else:
                backend_config = self._convert_to_openvla_config(self.config)

        elif backend_type == "openpi":
            if hasattr(self.config, "openpi") and self.config.openpi is not None:
                backend_config = self.config
            else:
                backend_config = self._convert_to_openpi_config(self.config)

        elif backend_type == "openvla-oft":
            if hasattr(self.config, "openvla_oft") and self.config.openvla_oft is not None:
                backend_config = self.config
            else:
                backend_config = self._convert_to_openvla_oft_config(self.config)

        elif backend_type == "minivla":
            # MiniVLA uses its config directly through get_backend
            backend_config = self.config

        elif backend_type in ("pi0", "pi0.5"):
            if hasattr(self.config, "pi0") and self.config.pi0 is not None:
                backend_config = self.config
            else:
                backend_config = self._convert_to_pi0_config(self.config)

        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

        self.backend = BackendClass(backend_config)
        logger.info(f"Backend created: {self.backend}")

    def _convert_to_openvla_config(self, config: UnifiedVLAConfig) -> OpenVLAConfig:
        """Convert UnifiedVLAConfig to OpenVLAConfig."""
        from crane_x7_vla.backends.openvla.config import OpenVLAConfig, OpenVLASpecificConfig

        # Create OpenVLA-specific config from backend_config if available
        openvla_specific = OpenVLASpecificConfig()
        if config.backend_config:
            for key, value in config.backend_config.items():
                if hasattr(openvla_specific, key):
                    setattr(openvla_specific, key, value)

        # Create OpenVLAConfig
        openvla_config = OpenVLAConfig(
            backend="openvla",
            data=config.data,
            training=config.training,
            output_dir=config.output_dir,
            experiment_name=config.experiment_name,
            seed=config.seed,
            resume_from_checkpoint=config.resume_from_checkpoint,
            openvla=openvla_specific,
        )

        return openvla_config

    def _convert_to_openpi_config(self, config: UnifiedVLAConfig) -> OpenPIConfig:
        """Convert UnifiedVLAConfig to OpenPIConfig."""
        from crane_x7_vla.backends.openpi.config import OpenPIConfig, OpenPISpecificConfig

        # Create OpenPI-specific config from backend_config if available
        openpi_specific = OpenPISpecificConfig()
        if config.backend_config:
            for key, value in config.backend_config.items():
                if hasattr(openpi_specific, key):
                    setattr(openpi_specific, key, value)

        # Create OpenPIConfig
        openpi_config = OpenPIConfig(
            backend="openpi",
            data=config.data,
            training=config.training,
            output_dir=config.output_dir,
            experiment_name=config.experiment_name,
            seed=config.seed,
            resume_from_checkpoint=config.resume_from_checkpoint,
            openpi=openpi_specific,
        )

        return openpi_config

    def _convert_to_openvla_oft_config(self, config: UnifiedVLAConfig) -> OpenVLAOFTConfig:
        """Convert UnifiedVLAConfig to OpenVLAOFTConfig."""
        from crane_x7_vla.backends.openvla_oft.config import OpenVLAOFTConfig, OpenVLAOFTSpecificConfig

        # Create OpenVLA-OFT specific config from backend_config if available
        openvla_oft_specific = OpenVLAOFTSpecificConfig()
        if config.backend_config:
            for key, value in config.backend_config.items():
                if hasattr(openvla_oft_specific, key):
                    setattr(openvla_oft_specific, key, value)
                # Handle nested configs
                elif key == "film_enabled" and hasattr(openvla_oft_specific, "film"):
                    openvla_oft_specific.film.enabled = value
                elif key == "proprio_enabled" and hasattr(openvla_oft_specific, "proprio"):
                    openvla_oft_specific.proprio.enabled = value
                elif key == "multi_image_enabled" and hasattr(openvla_oft_specific, "multi_image"):
                    openvla_oft_specific.multi_image.enabled = value
                elif key == "num_images" and hasattr(openvla_oft_specific, "multi_image"):
                    openvla_oft_specific.multi_image.num_images = value

        # Create OpenVLAOFTConfig
        openvla_oft_config = OpenVLAOFTConfig(
            backend="openvla-oft",
            data=config.data,
            training=config.training,
            output_dir=config.output_dir,
            experiment_name=config.experiment_name,
            seed=config.seed,
            resume_from_checkpoint=config.resume_from_checkpoint,
            openvla_oft=openvla_oft_specific,
        )

        return openvla_oft_config

    def _convert_to_pi0_config(self, config: UnifiedVLAConfig):
        """Convert UnifiedVLAConfig to Pi0Config."""
        from crane_x7_vla.backends.pi0.config import Pi0Config, Pi0SpecificConfig

        # Create Pi0-specific config from backend_config if available
        pi0_specific = Pi0SpecificConfig()
        if config.backend_config:
            for key, value in config.backend_config.items():
                if hasattr(pi0_specific, key):
                    setattr(pi0_specific, key, value)

        # Create Pi0Config
        pi0_config = Pi0Config(
            backend=config.backend,
            data=config.data,
            training=config.training,
            output_dir=config.output_dir,
            experiment_name=config.experiment_name,
            seed=config.seed,
            resume_from_checkpoint=config.resume_from_checkpoint,
            pi0=pi0_specific,
        )

        return pi0_config

    def train(self) -> dict[str, Any]:
        """
        Execute training.

        Returns:
            Dictionary containing training results
        """
        logger.info("=" * 60)
        logger.info(f"Starting training with {self.config.backend} backend")
        logger.info(f"Experiment: {self.config.experiment_name}")
        logger.info(f"Output directory: {self.config.output_dir}")
        logger.info("=" * 60)

        results = self.backend.train()

        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info(f"Results: {results}")
        logger.info("=" * 60)

        return results

    def evaluate(
        self, checkpoint_path: str | Path | None = None, test_data_path: str | Path | None = None
    ) -> dict[str, float]:
        """
        Evaluate the model.

        Args:
            checkpoint_path: Path to model checkpoint
            test_data_path: Path to test dataset

        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Starting evaluation...")
        metrics = self.backend.evaluate(checkpoint_path, test_data_path)
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def save_config(self, path: str | Path) -> None:
        """
        Save configuration to file.

        Args:
            path: Path to save configuration
        """
        path = Path(path)
        self.config.to_yaml(path)
        logger.info(f"Configuration saved to {path}")

    @classmethod
    def from_config_file(cls, config_path: str | Path) -> VLATrainer:
        """
        Create trainer from configuration file.

        Args:
            config_path: Path to configuration YAML file

        Returns:
            VLATrainer instance
        """
        config = UnifiedVLAConfig.from_yaml(config_path)
        return cls(config)
