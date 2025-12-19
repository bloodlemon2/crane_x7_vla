#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Abstract base class for VLA inference cores."""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Callable

import numpy as np
import torch

from crane_x7_vla.core.types import LogCallback


class BaseVLAInferenceCore(ABC):
    """Abstract base class for VLA inference cores.

    This class defines the interface for all VLA inference implementations,
    allowing seamless switching between different backends (OpenVLA, Pi0, etc.).

    Attributes:
        model_path: Path to the VLA model (local or HuggingFace Hub ID)
        device_name: Target device name ('cuda' or 'cpu')
        logger: Logger instance for debugging
        model: Loaded model (type depends on backend)
        device: PyTorch device object
    """

    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize base inference core.

        Args:
            model_path: Path to model (local path or HuggingFace Hub ID)
            device: Device for inference ('cuda' or 'cpu')
            logger: Optional logger instance
        """
        self.model_path = model_path
        self.device_name = device
        self.logger = logger or logging.getLogger(__name__)
        self.model = None
        self.device: Optional[torch.device] = None

    def setup_device(self) -> torch.device:
        """Setup PyTorch device based on availability and configuration.

        Returns:
            Configured PyTorch device
        """
        if torch.cuda.is_available() and self.device_name == 'cuda':
            self.device = torch.device('cuda')
            self.logger.info(f'Using CUDA device: {torch.cuda.get_device_name(0)}')
        else:
            self.device = torch.device('cpu')
            if self.device_name == 'cuda':
                self.logger.warning('CUDA not available, falling back to CPU')
            else:
                self.logger.info('Using CPU device')
        return self.device

    @abstractmethod
    def load_model(self) -> bool:
        """Load the VLA model.

        Returns:
            True on success, False on failure
        """
        pass

    @abstractmethod
    def predict_action(
        self,
        image: np.ndarray,
        instruction: str,
        state: Optional[np.ndarray] = None,
        log_callback: Optional[LogCallback] = None
    ) -> Optional[np.ndarray]:
        """Run inference and return predicted action.

        Args:
            image: RGB image as numpy array (H, W, 3)
            instruction: Task instruction string
            state: Robot state array (joint positions, 8-dim for CRANE-X7)
            log_callback: Optional callback for logging debug info

        Returns:
            Action array or None if inference fails
        """
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference.

        Returns:
            True if model is ready for inference
        """
        pass
