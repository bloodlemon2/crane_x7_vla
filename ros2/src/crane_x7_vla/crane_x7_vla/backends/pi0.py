#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Pi0/Pi0.5 inference core using flow matching action prediction."""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from crane_x7_vla.core.base import BaseVLAInferenceCore
from crane_x7_vla.core.types import LogCallback
from crane_x7_vla.utils.paths import get_vla_path
from crane_x7_vla.utils.image import image_to_tensor, compute_image_hash
from crane_x7_vla.utils.normalization import denormalize_action, load_norm_stats_from_config


class Pi0InferenceCore(BaseVLAInferenceCore):
    """Pi0/Pi0.5 inference core using flow matching action prediction.

    This class handles model loading and inference for Pi0 and Pi0.5 models,
    using flow matching ODE integration for action chunk prediction.

    Attributes:
        model_path: Path to the Pi0 model checkpoint
        device: PyTorch device for inference ('cuda' or 'cpu')
        model: Loaded Pi0Model
        config: Model configuration from checkpoint
    """

    # CRANE-X7 specific settings
    CRANE_X7_ACTION_DIM = 8

    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize Pi0 inference core.

        Args:
            model_path: Path to model checkpoint directory
            device: Device for inference ('cuda' or 'cpu')
            logger: Optional logger instance
        """
        super().__init__(model_path, device, logger)
        self.config = None
        self.tokenizer = None
        self.norm_stats = None

    def load_model(self) -> bool:
        """Load Pi0/Pi0.5 model from checkpoint.

        Returns:
            True if model loaded successfully, False otherwise
        """
        if not self.model_path:
            self.logger.error('Model path not specified!')
            return False

        if self.device is None:
            self.setup_device()

        model_path = Path(self.model_path)
        if not model_path.exists():
            self.logger.error(f'Model path does not exist: {model_path}')
            return False

        self.logger.info(f'Loading Pi0 model from {model_path}...')

        try:
            # Load checkpoint
            checkpoint_path = model_path / "checkpoint.pt"
            if not checkpoint_path.exists():
                self.logger.error(f'Checkpoint not found: {checkpoint_path}')
                return False

            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            # Extract config
            if 'config' not in checkpoint:
                self.logger.error('No config found in checkpoint')
                return False

            self.config = checkpoint['config']
            self.logger.info(f'Model type: {self.config.get("model_type", "unknown")}')

            # Import Pi0 model classes
            Pi0Model, Pi0ModelConfig = self._import_pi0_classes()

            # Create model config - use float32 for inference to avoid dtype mismatches
            model_config = Pi0ModelConfig(
                pi05=self.config.get('model_type') == 'pi0.5',
                paligemma_variant=self.config.get('paligemma_variant', 'gemma_2b'),
                action_expert_variant=self.config.get('action_expert_variant', 'gemma_300m'),
                action_dim=self.config.get('action_dim', 32),
                action_horizon=self.config.get('action_horizon', 50),
                max_token_len=self.config.get('max_token_len', 48),
                dtype='float32',  # Use float32 for inference stability
            )

            # Create and load model
            self.model = Pi0Model(model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.model = self.model.to(device=self.device, dtype=torch.float32)
            self.model.eval()

            self.logger.info(f'Model loaded: pi05={model_config.pi05}, '
                           f'action_dim={model_config.action_dim}, '
                           f'action_horizon={model_config.action_horizon}')

            # Load tokenizer for language processing
            self._load_tokenizer()

            # Load normalization statistics
            self._load_norm_stats(model_path)

            self.logger.info('Pi0 model loaded successfully')
            return True

        except Exception as e:
            self.logger.error(f'Failed to load Pi0 model: {e}')
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _import_pi0_classes(self):
        """Import Pi0 model classes, handling path setup.

        Uses importlib.util to directly load the VLA package's pi0 model,
        avoiding namespace conflicts with the ROS 2 package's pi0.py file.

        Returns:
            Tuple of (Pi0Model, Pi0ModelConfig) classes
        """
        import importlib.util

        vla_path = get_vla_path()
        model_file = vla_path / "src" / "crane_x7_vla" / "backends" / "pi0" / "model.py"

        if not model_file.exists():
            raise ImportError(f"Pi0 model file not found: {model_file}")

        # Load module directly using importlib.util
        spec = importlib.util.spec_from_file_location("pi0_model", model_file)
        pi0_model_module = importlib.util.module_from_spec(spec)
        sys.modules["pi0_model"] = pi0_model_module
        spec.loader.exec_module(pi0_model_module)

        return pi0_model_module.Pi0Model, pi0_model_module.Pi0ModelConfig

    def _load_tokenizer(self) -> None:
        """Load Gemma tokenizer for language encoding."""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "google/gemma-2b",
                trust_remote_code=True
            )
            self.logger.info('Loaded Gemma tokenizer')
        except Exception as e:
            self.logger.warning(f'Failed to load tokenizer: {e}')
            self.tokenizer = None

    def _load_norm_stats(self, model_path: Path) -> None:
        """Load action normalization statistics.

        Tries multiple paths in order:
        1. checkpoint_dir/dataset_statistics.json
        2. checkpoint_dir/../dataset_statistics.json (parent dir)
        3. Training data path from config (Docker path mapped)
        4. Common Docker data directories
        """
        # List of paths to try
        stats_paths = [
            model_path / "dataset_statistics.json",
            model_path.parent / "dataset_statistics.json",
        ]

        # Add Docker workspace paths as fallback
        workspace_data_dirs = [
            Path("/workspace/data/tfrecord_logs_2_10Hz"),
            Path("/workspace/data/tfrecord_logs"),
            Path("/workspace/data"),
        ]
        for data_dir in workspace_data_dirs:
            stats_paths.append(data_dir / "dataset_statistics.json")

        # Try each path
        for stats_path in stats_paths:
            if stats_path.exists():
                try:
                    with open(stats_path, 'r') as f:
                        self.norm_stats = json.load(f)
                    self.logger.info(f'Loaded normalization stats from {stats_path}')
                    return
                except Exception as e:
                    self.logger.warning(f'Failed to load stats from {stats_path}: {e}')

        # Try to load from training data path in config
        if self.config:
            stats = load_norm_stats_from_config(self.config, self.logger)
            if stats:
                self.norm_stats = stats
                return

        self.logger.warning('No normalization statistics found - actions will not be denormalized')

    def predict_action(
        self,
        image: np.ndarray,
        instruction: str,
        log_callback: Optional[LogCallback] = None
    ) -> Optional[np.ndarray]:
        """Run Pi0 inference and return predicted action.

        Args:
            image: RGB image as numpy array (H, W, 3)
            instruction: Task instruction string
            log_callback: Optional callback for logging debug info

        Returns:
            Action array (8-dim for CRANE-X7) or None if inference fails
        """
        if self.model is None:
            self.logger.error('Model not loaded')
            return None

        if self.tokenizer is None:
            self.logger.error('Tokenizer not loaded')
            return None

        try:
            # Debug logging
            if log_callback:
                image_hash = compute_image_hash(image)
                log_callback(
                    f'Pi0 inference: image_shape={image.shape}, '
                    f'image_hash={image_hash:04d}, '
                    f'instruction="{instruction}"'
                )

            # Prepare image using utility
            # Note: Keep image as float32 for SigLIP vision tower compatibility
            target_size = self.config.get('image_size', (224, 224))
            image_tensor = image_to_tensor(
                image,
                target_size=target_size,
                device=self.device_name,
            )

            # Prepare language tokens
            lang_tokens, lang_masks = self._prepare_language(instruction)

            # Prepare state (zero state for inference without prior state)
            state = self._prepare_state()

            # Run flow matching inference (optimized: 5 steps instead of 10)
            action_chunk = self.model.sample_actions(
                images=[image_tensor],
                img_masks=[torch.tensor([True], device=self.device)],
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
                state=state,
                num_steps=self.config.get('num_denoise_steps', 5),
            )

            # Extract first action from chunk and truncate to CRANE-X7 dims
            raw_action = action_chunk[0, 0, :self.CRANE_X7_ACTION_DIM].cpu().numpy()

            if log_callback:
                log_callback(f'Raw model output (normalized): {raw_action}')

            # Denormalize if stats available
            if self.norm_stats:
                action = self._denormalize_action(raw_action)
                if log_callback:
                    log_callback(f'Denormalized action: {action}')
            else:
                action = raw_action
                if log_callback:
                    log_callback('WARNING: No norm_stats - using raw action')

            if log_callback:
                log_callback(f'Final predicted action: {action}')

            return action

        except Exception as e:
            self.logger.error(f'Pi0 inference failed: {e}')
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _prepare_language(self, instruction: str) -> tuple:
        """Tokenize language instruction.

        Args:
            instruction: Task instruction string

        Returns:
            Tuple of (token_ids, attention_mask) as tensors
        """
        max_len = self.config.get('max_token_len', 48)

        encoding = self.tokenizer(
            instruction,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        lang_tokens = encoding["input_ids"].to(self.device)
        lang_masks = encoding["attention_mask"].bool().to(self.device)

        return lang_tokens, lang_masks

    def _prepare_state(self) -> torch.Tensor:
        """Prepare robot state tensor.

        For inference without prior state, returns zero state.

        Returns:
            State tensor [1, action_dim]
        """
        action_dim = self.config.get('action_dim', 32)
        state = torch.zeros(1, action_dim, device=self.device, dtype=torch.float32)
        return state

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Denormalize action using stored statistics.

        Args:
            action: Normalized action array

        Returns:
            Denormalized action array
        """
        mode = self.config.get('normalization_mode', 'quantile')
        return denormalize_action(
            action,
            self.norm_stats,
            mode=mode,
            stats_key='crane_x7',
            logger=self.logger,
        )

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self.model is not None and self.tokenizer is not None
