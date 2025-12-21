#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Pi0/Pi0.5 inference core using flow matching action prediction.

Architecture aligned with vla/src/crane_x7_vla/backends/pi0/model.py.

Key differences between Pi0 and Pi0.5:
- Pi0: Continuous state input via projection, timestep mixed via MLP
       max_token_len=48, discrete_state_input=False
- Pi0.5: Discrete state tokens, timestep via adaRMSNorm conditioning
         max_token_len=200, discrete_state_input=True
"""

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


# Default settings for Pi0 vs Pi0.5
PI0_DEFAULTS = {
    "max_token_len": 48,
    "discrete_state_input": False,
}
PI05_DEFAULTS = {
    "max_token_len": 200,
    "discrete_state_input": True,
}


class Pi0InferenceCore(BaseVLAInferenceCore):
    """Pi0/Pi0.5 inference core using flow matching action prediction.

    This class handles model loading and inference for Pi0 and Pi0.5 models,
    using flow matching ODE integration for action chunk prediction.

    Features:
    - Action chunk caching: Predicts multiple future actions, uses them sequentially
    - Reduces inference frequency by reusing predicted actions

    Architecture:
    - Pi0: Continuous state input, MLP for timestep processing, max_token_len=48
    - Pi0.5: Discrete state tokens, adaRMSNorm for timestep injection, max_token_len=200

    Attributes:
        model_path: Path to the Pi0 model checkpoint
        device: PyTorch device for inference ('cuda' or 'cpu')
        model: Loaded Pi0Model
        config: Model configuration from checkpoint
    """

    # Default settings (can be overridden via constructor)
    DEFAULT_ACTION_DIM = 8
    DEFAULT_CHUNK_USE_COUNT = 10

    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        logger: Optional[logging.Logger] = None,
        chunk_use_count: int = DEFAULT_CHUNK_USE_COUNT,
        action_dim: int = DEFAULT_ACTION_DIM,
    ):
        """Initialize Pi0 inference core.

        Args:
            model_path: Path to model checkpoint directory
            device: Device for inference ('cuda' or 'cpu')
            logger: Optional logger instance
            chunk_use_count: Number of actions to use from each predicted chunk
            action_dim: Robot action dimension (default: 8 for CRANE-X7)
        """
        super().__init__(model_path, device, logger)
        self.config = None
        self.tokenizer = None
        self.norm_stats = None

        # Robot-specific settings
        self._robot_action_dim = action_dim

        # Action chunk caching
        self._action_cache: Optional[np.ndarray] = None  # [horizon, action_dim]
        self._cache_index: int = 0  # Current index in cached chunk
        self._cache_instruction: Optional[str] = None  # Instruction for cached chunk
        self._cache_image_hash: Optional[int] = None  # Image hash for cached chunk
        self._chunk_use_count: int = chunk_use_count  # Actions to use per chunk

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
            model_type = self.config.get("model_type", "pi0")
            is_pi05 = model_type == "pi0.5"
            self.logger.info(f'Model type: {model_type}')

            # Get defaults based on model type (aligned with vla/config.py)
            defaults = PI05_DEFAULTS if is_pi05 else PI0_DEFAULTS

            # Import Pi0 model classes
            Pi0Model, Pi0ModelConfig = self._import_pi0_classes()

            # Create model config - use float32 for inference to avoid dtype mismatches
            # Settings aligned with vla/src/crane_x7_vla/backends/pi0/config.py
            # Include openpi_checkpoint to ensure correct model architecture (vocab_size, etc.)
            openpi_checkpoint = self.config.get('openpi_checkpoint', None)
            if openpi_checkpoint:
                self.logger.info(f'Using OpenPI checkpoint for model architecture: {openpi_checkpoint}')

            model_config = Pi0ModelConfig(
                pi05=is_pi05,
                paligemma_variant=self.config.get('paligemma_variant', 'gemma_2b'),
                action_expert_variant=self.config.get('action_expert_variant', 'gemma_300m'),
                action_dim=self.config.get('action_dim', 32),
                action_horizon=self.config.get('action_horizon', 50),
                max_token_len=self.config.get('max_token_len', defaults['max_token_len']),
                dtype='float32',  # Use float32 for inference stability
                use_pretrained=False,  # Skip HuggingFace pretrained loading
                openpi_checkpoint=openpi_checkpoint,  # Use OpenPI checkpoint for correct vocab_size
            )

            # Create model (OpenPI checkpoint will be loaded if specified for correct architecture)
            self.model = Pi0Model(model_config)
            # Load finetuned weights (overwriting OpenPI base weights)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.model = self.model.to(device=self.device, dtype=torch.float32)
            self.model.eval()

            # Log model configuration details
            discrete_state = self.config.get('discrete_state_input', defaults['discrete_state_input'])
            image_size = self.config.get('image_size', (224, 224))
            camera_names = self.config.get('camera_names', ['base_0_rgb'])
            self.logger.info(
                f'Model loaded: pi05={model_config.pi05}, '
                f'action_dim={model_config.action_dim}, '
                f'action_horizon={model_config.action_horizon}, '
                f'max_token_len={model_config.max_token_len}, '
                f'discrete_state_input={discrete_state}, '
                f'image_size={image_size}, '
                f'num_cameras={len(camera_names)}'
            )

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

        Uses importlib to load the VLA package's pi0 module with proper
        package context for relative imports.

        Returns:
            Tuple of (Pi0Model, Pi0ModelConfig) classes
        """
        import importlib.util

        vla_path = get_vla_path()
        vla_src_path = str(vla_path / "src")
        pi0_package_path = vla_path / "src" / "crane_x7_vla" / "backends" / "pi0"

        # Add VLA src to sys.path for submodule imports
        if vla_src_path not in sys.path:
            sys.path.insert(0, vla_src_path)
            self.logger.info(f"Added {vla_src_path} to sys.path")

        # Register parent packages in sys.modules to enable relative imports
        # Use 'vla_' prefix to avoid collision with ROS 2 crane_x7_vla package
        package_prefix = "vla_crane_x7_vla"

        # Create and register parent package hierarchy
        self._register_package(f"{package_prefix}", vla_path / "src" / "crane_x7_vla")
        self._register_package(f"{package_prefix}.backends", vla_path / "src" / "crane_x7_vla" / "backends")
        self._register_package(f"{package_prefix}.backends.pi0", pi0_package_path)
        self._register_package(f"{package_prefix}.backends.pi0.models_pytorch", pi0_package_path / "models_pytorch")

        # Now load the model module with proper package context
        model_file = pi0_package_path / "model.py"
        if not model_file.exists():
            raise ImportError(f"Pi0 model file not found: {model_file}")

        spec = importlib.util.spec_from_file_location(
            f"{package_prefix}.backends.pi0.model",
            model_file,
            submodule_search_locations=[str(pi0_package_path)]
        )
        module = importlib.util.module_from_spec(spec)
        module.__package__ = f"{package_prefix}.backends.pi0"
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        return module.Pi0Model, module.Pi0ModelConfig

    def _register_package(self, name: str, path: Path) -> None:
        """Register a package in sys.modules for relative import support."""
        import importlib.util
        import types

        if name in sys.modules:
            return

        # Create a module object for the package
        init_file = path / "__init__.py"
        if init_file.exists():
            spec = importlib.util.spec_from_file_location(
                name, init_file,
                submodule_search_locations=[str(path)]
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                # If __init__.py fails, create empty module
                pass
        else:
            # Create empty package module
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            module.__package__ = name
            sys.modules[name] = module

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
        3. data_root_dir from checkpoint config (Pi0TrainerConfig format)
        4. Common Docker data directories
        """
        # List of paths to try
        stats_paths = [
            model_path / "dataset_statistics.json",
            model_path.parent / "dataset_statistics.json",
        ]

        # Add data_root_dir from checkpoint config if available
        # Pi0TrainerConfig stores 'data_root_dir' (aligned with backend.py)
        # Translate training paths (/root/vla/) to inference paths (/workspace/)
        if self.config:
            data_root = self.config.get('data_root_dir')
            if data_root:
                # Translate /root/vla/ -> /workspace/ for inference environment
                if data_root.startswith('/root/vla/'):
                    data_root = data_root.replace('/root/vla/', '/workspace/')
                data_root_path = Path(data_root)
                stats_paths.append(data_root_path / "dataset_statistics.json")
                # Also check parent directory (with permission error handling)
                try:
                    if data_root_path.parent.exists():
                        stats_paths.append(data_root_path.parent / "dataset_statistics.json")
                except PermissionError:
                    pass  # Skip inaccessible paths

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

        # Try to load from training data path in config (legacy format)
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
        state: Optional[np.ndarray] = None,
        log_callback: Optional[LogCallback] = None
    ) -> Optional[np.ndarray]:
        """Run Pi0 inference and return predicted action.

        Uses action chunk caching to reduce inference frequency:
        - Predicts multiple future actions at once
        - Returns cached actions until chunk is exhausted
        - Re-predicts when cache is empty or instruction changes

        Args:
            image: RGB image as numpy array (H, W, 3)
            instruction: Task instruction string
            state: Robot state array (joint positions, 8-dim for CRANE-X7)
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

        # Compute image hash for cache validation
        image_hash = compute_image_hash(image)

        # Check if we can use cached action (fast path)
        if self._can_use_cached_action(instruction, image_hash):
            action = self._get_cached_action(log_callback)
            if log_callback:
                log_callback(f'Final predicted action (cached): {action}')
            return action

        # Need to run inference (slow path)
        try:
            # Debug logging
            if log_callback:
                log_callback(
                    f'Pi0 inference (new prediction): image_shape={image.shape}, '
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

            # Prepare state (normalize and pad to 32-dim)
            state_tensor = self._prepare_state(state, log_callback)

            # Replicate image for all camera views (model trained with multiple cameras)
            camera_names = self.config.get('camera_names', ['base_0_rgb'])
            num_cameras = len(camera_names)
            images = [image_tensor] * num_cameras
            img_masks = [torch.tensor([True], device=self.device)] * num_cameras

            # Run flow matching inference
            action_chunk = self.model.sample_actions(
                images=images,
                img_masks=img_masks,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
                state=state_tensor,
                num_steps=self.config.get('num_denoise_steps', 10),
            )

            # Cache the action chunk for future use
            self._cache_action_chunk(action_chunk, instruction, image_hash, log_callback)

            # Return the first action from cache
            action = self._get_cached_action(log_callback)

            if log_callback:
                log_callback(f'Final predicted action: {action}')

            return action

        except Exception as e:
            self.logger.error(f'Pi0 inference failed: {e}')
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _can_use_cached_action(self, instruction: str, image_hash: int) -> bool:
        """Check if we can use a cached action.

        Cache is valid if:
        1. Cache exists
        2. Instruction matches
        3. Cache index is within usable range

        Note: We intentionally do NOT check image_hash here.
        Flow matching generates actions from random noise, so re-running
        inference on every frame causes oscillating behavior.
        The action chunk should be used for chunk_use_count steps,
        then a new inference with the current image will be performed.

        Args:
            instruction: Current task instruction
            image_hash: Hash of current image (unused, kept for API compatibility)

        Returns:
            True if cached action can be used
        """
        if self._action_cache is None:
            return False
        if self._cache_instruction != instruction:
            return False
        if self._cache_index >= self._chunk_use_count:
            return False
        return True

    def _get_cached_action(self, log_callback: Optional[LogCallback] = None) -> np.ndarray:
        """Get the next action from cache.

        Args:
            log_callback: Optional callback for logging

        Returns:
            Action array from cache
        """
        action = self._action_cache[self._cache_index]
        if log_callback:
            log_callback(f'Using cached action [{self._cache_index}/{self._chunk_use_count}]')
        self._cache_index += 1
        return action

    def _cache_action_chunk(
        self,
        action_chunk: torch.Tensor,
        instruction: str,
        image_hash: int,
        log_callback: Optional[LogCallback] = None,
    ) -> None:
        """Cache a predicted action chunk.

        Args:
            action_chunk: Predicted action chunk [1, horizon, action_dim]
            instruction: Instruction used for prediction
            image_hash: Hash of image used for prediction
            log_callback: Optional callback for logging
        """
        # Extract and denormalize all actions in chunk
        raw_chunk = action_chunk[0, :, :self._robot_action_dim].cpu().numpy()

        # Denormalize each action
        if self.norm_stats:
            denorm_chunk = np.array([
                self._denormalize_action(raw_chunk[i])
                for i in range(raw_chunk.shape[0])
            ])
        else:
            denorm_chunk = raw_chunk

        self._action_cache = denorm_chunk
        self._cache_index = 0
        self._cache_instruction = instruction
        self._cache_image_hash = image_hash

        if log_callback:
            log_callback(f'Cached {len(denorm_chunk)} actions, will use {self._chunk_use_count}')

    def invalidate_cache(self) -> None:
        """Invalidate the action cache.

        Call this when the task changes or robot state is reset.
        """
        self._action_cache = None
        self._cache_index = 0
        self._cache_instruction = None
        self._cache_image_hash = None
        self.logger.debug('Action cache invalidated')

    def _prepare_language(self, instruction: str) -> tuple:
        """Tokenize language instruction.

        Args:
            instruction: Task instruction string

        Returns:
            Tuple of (token_ids, attention_mask) as tensors
        """
        # Get max_token_len from config, with model-type-aware defaults
        model_type = self.config.get('model_type', 'pi0')
        defaults = PI05_DEFAULTS if model_type == 'pi0.5' else PI0_DEFAULTS
        max_len = self.config.get('max_token_len', defaults['max_token_len'])

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

    def _prepare_state(
        self,
        state: Optional[np.ndarray] = None,
        log_callback: Optional[LogCallback] = None,
    ) -> torch.Tensor:
        """Prepare robot state tensor with normalization and padding.

        Matches training-time preprocessing:
        1. Normalize state using quantile statistics (same as actions)
        2. Pad from 8-dim to 32-dim

        Args:
            state: Robot state array (8-dim for CRANE-X7) or None for zero state
            log_callback: Optional callback for logging

        Returns:
            State tensor [1, action_dim] normalized and padded
        """
        action_dim = self.config.get('action_dim', 32)

        if state is None:
            # Fallback to zero state if no state provided
            if log_callback:
                log_callback('Warning: No state provided, using zero state')
            return torch.zeros(1, action_dim, device=self.device, dtype=torch.float32)

        # Normalize state (same normalization as actions during training)
        normalized_state = self._normalize_state(state)

        if log_callback:
            log_callback(f'State: raw={state[:4]}..., normalized={normalized_state[:4]}...')

        # Pad from 8-dim to 32-dim
        padded_state = np.zeros(action_dim, dtype=np.float32)
        padded_state[:len(normalized_state)] = normalized_state

        # Convert to tensor
        state_tensor = torch.tensor(padded_state, device=self.device, dtype=torch.float32)
        state_tensor = state_tensor.unsqueeze(0)  # [1, action_dim]

        return state_tensor

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state using the same statistics and mode as actions.

        Supports both quantile and zscore normalization modes:
        - quantile: normalized = 2 * (state - q01) / (q99 - q01) - 1
        - zscore: normalized = (state - mean) / std

        Args:
            state: Raw state array (8-dim)

        Returns:
            Normalized state array
        """
        if not self.norm_stats:
            return state.astype(np.float32)

        # Get normalization mode from config
        mode = self.config.get('normalization_mode', 'quantile') if self.config else 'quantile'

        # Get stats - handle nested structure
        stats = self.norm_stats
        if 'crane_x7' in stats:
            stats = stats['crane_x7']
        if 'action' in stats:
            stats = stats['action']

        if mode == 'zscore':
            # Z-score normalization: (state - mean) / std
            mean = np.array(stats.get('mean', []))
            std = np.array(stats.get('std', []))

            if len(mean) < len(state) or len(std) < len(state):
                self.logger.warning('Normalization stats dimension mismatch, using raw state')
                return state.astype(np.float32)

            mean = mean[:len(state)]
            std = std[:len(state)]

            # Avoid division by zero
            std = np.where(std < 1e-6, 1.0, std)
            normalized = (state - mean) / std

            # Clip to reasonable range (same as training)
            normalized = np.clip(normalized, -10.0, 10.0)
        else:
            # Quantile normalization: 2 * (state - q01) / (q99 - q01) - 1
            q01 = np.array(stats.get('q01', []))
            q99 = np.array(stats.get('q99', []))

            if len(q01) < len(state) or len(q99) < len(state):
                self.logger.warning('Normalization stats dimension mismatch, using raw state')
                return state.astype(np.float32)

            q01 = q01[:len(state)]
            q99 = q99[:len(state)]

            # Normalize to [-1, 1]
            range_val = q99 - q01
            range_val = np.where(range_val < 1e-6, 1.0, range_val)
            normalized = 2 * (state - q01) / range_val - 1

            # Clip to [-1, 1]
            normalized = np.clip(normalized, -1.0, 1.0)

        return normalized.astype(np.float32)

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

    @property
    def chunk_use_count(self) -> int:
        """Number of actions to use from each predicted chunk."""
        return self._chunk_use_count

    @chunk_use_count.setter
    def chunk_use_count(self, value: int) -> None:
        """Set number of actions to use from each predicted chunk.

        Args:
            value: Number of actions (1-50, typically 5-20)
        """
        action_horizon = self.config.get('action_horizon', 50) if self.config else 50
        self._chunk_use_count = max(1, min(value, action_horizon))
        self.invalidate_cache()  # Clear cache when changing setting

    @property
    def cache_status(self) -> dict:
        """Get current cache status for debugging."""
        return {
            'has_cache': self._action_cache is not None,
            'cache_index': self._cache_index,
            'chunk_use_count': self._chunk_use_count,
            'cache_instruction': self._cache_instruction,
            'cache_image_hash': self._cache_image_hash,
            'remaining_actions': max(0, self._chunk_use_count - self._cache_index) if self._action_cache is not None else 0,
        }
