# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
CRANE-X7 Policy for PyTorch VLA models.

Provides a Policy wrapper for running inference on CRANE-X7 robot
using PyTorch-based VLA models (Flow Matching, etc.).

Note: This module has no external dependencies on openpi or JAX.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import torch


class DataTransformFn(Protocol):
    """Protocol for data transformation functions."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]: ...


@dataclasses.dataclass(frozen=True)
class CraneX7InputTransform:
    """
    Transform CRANE-X7 observation format to model input format.

    This transform handles:
    - State padding from 8 DOF to target dimension (e.g., 32)
    - Image format conversion (single camera to multi-camera dict)
    - Image mask creation for missing cameras
    """

    # Source state dimension (CRANE-X7)
    source_dim: int = 8
    # Target state dimension (model)
    target_dim: int = 32
    # Expected camera names
    camera_names: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    # Image size for padding
    image_size: tuple[int, int] = (224, 224)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply input transformation."""
        result = dict(data)

        # Pad state from source_dim -> target_dim
        if "state" in result:
            state = np.asarray(result["state"])
            if state.shape[-1] < self.target_dim:
                pad_width = [(0, 0)] * (state.ndim - 1) + [(0, self.target_dim - state.shape[-1])]
                result["state"] = np.pad(state, pad_width, mode="constant")

        # Ensure images dict exists and is properly formatted
        if "image" not in result:
            result["image"] = {}

        # If only a single image is provided, map to primary camera
        if isinstance(result.get("image"), np.ndarray):
            result["image"] = {self.camera_names[0]: result["image"]}

        # Create image mask
        if "image_mask" not in result:
            result["image_mask"] = {}

        # Ensure all expected cameras exist
        for cam_name in self.camera_names:
            if cam_name not in result["image"]:
                # Create zero-padded image
                result["image"][cam_name] = np.zeros((*self.image_size, 3), dtype=np.uint8)
                result["image_mask"][cam_name] = False
            else:
                result["image_mask"][cam_name] = True

        return result


@dataclasses.dataclass(frozen=True)
class CraneX7OutputTransform:
    """
    Transform model output to CRANE-X7 action format.

    This transform handles:
    - Action truncation from target dimension to 8 DOF
    - Optional action chunk selection (first action only)
    """

    # Target action dimension (CRANE-X7)
    target_dim: int = 8
    # Whether to return only the first action from the chunk
    first_action_only: bool = False

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply output transformation."""
        result = dict(data)

        if "actions" in result:
            actions = np.asarray(result["actions"])

            # Truncate from model dim -> CRANE-X7 dim
            if actions.shape[-1] > self.target_dim:
                actions = actions[..., : self.target_dim]

            # Return only first action if requested
            if self.first_action_only and actions.ndim > 1:
                actions = actions[0]

            result["actions"] = actions

        return result


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt:
    """Inject default prompt if not provided."""

    default_prompt: str = "manipulate objects"

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Inject default prompt."""
        result = dict(data)
        if "prompt" not in result or result["prompt"] is None:
            result["prompt"] = self.default_prompt
        return result


class CraneX7Policy:
    """
    Policy wrapper for CRANE-X7 robot with PyTorch VLA models.

    Provides a unified interface for running inference with different
    PyTorch-based models (Flow Matching, etc.).

    This class is independent of openpi and JAX.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str | torch.device = "cuda",
        transforms: Sequence[Callable[[dict[str, Any]], dict[str, Any]]] = (),
        output_transforms: Sequence[Callable[[dict[str, Any]], dict[str, Any]]] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        source_dim: int = 8,
        target_dim: int = 32,
        first_action_only: bool = True,
    ):
        """
        Initialize CRANE-X7 Policy.

        Args:
            model: PyTorch model with sample_actions() method
            device: Device for inference
            transforms: Additional input transforms
            output_transforms: Additional output transforms
            sample_kwargs: Additional kwargs for model.sample_actions
            metadata: Policy metadata
            source_dim: CRANE-X7 action dimension (8)
            target_dim: Model action dimension (32)
            first_action_only: Return only first action from chunk
        """
        import torch

        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.sample_kwargs = sample_kwargs or {}
        self.metadata = metadata or {}

        # Build transform pipeline
        crane_x7_input = CraneX7InputTransform(
            source_dim=source_dim,
            target_dim=target_dim,
        )
        crane_x7_output = CraneX7OutputTransform(
            target_dim=source_dim,
            first_action_only=first_action_only,
        )

        self.input_transforms: list[Callable[[dict[str, Any]], dict[str, Any]]] = [crane_x7_input, *transforms]
        self.output_transforms: list[Callable[[dict[str, Any]], dict[str, Any]]] = [crane_x7_output, *output_transforms]

        # Store dimensions
        self.source_dim = source_dim
        self.target_dim = target_dim

    def infer(self, observation: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """
        Run inference on observation.

        Args:
            observation: Dict with 'state', 'image', optionally 'prompt'
            **kwargs: Additional sampling kwargs

        Returns:
            Dict with 'actions' key containing predicted actions
        """
        import torch

        # Apply input transforms
        data = dict(observation)
        for transform in self.input_transforms:
            data = transform(data)

        # Convert to tensors
        obs_tensor = self._to_tensor(data)

        # Run model
        self.model.eval()
        with torch.no_grad():
            sample_kwargs = {**self.sample_kwargs, **kwargs}
            actions = self.model.sample_actions(obs_tensor, **sample_kwargs)

        # Convert back to numpy
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        result: dict[str, Any] = {"actions": actions}

        # Apply output transforms
        for transform in self.output_transforms:
            result = transform(result)

        return result

    def _to_tensor(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert numpy arrays to tensors."""
        import torch

        result: dict[str, Any] = {}

        if "state" in data:
            state = data["state"]
            if isinstance(state, np.ndarray):
                state = torch.tensor(state, dtype=torch.float32)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            result["state"] = state.to(self.device)

        if "image" in data:
            result["image"] = {}
            for name, img in data["image"].items():
                if isinstance(img, np.ndarray):
                    img_tensor = torch.tensor(img, dtype=torch.float32)
                    if img_tensor.max() > 1.0:
                        img_tensor = img_tensor / 255.0
                    # Add batch dimension if needed
                    if img_tensor.dim() == 3:
                        img_tensor = img_tensor.unsqueeze(0)
                    result["image"][name] = img_tensor.to(self.device)
                else:
                    result["image"][name] = img

        if "prompt" in data:
            result["prompt"] = data["prompt"]

        return result

    def predict_action(
        self,
        state: np.ndarray,
        image: np.ndarray,
        prompt: str = "manipulate objects",
    ) -> np.ndarray:
        """
        Convenience method for single-step prediction.

        Args:
            state: Robot state [8]
            image: RGB image [H, W, 3]
            prompt: Task instruction

        Returns:
            Predicted action [8]
        """
        obs = {
            "state": state,
            "image": {"base_0_rgb": image},
            "prompt": prompt,
        }
        result = self.infer(obs)
        return result["actions"]


def create_crane_x7_policy(
    model: torch.nn.Module,
    *,
    transforms: Sequence[Callable[[dict[str, Any]], dict[str, Any]]] = (),
    output_transforms: Sequence[Callable[[dict[str, Any]], dict[str, Any]]] = (),
    device: str | torch.device = "cuda",
    first_action_only: bool = True,
    default_prompt: str = "manipulate objects",
    **kwargs: Any,
) -> CraneX7Policy:
    """
    Factory function to create a CRANE-X7 policy.

    Args:
        model: PyTorch model with sample_actions() method
        transforms: Additional input transforms
        output_transforms: Additional output transforms
        device: Device for inference
        first_action_only: Return only first action from chunk
        default_prompt: Default task instruction
        **kwargs: Additional kwargs passed to CraneX7Policy

    Returns:
        Configured CraneX7Policy instance
    """
    prompt_transform = InjectDefaultPrompt(default_prompt)
    all_transforms: list[Callable[[dict[str, Any]], dict[str, Any]]] = [prompt_transform, *transforms]

    return CraneX7Policy(
        model,
        transforms=all_transforms,
        output_transforms=output_transforms,
        device=device,
        first_action_only=first_action_only,
        **kwargs,
    )
