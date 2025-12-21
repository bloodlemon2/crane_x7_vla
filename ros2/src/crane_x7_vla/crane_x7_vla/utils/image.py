#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Image preparation utilities for VLA inference."""

from typing import Tuple, Optional

import numpy as np
from PIL import Image as PILImage
import torch


def prepare_image_for_inference(
    image: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
) -> PILImage.Image:
    """Prepare image for VLA model inference.

    Converts numpy array to PIL Image for model processing.

    Args:
        image: RGB image as numpy array (H, W, 3), uint8 or float
        target_size: Optional (width, height) tuple for resizing

    Returns:
        PIL Image in RGB format
    """
    # Convert to PIL Image
    pil_image = PILImage.fromarray(image).convert("RGB")

    # Resize if target size specified
    if target_size is not None:
        pil_image = pil_image.resize(target_size, PILImage.Resampling.BILINEAR)

    return pil_image


def image_to_tensor(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    normalize_range: Tuple[float, float] = (-1.0, 1.0),
    device: str = 'cuda',
) -> torch.Tensor:
    """Convert image to normalized tensor for Pi0 model input.

    Args:
        image: RGB image as numpy array (H, W, 3), uint8 or float
        target_size: (height, width) for resizing (matching training config format)
        normalize_range: (min, max) for normalization
        device: Target device for tensor

    Returns:
        Image tensor [1, C, H, W] normalized to specified range
    """
    # Convert (height, width) to (width, height) for PIL
    pil_target_size = (target_size[1], target_size[0])

    # Prepare PIL image
    pil_image = prepare_image_for_inference(image, pil_target_size)

    # Convert to float array
    img_array = np.array(pil_image, dtype=np.float32)

    # Normalize to [-1, 1] if input is uint8 (0-255)
    if img_array.max() > 1.0:
        min_val, max_val = normalize_range
        # Scale from [0, 255] to [min_val, max_val]
        img_array = (img_array / 127.5) - 1.0
        if normalize_range != (-1.0, 1.0):
            # Further scale if needed
            img_array = (img_array + 1.0) / 2.0 * (max_val - min_val) + min_val

    # Convert to CHW format
    img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1))
    img_tensor = img_tensor.unsqueeze(0).to(device, dtype=torch.float32)

    return img_tensor


def compute_image_hash(image: np.ndarray, mod: int = 10000) -> int:
    """Compute a simple hash of image data for debugging.

    Args:
        image: Image as numpy array
        mod: Modulo for hash (default 10000)

    Returns:
        Hash value modulo mod
    """
    return hash(image.tobytes()) % mod
