# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Image processing utilities without TensorFlow dependency.

Provides PIL/OpenCV-based image decoding, resizing, and basic processing
functions as replacements for tf.io.decode_jpeg, tf.image.resize, etc.
"""

import io
import logging
from typing import Literal

import cv2
import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)


def decode_jpeg(
    image_bytes: bytes,
    channels: int = 3,
) -> np.ndarray:
    """
    Decode JPEG image bytes to numpy array.

    Equivalent to tf.io.decode_jpeg(image_bytes, channels=3).

    Args:
        image_bytes: JPEG-encoded image bytes
        channels: Number of color channels (1 for grayscale, 3 for RGB)

    Returns:
        Decoded image as numpy array with shape (H, W, C), dtype uint8
    """
    if not image_bytes:
        raise ValueError("Empty image bytes")

    try:
        image = Image.open(io.BytesIO(image_bytes))

        if channels == 3:
            image = image.convert("RGB")
        elif channels == 1:
            image = image.convert("L")
        elif channels == 4:
            image = image.convert("RGBA")

        return np.array(image, dtype=np.uint8)

    except Exception as e:
        logger.error(f"Failed to decode JPEG image: {e}")
        raise


def decode_image(
    image_bytes: bytes,
    channels: int = 3,
) -> np.ndarray:
    """
    Decode image bytes (any format) to numpy array.

    Supports JPEG, PNG, and other formats supported by PIL.

    Args:
        image_bytes: Encoded image bytes
        channels: Number of color channels

    Returns:
        Decoded image as numpy array with shape (H, W, C), dtype uint8
    """
    return decode_jpeg(image_bytes, channels)


def resize_image(
    image: np.ndarray,
    size: tuple[int, int],
    interpolation: Literal["bilinear", "nearest", "bicubic", "area"] = "bilinear",
) -> np.ndarray:
    """
    Resize image to target size.

    Equivalent to tf.image.resize(image, size).

    Args:
        image: Input image as numpy array (H, W, C)
        size: Target size as (height, width)
        interpolation: Interpolation method

    Returns:
        Resized image as numpy array with shape (height, width, C), dtype uint8
    """
    interp_map = {
        "bilinear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
        "bicubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
    }

    interp = interp_map.get(interpolation, cv2.INTER_LINEAR)

    # cv2.resize expects (width, height)
    target_size = (size[1], size[0])

    resized = cv2.resize(image, target_size, interpolation=interp)

    return resized.astype(np.uint8)


def decode_and_resize(
    image_bytes: bytes,
    size: tuple[int, int],
    channels: int = 3,
) -> np.ndarray:
    """
    Decode and resize image in one operation.

    Convenience function combining decode_jpeg and resize_image.

    Args:
        image_bytes: JPEG-encoded image bytes
        size: Target size as (height, width)
        channels: Number of color channels

    Returns:
        Decoded and resized image as numpy array
    """
    image = decode_jpeg(image_bytes, channels)
    return resize_image(image, size)


def decode_raw(
    raw_bytes: bytes,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Decode raw bytes to numpy array.

    Equivalent to tf.io.decode_raw(raw_bytes, dtype).

    Args:
        raw_bytes: Raw bytes to decode
        dtype: Target numpy dtype

    Returns:
        Decoded array (1D, shape determined by byte length and dtype)
    """
    return np.frombuffer(raw_bytes, dtype=dtype)


def normalize_image(
    image: np.ndarray,
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> np.ndarray:
    """
    Normalize image to [-1, 1] range.

    Args:
        image: Input image as numpy array (H, W, C), dtype uint8
        mean: Mean values per channel
        std: Standard deviation per channel

    Returns:
        Normalized image as float32 array in [-1, 1] range
    """
    image = image.astype(np.float32) / 255.0
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    return (image - mean) / std


def denormalize_image(
    image: np.ndarray,
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> np.ndarray:
    """
    Denormalize image from [-1, 1] range back to uint8.

    Args:
        image: Normalized image as float32 array
        mean: Mean values used for normalization
        std: Standard deviation used for normalization

    Returns:
        Denormalized image as uint8 array in [0, 255] range
    """
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    image = image * std + mean
    image = np.clip(image * 255.0, 0, 255)
    return image.astype(np.uint8)


def to_pil_image(image: np.ndarray) -> Image.Image:
    """
    Convert numpy array to PIL Image.

    Args:
        image: Image as numpy array (H, W, C)

    Returns:
        PIL Image
    """
    return Image.fromarray(image)


def from_pil_image(image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to numpy array.

    Args:
        image: PIL Image

    Returns:
        Image as numpy array (H, W, C)
    """
    return np.array(image)
