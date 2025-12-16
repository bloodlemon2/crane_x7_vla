# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Image augmentation utilities without TensorFlow dependency.

Provides OpenCV/NumPy-based image augmentation matching OpenVLA's defaults
as replacements for tf.image.random_* functions.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class ImageAugmentationConfig:
    """
    Configuration for image augmentation matching OpenVLA defaults.

    These values match the DEFAULT_IMAGE_AUG_KWARGS from OpenVLA finetune.py:
    - random_resized_crop: scale=[0.9, 0.9], ratio=[1.0, 1.0]
    - random_brightness: [0.2]
    - random_contrast: [0.8, 1.2]
    - random_saturation: [0.8, 1.2]
    - random_hue: [0.05]
    """

    crop_scale: float = 0.9
    """Scale factor for random crop (0.9 = crop 90% of image)"""

    brightness_delta: float = 0.2
    """Maximum brightness adjustment (as fraction of 255)"""

    contrast_range: tuple[float, float] = (0.8, 1.2)
    """Range for random contrast adjustment"""

    saturation_range: tuple[float, float] = (0.8, 1.2)
    """Range for random saturation adjustment"""

    hue_delta: float = 0.05
    """Maximum hue shift (as fraction of 180 degrees)"""

    augment_order: list[str] = field(
        default_factory=lambda: [
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ]
    )
    """Order of augmentations to apply"""


class ImageAugmentor:
    """
    Image augmentation using OpenCV/NumPy (no TensorFlow dependency).

    This class provides augmentations matching OpenVLA's defaults,
    implemented using pure NumPy and OpenCV operations.

    Example:
        config = ImageAugmentationConfig(brightness_delta=0.3)
        augmentor = ImageAugmentor(target_size=(224, 224), config=config)
        augmented = augmentor(image)
    """

    def __init__(
        self,
        target_size: tuple[int, int],
        config: ImageAugmentationConfig | None = None,
    ):
        """
        Initialize image augmentor.

        Args:
            target_size: Target image size as (height, width)
            config: Augmentation configuration. Uses defaults if None.
        """
        self.target_size = target_size
        self.config = config or ImageAugmentationConfig()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply augmentation pipeline to image.

        Args:
            image: Input image as numpy array (H, W, C), dtype uint8

        Returns:
            Augmented image as numpy array (H, W, C), dtype uint8
        """
        for aug_name in self.config.augment_order:
            method = getattr(self, f"_{aug_name}", None)
            if method is not None:
                image = method(image)
        return image

    def _random_resized_crop(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random resized crop.

        Crops to crop_scale of original size, then resizes back to target_size.
        Matches tf.image.random_crop behavior.

        Args:
            image: Input image (H, W, C)

        Returns:
            Cropped and resized image (target_h, target_w, C)
        """
        h, w = image.shape[:2]
        crop_h = int(h * self.config.crop_scale)
        crop_w = int(w * self.config.crop_scale)

        # Random crop position
        top = np.random.randint(0, h - crop_h + 1) if h > crop_h else 0
        left = np.random.randint(0, w - crop_w + 1) if w > crop_w else 0

        # Crop
        cropped = image[top : top + crop_h, left : left + crop_w]

        # Resize to target size
        resized = cv2.resize(
            cropped,
            (self.target_size[1], self.target_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        return resized.astype(np.uint8)

    def _random_brightness(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random brightness adjustment.

        Matches tf.image.random_brightness behavior.

        Args:
            image: Input image (H, W, C), dtype uint8

        Returns:
            Brightness-adjusted image
        """
        delta = np.random.uniform(-self.config.brightness_delta, self.config.brightness_delta)
        delta_scaled = delta * 255.0

        image = image.astype(np.float32)
        image = image + delta_scaled
        image = np.clip(image, 0, 255)

        return image.astype(np.uint8)

    def _random_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random contrast adjustment.

        Matches tf.image.random_contrast behavior.
        Contrast adjustment: (image - mean) * factor + mean

        Args:
            image: Input image (H, W, C), dtype uint8

        Returns:
            Contrast-adjusted image
        """
        low, high = self.config.contrast_range
        factor = np.random.uniform(low, high)

        image = image.astype(np.float32)
        mean = image.mean()
        image = (image - mean) * factor + mean
        image = np.clip(image, 0, 255)

        return image.astype(np.uint8)

    def _random_saturation(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random saturation adjustment.

        Matches tf.image.random_saturation behavior.
        Converts to HSV, adjusts S channel, converts back.

        Args:
            image: Input image (H, W, C) in RGB, dtype uint8

        Returns:
            Saturation-adjusted image in RGB
        """
        low, high = self.config.saturation_range
        factor = np.random.uniform(low, high)

        # Convert RGB to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Adjust saturation (channel 1)
        hsv[:, :, 1] = hsv[:, :, 1] * factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

        # Convert back to RGB
        hsv = hsv.astype(np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return rgb

    def _random_hue(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random hue shift.

        Matches tf.image.random_hue behavior.
        Converts to HSV, shifts H channel, converts back.

        Args:
            image: Input image (H, W, C) in RGB, dtype uint8

        Returns:
            Hue-shifted image in RGB
        """
        delta = np.random.uniform(-self.config.hue_delta, self.config.hue_delta)
        # OpenCV hue is in [0, 180], TF uses [-1, 1] relative
        # delta is fraction, convert to degrees
        hue_shift = delta * 180

        # Convert RGB to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Shift hue (channel 0), wrap around [0, 180)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180

        # Convert back to RGB
        hsv = hsv.astype(np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return rgb


def apply_augmentation(
    image: np.ndarray,
    target_size: tuple[int, int],
    config: ImageAugmentationConfig | None = None,
) -> np.ndarray:
    """
    Convenience function to apply augmentation to a single image.

    Args:
        image: Input image as numpy array (H, W, C), dtype uint8
        target_size: Target image size as (height, width)
        config: Augmentation configuration. Uses defaults if None.

    Returns:
        Augmented image as numpy array
    """
    augmentor = ImageAugmentor(target_size, config)
    return augmentor(image)
