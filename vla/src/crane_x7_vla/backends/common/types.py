# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""
型定義.

prismaticからの互換性のための型エイリアス。
"""

from collections.abc import Callable

import torch
from PIL import Image


# ImageTransform型のエイリアス (prismaticからの互換性)
ImageTransform = Callable[[Image.Image], torch.Tensor]

__all__ = ["ImageTransform"]
