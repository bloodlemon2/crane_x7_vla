# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""
CRANE-X7用VLA定数.

OpenVLA-OFTのトレーニングと推論に必要な定数を定義。
公式openvla-oftリポジトリのprismatic/vla/constants.pyを参考に、
CRANE-X7ロボット向けにカスタマイズ。
"""

from enum import Enum


# Llama-2 トークン定数
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2  # '</s>'


class NormalizationType(str, Enum):
    """アクションと固有感覚状態の正規化方式."""

    NORMAL = "normal"  # 平均=0、標準偏差=1に正規化
    BOUNDS = "bounds"  # 区間[-1, 1]に正規化
    BOUNDS_Q99 = "bounds_q99"  # [1%分位点, ..., 99%分位点] -> [-1, ..., 1]


# CRANE-X7用定数
NUM_ACTIONS_CHUNK = 8  # アクションチャンク長(予測する将来ステップ数)
ACTION_DIM = 8  # アクション次元(関節7 + グリッパー1)
PROPRIO_DIM = 8  # 固有感覚次元(関節位置7 + グリッパー位置1)

# デフォルトの正規化方式
ACTION_PROPRIO_NORMALIZATION_TYPE = NormalizationType.BOUNDS_Q99
