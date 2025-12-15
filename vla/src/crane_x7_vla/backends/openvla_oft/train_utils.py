# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""
トレーニング/ファインチューニング用ユーティリティ関数.

公式openvla-oftリポジトリのprismatic/training/train_utils.pyを
CRANE-X7用にコピー・カスタマイズ。
"""

import torch

from .constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX


def get_current_action_mask(token_ids: torch.Tensor) -> torch.Tensor:
    """
    現在のアクション(最初のACTION_DIM個のトークン)のマスクを計算.

    Args:
        token_ids: ラベルのトークンID (batch_size, seq_len)

    Returns:
        現在のアクションを示すブールマスク (batch_size, seq_len)
    """
    # IGNORE_INDEXでない位置をマーク
    newline_positions = token_ids != IGNORE_INDEX

    # 累積和を計算してIGNORE_INDEX以降の領域を識別
    cumsum = torch.cumsum(newline_positions, dim=1)

    # 最初のACTION_DIM個のアクショントークンを選択するマスク
    mask = (cumsum >= 1) & (cumsum <= ACTION_DIM)

    # アクショントークンのみを抽出(ACTION_TOKEN_BEGIN_IDXより大きいトークン)
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    mask = action_tokens_only_mask * mask

    return mask


def get_next_actions_mask(token_ids: torch.Tensor) -> torch.Tensor:
    """
    将来のアクション(ACTION_DIM個より後のトークン)のマスクを計算.

    Args:
        token_ids: ラベルのトークンID (batch_size, seq_len)

    Returns:
        将来のアクションを示すブールマスク (batch_size, seq_len)
    """
    # IGNORE_INDEXでない位置をマーク
    newline_positions = token_ids != IGNORE_INDEX

    # 累積和を計算してIGNORE_INDEX以降の領域を識別
    cumsum = torch.cumsum(newline_positions, dim=1)

    # ACTION_DIM個より後のトークンを選択するマスク
    mask = cumsum > ACTION_DIM

    # アクショントークンのみを抽出
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    mask = action_tokens_only_mask * mask

    return mask


def compute_token_accuracy(
    predicted_token_ids: torch.Tensor,
    ground_truth_token_ids: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    マスクされた位置でのトークン予測精度を計算.

    Args:
        predicted_token_ids: 予測されたトークンID
        ground_truth_token_ids: 正解トークンID
        mask: 評価対象のブールマスク

    Returns:
        精度(0.0-1.0)
    """
    correct_preds = (predicted_token_ids == ground_truth_token_ids) & mask
    accuracy = correct_preds.sum().float() / mask.sum().float()
    return accuracy


def compute_actions_l1_loss(
    action_tokenizer,
    predicted_token_ids: torch.Tensor,
    ground_truth_token_ids: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    トークンIDをアクションにデコードしてL1損失を計算.

    Args:
        action_tokenizer: アクショントークナイザー
        predicted_token_ids: 予測されたトークンID
        ground_truth_token_ids: 正解トークンID
        mask: 評価対象のブールマスク

    Returns:
        L1損失
    """
    pred_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(predicted_token_ids[mask].cpu().numpy())
    )
    true_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(ground_truth_token_ids[mask].cpu().numpy())
    )
    l1_loss = torch.nn.functional.l1_loss(pred_continuous_actions, true_continuous_actions)
    return l1_loss
