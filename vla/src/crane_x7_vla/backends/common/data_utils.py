# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""
データユーティリティ.

prismatic/util/data_utils.pyとprismatic/vla/datasets/rlds/utils/data_utils.pyからの移植。
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

logger = logging.getLogger(__name__)


@dataclass
class PaddedCollatorForActionPrediction:
    """VLAトレーニング用のパディングコレーター."""

    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        dataset_names = [instance["dataset_name"] for instance in instances] if "dataset_name" in instances[0] else None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all(pv is not None for pv in pixel_values), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        output = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output


def save_dataset_statistics(dataset_statistics: dict[str, Any], run_dir: Path | str) -> None:
    """データセット統計をJSONに保存.

    Args:
        dataset_statistics: データセット統計の辞書
        run_dir: 保存先ディレクトリ
    """
    run_dir = Path(run_dir)
    out_path = run_dir / "dataset_statistics.json"

    # Deep copy to avoid modifying original
    stats_copy = {}
    for dataset_name, stats in dataset_statistics.items():
        stats_copy[dataset_name] = {}
        for key, value in stats.items():
            if isinstance(value, dict):
                stats_copy[dataset_name][key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        stats_copy[dataset_name][key][k] = v.tolist()
                    else:
                        stats_copy[dataset_name][key][k] = v
            elif isinstance(value, np.ndarray):
                stats_copy[dataset_name][key] = value.item() if value.ndim == 0 else value.tolist()
            else:
                stats_copy[dataset_name][key] = value

    with out_path.open("w") as f_json:
        json.dump(stats_copy, f_json, indent=2)

    logger.info(f"Saved dataset statistics file at path {out_path}")


__all__ = [
    "IGNORE_INDEX",
    "PaddedCollatorForActionPrediction",
    "save_dataset_statistics",
]
