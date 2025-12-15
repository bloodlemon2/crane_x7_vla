#!/bin/bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop
#
# =============================================================================
# Pi0 Local Sweep Template
# =============================================================================
#
# ローカル環境でW&B Sweepを実行するためのテンプレート。
# Pi0: PaliGemma + Expert Gemma + Flow Matchingベースのモデル
#
# 使用方法:
#   slurm-submit sweep start examples/sweeps/sweep_pi0.yaml \
#     --local \
#     --template examples/templates_local/pi0_sweep.sh \
#     --max-runs 5
#
# =============================================================================

set -euo pipefail

# =============================================================================
# Environment Setup
# =============================================================================
echo "=== Environment Information ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Working Directory: $(pwd)"
echo ""

# W&B Configuration
export WANDB_MODE=online
export WANDB_API_KEY={{WANDB_API_KEY}}
export WANDB_PROJECT={{WANDB_PROJECT}}
export WANDB_ENTITY={{WANDB_ENTITY}}

# Python/CUDA Configuration
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2

# データパス設定
DATA_ROOT=${DATA_ROOT:-{{DATA_ROOT}}}
OUTPUT_DIR=${OUTPUT_DIR:-{{OUTPUT_DIR}}}

# トレーニング設定（Sweepでオーバーライドされない固定パラメータ）
MAX_STEPS=${MAX_STEPS:-{{MAX_STEPS}}}
SAVE_INTERVAL=${SAVE_INTERVAL:-{{SAVE_INTERVAL}}}
EVAL_INTERVAL=${EVAL_INTERVAL:-{{EVAL_INTERVAL}}}
OVERFIT_CHECK_INTERVAL=${OVERFIT_CHECK_INTERVAL:-{{OVERFIT_CHECK_INTERVAL}}}

# デフォルト値
MAX_STEPS=${MAX_STEPS:-10000}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
EVAL_INTERVAL=${EVAL_INTERVAL:-100}
OVERFIT_CHECK_INTERVAL=${OVERFIT_CHECK_INTERVAL:-500}

echo "=== Sweep Configuration ==="
echo "SWEEP_ID: {{SWEEP_ID}}"
echo "RUN_NUMBER: {{RUN_NUMBER}}"
echo "WANDB_ENTITY: {{WANDB_ENTITY}}"
echo "WANDB_PROJECT: {{WANDB_PROJECT}}"
echo ""
echo "=== Data Configuration ==="
echo "DATA_ROOT: ${DATA_ROOT}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo ""

# =============================================================================
# GPU Information
# =============================================================================
echo "=== GPU Information ==="
nvidia-smi || echo "No GPU available"
echo ""

# =============================================================================
# Sweep Agent Execution
# =============================================================================
echo "=== Starting W&B Sweep Agent ==="

# crane_x7_vla の agent コマンドを使用してSweepからパラメータを取得し、トレーニングを実行
# wandb.agent()が内部で呼ばれ、RunがSweepに正しく関連付けられる
python -m crane_x7_vla.training.cli agent pi0 \
    --sweep-id "{{SWEEP_ID}}" \
    --entity "{{WANDB_ENTITY}}" \
    --project "{{WANDB_PROJECT}}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}/checkpoints" \
    --experiment-name "crane_x7_pi0_sweep_local" \
    --max-steps "${MAX_STEPS}"

echo "=== Job Completed ==="
echo "End time: $(date)"
