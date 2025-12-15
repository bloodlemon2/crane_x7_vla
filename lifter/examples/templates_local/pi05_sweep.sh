#!/bin/bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop
#
# =============================================================================
# Pi0.5 Local Sweep Template (Docker)
# =============================================================================
#
# ローカル環境でW&B SweepをDockerコンテナ内で実行するためのテンプレート。
# Pi0.5: Pi0 + adaRMSNorm + Discrete State入力
# Pi0よりmax_token_lenが大きい（200 vs 48）ため、メモリ使用量が高い
#
# 使用方法:
#   lifter sweep start examples/sweeps/sweep_pi05.yaml \
#     --local \
#     --template examples/templates_local/pi05_sweep.sh \
#     --max-runs 5
#
# 必要な環境変数:
#   - SLURM_CONTAINER: Dockerイメージ名
#   - DATA_ROOT: データディレクトリ（ホスト側パス）
#   - OUTPUT_DIR: 出力ディレクトリ（ホスト側パス）
#   - WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT: W&B設定
#   - HF_TOKEN: Hugging Face APIトークン（gemmaモデル用）
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

# コンテナイメージ
CONTAINER_IMAGE={{SLURM_CONTAINER}}

# W&B Configuration
WANDB_API_KEY={{WANDB_API_KEY}}
WANDB_PROJECT={{WANDB_PROJECT}}
WANDB_ENTITY={{WANDB_ENTITY}}

# Hugging Face Configuration
HF_TOKEN={{HF_TOKEN}}

# データパス設定（ホスト側）
DATA_ROOT=${DATA_ROOT:-{{DATA_ROOT}}}
OUTPUT_DIR=${OUTPUT_DIR:-{{OUTPUT_DIR}}}

# コンテナ内パス
CONTAINER_DATA_ROOT=/workspace/data
CONTAINER_OUTPUT_DIR=/workspace/outputs

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
echo "WANDB_ENTITY: ${WANDB_ENTITY}"
echo "WANDB_PROJECT: ${WANDB_PROJECT}"
echo ""
echo "=== Container Configuration ==="
echo "CONTAINER_IMAGE: ${CONTAINER_IMAGE}"
echo ""
echo "=== Data Configuration ==="
echo "DATA_ROOT (host): ${DATA_ROOT}"
echo "OUTPUT_DIR (host): ${OUTPUT_DIR}"
echo "DATA_ROOT (container): ${CONTAINER_DATA_ROOT}"
echo "OUTPUT_DIR (container): ${CONTAINER_OUTPUT_DIR}"
echo ""

# =============================================================================
# GPU Information
# =============================================================================
echo "=== GPU Information ==="
nvidia-smi || echo "No GPU available"
echo ""

# =============================================================================
# Sweep Agent Execution (Docker)
# =============================================================================
echo "=== Starting W&B Sweep Agent in Docker ==="

# crane_x7_vla の agent コマンドを使用してSweepからパラメータを取得し、トレーニングを実行
# wandb.agent()が内部で呼ばれ、RunがSweepに正しく関連付けられる
docker run --rm \
    --gpus all \
    --shm-size=16g \
    -e WANDB_MODE=online \
    -e WANDB_API_KEY="${WANDB_API_KEY}" \
    -e WANDB_PROJECT="${WANDB_PROJECT}" \
    -e WANDB_ENTITY="${WANDB_ENTITY}" \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e PYTHONUNBUFFERED=1 \
    -e TF_CPP_MIN_LOG_LEVEL=2 \
    -v "${DATA_ROOT}:${CONTAINER_DATA_ROOT}:ro" \
    -v "${OUTPUT_DIR}:${CONTAINER_OUTPUT_DIR}" \
    "${CONTAINER_IMAGE}" \
    python -m crane_x7_vla.training.cli agent pi0.5 \
        --sweep-id "{{SWEEP_ID}}" \
        --entity "${WANDB_ENTITY}" \
        --project "${WANDB_PROJECT}" \
        --data-root "${CONTAINER_DATA_ROOT}" \
        --output-dir "${CONTAINER_OUTPUT_DIR}/checkpoints" \
        --experiment-name "crane_x7_pi05_sweep_local" \
        --max-steps "${MAX_STEPS}"

echo "=== Job Completed ==="
echo "End time: $(date)"
