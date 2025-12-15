#!/bin/bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop
#
# =============================================================================
# Pi0 Training Template for Slurm
# =============================================================================
#
# このテンプレートはPi0トレーニングを実行します。
#
# Pi0: PaliGemma + Expert Gemma + Flow Matchingベースのモデル
#
# 使用方法:
#   slurm-submit submit jobs/pi0_train.sh
#
# =============================================================================

#SBATCH --job-name={{SLURM_JOB_PREFIX}}_pi0_train
#SBATCH --partition={{SLURM_PARTITION}}
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task={{SLURM_CPUS}}
#SBATCH --gpus-per-task={{SLURM_GPUS}}
#SBATCH --time={{SLURM_TIME}}
#SBATCH --output=logs/pi0_train_%j.out
#SBATCH --error=logs/pi0_train_%j.err

#SBATCH --container={{SLURM_CONTAINER}}

set -euo pipefail

# =============================================================================
# Environment Setup
# =============================================================================
echo "=========================================="
echo "Pi0 Training"
echo "=========================================="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-N/A}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "=========================================="

# 作業ディレクトリに移動
cd "${SLURM_SUBMIT_DIR:-{{SLURM_REMOTE_WORKDIR}}}"
echo "Working directory: $(pwd)"

# ログディレクトリを作成
mkdir -p logs

# 環境変数の設定
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2

# W&B設定（オプション）
export WANDB_API_KEY={{WANDB_API_KEY}}
export WANDB_PROJECT={{WANDB_PROJECT}}
export WANDB_ENTITY={{WANDB_ENTITY}}
export HF_TOKEN={{HF_TOKEN}}
export WANDB_MODE=${WANDB_MODE:-online}

# データパス設定
DATA_ROOT=${DATA_ROOT:-{{DATA_ROOT}}}
OUTPUT_DIR=${OUTPUT_DIR:-{{OUTPUT_DIR}}}


# トレーニング設定（Sweepでオーバーライド可能）
BATCH_SIZE=${BATCH_SIZE:-{{batch_size}}}
LEARNING_RATE=${LEARNING_RATE:-{{learning_rate}}}
MAX_STEPS=${MAX_STEPS:-{{MAX_STEPS}}}
SAVE_INTERVAL=${SAVE_INTERVAL:-{{SAVE_INTERVAL}}}
EVAL_INTERVAL=${EVAL_INTERVAL:-{{EVAL_INTERVAL}}}

# デフォルト値（テンプレートプレースホルダが未置換の場合）
BATCH_SIZE=${BATCH_SIZE:-4}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
MAX_STEPS=${MAX_STEPS:-10000}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
EVAL_INTERVAL=${EVAL_INTERVAL:-100}

echo ""
echo "=== Configuration ==="
echo "DATA_ROOT: ${DATA_ROOT}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "LEARNING_RATE: ${LEARNING_RATE}"
echo "MAX_STEPS: ${MAX_STEPS}"
echo ""

# =============================================================================
# Pi0 Training
# =============================================================================
echo "=========================================="
echo "Pi0 Training"
echo "=========================================="

# GPUメモリ情報を表示
nvidia-smi || true

echo ""
echo "Starting Pi0 training..."
echo "  Batch size: ${BATCH_SIZE}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Max steps: ${MAX_STEPS}"
echo "  Save interval: ${SAVE_INTERVAL}"
echo "  Eval interval: ${EVAL_INTERVAL}"
echo ""

# Pi0トレーニングを実行
python -m crane_x7_vla.training.cli train pi0 \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}/checkpoints" \
    --experiment-name "crane_x7_pi0" \
    --batch-size "${BATCH_SIZE}" \
    --learning-rate "${LEARNING_RATE}" \
    --max-steps "${MAX_STEPS}"

TRAIN_EXIT_CODE=$?

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=========================================="
echo "Training Completed"
echo "=========================================="
echo "Exit code: ${TRAIN_EXIT_CODE}"
echo "End time: $(date)"
echo ""
echo "Outputs:"
echo "  Checkpoints: ${OUTPUT_DIR}/checkpoints"
echo "=========================================="

exit ${TRAIN_EXIT_CODE}
