#!/bin/bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop
#
# =============================================================================
# Pi0 Sweep Job Template for Slurm
# =============================================================================
#
# このテンプレートはW&B Sweep統合用のSlurmジョブスクリプトです。
# crane_x7_vla.training.cli の agent コマンドを使用して
# Sweepからパラメータを取得し、RunをSweepに正しく関連付けます。
#
# Pi0: PaliGemma + Expert Gemma + Flow Matchingベースのモデル
#
# 使用方法:
#   slurm-submit sweep start examples/sweeps/sweep_pi0.yaml \
#     --template examples/templates/pi0_sweep.sh --max-runs 10
#
# =============================================================================

#SBATCH --job-name={{SLURM_JOB_PREFIX}}_pi0_sweep
#SBATCH --partition={{SLURM_PARTITION}}
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task={{SLURM_CPUS}}
#SBATCH --gpus-per-task={{SLURM_GPUS}}
#SBATCH --time={{SLURM_TIME}}
#SBATCH --output=logs/pi0_sweep_%j.out
#SBATCH --error=logs/pi0_sweep_%j.err

#SBATCH --container={{SLURM_CONTAINER}}

set -euo pipefail

# =============================================================================
# Environment Setup
# =============================================================================
echo "=== Environment Information ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-N/A}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
cat /etc/os-release 2>/dev/null || true
echo ""

# 作業ディレクトリに移動
cd "{{SLURM_CONTAINER_WORKDIR}}"
echo "Working directory: $(pwd)"

# ログディレクトリを作成
mkdir -p logs

# W&B Configuration
export WANDB_MODE=online
export WANDB_API_KEY={{WANDB_API_KEY}}
export WANDB_PROJECT={{WANDB_PROJECT}}
export WANDB_ENTITY={{WANDB_ENTITY}}
export HF_TOKEN={{HF_TOKEN}}

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

pip list

echo "=== Sweep Configuration ==="
echo "SWEEP_ID: {{SWEEP_ID}}"
echo "WANDB_ENTITY: {{WANDB_ENTITY}}"
echo "WANDB_PROJECT: {{WANDB_PROJECT}}"
echo ""
echo "=== Data Configuration ==="
echo "DATA_ROOT: ${DATA_ROOT}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo ""

# =============================================================================
# Sweep Agent Execution
# =============================================================================
echo "=== Starting W&B Sweep Agent ==="
echo "Sweep ID: {{SWEEP_ID}}"
echo "Entity: {{WANDB_ENTITY}}"
echo "Project: {{WANDB_PROJECT}}"
echo ""

# GPUメモリ情報を表示
nvidia-smi || true

# crane_x7_vla の agent コマンドを使用してSweepからパラメータを取得し、トレーニングを実行
# wandb.agent()が内部で呼ばれ、RunがSweepに正しく関連付けられる
python -m crane_x7_vla.training.cli agent pi0 \
    --sweep-id "{{SWEEP_ID}}" \
    --entity "{{WANDB_ENTITY}}" \
    --project "{{WANDB_PROJECT}}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}/checkpoints" \
    --experiment-name "crane_x7_pi0_sweep" \
    --training-max-steps "${MAX_STEPS}" \
    --training-save-interval "${SAVE_INTERVAL}" \
    --training-eval-interval "${EVAL_INTERVAL}" \
    --overfitting-overfit-check-interval "${OVERFIT_CHECK_INTERVAL}" \
    --training-gradient-checkpointing

echo "=== Job Completed ==="
echo "End time: $(date)"
