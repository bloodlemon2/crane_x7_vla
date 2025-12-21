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
cat /etc/os-release 2>/dev/null || true
echo "=========================================="

# 作業ディレクトリに移動
WORKDIR="${SLURM_SUBMIT_DIR:-{{SLURM_REMOTE_WORKDIR}}}"
cd "${WORKDIR}"
echo "Working directory: $(pwd)"

# ログディレクトリを作成
mkdir -p logs

# =============================================================================
# Repository Clone/Update
# =============================================================================
REPO_URL="{{GIT_REPO_URL}}"
REPO_BRANCH="{{GIT_BRANCH}}"
REPO_REF="${GIT_REF:-}"  # オプション: 特定のコミットハッシュやタグ
REPO_DIR="crane_x7_vla"

# デフォルト値
REPO_URL=${REPO_URL:-https://github.com/NOPLAB/crane_x7_vla.git}
REPO_BRANCH=${REPO_BRANCH:-main}

echo "=== Repository Setup ==="
echo "REPO_URL: ${REPO_URL}"
echo "REPO_BRANCH: ${REPO_BRANCH}"
echo "REPO_REF: ${REPO_REF:-HEAD}"
echo ""

if [ -d "${REPO_DIR}/.git" ]; then
    echo "Repository already exists. Updating..."
    cd "${REPO_DIR}"
    git fetch origin
    git checkout "${REPO_BRANCH}"
    git reset --hard "origin/${REPO_BRANCH}"
    if [ -n "${REPO_REF}" ]; then
        echo "Checking out specific ref: ${REPO_REF}"
        git checkout "${REPO_REF}"
    fi
    git submodule update --init --recursive
    cd ..
else
    echo "Cloning repository..."
    git clone --branch "${REPO_BRANCH}" --recursive "${REPO_URL}" "${REPO_DIR}"
    if [ -n "${REPO_REF}" ]; then
        cd "${REPO_DIR}"
        echo "Checking out specific ref: ${REPO_REF}"
        git checkout "${REPO_REF}"
        cd ..
    fi
fi

echo "Repository is ready at: $(pwd)/${REPO_DIR}"
echo "Current commit: $(cd ${REPO_DIR} && git rev-parse HEAD)"
echo ""

# =============================================================================
# Copy Code to /workspace/vla
# =============================================================================
echo "=== Copying Code to /workspace/vla ==="
VLA_WORKSPACE="/workspace/vla"

# コードをDockerイメージのパスにコピー
cp -r "${REPO_DIR}/vla/src/crane_x7_vla" "${VLA_WORKSPACE}/src/"
cp -r "${REPO_DIR}/vla/configs" "${VLA_WORKSPACE}/"

# transformers_replaceをtransformersにパッチ
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
TRANSFORMERS_PATH="/usr/local/lib/python${PYTHON_VERSION}/dist-packages/transformers"
if [ -d "${VLA_WORKSPACE}/src/crane_x7_vla/backends/pi0/models_pytorch/transformers_replace" ]; then
    echo "Patching transformers with transformers_replace..."
    sudo cp -r "${VLA_WORKSPACE}/src/crane_x7_vla/backends/pi0/models_pytorch/transformers_replace"/* "${TRANSFORMERS_PATH}/" || \
    cp -r "${VLA_WORKSPACE}/src/crane_x7_vla/backends/pi0/models_pytorch/transformers_replace"/* "${TRANSFORMERS_PATH}/" 2>/dev/null || true
fi

echo "Code copied to: ${VLA_WORKSPACE}"
echo ""

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
    --training-batch-size "${BATCH_SIZE}" \
    --training-learning-rate "${LEARNING_RATE}" \
    --training-max-steps "${MAX_STEPS}" \
    --training-save-interval "${SAVE_INTERVAL}" \
    --training-eval-interval "${EVAL_INTERVAL}" \
    --training-gradient-checkpointing

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
