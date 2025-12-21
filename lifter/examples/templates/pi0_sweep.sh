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

# =============================================================================
# OpenPI Checkpoint Download/Convert
# =============================================================================
# OpenPIの事前学習済みモデル（10k+時間のロボットデータで学習）を使用
OPENPI_CHECKPOINT=${OPENPI_CHECKPOINT:-"pi0_base"}
SKIP_OPENPI_DOWNLOAD=${SKIP_OPENPI_DOWNLOAD:-false}

# キャッシュディレクトリを設定
export CRANE_X7_VLA_CACHE=/root/.cache/crane_x7_vla/openpi

echo "=== OpenPI Checkpoint Setup ==="
echo "OPENPI_CHECKPOINT: ${OPENPI_CHECKPOINT}"
echo "SKIP_OPENPI_DOWNLOAD: ${SKIP_OPENPI_DOWNLOAD}"
echo ""

if [ "${SKIP_OPENPI_DOWNLOAD}" != "true" ]; then
    # Google Cloud SDKをインストール（gsutilがなければ）
    if ! command -v gsutil &> /dev/null; then
        echo "Installing Google Cloud SDK..."
        if [ -f /etc/debian_version ]; then
            # Debian/Ubuntu
            apt-get update && apt-get install -y apt-transport-https ca-certificates gnupg curl
            curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
            echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | tee /etc/apt/sources.list.d/google-cloud-sdk.list
            apt-get update && apt-get install -y google-cloud-sdk
        else
            # フォールバック: pip経由でgoogle-cloud-storage
            pip install google-cloud-storage
        fi
    fi

    # JAX依存関係をインストール（変換に必要）
    # Note: 独自の変換コードを使用するため、OpenPIのモデルコード依存は不要
    echo "Installing dependencies for checkpoint conversion..."
    pip install orbax-checkpoint || true

    # インストール確認
    python3 -c "import orbax.checkpoint; import safetensors; print('All dependencies installed')"

    # Pythonでチェックポイントをダウンロード/変換
    echo "Downloading and converting OpenPI checkpoint: ${OPENPI_CHECKPOINT}..."
    python3 << EOF
import sys
sys.path.insert(0, "${VLA_WORKSPACE}/src")

from crane_x7_vla.backends.pi0.checkpoint_utils import (
    download_checkpoint,
    convert_jax_to_pytorch,
    get_cache_dir,
)
import pathlib

checkpoint_name = "${OPENPI_CHECKPOINT}"
cache_dir = get_cache_dir()

print(f"Cache directory: {cache_dir}")

# JAXチェックポイントをダウンロード
jax_path = download_checkpoint(checkpoint_name)
print(f"JAX checkpoint downloaded to: {jax_path}")

# PyTorchに変換
pytorch_path = cache_dir / "pytorch" / checkpoint_name
config_name = "pi0_base" if "pi0" in checkpoint_name and "pi05" not in checkpoint_name else "pi05_base"
pytorch_path = convert_jax_to_pytorch(jax_path, pytorch_path, config_name=config_name)
print(f"PyTorch checkpoint saved to: {pytorch_path}")
EOF

    if [ $? -eq 0 ]; then
        echo "OpenPI checkpoint ready!"
    else
        echo "Warning: Failed to download/convert OpenPI checkpoint. Continuing without pretrained weights..."
    fi
else
    echo "Skipping OpenPI checkpoint download (SKIP_OPENPI_DOWNLOAD=true)"
fi
echo ""

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
    --training-gradient-checkpointing \
    --openpi-checkpoint "${OPENPI_CHECKPOINT}"

echo "=== Job Completed ==="
echo "End time: $(date)"
