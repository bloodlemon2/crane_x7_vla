#!/bin/bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop
#
# =============================================================================
# Pi0 Local Sweep Template (Docker)
# =============================================================================
#
# ローカル環境でW&B SweepをDockerコンテナ内で実行するためのテンプレート。
# Pi0: PaliGemma + Expert Gemma + Flow Matchingベースのモデル
#
# 使用方法:
#   lifter sweep start examples/sweeps/sweep_pi0.yaml \
#     --local \
#     --template examples/templates_local/pi0_sweep.sh \
#     --max-runs 5
#
# 必要な環境変数:
#   - SLURM_CONTAINER: Dockerイメージ名
#   - DATA_ROOT: データディレクトリ（ホスト側パス）
#   - OUTPUT_DIR: 出力ディレクトリ（ホスト側パス）
#   - WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT: W&B設定
#   - HF_TOKEN: Hugging Face APIトークン（gemmaモデル用）
#
# オプション環境変数:
#   - OPENPI_CACHE_DIR: OpenPIチェックポイントキャッシュ（デフォルト: ~/.cache/crane_x7_vla/openpi）
#   - OPENPI_CHECKPOINT: 使用するOpenPIチェックポイント名（デフォルト: pi0_base）
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

# OpenPIキャッシュディレクトリ（ホスト側）
OPENPI_CACHE_DIR=${OPENPI_CACHE_DIR:-${HOME}/.cache/crane_x7_vla/openpi}
CONTAINER_CACHE_DIR=/root/.cache/crane_x7_vla/openpi

# トレーニング設定（Sweepでオーバーライドされない固定パラメータ）
MAX_STEPS=${MAX_STEPS:-{{MAX_STEPS}}}
SAVE_INTERVAL=${SAVE_INTERVAL:-{{SAVE_INTERVAL}}}
EVAL_INTERVAL=${EVAL_INTERVAL:-{{EVAL_INTERVAL}}}
OVERFIT_CHECK_INTERVAL=${OVERFIT_CHECK_INTERVAL:-{{OVERFIT_CHECK_INTERVAL}}}

# OpenPIチェックポイント設定
OPENPI_CHECKPOINT=${OPENPI_CHECKPOINT:-"pi0_base"}

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
# キャッシュディレクトリを作成
mkdir -p "${OPENPI_CACHE_DIR}"

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
    -e CRANE_X7_VLA_CACHE="${CONTAINER_CACHE_DIR}" \
    -e OPENPI_CHECKPOINT="${OPENPI_CHECKPOINT}" \
    -v "${DATA_ROOT}:${CONTAINER_DATA_ROOT}:ro" \
    -v "${OUTPUT_DIR}:${CONTAINER_OUTPUT_DIR}" \
    -v "${OPENPI_CACHE_DIR}:${CONTAINER_CACHE_DIR}" \
    "${CONTAINER_IMAGE}" \
    bash -c '
set -euo pipefail

echo "=== OpenPI Checkpoint Setup ==="
echo "OPENPI_CHECKPOINT: ${OPENPI_CHECKPOINT}"

# JAX依存関係をインストール（変換に必要）
echo "Installing dependencies for checkpoint conversion..."
pip install orbax-checkpoint || true

# インストール確認
python3 -c "import orbax.checkpoint; import safetensors; print(\"All dependencies installed\")"

# Pythonでチェックポイントをダウンロード/変換
echo "Downloading and converting OpenPI checkpoint: ${OPENPI_CHECKPOINT}..."
python3 << EOF
from crane_x7_vla.backends.pi0.checkpoint_utils import (
    download_checkpoint,
    convert_jax_to_pytorch,
    get_cache_dir,
)

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

echo "OpenPI checkpoint ready!"
echo ""

# トレーニング実行
python -m crane_x7_vla.training.cli agent pi0 \
    --sweep-id "{{SWEEP_ID}}" \
    --entity "'"${WANDB_ENTITY}"'" \
    --project "'"${WANDB_PROJECT}"'" \
    --data-root "'"${CONTAINER_DATA_ROOT}"'" \
    --output-dir "'"${CONTAINER_OUTPUT_DIR}"'/checkpoints" \
    --experiment-name "crane_x7_pi0_sweep_local" \
    --training-max-steps "'"${MAX_STEPS}"'" \
    --training-save-interval "'"${SAVE_INTERVAL}"'" \
    --training-eval-interval "'"${EVAL_INTERVAL}"'" \
    --overfitting-overfit-check-interval "'"${OVERFIT_CHECK_INTERVAL}"'" \
    --training-gradient-checkpointing \
    --openpi-checkpoint "${OPENPI_CHECKPOINT}"
'

echo "=== Job Completed ==="
echo "End time: $(date)"
