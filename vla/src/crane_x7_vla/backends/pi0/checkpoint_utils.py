# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Utilities for downloading and converting OpenPI checkpoints.

This module provides standalone JAX→PyTorch checkpoint conversion
without requiring the full OpenPI model code and its dependencies.
"""

import dataclasses
import json
import logging
import os
import pathlib
import shutil
import subprocess
from typing import Literal

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors


logger = logging.getLogger(__name__)

# OpenPI checkpoint URLs
OPENPI_CHECKPOINTS = {
    # Base models (for fine-tuning)
    "pi0_base": "gs://openpi-assets/checkpoints/pi0_base",
    "pi0_fast_base": "gs://openpi-assets/checkpoints/pi0_fast_base",
    "pi05_base": "gs://openpi-assets/checkpoints/pi05_base",
    # Fine-tuned models
    "pi0_droid": "gs://openpi-assets/checkpoints/pi0_droid",
    "pi0_fast_droid": "gs://openpi-assets/checkpoints/pi0_fast_droid",
    "pi05_droid": "gs://openpi-assets/checkpoints/pi05_droid",
    "pi05_libero": "gs://openpi-assets/checkpoints/pi05_libero",
    "pi0_aloha_towel": "gs://openpi-assets/checkpoints/pi0_aloha_towel",
    "pi0_aloha_tupperware": "gs://openpi-assets/checkpoints/pi0_aloha_tupperware",
    "pi0_aloha_pen_uncap": "gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap",
}

# Default cache directory
DEFAULT_CACHE_DIR = pathlib.Path.home() / ".cache" / "crane_x7_vla" / "openpi"


# =============================================================================
# Model Configuration (hardcoded to avoid OpenPI dependencies)
# =============================================================================


@dataclasses.dataclass
class VisionConfig:
    """SigLIP Vision configuration."""

    hidden_size: int = 1152
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    intermediate_size: int = 4304
    patch_size: int = 14
    projection_dim: int = 2048


@dataclasses.dataclass
class TextConfig:
    """Gemma text configuration for PaliGemma."""

    hidden_size: int = 2048
    num_hidden_layers: int = 18
    num_attention_heads: int = 8
    head_dim: int = 256
    intermediate_size: int = 16384


@dataclasses.dataclass
class GemmaExpertConfig:
    """Gemma 300M expert configuration."""

    width: int = 1024
    depth: int = 18
    mlp_dim: int = 4096
    num_heads: int = 8
    num_kv_heads: int = 1
    head_dim: int = 256

    @property
    def hidden_size(self) -> int:
        return self.width

    @property
    def num_hidden_layers(self) -> int:
        return self.depth

    @property
    def num_attention_heads(self) -> int:
        return self.num_heads


@dataclasses.dataclass
class PaliGemmaConfig:
    """Combined PaliGemma configuration."""

    vision_config: VisionConfig = dataclasses.field(default_factory=VisionConfig)
    text_config: TextConfig = dataclasses.field(default_factory=TextConfig)


# =============================================================================
# Utility Functions
# =============================================================================


def get_cache_dir() -> pathlib.Path:
    """Get the cache directory for OpenPI checkpoints."""
    cache_dir = pathlib.Path(os.getenv("CRANE_X7_VLA_CACHE", str(DEFAULT_CACHE_DIR)))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """Flatten a nested dictionary with separator-joined keys."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _unflatten_dict(d: dict, sep: str = "/") -> dict:
    """Unflatten a flat dictionary with separator-joined keys."""
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


# =============================================================================
# Download Functions
# =============================================================================


def download_checkpoint(
    checkpoint_name: str,
    force_download: bool = False,
) -> pathlib.Path:
    """Download an OpenPI checkpoint from GCS.

    Args:
        checkpoint_name: Name of the checkpoint (e.g., "pi0_base", "pi05_base")
        force_download: If True, re-download even if cached

    Returns:
        Path to the downloaded checkpoint directory
    """
    if checkpoint_name not in OPENPI_CHECKPOINTS:
        raise ValueError(f"Unknown checkpoint: {checkpoint_name}. " f"Available: {list(OPENPI_CHECKPOINTS.keys())}")

    gcs_url = OPENPI_CHECKPOINTS[checkpoint_name]
    cache_dir = get_cache_dir()
    local_path = cache_dir / "jax" / checkpoint_name

    if local_path.exists() and not force_download:
        logger.info(f"Using cached checkpoint: {local_path}")
        return local_path

    logger.info(f"Downloading {checkpoint_name} from {gcs_url}...")

    # Use gsutil for downloading
    try:
        if local_path.exists():
            shutil.rmtree(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            ["gsutil", "-m", "cp", "-r", gcs_url, str(local_path.parent)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"gsutil failed: {result.stderr}")

        # gsutil creates a subdirectory with the checkpoint name
        downloaded_path = local_path.parent / checkpoint_name.split("/")[-1]
        if downloaded_path != local_path and downloaded_path.exists():
            shutil.move(str(downloaded_path), str(local_path))

        logger.info(f"Downloaded to: {local_path}")
        return local_path

    except FileNotFoundError:
        logger.error("gsutil not found. Please install Google Cloud SDK: " "https://cloud.google.com/sdk/docs/install")
        raise


# =============================================================================
# JAX Checkpoint Loading (minimal dependencies)
# =============================================================================


def _load_jax_checkpoint(checkpoint_path: pathlib.Path) -> dict:
    """Load JAX checkpoint using orbax.

    Args:
        checkpoint_path: Path to the checkpoint directory

    Returns:
        Flattened parameter dictionary
    """
    import orbax.checkpoint as ocp

    # Try with 'params' subdirectory first
    params_path = checkpoint_path / "params"
    if not params_path.exists():
        params_path = checkpoint_path

    # Use StandardCheckpointer for simpler restoration
    checkpointer = ocp.StandardCheckpointer()

    # Restore without specifying item structure (let orbax infer it)
    try:
        # Try restoring directly
        restored = checkpointer.restore(params_path)
    except Exception as e:
        logger.warning(f"StandardCheckpointer failed: {e}, trying PyTreeCheckpointer")
        # Fallback to PyTreeCheckpointer with abstract restore
        with ocp.PyTreeCheckpointer() as ckptr:
            restored = ckptr.restore(params_path)

    # Handle different checkpoint structures
    params = restored.get("params", restored) if isinstance(restored, dict) else restored

    # Flatten and remove 'value' suffix if present
    flat_params = _flatten_dict(params, sep="/")

    # Check if keys end with 'value' (nnx.State format)
    cleaned_params = {}
    for key, value in flat_params.items():
        new_key = key[: -len("/value")] if key.endswith("/value") else key
        # Convert to numpy if it's a JAX array
        if hasattr(value, "numpy") or not isinstance(value, np.ndarray):
            value = np.array(value)
        cleaned_params[new_key] = value

    return cleaned_params


# =============================================================================
# PaliGemma Weight Conversion
# =============================================================================


def _convert_paligemma_weights(
    state_dict: dict[str, np.ndarray],
    config: PaliGemmaConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
    """Convert PaliGemma JAX parameters to PyTorch format.

    Args:
        state_dict: Flattened JAX state dict
        config: PaliGemma configuration

    Returns:
        Tuple of (converted PyTorch state dict, remaining expert params)
    """
    # Check for /value suffix pattern
    suffix = "/value" if "img/embedding/kernel/value" in state_dict else ""

    # Vision encoder: patch embeddings
    jax_key = f"img/embedding/kernel{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose(3, 2, 0, 1)

    jax_key = f"img/embedding/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # Positional embeddings
    jax_key = f"img/pos_embedding{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_embedding.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).reshape(-1, config.vision_config.hidden_size)

    # Vision encoder layers
    encoderblock_layernorm0_scale = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/scale{suffix}")
    encoderblock_layernorm0_bias = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/bias{suffix}")
    encoderblock_layernorm1_scale = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/scale{suffix}")
    encoderblock_layernorm1_bias = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/bias{suffix}")

    encoderblock_mlp_dense0_kernel = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel{suffix}")
    encoderblock_mlp_dense0_bias = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias{suffix}")
    encoderblock_mlp_dense1_kernel = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel{suffix}")
    encoderblock_mlp_dense1_bias = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias{suffix}")

    encoderblock_attention_0_key_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel{suffix}"
    )
    encoderblock_attention_0_key_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias{suffix}"
    )
    encoderblock_attention_0_value_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel{suffix}"
    )
    encoderblock_attention_0_value_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias{suffix}"
    )
    encoderblock_attention_0_query_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel{suffix}"
    )
    encoderblock_attention_0_query_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias{suffix}"
    )
    encoderblock_attention_0_out_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel{suffix}"
    )
    encoderblock_attention_0_out_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias{suffix}"
    )

    hidden_size = config.vision_config.hidden_size
    for i in range(config.vision_config.num_hidden_layers):
        prefix = f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}"

        state_dict[f"{prefix}.layer_norm1.weight"] = encoderblock_layernorm0_scale[i].transpose()
        state_dict[f"{prefix}.layer_norm1.bias"] = encoderblock_layernorm0_bias[i]
        state_dict[f"{prefix}.layer_norm2.weight"] = encoderblock_layernorm1_scale[i].transpose()
        state_dict[f"{prefix}.layer_norm2.bias"] = encoderblock_layernorm1_bias[i]

        state_dict[f"{prefix}.mlp.fc1.weight"] = encoderblock_mlp_dense0_kernel[i].transpose()
        state_dict[f"{prefix}.mlp.fc1.bias"] = encoderblock_mlp_dense0_bias[i]
        state_dict[f"{prefix}.mlp.fc2.weight"] = encoderblock_mlp_dense1_kernel[i].transpose()
        state_dict[f"{prefix}.mlp.fc2.bias"] = encoderblock_mlp_dense1_bias[i]

        state_dict[f"{prefix}.self_attn.k_proj.weight"] = (
            encoderblock_attention_0_key_kernel[i].reshape(-1, hidden_size).transpose()
        )
        state_dict[f"{prefix}.self_attn.k_proj.bias"] = (
            encoderblock_attention_0_key_bias[i].reshape(-1, hidden_size).reshape(-1)
        )
        state_dict[f"{prefix}.self_attn.v_proj.weight"] = (
            encoderblock_attention_0_value_kernel[i].reshape(-1, hidden_size).transpose()
        )
        state_dict[f"{prefix}.self_attn.v_proj.bias"] = (
            encoderblock_attention_0_value_bias[i].reshape(-1, hidden_size).reshape(-1)
        )
        state_dict[f"{prefix}.self_attn.q_proj.weight"] = (
            encoderblock_attention_0_query_kernel[i].reshape(-1, hidden_size).transpose()
        )
        state_dict[f"{prefix}.self_attn.q_proj.bias"] = (
            encoderblock_attention_0_query_bias[i].reshape(-1, hidden_size).reshape(-1)
        )
        state_dict[f"{prefix}.self_attn.out_proj.weight"] = (
            encoderblock_attention_0_out_kernel[i].reshape(-1, hidden_size).transpose()
        )
        state_dict[f"{prefix}.self_attn.out_proj.bias"] = (
            encoderblock_attention_0_out_bias[i].reshape(-1, hidden_size).reshape(-1)
        )

    # Vision encoder post-layernorm
    jax_key = f"img/Transformer/encoder_norm/scale{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose()

    jax_key = f"img/Transformer/encoder_norm/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # Multimodal projector
    jax_key = f"img/head/kernel{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose()

    jax_key = f"img/head/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # Text decoder (Gemma) - embeddings
    jax_key = f"llm/embedder/input_embedding{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # Text decoder layers
    llm_attention_attn_vec_einsum = state_dict.pop(f"llm/layers/attn/attn_vec_einsum/w{suffix}")
    llm_attention_kv_einsum = state_dict.pop(f"llm/layers/attn/kv_einsum/w{suffix}")
    llm_attention_q_einsum = state_dict.pop(f"llm/layers/attn/q_einsum/w{suffix}")

    llm_mlp_gating_einsum = state_dict.pop(f"llm/layers/mlp/gating_einsum{suffix}")
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp/linear{suffix}")

    llm_input_layernorm = state_dict.pop(f"llm/layers/pre_attention_norm/scale{suffix}")
    llm_post_attention_layernorm = state_dict.pop(f"llm/layers/pre_ffw_norm/scale{suffix}")

    text_hidden_size = config.text_config.hidden_size
    text_num_heads = config.text_config.num_attention_heads
    text_head_dim = config.text_config.head_dim

    for i in range(config.text_config.num_hidden_layers):
        prefix = f"paligemma_with_expert.paligemma.model.language_model.layers.{i}"

        q_proj_weight_reshaped = (
            llm_attention_q_einsum[i].transpose(0, 2, 1).reshape(text_num_heads * text_head_dim, text_hidden_size)
        )
        state_dict[f"{prefix}.self_attn.q_proj.weight"] = q_proj_weight_reshaped

        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[f"{prefix}.self_attn.k_proj.weight"] = k_proj_weight_reshaped

        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[f"{prefix}.self_attn.v_proj.weight"] = v_proj_weight_reshaped

        o_proj_weight_reshaped = (
            llm_attention_attn_vec_einsum[i]
            .transpose(2, 0, 1)
            .reshape(text_num_heads * text_head_dim, text_hidden_size)
        )
        state_dict[f"{prefix}.self_attn.o_proj.weight"] = o_proj_weight_reshaped

        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[f"{prefix}.mlp.gate_proj.weight"] = gate_proj_weight.transpose()

        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[f"{prefix}.mlp.up_proj.weight"] = up_proj_weight.transpose()

        state_dict[f"{prefix}.mlp.down_proj.weight"] = llm_mlp_linear[i].transpose()

        state_dict[f"{prefix}.input_layernorm.weight"] = llm_input_layernorm[i]
        state_dict[f"{prefix}.post_attention_layernorm.weight"] = llm_post_attention_layernorm[i]

    # Final norm
    jax_key = f"llm/final_norm/scale{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.language_model.norm.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # Separate expert-related keys
    expert_keys = [
        f"llm/final_norm_1/scale{suffix}",
        f"llm/final_norm_1/Dense_0/bias{suffix}",
        f"llm/final_norm_1/Dense_0/kernel{suffix}",
        f"llm/layers/attn/attn_vec_einsum_1/w{suffix}",
        f"llm/layers/attn/kv_einsum_1/w{suffix}",
        f"llm/layers/attn/q_einsum_1/w{suffix}",
        f"llm/layers/mlp_1/gating_einsum{suffix}",
        f"llm/layers/mlp_1/linear{suffix}",
        f"llm/layers/pre_attention_norm_1/scale{suffix}",
        f"llm/layers/pre_attention_norm_1/Dense_0/bias{suffix}",
        f"llm/layers/pre_attention_norm_1/Dense_0/kernel{suffix}",
        f"llm/layers/pre_ffw_norm_1/scale{suffix}",
        f"llm/layers/pre_ffw_norm_1/Dense_0/bias{suffix}",
        f"llm/layers/pre_ffw_norm_1/Dense_0/kernel{suffix}",
    ]

    expert_dict = {}
    final_state_dict = {}

    for key, value in state_dict.items():
        if key in expert_keys:
            expert_dict[key] = value
        else:
            final_state_dict[key] = torch.from_numpy(np.array(value))

    return final_state_dict, expert_dict


# =============================================================================
# Expert Gemma Weight Conversion
# =============================================================================


def _convert_gemma_expert_weights(
    state_dict: dict[str, np.ndarray],
    config: GemmaExpertConfig,
    pi05: bool = False,
) -> dict[str, torch.Tensor]:
    """Convert Gemma expert JAX parameters to PyTorch format.

    Args:
        state_dict: Remaining state dict with expert params
        config: Gemma expert configuration
        pi05: Whether this is a Pi0.5 model (uses adaptive normalization)

    Returns:
        Converted PyTorch state dict
    """
    num_expert = 1
    suffix = "/value" if f"llm/layers/attn/attn_vec_einsum_{num_expert}/w/value" in state_dict else ""

    llm_attention_attn_vec_einsum = state_dict.pop(f"llm/layers/attn/attn_vec_einsum_{num_expert}/w{suffix}")
    llm_attention_kv_einsum = state_dict.pop(f"llm/layers/attn/kv_einsum_{num_expert}/w{suffix}")
    llm_attention_q_einsum = state_dict.pop(f"llm/layers/attn/q_einsum_{num_expert}/w{suffix}")

    llm_mlp_gating_einsum = state_dict.pop(f"llm/layers/mlp_{num_expert}/gating_einsum{suffix}")
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp_{num_expert}/linear{suffix}")

    # Handle normalization layers based on model type
    if pi05:
        # Pi0.5 with adaptive normalization (Dense layers)
        llm_input_layernorm_bias = state_dict.pop(f"llm/layers/pre_attention_norm_{num_expert}/Dense_0/bias{suffix}")
        llm_post_attention_layernorm_bias = state_dict.pop(f"llm/layers/pre_ffw_norm_{num_expert}/Dense_0/bias{suffix}")
        llm_input_layernorm_kernel = state_dict.pop(
            f"llm/layers/pre_attention_norm_{num_expert}/Dense_0/kernel{suffix}"
        )
        llm_post_attention_layernorm_kernel = state_dict.pop(
            f"llm/layers/pre_ffw_norm_{num_expert}/Dense_0/kernel{suffix}"
        )
    else:
        # Regular Pi0 with standard RMSNorm (scale only)
        llm_input_layernorm = state_dict.pop(f"llm/layers/pre_attention_norm_{num_expert}/scale{suffix}")
        llm_post_attention_layernorm = state_dict.pop(f"llm/layers/pre_ffw_norm_{num_expert}/scale{suffix}")

    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = config.head_dim

    for i in range(config.num_hidden_layers):
        prefix = f"paligemma_with_expert.gemma_expert.model.layers.{i}"

        q_proj_weight_reshaped = llm_attention_q_einsum[i].transpose(0, 2, 1).reshape(num_heads * head_dim, hidden_size)
        state_dict[f"{prefix}.self_attn.q_proj.weight"] = q_proj_weight_reshaped

        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[f"{prefix}.self_attn.k_proj.weight"] = k_proj_weight_reshaped

        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[f"{prefix}.self_attn.v_proj.weight"] = v_proj_weight_reshaped

        o_proj_weight_reshaped = (
            llm_attention_attn_vec_einsum[i].reshape(num_heads * head_dim, hidden_size).transpose(1, 0)
        )
        state_dict[f"{prefix}.self_attn.o_proj.weight"] = o_proj_weight_reshaped

        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[f"{prefix}.mlp.gate_proj.weight"] = gate_proj_weight.transpose()

        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[f"{prefix}.mlp.up_proj.weight"] = up_proj_weight.transpose()

        state_dict[f"{prefix}.mlp.down_proj.weight"] = llm_mlp_linear[i].transpose()

        if pi05:
            # Pi0.5 with adaptive normalization - use Dense layer parameters
            state_dict[f"{prefix}.input_layernorm.dense.bias"] = llm_input_layernorm_bias[i]
            state_dict[f"{prefix}.post_attention_layernorm.dense.bias"] = llm_post_attention_layernorm_bias[i]
            state_dict[f"{prefix}.input_layernorm.dense.weight"] = llm_input_layernorm_kernel[i].transpose()
            state_dict[f"{prefix}.post_attention_layernorm.dense.weight"] = llm_post_attention_layernorm_kernel[
                i
            ].transpose()
        else:
            # Regular Pi0 with standard RMSNorm
            state_dict[f"{prefix}.input_layernorm.weight"] = llm_input_layernorm[i]
            state_dict[f"{prefix}.post_attention_layernorm.weight"] = llm_post_attention_layernorm[i]

    # Final norm layer
    if pi05:
        # Pi0.5 with adaptive normalization
        final_norm_bias = state_dict.pop(f"llm/final_norm_{num_expert}/Dense_0/bias{suffix}")
        final_norm_kernel = state_dict.pop(f"llm/final_norm_{num_expert}/Dense_0/kernel{suffix}")
        state_dict["paligemma_with_expert.gemma_expert.model.norm.dense.bias"] = final_norm_bias
        state_dict["paligemma_with_expert.gemma_expert.model.norm.dense.weight"] = final_norm_kernel.transpose()
    else:
        # Regular Pi0 with standard RMSNorm
        state_dict["paligemma_with_expert.gemma_expert.model.norm.weight"] = state_dict.pop(
            f"llm/final_norm_{num_expert}/scale{suffix}"
        )

    # Convert all remaining numpy arrays to tensors
    final_state_dict = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            final_state_dict[key] = torch.from_numpy(np.array(value))
        else:
            final_state_dict[key] = value

    return final_state_dict


# =============================================================================
# Projection Layer Conversion
# =============================================================================


def _convert_projection_params(
    params: dict[str, np.ndarray],
    pi05: bool = False,
) -> dict[str, torch.Tensor]:
    """Convert projection layer parameters.

    Args:
        params: Raw params dict with projection layers
        pi05: Whether this is a Pi0.5 model

    Returns:
        Converted projection parameters
    """
    if pi05:
        keys = [
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
            "time_mlp_out",
        ]
    else:
        keys = [
            "state_proj",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ]

    projection_params = {}

    for key in keys:
        if key not in params:
            logger.warning(f"Projection key {key} not found in params")
            continue

        proj_data = params[key]

        # Handle nested dict structure
        if isinstance(proj_data, dict):
            kernel_params = proj_data.get("kernel", {})
            bias_params = proj_data.get("bias", {})

            if isinstance(kernel_params, dict) and "value" in kernel_params:
                weight = kernel_params["value"]
                bias = bias_params["value"]
            else:
                weight = kernel_params
                bias = bias_params
        else:
            # Skip if not the expected structure
            logger.warning(f"Unexpected structure for projection key {key}")
            continue

        # Convert to numpy if needed
        if hasattr(weight, "numpy"):
            weight = np.array(weight)
        if hasattr(bias, "numpy"):
            bias = np.array(bias)

        projection_params[f"{key}.weight"] = torch.from_numpy(np.array(weight)).T
        projection_params[f"{key}.bias"] = torch.from_numpy(np.array(bias))

    return projection_params


# =============================================================================
# Main Conversion Function
# =============================================================================


def convert_jax_to_pytorch(
    jax_checkpoint_path: pathlib.Path,
    output_path: pathlib.Path | None = None,
    config_name: str = "pi0_base",
    precision: Literal["float32", "bfloat16"] = "bfloat16",
) -> pathlib.Path:
    """Convert a JAX checkpoint to PyTorch format.

    This is a standalone conversion that doesn't require OpenPI model code.
    Only requires: orbax-checkpoint, numpy, torch, safetensors

    Args:
        jax_checkpoint_path: Path to the JAX checkpoint
        output_path: Path to save the converted checkpoint (default: auto-generated)
        config_name: Config name to determine model type ("pi0_base" or "pi05_base")
        precision: Model precision

    Returns:
        Path to the converted PyTorch checkpoint
    """
    if output_path is None:
        cache_dir = get_cache_dir()
        output_path = cache_dir / "pytorch" / jax_checkpoint_path.name

    if output_path.exists() and (output_path / "model.safetensors").exists():
        logger.info(f"Using cached PyTorch checkpoint: {output_path}")
        return output_path

    logger.info(f"Converting {jax_checkpoint_path} to PyTorch format...")
    logger.info(f"Config: {config_name}, Precision: {precision}")

    pi05 = "pi05" in config_name

    # Load JAX checkpoint
    raw_params = _load_jax_checkpoint(jax_checkpoint_path)

    # Separate PaliGemma params and projection params
    paligemma_params = {}
    projection_params_raw = {}

    for key, value in raw_params.items():
        if key.startswith("PaliGemma/"):
            # Remove "PaliGemma/" prefix and flatten
            new_key = key[len("PaliGemma/") :]
            paligemma_params[new_key] = value
        elif "/" not in key or key.split("/")[0] in [
            "state_proj",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
            "time_mlp_in",
            "time_mlp_out",
        ]:
            # Projection params - need to unflatten
            parts = key.split("/")
            root_key = parts[0]
            if root_key not in projection_params_raw:
                projection_params_raw[root_key] = {}

            if len(parts) == 2:
                projection_params_raw[root_key][parts[1]] = value
            elif len(parts) == 3:
                if parts[1] not in projection_params_raw[root_key]:
                    projection_params_raw[root_key][parts[1]] = {}
                projection_params_raw[root_key][parts[1]][parts[2]] = value

    # Create configs
    paligemma_config = PaliGemmaConfig()
    gemma_expert_config = GemmaExpertConfig()

    # Convert PaliGemma weights
    pytorch_params, expert_params = _convert_paligemma_weights(paligemma_params, paligemma_config)

    # Convert Gemma expert weights
    gemma_params = _convert_gemma_expert_weights(expert_params, gemma_expert_config, pi05=pi05)
    pytorch_params.update(gemma_params)

    # Convert projection params
    projection_params = _convert_projection_params(projection_params_raw, pi05=pi05)
    pytorch_params.update(projection_params)

    # Apply precision
    dtype = torch.float32 if precision == "float32" else torch.bfloat16
    for key in pytorch_params:
        pytorch_params[key] = pytorch_params[key].to(dtype)

    # Save
    output_path.mkdir(parents=True, exist_ok=True)
    save_safetensors(pytorch_params, str(output_path / "model.safetensors"))

    # Copy assets folder if it exists
    assets_source = jax_checkpoint_path.parent / "assets"
    if assets_source.exists():
        assets_dest = output_path / "assets"
        if assets_dest.exists():
            shutil.rmtree(assets_dest)
        shutil.copytree(assets_source, assets_dest)

    # Save config as JSON for reference
    config_dict = {
        "model_type": "pi05" if pi05 else "pi0",
        "precision": precision,
        "paligemma_config": {
            "vision_hidden_size": paligemma_config.vision_config.hidden_size,
            "vision_num_layers": paligemma_config.vision_config.num_hidden_layers,
            "text_hidden_size": paligemma_config.text_config.hidden_size,
            "text_num_layers": paligemma_config.text_config.num_hidden_layers,
        },
        "gemma_expert_config": {
            "hidden_size": gemma_expert_config.hidden_size,
            "num_layers": gemma_expert_config.num_hidden_layers,
            "num_heads": gemma_expert_config.num_attention_heads,
        },
    }
    with (output_path / "config.json").open("w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"Converted to: {output_path}")
    logger.info(f"Total parameters: {len(pytorch_params)}")

    return output_path


# =============================================================================
# High-level API
# =============================================================================


def load_openpi_checkpoint(
    checkpoint_name: str,
    device: str = "cpu",
    force_download: bool = False,
    force_convert: bool = False,
) -> dict[str, torch.Tensor]:
    """Download, convert, and load an OpenPI checkpoint.

    Args:
        checkpoint_name: Name of the checkpoint (e.g., "pi0_base", "pi05_base")
        device: Device to load the tensors to
        force_download: Re-download even if cached
        force_convert: Re-convert even if cached

    Returns:
        State dict with model weights
    """
    # Determine config name for conversion
    config_name = "pi05_base" if "pi05" in checkpoint_name else "pi0_base"

    # Check for pre-converted PyTorch checkpoint
    cache_dir = get_cache_dir()
    pytorch_path = cache_dir / "pytorch" / checkpoint_name

    if pytorch_path.exists() and (pytorch_path / "model.safetensors").exists() and not force_convert:
        logger.info(f"Loading cached PyTorch checkpoint: {pytorch_path}")
        state_dict = load_safetensors(str(pytorch_path / "model.safetensors"), device=device)
        return state_dict

    # Download JAX checkpoint if needed
    jax_path = download_checkpoint(checkpoint_name, force_download=force_download)

    # Convert to PyTorch
    pytorch_path = convert_jax_to_pytorch(
        jax_path,
        output_path=pytorch_path,
        config_name=config_name,
    )

    # Load the converted checkpoint
    state_dict = load_safetensors(str(pytorch_path / "model.safetensors"), device=device)
    return state_dict


def get_available_checkpoints() -> dict[str, str]:
    """Get a list of available OpenPI checkpoints.

    Returns:
        Dict mapping checkpoint names to GCS URLs
    """
    return OPENPI_CHECKPOINTS.copy()


def map_openpi_to_crane_x7(
    openpi_state_dict: dict[str, torch.Tensor],
    pi05: bool = False,
) -> dict[str, torch.Tensor]:
    """Map OpenPI state dict keys to CRANE-X7 VLA model keys.

    The OpenPI model structure matches ours since we based our implementation
    on OpenPI, so most keys should map directly.

    Args:
        openpi_state_dict: State dict from OpenPI checkpoint
        pi05: Whether this is a Pi0.5 model

    Returns:
        State dict compatible with our Pi0Model
    """
    # The key prefixes should be compatible
    # OpenPI uses: paligemma_with_expert.paligemma.*, paligemma_with_expert.gemma_expert.*
    # Our model uses the same structure

    mapped_dict = {}

    for key, value in openpi_state_dict.items():
        # Direct mapping for most keys
        mapped_dict[key] = value

    return mapped_dict
