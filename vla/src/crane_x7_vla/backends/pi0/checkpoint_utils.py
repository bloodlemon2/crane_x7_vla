# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Utilities for downloading and converting OpenPI checkpoints."""

import logging
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Literal

import torch
from safetensors.torch import load_file as load_safetensors


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


def get_cache_dir() -> pathlib.Path:
    """Get the cache directory for OpenPI checkpoints."""
    cache_dir = pathlib.Path(os.getenv("CRANE_X7_VLA_CACHE", str(DEFAULT_CACHE_DIR)))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


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


def convert_jax_to_pytorch(
    jax_checkpoint_path: pathlib.Path,
    output_path: pathlib.Path | None = None,
    config_name: str = "pi0_base",
    precision: Literal["float32", "bfloat16"] = "bfloat16",
) -> pathlib.Path:
    """Convert a JAX checkpoint to PyTorch format using OpenPI's converter.

    Args:
        jax_checkpoint_path: Path to the JAX checkpoint
        output_path: Path to save the converted checkpoint (default: auto-generated)
        config_name: OpenPI config name for the model
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

    # Find the OpenPI convert script
    openpi_path = pathlib.Path(__file__).parent.parent.parent.parent / "openpi"
    convert_script = openpi_path / "examples" / "convert_jax_model_to_pytorch.py"

    if not convert_script.exists():
        raise FileNotFoundError(
            f"OpenPI convert script not found at {convert_script}. " "Make sure the openpi submodule is initialized."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run the conversion script
    env = os.environ.copy()
    env["PYTHONPATH"] = str(openpi_path / "src") + ":" + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [
            sys.executable,
            str(convert_script),
            "--checkpoint_dir",
            str(jax_checkpoint_path / "params" if (jax_checkpoint_path / "params").exists() else jax_checkpoint_path),
            "--config_name",
            config_name,
            "--output_path",
            str(output_path),
            "--precision",
            precision,
        ],
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Conversion failed: {result.stderr}")
        raise RuntimeError(f"Checkpoint conversion failed: {result.stderr}")

    logger.info(f"Converted to: {output_path}")
    return output_path


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
    if "pi05" in checkpoint_name:
        config_name = "pi05_base" if "base" in checkpoint_name else f"pi05_{checkpoint_name.split('_')[-1]}"
    else:
        config_name = "pi0_base" if "base" in checkpoint_name else checkpoint_name

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
