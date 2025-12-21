#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Merge LoRA adapters into base VLA models.

This script merges LoRA adapter weights saved during training into the base
model to create a standalone model for inference.

Supported backends:
- openvla: OpenVLA models
- pi0: Pi0/Pi0.5 models (merges expert and/or VLM adapters)

Usage:
    # OpenVLA
    python -m crane_x7_vla.scripts.merge_lora \
        --adapter_path /path/to/lora_adapters \
        --output_path /path/to/merged_model \
        --backend openvla \
        --base_model openvla/openvla-7b

    # Pi0
    python -m crane_x7_vla.scripts.merge_lora \
        --adapter_path /path/to/checkpoint/lora_adapters \
        --output_path /path/to/merged_checkpoint \
        --backend pi0

Example:
    python -m crane_x7_vla.scripts.merge_lora \
        --adapter_path outputs/crane_x7_openvla/lora_adapters \
        --output_path outputs/crane_x7_openvla_merged
"""

import argparse
import gc
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel


def register_openvla_classes():
    """Register OpenVLA classes with HuggingFace Auto classes."""
    # Lazy import to avoid dependency issues
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def merge_openvla_lora(
    adapter_path: Path,
    output_path: Path,
    base_model: str,
    copy_processor: bool,
    copy_statistics: bool,
) -> None:
    """Merge LoRA adapters for OpenVLA models."""
    from transformers import AutoModelForVision2Seq

    # Validate adapter path
    adapter_config_file = adapter_path / "adapter_config.json"
    if not adapter_config_file.exists():
        raise ValueError(f"No adapter_config.json found in {adapter_path}")

    # Register OpenVLA classes
    print("Registering OpenVLA classes...")
    register_openvla_classes()

    # Load base model on CPU to avoid GPU memory issues
    print(f"Loading base model from {base_model}...")
    base_vla = AutoModelForVision2Seq.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
        device_map="cpu",
    )

    # Load LoRA adapters
    print(f"Loading LoRA adapters from {adapter_path}...")
    merged_vla = PeftModel.from_pretrained(
        base_vla,
        adapter_path,
        device_map="cpu",
    )

    # Merge and unload adapters
    print("Merging LoRA weights into base model...")
    merged_vla = merged_vla.merge_and_unload()

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Save merged model
    print(f"Saving merged model to {output_path}...")
    merged_vla.save_pretrained(output_path)

    # Copy processor files if they exist in parent directory
    parent_dir = adapter_path.parent
    processor_files = [
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ]

    if copy_processor:
        print("Copying processor files...")
        for filename in processor_files:
            src_file = parent_dir / filename
            if src_file.exists():
                shutil.copy(src_file, output_path / filename)
                print(f"  Copied: {filename}")

    # Copy dataset statistics if they exist
    if copy_statistics:
        stats_file = parent_dir / "dataset_statistics.json"
        if stats_file.exists():
            shutil.copy(stats_file, output_path / "dataset_statistics.json")
            print("  Copied: dataset_statistics.json")

    # Clean up to free memory
    del base_vla, merged_vla
    gc.collect()
    torch.cuda.empty_cache()


def merge_pi0_lora(
    adapter_path: Path,
    output_path: Path,
    checkpoint_path: Path | None = None,
) -> None:
    """Merge LoRA adapters for Pi0/Pi0.5 models.

    Pi0 LoRA adapters are saved in subdirectories:
    - lora_adapters/gemma_expert/  (action expert LoRA)
    - lora_adapters/paligemma_lm/  (VLM LoRA, optional)
    """
    from crane_x7_vla.backends.pi0.model import Pi0Model, Pi0ModelConfig

    # Load checkpoint to get config
    if checkpoint_path is None:
        # Try to find checkpoint.pt in parent directory
        checkpoint_path = adapter_path.parent / "checkpoint.pt"

    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})

    # Create model from config
    model_config = Pi0ModelConfig(
        pi05=config.get("model_type", "pi0") == "pi0.5",
        paligemma_variant=config.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=config.get("action_expert_variant", "gemma_300m"),
        action_dim=config.get("action_dim", 32),
        action_horizon=config.get("action_horizon", 50),
        max_token_len=config.get("max_token_len", 48),
        dtype=config.get("precision", "bfloat16"),
    )

    print("Creating Pi0 model...")
    model = Pi0Model(model_config)

    # Check for LoRA adapters
    expert_lora_path = adapter_path / "gemma_expert"
    vlm_lora_path = adapter_path / "paligemma_lm"

    # Load and merge expert LoRA
    if expert_lora_path.exists():
        print(f"Loading and merging Expert LoRA from {expert_lora_path}...")
        model.paligemma_with_expert.gemma_expert = PeftModel.from_pretrained(
            model.paligemma_with_expert.gemma_expert,
            expert_lora_path,
        )
        model.paligemma_with_expert.gemma_expert = model.paligemma_with_expert.gemma_expert.merge_and_unload()
        print("  Merged Expert LoRA successfully")

    # Load and merge VLM LoRA if present
    if vlm_lora_path.exists():
        print(f"Loading and merging VLM LoRA from {vlm_lora_path}...")
        model.paligemma_with_expert.paligemma.language_model = PeftModel.from_pretrained(
            model.paligemma_with_expert.paligemma.language_model,
            vlm_lora_path,
        )
        model.paligemma_with_expert.paligemma.language_model = (
            model.paligemma_with_expert.paligemma.language_model.merge_and_unload()
        )
        print("  Merged VLM LoRA successfully")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Save merged checkpoint
    print(f"Saving merged checkpoint to {output_path}...")
    merged_checkpoint = {
        "model_state_dict": model.state_dict(),
        "global_step": checkpoint.get("global_step", 0),
        "epoch": checkpoint.get("epoch", 0),
        "config": config,
        "use_lora": False,  # LoRA has been merged
    }
    torch.save(merged_checkpoint, output_path / "checkpoint.pt")

    # Copy config.json if exists
    config_json_path = adapter_path.parent / "config.json"
    if config_json_path.exists():
        with config_json_path.open() as f:
            cfg_dict = json.load(f)
        cfg_dict["use_lora"] = False  # Mark as merged
        with (output_path / "config.json").open("w") as f:
            json.dump(cfg_dict, f, indent=2, default=str)
        print("  Copied and updated config.json")

    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()


def merge_lora_adapters(
    adapter_path: str,
    output_path: str,
    backend: str = "openvla",
    base_model: str = "openvla/openvla-7b",
    copy_processor: bool = True,
    copy_statistics: bool = True,
) -> None:
    """
    Merge LoRA adapters into base model.

    Args:
        adapter_path: Path to directory containing LoRA adapter weights
        output_path: Path to save merged model
        backend: Backend type ('openvla' or 'pi0')
        base_model: HuggingFace model ID or path to base model (for OpenVLA)
        copy_processor: Whether to copy processor files from adapter directory
        copy_statistics: Whether to copy dataset statistics from adapter directory
    """
    adapter_path = Path(adapter_path)
    output_path = Path(output_path)

    # Validate adapter path
    if not adapter_path.exists():
        raise ValueError(f"Adapter path does not exist: {adapter_path}")

    print("=" * 60)
    print("LoRA Merge Script")
    print("=" * 60)
    print(f"Backend: {backend}")
    print(f"Adapter path: {adapter_path}")
    print(f"Output path: {output_path}")
    if backend == "openvla":
        print(f"Base model: {base_model}")
    print("=" * 60)

    if backend == "openvla":
        merge_openvla_lora(adapter_path, output_path, base_model, copy_processor, copy_statistics)
    elif backend in ("pi0", "pi0.5"):
        merge_pi0_lora(adapter_path, output_path)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    print("=" * 60)
    print("Merge completed successfully!")
    print(f"Merged model saved to: {output_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters into base VLA model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OpenVLA
  python -m crane_x7_vla.scripts.merge_lora \\
      --adapter_path outputs/crane_x7_openvla/lora_adapters \\
      --output_path outputs/crane_x7_openvla_merged \\
      --backend openvla

  # Pi0
  python -m crane_x7_vla.scripts.merge_lora \\
      --adapter_path outputs/crane_x7_pi0/checkpoint_10000/lora_adapters \\
      --output_path outputs/crane_x7_pi0_merged \\
      --backend pi0
        """,
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to directory containing LoRA adapter weights",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save merged model",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="openvla",
        choices=["openvla", "pi0", "pi0.5"],
        help="Backend type (default: openvla)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="openvla/openvla-7b",
        help="HuggingFace model ID or path to base model (for OpenVLA, default: openvla/openvla-7b)",
    )
    parser.add_argument(
        "--no_copy_processor",
        action="store_true",
        help="Do not copy processor files from adapter directory (OpenVLA only)",
    )
    parser.add_argument(
        "--no_copy_statistics",
        action="store_true",
        help="Do not copy dataset statistics from adapter directory (OpenVLA only)",
    )

    args = parser.parse_args()

    merge_lora_adapters(
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        backend=args.backend,
        base_model=args.base_model,
        copy_processor=not args.no_copy_processor,
        copy_statistics=not args.no_copy_statistics,
    )


if __name__ == "__main__":
    main()
