# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Pi0/Pi0.5 Model Implementation.

Based on OpenPI's pi0_pytorch.py implementation.
Uses PaliGemma + Expert Gemma architecture with flow matching.

Key components:
- PaliGemmaWithExpertModel: VLM + Action Expert
- Flow matching for action prediction
- Support for both Pi0 (continuous state) and Pi0.5 (discrete state + adaRMSNorm)
"""

import logging
import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


logger = logging.getLogger(__name__)


@dataclass
class Pi0ModelConfig:
    """Configuration for Pi0 model."""

    pi05: bool = False
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48
    dtype: str = "bfloat16"


# =============================================================================
# Gemma Configuration
# =============================================================================


@dataclass
class GemmaConfig:
    """Gemma model configuration."""

    width: int
    mlp_dim: int
    num_heads: int
    head_dim: int
    depth: int
    num_kv_heads: int


GEMMA_CONFIGS: dict[str, GemmaConfig] = {
    "gemma_2b": GemmaConfig(
        width=2048,
        mlp_dim=16384,
        num_heads=8,
        head_dim=256,
        depth=18,
        num_kv_heads=1,
    ),
    "gemma_2b_lora": GemmaConfig(
        width=2048,
        mlp_dim=16384,
        num_heads=8,
        head_dim=256,
        depth=18,
        num_kv_heads=1,
    ),
    "gemma_300m": GemmaConfig(
        width=1024,
        mlp_dim=4096,
        num_heads=8,
        head_dim=128,
        depth=18,
        num_kv_heads=1,
    ),
    "gemma_300m_lora": GemmaConfig(
        width=1024,
        mlp_dim=4096,
        num_heads=8,
        head_dim=128,
        depth=18,
        num_kv_heads=1,
    ),
}


def get_gemma_config(variant: str) -> GemmaConfig:
    """Get Gemma configuration by variant name."""
    if variant not in GEMMA_CONFIGS:
        raise ValueError(f"Unknown Gemma variant: {variant}. Available: {list(GEMMA_CONFIGS.keys())}")
    return GEMMA_CONFIGS[variant]


# =============================================================================
# Utility Functions
# =============================================================================


def get_safe_dtype(target_dtype: torch.dtype, device_type: str) -> torch.dtype:
    """Get a safe dtype for the given device type."""
    if device_type == "cpu" and target_dtype == torch.bfloat16:
        return torch.float32
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: torch.device,
) -> Tensor:
    """Compute sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape (batch_size,)")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha: float, beta: float, bsize: int, device: torch.device) -> torch.Tensor:
    """Sample from Beta distribution."""
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """Create 2D attention masks from padding and attention masks.

    Tokens can attend to valid input tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention patterns.

    Args:
        pad_masks: bool[B, N] - True if part of the input, False if padding
        att_masks: int32[B, N] - 1 where previous tokens cannot attend, 0 for shared attention

    Returns:
        bool[B, N, N] - 2D attention mask
    """
    if att_masks.ndim != 2:
        raise ValueError(f"att_masks must be 2D, got {att_masks.ndim}D")
    if pad_masks.ndim != 2:
        raise ValueError(f"pad_masks must be 2D, got {pad_masks.ndim}D")

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


# =============================================================================
# PaliGemma with Expert Model
# =============================================================================


class PaliGemmaWithExpertModel(nn.Module):
    """PaliGemma VLM combined with Action Expert Gemma.

    This model combines:
    - PaliGemma: Vision-language model for processing images and text
    - Expert Gemma: Smaller model for action prediction with optional adaRMSNorm

    Uses joint attention mechanism from openpi for efficient inference.
    """

    def __init__(
        self,
        vlm_config: GemmaConfig,
        action_expert_config: GemmaConfig,
        use_adarms: list[bool] | None = None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        super().__init__()

        if use_adarms is None:
            use_adarms = [False, False]

        from transformers import GemmaForCausalLM, PaliGemmaForConditionalGeneration
        from transformers.models.auto import CONFIG_MAPPING

        # Create PaliGemma config
        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        # Create Action Expert config
        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None  # Share embeddings with PaliGemma

        # Projection layer: VLM hidden dim -> Expert hidden dim
        self.vlm_to_expert_proj = nn.Linear(vlm_config.width, action_expert_config.width)

        self._apply_precision(precision)
        self._debug_gc_printed = False

    def _apply_precision(self, precision: Literal["bfloat16", "float32"]) -> None:
        """Apply precision settings to the model."""
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        # Keep certain parameters in float32 for stability
        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        """Embed image using SigLIP vision tower."""
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed language tokens."""
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor | None], list[torch.FloatTensor] | None]:
        """Forward pass through PaliGemma and Expert Gemma.

        Uses a simplified approach:
        - Process prefix (image + language) through PaliGemma
        - Process suffix (state + actions) through Expert Gemma with prefix context
        """
        if adarms_cond is None:
            adarms_cond = [None, None]

        # Prefix-only forward (for KV caching)
        if inputs_embeds[1] is None:
            self.paligemma.language_model.config._attn_implementation = "eager"
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0],
            )
            return [prefix_output.last_hidden_state, None], prefix_output.past_key_values

        # Suffix-only forward (using cached prefix KV)
        if inputs_embeds[0] is None:
            self.gemma_expert.model.config._attn_implementation = "eager"
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1],
            )
            return [None, suffix_output.last_hidden_state], None

        # Joint forward: Project prefix to Expert dim, concatenate with suffix, process through Expert Gemma
        # 1. Process prefix through PaliGemma
        self.paligemma.language_model.config._attn_implementation = "eager"
        prefix_output = self.paligemma.language_model.forward(
            inputs_embeds=inputs_embeds[0],
            attention_mask=None,  # Prefix attends to itself only
            position_ids=None,
            past_key_values=None,
            use_cache=False,
            adarms_cond=adarms_cond[0],
        )
        prefix_hidden = prefix_output.last_hidden_state

        # 2. Project prefix to Expert Gemma's dimension
        prefix_proj = self.vlm_to_expert_proj(prefix_hidden)

        # 3. Concatenate projected prefix + suffix
        combined_embeds = torch.cat([prefix_proj, inputs_embeds[1]], dim=1)

        # 4. Process combined through Expert Gemma
        self.gemma_expert.model.config._attn_implementation = "eager"
        combined_output = self.gemma_expert.model.forward(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            adarms_cond=adarms_cond[1],
        )

        # 5. Extract suffix portion of output
        prefix_len = prefix_proj.shape[1]
        suffix_hidden = combined_output.last_hidden_state[:, prefix_len:]

        return [None, suffix_hidden], None


# =============================================================================
# Pi0 Model
# =============================================================================


class Pi0Model(nn.Module):
    """Pi0/Pi0.5 Model for action prediction.

    Implements flow matching for action chunk prediction using:
    - PaliGemma for vision-language understanding
    - Expert Gemma for action prediction
    - Optional adaRMSNorm for Pi0.5 timestep injection

    Key differences between Pi0 and Pi0.5:
    - Pi0: Continuous state input via projection, timestep mixed via MLP
    - Pi0.5: Discrete state tokens, timestep via adaRMSNorm conditioning
    """

    def __init__(self, config: Pi0ModelConfig):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon

        # Get Gemma configs
        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        # Create PaliGemma with Expert
        use_adarms = [False, True] if config.pi05 else [False, False]
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=use_adarms,
            precision=config.dtype,
        )

        # Action projection layers
        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        # Timestep processing layers (different for Pi0 vs Pi0.5)
        if self.pi05:
            # Pi0.5: Time MLP for adaRMSNorm conditioning
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            # Pi0: State projection and action-time MLP
            self.state_proj = nn.Linear(config.action_dim, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        self.gradient_checkpointing_enabled = False

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logger.info("Enabled gradient checkpointing for Pi0 model")

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logger.info("Disabled gradient checkpointing for Pi0 model")

    def _prepare_attention_masks_4d(self, att_2d_masks: torch.Tensor) -> torch.Tensor:
        """Convert 2D attention masks to 4D format for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def sample_noise(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """Sample Gaussian noise."""
        return torch.randn(shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps from Beta distribution."""
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images and language tokens for prefix.

        Args:
            images: List of image tensors [B, C, H, W]
            img_masks: List of image masks [B]
            lang_tokens: Language token IDs [B, seq_len]
            lang_masks: Language attention masks [B, seq_len]

        Returns:
            Tuple of (embeddings, pad_masks, att_masks)
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self.paligemma_with_expert.embed_image(img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs  # Image tokens attend to each other

        # Process language tokens
        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]  # Full attention between image and language

        # Concatenate
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(
        self,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Embed state, noisy actions, and timestep for suffix.

        Args:
            state: Robot state [B, state_dim]
            noisy_actions: Noisy action chunk [B, horizon, action_dim]
            timestep: Flow matching timestep [B]

        Returns:
            Tuple of (embeddings, pad_masks, att_masks, adarms_cond)
        """
        embs = []
        pad_masks = []
        att_masks = []

        device = timestep.device
        bsize = noisy_actions.shape[0]

        # Pi0: Add state token
        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)
            state_emb = self.state_proj(state)[:, None, :]
            embs.append(state_emb)
            pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
            att_masks += [1]  # State doesn't attend to previous tokens

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
            device=device,
        )
        time_emb = time_emb.to(dtype=timestep.dtype)

        # Project noisy actions
        action_emb = self.action_in_proj(noisy_actions)

        # Process timestep differently for Pi0 vs Pi0.5
        if not self.pi05:
            # Pi0: Mix timestep with actions via MLP
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)
            action_time_emb = self.action_time_mlp_in(action_time_emb)
            action_time_emb = F.silu(action_time_emb)
            action_time_emb = self.action_time_mlp_out(action_time_emb)
            adarms_cond = None
        else:
            # Pi0.5: Time MLP for adaRMSNorm conditioning
            time_emb = self.time_mlp_in(time_emb)
            time_emb = F.silu(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = F.silu(time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        embs.append(action_time_emb)
        pad_masks.append(torch.ones(bsize, self.action_horizon, dtype=torch.bool, device=device))
        att_masks += [1] + ([0] * (self.action_horizon - 1))  # Causal attention for actions

        # Concatenate
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for training with flow matching loss.

        Args:
            images: List of image tensors [B, C, H, W]
            img_masks: List of image masks [B]
            lang_tokens: Language token IDs [B, seq_len]
            lang_masks: Language attention masks [B, seq_len]
            state: Robot state [B, state_dim]
            actions: Target action chunk [B, horizon, action_dim]
            noise: Optional pre-sampled noise
            time: Optional pre-sampled timesteps

        Returns:
            Flow matching loss [B, horizon, action_dim]
        """
        device = actions.device
        bsize = actions.shape[0]

        # Sample noise and time if not provided
        if noise is None:
            noise = self.sample_noise(actions.shape, device)
        if time is None:
            time = self.sample_time(bsize, device)

        # Compute noisy actions: x_t = t * noise + (1 - t) * actions
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions

        # Target velocity: u_t = noise - actions
        u_t = noise - actions

        # Embed prefix (images + language)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        # Embed suffix (state + noisy actions + time)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)

        # Match dtypes
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # Create attention masks
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Forward through model
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        # Extract action predictions
        suffix_out = suffix_out[:, -self.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)

        # Compute MSE loss
        return F.mse_loss(u_t, v_t, reduction="none")

    def _get_model_dtype(self) -> torch.dtype:
        """Get model weight dtype (cached)."""
        if not hasattr(self, "_cached_model_dtype"):
            self._cached_model_dtype = self.paligemma_with_expert.paligemma.language_model.layers[
                0
            ].self_attn.q_proj.weight.dtype
        return self._cached_model_dtype

    def _denoise_step_cached(
        self,
        state: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        prefix_proj: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Single denoising step with cached prefix projection.

        This uses the pre-computed projected prefix, avoiding redundant
        PaliGemma processing during ODE integration.

        Args:
            state: Robot state [B, state_dim]
            x_t: Current noisy actions [B, horizon, action_dim]
            timestep: Flow matching timestep [B]
            prefix_proj: Pre-computed projected prefix [B, prefix_len, expert_dim]
            prefix_pad_masks: Prefix padding masks [B, prefix_len]
            prefix_att_masks: Prefix attention masks [B, prefix_len]
        """
        # Embed suffix for current x_t
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        # Match dtype
        model_dtype = self._get_model_dtype()
        if model_dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_proj = prefix_proj.to(dtype=torch.bfloat16)

        # Concatenate projected prefix + suffix
        combined_embs = torch.cat([prefix_proj, suffix_embs], dim=1)

        # Create attention masks
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Forward through Expert Gemma only
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        combined_output = self.paligemma_with_expert.gemma_expert.model.forward(
            inputs_embeds=combined_embs,
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            adarms_cond=adarms_cond,
        )

        # Extract velocity prediction from action tokens (suffix portion)
        prefix_len = prefix_proj.shape[1]
        suffix_out = combined_output.last_hidden_state[:, prefix_len:]
        suffix_out = suffix_out[:, -self.action_horizon :]
        return self.action_out_proj(suffix_out.float())

    @torch.no_grad()
    def sample_actions(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        num_steps: int = 5,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample actions using flow matching ODE integration.

        Optimized for inference:
        - Processes prefix through PaliGemma ONCE
        - Projects prefix to Expert Gemma dimension
        - Reuses projected prefix for each denoising step (Expert Gemma only)

        Args:
            images: List of image tensors [B, C, H, W]
            img_masks: List of image masks [B]
            lang_tokens: Language token IDs [B, seq_len]
            lang_masks: Language attention masks [B, seq_len]
            state: Robot state [B, state_dim]
            num_steps: Number of integration steps (default: 5)
            noise: Optional initial noise

        Returns:
            Sampled action chunk [B, horizon, action_dim]
        """
        device = state.device
        bsize = state.shape[0]

        # Initialize from noise
        if noise is None:
            actions_shape = (bsize, self.action_horizon, self.action_dim)
            noise = self.sample_noise(actions_shape, device)
        x_t = noise

        # Embed prefix (images + language)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        # Convert prefix to model dtype
        model_dtype = self._get_model_dtype()
        if model_dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # Process prefix through PaliGemma ONCE (most expensive part)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        prefix_output = self.paligemma_with_expert.paligemma.language_model.forward(
            inputs_embeds=prefix_embs,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            use_cache=False,
            adarms_cond=None,
        )
        prefix_hidden = prefix_output.last_hidden_state

        # Project prefix to Expert Gemma dimension ONCE
        prefix_proj = self.paligemma_with_expert.vlm_to_expert_proj(prefix_hidden)

        # Pre-compute timesteps
        dt = -1.0 / num_steps
        timesteps = [torch.tensor([1.0 + i * dt] * bsize, dtype=torch.float32, device=device) for i in range(num_steps)]

        # Euler integration - only Expert Gemma runs each step
        for timestep in timesteps:
            v_t = self._denoise_step_cached(
                state,
                x_t,
                timestep,
                prefix_proj,
                prefix_pad_masks,
                prefix_att_masks,
            )
            x_t = x_t + dt * v_t

        return x_t
