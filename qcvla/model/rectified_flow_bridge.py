"""
Rectified Flow Bridge: Adapted from RectifiedFlow for VLM feature dynamics prediction.

Based on: https://github.com/gnobitab/RectifiedFlow
Paper: "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"

Key Adaptations:
1. Replace UNet (2D) with lightweight MLP/Transformer (1D features)
2. Add conditioning on state, action, horizon
3. Keep RectifiedFlow loss and sampling logic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


# ============================================================================
# Time Embedding (from RectifiedFlow)
# ============================================================================

def get_timestep_embedding(timesteps, embedding_dim):
    """
    Build sinusoidal embeddings (from Fairseq).
    Matches the implementation in RectifiedFlow.
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1, 0, 0))

    return emb


# ============================================================================
# Model Architecture: Replace UNet with Lightweight MLP for Features
# ============================================================================

class ResidualMLPBlock(nn.Module):
    """MLP block with residual connection and time conditioning."""

    def __init__(self, dim: int, hidden_dim: int, time_dim: int, dropout: float = 0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

        # Time conditioning via scale and shift
        self.time_proj = nn.Linear(time_dim, dim * 2)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x: [B, seq_len, dim]
        t_emb: [B, time_dim]
        """
        # Time conditioning
        scale_shift = self.time_proj(t_emb).unsqueeze(1)  # [B, 1, dim*2]
        scale, shift = scale_shift.chunk(2, dim=-1)

        # First residual block with time modulation
        h = self.norm1(x)
        h = h * (1 + scale) + shift
        x = x + self.mlp1(h)

        # Second residual block
        x = x + self.mlp2(self.norm2(x))

        return x


# ============================================================================
# DiT-style Blocks (from "Scalable Diffusion Models with Transformers")
# ============================================================================

def modulate(x, shift, scale):
    """Apply adaptive modulation: x * (1 + scale) + shift"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    DiT block with adaptive layer norm (adaLN-Zero).

    Based on: "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Self-attention
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # FFN
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

        # AdaLN modulation: 6 params (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
        # Zero-initialize the modulation (DiT key insight)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: [B, seq_len, hidden_dim] - input features
        c: [B, hidden_dim] - conditioning (time + other conditions)
        """
        # Get modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)

        # Self-attention with adaLN
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # FFN with adaLN
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        return x


class DiTCrossBlock(nn.Module):
    """
    DiT block with cross-attention to stable context.

    Adds cross-attention to layer 8 (stable) features.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Self-attention
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Cross-attention to stable context
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm_stable = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # FFN
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

        # AdaLN modulation: 9 params (3 for self-attn, 3 for cross-attn, 3 for FFN)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 9 * hidden_dim),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, stable: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: [B, seq_len, hidden_dim] - dynamic features (layer 16)
        stable: [B, seq_len, hidden_dim] - stable features (layer 8)
        c: [B, hidden_dim] - conditioning
        """
        # Get modulation parameters
        mods = self.adaLN_modulation(c).chunk(9, dim=-1)
        shift_sa, scale_sa, gate_sa = mods[0], mods[1], mods[2]
        shift_ca, scale_ca, gate_ca = mods[3], mods[4], mods[5]
        shift_mlp, scale_mlp, gate_mlp = mods[6], mods[7], mods[8]

        # Self-attention
        x_norm = modulate(self.norm1(x), shift_sa, scale_sa)
        sa_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + gate_sa.unsqueeze(1) * sa_out

        # Cross-attention to stable context
        x_norm = modulate(self.norm2(x), shift_ca, scale_ca)
        stable_norm = self.norm_stable(stable)
        ca_out, _ = self.cross_attn(x_norm, stable_norm, stable_norm)
        x = x + gate_ca.unsqueeze(1) * ca_out

        # FFN
        x_norm = modulate(self.norm3(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        return x


class DiTFinalLayer(nn.Module):
    """Final layer of DiT with adaLN."""

    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        # Zero-initialize output
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class FeatureVelocityModel(nn.Module):
    """
    Velocity model for VLM feature prediction.
    Replaces UNet from RectifiedFlow with lightweight MLP.

    Input: [B, seq_len, feature_dim] - noisy/interpolated features
    Output: [B, seq_len, feature_dim] - predicted velocity
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        seq_len: int = 204,
        hidden_dim: int = 512,  # Reduced for speed
        time_dim: int = 128,    # Reduced for speed
        cond_dim: int = 128,    # Reduced for speed
        num_blocks: int = 2,    # Reduced for speed
        dropout: float = 0.0,
        # Conditioning dimensions
        state_dim: int = 128,
        action_dim: int = 128,
        max_horizon: int = 10,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.time_dim = time_dim
        self.cond_dim = cond_dim

        # Time embedding (matches RectifiedFlow: t * 999)
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # Condition embeddings
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, cond_dim // 2),
            nn.GELU(),
        )
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, cond_dim // 2),
            nn.GELU(),
        )
        self.horizon_embed = nn.Embedding(max_horizon + 1, cond_dim // 4)

        # Fuse all conditions
        total_cond_dim = time_dim + cond_dim // 2 + cond_dim // 2 + cond_dim // 4
        self.cond_fuse = nn.Sequential(
            nn.Linear(total_cond_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input projection
        self.input_proj = nn.Linear(feature_dim, hidden_dim)

        # Residual MLP blocks
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, hidden_dim * 2, time_dim, dropout)
            for _ in range(num_blocks)
        ])

        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, feature_dim)

        # Initialize output to zero for residual-like start
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        self._count_parameters()

    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"FeatureVelocityModel: {total/1e6:.2f}M params")

    def forward(
        self,
        x: torch.Tensor,           # [B, seq_len, feature_dim] or [B, C, H, W] for compatibility
        t: torch.Tensor,           # [B] timesteps (scaled by 999 in RectifiedFlow)
        state: Optional[torch.Tensor] = None,    # [B, state_dim]
        action: Optional[torch.Tensor] = None,   # [B, action_dim]
        horizon: Optional[torch.Tensor] = None,  # [B] integer horizon
    ) -> torch.Tensor:
        """
        Predict velocity field at x_t.

        Compatible with RectifiedFlow training which passes t * 999.
        """
        B = x.shape[0]
        device = x.device

        # Handle input shape (RectifiedFlow passes [B, C, H, W])
        if x.dim() == 4:
            # Reshape from [B, C, H, W] to [B, seq_len, feature_dim]
            x = x.view(B, self.seq_len, self.feature_dim)

        # Time embedding (t comes as t * 999 from RectifiedFlow)
        t_normalized = t / 999.0  # Normalize back to [0, 1]
        t_emb = get_timestep_embedding(t_normalized, self.time_dim)
        t_emb = self.time_embed(t_emb)  # [B, time_dim]

        # Condition embeddings
        if state is not None:
            state_emb = self.state_embed(state)
        else:
            state_emb = torch.zeros(B, self.cond_dim // 2, device=device)

        if action is not None:
            action_emb = self.action_embed(action)
        else:
            action_emb = torch.zeros(B, self.cond_dim // 2, device=device)

        if horizon is not None:
            horizon_emb = self.horizon_embed(horizon.long())
        else:
            horizon_emb = torch.zeros(B, self.cond_dim // 4, device=device)

        # Fuse all conditions
        cond = torch.cat([t_emb, state_emb, action_emb, horizon_emb], dim=-1)
        cond = self.cond_fuse(cond)  # [B, time_dim]

        # Forward through network
        h = self.input_proj(x)  # [B, seq_len, hidden_dim]

        for block in self.blocks:
            h = block(h, cond)

        # Output velocity
        h = self.output_norm(h)
        velocity = self.output_proj(h)  # [B, seq_len, feature_dim]

        # Reshape back if input was 4D
        if len(x.shape) == 4:
            velocity = velocity.view(B, self.seq_len, self.feature_dim)

        return velocity


class MultiLayerFeatureVelocityModel(nn.Module):
    """
    Velocity model with multi-layer hidden state input.

    Uses learned layer weighting to combine information from multiple VLM layers.
    Input: [B, num_layers, seq_len, feature_dim] - multi-layer features
    Output: [B, seq_len, feature_dim] - predicted velocity (for last layer)
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        seq_len: int = 204,
        num_layers: int = 4,  # Number of input layers (e.g., last 4 layers)
        hidden_dim: int = 512,
        time_dim: int = 128,
        cond_dim: int = 128,
        num_blocks: int = 2,
        dropout: float = 0.0,
        state_dim: int = 128,
        action_dim: int = 128,
        max_horizon: int = 10,
        layer_weighting: str = "learned",  # "learned", "attention", or "concat"
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.time_dim = time_dim
        self.cond_dim = cond_dim
        self.layer_weighting = layer_weighting

        # Layer weighting mechanisms
        if layer_weighting == "learned":
            # Simple learned weights (softmax normalized)
            self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        elif layer_weighting == "attention":
            # Attention-based layer weighting (conditioned on t)
            self.layer_query = nn.Linear(time_dim, feature_dim)
            self.layer_key = nn.Linear(feature_dim, feature_dim)
        elif layer_weighting == "concat":
            # Concatenate and project
            self.layer_proj = nn.Linear(feature_dim * num_layers, feature_dim)
        else:
            raise ValueError(f"Unknown layer_weighting: {layer_weighting}")

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # Condition embeddings
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, cond_dim // 2),
            nn.GELU(),
        )
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, cond_dim // 2),
            nn.GELU(),
        )
        self.horizon_embed = nn.Embedding(max_horizon + 1, cond_dim // 4)

        # Fuse all conditions
        total_cond_dim = time_dim + cond_dim // 2 + cond_dim // 2 + cond_dim // 4
        self.cond_fuse = nn.Sequential(
            nn.Linear(total_cond_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input projection
        self.input_proj = nn.Linear(feature_dim, hidden_dim)

        # Residual MLP blocks
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, hidden_dim * 2, time_dim, dropout)
            for _ in range(num_blocks)
        ])

        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, feature_dim)

        # Initialize output to zero
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        self._count_parameters()

    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"MultiLayerFeatureVelocityModel: {total/1e6:.2f}M params "
              f"(layers={self.num_layers}, weighting={self.layer_weighting})")

    def combine_layers(
        self,
        multi_layer_features: torch.Tensor,  # [B, num_layers, seq_len, feature_dim]
        t_emb: torch.Tensor,  # [B, time_dim]
    ) -> torch.Tensor:
        """Combine multi-layer features into single representation."""
        B, L, S, D = multi_layer_features.shape

        if self.layer_weighting == "learned":
            # Softmax over learned weights
            weights = F.softmax(self.layer_weights, dim=0)  # [num_layers]
            # Weighted sum: [B, seq_len, feature_dim]
            combined = torch.einsum('blsd,l->bsd', multi_layer_features, weights)

        elif self.layer_weighting == "attention":
            # Query from time embedding
            query = self.layer_query(t_emb)  # [B, feature_dim]

            # Pool each layer to get keys
            layer_pooled = multi_layer_features.mean(dim=2)  # [B, num_layers, feature_dim]
            keys = self.layer_key(layer_pooled)  # [B, num_layers, feature_dim]

            # Attention scores
            scores = torch.einsum('bd,bld->bl', query, keys) / (D ** 0.5)  # [B, num_layers]
            weights = F.softmax(scores, dim=-1)  # [B, num_layers]

            # Weighted sum
            combined = torch.einsum('blsd,bl->bsd', multi_layer_features, weights)

        elif self.layer_weighting == "concat":
            # Concatenate along feature dim
            concat = multi_layer_features.permute(0, 2, 1, 3).reshape(B, S, L * D)
            combined = self.layer_proj(concat)  # [B, seq_len, feature_dim]

        return combined

    def forward(
        self,
        x: torch.Tensor,  # [B, num_layers, seq_len, feature_dim] or [B, seq_len, feature_dim]
        t: torch.Tensor,  # [B] timesteps
        state: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        horizon: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict velocity field from multi-layer features.
        """
        B = x.shape[0]
        device = x.device

        # Handle single-layer input for compatibility
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [B, 1, seq_len, feature_dim]

        # Time embedding
        t_normalized = t / 999.0
        t_emb = get_timestep_embedding(t_normalized, self.time_dim)
        t_emb = self.time_embed(t_emb)

        # Condition embeddings
        if state is not None:
            state_emb = self.state_embed(state)
        else:
            state_emb = torch.zeros(B, self.cond_dim // 2, device=device)

        if action is not None:
            action_emb = self.action_embed(action)
        else:
            action_emb = torch.zeros(B, self.cond_dim // 2, device=device)

        if horizon is not None:
            horizon_emb = self.horizon_embed(horizon.long())
        else:
            horizon_emb = torch.zeros(B, self.cond_dim // 4, device=device)

        # Fuse conditions
        cond = torch.cat([t_emb, state_emb, action_emb, horizon_emb], dim=-1)
        cond = self.cond_fuse(cond)

        # Combine multi-layer features
        combined = self.combine_layers(x, t_emb)  # [B, seq_len, feature_dim]

        # Forward through network
        h = self.input_proj(combined)

        for block in self.blocks:
            h = block(h, cond)

        # Output velocity
        h = self.output_norm(h)
        velocity = self.output_proj(h)

        return velocity

    def get_layer_weights(self) -> torch.Tensor:
        """Get current layer weights for visualization."""
        if self.layer_weighting == "learned":
            return F.softmax(self.layer_weights, dim=0)
        else:
            return None


class StableDynamicFlowModel(nn.Module):
    """
    Stable-Dynamic Decomposition for VLM feature prediction.

    Key insight: Layer 8 is FROZEN (0% change), use as stable context anchor.

    Conditions:
    - state: robot proprioception (previous)
    - action: robot action (previous)
    - horizon: prediction steps (previous)
    - time: rectified flow time (previous)
    - stable_context: layer 8 features (NEW - frozen anchor)

    Input:
    - x: [B, seq_len, feature_dim] - layer 16 features (dynamic, to predict)
    - stable_features: [B, seq_len, feature_dim] - layer 8 features (frozen anchor)

    Output: [B, seq_len, feature_dim] - predicted velocity for layer 16
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        seq_len: int = 204,
        hidden_dim: int = 512,
        time_dim: int = 128,
        cond_dim: int = 128,
        num_blocks: int = 2,
        dropout: float = 0.0,
        # Previous conditions
        state_dim: int = 128,
        action_dim: int = 128,
        max_horizon: int = 10,
        # New: stable context options
        use_stable_context: bool = True,
        stable_context_dim: int = 256,  # Compressed stable context
        # Optional: auxiliary layer inputs
        use_early_layer: bool = False,  # Layer 1 for spatial dynamics
        use_pre_final_layer: bool = False,  # Layer 15 for semantic
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.time_dim = time_dim
        self.cond_dim = cond_dim
        self.use_stable_context = use_stable_context
        self.use_early_layer = use_early_layer
        self.use_pre_final_layer = use_pre_final_layer

        # Time embedding (from RectifiedFlow)
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # Previous condition embeddings (keep all)
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, cond_dim // 2),
            nn.GELU(),
        )
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, cond_dim // 2),
            nn.GELU(),
        )
        self.horizon_embed = nn.Embedding(max_horizon + 1, cond_dim // 4)

        # NEW: Stable context encoder (layer 8 -> compressed)
        if use_stable_context:
            self.stable_encoder = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, stable_context_dim),
            )
            # Pool stable context to global vector
            self.stable_pool = nn.Sequential(
                nn.Linear(stable_context_dim, stable_context_dim),
                nn.GELU(),
            )

        # Optional: Early layer encoder (layer 1 - spatial dynamics)
        if use_early_layer:
            self.early_encoder = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, cond_dim // 4),
            )

        # Optional: Pre-final layer encoder (layer 15 - semantic)
        if use_pre_final_layer:
            self.pre_final_encoder = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, cond_dim // 4),
            )

        # Calculate total condition dimension
        total_cond_dim = time_dim + cond_dim // 2 + cond_dim // 2 + cond_dim // 4
        if use_stable_context:
            total_cond_dim += stable_context_dim
        if use_early_layer:
            total_cond_dim += cond_dim // 4
        if use_pre_final_layer:
            total_cond_dim += cond_dim // 4

        # Fuse all conditions
        self.cond_fuse = nn.Sequential(
            nn.Linear(total_cond_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input projection (layer 16 features)
        self.input_proj = nn.Linear(feature_dim, hidden_dim)

        # Optional: Cross-attention with stable context
        if use_stable_context:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )
            self.cross_attn_norm = nn.LayerNorm(hidden_dim)
            self.stable_proj = nn.Linear(stable_context_dim, hidden_dim)

        # Residual MLP blocks
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, hidden_dim * 2, time_dim, dropout)
            for _ in range(num_blocks)
        ])

        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, feature_dim)

        # Initialize output to zero for residual-like start
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        self._count_parameters()

    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        config = f"stable={self.use_stable_context}, early={self.use_early_layer}, pre_final={self.use_pre_final_layer}"
        print(f"StableDynamicFlowModel: {total/1e6:.2f}M params ({config})")

    def forward(
        self,
        x: torch.Tensor,                    # [B, seq_len, feature_dim] layer 16
        t: torch.Tensor,                    # [B] timesteps
        state: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        horizon: Optional[torch.Tensor] = None,
        # NEW: multi-layer inputs
        stable_features: Optional[torch.Tensor] = None,  # [B, seq_len, feature_dim] layer 8
        early_features: Optional[torch.Tensor] = None,   # [B, seq_len, feature_dim] layer 1
        pre_final_features: Optional[torch.Tensor] = None,  # [B, seq_len, feature_dim] layer 15
    ) -> torch.Tensor:
        """
        Predict velocity field for layer 16, conditioned on all available information.
        """
        B = x.shape[0]
        device = x.device

        # Handle input shape compatibility
        if x.dim() == 4:
            x = x.view(B, self.seq_len, self.feature_dim)

        # Time embedding
        t_normalized = t / 999.0
        t_emb = get_timestep_embedding(t_normalized, self.time_dim)
        t_emb = self.time_embed(t_emb)  # [B, time_dim]

        # Previous condition embeddings
        if state is not None:
            state_emb = self.state_embed(state)
        else:
            state_emb = torch.zeros(B, self.cond_dim // 2, device=device)

        if action is not None:
            action_emb = self.action_embed(action)
        else:
            action_emb = torch.zeros(B, self.cond_dim // 2, device=device)

        if horizon is not None:
            horizon_emb = self.horizon_embed(horizon.long())
        else:
            horizon_emb = torch.zeros(B, self.cond_dim // 4, device=device)

        # Collect all conditions
        cond_parts = [t_emb, state_emb, action_emb, horizon_emb]

        # NEW: Stable context from layer 8
        stable_context_seq = None
        if self.use_stable_context and stable_features is not None:
            # Encode stable features
            stable_encoded = self.stable_encoder(stable_features)  # [B, seq, stable_dim]
            stable_context_seq = stable_encoded  # Keep for cross-attention

            # Pool to global condition
            stable_pooled = stable_encoded.mean(dim=1)  # [B, stable_dim]
            stable_cond = self.stable_pool(stable_pooled)
            cond_parts.append(stable_cond)
        elif self.use_stable_context:
            # No stable features provided, use zeros
            cond_parts.append(torch.zeros(B, 256, device=device))

        # Optional: Early layer (spatial dynamics)
        if self.use_early_layer and early_features is not None:
            early_encoded = self.early_encoder(early_features)
            early_pooled = early_encoded.mean(dim=1)
            cond_parts.append(early_pooled)
        elif self.use_early_layer:
            cond_parts.append(torch.zeros(B, self.cond_dim // 4, device=device))

        # Optional: Pre-final layer (semantic)
        if self.use_pre_final_layer and pre_final_features is not None:
            pre_final_encoded = self.pre_final_encoder(pre_final_features)
            pre_final_pooled = pre_final_encoded.mean(dim=1)
            cond_parts.append(pre_final_pooled)
        elif self.use_pre_final_layer:
            cond_parts.append(torch.zeros(B, self.cond_dim // 4, device=device))

        # Fuse all conditions
        cond = torch.cat(cond_parts, dim=-1)
        cond = self.cond_fuse(cond)  # [B, time_dim]

        # Forward through network
        h = self.input_proj(x)  # [B, seq_len, hidden_dim]

        # Cross-attention with stable context (if available)
        if self.use_stable_context and stable_context_seq is not None:
            stable_kv = self.stable_proj(stable_context_seq)  # [B, seq, hidden_dim]
            h_attn, _ = self.cross_attn(h, stable_kv, stable_kv)
            h = self.cross_attn_norm(h + h_attn)

        # Residual blocks
        for block in self.blocks:
            h = block(h, cond)

        # Output velocity
        h = self.output_norm(h)
        velocity = self.output_proj(h)

        return velocity


class FeatureVelocityModelFast(nn.Module):
    """
    Ultra-lightweight velocity model for fast inference (<5ms).

    Uses simple MLP without complex conditioning for maximum speed.
    Processes features token-wise with shared MLP + global condition.
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_dim: int = 1024,
        cond_dim: int = 64,
        state_dim: int = 128,
        action_dim: int = 128,
        max_horizon: int = 10,
    ):
        super().__init__()

        self.feature_dim = feature_dim

        # Simple condition encoding
        self.cond_encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim + cond_dim, hidden_dim),
            nn.GELU(),
        )

        # Time embedding (simple)
        self.time_embed = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.GELU(),
        )

        # Horizon embedding
        self.horizon_embed = nn.Embedding(max_horizon + 1, cond_dim // 2)

        # Simple MLP: feature + cond -> velocity
        # Process all tokens in parallel
        self.velocity_net = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # Initialize output small
        nn.init.xavier_uniform_(self.velocity_net[-1].weight, gain=0.1)
        nn.init.zeros_(self.velocity_net[-1].bias)

        self._count_parameters()

    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"FeatureVelocityModelFast: {total/1e6:.2f}M params")

    def forward(
        self,
        x: torch.Tensor,           # [B, seq_len, feature_dim]
        t: torch.Tensor,           # [B] timesteps
        state: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        horizon: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, seq_len, _ = x.shape
        device = x.device

        # Time embedding
        t_normalized = (t / 999.0).unsqueeze(-1)  # [B, 1]
        t_emb = self.time_embed(t_normalized)  # [B, cond_dim]

        # Condition embedding
        if state is None:
            state = torch.zeros(B, 128, device=device)
        if action is None:
            action = torch.zeros(B, 128, device=device)
        if horizon is None:
            horizon = torch.ones(B, device=device).long()

        h_emb = self.horizon_embed(horizon)  # [B, cond_dim//2]

        # Combine conditions
        cond_input = torch.cat([state, action, t_emb, h_emb], dim=-1)
        cond = self.cond_encoder(cond_input)  # [B, hidden_dim]

        # Expand condition to all tokens
        cond_expand = cond.unsqueeze(1).expand(-1, seq_len, -1)  # [B, seq_len, hidden_dim]

        # Concatenate features with condition
        x_cond = torch.cat([x, cond_expand], dim=-1)  # [B, seq_len, feature_dim + hidden_dim]

        # Predict velocity
        velocity = self.velocity_net(x_cond)  # [B, seq_len, feature_dim]

        return velocity


# ============================================================================
# Rectified Flow Wrapper (adapted from sde_lib.py)
# ============================================================================

class RectifiedFlowBridge:
    """
    Rectified Flow for feature dynamics prediction.

    Adapted from RectifiedFlow to work with VLM features instead of images.
    """

    def __init__(
        self,
        init_type: str = 'gaussian',
        noise_scale: float = 1.0,
        reflow_flag: bool = False,
        reflow_t_schedule: str = 'uniform',
        reflow_loss: str = 'l2',
        sigma_var: float = 0.0,
        sample_N: int = 1,  # Number of sampling steps (1 for one-step generation)
    ):
        self.init_type = init_type
        self.noise_scale = noise_scale
        self.sigma_t = lambda t: (1. - t) * sigma_var
        self.sample_N = sample_N

        self.reflow_flag = reflow_flag
        self.reflow_t_schedule = reflow_t_schedule
        self.reflow_loss = reflow_loss

        print(f'RectifiedFlowBridge: init={init_type}, noise_scale={noise_scale}, '
              f'reflow={reflow_flag}, sample_N={sample_N}')

    @property
    def T(self):
        return 1.0

    def get_z0(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Get initial noise distribution.
        For feature prediction: z0 = current features (not noise!)

        For standard RF: z0 ~ N(0, I)
        For our case: z0 = features_t0 (source features)
        """
        if self.init_type == 'gaussian':
            return torch.randn_like(batch) * self.noise_scale
        elif self.init_type == 'source':
            # Use source features directly (for reflow/distillation)
            return batch
        else:
            raise NotImplementedError(f"Init type {self.init_type} not implemented")

    def get_train_tuple(
        self,
        z0: torch.Tensor,      # Source (noise or features_t0)
        z1: torch.Tensor,      # Target (features_t1)
        eps: float = 1e-3,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get training tuple (t, x_t, target_velocity) for rectified flow.

        x_t = t * z1 + (1 - t) * z0
        target = z1 - z0
        """
        B = z0.shape[0]
        device = z0.device

        if self.reflow_flag:
            if self.reflow_t_schedule == 't0':
                # Distill for t = 0 (1-step generation)
                t = torch.zeros(B, device=device) * (self.T - eps) + eps
            elif self.reflow_t_schedule == 't1':
                # Reverse distill for t = 1
                t = torch.ones(B, device=device) * (self.T - eps)
            elif self.reflow_t_schedule == 'uniform':
                # Standard reflow training
                t = torch.rand(B, device=device) * (self.T - eps) + eps
            else:
                raise NotImplementedError(f"Schedule {self.reflow_t_schedule} not implemented")
        else:
            # Standard rectified flow: uniform t
            t = torch.rand(B, device=device) * (self.T - eps) + eps

        # Interpolate
        t_expand = t.view(-1, *([1] * (z0.dim() - 1)))
        x_t = t_expand * z1 + (1. - t_expand) * z0

        # Target velocity
        target = z1 - z0

        return t, x_t, target

    @torch.no_grad()
    def sample_euler(
        self,
        model: nn.Module,
        z0: torch.Tensor,
        N: Optional[int] = None,
        state: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        horizon: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample using Euler method.

        z_1 = z_0 + sum_{i=0}^{N-1} v(z_t, t) * dt

        For N=1 (one-step): z_1 = z_0 + v(z_0, 0)
        """
        if N is None:
            N = self.sample_N

        dt = 1.0 / N
        x = z0.clone()

        model.eval()

        for i in range(N):
            t = torch.ones(z0.shape[0], device=z0.device) * (i / N)
            t_input = t * 999  # RectifiedFlow convention

            velocity = model(x, t_input, state=state, action=action, horizon=horizon)
            x = x + velocity * dt

        return x

    @torch.no_grad()
    def sample_one_step(
        self,
        model: nn.Module,
        z0: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        horizon: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        One-step generation: z_1 = z_0 + v(z_0, 0)

        This is the key for fast inference!
        """
        model.eval()
        t = torch.zeros(z0.shape[0], device=z0.device)
        t_input = t * 999  # Will be 0

        velocity = model(z0, t_input, state=state, action=action, horizon=horizon)
        return z0 + velocity


# ============================================================================
# Loss Function
# ============================================================================

class RectifiedFlowLoss(nn.Module):
    """
    Loss function for Rectified Flow training.

    Standard: L2 loss on velocity
    Optional: Cosine similarity for monitoring
    """

    def __init__(self, loss_type: str = 'l2'):
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        pred_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss.

        pred_velocity: [B, seq_len, feature_dim]
        target_velocity: [B, seq_len, feature_dim]
        mask: [B, seq_len] optional attention mask
        """
        if self.loss_type == 'l2':
            if mask is not None:
                mask_expand = mask.unsqueeze(-1)
                loss = ((pred_velocity - target_velocity) ** 2 * mask_expand).sum()
                loss = loss / mask_expand.sum().clamp(min=1)
            else:
                loss = F.mse_loss(pred_velocity, target_velocity)
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented")

        # Metrics
        with torch.no_grad():
            # Velocity MSE
            vel_mse = F.mse_loss(pred_velocity, target_velocity)

            # Feature cosine (if we add velocity to source)
            # This is a proxy metric
            cos_sim = F.cosine_similarity(
                pred_velocity.flatten(1),
                target_velocity.flatten(1),
                dim=1
            ).mean()

        metrics = {
            'loss': loss,
            'velocity_mse': vel_mse,
            'velocity_cosine': cos_sim,
        }

        return loss, metrics


# ============================================================================
# Factory Function
# ============================================================================

def create_rectified_flow_bridge(
    feature_dim: int = 2048,
    seq_len: int = 204,
    hidden_dim: int = 512,
    num_blocks: int = 2,
    state_dim: int = 128,
    action_dim: int = 128,
    device: str = "cuda",
) -> Tuple[FeatureVelocityModel, RectifiedFlowBridge]:
    """Create model and flow wrapper."""

    model = FeatureVelocityModel(
        feature_dim=feature_dim,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        state_dim=state_dim,
        action_dim=action_dim,
    ).to(device)

    flow = RectifiedFlowBridge(
        init_type='gaussian',
        noise_scale=1.0,
        sample_N=1,
    )

    return model, flow


def create_stable_dynamic_bridge(
    feature_dim: int = 2048,
    seq_len: int = 204,
    hidden_dim: int = 512,
    num_blocks: int = 2,
    state_dim: int = 128,
    action_dim: int = 128,
    # Stable-dynamic options
    use_stable_context: bool = True,
    use_early_layer: bool = False,
    use_pre_final_layer: bool = False,
    device: str = "cuda",
) -> Tuple[StableDynamicFlowModel, RectifiedFlowBridge]:
    """
    Create Stable-Dynamic Flow model.

    Configurations:
    - Minimal: use_stable_context=True only (layer 8 as anchor)
    - Full: all layers [1, 8, 15, 16]
    """
    model = StableDynamicFlowModel(
        feature_dim=feature_dim,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        state_dim=state_dim,
        action_dim=action_dim,
        use_stable_context=use_stable_context,
        use_early_layer=use_early_layer,
        use_pre_final_layer=use_pre_final_layer,
    ).to(device)

    flow = RectifiedFlowBridge(
        init_type='gaussian',
        noise_scale=1.0,
        sample_N=1,
    )

    return model, flow


def create_multilayer_flow_bridge(
    feature_dim: int = 2048,
    seq_len: int = 204,
    num_layers: int = 4,
    hidden_dim: int = 512,
    num_blocks: int = 2,
    state_dim: int = 128,
    action_dim: int = 128,
    layer_weighting: str = "learned",
    device: str = "cuda",
) -> Tuple[MultiLayerFeatureVelocityModel, RectifiedFlowBridge]:
    """Create multi-layer model and flow wrapper."""

    model = MultiLayerFeatureVelocityModel(
        feature_dim=feature_dim,
        seq_len=seq_len,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        state_dim=state_dim,
        action_dim=action_dim,
        layer_weighting=layer_weighting,
    ).to(device)

    flow = RectifiedFlowBridge(
        init_type='gaussian',
        noise_scale=1.0,
        sample_N=1,
    )

    return model, flow


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("Testing RectifiedFlowBridge...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model and flow
    model, flow = create_rectified_flow_bridge(device=device)

    # Test data
    B, seq_len, feat_dim = 4, 204, 2048
    z0 = torch.randn(B, seq_len, feat_dim, device=device)  # Source
    z1 = torch.randn(B, seq_len, feat_dim, device=device)  # Target
    state = torch.randn(B, 128, device=device)
    action = torch.randn(B, 128, device=device)
    horizon = torch.ones(B, device=device).long()

    # Test training forward
    t, x_t, target = flow.get_train_tuple(z0, z1)
    print(f"Training tuple: t={t.shape}, x_t={x_t.shape}, target={target.shape}")

    pred_velocity = model(x_t, t * 999, state=state, action=action, horizon=horizon)
    print(f"Predicted velocity: {pred_velocity.shape}")

    # Test loss
    criterion = RectifiedFlowLoss()
    loss, metrics = criterion(pred_velocity, target)
    print(f"Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")

    # Test one-step sampling
    with torch.no_grad():
        z1_pred = flow.sample_one_step(model, z0, state=state, action=action, horizon=horizon)
        print(f"One-step prediction: {z1_pred.shape}")

    # Benchmark latency
    import time
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = flow.sample_one_step(model, z0, state=state, action=action, horizon=horizon)

        if device == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        n_iters = 100
        for _ in range(n_iters):
            _ = flow.sample_one_step(model, z0, state=state, action=action, horizon=horizon)

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = (time.time() - start) / n_iters * 1000
        print(f"One-step inference latency: {elapsed:.2f}ms")

    print("Test passed!")


# ============================================================================
# Action-Aware Quantization-Inspired Enhancements
# Based on QVLA: "Not All Channels Are Equal" (ICLR 2026)
# ============================================================================

class ChannelImportanceModule(nn.Module):
    """
    Channel importance weighting following QVLA methodology.

    QVLA insight: "Not All Channels Are Equal" - channels that cause larger
    action deviations when perturbed should receive higher importance weights
    during training (i.e., errors in these channels are penalized more).

    Supports three modes:
    1. Learnable: Start uniform, learn weights during training
    2. Fixed QVLA: Load precomputed sensitivity weights from QVLA analysis
    3. Hybrid: Initialize from QVLA, fine-tune during training

    The weighting follows: weighted_loss = sum(w_c * error_c)
    where w_c is proportional to channel c's action sensitivity.
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        seq_len: int = 204,
        learn_per_position: bool = False,
        init_uniform: bool = True,
        sensitivity_path: Optional[str] = None,  # Path to QVLA sensitivity .npy
        fixed_weights: bool = False,  # If True, don't learn weights
        temperature: float = 1.0,  # Softmax temperature for weight sharpness
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.learn_per_position = learn_per_position
        self.fixed_weights = fixed_weights
        self.temperature = temperature

        # Initialize importance logits
        if learn_per_position:
            init_logits = torch.zeros(seq_len, feature_dim)
        else:
            init_logits = torch.zeros(feature_dim)

        # Load QVLA sensitivity weights if provided
        if sensitivity_path is not None:
            self._load_qvla_sensitivity(init_logits, sensitivity_path)
        elif not init_uniform:
            # Random initialization
            nn.init.normal_(init_logits, mean=0, std=0.1)

        if fixed_weights:
            # Register as buffer (not learned)
            self.register_buffer('importance_logits', init_logits)
        else:
            # Register as parameter (learned)
            self.importance_logits = nn.Parameter(init_logits)

    def _load_qvla_sensitivity(self, sensitivity_tensor: torch.Tensor, path: str):
        """Load precomputed QVLA sensitivity weights."""
        import numpy as np

        if path.endswith('.npy'):
            sensitivity = np.load(path)
        elif path.endswith('.npz'):
            data = np.load(path)
            # Use combined sensitivity if available
            if 'combined' in data:
                sensitivity = data['combined']
            elif 'zero_out' in data:
                sensitivity = data['zero_out']
            else:
                sensitivity = list(data.values())[0]
        else:
            raise ValueError(f"Unknown sensitivity file format: {path}")

        sensitivity = torch.from_numpy(sensitivity).float()
        sensitivity = sensitivity.clamp(min=0)  # Ensure non-negative

        # Store raw sensitivity values directly (no log transform)
        # Normalization happens in get_importance_weights()
        if self.learn_per_position:
            sensitivity_tensor.copy_(sensitivity.unsqueeze(0).expand(self.seq_len, -1))
        else:
            sensitivity_tensor.copy_(sensitivity)

        print(f"Loaded QVLA sensitivity from {path}")
        print(f"  Raw range: [{sensitivity.min():.6f}, {sensitivity.max():.6f}]")
        print(f"  Dynamic range: {sensitivity.max() / (sensitivity.min() + 1e-10):.1f}x")

    def get_importance_weights(self) -> torch.Tensor:
        """
        Get normalized importance weights using direct scaling.

        Returns weights that:
        - Sum to feature_dim (preserving loss scale)
        - Higher weight = more important channel (penalize errors more)
        - Preserves relative importance ratios from QVLA sensitivity
        """
        # Direct normalization - preserves relative ratios better than softmax
        raw_weights = self.importance_logits.clamp(min=0)

        # Apply temperature as power scaling (T<1 sharpens, T>1 smooths)
        if self.temperature != 1.0:
            raw_weights = raw_weights ** (1.0 / self.temperature)

        # Normalize to sum to feature_dim (preserves loss scale)
        if self.learn_per_position:
            weights = raw_weights / (raw_weights.sum(dim=-1, keepdim=True) + 1e-10) * self.feature_dim
        else:
            weights = raw_weights / (raw_weights.sum() + 1e-10) * self.feature_dim

        return weights

    def forward(self, error: torch.Tensor) -> torch.Tensor:
        """
        Apply QVLA-style channel importance weighting to error tensor.

        Args:
            error: [B, seq_len, feature_dim] squared error

        Returns:
            Weighted error [B, seq_len, feature_dim]

        Note: Higher importance = higher weight = error in this channel
        contributes more to loss (following QVLA's action-sensitivity principle)
        """
        weights = self.get_importance_weights()

        if self.learn_per_position:
            return error * weights.unsqueeze(0)
        else:
            return error * weights.unsqueeze(0).unsqueeze(0)

    def get_top_channels(self, k: int = 100) -> torch.Tensor:
        """Get indices of top-k most important channels."""
        weights = self.get_importance_weights()
        if self.learn_per_position:
            weights = weights.mean(dim=0)
        _, indices = torch.topk(weights, k)
        return indices

    def get_importance_stats(self) -> Dict[str, float]:
        """Get statistics about current importance weights."""
        weights = self.get_importance_weights()
        if self.learn_per_position:
            weights = weights.mean(dim=0)

        weights_np = weights.detach().cpu().numpy()
        sorted_w = np.sort(weights_np)[::-1]

        return {
            'mean': float(weights_np.mean()),
            'std': float(weights_np.std()),
            'min': float(weights_np.min()),
            'max': float(weights_np.max()),
            'dynamic_range': float(weights_np.max() / (weights_np.min() + 1e-10)),
            'top10_frac': float(sorted_w[:10].sum() / sorted_w.sum()),
            'top50_frac': float(sorted_w[:50].sum() / sorted_w.sum()),
            'top100_frac': float(sorted_w[:100].sum() / sorted_w.sum()),
        }


class ActionReconstructionHead(nn.Module):
    """
    Auxiliary head to reconstruct actions from predicted features.

    This provides an action-space signal to guide the latent bridge training,
    ensuring predicted features are useful for action prediction.
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        seq_len: int = 204,
        action_dim: int = 7,  # Robot action dimension
        action_horizon: int = 16,  # Number of action steps to predict
        hidden_dim: int = 512,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.action_dim = action_dim
        self.action_horizon = action_horizon

        # Pool features to fixed size then predict actions
        self.feature_pool = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Attention pooling over sequence
        self.attn_pool = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
        )
        self.query = nn.Parameter(torch.randn(1, action_horizon, hidden_dim))

        # Action prediction
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict actions from features.

        Args:
            features: [B, seq_len, feature_dim]

        Returns:
            actions: [B, action_horizon, action_dim]
        """
        B = features.shape[0]

        # Project features
        h = self.feature_pool(features)  # [B, seq_len, hidden_dim]

        # Attention pooling to action horizon
        query = self.query.expand(B, -1, -1)  # [B, action_horizon, hidden_dim]
        pooled, _ = self.attn_pool(query, h, h)  # [B, action_horizon, hidden_dim]

        # Predict actions
        actions = self.action_head(pooled)  # [B, action_horizon, action_dim]

        return actions


class ActionAwareLoss(nn.Module):
    """
    Action-aware loss function combining:
    1. Feature reconstruction loss (MSE with QVLA channel importance weighting)
    2. Action reconstruction auxiliary loss

    Based on QVLA's insight: "Not All Channels Are Equal"
    Channels with higher action sensitivity receive higher loss weights.
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        seq_len: int = 204,
        action_dim: int = 7,
        action_horizon: int = 16,
        use_channel_importance: bool = True,
        use_action_aux_loss: bool = True,
        action_loss_weight: float = 0.1,
        learn_per_position: bool = False,
        # QVLA-specific parameters
        sensitivity_path: Optional[str] = None,  # Path to precomputed sensitivity
        fixed_importance: bool = False,  # If True, don't learn importance weights
        importance_temperature: float = 1.0,  # Sharpness of importance distribution
    ):
        super().__init__()

        self.use_channel_importance = use_channel_importance
        self.use_action_aux_loss = use_action_aux_loss
        self.action_loss_weight = action_loss_weight

        # Channel importance module with QVLA support
        if use_channel_importance:
            self.channel_importance = ChannelImportanceModule(
                feature_dim=feature_dim,
                seq_len=seq_len,
                learn_per_position=learn_per_position,
                sensitivity_path=sensitivity_path,
                fixed_weights=fixed_importance,
                temperature=importance_temperature,
            )

        # Action reconstruction head
        if use_action_aux_loss:
            self.action_head = ActionReconstructionHead(
                feature_dim=feature_dim,
                seq_len=seq_len,
                action_dim=action_dim,
                action_horizon=action_horizon,
            )

    def forward(
        self,
        pred_velocity: torch.Tensor,  # [B, seq_len, feature_dim]
        target_velocity: torch.Tensor,  # [B, seq_len, feature_dim]
        target_features: Optional[torch.Tensor] = None,  # [B, seq_len, feature_dim] for action aux
        target_actions: Optional[torch.Tensor] = None,  # [B, action_horizon, action_dim]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute action-aware loss.

        Args:
            pred_velocity: Predicted velocity from flow model
            target_velocity: True velocity (x1 - x0)
            target_features: Target features (x1) for action reconstruction
            target_actions: Ground truth actions for auxiliary loss

        Returns:
            loss: Scalar loss
            metrics: Dict of loss components
        """
        metrics = {}

        # Feature reconstruction loss
        error = (pred_velocity - target_velocity) ** 2  # [B, seq_len, feature_dim]

        if self.use_channel_importance:
            # Apply learned channel importance weighting
            weighted_error = self.channel_importance(error)
            feature_loss = weighted_error.mean()

            # Track importance statistics
            with torch.no_grad():
                weights = self.channel_importance.get_importance_weights()
                if weights.dim() > 1:
                    weights = weights.mean(dim=0)
                metrics['importance_std'] = weights.std().item()
                metrics['importance_max'] = weights.max().item()
                metrics['importance_min'] = weights.min().item()
        else:
            feature_loss = error.mean()

        metrics['feature_loss'] = feature_loss.item()

        total_loss = feature_loss

        # Action reconstruction auxiliary loss
        if self.use_action_aux_loss and target_features is not None and target_actions is not None:
            # Predict actions from target features
            pred_actions = self.action_head(target_features)

            # Match action horizon
            min_horizon = min(pred_actions.shape[1], target_actions.shape[1])
            action_loss = F.mse_loss(
                pred_actions[:, :min_horizon],
                target_actions[:, :min_horizon]
            )

            metrics['action_loss'] = action_loss.item()
            total_loss = total_loss + self.action_loss_weight * action_loss

        metrics['total_loss'] = total_loss.item()

        return total_loss, metrics

    def get_channel_importance(self) -> Optional[torch.Tensor]:
        """Get current channel importance weights for analysis."""
        if self.use_channel_importance:
            return self.channel_importance.get_importance_weights()
        return None


# ============================================================================
# DiT-style Stable-Dynamic Flow Model
# ============================================================================

class DiTStableDynamicFlowModel(nn.Module):
    """
    DiT-style Stable-Dynamic Flow Model.

    Based on "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023).

    Key features:
    - AdaLN-Zero for conditioning (time, state, action, horizon)
    - Self-attention within layer 16 features
    - Cross-attention to layer 8 (stable) features
    - Zero-initialized output for residual-like start
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        seq_len: int = 204,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_blocks: int = 4,  # DiT typically uses more blocks
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        # Conditioning
        state_dim: int = 128,
        action_dim: int = 128,
        max_horizon: int = 10,
        # QVLA
        use_channel_importance: bool = True,
        sensitivity_path: Optional[str] = None,
        fixed_importance: bool = False,
        importance_temperature: float = 1.0,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.use_channel_importance = use_channel_importance

        # Input projection (layer 16 dynamic features)
        self.input_proj = nn.Linear(feature_dim, hidden_dim)

        # Stable context projection (layer 8 frozen features)
        self.stable_proj = nn.Linear(feature_dim, hidden_dim)

        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Condition embeddings
        time_dim = hidden_dim
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )
        self.horizon_embed = nn.Embedding(max_horizon + 1, hidden_dim // 4)

        # Fuse all conditions to hidden_dim
        cond_in_dim = hidden_dim + hidden_dim // 2 + hidden_dim // 2 + hidden_dim // 4
        self.cond_fuse = nn.Sequential(
            nn.Linear(cond_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # DiT blocks with cross-attention to stable context
        self.blocks = nn.ModuleList([
            DiTCrossBlock(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_blocks)
        ])

        # Final layer
        self.final_layer = DiTFinalLayer(hidden_dim, feature_dim)

        # Channel importance (QVLA)
        if use_channel_importance:
            self.channel_importance = ChannelImportanceModule(
                feature_dim=feature_dim,
                seq_len=seq_len,
                learn_per_position=False,
                sensitivity_path=sensitivity_path,
                fixed_weights=fixed_importance,
                temperature=importance_temperature,
            )

        self._count_parameters()

    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"DiTStableDynamicFlowModel: {total/1e6:.2f}M params")
        print(f"  - hidden_dim={self.hidden_dim}, blocks={len(self.blocks)}")

    def forward(
        self,
        x: torch.Tensor,           # [B, seq_len, feature_dim] - layer 16 (dynamic)
        t: torch.Tensor,           # [B] timesteps (scaled by 999)
        state: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        horizon: Optional[torch.Tensor] = None,
        stable_features: Optional[torch.Tensor] = None,  # [B, seq_len, feature_dim] - layer 8
        early_features: Optional[torch.Tensor] = None,   # unused, for compatibility
        pre_final_features: Optional[torch.Tensor] = None,  # unused
    ) -> torch.Tensor:
        """Predict velocity field for layer 16."""
        B = x.shape[0]
        device = x.device

        # Handle input shape
        if x.dim() == 4:
            x = x.view(B, self.seq_len, self.feature_dim)

        # Project inputs
        x = self.input_proj(x) + self.pos_embed  # [B, seq_len, hidden_dim]

        if stable_features is not None:
            if stable_features.dim() == 4:
                stable_features = stable_features.view(B, self.seq_len, self.feature_dim)
            stable = self.stable_proj(stable_features) + self.pos_embed
        else:
            stable = x  # fallback

        # Time embedding
        t_normalized = t / 999.0
        t_emb = get_timestep_embedding(t_normalized, self.hidden_dim)
        t_emb = self.time_embed(t_emb)  # [B, hidden_dim]

        # Condition embeddings
        if state is not None:
            state_emb = self.state_embed(state)
        else:
            state_emb = torch.zeros(B, self.hidden_dim // 2, device=device)

        if action is not None:
            action_emb = self.action_embed(action)
        else:
            action_emb = torch.zeros(B, self.hidden_dim // 2, device=device)

        if horizon is not None:
            horizon_emb = self.horizon_embed(horizon.clamp(0, 10))
        else:
            horizon_emb = torch.zeros(B, self.hidden_dim // 4, device=device)

        # Fuse conditions
        c = torch.cat([t_emb, state_emb, action_emb, horizon_emb], dim=-1)
        c = self.cond_fuse(c)  # [B, hidden_dim]

        # DiT blocks
        for block in self.blocks:
            x = block(x, stable, c)

        # Final layer
        velocity = self.final_layer(x, c)

        return velocity

    def get_importance_weights(self) -> Optional[torch.Tensor]:
        """Get channel importance weights."""
        if self.use_channel_importance:
            return self.channel_importance.get_importance_weights()
        return None

    def compute_loss(
        self,
        pred_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
        pred_features: Optional[torch.Tensor] = None,
        target_actions: Optional[torch.Tensor] = None,
        action_loss_weight: float = 0.1,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss with channel importance weighting.

        Args:
            pred_velocity: [B, seq_len, feature_dim] predicted velocity
            target_velocity: [B, seq_len, feature_dim] target velocity
            pred_features: unused (for API compatibility)
            target_actions: unused (for API compatibility)
            action_loss_weight: unused (for API compatibility)

        Returns:
            loss, metrics dict
        """
        metrics = {}

        # Feature reconstruction loss with channel importance
        error = (pred_velocity - target_velocity) ** 2

        if self.use_channel_importance:
            # Apply channel importance weighting
            weighted_error = self.channel_importance(error)
            loss = weighted_error.mean()

            # Track importance stats
            with torch.no_grad():
                weights = self.channel_importance.get_importance_weights()
                metrics['ch_importance_std'] = weights.std().item()
                metrics['ch_importance_max'] = weights.max().item()
                metrics['ch_importance_min'] = weights.min().item()
        else:
            loss = error.mean()

        metrics['feature_loss'] = loss.item()
        metrics['total_loss'] = loss.item()

        return loss, metrics


def create_dit_stable_dynamic_bridge(
    feature_dim: int = 2048,
    seq_len: int = 204,
    hidden_dim: int = 512,
    num_heads: int = 8,
    num_blocks: int = 4,
    mlp_ratio: float = 4.0,
    state_dim: int = 128,
    action_dim: int = 128,
    # QVLA options
    use_channel_importance: bool = True,
    sensitivity_path: Optional[str] = None,
    fixed_importance: bool = False,
    importance_temperature: float = 1.0,
    device: str = "cuda",
) -> Tuple[DiTStableDynamicFlowModel, RectifiedFlowBridge]:
    """
    Create DiT-style Stable-Dynamic Flow model.

    Based on "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023).
    """
    model = DiTStableDynamicFlowModel(
        feature_dim=feature_dim,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        mlp_ratio=mlp_ratio,
        state_dim=state_dim,
        action_dim=action_dim,
        use_channel_importance=use_channel_importance,
        sensitivity_path=sensitivity_path,
        fixed_importance=fixed_importance,
        importance_temperature=importance_temperature,
    ).to(device)

    flow = RectifiedFlowBridge(
        init_type='gaussian',
        noise_scale=1.0,
        sample_N=1,
    )

    return model, flow


class ActionAwareStableDynamicFlowModel(StableDynamicFlowModel):
    """
    Stable-Dynamic Flow Model with Action-Aware enhancements.

    Extends StableDynamicFlowModel with:
    1. Integrated channel importance learning
    2. Action reconstruction auxiliary head
    3. Action-space aligned loss computation

    Based on insights from QVLA (ICLR 2026).
    """

    def __init__(
        self,
        # Inherited params
        feature_dim: int = 2048,
        seq_len: int = 204,
        hidden_dim: int = 512,
        time_dim: int = 128,
        cond_dim: int = 128,
        num_blocks: int = 2,
        dropout: float = 0.0,
        state_dim: int = 128,
        action_dim: int = 128,
        max_horizon: int = 10,
        use_stable_context: bool = True,
        stable_context_dim: int = 256,
        use_early_layer: bool = False,
        use_pre_final_layer: bool = False,
        # Action-aware params
        robot_action_dim: int = 7,
        action_horizon: int = 16,
        use_channel_importance: bool = True,
        use_action_aux_loss: bool = True,
        action_loss_weight: float = 0.1,
        # QVLA-specific params
        sensitivity_path: Optional[str] = None,
        fixed_importance: bool = False,
        importance_temperature: float = 1.0,
    ):
        super().__init__(
            feature_dim=feature_dim,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            cond_dim=cond_dim,
            num_blocks=num_blocks,
            dropout=dropout,
            state_dim=state_dim,
            action_dim=action_dim,
            max_horizon=max_horizon,
            use_stable_context=use_stable_context,
            stable_context_dim=stable_context_dim,
            use_early_layer=use_early_layer,
            use_pre_final_layer=use_pre_final_layer,
        )

        self.robot_action_dim = robot_action_dim
        self.action_horizon = action_horizon
        self.use_channel_importance = use_channel_importance
        self.use_action_aux_loss = use_action_aux_loss
        self.action_loss_weight = action_loss_weight

        # Channel importance module with QVLA support
        if use_channel_importance:
            self.channel_importance = ChannelImportanceModule(
                feature_dim=feature_dim,
                seq_len=seq_len,
                learn_per_position=False,
                sensitivity_path=sensitivity_path,
                fixed_weights=fixed_importance,
                temperature=importance_temperature,
            )

        # Action reconstruction head (from predicted features)
        if use_action_aux_loss:
            self.action_head = ActionReconstructionHead(
                feature_dim=feature_dim,
                seq_len=seq_len,
                action_dim=robot_action_dim,
                action_horizon=action_horizon,
                hidden_dim=hidden_dim,
            )

        print(f"ActionAwareStableDynamicFlowModel (QVLA-style):")
        print(f"  - channel_importance={use_channel_importance}")
        print(f"  - action_aux={use_action_aux_loss}, weight={action_loss_weight}")
        if sensitivity_path:
            print(f"  - QVLA sensitivity from: {sensitivity_path}")
            print(f"  - fixed_importance={fixed_importance}, temperature={importance_temperature}")

    def compute_loss(
        self,
        pred_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
        pred_features: Optional[torch.Tensor] = None,
        target_actions: Optional[torch.Tensor] = None,
        action_loss_weight: float = 0.1,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute action-aware loss.

        Args:
            pred_velocity: [B, seq_len, feature_dim]
            target_velocity: [B, seq_len, feature_dim]
            pred_features: [B, seq_len, feature_dim] predicted x1 (for action aux)
            target_actions: [B, action_horizon, robot_action_dim]

        Returns:
            loss, metrics dict
        """
        metrics = {}

        # Feature reconstruction loss
        error = (pred_velocity - target_velocity) ** 2

        if self.use_channel_importance:
            weighted_error = self.channel_importance(error)
            feature_loss = weighted_error.mean()

            # Track importance stats
            with torch.no_grad():
                weights = self.channel_importance.get_importance_weights()
                metrics['ch_importance_std'] = weights.std().item()
                metrics['ch_importance_max'] = weights.max().item()
        else:
            feature_loss = error.mean()

        metrics['feature_loss'] = feature_loss.item()
        total_loss = feature_loss

        # Action auxiliary loss
        if self.use_action_aux_loss and pred_features is not None and target_actions is not None:
            pred_actions = self.action_head(pred_features)
            min_h = min(pred_actions.shape[1], target_actions.shape[1])
            action_loss = F.mse_loss(pred_actions[:, :min_h], target_actions[:, :min_h])

            metrics['action_aux_loss'] = action_loss.item()
            total_loss = total_loss + action_loss_weight * action_loss

        metrics['total_loss'] = total_loss.item()

        return total_loss, metrics

    def get_channel_importance_analysis(self) -> Dict[str, torch.Tensor]:
        """Get channel importance for analysis."""
        if not self.use_channel_importance:
            return {}

        weights = self.channel_importance.get_importance_weights()
        top_indices = self.channel_importance.get_top_channels(k=100)

        return {
            'weights': weights.detach().cpu(),
            'top_100_indices': top_indices.detach().cpu(),
            'mean': weights.mean().item(),
            'std': weights.std().item(),
        }
