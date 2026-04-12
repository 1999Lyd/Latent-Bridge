"""
π0.5 Multi-Layer KV Cache Bridge.

Adapts GR00T's DiT Stable-Dynamic bridge for π0.5's KV cache structure.

Key differences from GR00T:
1. GR00T: Predicts layer 16 hidden state from layer 10 (single layer → single layer)
2. π0.5: Predicts all 18 layers of KV cache (multi-layer → multi-layer)

π0.5 KV cache structure:
- 18 layers, each with 1 KV head (GQA)
- Per layer: [batch, 1, seq_len, head_dim=256]
- Total: [batch, 18, seq_len, 256] when stacked

Design: Shared DiT bridge with layer conditioning
- Same weights process all layers
- Layer embedding conditions which layer is being predicted
- Cross-attention between layers for information sharing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass


# ============================================================================
# Utility Functions (from GR00T bridge)
# ============================================================================

def get_timestep_embedding(timesteps, embedding_dim):
    """Sinusoidal time embeddings."""
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


def modulate(x, shift, scale):
    """AdaLN modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ============================================================================
# DiT Blocks (adapted from GR00T)
# ============================================================================

class DiTBlock(nn.Module):
    """DiT block with adaLN-Zero conditioning."""

    def __init__(self, hidden_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

        # AdaLN: 6 params (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)

        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out

        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        return x


class DiTCrossLayerBlock(nn.Module):
    """
    DiT block with cross-attention between layers.

    For π0.5: allows information flow between different KV layers.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()

        # Self-attention within layer
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # Cross-attention to other layers (context)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm_ctx = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

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

        # AdaLN: 9 params
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 9 * hidden_dim),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: [B, seq_len, hidden_dim] - current layer features
        context: [B, seq_len, hidden_dim] - aggregated other layers
        c: [B, hidden_dim] - conditioning
        """
        mods = self.adaLN_modulation(c).chunk(9, dim=-1)
        shift_sa, scale_sa, gate_sa = mods[0], mods[1], mods[2]
        shift_ca, scale_ca, gate_ca = mods[3], mods[4], mods[5]
        shift_mlp, scale_mlp, gate_mlp = mods[6], mods[7], mods[8]

        # Self-attention
        x_norm = modulate(self.norm1(x), shift_sa, scale_sa)
        sa_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + gate_sa.unsqueeze(1) * sa_out

        # Cross-attention to context
        x_norm = modulate(self.norm2(x), shift_ca, scale_ca)
        ctx_norm = self.norm_ctx(context)
        ca_out, _ = self.cross_attn(x_norm, ctx_norm, ctx_norm)
        x = x + gate_ca.unsqueeze(1) * ca_out

        # FFN
        x_norm = modulate(self.norm3(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        return x


class DiTFinalLayer(nn.Module):
    """Final output layer with adaLN."""

    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


# ============================================================================
# π0.5 Multi-Layer KV Bridge
# ============================================================================

class Pi05MultiLayerKVBridge(nn.Module):
    """
    Multi-layer KV cache bridge for π0.5.

    Predicts fresh KV cache from stale KV cache for all layers.
    Uses shared weights with layer conditioning (parameter efficient).

    Input: stale_kv [B, num_layers, seq_len, kv_dim]
    Output: delta_kv [B, num_layers, seq_len, kv_dim] (velocity)

    Architecture:
    1. Project each layer's KV to hidden_dim
    2. Add layer embedding + positional embedding
    3. Process with DiT blocks (shared across layers)
    4. Cross-attention between layers for information sharing
    5. Project back to KV dimension
    """

    def __init__(
        self,
        # π0.5 KV cache dimensions
        num_layers: int = 18,
        kv_dim: int = 256,  # head_dim (since num_kv_heads=1)
        seq_len: int = 512,  # KV cache sequence length
        # Bridge architecture
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_blocks: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        # Conditioning
        state_dim: int = 8,  # Robot state dimension
        max_horizon: int = 10,
        # Layer selection (predict subset of layers)
        predict_layers: Optional[List[int]] = None,  # None = all layers
        # Channel importance (QVLA-style)
        use_channel_importance: bool = False,
        sensitivity_path: Optional[str] = None,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.kv_dim = kv_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # Which layers to predict
        if predict_layers is None:
            self.predict_layers = list(range(num_layers))
        else:
            self.predict_layers = predict_layers
        self.num_predict_layers = len(self.predict_layers)

        # Input projection (KV → hidden)
        self.input_proj = nn.Linear(kv_dim, hidden_dim)

        # Layer embedding (which layer is being processed)
        self.layer_embed = nn.Embedding(num_layers, hidden_dim)

        # Positional embedding (sequence position)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # State embedding
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )

        # Horizon embedding
        self.horizon_embed = nn.Embedding(max_horizon + 1, hidden_dim // 4)

        # Fuse conditions
        cond_in_dim = hidden_dim + hidden_dim // 2 + hidden_dim // 4 + hidden_dim  # + layer_emb
        self.cond_fuse = nn.Sequential(
            nn.Linear(cond_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # DiT blocks with cross-layer attention
        self.blocks = nn.ModuleList([
            DiTCrossLayerBlock(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_blocks)
        ])

        # Layer aggregation for cross-attention context
        self.layer_aggregator = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.layer_agg_norm = nn.LayerNorm(hidden_dim)

        # Output projection (hidden → KV)
        self.final_layer = DiTFinalLayer(hidden_dim, kv_dim)

        # Channel importance (optional)
        self.use_channel_importance = use_channel_importance
        if use_channel_importance and sensitivity_path:
            self._load_channel_importance(sensitivity_path)

        self._count_parameters()

    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"Pi05MultiLayerKVBridge: {total/1e6:.2f}M params")
        print(f"  - num_layers={self.num_layers}, predict_layers={self.num_predict_layers}")
        print(f"  - hidden_dim={self.hidden_dim}, blocks={len(self.blocks)}")

    def _load_channel_importance(self, path: str):
        """Load precomputed channel importance from sensitivity analysis."""
        import numpy as np
        data = np.load(path)
        if 'hidden_combined' in data:
            importance = torch.from_numpy(data['hidden_combined']).float()
        else:
            importance = torch.ones(self.kv_dim)

        # Normalize to [0.5, 1.5] range
        importance = importance / (importance.mean() + 1e-10)
        importance = importance.clamp(0.5, 2.0)

        self.register_buffer('channel_importance', importance)
        print(f"Loaded channel importance from {path}")

    def _aggregate_layers(self, layer_features: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Aggregate information from other layers for cross-attention.

        layer_features: [B, num_layers, seq_len, hidden_dim]
        layer_idx: current layer being processed

        Returns: [B, seq_len, hidden_dim] aggregated context
        """
        B, L, S, D = layer_features.shape

        # Pool each layer to single vector
        layer_pooled = layer_features.mean(dim=2)  # [B, L, D]

        # Query is current layer, keys/values are all layers
        query = layer_pooled[:, layer_idx:layer_idx+1, :]  # [B, 1, D]

        # Attention over layers
        ctx, _ = self.layer_aggregator(query, layer_pooled, layer_pooled)  # [B, 1, D]
        ctx = self.layer_agg_norm(ctx)

        # Expand to sequence length
        ctx = ctx.expand(-1, S, -1)  # [B, S, D]

        return ctx

    def forward(
        self,
        stale_kv: torch.Tensor,  # [B, num_layers, seq_len, kv_dim]
        t: torch.Tensor,  # [B] timesteps (0-999)
        state: Optional[torch.Tensor] = None,  # [B, state_dim]
        horizon: Optional[torch.Tensor] = None,  # [B]
    ) -> torch.Tensor:
        """
        Predict KV cache velocity (delta from stale to fresh).

        Returns: velocity [B, num_layers, seq_len, kv_dim]
        """
        B, L, S, D = stale_kv.shape
        device = stale_kv.device

        # Project all layers to hidden space
        # [B, L, S, D] -> [B, L, S, hidden_dim]
        h = self.input_proj(stale_kv)

        # Add positional embedding (shared across layers)
        h = h + self.pos_embed[:, :S, :].unsqueeze(1)

        # Time embedding
        t_normalized = t / 999.0
        t_emb = get_timestep_embedding(t_normalized, self.hidden_dim)
        t_emb = self.time_embed(t_emb)  # [B, hidden_dim]

        # State embedding
        if state is not None:
            state_emb = self.state_embed(state)
        else:
            state_emb = torch.zeros(B, self.hidden_dim // 2, device=device)

        # Horizon embedding
        if horizon is not None:
            horizon_emb = self.horizon_embed(horizon.clamp(0, 10))
        else:
            horizon_emb = torch.zeros(B, self.hidden_dim // 4, device=device)

        # Process each layer with shared weights
        outputs = []

        for layer_idx in self.predict_layers:
            # Get layer embedding
            layer_emb = self.layer_embed(torch.tensor([layer_idx], device=device))
            layer_emb = layer_emb.expand(B, -1)  # [B, hidden_dim]

            # Fuse all conditions
            c = torch.cat([t_emb, state_emb, horizon_emb, layer_emb], dim=-1)
            c = self.cond_fuse(c)  # [B, hidden_dim]

            # Get current layer features
            x = h[:, layer_idx]  # [B, S, hidden_dim]

            # Aggregate context from other layers
            context = self._aggregate_layers(h, layer_idx)

            # Process through DiT blocks
            for block in self.blocks:
                x = block(x, context, c)

            # Project to output
            out = self.final_layer(x, c)  # [B, S, kv_dim]
            outputs.append(out)

        # Stack outputs [B, num_predict_layers, S, kv_dim]
        velocity = torch.stack(outputs, dim=1)

        # If predicting subset, expand to full layers (others = 0)
        if len(self.predict_layers) < self.num_layers:
            full_velocity = torch.zeros(B, self.num_layers, S, D, device=device)
            for i, layer_idx in enumerate(self.predict_layers):
                full_velocity[:, layer_idx] = velocity[:, i]
            velocity = full_velocity

        return velocity

    def predict_fresh_kv(
        self,
        stale_kv: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        horizon: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        One-step prediction: fresh_kv = stale_kv + velocity(stale_kv, t=0)
        """
        B = stale_kv.shape[0]
        device = stale_kv.device

        t = torch.zeros(B, device=device)
        velocity = self(stale_kv, t, state, horizon)

        return stale_kv + velocity


class Pi05ParallelKVBridge(nn.Module):
    """
    Parallelized multi-layer KV bridge for π0.5.

    Key optimization: Process all layers in parallel by treating layers as batch dim.
    - Input: [B, L, S, D] -> reshape to [B*L, S, D]
    - Add layer embedding to distinguish layers
    - Process through DiT blocks (all layers in parallel)
    - Reshape back to [B, L, S, D]

    Cross-layer information: Global context pooled across all layers.
    """

    def __init__(
        self,
        num_layers: int = 18,
        kv_dim: int = 256,
        seq_len: int = 512,
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_blocks: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        state_dim: int = 8,
        max_horizon: int = 10,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.kv_dim = kv_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(kv_dim, hidden_dim)

        # Layer embedding (critical for distinguishing layers)
        self.layer_embed = nn.Embedding(num_layers, hidden_dim)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # State embedding
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Horizon embedding
        self.horizon_embed = nn.Embedding(max_horizon + 1, hidden_dim)

        # Global context: pool across layers for cross-layer information
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # DiT blocks (standard, no cross-attention - cross-layer info via embeddings)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_blocks)
        ])

        # Output projection
        self.final_layer = DiTFinalLayer(hidden_dim, kv_dim)

        self._count_parameters()

    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"Pi05ParallelKVBridge: {total/1e6:.2f}M params")
        print(f"  - num_layers={self.num_layers}, hidden_dim={self.hidden_dim}")
        print(f"  - blocks={len(self.blocks)} (parallel over all layers)")

    def forward(
        self,
        stale_kv: torch.Tensor,  # [B, num_layers, seq_len, kv_dim]
        t: torch.Tensor,  # [B] timesteps
        state: Optional[torch.Tensor] = None,
        horizon: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, S, D = stale_kv.shape
        device = stale_kv.device

        # Project to hidden dim: [B, L, S, D] -> [B, L, S, H]
        h = self.input_proj(stale_kv)

        # Add positional embedding (broadcast over layers)
        h = h + self.pos_embed[:, :S, :].unsqueeze(1)

        # Add layer embedding: [L, H] -> [1, L, 1, H]
        layer_emb = self.layer_embed(torch.arange(L, device=device))
        h = h + layer_emb.unsqueeze(0).unsqueeze(2)

        # Global context: pool across all layers and positions
        global_ctx = h.mean(dim=(1, 2))  # [B, H]
        global_ctx = self.global_pool(global_ctx)  # [B, H]

        # Time embedding
        t_normalized = t / 999.0
        t_emb = get_timestep_embedding(t_normalized, self.hidden_dim)
        t_emb = self.time_embed(t_emb)  # [B, H]

        # State embedding
        if state is not None:
            state_emb = self.state_embed(state)  # [B, H]
        else:
            state_emb = torch.zeros(B, self.hidden_dim, device=device)

        # Horizon embedding
        if horizon is not None:
            horizon_emb = self.horizon_embed(horizon.clamp(0, 10))  # [B, H]
        else:
            horizon_emb = torch.zeros(B, self.hidden_dim, device=device)

        # Combined conditioning
        c = t_emb + state_emb + horizon_emb + global_ctx  # [B, H]

        # Reshape for parallel processing: [B, L, S, H] -> [B*L, S, H]
        h = h.reshape(B * L, S, self.hidden_dim)

        # Expand conditioning for all layers: [B, H] -> [B*L, H]
        c_expanded = c.unsqueeze(1).expand(-1, L, -1).reshape(B * L, self.hidden_dim)

        # Process through DiT blocks (all layers in parallel!)
        for block in self.blocks:
            h = block(h, c_expanded)

        # Output projection
        out = self.final_layer(h, c_expanded)  # [B*L, S, D]

        # Reshape back: [B*L, S, D] -> [B, L, S, D]
        velocity = out.reshape(B, L, S, D)

        return velocity

    def predict_fresh_kv(
        self,
        stale_kv: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        horizon: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One-step prediction."""
        B = stale_kv.shape[0]
        t = torch.zeros(B, device=stale_kv.device)
        velocity = self(stale_kv, t, state, horizon)
        return stale_kv + velocity


class Pi05FastMLPBridge(nn.Module):
    """
    Ultra-fast MLP-based bridge for π0.5 KV cache prediction.

    Key design for speed:
    1. NO attention - pure MLP (O(n) not O(n²))
    2. Process all layers and positions in parallel
    3. Minimal overhead from conditioning

    Target: <20ms to match Action Expert latency
    """

    def __init__(
        self,
        num_layers: int = 18,
        kv_dim: int = 256,
        hidden_dim: int = 1024,
        num_blocks: int = 4,
        state_dim: int = 8,
        max_horizon: int = 10,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.kv_dim = kv_dim
        self.hidden_dim = hidden_dim

        # Layer embedding (added to input)
        self.layer_embed = nn.Embedding(num_layers, kv_dim)

        # Time embedding (small, added to hidden)
        self.time_embed = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.SiLU(),
        )

        # State embedding
        self.state_embed = nn.Linear(state_dim, hidden_dim)

        # Main MLP: processes each position independently
        # Input: kv_dim (with layer embedding added)
        # Output: kv_dim (velocity)
        self.input_proj = nn.Linear(kv_dim, hidden_dim)

        # Residual MLP blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            ))

        # Output
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, kv_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        self._count_parameters()

    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"Pi05FastMLPBridge: {total/1e6:.2f}M params")
        print(f"  - hidden_dim={self.hidden_dim}, blocks={len(self.blocks)}")

    def forward(
        self,
        stale_kv: torch.Tensor,  # [B, num_layers, seq_len, kv_dim]
        t: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        horizon: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, S, D = stale_kv.shape
        device = stale_kv.device

        # Add layer embedding to input: [L, D] -> [1, L, 1, D]
        layer_emb = self.layer_embed(torch.arange(L, device=device))
        x = stale_kv + layer_emb.unsqueeze(0).unsqueeze(2)

        # Flatten to [B*L*S, D] for efficient MLP
        x = x.reshape(B * L * S, D)

        # Project to hidden
        h = self.input_proj(x)  # [B*L*S, hidden_dim]

        # Add conditioning (broadcast to all positions)
        t_normalized = t / 999.0
        t_emb = get_timestep_embedding(t_normalized, 64)
        t_emb = self.time_embed(t_emb)  # [B, hidden_dim]

        if state is not None:
            state_emb = self.state_embed(state)  # [B, hidden_dim]
            cond = t_emb + state_emb
        else:
            cond = t_emb

        # Broadcast conditioning to all positions
        cond = cond.unsqueeze(1).unsqueeze(1).expand(B, L, S, -1)
        cond = cond.reshape(B * L * S, -1)
        h = h + cond

        # MLP blocks with residual connections
        for block in self.blocks:
            h = h + block(h)

        # Output
        h = self.output_norm(h)
        velocity = self.output_proj(h)  # [B*L*S, D]

        # Reshape back
        velocity = velocity.reshape(B, L, S, D)

        return velocity

    def predict_fresh_kv(
        self,
        stale_kv: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        horizon: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One-step prediction."""
        B = stale_kv.shape[0]
        t = torch.zeros(B, device=stale_kv.device)
        velocity = self(stale_kv, t, state, horizon)
        return stale_kv + velocity


class Pi05SharedLayerBridge(nn.Module):
    """
    Simplified per-layer bridge with shared weights.

    More efficient than full cross-layer attention.
    Processes each layer independently with layer-conditioned MLP.
    """

    def __init__(
        self,
        num_layers: int = 18,
        kv_dim: int = 256,
        seq_len: int = 512,
        hidden_dim: int = 512,
        num_blocks: int = 4,
        state_dim: int = 8,
        max_horizon: int = 10,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.kv_dim = kv_dim
        self.hidden_dim = hidden_dim

        # Layer embedding
        self.layer_embed = nn.Embedding(num_layers, hidden_dim)

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # State embedding
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.SiLU(),
        )

        # Horizon embedding
        self.horizon_embed = nn.Embedding(max_horizon + 1, hidden_dim // 4)

        # Shared velocity predictor (processes token-wise)
        cond_dim = hidden_dim + hidden_dim // 2 + hidden_dim // 4
        self.velocity_net = nn.Sequential(
            nn.Linear(kv_dim + cond_dim, hidden_dim),
            nn.GELU(),
        )

        # Residual blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(num_blocks)
        ])
        self.block_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_blocks)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, kv_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        self._count_parameters()

    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"Pi05SharedLayerBridge: {total/1e6:.2f}M params")

    def forward(
        self,
        stale_kv: torch.Tensor,  # [B, num_layers, seq_len, kv_dim]
        t: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        horizon: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, S, D = stale_kv.shape
        device = stale_kv.device

        # Time embedding
        t_normalized = t / 999.0
        t_emb = get_timestep_embedding(t_normalized, self.hidden_dim)
        t_emb = self.time_embed(t_emb)

        # State embedding
        if state is not None:
            state_emb = self.state_embed(state)
        else:
            state_emb = torch.zeros(B, self.hidden_dim // 2, device=device)

        # Horizon embedding
        if horizon is not None:
            horizon_emb = self.horizon_embed(horizon.clamp(0, 10))
        else:
            horizon_emb = torch.zeros(B, self.hidden_dim // 4, device=device)

        outputs = []

        for layer_idx in range(L):
            # Layer embedding
            layer_emb = self.layer_embed(torch.tensor([layer_idx], device=device))
            layer_emb = layer_emb.expand(B, -1)

            # Combined condition
            cond = torch.cat([t_emb + layer_emb, state_emb, horizon_emb], dim=-1)  # [B, cond_dim]
            cond = cond.unsqueeze(1).expand(-1, S, -1)  # [B, S, cond_dim]

            # Input: KV + condition
            x = stale_kv[:, layer_idx]  # [B, S, D]
            x = torch.cat([x, cond], dim=-1)  # [B, S, D + cond_dim]

            # Process
            h = self.velocity_net(x)  # [B, S, hidden_dim]

            for block, norm in zip(self.blocks, self.block_norms):
                h = h + block(norm(h))

            # Output
            out = self.output_proj(h)  # [B, S, D]
            outputs.append(out)

        return torch.stack(outputs, dim=1)  # [B, L, S, D]


# ============================================================================
# Rectified Flow for KV Bridge
# ============================================================================

class KVRectifiedFlow:
    """Rectified Flow wrapper for KV cache prediction."""

    def __init__(self, sample_N: int = 1):
        self.sample_N = sample_N

    def get_train_tuple(
        self,
        stale_kv: torch.Tensor,  # z0
        fresh_kv: torch.Tensor,  # z1
        eps: float = 1e-3,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get (t, x_t, target_velocity) for training.

        x_t = t * z1 + (1-t) * z0
        target = z1 - z0
        """
        B = stale_kv.shape[0]
        device = stale_kv.device

        t = torch.rand(B, device=device) * (1.0 - eps) + eps
        t_expand = t.view(-1, 1, 1, 1)

        x_t = t_expand * fresh_kv + (1 - t_expand) * stale_kv
        target = fresh_kv - stale_kv

        return t * 999, x_t, target  # Scale t to 0-999

    @torch.no_grad()
    def sample_one_step(
        self,
        model: nn.Module,
        stale_kv: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        horizon: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One-step generation: fresh = stale + v(stale, t=0)"""
        B = stale_kv.shape[0]
        t = torch.zeros(B, device=stale_kv.device)

        velocity = model(stale_kv, t, state, horizon)
        return stale_kv + velocity


# ============================================================================
# Loss Function
# ============================================================================

class KVBridgeLoss(nn.Module):
    """Loss function for KV bridge training."""

    def __init__(
        self,
        use_layer_importance: bool = True,
        layer_importance: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.use_layer_importance = use_layer_importance

        if layer_importance is not None:
            self.register_buffer('layer_importance', layer_importance)
        else:
            self.layer_importance = None

    def forward(
        self,
        pred_velocity: torch.Tensor,  # [B, L, S, D]
        target_velocity: torch.Tensor,
        layer_sensitivity: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss with optional layer importance weighting.
        """
        error = (pred_velocity - target_velocity) ** 2

        # Layer importance weighting
        if self.use_layer_importance and layer_sensitivity is not None:
            # Normalize sensitivity to weights
            weights = layer_sensitivity / (layer_sensitivity.sum() + 1e-10)
            weights = weights.view(1, -1, 1, 1)
            error = error * weights * error.shape[1]  # Scale to preserve magnitude

        loss = error.mean()

        # Metrics
        with torch.no_grad():
            per_layer_loss = error.mean(dim=(0, 2, 3))
            cosine = F.cosine_similarity(
                pred_velocity.flatten(1),
                target_velocity.flatten(1),
                dim=1
            ).mean()

        metrics = {
            'loss': loss.item(),
            'cosine': cosine.item(),
        }

        return loss, metrics


# ============================================================================
# Factory Functions
# ============================================================================

# Model size configurations for attention-based bridges
BRIDGE_CONFIGS = {
    "tiny": {  # ~8M params - for testing
        "hidden_dim": 256,
        "num_blocks": 4,
        "num_heads": 8,
        "mlp_ratio": 4.0,
    },
    "small": {  # ~18M params
        "hidden_dim": 384,
        "num_blocks": 6,
        "num_heads": 8,
        "mlp_ratio": 4.0,
    },
    "medium": {  # ~32M params
        "hidden_dim": 512,
        "num_blocks": 6,
        "num_heads": 8,
        "mlp_ratio": 4.0,
    },
    "medium-large": {  # ~55M params
        "hidden_dim": 640,
        "num_blocks": 6,
        "num_heads": 10,
        "mlp_ratio": 4.0,
    },
    "large": {  # ~94M params
        "hidden_dim": 768,
        "num_blocks": 8,
        "num_heads": 12,
        "mlp_ratio": 4.0,
    },
    "xlarge": {  # ~166M params
        "hidden_dim": 1024,
        "num_blocks": 8,
        "num_heads": 16,
        "mlp_ratio": 4.0,
    },
}

# Fast MLP bridge configurations (no attention, O(n) complexity)
MLP_BRIDGE_CONFIGS = {
    "mlp-small": {  # ~5M params, target <5ms
        "hidden_dim": 512,
        "num_blocks": 3,
    },
    "mlp-medium": {  # ~15M params, target <10ms
        "hidden_dim": 1024,
        "num_blocks": 4,
    },
    "mlp-large": {  # ~30M params, target <15ms
        "hidden_dim": 1536,
        "num_blocks": 4,
    },
    "mlp-xlarge": {  # ~60M params, target <20ms
        "hidden_dim": 2048,
        "num_blocks": 4,
    },
}


def create_pi05_kv_bridge(
    num_layers: int = 18,
    kv_dim: int = 256,
    seq_len: int = 512,
    state_dim: int = 8,
    model_size: str = "mlp-large",  # Default to fast MLP
    bridge_type: str = "mlp",  # "mlp" (fast), "parallel", "full", "shared"
    device: str = "cuda",
    # Override specific params if needed
    hidden_dim: Optional[int] = None,
    num_blocks: Optional[int] = None,
    num_heads: Optional[int] = None,
) -> Tuple[nn.Module, KVRectifiedFlow]:
    """
    Create π0.5 KV bridge model and flow.

    Bridge types:
    - "mlp": Ultra-fast MLP bridge, NO attention (RECOMMENDED, <20ms)
    - "parallel": Fast parallel attention (~40-100ms)
    - "full": Cross-layer attention (slowest, ~150ms)
    - "shared": Per-layer MLP with loop

    Model sizes for MLP bridge:
    - "mlp-small" (~5M): <5ms, basic capacity
    - "mlp-medium" (~15M): <10ms, good balance
    - "mlp-large" (~30M): <15ms, recommended
    - "mlp-xlarge" (~60M): <20ms, maximum capacity

    For comparison:
    - π0.5 Action Expert (1 step): ~19ms
    - π0.5 VLM Total: ~57ms
    """
    # Determine which config to use
    if model_size.startswith("mlp-"):
        config = MLP_BRIDGE_CONFIGS.get(model_size, MLP_BRIDGE_CONFIGS["mlp-large"])
        bridge_type = "mlp"  # Force MLP type
    else:
        config = BRIDGE_CONFIGS.get(model_size, BRIDGE_CONFIGS["large"])

    # Allow overrides
    _hidden_dim = hidden_dim if hidden_dim is not None else config["hidden_dim"]
    _num_blocks = num_blocks if num_blocks is not None else config["num_blocks"]

    print(f"Creating π0.5 KV Bridge ({model_size}, {bridge_type}):")
    print(f"  hidden_dim={_hidden_dim}, blocks={_num_blocks}")

    if bridge_type == "mlp":
        model = Pi05FastMLPBridge(
            num_layers=num_layers,
            kv_dim=kv_dim,
            hidden_dim=_hidden_dim,
            num_blocks=_num_blocks,
            state_dim=state_dim,
        ).to(device)
    elif bridge_type == "parallel":
        _num_heads = num_heads if num_heads is not None else config.get("num_heads", 8)
        _mlp_ratio = config.get("mlp_ratio", 4.0)
        model = Pi05ParallelKVBridge(
            num_layers=num_layers,
            kv_dim=kv_dim,
            seq_len=seq_len,
            hidden_dim=_hidden_dim,
            num_heads=_num_heads,
            num_blocks=_num_blocks,
            mlp_ratio=_mlp_ratio,
            state_dim=state_dim,
        ).to(device)
    elif bridge_type == "full":
        _num_heads = num_heads if num_heads is not None else config.get("num_heads", 8)
        _mlp_ratio = config.get("mlp_ratio", 4.0)
        model = Pi05MultiLayerKVBridge(
            num_layers=num_layers,
            kv_dim=kv_dim,
            seq_len=seq_len,
            hidden_dim=_hidden_dim,
            num_heads=_num_heads,
            num_blocks=_num_blocks,
            mlp_ratio=_mlp_ratio,
            state_dim=state_dim,
        ).to(device)
    else:
        model = Pi05SharedLayerBridge(
            num_layers=num_layers,
            kv_dim=kv_dim,
            seq_len=seq_len,
            hidden_dim=_hidden_dim,
            num_blocks=_num_blocks,
            state_dim=state_dim,
        ).to(device)

    flow = KVRectifiedFlow(sample_N=1)

    return model, flow


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Pi05 KV Bridge - Model Size Comparison")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # π0.5 dimensions
    B, L, S, D = 2, 18, 512, 256
    state_dim = 8

    # Compare model sizes
    print("\nModel size comparison:")
    print("-" * 40)
    for size_name in ["tiny", "small", "medium", "large", "xlarge"]:
        model, _ = create_pi05_kv_bridge(
            num_layers=L,
            kv_dim=D,
            seq_len=S,
            state_dim=state_dim,
            model_size=size_name,
            bridge_type="full",
            device="cpu",  # Just for param counting
        )
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  {size_name:8s}: {params:6.1f}M params")
        del model

    print("\n" + "=" * 60)
    print("Testing 'large' configuration (recommended)")
    print("=" * 60)

    # Create model with recommended size
    model, flow = create_pi05_kv_bridge(
        num_layers=L,
        kv_dim=D,
        seq_len=S,
        state_dim=state_dim,
        model_size="large",  # 60M params
        bridge_type="full",
        device=device,
    )

    # Test data
    stale_kv = torch.randn(B, L, S, D, device=device)
    fresh_kv = torch.randn(B, L, S, D, device=device)
    state = torch.randn(B, state_dim, device=device)
    horizon = torch.ones(B, device=device).long()

    # Test training
    t, x_t, target = flow.get_train_tuple(stale_kv, fresh_kv)
    pred_velocity = model(x_t, t, state, horizon)
    print(f"Pred velocity shape: {pred_velocity.shape}")

    # Test loss
    criterion = KVBridgeLoss()
    loss, metrics = criterion(pred_velocity, target)
    print(f"Loss: {loss.item():.4f}, Cosine: {metrics['cosine']:.4f}")

    # Test inference
    with torch.no_grad():
        fresh_pred = flow.sample_one_step(model, stale_kv, state, horizon)
        print(f"Predicted fresh KV shape: {fresh_pred.shape}")

    # Benchmark
    import time
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = flow.sample_one_step(model, stale_kv, state, horizon)

        if device == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        for _ in range(50):
            _ = flow.sample_one_step(model, stale_kv, state, horizon)

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = (time.time() - start) / 50 * 1000
        print(f"Inference latency: {elapsed:.2f}ms")

    print("Test passed!")
