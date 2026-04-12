#!/usr/bin/env python3
"""
π0 Bridge KV: Directly predict pre-RoPE K and V deltas for all layers.

Instead of predicting hidden states and reconstructing KV (which amplifies errors),
directly predict the KV cache entries.

Per layer output: [seq, 512] (256 for pre-RoPE K + 256 for V)
Total: 18 layers × seq × 512 = much smaller than 18 × seq × 2048

RoPE is applied to predicted K after bridge output (deterministic given position).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdaLNBlock(nn.Module):
    """DiT block with AdaLN + cross-attention. Uses nn.MHA for compatibility with saved checkpoints.
    Apply torch.compile + bfloat16 at inference for flash attention speedup."""
    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, ctx, c):
        shifts = self.adaLN(c).unsqueeze(1).chunk(6, dim=-1)
        s1, sc1, g1, s2, sc2, g2 = shifts
        h = self.norm1(x) * (1 + sc1) + s1
        x = x + g1 * self.attn(h, h, h)[0]
        h = self.norm2(x)
        x = x + self.cross_attn(h, ctx, ctx)[0]
        h = self.norm3(x) * (1 + sc2) + s2
        x = x + g2 * self.mlp(h)
        return x


class Pi0BridgeKV(nn.Module):
    """
    Predict pre-RoPE K and V deltas for all 18 Gemma layers.

    Input: fresh embedding delta + previous KV (compressed) + state/action
    Output: per-layer (pre-RoPE K delta, V delta) [18, seq, 512]
    """

    def __init__(
        self,
        kv_dim: int = 256,        # per-head KV dim (Gemma: 256)
        num_layers: int = 18,
        seq_len: int = 768,
        emb_dim: int = 2048,       # embedding dimension (for input projection)
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_blocks: int = 10,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        state_dim: int = 8,
        action_dim: int = 7,
        no_vision: bool = False,   # skip vision embedding input
        no_state: bool = False,    # skip state conditioning
        no_action: bool = False,   # skip action conditioning
    ):
        super().__init__()
        self.kv_dim = kv_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.out_dim = kv_dim * 2  # K + V per layer
        self.no_vision = no_vision
        self.no_state = no_state
        self.no_action = no_action

        if not no_vision:
            # Input: embedding delta (what changed visually)
            self.emb_delta_proj = nn.Linear(emb_dim, hidden_dim)
            # Context: current fresh embedding
            self.curr_emb_proj = nn.Linear(emb_dim, hidden_dim)

        # Previous KV summary as additional input
        # Compress prev KV [18, seq, 512] → [seq, hidden] via learned projection
        self.prev_kv_proj = nn.Linear(num_layers * kv_dim * 2, hidden_dim)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Layer embedding (which layer's KV to predict)
        self.layer_embed = nn.Embedding(num_layers, hidden_dim)

        # Conditioning
        self.cond_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )

        # Backbone
        self.blocks = nn.ModuleList([
            AdaLNBlock(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_blocks)
        ])

        # Per-layer output heads (predict K_delta + V_delta)
        self.layer_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, self.out_dim),
            ) for _ in range(num_layers)
        ])
        # Zero-init output → starts at copy baseline (delta=0)
        for head in self.layer_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

        total = sum(p.numel() for p in self.parameters())
        print(f"Pi0BridgeKV: {total/1e6:.1f}M params")
        print(f"  {num_layers} layers × {kv_dim*2} KV dim, {num_blocks} blocks, hidden={hidden_dim}")

    def forward(self, emb_delta, curr_emb, prev_kv_flat, state, action):
        """
        Args:
            emb_delta:    [B, S, emb_dim] — curr_emb - prev_emb (ignored if no_vision)
            curr_emb:     [B, S, emb_dim] — fresh embedding (ignored if no_vision)
            prev_kv_flat: [B, S, 18*512] — flattened previous KV (pre-RoPE K + V)
            state:        [B, state_dim]
            action:       [B, action_dim]

        Returns:
            kv_deltas: list of 18 tensors [B, S, 512] (pre-RoPE K_delta + V_delta)
        """
        B, S = prev_kv_flat.shape[:2]
        pos = self.pos_embed[:, :S]

        if self.no_vision:
            # No vision: input is only prev KV
            x = self.prev_kv_proj(prev_kv_flat) + pos
            ctx = x  # self-attend
        else:
            # Input: embedding delta + compressed prev KV
            x = self.emb_delta_proj(emb_delta) + self.prev_kv_proj(prev_kv_flat) + pos
            # Context: fresh embedding
            ctx = self.curr_emb_proj(curr_emb) + pos

        # Conditioning (zero out state/action if ablated)
        if self.no_state:
            state = torch.zeros_like(state)
        if self.no_action:
            action = torch.zeros_like(action)
        c = self.cond_net(torch.cat([state, action], dim=-1))

        # Backbone
        for block in self.blocks:
            x = block(x, ctx, c)

        # Per-layer output
        kv_deltas = []
        for l in range(self.num_layers):
            delta = self.layer_heads[l](x)  # [B, S, 512]
            kv_deltas.append(delta)

        return kv_deltas

    def split_kv(self, kv_delta):
        """Split [B, S, 512] into pre-RoPE K delta [B, S, 256] and V delta [B, S, 256]."""
        return kv_delta[..., :self.kv_dim], kv_delta[..., self.kv_dim:]


if __name__ == "__main__":
    model = Pi0BridgeKV(
        kv_dim=256, num_layers=18, seq_len=768, emb_dim=2048,
        hidden_dim=768, num_heads=12, num_blocks=10,
        state_dim=8, action_dim=7,
    )

    B, S = 1, 768
    emb_d = torch.randn(B, S, 2048)
    curr = torch.randn(B, S, 2048)
    prev_kv = torch.randn(B, S, 18 * 512)
    state = torch.randn(B, 8)
    action = torch.randn(B, 7)

    deltas = model(emb_d, curr, prev_kv, state, action)
    print(f"\n{len(deltas)} layer outputs, each {deltas[0].shape}")
    k_d, v_d = model.split_kv(deltas[0])
    print(f"K_delta: {k_d.shape}, V_delta: {v_d.shape}")
    print(f"Norms: K={k_d.norm():.4f}, V={v_d.norm():.4f} (should be ~0 at init)")
