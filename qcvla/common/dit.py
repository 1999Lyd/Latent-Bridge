"""
Clean DiT (Diffusion Transformer) blocks for flow matching.

Based on "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023).
Simplified implementation without channel importance or complex conditioning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def get_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal timestep embeddings.

    Args:
        timesteps: [B] tensor of timesteps (0-1 range)
        dim: embedding dimension

    Returns:
        [B, dim] embedding tensor
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization with zero-initialized modulation.

    Applies: x = x * (1 + scale) + shift, then gates output.
    """

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        # 3 outputs: shift, scale, gate
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 3),
        )
        # Zero-initialize for residual learning
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        Args:
            x: [B, seq_len, hidden_dim]
            cond: [B, cond_dim]
        Returns:
            normalized x, gate for output
        """
        shift, scale, gate = self.proj(cond).chunk(3, dim=-1)
        # Expand for sequence dimension
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
        gate = gate.unsqueeze(1)

        x = self.norm(x) * (1 + scale) + shift
        return x, gate


class DiTBlock(nn.Module):
    """
    DiT block with self-attention and FFN.

    Uses AdaLN-Zero for conditioning.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Self-attention
        self.adaln1 = AdaLNZero(hidden_dim, hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.attn_out = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)

        # FFN
        self.adaln2 = AdaLNZero(hidden_dim, hidden_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, hidden_dim]
            cond: [B, hidden_dim] conditioning
        """
        B, L, D = x.shape

        # Self-attention with AdaLN
        x_norm, gate1 = self.adaln1(x, cond)
        qkv = self.qkv(x_norm).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, L, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        out = self.attn_out(out)
        x = x + gate1 * out

        # FFN with AdaLN
        x_norm, gate2 = self.adaln2(x, cond)
        x = x + gate2 * self.ffn(x_norm)

        return x


class DiTCrossBlock(nn.Module):
    """
    DiT block with self-attention, cross-attention to stable context, and FFN.

    Uses AdaLN-Zero for conditioning.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Self-attention
        self.adaln1 = AdaLNZero(hidden_dim, hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.attn_out = nn.Linear(hidden_dim, hidden_dim)

        # Cross-attention to stable context
        self.adaln_cross = AdaLNZero(hidden_dim, hidden_dim)
        self.q_cross = nn.Linear(hidden_dim, hidden_dim)
        self.kv_cross = nn.Linear(hidden_dim, hidden_dim * 2)
        self.cross_out = nn.Linear(hidden_dim, hidden_dim)

        # FFN
        self.adaln2 = AdaLNZero(hidden_dim, hidden_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        stable: torch.Tensor,
        cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, hidden_dim] - dynamic features
            stable: [B, seq_len, hidden_dim] - stable layer context
            cond: [B, hidden_dim] - conditioning (time + state + action + horizon)
        """
        B, L, D = x.shape

        # Self-attention
        x_norm, gate1 = self.adaln1(x, cond)
        qkv = self.qkv(x_norm).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        out = self.attn_out(out)
        x = x + gate1 * out

        # Cross-attention to stable context
        x_norm, gate_cross = self.adaln_cross(x, cond)
        q = self.q_cross(x_norm).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_cross(stable).reshape(B, L, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        out = self.cross_out(out)
        x = x + gate_cross * out

        # FFN
        x_norm, gate2 = self.adaln2(x, cond)
        x = x + gate2 * self.ffn(x_norm)

        return x


class FinalLayer(nn.Module):
    """
    Final layer with AdaLN and linear projection.
    Zero-initialized for residual-like start.
    """

    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        self.out = nn.Linear(hidden_dim, out_dim)

        # Zero-initialize
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, hidden_dim]
            cond: [B, hidden_dim]
        """
        shift, scale = self.proj(cond).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.out(x)
