"""
Clean FlowBridge model for GR00T.

Predicts velocity field for latent feature dynamics.
Uses stable layer (layer 10) as cross-attention context.
Trained with t=0 focused sampling for one-step inference.
"""

import torch
import torch.nn as nn
from typing import Optional

from ..common.dit import (
    get_timestep_embedding,
    DiTCrossBlock,
    FinalLayer,
)


class FlowBridge(nn.Module):
    """
    Clean velocity prediction model for flow matching.

    Architecture:
        - Input projection for dynamic features (layer 16)
        - Stable projection for context features (layer 10)
        - Learnable positional embeddings
        - DiT blocks with self-attention + cross-attention
        - AdaLN-Zero conditioning on [time, state, action, horizon]
        - Zero-initialized output for residual-like start

    Inference:
        z_{t+h} = z_t + v(z_t, t=0, stable, state, action, horizon)
    """

    def __init__(
        self,
        latent_dim: int = 2048,
        seq_len: int = 204,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        state_dim: int = 8,
        action_dim: int = 7,
        max_horizon: int = 5,
    ):
        """
        Args:
            latent_dim: VLM hidden dimension (2048 for GR00T)
            seq_len: sequence length (204 for GR00T)
            hidden_dim: transformer hidden dimension
            num_heads: number of attention heads
            num_layers: number of DiT blocks
            mlp_ratio: FFN hidden dim = hidden_dim * mlp_ratio
            dropout: dropout rate
            state_dim: robot state dimension
            action_dim: robot action dimension
            max_horizon: maximum prediction horizon
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # Input projections
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.stable_proj = nn.Linear(latent_dim, hidden_dim)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Conditioning embeddings
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

        # Fuse all conditions
        cond_in_dim = hidden_dim + hidden_dim // 2 + hidden_dim // 2 + hidden_dim // 4
        self.cond_fuse = nn.Sequential(
            nn.Linear(cond_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # DiT blocks with cross-attention
        self.blocks = nn.ModuleList([
            DiTCrossBlock(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        # Final layer
        self.final_layer = FinalLayer(hidden_dim, latent_dim)

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"FlowBridge: {n_params / 1e6:.2f}M params")
        print(f"  hidden_dim={hidden_dim}, num_layers={num_layers}")

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        z_stable: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        horizon: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict velocity field.

        Args:
            z_t: [B, seq_len, latent_dim] - current/interpolated features
            t: [B] - timestep (0-1 range, will be scaled internally)
            z_stable: [B, seq_len, latent_dim] - stable layer context
            state: [B, state_dim] - robot state (optional)
            action: [B, action_dim] - robot action (optional)
            horizon: [B] - prediction horizon (optional)

        Returns:
            [B, seq_len, latent_dim] - predicted velocity
        """
        B = z_t.shape[0]
        device = z_t.device

        # Handle input shape [B, seq_len, latent_dim]
        if z_t.dim() == 4:  # [B, C, H, W] format
            z_t = z_t.view(B, self.seq_len, self.latent_dim)
        if z_stable.dim() == 4:
            z_stable = z_stable.view(B, self.seq_len, self.latent_dim)

        # Project inputs
        x = self.input_proj(z_t) + self.pos_embed
        stable = self.stable_proj(z_stable) + self.pos_embed

        # Time embedding (scale t to match sinusoidal range)
        t_emb = get_timestep_embedding(t, self.hidden_dim)
        t_emb = self.time_embed(t_emb)

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
            horizon_emb = self.horizon_embed(horizon.clamp(0, 5).long())
        else:
            horizon_emb = torch.zeros(B, self.hidden_dim // 4, device=device)

        # Fuse conditions
        cond = torch.cat([t_emb, state_emb, action_emb, horizon_emb], dim=-1)
        cond = self.cond_fuse(cond)

        # DiT blocks
        for block in self.blocks:
            x = block(x, stable, cond)

        # Final layer
        velocity = self.final_layer(x, cond)

        return velocity


def create_flow_bridge(
    latent_dim: int = 2048,
    seq_len: int = 204,
    hidden_dim: int = 512,
    num_layers: int = 4,
    state_dim: int = 8,
    action_dim: int = 7,
    device: str = "cuda",
) -> FlowBridge:
    """Factory function to create FlowBridge model."""
    model = FlowBridge(
        latent_dim=latent_dim,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        state_dim=state_dim,
        action_dim=action_dim,
    ).to(device)
    return model
