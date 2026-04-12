"""
Single-step Flow Bridge for GR00T.

Key changes from v1:
- No horizon embedding (h=1 only)
- Simpler architecture
- Designed for autoregressive inference
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


def get_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / half)
    args = timesteps[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class RMSNorm(nn.Module):
    """RMS Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(2, dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return self.scale * x / (norm + self.eps)


class AdaLN(nn.Module):
    """Adaptive Layer Normalization."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = RMSNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim * 2)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scale, shift = self.proj(c).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), scale


class CrossAttention(nn.Module):
    """Cross attention with stable features."""
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.kv_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(context).view(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)

        return self.out_proj(out)


class DiTBlock(nn.Module):
    """DiT block with cross-attention to stable features."""
    def __init__(self, hidden_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.adaln1 = AdaLN(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        self.adaln2 = AdaLN(hidden_dim)
        self.cross_attn = CrossAttention(hidden_dim, num_heads)

        self.adaln3 = AdaLN(hidden_dim)
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, stable: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x_norm, _ = self.adaln1(x, c)
        x = x + self.attn(x_norm, x_norm, x_norm, need_weights=False)[0]

        # Cross-attention with stable features
        x_norm, _ = self.adaln2(x, c)
        x = x + self.cross_attn(x_norm, stable)

        # MLP
        x_norm, _ = self.adaln3(x, c)
        x = x + self.mlp(x_norm)

        return x


class FinalLayer(nn.Module):
    """Final layer with AdaLN."""
    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.norm = RMSNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(c).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.out(x)


class SingleStepBridge(nn.Module):
    """
    Single-step feature prediction model.

    Predicts z_{t+1} from z_t using flow matching.
    No horizon embedding - designed for h=1 only.
    At inference, run autoregressively for multi-step.
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        seq_len: int = 204,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        state_dim: int = 8,
        action_dim: int = 7,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # Input projections
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.stable_proj = nn.Linear(feature_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Condition embeddings (no horizon!)
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

        # Condition fusion (no horizon, so smaller input)
        # t_emb: hidden_dim, state_emb: hidden_dim//2, action_emb: hidden_dim//2
        self.cond_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])

        # Output
        self.final_layer = FinalLayer(hidden_dim, feature_dim)

        self._init_weights()
        self._print_params()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_init)
        # Zero init final layer
        nn.init.zeros_(self.final_layer.out.weight)
        nn.init.zeros_(self.final_layer.out.bias)

    def _print_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        print(f"SingleStepBridge: {n_params/1e6:.2f}M params")
        print(f"  hidden_dim={self.hidden_dim}, layers={len(self.blocks)}")

    def forward(
        self,
        x: torch.Tensor,           # [B, seq_len, feature_dim] - current features
        t: torch.Tensor,           # [B] timesteps (0-1 range for flow matching)
        stable_features: torch.Tensor,  # [B, seq_len, feature_dim] - stable layer
        state: Optional[torch.Tensor] = None,   # [B, state_dim]
        action: Optional[torch.Tensor] = None,  # [B, action_dim]
    ) -> torch.Tensor:
        """Predict velocity field."""
        B = x.shape[0]
        device = x.device

        # Project inputs
        x = self.input_proj(x) + self.pos_embed
        stable = self.stable_proj(stable_features) + self.pos_embed

        # Time embedding (scale t to match training)
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

        # Fuse conditions
        c = torch.cat([t_emb, state_emb, action_emb], dim=-1)
        c = self.cond_fuse(c)

        # DiT blocks
        for block in self.blocks:
            x = block(x, stable, c)

        # Output velocity
        velocity = self.final_layer(x, c)

        return velocity

    @torch.no_grad()
    def predict_one_step(
        self,
        z_t: torch.Tensor,
        stable_features: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        One-step prediction: z_{t+1} = z_t + v(z_t, t=0)
        """
        B = z_t.shape[0]
        t = torch.zeros(B, device=z_t.device)
        velocity = self.forward(z_t, t, stable_features, state, action)
        return z_t + velocity

    @torch.no_grad()
    def predict_n_steps(
        self,
        z_0: torch.Tensor,
        stable_features: torch.Tensor,
        n_steps: int,
        states: Optional[torch.Tensor] = None,   # [B, n_steps, state_dim] or [B, state_dim]
        actions: Optional[torch.Tensor] = None,  # [B, n_steps, action_dim] or [B, action_dim]
    ) -> torch.Tensor:
        """
        Autoregressive prediction for n steps.

        Args:
            z_0: Initial features [B, seq_len, feature_dim]
            stable_features: Stable layer features [B, seq_len, feature_dim]
            n_steps: Number of steps to predict
            states: Robot states for each step
            actions: Robot actions for each step

        Returns:
            z_n: Predicted features after n steps
        """
        z_t = z_0
        B = z_0.shape[0]

        for step in range(n_steps):
            # Get state/action for this step
            if states is not None:
                if states.dim() == 3:
                    state = states[:, min(step, states.shape[1]-1)]
                else:
                    state = states
            else:
                state = None

            if actions is not None:
                if actions.dim() == 3:
                    action = actions[:, min(step, actions.shape[1]-1)]
                else:
                    action = actions
            else:
                action = None

            # Predict next step
            z_t = self.predict_one_step(z_t, stable_features, state, action)

        return z_t
