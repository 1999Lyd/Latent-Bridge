"""
Flow matching utilities with t=0 focused sampling.

Key insight: Standard rectified flow samples t uniformly in [0, 1], but
inference uses t=0. This causes poor performance at t=0.

Solution: Sample t=0 with 50% probability during training.
"""

import torch
from typing import Tuple


def sample_t_focused(
    batch_size: int,
    device: torch.device,
    t0_prob: float = 0.5,
) -> torch.Tensor:
    """
    Sample timesteps with focus on t=0.

    Args:
        batch_size: number of samples
        device: torch device
        t0_prob: probability of sampling t=0 (default 0.5)

    Returns:
        [B] tensor of timesteps in [0, 1]
    """
    t = torch.rand(batch_size, device=device)

    # With probability t0_prob, set t=0
    mask = torch.rand(batch_size, device=device) < t0_prob
    t[mask] = 0.0

    return t


def get_train_tuple(
    z0: torch.Tensor,
    z1: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get training tuple for rectified flow.

    Args:
        z0: [B, ...] source features (t=0)
        z1: [B, ...] target features (t=1)
        t: [B] timesteps

    Returns:
        t: [B] timesteps (unchanged)
        z_t: [B, ...] interpolated features
        target: [B, ...] target velocity (z1 - z0)
    """
    # Expand t for broadcasting
    t_expand = t.view(-1, *([1] * (z0.dim() - 1)))

    # Linear interpolation: z_t = (1 - t) * z0 + t * z1
    z_t = (1 - t_expand) * z0 + t_expand * z1

    # Target velocity is always z1 - z0 (the straight line)
    target = z1 - z0

    return t, z_t, target


def one_step_sample(
    model: torch.nn.Module,
    z0: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """
    One-step generation: z1 = z0 + v(z0, t=0).

    Args:
        model: velocity model
        z0: [B, ...] starting features
        **kwargs: additional model inputs (state, action, horizon, stable)

    Returns:
        [B, ...] predicted z1
    """
    B = z0.shape[0]
    device = z0.device

    # t=0 for one-step generation
    t = torch.zeros(B, device=device)

    # Predict velocity
    velocity = model(z0, t, **kwargs)

    # Add velocity to get prediction
    return z0 + velocity
