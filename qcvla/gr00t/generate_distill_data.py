#!/usr/bin/env python3
"""
Generate distillation data using teacher model with multi-step ODE.
Multi-GPU version with batching for faster processing.
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qcvla.model.rectified_flow_bridge import DiTStableDynamicFlowModel


def load_teacher(checkpoint_path: str, device: str) -> DiTStableDynamicFlowModel:
    """Load teacher model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})

    model = DiTStableDynamicFlowModel(
        feature_dim=2048,
        seq_len=204,
        hidden_dim=config.get('hidden_dim', 512),
        num_blocks=config.get('num_blocks', 4),
        num_heads=config.get('num_heads', 8),
        mlp_ratio=config.get('mlp_ratio', 4.0),
        state_dim=config.get('state_dim', 8),
        action_dim=config.get('action_dim', 7),
        use_channel_importance=False,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    print(f"Loaded teacher from {checkpoint_path}")
    print(f"  Config: hidden_dim={config.get('hidden_dim')}, num_blocks={config.get('num_blocks')}")

    return model


@torch.no_grad()
def ode_solve_batch(
    model: DiTStableDynamicFlowModel,
    z_0: torch.Tensor,
    stable: torch.Tensor,
    state: torch.Tensor,
    action: torch.Tensor,
    horizon: torch.Tensor,
    num_steps: int = 10,
) -> torch.Tensor:
    """
    Batched Euler ODE integration from t=0 to t=1.
    """
    dt = 1.0 / num_steps
    z_t = z_0.clone()

    for i in range(num_steps):
        t = torch.full((z_t.shape[0],), i * dt, device=z_t.device)
        velocity = model(
            z_t, t * 999,
            state=state,
            action=action,
            horizon=horizon,
            stable_features=stable
        )
        z_t = z_t + dt * velocity

    return z_t


def pad_or_truncate(x: np.ndarray, seq_len: int = 204) -> np.ndarray:
    """Pad or truncate to seq_len."""
    if x.shape[0] < seq_len:
        pad_len = seq_len - x.shape[0]
        return np.pad(x, ((0, pad_len), (0, 0)), mode='constant')
    return x[:seq_len]


def process_episode(
    ep_key: str,
    src_h5: h5py.File,
    model: DiTStableDynamicFlowModel,
    device: str,
    stable_layer_idx: int,
    target_layer_idx: int,
    num_steps: int,
    batch_size: int = 32,
):
    """Process a single episode and return distillation data."""
    ep_group = src_h5[ep_key]
    features = ep_group['multilayer_features'][:]
    states = ep_group['states'][:]
    actions = ep_group['actions'][:]

    T = features.shape[0]

    # Collect all samples for this episode
    all_z0 = []
    all_stable = []
    all_state = []
    all_action = []
    all_horizon = []
    all_info = []

    for horizon in [1, 2, 3]:
        for t0 in range(T - horizon):
            t1 = t0 + horizon

            z_0_np = pad_or_truncate(features[t0, target_layer_idx])
            stable_np = pad_or_truncate(features[t0, stable_layer_idx])

            all_z0.append(z_0_np)
            all_stable.append(stable_np)
            all_state.append(states[t0])
            all_action.append(actions[t0])
            all_horizon.append(horizon)
            all_info.append((t0, t1, horizon))

    if not all_z0:
        return None, None, features, states, actions

    # Process in batches
    all_z0 = np.stack(all_z0)
    all_stable = np.stack(all_stable)
    all_state = np.stack(all_state)
    all_action = np.stack(all_action)
    all_horizon = np.array(all_horizon)

    n_samples = len(all_z0)
    distill_targets = []

    for i in range(0, n_samples, batch_size):
        end_i = min(i + batch_size, n_samples)

        z0_batch = torch.from_numpy(all_z0[i:end_i]).float().to(device)
        stable_batch = torch.from_numpy(all_stable[i:end_i]).float().to(device)
        state_batch = torch.from_numpy(all_state[i:end_i]).float().to(device)
        action_batch = torch.from_numpy(all_action[i:end_i]).float().to(device)
        horizon_batch = torch.from_numpy(all_horizon[i:end_i]).long().to(device)

        z1_distill = ode_solve_batch(
            model, z0_batch, stable_batch, state_batch, action_batch, horizon_batch,
            num_steps=num_steps
        )

        distill_targets.append(z1_distill.cpu().numpy())

    distill_targets = np.concatenate(distill_targets, axis=0)
    distill_info = np.array(all_info, dtype=np.int32)

    return distill_targets, distill_info, features, states, actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_path', type=str,
                        default='outputs/stable_dynamic_bridge/best_model.pt')
    parser.add_argument('--data_path', type=str,
                        default='outputs/latent_bridge_data/multilayer_train_data.h5')
    parser.add_argument('--output_path', type=str,
                        default='outputs/latent_bridge_data/distill_data.h5')
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_gpus', type=int, default=4)
    parser.add_argument('--max_episodes', type=int, default=None)
    args = parser.parse_args()

    # Get available GPUs
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    print(f"Using {num_gpus} GPUs")

    # Load teacher on each GPU
    teachers = []
    for i in range(num_gpus):
        device = f'cuda:{i}'
        teacher = load_teacher(args.teacher_path, device)
        teachers.append((teacher, device))

    # Open source data
    print(f"Loading data from {args.data_path}...")
    src_h5 = h5py.File(args.data_path, 'r')

    # Get episode keys
    ep_keys = sorted([k for k in src_h5.keys() if k.startswith('episode_')])
    if args.max_episodes:
        ep_keys = ep_keys[:args.max_episodes]
    print(f"Processing {len(ep_keys)} episodes")

    # Layer indices
    stable_layer_idx = 1
    target_layer_idx = 3

    # Create output file
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    out_h5 = h5py.File(args.output_path, 'w')

    total_pairs = 0

    # Process episodes with round-robin GPU assignment
    pbar = tqdm(ep_keys, desc="Episodes")
    for idx, ep_key in enumerate(pbar):
        # Round-robin GPU assignment
        gpu_idx = idx % num_gpus
        model, device = teachers[gpu_idx]

        distill_targets, distill_info, features, states, actions = process_episode(
            ep_key, src_h5, model, device,
            stable_layer_idx, target_layer_idx,
            args.num_steps, args.batch_size
        )

        # Create output episode group
        out_ep = out_h5.create_group(ep_key)
        out_ep.create_dataset('multilayer_features', data=features)
        out_ep.create_dataset('states', data=states)
        out_ep.create_dataset('actions', data=actions)

        if distill_targets is not None:
            out_ep.create_dataset('distill_targets', data=distill_targets)
            out_ep.create_dataset('distill_info', data=distill_info)
            total_pairs += len(distill_targets)

        pbar.set_postfix({'pairs': total_pairs, 'gpu': gpu_idx})

    src_h5.close()
    out_h5.close()

    print(f"\nDistillation data saved to {args.output_path}")
    print(f"Total pairs: {total_pairs}")


if __name__ == '__main__':
    main()
