#!/usr/bin/env python3
"""
Train student model on distilled data with t=0 only.

Reflow-style distillation training:
- Student learns: z_0 + v(z_0, t=0) → z_1' (teacher output)
- No t sampling - always t=0
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .bridge import FlowBridge


class DistillDataset(Dataset):
    """Dataset for distillation training."""

    def __init__(
        self,
        h5_path: str,
        episode_keys: list = None,
        stable_layer_idx: int = 1,
        target_layer_idx: int = 3,
    ):
        """
        Args:
            h5_path: Path to distillation data HDF5 file
            episode_keys: List of episode keys to include (if None, use all)
            stable_layer_idx: Index of stable layer in multilayer_features
            target_layer_idx: Index of target layer in multilayer_features
        """
        self.h5_path = h5_path
        self.stable_layer_idx = stable_layer_idx
        self.target_layer_idx = target_layer_idx

        # Build index from specified episodes only
        self.index = []
        with h5py.File(h5_path, 'r') as f:
            # If no episode keys specified, use all
            if episode_keys is None:
                episode_keys = sorted([k for k in f.keys() if k.startswith('episode_')])

            for ep_key in episode_keys:
                if ep_key not in f:
                    continue
                ep = f[ep_key]
                if 'distill_info' not in ep:
                    continue

                distill_info = ep['distill_info'][:]  # [N, 3] = (t0, t1, horizon)
                for i in range(len(distill_info)):
                    self.index.append((ep_key, i))

        print(f"Distill dataset: {len(self.index)} samples from {len(episode_keys)} episodes")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep_key, sample_idx = self.index[idx]

        with h5py.File(self.h5_path, 'r') as f:
            ep = f[ep_key]

            # Get distill info
            t0, t1, horizon = ep['distill_info'][sample_idx]

            # Get features
            features = ep['multilayer_features']
            target_t0 = features[t0, self.target_layer_idx]
            stable_t0 = features[t0, self.stable_layer_idx]

            # Get distilled target (teacher ODE output)
            distill_target = ep['distill_targets'][sample_idx]

            # Get state/action
            state = ep['states'][t0]
            action = ep['actions'][t0]

        return {
            'target_t0': torch.from_numpy(target_t0).float(),
            'distill_target': torch.from_numpy(distill_target).float(),
            'stable_t0': torch.from_numpy(stable_t0).float(),
            'state': torch.from_numpy(state).float(),
            'action': torch.from_numpy(action).float(),
            'horizon': torch.tensor(horizon).float(),
        }


def setup_ddp():
    """Initialize DDP."""
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp():
    """Cleanup DDP."""
    dist.destroy_process_group()


def train_epoch(
    model: DDP,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    rank: int = 0,
) -> dict:
    """Train for one epoch with t=0 only."""
    model.train()
    total_loss = 0
    total_samples = 0
    all_cosines = []

    pbar = tqdm(dataloader, desc="Training", disable=(rank != 0))
    for batch in pbar:
        # Move to device
        target_t0 = batch['target_t0'].to(device)
        distill_target = batch['distill_target'].to(device)
        stable_t0 = batch['stable_t0'].to(device)
        state = batch['state'].to(device)
        action = batch['action'].to(device)
        horizon = batch['horizon'].to(device)

        B = target_t0.shape[0]

        # Always t=0 for distillation
        t = torch.zeros(B, device=device)

        # Forward pass
        pred_velocity = model(
            target_t0, t, stable_t0,
            state=state, action=action, horizon=horizon
        )

        # Predict z_1 from z_0 + velocity
        pred_t1 = target_t0 + pred_velocity

        # MSE loss against teacher output
        loss = F.mse_loss(pred_t1, distill_target)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item() * B
        total_samples += B

        with torch.no_grad():
            cos_sim = F.cosine_similarity(
                pred_t1.flatten(1), distill_target.flatten(1), dim=1
            ).mean().item()
            all_cosines.append(cos_sim)

        if rank == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'cos': f'{cos_sim:.4f}'})

    # Aggregate across ranks
    total_loss_tensor = torch.tensor([total_loss], device=device)
    total_samples_tensor = torch.tensor([total_samples], device=device)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

    return {
        'loss': total_loss_tensor.item() / total_samples_tensor.item(),
        'feature_cosine': np.mean(all_cosines),
    }


@torch.no_grad()
def validate(
    model: DDP,
    dataloader: DataLoader,
    device: str,
    rank: int = 0,
) -> dict:
    """Validate with one-step generation."""
    model.eval()

    all_cosines = []
    all_velocity_cosines = []

    for batch in tqdm(dataloader, desc="Validating", disable=(rank != 0)):
        target_t0 = batch['target_t0'].to(device)
        distill_target = batch['distill_target'].to(device)
        stable_t0 = batch['stable_t0'].to(device)
        state = batch['state'].to(device)
        action = batch['action'].to(device)
        horizon = batch['horizon'].to(device)

        B = target_t0.shape[0]

        # Always t=0
        t = torch.zeros(B, device=device)
        pred_velocity = model(
            target_t0, t, stable_t0,
            state=state, action=action, horizon=horizon
        )
        pred_t1 = target_t0 + pred_velocity

        # Feature cosine (pred vs teacher target)
        feature_cos = F.cosine_similarity(
            pred_t1.flatten(1), distill_target.flatten(1), dim=1
        )

        # Velocity cosine
        target_velocity = distill_target - target_t0
        velocity_cos = F.cosine_similarity(
            pred_velocity.flatten(1), target_velocity.flatten(1), dim=1
        )

        all_cosines.extend(feature_cos.cpu().numpy())
        all_velocity_cosines.extend(velocity_cos.cpu().numpy())

    return {
        'val_cosine': np.mean(all_cosines),
        'val_velocity_cosine': np.mean(all_velocity_cosines),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='outputs/latent_bridge_data/distill_data.h5')
    parser.add_argument('--output_dir', type=str, default='outputs/gr00t_distill')

    # Model
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--state_dim', type=int, default=8)
    parser.add_argument('--action_dim', type=int, default=7)

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    # Setup DDP
    rank, world_size, local_rank = setup_ddp()
    device = f'cuda:{local_rank}'

    if rank == 0:
        print(f"Training distilled model with DDP ({world_size} GPUs)")
        print(f"t=0 only (reflow distillation)")
        os.makedirs(args.output_dir, exist_ok=True)

    # Get episode keys and split by episode (not by sample!)
    if rank == 0:
        print(f"Loading distill data from {args.data_path}...")
        print("Splitting by EPISODE to avoid data leakage...")

    with h5py.File(args.data_path, 'r') as f:
        all_episode_keys = sorted([k for k in f.keys() if k.startswith('episode_')])

    n_episodes = len(all_episode_keys)
    n_val_episodes = int(n_episodes * args.val_split)
    n_train_episodes = n_episodes - n_val_episodes

    # Use last episodes for validation (deterministic split)
    train_episode_keys = all_episode_keys[:n_train_episodes]
    val_episode_keys = all_episode_keys[n_train_episodes:]

    if rank == 0:
        print(f"Total episodes: {n_episodes}")
        print(f"Train episodes: {n_train_episodes}, Val episodes: {n_val_episodes}")

    # Create separate datasets for train and val
    train_dataset = DistillDataset(args.data_path, episode_keys=train_episode_keys)
    val_dataset = DistillDataset(args.data_path, episode_keys=val_episode_keys)

    if rank == 0:
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Distributed samplers
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model
    if rank == 0:
        print("Creating FlowBridge model...")

    model = FlowBridge(
        latent_dim=2048,
        seq_len=204,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
    ).to(device)

    model = DDP(model, device_ids=[local_rank])

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Save config
    if rank == 0:
        config = vars(args)
        config['world_size'] = world_size
        config['distillation'] = True
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    # Training loop
    best_val_cosine = 0

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{args.epochs}")
            print(f"{'='*60}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, rank=rank)

        if rank == 0:
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Train Feature Cosine: {train_metrics['feature_cosine']:.4f}")

        # Validate
        if rank == 0:
            val_metrics = validate(model, val_loader, device, rank=rank)
            print(f"Val Cosine: {val_metrics['val_cosine']:.4f}")
            print(f"Val Velocity Cosine: {val_metrics['val_velocity_cosine']:.4f}")

            # Save best
            if val_metrics['val_cosine'] > best_val_cosine:
                best_val_cosine = val_metrics['val_cosine']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_cosine': val_metrics['val_cosine'],
                    'config': vars(args),
                }, os.path.join(args.output_dir, 'best_model.pt'))
                print(f"New best model saved! Val Cosine: {best_val_cosine:.4f}")

            # Save latest
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cosine': val_metrics['val_cosine'],
                'config': vars(args),
            }, os.path.join(args.output_dir, 'latest_model.pt'))

        dist.barrier()

    if rank == 0:
        print(f"\nDistillation complete! Best Val Cosine: {best_val_cosine:.4f}")

    cleanup_ddp()


if __name__ == '__main__':
    main()
