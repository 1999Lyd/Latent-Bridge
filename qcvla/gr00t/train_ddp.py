#!/usr/bin/env python3
"""
DDP Training script for GR00T FlowBridge.

Key features:
- t=0 focused sampling (50% at t=0)
- Simple MSE loss
- Multi-GPU with DistributedDataParallel
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import h5py
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from ..common.dataset import BridgeDataset
from ..common.flow import sample_t_focused, get_train_tuple
from .bridge import FlowBridge


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
    t0_prob: float = 0.5,
    rank: int = 0,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_samples = 0
    all_cosines = []

    pbar = tqdm(dataloader, desc="Training", disable=(rank != 0))
    for batch in pbar:
        # Move to device
        target_t0 = batch['target_t0'].to(device)
        target_t1 = batch['target_t1'].to(device)
        stable_t0 = batch['stable_t0'].to(device)
        state = batch['state'].to(device)
        action = batch['action'].to(device)
        horizon = batch['horizon'].to(device)

        B = target_t0.shape[0]

        # Sample t with focus on t=0
        t = sample_t_focused(B, device, t0_prob=t0_prob)

        # Get training tuple
        t, z_t, target_velocity = get_train_tuple(target_t0, target_t1, t)

        # Forward pass
        pred_velocity = model(
            z_t, t, stable_t0,
            state=state, action=action, horizon=horizon
        )

        # MSE loss (direction + magnitude)
        loss = F.mse_loss(pred_velocity, target_velocity)

        # Track cosine and magnitude for monitoring
        with torch.no_grad():
            cos_sim = F.cosine_similarity(
                pred_velocity.flatten(1), target_velocity.flatten(1), dim=1
            )
            pred_mag = pred_velocity.flatten(1).norm(dim=1)
            target_mag = target_velocity.flatten(1).norm(dim=1)
            mag_ratio = pred_mag / (target_mag + 1e-8)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item() * B
        total_samples += B
        all_cosines.append(cos_sim.mean().item())
        avg_mag = mag_ratio.mean().item()

        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cos': f'{cos_sim.mean().item():.4f}',
                'mag': f'{avg_mag:.2f}'
            })

    # Aggregate across ranks
    total_loss_tensor = torch.tensor([total_loss], device=device)
    total_samples_tensor = torch.tensor([total_samples], device=device)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

    return {
        'loss': total_loss_tensor.item() / total_samples_tensor.item(),
        'velocity_cosine': np.mean(all_cosines),
    }


def validate(
    model: DDP,
    dataloader: DataLoader,
    device: str,
    rank: int = 0,
) -> dict:
    """Validate with one-step generation."""
    model.eval()

    all_feature_cosines = []
    all_velocity_cosines = []
    all_copy_cosines = []
    cosines_by_horizon = {1: [], 2: [], 3: []}
    velocity_by_horizon = {1: [], 2: [], 3: []}
    copy_by_horizon = {1: [], 2: [], 3: []}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", disable=(rank != 0)):
            target_t0 = batch['target_t0'].to(device)
            target_t1 = batch['target_t1'].to(device)
            stable_t0 = batch['stable_t0'].to(device)
            state = batch['state'].to(device)
            action = batch['action'].to(device)
            horizon = batch['horizon'].to(device)

            B = target_t0.shape[0]

            # One-step generation at t=0
            t = torch.zeros(B, device=device)
            pred_velocity = model(
                target_t0, t, stable_t0,
                state=state, action=action, horizon=horizon
            )
            pred_t1 = target_t0 + pred_velocity

            # Target velocity
            target_velocity = target_t1 - target_t0

            # Velocity cosine (matches training metric)
            velocity_cos = F.cosine_similarity(
                pred_velocity.flatten(1), target_velocity.flatten(1), dim=1
            )

            # Feature cosine
            feature_cos = F.cosine_similarity(
                pred_t1.flatten(1), target_t1.flatten(1), dim=1
            )

            # Copy baseline
            copy_cos = F.cosine_similarity(
                target_t0.flatten(1), target_t1.flatten(1), dim=1
            )

            all_velocity_cosines.extend(velocity_cos.cpu().numpy())
            all_feature_cosines.extend(feature_cos.cpu().numpy())
            all_copy_cosines.extend(copy_cos.cpu().numpy())

            # Per-horizon
            for i, h in enumerate(horizon.cpu().numpy()):
                h = int(h)
                if h in cosines_by_horizon:
                    cosines_by_horizon[h].append(feature_cos[i].item())
                    velocity_by_horizon[h].append(velocity_cos[i].item())
                    copy_by_horizon[h].append(copy_cos[i].item())

    # Gather metrics from all ranks
    metrics = {
        'val_cosine': np.mean(all_feature_cosines) if all_feature_cosines else 0,
        'val_cosine_std': np.std(all_feature_cosines) if all_feature_cosines else 0,
        'val_velocity_cosine': np.mean(all_velocity_cosines) if all_velocity_cosines else 0,
        'copy_baseline': np.mean(all_copy_cosines) if all_copy_cosines else 0,
    }

    # Per-horizon
    for h in [1, 2, 3]:
        if cosines_by_horizon[h]:
            metrics[f'val_cosine_h{h}'] = np.mean(cosines_by_horizon[h])
            metrics[f'val_velocity_h{h}'] = np.mean(velocity_by_horizon[h])
            metrics[f'copy_baseline_h{h}'] = np.mean(copy_by_horizon[h])
            metrics[f'improvement_h{h}'] = metrics[f'val_cosine_h{h}'] - metrics[f'copy_baseline_h{h}']

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='outputs/latent_bridge_data/multilayer_train_data.h5')
    parser.add_argument('--output_dir', type=str, default='outputs/gr00t_bridge')

    # Model
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--state_dim', type=int, default=8)
    parser.add_argument('--action_dim', type=int, default=7)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--t0_prob', type=float, default=0.5,
                        help='Probability of sampling t=0')
    parser.add_argument('--val_split', type=float, default=0.1)

    # Misc
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--max_samples', type=int, default=None)

    args = parser.parse_args()

    # Setup DDP
    rank, world_size, local_rank = setup_ddp()
    device = f'cuda:{local_rank}'

    if rank == 0:
        print(f"Training with DDP ({world_size} GPUs)")
        os.makedirs(args.output_dir, exist_ok=True)

    # Get episode keys and split by episode (not by sample!)
    if rank == 0:
        print(f"Loading data from {args.data_path}...")
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
    train_dataset = BridgeDataset(
        args.data_path,
        seq_len=204,
        stable_layer_idx=1,  # Layer 10
        target_layer_idx=3,  # Layer 16
        horizons=[1, 2, 3],
        max_samples=args.max_samples,
        episode_keys=train_episode_keys,
    )
    val_dataset = BridgeDataset(
        args.data_path,
        seq_len=204,
        stable_layer_idx=1,
        target_layer_idx=3,
        horizons=[1, 2, 3],
        episode_keys=val_episode_keys,
    )

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

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank])

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Resume
    start_epoch = 0
    best_val_cosine = 0
    if args.resume and os.path.exists(args.resume):
        if rank == 0:
            print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.module.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_cosine = ckpt.get('val_cosine', 0)
        if rank == 0:
            print(f"Resumed from epoch {start_epoch}, best_val_cosine={best_val_cosine:.4f}")

    # Save config
    if rank == 0:
        config = vars(args)
        config['world_size'] = world_size
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{args.epochs}")
            print(f"{'='*60}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            t0_prob=args.t0_prob, rank=rank,
        )

        if rank == 0:
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Train Velocity Cosine: {train_metrics['velocity_cosine']:.4f}")

        # Validate (only on rank 0 for simplicity)
        if rank == 0:
            val_metrics = validate(model, val_loader, device, rank=rank)
            print(f"Val Cosine: {val_metrics['val_cosine']:.4f} ± {val_metrics['val_cosine_std']:.4f}")
            print(f"Val Velocity Cosine: {val_metrics['val_velocity_cosine']:.4f}")
            print(f"Copy Baseline: {val_metrics['copy_baseline']:.4f}")
            improvement = val_metrics['val_cosine'] - val_metrics['copy_baseline']
            print(f"Improvement: {improvement:.4f}")

            for h in [1, 2, 3]:
                if f'val_cosine_h{h}' in val_metrics:
                    print(f"  H{h}: val={val_metrics[f'val_cosine_h{h}']:.4f}, "
                          f"vel={val_metrics[f'val_velocity_h{h}']:.4f}, "
                          f"copy={val_metrics[f'copy_baseline_h{h}']:.4f}, "
                          f"impr={val_metrics[f'improvement_h{h}']:.4f}")

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

        # Sync before next epoch
        dist.barrier()

    if rank == 0:
        print(f"\nTraining complete! Best Val Cosine: {best_val_cosine:.4f}")

    cleanup_ddp()


if __name__ == '__main__':
    main()
