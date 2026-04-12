#!/usr/bin/env python3
"""
Training script for SingleStepBridge (h=1 only).

Key differences from v1:
- Only h=1 prediction (no horizon embedding)
- Simpler training loop
- Autoregressive validation
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

from .bridge_v2 import SingleStepBridge
from .dataset_v2 import SingleStepDataset


def setup_ddp():
    """Initialize DDP. Returns None for single-GPU mode."""
    if 'RANK' not in os.environ:
        # Single GPU mode
        torch.cuda.set_device(0)
        return 0, 1, 0

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def is_ddp():
    """Check if running in DDP mode."""
    return 'RANK' in os.environ


def cleanup_ddp():
    if is_ddp():
        dist.destroy_process_group()


def sample_t_focused(batch_size: int, device: str, t0_prob: float = 0.5) -> torch.Tensor:
    """Sample t with focus on t=0."""
    t = torch.rand(batch_size, device=device)
    mask = torch.rand(batch_size, device=device) < t0_prob
    t[mask] = 0.0
    return t


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
        target_t0 = batch['target_t0'].to(device)
        target_t1 = batch['target_t1'].to(device)
        stable_t0 = batch['stable_t0'].to(device)
        state = batch['state'].to(device)
        action = batch['action'].to(device)

        B = target_t0.shape[0]

        # Sample t with focus on t=0
        t = sample_t_focused(B, device, t0_prob=t0_prob)

        # Linear interpolation for flow matching
        z_t = (1 - t.view(-1, 1, 1)) * target_t0 + t.view(-1, 1, 1) * target_t1
        target_velocity = target_t1 - target_t0

        # Forward pass
        pred_velocity = model(z_t, t, stable_t0, state=state, action=action)

        # MSE loss
        loss = F.mse_loss(pred_velocity, target_velocity)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            cos_sim = F.cosine_similarity(
                pred_velocity.flatten(1), target_velocity.flatten(1), dim=1
            ).mean().item()

        total_loss += loss.item() * B
        total_samples += B
        all_cosines.append(cos_sim)

        if rank == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'cos': f'{cos_sim:.4f}'})

    # Aggregate across ranks (only in DDP mode)
    if is_ddp():
        total_loss_tensor = torch.tensor([total_loss], device=device)
        total_samples_tensor = torch.tensor([total_samples], device=device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
        avg_loss = total_loss_tensor.item() / total_samples_tensor.item()
    else:
        avg_loss = total_loss / total_samples

    return {
        'loss': avg_loss,
        'velocity_cosine': np.mean(all_cosines),
    }


@torch.no_grad()
def validate(
    model: DDP,
    dataloader: DataLoader,
    device: str,
    rank: int = 0,
) -> dict:
    """
    Validate with:
    1. One-step prediction (z_0 -> z_1)
    2. Autoregressive for h=2,3 (simulated)
    """
    model.eval()

    # We need to collect consecutive samples for AR validation
    # For simplicity, just do one-step validation here
    all_feature_cosines = []
    all_velocity_cosines = []
    all_copy_cosines = []

    for batch in tqdm(dataloader, desc="Validating", disable=(rank != 0)):
        target_t0 = batch['target_t0'].to(device)
        target_t1 = batch['target_t1'].to(device)
        stable_t0 = batch['stable_t0'].to(device)
        state = batch['state'].to(device)
        action = batch['action'].to(device)

        B = target_t0.shape[0]

        # One-step prediction at t=0
        t = torch.zeros(B, device=device)
        pred_velocity = model(target_t0, t, stable_t0, state=state, action=action)
        pred_t1 = target_t0 + pred_velocity

        # Target velocity
        target_velocity = target_t1 - target_t0

        # Velocity cosine
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

    metrics = {
        'val_cosine': np.mean(all_feature_cosines),
        'val_cosine_std': np.std(all_feature_cosines),
        'val_velocity_cosine': np.mean(all_velocity_cosines),
        'copy_baseline': np.mean(all_copy_cosines),
        'improvement': np.mean(all_feature_cosines) - np.mean(all_copy_cosines),
    }

    return metrics


@torch.no_grad()
def validate_autoregressive(
    model: DDP,
    h5_path: str,
    val_episode_keys: list,
    device: str,
    max_samples: int = 200,
    rank: int = 0,
) -> dict:
    """
    Validate autoregressive prediction for h=2,3.

    Load consecutive frames and predict autoregressively.
    """
    model.eval()

    stable_layer_idx = 1
    target_layer_idx = 3
    seq_len = 204

    def pad_or_truncate(x, seq_len=204):
        if x.shape[0] > seq_len:
            return x[:seq_len]
        elif x.shape[0] < seq_len:
            pad = np.zeros((seq_len - x.shape[0], x.shape[1]), dtype=x.dtype)
            return np.concatenate([x, pad], axis=0)
        return x

    results = {h: {'feature_cos': [], 'velocity_cos': [], 'copy_cos': []} for h in [1, 2, 3]}

    with h5py.File(h5_path, 'r') as f:
        sample_count = 0
        for ep_key in val_episode_keys:
            if sample_count >= max_samples:
                break

            ep = f[ep_key]
            features = ep['multilayer_features'][:]
            states = ep['states'][:]
            actions = ep['actions'][:]
            T = features.shape[0]

            for t0 in range(min(T - 3, 10)):  # Sample some positions per episode
                if sample_count >= max_samples:
                    break

                # Load ground truth for t0, t1, t2, t3
                z0 = pad_or_truncate(features[t0, target_layer_idx])
                z1 = pad_or_truncate(features[t0 + 1, target_layer_idx])
                z2 = pad_or_truncate(features[t0 + 2, target_layer_idx])
                z3 = pad_or_truncate(features[t0 + 3, target_layer_idx])
                stable = pad_or_truncate(features[t0, stable_layer_idx])

                # Convert to tensors
                z0_t = torch.from_numpy(z0).float().unsqueeze(0).to(device)
                z1_t = torch.from_numpy(z1).float().unsqueeze(0).to(device)
                z2_t = torch.from_numpy(z2).float().unsqueeze(0).to(device)
                z3_t = torch.from_numpy(z3).float().unsqueeze(0).to(device)
                stable_t = torch.from_numpy(stable).float().unsqueeze(0).to(device)
                state_t = torch.from_numpy(states[t0]).float().unsqueeze(0).to(device)
                action_t = torch.from_numpy(actions[t0]).float().unsqueeze(0).to(device)

                # h=1: z0 -> z1
                t = torch.zeros(1, device=device)
                v1 = model(z0_t, t, stable_t, state=state_t, action=action_t)
                z1_pred = z0_t + v1

                # h=2: z1_pred -> z2 (autoregressive)
                v2 = model(z1_pred, t, stable_t, state=state_t, action=action_t)
                z2_pred = z1_pred + v2

                # h=3: z2_pred -> z3 (autoregressive)
                v3 = model(z2_pred, t, stable_t, state=state_t, action=action_t)
                z3_pred = z2_pred + v3

                # Compute metrics for each horizon
                for h, (pred, gt) in enumerate([(z1_pred, z1_t), (z2_pred, z2_t), (z3_pred, z3_t)], 1):
                    feature_cos = F.cosine_similarity(pred.flatten(1), gt.flatten(1), dim=1).item()
                    # Velocity is cumulative delta from z0
                    pred_delta = pred - z0_t
                    gt_delta = gt - z0_t
                    vel_cos = F.cosine_similarity(pred_delta.flatten(1), gt_delta.flatten(1), dim=1).item()
                    copy_cos = F.cosine_similarity(z0_t.flatten(1), gt.flatten(1), dim=1).item()

                    results[h]['feature_cos'].append(feature_cos)
                    results[h]['velocity_cos'].append(vel_cos)
                    results[h]['copy_cos'].append(copy_cos)

                sample_count += 1

    # Aggregate
    metrics = {}
    for h in [1, 2, 3]:
        if results[h]['feature_cos']:
            metrics[f'ar_h{h}_feature_cos'] = np.mean(results[h]['feature_cos'])
            metrics[f'ar_h{h}_velocity_cos'] = np.mean(results[h]['velocity_cos'])
            metrics[f'ar_h{h}_copy_baseline'] = np.mean(results[h]['copy_cos'])
            metrics[f'ar_h{h}_improvement'] = metrics[f'ar_h{h}_feature_cos'] - metrics[f'ar_h{h}_copy_baseline']

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='outputs/latent_bridge_data/multilayer_train_data.h5')
    parser.add_argument('--output_dir', type=str, default='outputs/gr00t_bridge_v2')

    # Model
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--state_dim', type=int, default=8)
    parser.add_argument('--action_dim', type=int, default=7)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--t0_prob', type=float, default=0.5)
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
        print(f"Training SingleStepBridge (h=1 only) with DDP ({world_size} GPUs)")
        os.makedirs(args.output_dir, exist_ok=True)

    # Episode-based split
    if rank == 0:
        print(f"Loading data from {args.data_path}...")

    with h5py.File(args.data_path, 'r') as f:
        all_episode_keys = sorted([k for k in f.keys() if k.startswith('episode_')])

    n_episodes = len(all_episode_keys)
    n_val_episodes = int(n_episodes * args.val_split)
    n_train_episodes = n_episodes - n_val_episodes

    train_episode_keys = all_episode_keys[:n_train_episodes]
    val_episode_keys = all_episode_keys[n_train_episodes:]

    if rank == 0:
        print(f"Episodes: {n_train_episodes} train, {n_val_episodes} val")

    # Datasets
    train_dataset = SingleStepDataset(
        args.data_path,
        seq_len=204,
        max_samples=args.max_samples,
        episode_keys=train_episode_keys,
    )
    val_dataset = SingleStepDataset(
        args.data_path,
        seq_len=204,
        episode_keys=val_episode_keys,
    )

    if rank == 0:
        print(f"Samples: {len(train_dataset)} train, {len(val_dataset)} val")

    # Samplers and loaders
    if is_ddp():
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Model
    if rank == 0:
        print("Creating SingleStepBridge model...")

    model = SingleStepBridge(
        feature_dim=2048,
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
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # Resume
    start_epoch = 0
    best_val_cosine = 0
    if args.resume and os.path.exists(args.resume):
        if rank == 0:
            print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.module.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_cosine = ckpt.get('val_cosine', 0)

    # Save config
    if rank == 0:
        config = vars(args)
        config['world_size'] = world_size
        config['version'] = 'v2_single_step'
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

        # Validate
        if rank == 0:
            val_metrics = validate(model, val_loader, device, rank=rank)
            print(f"Val Feature Cosine: {val_metrics['val_cosine']:.4f} ± {val_metrics['val_cosine_std']:.4f}")
            print(f"Val Velocity Cosine: {val_metrics['val_velocity_cosine']:.4f}")
            print(f"Copy Baseline: {val_metrics['copy_baseline']:.4f}")
            print(f"Improvement: {val_metrics['improvement']:.4f}")

            # Autoregressive validation
            ar_metrics = validate_autoregressive(
                model, args.data_path, val_episode_keys, device, max_samples=200, rank=rank
            )
            print("\nAutoregressive validation:")
            for h in [1, 2, 3]:
                if f'ar_h{h}_feature_cos' in ar_metrics:
                    print(f"  H{h}: feat={ar_metrics[f'ar_h{h}_feature_cos']:.4f}, "
                          f"vel={ar_metrics[f'ar_h{h}_velocity_cos']:.4f}, "
                          f"copy={ar_metrics[f'ar_h{h}_copy_baseline']:.4f}, "
                          f"impr={ar_metrics[f'ar_h{h}_improvement']:.4f}")

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
                'ar_metrics': ar_metrics,
                'config': vars(args),
            }, os.path.join(args.output_dir, 'latest_model.pt'))

        dist.barrier()

    if rank == 0:
        print(f"\nTraining complete! Best Val Cosine: {best_val_cosine:.4f}")

    cleanup_ddp()


if __name__ == '__main__':
    main()
