#!/usr/bin/env python3
"""
Training script for GR00T FlowBridge.

Key features:
- t=0 focused sampling (50% at t=0)
- Simple MSE loss
- One-step validation
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from ..common.dataset import BridgeDataset
from ..common.flow import sample_t_focused, get_train_tuple
from .bridge import FlowBridge


def train_epoch(
    model: FlowBridge,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    t0_prob: float = 0.5,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_samples = 0
    all_cosines = []

    pbar = tqdm(dataloader, desc="Training")
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

        # MSE loss
        loss = F.mse_loss(pred_velocity, target_velocity)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item() * B
        total_samples += B

        with torch.no_grad():
            cos_sim = F.cosine_similarity(
                pred_velocity.flatten(1), target_velocity.flatten(1), dim=1
            ).mean().item()
            all_cosines.append(cos_sim)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'cos': f'{cos_sim:.4f}'})

    return {
        'loss': total_loss / total_samples,
        'velocity_cosine': np.mean(all_cosines),
    }


def validate(
    model: FlowBridge,
    dataloader: DataLoader,
    device: str,
) -> dict:
    """Validate with one-step generation."""
    model.eval()

    all_feature_cosines = []
    all_copy_cosines = []
    cosines_by_horizon = {1: [], 2: [], 3: []}
    copy_by_horizon = {1: [], 2: [], 3: []}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
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

            # Feature cosine
            feature_cos = F.cosine_similarity(
                pred_t1.flatten(1), target_t1.flatten(1), dim=1
            )

            # Copy baseline
            copy_cos = F.cosine_similarity(
                target_t0.flatten(1), target_t1.flatten(1), dim=1
            )

            all_feature_cosines.extend(feature_cos.cpu().numpy())
            all_copy_cosines.extend(copy_cos.cpu().numpy())

            # Per-horizon
            for i, h in enumerate(horizon.cpu().numpy()):
                h = int(h)
                if h in cosines_by_horizon:
                    cosines_by_horizon[h].append(feature_cos[i].item())
                    copy_by_horizon[h].append(copy_cos[i].item())

    metrics = {
        'val_cosine': np.mean(all_feature_cosines),
        'val_cosine_std': np.std(all_feature_cosines),
        'copy_baseline': np.mean(all_copy_cosines),
    }

    # Per-horizon
    for h in [1, 2, 3]:
        if cosines_by_horizon[h]:
            metrics[f'val_cosine_h{h}'] = np.mean(cosines_by_horizon[h])
            metrics[f'copy_baseline_h{h}'] = np.mean(copy_by_horizon[h])
            metrics[f'improvement_h{h}'] = metrics[f'val_cosine_h{h}'] - metrics[f'copy_baseline_h{h}']

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='outputs/latent_bridge_data/multilayer_train_data.h5')
    parser.add_argument('--output_dir', type=str, default='outputs/gr00t_bridge')
    parser.add_argument('--device', type=str, default='cuda:0')

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

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    # Dataset
    print(f"Loading data from {args.data_path}...")
    dataset = BridgeDataset(
        args.data_path,
        seq_len=204,
        stable_layer_idx=1,  # Layer 10
        target_layer_idx=3,  # Layer 16
        horizons=[1, 2, 3],
        max_samples=args.max_samples,
    )
    print(f"Dataset size: {len(dataset)}")

    # Split
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model
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
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_cosine = ckpt.get('val_cosine', 0)
        print(f"Resumed from epoch {start_epoch}, best_val_cosine={best_val_cosine:.4f}")

    # Save config
    config = vars(args)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            t0_prob=args.t0_prob,
        )
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"Train Velocity Cosine: {train_metrics['velocity_cosine']:.4f}")

        # Validate
        val_metrics = validate(model, val_loader, device)
        print(f"Val Cosine: {val_metrics['val_cosine']:.4f} ± {val_metrics['val_cosine_std']:.4f}")
        print(f"Copy Baseline: {val_metrics['copy_baseline']:.4f}")
        improvement = val_metrics['val_cosine'] - val_metrics['copy_baseline']
        print(f"Improvement: {improvement:.4f}")

        for h in [1, 2, 3]:
            if f'val_cosine_h{h}' in val_metrics:
                print(f"  H{h}: val={val_metrics[f'val_cosine_h{h}']:.4f}, "
                      f"copy={val_metrics[f'copy_baseline_h{h}']:.4f}, "
                      f"impr={val_metrics[f'improvement_h{h}']:.4f}")

        # Save best
        if val_metrics['val_cosine'] > best_val_cosine:
            best_val_cosine = val_metrics['val_cosine']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cosine': val_metrics['val_cosine'],
                'config': config,
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"New best model saved! Val Cosine: {best_val_cosine:.4f}")

        # Save latest
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_cosine': val_metrics['val_cosine'],
            'config': config,
        }, os.path.join(args.output_dir, 'latest_model.pt'))

    print(f"\nTraining complete! Best Val Cosine: {best_val_cosine:.4f}")


if __name__ == '__main__':
    main()
