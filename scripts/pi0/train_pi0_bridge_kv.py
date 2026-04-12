#!/usr/bin/env python3
"""
Train Pi0BridgeKV: predict pre-RoPE K+V deltas directly.

Usage:
    CUDA_VISIBLE_DEVICES=4 python scripts/train_pi0_bridge_kv.py \
        --data_path ./outputs/pi0_bridge_data/kv_spatial.h5 \
        --output_dir ./outputs/pi0_bridge_kv
"""
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'baseline', 'openpi', 'src'))

import argparse, json, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class Pi0KVDataset(Dataset):
    """In-memory dataset for KV delta prediction. Loads all data into RAM."""
    def __init__(self, h5_path, success_only=True):
        import h5py

        logger.info("Loading all data into RAM from %s...", h5_path)
        self.samples = []

        with h5py.File(h5_path, 'r') as f:
            episodes = sorted([k for k in f.keys() if k.startswith('episode')])
            n_ok = 0
            for ep_key in episodes:
                ep = f[ep_key]
                if success_only and not ep.attrs.get('success', False):
                    continue
                n_ok += 1

                kv = ep['kv'][:]           # [n_infer, 18, 768, 512] float16
                emb = ep['embedding'][:]   # [n_infer, 768, 2048] float16
                states = ep['state'][:]    # [n_steps, 8]
                actions = ep['action'][:]  # [n_steps, 7]

                n_infer = kv.shape[0]
                n_steps = states.shape[0]
                replan = max(1, n_steps // n_infer)

                for t in range(n_infer - 1):
                    step_idx = min(t * replan, n_steps - 1)
                    self.samples.append({
                        'prev_kv': kv[t],           # [18, 768, 512] float16
                        'next_kv': kv[t+1],
                        'prev_emb': emb[t],          # [768, 2048] float16
                        'curr_emb': emb[t+1],
                        'state': states[step_idx],    # [8]
                        'action': actions[step_idx],  # [7]
                    })

        logger.info("  %d episodes, %d samples loaded into RAM", n_ok, len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        prev_kv = s['prev_kv'].astype(np.float32)
        next_kv = s['next_kv'].astype(np.float32)
        return {
            'prev_kv': prev_kv,
            'next_kv': next_kv,
            'delta_kv': next_kv - prev_kv,
            'prev_emb': s['prev_emb'].astype(np.float32),
            'curr_emb': s['curr_emb'].astype(np.float32),
            'state': s['state'].astype(np.float32),
            'action': s['action'].astype(np.float32),
        }


class Pi0KVDaggerDataset(Dataset):
    """In-memory DAgger dataset: only loads consecutive pairs (prev_kv[t], oracle_kv[t+1])."""
    def __init__(self, h5_path):
        import h5py
        logger.info("Loading DAgger pairs from %s (memory-efficient)...", h5_path)
        self.samples = []
        with h5py.File(h5_path, 'r') as f:
            episodes = sorted([k for k in f.keys() if k.startswith('episode')])
            for ep_key in episodes:
                ep = f[ep_key]
                if 'kv' in ep and 'oracle_kv' in ep:
                    # Read full episode at once (faster than per-slice for gzip)
                    kv_all = ep['kv'][:]         # [T, 18, 768, 512]
                    oracle_all = ep['oracle_kv'][:]
                    emb_all = ep['embedding'][:]
                    states = ep['state'][:]
                    actions = ep['action'][:]
                    for t in range(len(kv_all) - 1):
                        self.samples.append({
                            'prev_kv': kv_all[t],
                            'next_kv': oracle_all[t+1],
                            'prev_emb': emb_all[t],
                            'curr_emb': emb_all[t+1],
                            'state': states[t],
                            'action': actions[t],
                        })
                    del kv_all, oracle_all, emb_all  # free immediately
                else:
                    prev_kv = ep['prev_kv'][:]
                    target_kv = ep['target_kv'][:]
                    prev_emb = ep['prev_emb'][:]
                    curr_emb = ep['curr_emb'][:]
                    states = ep['state'][:]
                    actions = ep['action'][:]
                    for t in range(prev_kv.shape[0]):
                        self.samples.append({
                            'prev_kv': prev_kv[t],
                            'next_kv': target_kv[t],
                            'prev_emb': prev_emb[t],
                            'curr_emb': curr_emb[t],
                            'state': states[t],
                            'action': actions[t],
                        })
                    del prev_kv, target_kv, prev_emb, curr_emb
        logger.info("  %d DAgger pairs loaded", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        prev_kv = s['prev_kv'].astype(np.float32)
        next_kv = s['next_kv'].astype(np.float32)
        return {
            'prev_kv': prev_kv,
            'next_kv': next_kv,
            'delta_kv': next_kv - prev_kv,
            'prev_emb': s['prev_emb'].astype(np.float32),
            'curr_emb': s['curr_emb'].astype(np.float32),
            'state': s['state'].astype(np.float32),
            'action': s['action'].astype(np.float32),
        }


def compute_loss(model, batch, device, no_delta=False):
    prev_kv = batch['prev_kv'].to(device)       # [B, 18, S, 512]
    next_kv = batch['next_kv'].to(device)
    delta_kv = batch['delta_kv'].to(device)
    prev_emb = batch['prev_emb'].to(device)
    curr_emb = batch['curr_emb'].to(device)
    state = batch['state'].to(device)
    action = batch['action'].to(device)

    emb_delta = curr_emb - prev_emb
    # Flatten prev_kv for input: [B, S, 18*512]
    B, L, S, D = prev_kv.shape
    prev_kv_flat = prev_kv.permute(0, 2, 1, 3).reshape(B, S, L * D)

    pred_deltas = model(emb_delta, curr_emb, prev_kv_flat, state, action)

    total_mse = 0
    total_cos = 0
    for l in range(18):
        pred_d = pred_deltas[l]           # [B, S, 512]
        tgt_next = next_kv[:, l]

        if no_delta:
            # Predict full KV directly: model output IS next_kv, no residual
            pred_next = pred_d  # no addition to prev_kv
            total_mse += F.mse_loss(pred_next, tgt_next)
            total_cos += F.cosine_similarity(pred_next, tgt_next, dim=-1).mean()
        else:
            # Delta prediction (default)
            tgt_d = delta_kv[:, l]
            pred_next = prev_kv[:, l] + pred_d
            total_mse += F.mse_loss(pred_d, tgt_d)
            total_cos += F.cosine_similarity(pred_next, tgt_next, dim=-1).mean()

    avg_mse = total_mse / 18
    avg_cos = total_cos / 18
    loss = avg_mse + (1.0 - avg_cos)
    return loss, {'loss': loss.item(), 'mse': avg_mse.item(), 'cos': avg_cos.item()}


def evaluate(model, val_loader, device, no_delta=False):
    model.eval()
    total = {'loss': 0, 'mse': 0, 'cos': 0}
    per_layer_cos = np.zeros(18)
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            loss, m = compute_loss(model, batch, device, no_delta=no_delta)
            bs = batch['state'].shape[0]
            for k in total: total[k] += m[k] * bs
            n += bs

            # Per-layer
            prev_kv = batch['prev_kv'].to(device)
            next_kv = batch['next_kv'].to(device)
            prev_emb = batch['prev_emb'].to(device)
            curr_emb = batch['curr_emb'].to(device)
            B, L, S, D = prev_kv.shape
            prev_kv_flat = prev_kv.permute(0, 2, 1, 3).reshape(B, S, L * D)
            pred = model(curr_emb - prev_emb, curr_emb, prev_kv_flat,
                        batch['state'].to(device), batch['action'].to(device))
            for l in range(18):
                pred_next = prev_kv[:, l] + pred[l]
                c = F.cosine_similarity(pred_next, next_kv[:, l], dim=-1).mean().item()
                per_layer_cos[l] += c * bs

    for k in total: total[k] /= n
    per_layer_cos /= n
    model.train()
    return total, per_layer_cos


def copy_baseline(val_loader, device):
    total_cos = 0
    per_layer = np.zeros(18)
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            prev = batch['prev_kv'].to(device)
            nxt = batch['next_kv'].to(device)
            bs = prev.shape[0]
            for l in range(18):
                c = F.cosine_similarity(prev[:, l], nxt[:, l], dim=-1).mean().item()
                per_layer[l] += c * bs
                total_cos += c * bs
            n += bs
    return total_cos / (n * 18), per_layer / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--dagger_path', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--num_blocks', type=int, default=10)
    parser.add_argument('--no_vision', action='store_true', help='Skip vision embedding input')
    parser.add_argument('--no_state', action='store_true', help='Skip state conditioning (ablation)')
    parser.add_argument('--no_action', action='store_true', help='Skip action conditioning (ablation)')
    parser.add_argument('--no_delta', action='store_true', help='Predict full KV instead of delta (ablation)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    sync_dataset = Pi0KVDataset(args.data_path, success_only=True)
    n_val = max(1, int(len(sync_dataset) * args.val_ratio))
    n_train = len(sync_dataset) - n_val
    train_sync, val_ds = random_split(sync_dataset, [n_train, n_val],
                                       generator=torch.Generator().manual_seed(args.seed))

    if args.dagger_path:
        dagger_dataset = Pi0KVDaggerDataset(args.dagger_path)
        from torch.utils.data import ConcatDataset
        train_ds = ConcatDataset([train_sync, dagger_dataset])
        logger.info("Train: %d sync + %d dagger = %d, Val: %d",
                    n_train, len(dagger_dataset), len(train_ds), n_val)
    else:
        train_ds = train_sync
        logger.info("Train: %d, Val: %d", n_train, n_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
    from pi0_bridge_kv import Pi0BridgeKV
    model = Pi0BridgeKV(
        kv_dim=256, num_layers=18, seq_len=768, emb_dim=2048,
        hidden_dim=args.hidden_dim, num_heads=12, num_blocks=args.num_blocks,
        state_dim=8, action_dim=7, no_vision=args.no_vision,
        no_state=args.no_state, no_action=args.no_action,
    ).to(args.device)

    if args.resume:
        logger.info("Resuming from %s", args.resume)
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    def lr_lambda(step):
        if step < warmup_steps: return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger.info("Computing copy baseline...")
    copy_cos, copy_per_layer = copy_baseline(val_loader, args.device)
    logger.info("Copy baseline avg cos: %.4f", copy_cos)
    logger.info("  L0=%.4f L5=%.4f L10=%.4f L17=%.4f",
                copy_per_layer[0], copy_per_layer[5], copy_per_layer[10], copy_per_layer[17])

    best_val_cos = 0.0
    for epoch in range(args.epochs):
        model.train()
        ep_loss, ep_cos, nb = 0, 0, 0
        t0 = time.time()
        for bi, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss, m = compute_loss(model, batch, args.device, no_delta=args.no_delta)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            ep_loss += m['loss']; ep_cos += m['cos']; nb += 1
            if (bi+1) % args.log_interval == 0:
                logger.info("  [%d/%d] loss=%.4f cos=%.4f lr=%.2e",
                          bi+1, len(train_loader), m['loss'], m['cos'], scheduler.get_last_lr()[0])

        ep_loss /= nb; ep_cos /= nb
        val_m, val_pl = evaluate(model, val_loader, args.device, no_delta=args.no_delta)
        pl_str = " ".join(f"L{l}={val_pl[l]:.4f}" for l in [0, 5, 10, 17])
        logger.info("Epoch %d/%d (%.0fs) — train cos=%.4f | val cos=%.4f (copy=%.4f) | %s",
                    epoch+1, args.epochs, time.time()-t0, ep_cos, val_m['cos'], copy_cos, pl_str)

        if val_m['cos'] > best_val_cos:
            best_val_cos = val_m['cos']
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch+1, 'val_cos': val_m['cos'],
                'val_per_layer': val_pl.tolist(),
                'config': vars(args),
            }, os.path.join(args.output_dir, 'best_model.pt'))
            logger.info("  * New best: %.4f", best_val_cos)

    logger.info("Done. Best val cos=%.4f (copy=%.4f)", best_val_cos, copy_cos)

if __name__ == '__main__':
    main()
