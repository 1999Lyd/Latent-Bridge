#!/usr/bin/env python3
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Generate DAgger data for π0 KV bridge by rolling out bridge on existing data.

For each episode:
1. Start from fresh KV at t=0
2. Autoregressively predict KV using bridge: pred_kv[t+1] = pred_kv[t] + bridge(delta)
3. At each step, pair (bridge_input=pred_kv[t], target=oracle_kv[t+1])

This creates training data on the bridge's OWN distribution, not the oracle's.

Usage:
    CUDA_VISIBLE_DEVICES=4 python scripts/generate_pi0_dagger_kv.py \
        --data_path ./outputs/pi0_bridge_data/kv_spatial.h5 \
        --bridge_path ./outputs/pi0_bridge_kv/best_model.pt \
        --output_path ./outputs/pi0_bridge_data/dagger_kv_spatial.h5
"""
import os, sys
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'baseline', 'openpi', 'src'))

import argparse, torch, numpy as np, h5py, logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--bridge_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    torch.cuda.set_device(args.device)

    # Load bridge
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
    from pi0_bridge_kv import Pi0BridgeKV
    ckpt = torch.load(args.bridge_path, map_location=args.device, weights_only=False)
    bcfg = ckpt['config']
    bridge = Pi0BridgeKV(
        kv_dim=256, num_layers=18, seq_len=768, emb_dim=2048,
        hidden_dim=bcfg['hidden_dim'], num_heads=12, num_blocks=bcfg['num_blocks'],
        state_dim=8, action_dim=7,
    ).to(args.device).eval()
    bridge.load_state_dict(ckpt['model_state_dict'])
    logger.info("Bridge loaded: val_cos=%.4f", ckpt['val_cos'])

    # Load data
    logger.info("Loading data...")
    fin = h5py.File(args.data_path, 'r')
    episodes = sorted([k for k in fin.keys() if k.startswith('episode')])

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    fout = h5py.File(args.output_path, 'w')

    n_samples = 0
    for ep_key in tqdm(episodes):
        ep = fin[ep_key]
        if not ep.attrs.get('success', False):
            continue

        kv = ep['kv'][:].astype(np.float32)        # [T, 18, 768, 512]
        emb = ep['embedding'][:].astype(np.float32) # [T, 768, 2048]
        states = ep['state'][:]
        actions = ep['action'][:]
        T = kv.shape[0]
        replan = max(1, states.shape[0] // T)

        # Roll out bridge autoregressively
        # pred_kv starts as oracle at t=0, then bridge predicts
        pred_kv = torch.from_numpy(kv[0]).to(args.device)  # [18, 768, 512]

        dagger_prev_kv = []
        dagger_prev_emb = []
        dagger_curr_emb = []
        dagger_target_kv = []
        dagger_state = []
        dagger_action = []

        for t in range(T - 1):
            step_idx = min(t * replan, states.shape[0] - 1)

            prev_emb_t = torch.from_numpy(emb[t]).to(args.device)
            curr_emb_t = torch.from_numpy(emb[t+1]).to(args.device)
            oracle_next = torch.from_numpy(kv[t+1]).to(args.device)
            st = torch.from_numpy(states[step_idx].astype(np.float32)).to(args.device)
            ac = torch.from_numpy(actions[step_idx].astype(np.float32)).to(args.device)

            # Save DAgger pair: (bridge_predicted_kv[t], oracle_kv[t+1])
            dagger_prev_kv.append(pred_kv.cpu().numpy().astype(np.float16))
            dagger_prev_emb.append(emb[t].astype(np.float16))
            dagger_curr_emb.append(emb[t+1].astype(np.float16))
            dagger_target_kv.append(kv[t+1].astype(np.float16))
            dagger_state.append(states[step_idx].astype(np.float32))
            dagger_action.append(actions[step_idx].astype(np.float32))

            # Bridge prediction (autoregressive)
            with torch.no_grad():
                emb_delta = (curr_emb_t - prev_emb_t).unsqueeze(0).float()
                curr_emb_f = curr_emb_t.unsqueeze(0).float()
                S = pred_kv.shape[1]
                prev_flat = pred_kv.permute(1, 0, 2).reshape(1, S, -1).float()
                kv_deltas = bridge(emb_delta, curr_emb_f, prev_flat,
                                   st.unsqueeze(0), ac.unsqueeze(0))

            # Update pred_kv with bridge prediction
            for l in range(18):
                pred_kv[l] = pred_kv[l] + kv_deltas[l][0].to(pred_kv.dtype)

            n_samples += 1

        # Save episode
        if dagger_prev_kv:
            g = fout.create_group(ep_key)
            g.create_dataset('prev_kv', data=np.stack(dagger_prev_kv), compression='gzip', compression_opts=1)
            g.create_dataset('prev_emb', data=np.stack(dagger_prev_emb), compression='gzip', compression_opts=1)
            g.create_dataset('curr_emb', data=np.stack(dagger_curr_emb), compression='gzip', compression_opts=1)
            g.create_dataset('target_kv', data=np.stack(dagger_target_kv), compression='gzip', compression_opts=1)
            g.create_dataset('state', data=np.stack(dagger_state))
            g.create_dataset('action', data=np.stack(dagger_action))
            for k in ep.attrs:
                g.attrs[k] = ep.attrs[k]
            fout.flush()

    fout.attrs['n_samples'] = n_samples
    fout.close()
    fin.close()
    logger.info("Done: %d DAgger samples saved to %s", n_samples, args.output_path)

if __name__ == '__main__':
    main()
