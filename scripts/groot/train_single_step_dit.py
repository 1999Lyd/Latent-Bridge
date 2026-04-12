#!/usr/bin/env python3
"""
Train Single-Step Delta Predictor using DiT architecture.
Reuses proven DiTCrossBlock that works with DataParallel.

Key simplifications from DiTStableDynamicFlowModel:
- No time embedding (direct delta, not flow)
- No horizon embedding (h=1 only)
- Predicts delta = z1 - z0 directly

Increased capacity for better performance.
"""

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.insert(0, PROJECT_ROOT)

# NCCL settings to prevent hangs
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Import proven DiT components
from qcvla.model.rectified_flow_bridge import DiTCrossBlock, DiTFinalLayer


class SingleStepDataset(Dataset):
    """Dataset for h=1 only. Supports both preloading and lazy loading."""

    def __init__(self, h5_path: str, seq_len: int = 204, max_samples: int = None, preload: bool = True,
                 target_layer_idx: int = -1, stable_layer_idx: int = 0, use_vision: bool = False,
                 image_only: bool = False):
        """
        Args:
            target_layer_idx: Index into multilayer array for target (dynamic) layer.
                              -1 means last layer.
            stable_layer_idx: Index into multilayer array for stable (anchor) layer.
            use_vision: If True, load vision encoder features (vision_features key in H5).
            image_only: If True, extract only image tokens (using image_mask). Makes bridge
                        seq_len-agnostic across datasets. seq_len will be set to num_image_tokens.
        """
        import h5py
        self.image_only = image_only
        self.seq_len = seq_len
        self.h5_path = h5_path
        self.preload = preload
        self.target_layer_idx = target_layer_idx
        self.stable_layer_idx = stable_layer_idx
        self.use_vision = use_vision

        if preload:
            print(f"Preloading all data from {h5_path}...")
            self._preload_data(h5_path, max_samples)
        else:
            print(f"Building index from {h5_path} (lazy loading)...")
            self._build_index(h5_path, max_samples)

    @staticmethod
    def _get_feature_key(ep):
        """Find the multilayer features key (handles naming variations)."""
        for key in ['multilayer_features', 'multi_layer_features']:
            if key in ep:
                return key
        return None

    @staticmethod
    def _get_key(ep, candidates):
        """Find first matching key from candidates."""
        for key in candidates:
            if key in ep:
                return key
        return None

    def _preload_data(self, h5_path, max_samples):
        """Preload all data into memory."""
        import h5py

        all_target_t0 = []
        all_target_t1 = []
        all_stable_t0 = []
        all_states = []
        all_actions = []
        all_vision_t1 = []
        all_image_masks = []

        with h5py.File(h5_path, 'r') as f:
            episode_keys = sorted([k for k in f.keys() if k.startswith('episode_')])
            print(f"Found {len(episode_keys)} episodes")

            # Check if vision features exist
            has_vision = False
            if self.use_vision:
                sample_ep = f[episode_keys[0]]
                vision_key = self._get_key(sample_ep, ['vision_features', 'vision_encoder_features'])
                has_vision = vision_key is not None
                if not has_vision:
                    print("WARNING: --use_vision set but no vision_features in H5. "
                          "Re-collect data with --save_vision. Falling back to blind mode.")
                    self.use_vision = False
                else:
                    print(f"Vision features found (key: {vision_key})")

            for ep_idx, ep_key in enumerate(episode_keys):
                if ep_idx % 50 == 0:
                    print(f"  Loading episode {ep_idx}/{len(episode_keys)}...")

                ep = f[ep_key]
                feat_key = self._get_feature_key(ep)
                if feat_key is None:
                    continue

                features = ep[feat_key]
                T = features.shape[0]
                num_layers = features.shape[1]

                # Resolve layer indices (-1 means last)
                tgt_idx = self.target_layer_idx if self.target_layer_idx >= 0 else num_layers + self.target_layer_idx
                stb_idx = self.stable_layer_idx if self.stable_layer_idx >= 0 else num_layers + self.stable_layer_idx

                # Load entire episode at once
                target_layer = np.array(features[:, tgt_idx], dtype=np.float32)
                stable_layer = np.array(features[:, stb_idx], dtype=np.float32)
                state_key = self._get_key(ep, ['states', 'state'])
                action_key = self._get_key(ep, ['actions', 'action'])
                states = np.array(ep[state_key], dtype=np.float32)
                actions = np.array(ep[action_key], dtype=np.float32)

                # Image mask (same for all steps within an episode)
                image_mask = None
                img_indices = None
                if 'image_mask' in ep:
                    raw_mask = np.array(ep['image_mask'][0], dtype=np.float32)
                    while raw_mask.ndim > 1:
                        raw_mask = raw_mask[0]  # handle [1, seq] shape

                    if self.image_only:
                        # Extract image token indices — bridge will only process these
                        img_indices = np.where(raw_mask > 0.5)[0]
                        n_img = len(img_indices)
                        if ep_idx == 0:
                            self.seq_len = n_img  # set seq_len to number of image tokens
                            print(f"  Image-only mode: {n_img} image tokens "
                                  f"(from {len(raw_mask)} total)")
                        image_mask = np.ones(n_img, dtype=np.float32)
                    else:
                        # Pad/truncate to seq_len
                        if raw_mask.shape[0] >= self.seq_len:
                            image_mask = raw_mask[:self.seq_len]
                        else:
                            image_mask = np.zeros(self.seq_len, dtype=np.float32)
                            image_mask[:raw_mask.shape[0]] = raw_mask

                # Vision features (from VLM's vision encoder, pre-LLM)
                vision_data = None
                if has_vision:
                    v_key = self._get_key(ep, ['vision_features', 'vision_encoder_features'])
                    vision_data = np.array(ep[v_key], dtype=np.float32)

                for t in range(T - 1):
                    if self.image_only and img_indices is not None:
                        # Extract only image tokens
                        all_target_t0.append(target_layer[t][img_indices])
                        all_target_t1.append(target_layer[t + 1][img_indices])
                        all_stable_t0.append(stable_layer[t][img_indices])
                    else:
                        all_target_t0.append(self._pad_or_truncate(target_layer[t], self.seq_len))
                        all_target_t1.append(self._pad_or_truncate(target_layer[t + 1], self.seq_len))
                        all_stable_t0.append(self._pad_or_truncate(stable_layer[t], self.seq_len))
                    all_states.append(states[t])
                    all_actions.append(actions[t])

                    if image_mask is not None:
                        all_image_masks.append(image_mask)

                    if vision_data is not None:
                        # vision at t+1: the observation when target z_{t+1} was computed
                        all_vision_t1.append(vision_data[t + 1])

                    if max_samples and len(all_target_t0) >= max_samples:
                        break
                if max_samples and len(all_target_t0) >= max_samples:
                    break

        # Convert to tensors
        print(f"Converting {len(all_target_t0)} samples to tensors...")
        self.data = {
            'target_t0': torch.from_numpy(np.stack(all_target_t0)),
            'target_t1': torch.from_numpy(np.stack(all_target_t1)),
            'stable_t0': torch.from_numpy(np.stack(all_stable_t0)),
            'states': torch.from_numpy(np.stack(all_states)),
            'actions': torch.from_numpy(np.stack(all_actions)),
        }
        if all_image_masks:
            self.data['image_mask'] = torch.from_numpy(np.stack(all_image_masks))
            n_img = int(self.data['image_mask'][0].sum())
            print(f"  Image mask loaded: {n_img} image tokens, {self.seq_len - n_img} text tokens")
        if all_vision_t1:
            self.data['vision_t1'] = torch.from_numpy(np.stack(all_vision_t1))
            print(f"  Vision features: {self.data['vision_t1'].shape}")
        print(f"Preloaded {len(self.data['target_t0'])} samples")

    def _build_index(self, h5_path, max_samples):
        """Build index for lazy loading."""
        import h5py
        self.index = []

        with h5py.File(h5_path, 'r') as f:
            episode_keys = sorted([k for k in f.keys() if k.startswith('episode_')])
            print(f"Found {len(episode_keys)} episodes")

            for ep_key in tqdm(episode_keys, desc="Indexing"):
                ep = f[ep_key]
                feat_key = self._get_feature_key(ep)
                if feat_key is None:
                    continue

                T = ep[feat_key].shape[0]

                for t in range(T - 1):
                    self.index.append((ep_key, t))
                    if max_samples and len(self.index) >= max_samples:
                        break

                if max_samples and len(self.index) >= max_samples:
                    break

        print(f"Index built: {len(self.index)} samples")

    def _pad_or_truncate(self, arr, target_len):
        if arr.shape[0] == target_len:
            return arr
        elif arr.shape[0] > target_len:
            return arr[:target_len]
        else:
            pad = np.zeros((target_len - arr.shape[0],) + arr.shape[1:], dtype=arr.dtype)
            return np.concatenate([arr, pad], axis=0)

    def _get_h5_handle(self):
        """Get or create h5 file handle (one per worker process)."""
        if not hasattr(self, '_h5_handle') or self._h5_handle is None:
            import h5py
            self._h5_handle = h5py.File(self.h5_path, 'r')
        return self._h5_handle

    def __len__(self):
        if self.preload:
            return len(self.data['target_t0'])
        return len(self.index)

    def __getitem__(self, idx):
        if self.preload:
            sample = {
                'target_t0': self.data['target_t0'][idx],
                'target_t1': self.data['target_t1'][idx],
                'stable_t0': self.data['stable_t0'][idx],
                'state': self.data['states'][idx],
                'action': self.data['actions'][idx],
            }
            if 'image_mask' in self.data:
                sample['image_mask'] = self.data['image_mask'][idx]
            if 'vision_t1' in self.data:
                sample['vision_t1'] = self.data['vision_t1'][idx]
            return sample

        # Lazy loading
        f = self._get_h5_handle()
        ep_key, t = self.index[idx]
        ep = f[ep_key]

        feat_key = self._get_feature_key(ep)
        features = ep[feat_key]
        num_layers = features.shape[1]
        tgt_idx = self.target_layer_idx if self.target_layer_idx >= 0 else num_layers + self.target_layer_idx
        stb_idx = self.stable_layer_idx if self.stable_layer_idx >= 0 else num_layers + self.stable_layer_idx

        target_t0 = self._pad_or_truncate(np.array(features[t, tgt_idx], dtype=np.float32), self.seq_len)
        target_t1 = self._pad_or_truncate(np.array(features[t + 1, tgt_idx], dtype=np.float32), self.seq_len)
        stable_t0 = self._pad_or_truncate(np.array(features[t, stb_idx], dtype=np.float32), self.seq_len)
        state_key = self._get_key(ep, ['states', 'state'])
        action_key = self._get_key(ep, ['actions', 'action'])
        state = np.array(ep[state_key][t], dtype=np.float32)
        action = np.array(ep[action_key][t], dtype=np.float32)

        sample = {
            'target_t0': torch.from_numpy(target_t0),
            'target_t1': torch.from_numpy(target_t1),
            'stable_t0': torch.from_numpy(stable_t0),
            'state': torch.from_numpy(state),
            'action': torch.from_numpy(action),
        }

        if self.use_vision:
            v_key = self._get_key(ep, ['vision_features', 'vision_encoder_features'])
            if v_key and t + 1 < ep[v_key].shape[0]:
                vision = np.array(ep[v_key][t + 1], dtype=np.float32)
                sample['vision_t1'] = torch.from_numpy(vision)

        return sample


class DAggerDataset(Dataset):
    """Dataset for DAgger-collected bridge data.

    Reads H5 files from collect_dagger_bridge_data.py with format:
      episode_XXXX/z_input  [steps, seq, dim]  - bridge's input (own prediction or VLM)
      episode_XXXX/z_gt     [steps, seq, dim]  - VLM ground truth on same observation
      episode_XXXX/stable   [steps, seq, dim]  - stable layer features
      episode_XXXX/state    [steps, 8]          - robot state
      episode_XXXX/action   [steps, 7]          - previous action
      episode_XXXX/image_mask [steps, seq]      - image token mask (optional)
      episode_XXXX/is_vlm_step [steps]          - whether step was VLM (True) or bridge (False)

    Training pairs: (z_input[t], z_gt[t+1]) — the bridge at step t uses z_input[t]
    and should predict delta such that z_input[t] + delta ≈ VLM(obs_{t+1}).
    This matches the original sync training where (z0_t, z1_{t+1}) are consecutive.

    Maps to same output format as SingleStepDataset:
      target_t0 = z_input[t], target_t1 = z_gt[t+1], stable_t0 = stable[t], state[t], action[t]
    """

    def __init__(self, h5_path: str, seq_len: int = 204, max_samples: int = None,
                 bridge_only: bool = False, hard_only: bool = False,
                 hard_cos_threshold: float = 0.96, image_only: bool = False):
        """
        Args:
            bridge_only: If True, only use steps where the input was a bridge
                        prediction (not fresh VLM). These are the steps where
                        the bridge had to work from its own noisy output.
            hard_only: If True, only keep samples where the cosine between
                      z_input[t] and z_gt[t+1] (on image tokens) is below
                      hard_cos_threshold. Filters out easy samples.
            hard_cos_threshold: Cosine threshold for hard sample filtering.
                               Samples with cos >= threshold are discarded.
            image_only: If True, extract only image tokens using image_mask.
        """
        import h5py
        self.seq_len = seq_len
        self.image_only = image_only
        self.bridge_only = bridge_only
        self.hard_only = hard_only
        self.hard_cos_threshold = hard_cos_threshold

        all_z_input = []
        all_z_gt_next = []
        all_stable = []
        all_states = []
        all_actions = []
        all_image_masks = []

        print(f"Loading DAgger data from {h5_path}...")
        with h5py.File(h5_path, 'r') as f:
            episode_keys = sorted([k for k in f.keys() if k.startswith('episode_')])
            print(f"  Found {len(episode_keys)} episodes")

            for ep_idx, ep_key in enumerate(tqdm(episode_keys, desc="Loading DAgger")):
                ep = f[ep_key]
                z_input = np.array(ep['z_input'], dtype=np.float32)  # [steps, seq, dim]
                z_gt = np.array(ep['z_gt'], dtype=np.float32)
                stable = np.array(ep['stable'], dtype=np.float32)
                state = np.array(ep['state'], dtype=np.float32)
                action = np.array(ep['action'], dtype=np.float32)
                is_vlm = np.array(ep['is_vlm_step']) if 'is_vlm_step' in ep else None

                has_mask = 'image_mask' in ep
                img_indices = None
                if has_mask:
                    image_mask = np.array(ep['image_mask'], dtype=np.float32)
                    if image_only:
                        # Get image token indices from first step's mask
                        mask_0 = image_mask[0].flatten() if image_mask.ndim > 1 else image_mask[0]
                        img_indices = np.where(mask_0 > 0.5)[0]
                        if ep_idx == 0:
                            self.seq_len = len(img_indices)
                            print(f"  DAgger image-only mode: {len(img_indices)} image tokens")

                T = len(z_input)
                # Pair z_input[t] with z_gt[t+1]: bridge predicts next step's VLM output
                for t in range(T - 1):
                    if bridge_only and is_vlm is not None and is_vlm[t]:
                        continue

                    # Hard sample filtering
                    if hard_only and has_mask:
                        mask_t = image_mask[t] if image_mask.ndim == 2 else image_mask[t].flatten()
                        img_mask = mask_t[:min(len(mask_t), seq_len)] > 0.5
                        inp = z_input[t][:len(img_mask)][img_mask]
                        gt_next = z_gt[t + 1][:len(img_mask)][img_mask]
                        if inp.size > 0 and gt_next.size > 0:
                            cos = np.sum(inp * gt_next) / (np.linalg.norm(inp) * np.linalg.norm(gt_next) + 1e-8)
                            if cos >= hard_cos_threshold:
                                continue

                    if image_only and img_indices is not None:
                        all_z_input.append(z_input[t][img_indices])
                        all_z_gt_next.append(z_gt[t + 1][img_indices])
                        all_stable.append(stable[t][img_indices])
                    else:
                        all_z_input.append(self._pad_or_truncate(z_input[t], seq_len))
                        all_z_gt_next.append(self._pad_or_truncate(z_gt[t + 1], seq_len))
                        all_stable.append(self._pad_or_truncate(stable[t], seq_len))
                    all_states.append(state[t])
                    all_actions.append(action[t])

                    if has_mask:
                        if image_only and img_indices is not None:
                            all_image_masks.append(np.ones(len(img_indices), dtype=np.float32))
                        else:
                            mask = image_mask[t] if image_mask.ndim == 2 else image_mask[t].flatten()
                            if mask.shape[0] >= seq_len:
                                all_image_masks.append(mask[:seq_len])
                            else:
                                padded = np.zeros(seq_len, dtype=np.float32)
                            padded[:mask.shape[0]] = mask
                            all_image_masks.append(padded)

                    if max_samples and len(all_z_input) >= max_samples:
                        break
                if max_samples and len(all_z_input) >= max_samples:
                    break

        filter_str = []
        if bridge_only: filter_str.append('bridge-only')
        if hard_only: filter_str.append(f'hard-only(cos<{hard_cos_threshold})')
        if not filter_str: filter_str.append('all steps')
        print(f"  Loaded {len(all_z_input)} DAgger samples ({', '.join(filter_str)})")

        self.data = {
            'target_t0': torch.from_numpy(np.stack(all_z_input)),
            'target_t1': torch.from_numpy(np.stack(all_z_gt_next)),
            'stable_t0': torch.from_numpy(np.stack(all_stable)),
            'states': torch.from_numpy(np.stack(all_states)),
            'actions': torch.from_numpy(np.stack(all_actions)),
        }
        if all_image_masks:
            self.data['image_mask'] = torch.from_numpy(np.stack(all_image_masks))

    def _pad_or_truncate(self, arr, target_len):
        if arr.shape[0] == target_len:
            return arr
        elif arr.shape[0] > target_len:
            return arr[:target_len]
        else:
            pad = np.zeros((target_len - arr.shape[0],) + arr.shape[1:], dtype=arr.dtype)
            return np.concatenate([arr, pad], axis=0)

    def __len__(self):
        return len(self.data['target_t0'])

    def __getitem__(self, idx):
        sample = {
            'target_t0': self.data['target_t0'][idx],
            'target_t1': self.data['target_t1'][idx],
            'stable_t0': self.data['stable_t0'][idx],
            'state': self.data['states'][idx],
            'action': self.data['actions'][idx],
        }
        if 'image_mask' in self.data:
            sample['image_mask'] = self.data['image_mask'][idx]
        return sample


class SingleStepDiT(nn.Module):
    """
    Single-step delta predictor using DiT architecture.
    Predicts delta = z1 - z0 directly (no time, no horizon).
    Uses proven DiTCrossBlock that works with DataParallel.
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        seq_len: int = 204,
        hidden_dim: int = 768,      # Increased from 512
        num_heads: int = 12,        # Increased from 8
        num_blocks: int = 12,       # Increased from 4
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        state_dim: int = 8,
        action_dim: int = 7,
        low_rank: int = 0,          # 0 = disabled, e.g. 100 for rank-100 bottleneck
        num_image_tokens: int = 162, # Only predict deltas for image tokens
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.low_rank = low_rank
        self.num_image_tokens = num_image_tokens

        # Input projection
        self.input_proj = nn.Linear(feature_dim, hidden_dim)

        # Stable context projection
        self.stable_proj = nn.Linear(feature_dim, hidden_dim)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Condition embedding (state + action only, no time/horizon)
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Fuse conditions
        self.cond_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # DiT blocks with cross-attention
        self.blocks = nn.ModuleList([
            DiTCrossBlock(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_blocks)
        ])

        # Output: low-rank bottleneck or direct projection
        if low_rank > 0:
            # Low-rank factored output: hidden → rank → feature_dim
            # Forces model to learn the dominant delta subspace
            self.final_layer = DiTFinalLayer(hidden_dim, low_rank)
            self.rank_up = nn.Linear(low_rank, feature_dim)
            nn.init.zeros_(self.rank_up.weight)
            nn.init.zeros_(self.rank_up.bias)
        else:
            # Direct projection (original)
            self.final_layer = DiTFinalLayer(hidden_dim, feature_dim)

        self._count_parameters()

    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"SingleStepDiT: {total/1e6:.2f}M params")
        print(f"  - hidden_dim={self.hidden_dim}, blocks={len(self.blocks)}")

    def forward(self, z0, stable_features, state, action):
        """
        z0: [B, seq_len, feature_dim] - current features
        stable_features: [B, seq_len, feature_dim] - stable layer features
        state: [B, state_dim]
        action: [B, action_dim]
        Returns: delta [B, seq_len, feature_dim] (full seq including text tokens as zeros)
        """
        # Project inputs
        x = self.input_proj(z0) + self.pos_embed
        stable = self.stable_proj(stable_features) + self.pos_embed

        # Condition embeddings
        state_emb = self.state_embed(state)
        action_emb = self.action_embed(action)
        c = self.cond_fuse(torch.cat([state_emb, action_emb], dim=-1))

        # DiT blocks
        for block in self.blocks:
            x = block(x, stable, c)

        # Final layer -> delta
        delta = self.final_layer(x, c)

        # Low-rank: project from rank-space back to feature_dim
        if self.low_rank > 0:
            delta = self.rank_up(delta)

        return delta


class VisionConditionedBridge(SingleStepDiT):
    """
    SingleStepDiT with optional visual input from VLM's vision encoder.

    At inference, the VLM's vision encoder (~430M params) runs every step
    to provide current-observation context, while the expensive LLM (~1.7B)
    is skipped. The bridge uses visual tokens as additional cross-attention
    context alongside stable features.

    Cost: vision_encoder (430M) + bridge (160M) ≈ 590M → 3.6x cheaper than full VLM.
    vs blind bridge (144M, 15x cheaper) — this trades speed for accuracy.

    Architecture: visual tokens are concatenated to stable features and fed
    as K,V to the existing DiTCrossBlock cross-attention. No architectural
    changes to the blocks themselves.
    """

    def __init__(self, vision_dim: int = 2048, **kwargs):
        super().__init__(**kwargs)
        self.vision_proj = nn.Linear(vision_dim, self.hidden_dim)
        # Re-count with vision projection
        total = sum(p.numel() for p in self.parameters())
        print(f"VisionConditionedBridge: {total/1e6:.2f}M params (vision_dim={vision_dim})")

    def forward(self, z0, stable_features, state, action, vision_features=None):
        """
        Same as SingleStepDiT.forward, with optional vision_features.

        vision_features: [B, N_vis, vision_dim] from VLM's vision encoder
                        (extract_feature output). Concatenated to stable
                        context for cross-attention.
        """
        x = self.input_proj(z0) + self.pos_embed
        stable = self.stable_proj(stable_features) + self.pos_embed

        # Build cross-attention context
        if vision_features is not None:
            vis = self.vision_proj(vision_features)
            context = torch.cat([stable, vis], dim=1)  # [B, seq+N_vis, hidden]
        else:
            context = stable

        state_emb = self.state_embed(state)
        action_emb = self.action_embed(action)
        c = self.cond_fuse(torch.cat([state_emb, action_emb], dim=-1))

        for block in self.blocks:
            x = block(x, context, c)

        delta = self.final_layer(x, c)

        # Low-rank: project from rank-space back to feature_dim
        if self.low_rank > 0:
            delta = self.rank_up(delta)

        return delta


# ============================================================================
# Distillation Losses
# ============================================================================

def distillation_loss(pred, target, alpha_mse=1.0, alpha_cos=0.5):
    """
    Combined distillation loss for feature matching.

    Three loss components with different roles:

    1. MSE (alpha_mse): Penalizes magnitude errors. Ensures predicted features
       have correct scale. Can be dominated by a few high-error tokens.

    2. Cosine (alpha_cos): Direction-sensitive. The action head is more sensitive
       to feature direction than magnitude (cosine similarity drives action
       similarity). Per-token cosine averaged over sequence.

    3. Action-level (not implemented here, see ActionProxy): Functional
       distillation — directly optimizes "do predicted features produce the
       same actions?" Automatically learns which feature dimensions matter
       most. Requires a frozen action head or lightweight proxy.

    Returns:
        loss: Combined scalar loss
        metrics: Dict with component losses for logging
    """
    # Feature MSE
    loss_mse = F.mse_loss(pred, target)

    # Per-token cosine loss (direction matching)
    cos_sim = F.cosine_similarity(pred, target, dim=-1)  # [B, seq]
    loss_cos = (1 - cos_sim).mean()

    loss = alpha_mse * loss_mse + alpha_cos * loss_cos

    return loss, {
        'mse': loss_mse.item(),
        'cos_loss': loss_cos.item(),
        'cos_sim_mean': cos_sim.mean().item(),
    }


class ActionProxy(nn.Module):
    """
    Lightweight MLP approximating action_head(features) → action.

    Trained offline on (backbone_features, oracle_actions) from sync data.
    Used as a cheap differentiable proxy for functional distillation loss
    during bridge training, avoiding the expensive 32-layer DiT action head.

    Training: python scripts/train_single_step_dit.py --train_action_proxy ...
    Usage:    python scripts/train_single_step_dit.py --action_proxy_path proxy.pt ...

    ~1.5M params, negligible compute vs the 144M bridge.
    """

    def __init__(self, feature_dim: int = 2048, hidden_dim: int = 512, action_dim: int = 7):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        total = sum(p.numel() for p in self.parameters())
        print(f"ActionProxy: {total/1e6:.2f}M params")

    def forward(self, features):
        """features: [B, seq, feature_dim] → [B, action_dim]"""
        pooled = self.pool(features.transpose(1, 2)).squeeze(-1)  # [B, feature_dim]
        return self.mlp(pooled)


def distillation_loss_with_action(pred, target, action_proxy, gt_action,
                                   alpha_mse=1.0, alpha_cos=0.5, alpha_action=0.1):
    """
    Distillation loss with functional (action-level) component.

    The action proxy provides gradient signal about WHICH feature dimensions
    matter for action prediction. Features that don't affect actions get
    lower implicit weight.

    Args:
        pred: [B, seq, dim] predicted features
        target: [B, seq, dim] ground truth features
        action_proxy: Frozen ActionProxy model
        gt_action: [B, action_dim] ground truth action
        alpha_action: Weight for action loss. Start low (0.1) to avoid
                     dominating feature matching in early training.
    """
    loss, metrics = distillation_loss(pred, target, alpha_mse, alpha_cos)

    # Action-level loss: gradient flows through pred → action_proxy → loss
    pred_action = action_proxy(pred)
    loss_action = F.mse_loss(pred_action, gt_action)
    loss = loss + alpha_action * loss_action
    metrics['action_loss'] = loss_action.item()

    return loss, metrics


def train_epoch(model, dataloader, optimizer, device, use_flow=True,
                use_vision=False, loss_type='mse', alpha_cos=0.5,
                action_proxy=None, alpha_action=0.1,
                num_image_tokens=162, normalize_targets=False,
                delta_stats=None):
    """Train one epoch.

    Args:
        num_image_tokens: Number of image tokens (first N tokens). Loss is
            computed only on these tokens. Text tokens (rest) have zero delta
            and dilute the loss if included.
        normalize_targets: If True, normalize target deltas by pre-computed
            per-dim std for stable gradients (deltas can have magnitude ~4700).
        delta_stats: Dict with 'mean' and 'std' tensors [feature_dim] for
            normalization. Required if normalize_targets=True.
    """
    model.train()
    total_loss = 0
    total_samples = 0
    all_cosines = []
    loss_components = {'mse': [], 'cos_loss': [], 'action_loss': []}

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        z0 = batch['target_t0'].to(device)
        z1 = batch['target_t1'].to(device)
        stable = batch['stable_t0'].to(device)
        state = batch['state'].to(device)
        action = batch['action'].to(device)

        # Image mask: [B, seq_len] float, 1.0 for image tokens, 0.0 for text
        img_mask = batch.get('image_mask', None)
        if img_mask is not None:
            img_mask = img_mask.to(device)  # [B, seq_len]

        vision = None
        if use_vision and 'vision_t1' in batch:
            vision = batch['vision_t1'].to(device)

        B = z0.shape[0]

        target_velocity = z1 - z0  # velocity = delta

        if use_flow:
            # Flow matching: sample random t, interpolate input
            t = torch.rand(B, device=device)
            x_t = (1 - t.view(-1, 1, 1)) * z0 + t.view(-1, 1, 1) * z1
            pred_velocity = model(x_t, stable, state, action, vision) if use_vision else \
                            model(x_t, stable, state, action)
        else:
            pred_velocity = model(z0, stable, state, action, vision) if use_vision else \
                            model(z0, stable, state, action)

        # Apply image mask to focus loss on image tokens only
        # Text tokens have ~zero delta (cosine 0.9999) and waste capacity
        if img_mask is not None:
            mask_3d = img_mask.unsqueeze(-1)  # [B, seq, 1]
            pred_masked = pred_velocity * mask_3d
            target_masked = target_velocity * mask_3d
            n_img_tokens = img_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        else:
            pred_masked = pred_velocity
            target_masked = target_velocity

        # Normalize targets for stable gradients
        if normalize_targets and delta_stats is not None:
            std = delta_stats['std'].to(device)  # [feature_dim]
            std = std.clamp(min=1e-6)
            target_for_loss = target_masked / std.unsqueeze(0).unsqueeze(0)
            pred_for_loss = pred_masked / std.unsqueeze(0).unsqueeze(0)
        else:
            pred_for_loss = pred_masked
            target_for_loss = target_masked

        # Loss (on image tokens only via mask)
        if loss_type == 'distill_v2':
            # v2 loss: masked MSE + per-token cosine on image tokens
            loss_mse = F.mse_loss(pred_for_loss, target_for_loss)
            # Cosine on reconstructed features (image tokens only)
            pred_z1 = z0 + pred_velocity
            if img_mask is not None:
                # Per-token cosine, masked to image tokens
                cos_per_token = F.cosine_similarity(pred_z1, z1, dim=-1)  # [B, seq]
                cos_masked = (cos_per_token * img_mask).sum(dim=1) / n_img_tokens.squeeze(1)
                loss_cos = (1 - cos_masked).mean()
                cos_sim_mean = cos_masked.mean().item()
            else:
                cos_per_token = F.cosine_similarity(pred_z1, z1, dim=-1)
                loss_cos = (1 - cos_per_token).mean()
                cos_sim_mean = cos_per_token.mean().item()
            loss = loss_mse + alpha_cos * loss_cos
            metrics = {
                'mse': loss_mse.item(),
                'cos_loss': loss_cos.item(),
                'cos_sim_mean': cos_sim_mean,
            }
            if action_proxy is not None:
                pred_action = action_proxy(pred_z1)
                loss_action = F.mse_loss(pred_action, action)
                loss = loss + alpha_action * loss_action
                metrics['action_loss'] = loss_action.item()
        elif loss_type == 'distill' and action_proxy is not None:
            loss, metrics = distillation_loss_with_action(
                pred_for_loss, target_for_loss, action_proxy, action,
                alpha_cos=alpha_cos, alpha_action=alpha_action,
            )
        elif loss_type == 'distill':
            loss, metrics = distillation_loss(
                pred_for_loss, target_for_loss, alpha_cos=alpha_cos,
            )
        else:
            loss = F.mse_loss(pred_for_loss, target_for_loss)
            metrics = {}

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Metrics: delta cosine on image tokens
        with torch.no_grad():
            cos_sim = F.cosine_similarity(
                pred_masked.flatten(1), target_masked.flatten(1), dim=1
            ).mean().item()

        total_loss += loss.item() * B
        total_samples += B
        all_cosines.append(cos_sim)
        for k, v in metrics.items():
            if k in loss_components:
                loss_components[k].append(v)

        postfix = {'loss': f'{loss.item():.4f}', 'cos': f'{cos_sim:.4f}'}
        if 'cos_sim_mean' in metrics:
            postfix['ft_cos'] = f'{metrics["cos_sim_mean"]:.4f}'
        pbar.set_postfix(postfix)

    result = {
        'loss': total_loss / total_samples,
        'delta_cosine': np.mean(all_cosines),
    }
    for k, v in loss_components.items():
        if v:
            result[k] = np.mean(v)
    return result


@torch.no_grad()
def validate(model, dataloader, device, use_vision=False, num_image_tokens=162):
    """Validate with feature cosine on image tokens only."""
    model.eval()
    all_feature_cos = []
    all_copy_cos = []
    all_img_token_cos = []
    all_delta_magnitude_ratio = []

    for batch in tqdm(dataloader, desc="Validating"):
        z0 = batch['target_t0'].to(device)
        z1 = batch['target_t1'].to(device)
        stable = batch['stable_t0'].to(device)
        state = batch['state'].to(device)
        action = batch['action'].to(device)

        # Image mask from data (preferred) or None
        img_mask = batch.get('image_mask', None)
        if img_mask is not None:
            img_mask = img_mask.to(device)  # [B, seq_len]

        vision = None
        if use_vision and 'vision_t1' in batch:
            vision = batch['vision_t1'].to(device)

        if use_vision and vision is not None:
            pred_delta = model(z0, stable, state, action, vision)
        else:
            pred_delta = model(z0, stable, state, action)
        pred_z1 = z0 + pred_delta

        if img_mask is not None:
            # Mask-based: apply mask to select image tokens only
            mask_3d = img_mask.unsqueeze(-1)  # [B, seq, 1]
            pred_z1_masked = pred_z1 * mask_3d
            z1_masked = z1 * mask_3d
            z0_masked = z0 * mask_3d
            pred_delta_masked = pred_delta * mask_3d
            true_delta_masked = (z1 - z0) * mask_3d

            feature_cos = F.cosine_similarity(pred_z1_masked.flatten(1), z1_masked.flatten(1), dim=1)
            copy_cos = F.cosine_similarity(z0_masked.flatten(1), z1_masked.flatten(1), dim=1)

            # Per-token cosine on image tokens
            cos_per_tok = F.cosine_similarity(pred_z1, z1, dim=-1)  # [B, seq]
            n_img = img_mask.sum(dim=1).clamp(min=1)
            img_token_cos = (cos_per_tok * img_mask).sum(dim=1) / n_img

            true_norm = true_delta_masked.flatten(1).norm(dim=1)
            pred_norm = pred_delta_masked.flatten(1).norm(dim=1)
        else:
            feature_cos = F.cosine_similarity(pred_z1.flatten(1), z1.flatten(1), dim=1)
            copy_cos = F.cosine_similarity(z0.flatten(1), z1.flatten(1), dim=1)
            img_token_cos = F.cosine_similarity(pred_z1, z1, dim=-1).mean(dim=1)
            true_norm = (z1 - z0).flatten(1).norm(dim=1)
            pred_norm = pred_delta.flatten(1).norm(dim=1)

        ratio = pred_norm / true_norm.clamp(min=1e-8)

        all_feature_cos.extend(feature_cos.cpu().numpy())
        all_copy_cos.extend(copy_cos.cpu().numpy())
        all_img_token_cos.extend(img_token_cos.cpu().numpy())
        all_delta_magnitude_ratio.extend(ratio.cpu().numpy())

    return {
        'feature_cosine': np.mean(all_feature_cos),
        'img_token_cosine': np.mean(all_img_token_cos),
        'copy_baseline': np.mean(all_copy_cos),
        'improvement': np.mean(all_feature_cos) - np.mean(all_copy_cos),
        'delta_mag_ratio': np.mean(all_delta_magnitude_ratio),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/single_step_bridge')
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--num_blocks', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_gpus', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--weights_only_resume', action='store_true',
                        help='When resuming, only load model weights (reset optimizer, epoch, best_val). '
                             'Use for DAgger training where data distribution changes.')
    parser.add_argument('--reset_best', action='store_true',
                        help='Reset best_val_cosine to 0 when resuming (save best even if below resumed score)')
    parser.add_argument('--lazy_load', action='store_true', help='Use lazy loading instead of preloading')
    parser.add_argument('--target_layer_idx', type=int, default=-1,
                        help='Index into multilayer array for target layer (-1 = last)')
    parser.add_argument('--stable_layer_idx', type=int, default=0,
                        help='Index into multilayer array for stable layer')
    # Vision-conditioned bridge options
    parser.add_argument('--use_vision', action='store_true',
                        help='Use VisionConditionedBridge with visual input. '
                             'Data must be collected with --save_vision.')
    parser.add_argument('--loss', choices=['mse', 'distill', 'distill_v2'], default='mse',
                        help='Loss function: mse (standard), distill (MSE + cosine), '
                             'distill_v2 (image-only MSE + feature cosine, recommended)')
    parser.add_argument('--alpha_cos', type=float, default=0.5,
                        help='Weight for cosine loss in distillation mode')
    parser.add_argument('--alpha_action', type=float, default=0.1,
                        help='Weight for action proxy loss in distillation mode')
    parser.add_argument('--action_proxy_path', type=str, default=None,
                        help='Path to trained ActionProxy checkpoint for functional distillation')
    # v2 bridge training options
    parser.add_argument('--low_rank', type=int, default=0,
                        help='Low-rank bottleneck dimension (0=disabled, e.g. 100 for rank-100). '
                             'Forces model to learn dominant delta subspace.')
    parser.add_argument('--seq_len', type=int, default=0,
                        help='Sequence length (0=auto-detect from data). '
                             'LIBERO-10=204, LIBERO-Spatial=211. Must match VLM output.')
    parser.add_argument('--num_image_tokens', type=int, default=162,
                        help='Number of image tokens (loss computed only on these). '
                             'Text tokens have zero delta and dilute the loss.')
    parser.add_argument('--normalize_targets', action='store_true',
                        help='Normalize delta targets by per-dim std for stable gradients. '
                             'Critical when delta magnitudes are ~4700.')
    parser.add_argument('--no_flow', action='store_true',
                        help='Disable flow matching (use direct delta prediction). '
                             'RECOMMENDED: flow matching allows a shortcut f(x_t)=x_t-z0 '
                             'that produces near-zero at inference (t=0).')
    # DAgger training options
    parser.add_argument('--dagger_data_path', type=str, nargs='+', default=None,
                        help='Path(s) to DAgger H5 data from collect_dagger_bridge_data.py. '
                             'Multiple paths supported. Mixed with sync data for training.')
    parser.add_argument('--dagger_bridge_only', action='store_true',
                        help='Only use bridge steps from DAgger data (skip VLM steps '
                             'where z_input==z_gt, i.e. zero delta).')
    parser.add_argument('--dagger_hard_only', action='store_true',
                        help='Only use hard DAgger samples where cosine(z_input[t], z_gt[t+1]) '
                             'is below --dagger_hard_threshold.')
    parser.add_argument('--dagger_hard_threshold', type=float, default=0.96,
                        help='Cosine threshold for hard sample filtering (default: 0.96).')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_type = "VisionConditionedBridge" if args.use_vision else "SingleStepDiT"
    print("=" * 60)
    print(f"Training {model_type}")
    print(f"Hidden dim: {args.hidden_dim}, Blocks: {args.num_blocks}, Heads: {args.num_heads}")
    print(f"Vision: {args.use_vision}, Loss: {args.loss}")
    if args.low_rank > 0:
        print(f"  Low-rank bottleneck: {args.low_rank}")
    print(f"  Image tokens: {args.num_image_tokens}, Normalize: {args.normalize_targets}")
    print(f"  Flow matching: {not args.no_flow}")
    if args.loss in ('distill', 'distill_v2'):
        print(f"  alpha_cos: {args.alpha_cos}, alpha_action: {args.alpha_action}")
        if args.action_proxy_path:
            print(f"  action_proxy: {args.action_proxy_path}")
    if args.dagger_data_path:
        print(f"  DAgger data: {args.dagger_data_path} ({len(args.dagger_data_path)} files, bridge_only={args.dagger_bridge_only}, hard_only={args.dagger_hard_only}, hard_threshold={args.dagger_hard_threshold})")
    print(f"GPUs: {args.num_gpus}")
    print(f"Preload: {not args.lazy_load}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {device}")
    print("=" * 60)

    # Dataset(s)
    use_image_only = args.seq_len == 0  # auto-detect = image-only mode
    datasets = []
    dataset = SingleStepDataset(
        args.data_path, max_samples=args.max_samples, preload=not args.lazy_load,
        target_layer_idx=args.target_layer_idx, stable_layer_idx=args.stable_layer_idx,
        use_vision=args.use_vision, image_only=use_image_only,
    )
    datasets.append(dataset)
    # Get actual seq_len from dataset (auto-detected if image_only)
    actual_seq_len = dataset.seq_len
    print(f"Sync dataset: {len(dataset)} samples, seq_len={actual_seq_len}")

    if args.dagger_data_path:
        for dpath in args.dagger_data_path:
            if os.path.exists(dpath):
                dagger_dataset = DAggerDataset(
                    dpath, seq_len=actual_seq_len,
                    max_samples=args.max_samples,
                    bridge_only=args.dagger_bridge_only,
                    hard_only=args.dagger_hard_only,
                    hard_cos_threshold=args.dagger_hard_threshold,
                    image_only=use_image_only,
                )
                datasets.append(dagger_dataset)
                print(f"DAgger dataset ({dpath}): {len(dagger_dataset)} samples")
            else:
                print(f"WARNING: DAgger data not found: {dpath}")

    if len(datasets) > 1:
        from torch.utils.data import ConcatDataset
        dataset = ConcatDataset(datasets)
        print(f"Combined dataset: {len(dataset)} samples")

    # Split
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Compute delta stats for normalization (on image tokens only)
    delta_stats = None
    if args.normalize_targets:
        print("Computing delta statistics for normalization...")
        N_img = args.num_image_tokens
        all_deltas = []
        tmp_loader = DataLoader(train_dataset, batch_size=256, shuffle=False,
                                num_workers=args.num_workers, pin_memory=False)
        for batch in tqdm(tmp_loader, desc="Computing delta stats"):
            delta = (batch['target_t1'] - batch['target_t0'])[:, :N_img, :]  # [B, N_img, 2048]
            all_deltas.append(delta.reshape(-1, delta.shape[-1]))
        all_deltas = torch.cat(all_deltas, dim=0)  # [N_total, 2048]
        delta_std = all_deltas.std(dim=0)  # [2048]
        delta_mean = all_deltas.mean(dim=0)  # [2048]
        delta_stats = {'mean': delta_mean, 'std': delta_std}
        print(f"  Delta std: min={delta_std.min():.4f}, max={delta_std.max():.4f}, "
              f"mean={delta_std.mean():.4f}")
        print(f"  Delta magnitude: {all_deltas.norm(dim=1).mean():.1f}")
        del all_deltas, tmp_loader

    # Model
    if args.use_vision:
        model = VisionConditionedBridge(
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            low_rank=args.low_rank,
            num_image_tokens=args.num_image_tokens,
        ).to(device)
    else:
        model = SingleStepDiT(
            seq_len=actual_seq_len,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            low_rank=args.low_rank,
            num_image_tokens=args.num_image_tokens if not use_image_only else actual_seq_len,
        ).to(device)

    # Optional action proxy for functional distillation
    action_proxy = None
    if args.action_proxy_path and os.path.exists(args.action_proxy_path):
        proxy_ckpt = torch.load(args.action_proxy_path, map_location=device, weights_only=False)
        action_proxy = ActionProxy(
            feature_dim=proxy_ckpt.get('feature_dim', 2048),
            hidden_dim=proxy_ckpt.get('hidden_dim', 512),
            action_dim=proxy_ckpt.get('action_dim', 7),
        ).to(device)
        action_proxy.load_state_dict(proxy_ckpt['model_state_dict'])
        action_proxy.eval()
        for p in action_proxy.parameters():
            p.requires_grad = False
        print(f"Loaded ActionProxy from {args.action_proxy_path}")

    # Multi-GPU
    if args.num_gpus > 1:
        print(f"Using DataParallel with {args.num_gpus} GPUs")
        model = nn.DataParallel(model, device_ids=list(range(args.num_gpus)))

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    best_val_cosine = 0

    # Resume
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model_state = ckpt['model_state_dict']
        # Handle DataParallel
        if args.num_gpus > 1 and not list(model_state.keys())[0].startswith('module.'):
            model_state = {f'module.{k}': v for k, v in model_state.items()}
        model.load_state_dict(model_state)
        if args.weights_only_resume:
            print(f"Weights-only resume: reset optimizer, epoch=0, best_val=0")
        else:
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                # Override LR to match args.lr — load_state_dict restores the
                # checkpoint's LR which may differ from the intended fine-tuning LR.
                # CosineAnnealingLR decays relative to current LR, so this is critical.
                for pg in optimizer.param_groups:
                    pg['lr'] = args.lr
                print(f"  Optimizer loaded, LR overridden to {args.lr}")
            start_epoch = ckpt.get('epoch', 0)
            if args.reset_best:
                best_val_cosine = 0
                print(f"Resumed from epoch {start_epoch}, best val cos RESET to 0")
            else:
                best_val_cosine = ckpt.get('val_cosine', 0)
                print(f"Resumed from epoch {start_epoch}, best val cos: {best_val_cosine:.4f}")

    # Create scheduler AFTER resume so it picks up the correct LR
    remaining_epochs = args.epochs - start_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=remaining_epochs, eta_min=args.lr * 0.01
    )
    print(f"Scheduler: T_max={remaining_epochs}, LR={optimizer.param_groups[0]['lr']:.2e} -> {args.lr * 0.01:.2e}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            use_flow=not args.no_flow,
            use_vision=args.use_vision, loss_type=args.loss,
            alpha_cos=args.alpha_cos, action_proxy=action_proxy,
            alpha_action=args.alpha_action,
            num_image_tokens=args.num_image_tokens,
            normalize_targets=args.normalize_targets,
            delta_stats=delta_stats,
        )
        val_metrics = validate(model, val_loader, device, use_vision=args.use_vision,
                               num_image_tokens=args.num_image_tokens)

        scheduler.step()

        print(f"Train Loss: {train_metrics['loss']:.4f}, Train Delta Cos: {train_metrics['delta_cosine']:.4f}")
        print(f"Val Feature Cos: {val_metrics['feature_cosine']:.4f}, "
              f"Copy Baseline: {val_metrics['copy_baseline']:.4f}, "
              f"Improvement: {val_metrics['improvement']:.4f}, "
              f"Delta Mag Ratio: {val_metrics['delta_mag_ratio']:.4f}")

        # Save
        model_to_save = model.module if hasattr(model, 'module') else model
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_cosine': val_metrics['feature_cosine'],
            'train_loss': train_metrics['loss'],
            'config': {
                'feature_dim': 2048,
                'seq_len': actual_seq_len,
                'image_only': use_image_only,
                'hidden_dim': args.hidden_dim,
                'num_blocks': args.num_blocks,
                'num_heads': args.num_heads,
                'state_dim': 8,
                'action_dim': 7,
                'use_vision': args.use_vision,
                'loss': args.loss,
                'low_rank': args.low_rank,
                'num_image_tokens': args.num_image_tokens,
                'normalize_targets': args.normalize_targets,
            },
        }

        if delta_stats is not None:
            checkpoint['delta_stats'] = {k: v.cpu() for k, v in delta_stats.items()}

        if val_metrics['feature_cosine'] > best_val_cosine:
            best_val_cosine = val_metrics['feature_cosine']
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model_dit.pt'))
            print(f"New best! Val Cos: {best_val_cosine:.4f}")

        torch.save(checkpoint, os.path.join(args.output_dir, 'latest_model_dit.pt'))

    print(f"\nDone! Best Val Cos: {best_val_cosine:.4f}")


if __name__ == "__main__":
    main()
