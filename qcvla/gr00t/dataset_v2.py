"""
Dataset for single-step bridge training (h=1 only).

Simpler than v1 - no multi-horizon sampling.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, List
from tqdm import tqdm


class SingleStepDataset(Dataset):
    """
    Dataset for single-step (h=1) prediction.

    Each sample is a consecutive frame pair (z_t, z_{t+1}).
    """

    def __init__(
        self,
        h5_path: str,
        seq_len: int = 204,
        stable_layer_idx: int = 1,  # Layer 10
        target_layer_idx: int = 3,  # Layer 16
        max_samples: Optional[int] = None,
        episode_keys: Optional[List[str]] = None,
    ):
        self.h5_path = h5_path
        self.seq_len = seq_len
        self.stable_layer_idx = stable_layer_idx
        self.target_layer_idx = target_layer_idx
        self.episode_keys = episode_keys

        # Build index
        self.index = self._build_index(max_samples)
        self._h5_handle = None

    def _build_index(self, max_samples: Optional[int]) -> List:
        """Build index of (episode_key, t) tuples for h=1 pairs."""
        index = []

        with h5py.File(self.h5_path, 'r') as f:
            if self.episode_keys is not None:
                episode_keys = self.episode_keys
            else:
                episode_keys = sorted([k for k in f.keys() if k.startswith('episode_')])

            for ep_key in tqdm(episode_keys, desc="Indexing (h=1)"):
                ep = f[ep_key]

                if 'multilayer_features' in ep:
                    T = ep['multilayer_features'].shape[0]
                else:
                    continue

                # Only h=1: consecutive pairs
                for t in range(T - 1):
                    index.append((ep_key, t))

                    if max_samples and len(index) >= max_samples:
                        return index

        return index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._h5_handle is None:
            self._h5_handle = h5py.File(self.h5_path, 'r')

        ep_key, t = self.index[idx]
        ep = self._h5_handle[ep_key]

        # Load features
        features = ep['multilayer_features']
        target_t0 = np.array(features[t, self.target_layer_idx], dtype=np.float32)
        target_t1 = np.array(features[t + 1, self.target_layer_idx], dtype=np.float32)
        stable_t0 = np.array(features[t, self.stable_layer_idx], dtype=np.float32)

        # Pad/truncate
        target_t0 = self._pad_or_truncate(target_t0)
        target_t1 = self._pad_or_truncate(target_t1)
        stable_t0 = self._pad_or_truncate(stable_t0)

        # Load state and action
        states = ep['states'] if 'states' in ep else None
        actions = ep['actions'] if 'actions' in ep else None

        if states is not None:
            state = np.array(states[t], dtype=np.float32)
        else:
            state = np.zeros(8, dtype=np.float32)

        if actions is not None:
            action = np.array(actions[t], dtype=np.float32)
        else:
            action = np.zeros(7, dtype=np.float32)

        return {
            'target_t0': torch.from_numpy(target_t0),
            'target_t1': torch.from_numpy(target_t1),
            'stable_t0': torch.from_numpy(stable_t0),
            'state': torch.from_numpy(state),
            'action': torch.from_numpy(action),
        }

    def _pad_or_truncate(self, x: np.ndarray) -> np.ndarray:
        if len(x) > self.seq_len:
            return x[:self.seq_len]
        elif len(x) < self.seq_len:
            pad = np.zeros((self.seq_len - len(x), x.shape[1]), dtype=x.dtype)
            return np.concatenate([x, pad], axis=0)
        return x

    def __del__(self):
        if self._h5_handle is not None:
            self._h5_handle.close()
