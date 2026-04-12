"""
Dataset for bridge training.

Loads multi-layer features from HDF5 file with lazy loading for efficiency.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, List
from tqdm import tqdm


class BridgeDataset(Dataset):
    """
    Dataset for flow bridge training.

    Loads pairs of (z0, z1) features from different timesteps within episodes.
    Uses lazy loading - only builds index at init, loads data on-demand.

    Data format (HDF5):
        episode_XXXX/
            multilayer_features: [T, num_layers, seq_len, feature_dim]
            states: [T, state_dim]
            actions: [T, action_dim]

    Layer mapping (GR00T):
        0: layer 1 (early)
        1: layer 10 (stable)
        2: layer 15 (pre-final)
        3: layer 16 (target)
    """

    def __init__(
        self,
        h5_path: str,
        seq_len: int = 204,
        stable_layer_idx: int = 1,  # Index in multilayer_features
        target_layer_idx: int = 3,  # Index in multilayer_features
        horizons: List[int] = [1, 2, 3],
        max_samples: Optional[int] = None,
        episode_keys: Optional[List[str]] = None,
    ):
        """
        Args:
            h5_path: path to HDF5 file
            seq_len: sequence length to pad/truncate to
            stable_layer_idx: which layer index is stable (default 1 = layer 10)
            target_layer_idx: which layer index is target (default 3 = layer 16)
            horizons: list of prediction horizons to sample
            max_samples: limit total samples (for debugging)
            episode_keys: list of episode keys to include (if None, use all)
        """
        self.h5_path = h5_path
        self.seq_len = seq_len
        self.stable_layer_idx = stable_layer_idx
        self.target_layer_idx = target_layer_idx
        self.horizons = horizons
        self.episode_keys = episode_keys

        # Build index (lazy loading)
        self.index = self._build_index(max_samples)

        # HDF5 handle (opened on first access)
        self._h5_handle = None

    def _build_index(self, max_samples: Optional[int]) -> List:
        """Build index of (episode_key, t0, t1, horizon) tuples."""
        index = []

        with h5py.File(self.h5_path, 'r') as f:
            # Use provided episode keys or get all from file
            if self.episode_keys is not None:
                episode_keys = self.episode_keys
            else:
                episode_keys = sorted([k for k in f.keys() if k.startswith('episode_')])

            for ep_key in tqdm(episode_keys, desc="Indexing"):
                ep = f[ep_key]

                # Get episode length
                if 'multilayer_features' in ep:
                    T = ep['multilayer_features'].shape[0]
                elif 'backbone_features' in ep:
                    T = ep['backbone_features'].shape[0]
                else:
                    continue

                # Add samples for each horizon
                for horizon in self.horizons:
                    for t0 in range(T - horizon):
                        t1 = t0 + horizon
                        index.append((ep_key, t0, t1, horizon))

                        if max_samples and len(index) >= max_samples:
                            return index

        return index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns dict with:
            target_t0: [seq_len, feature_dim] - target layer at t0
            target_t1: [seq_len, feature_dim] - target layer at t1
            stable_t0: [seq_len, feature_dim] - stable layer at t0
            state: [state_dim] - robot state at t0
            action: [action_dim] - robot action at t0
            horizon: int - prediction horizon
        """
        # Open HDF5 if not already open
        if self._h5_handle is None:
            self._h5_handle = h5py.File(self.h5_path, 'r')

        ep_key, t0, t1, horizon = self.index[idx]
        ep = self._h5_handle[ep_key]

        # Load features
        if 'multilayer_features' in ep:
            features = ep['multilayer_features']
            target_t0 = np.array(features[t0, self.target_layer_idx], dtype=np.float32)
            target_t1 = np.array(features[t1, self.target_layer_idx], dtype=np.float32)
            stable_t0 = np.array(features[t0, self.stable_layer_idx], dtype=np.float32)
        else:
            # Fallback: use backbone_features for both
            features = ep['backbone_features']
            target_t0 = np.array(features[t0], dtype=np.float32)
            target_t1 = np.array(features[t1], dtype=np.float32)
            stable_t0 = target_t0.copy()

        # Pad/truncate to seq_len
        target_t0 = self._pad_or_truncate(target_t0)
        target_t1 = self._pad_or_truncate(target_t1)
        stable_t0 = self._pad_or_truncate(stable_t0)

        # Load state and action
        states = ep['states'] if 'states' in ep else ep.get('state', None)
        actions = ep['actions'] if 'actions' in ep else ep.get('action', None)

        if states is not None:
            state = np.array(states[t0], dtype=np.float32)
        else:
            state = np.zeros(8, dtype=np.float32)

        if actions is not None:
            action = np.array(actions[t0], dtype=np.float32)
        else:
            action = np.zeros(7, dtype=np.float32)

        return {
            'target_t0': torch.from_numpy(target_t0),
            'target_t1': torch.from_numpy(target_t1),
            'stable_t0': torch.from_numpy(stable_t0),
            'state': torch.from_numpy(state),
            'action': torch.from_numpy(action),
            'horizon': torch.tensor(horizon, dtype=torch.long),
        }

    def _pad_or_truncate(self, x: np.ndarray) -> np.ndarray:
        """Pad or truncate to seq_len."""
        if len(x) > self.seq_len:
            return x[:self.seq_len]
        elif len(x) < self.seq_len:
            pad = np.zeros((self.seq_len - len(x), x.shape[1]), dtype=x.dtype)
            return np.concatenate([x, pad], axis=0)
        return x

    def __del__(self):
        """Close HDF5 handle."""
        if self._h5_handle is not None:
            self._h5_handle.close()
