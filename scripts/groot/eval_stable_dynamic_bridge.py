#!/usr/bin/env python3
"""
Evaluate Stable-Dynamic Flow Bridge on LIBERO-10.

Compares three modes:
1. Sync: Fresh VLM features every step (baseline)
2. Async: Cache VLM features for N steps (naive caching)
3. Bridge: Use Stable-Dynamic Flow Bridge to predict features for N steps
"""

import argparse
import copy
import os
import sys
from pathlib import Path
from collections import deque, defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import time
from tqdm import tqdm
from typing import Dict, Optional
import json
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks" / "Isaac-GR00T"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import BatchFeature

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pad_or_truncate(x: np.ndarray, seq_len: int = 204) -> np.ndarray:
    """Pad or truncate to seq_len."""
    if x.shape[0] < seq_len:
        pad_len = seq_len - x.shape[0]
        return np.pad(x, ((0, pad_len), (0, 0)), mode='constant')
    return x[:seq_len]


@torch.no_grad()
def offline_eval_ode(
    model,
    data_path: str,
    device: str,
    num_steps_list: list = [1, 5, 10],
    max_samples: int = 500,
    batch_size: int = 32,
    stable_layer_idx: int = 1,
    target_layer_idx: int = 3,
):
    """
    Offline evaluation: measure cosine similarity between predicted and ground truth.

    This evaluates the teacher model's ODE quality on held-out data.
    """
    model.eval()

    # Load data
    logger.info(f"Loading data from {data_path}...")
    h5_file = h5py.File(data_path, 'r')

    ep_keys = sorted([k for k in h5_file.keys() if k.startswith('episode_')])
    logger.info(f"Found {len(ep_keys)} episodes")

    # Use last 10% as validation
    n_val = max(1, int(len(ep_keys) * 0.1))
    val_ep_keys = ep_keys[-n_val:]
    logger.info(f"Using {n_val} validation episodes")

    # Collect samples
    all_z0, all_z1, all_stable = [], [], []
    all_state, all_action, all_horizon = [], [], []

    for ep_key in val_ep_keys:
        ep = h5_file[ep_key]
        features = ep['multilayer_features'][:]
        states = ep['states'][:]
        actions = ep['actions'][:]
        T = features.shape[0]

        for horizon in [1, 2, 3]:
            for t0 in range(T - horizon):
                t1 = t0 + horizon

                z0 = pad_or_truncate(features[t0, target_layer_idx])
                z1 = pad_or_truncate(features[t1, target_layer_idx])
                stable = pad_or_truncate(features[t0, stable_layer_idx])

                all_z0.append(z0)
                all_z1.append(z1)
                all_stable.append(stable)
                all_state.append(states[t0])
                all_action.append(actions[t0])
                all_horizon.append(horizon)

                if len(all_z0) >= max_samples:
                    break
            if len(all_z0) >= max_samples:
                break
        if len(all_z0) >= max_samples:
            break

    h5_file.close()

    n_samples = len(all_z0)
    logger.info(f"Collected {n_samples} samples")

    all_z0 = np.stack(all_z0)
    all_z1 = np.stack(all_z1)
    all_stable = np.stack(all_stable)
    all_state = np.stack(all_state)
    all_action = np.stack(all_action)
    all_horizon = np.array(all_horizon)

    # Compute copy baseline
    copy_cosines = []
    for i in range(0, n_samples, batch_size):
        end_i = min(i + batch_size, n_samples)
        z0_batch = torch.from_numpy(all_z0[i:end_i]).float().to(device)
        z1_batch = torch.from_numpy(all_z1[i:end_i]).float().to(device)
        cos = F.cosine_similarity(z0_batch.flatten(1), z1_batch.flatten(1), dim=1)
        copy_cosines.extend(cos.cpu().numpy())

    logger.info(f"\nCopy baseline (z0 as prediction): {np.mean(copy_cosines):.4f}")

    results = {"copy_baseline": float(np.mean(copy_cosines))}

    # Evaluate with different number of ODE steps
    for num_steps in num_steps_list:
        logger.info(f"\n{'='*60}")
        logger.info(f"ODE Steps: {num_steps}")
        logger.info(f"{'='*60}")

        feature_cosines = []
        velocity_cosines = []
        cosines_by_horizon = {1: [], 2: [], 3: []}
        vel_by_horizon = {1: [], 2: [], 3: []}

        for i in tqdm(range(0, n_samples, batch_size), desc=f"Steps={num_steps}"):
            end_i = min(i + batch_size, n_samples)

            z0_batch = torch.from_numpy(all_z0[i:end_i]).float().to(device)
            z1_batch = torch.from_numpy(all_z1[i:end_i]).float().to(device)
            stable_batch = torch.from_numpy(all_stable[i:end_i]).float().to(device)
            state_batch = torch.from_numpy(all_state[i:end_i]).float().to(device)
            action_batch = torch.from_numpy(all_action[i:end_i]).float().to(device)
            horizon_batch = torch.from_numpy(all_horizon[i:end_i]).long().to(device)

            B = z0_batch.shape[0]

            # Run ODE
            if num_steps == 1:
                # One-step: z_1 = z_0 + v(z_0, t=0)
                t = torch.zeros(B, device=device)
                velocity = model(
                    z0_batch, t * 999,
                    state=state_batch,
                    action=action_batch,
                    horizon=horizon_batch,
                    stable_features=stable_batch,
                )
                pred_z1 = z0_batch + velocity
            else:
                # Multi-step Euler ODE
                dt = 1.0 / num_steps
                z_t = z0_batch.clone()
                for step_i in range(num_steps):
                    t = torch.full((B,), step_i * dt, device=device)
                    velocity = model(
                        z_t, t * 999,
                        state=state_batch,
                        action=action_batch,
                        horizon=horizon_batch,
                        stable_features=stable_batch,
                    )
                    z_t = z_t + dt * velocity
                pred_z1 = z_t

            # Feature cosine (pred_z1 vs ground truth z1)
            feature_cos = F.cosine_similarity(
                pred_z1.flatten(1), z1_batch.flatten(1), dim=1
            )

            # Velocity cosine (predicted delta vs ground truth delta)
            pred_velocity = pred_z1 - z0_batch
            gt_velocity = z1_batch - z0_batch
            vel_cos = F.cosine_similarity(
                pred_velocity.flatten(1), gt_velocity.flatten(1), dim=1
            )

            feature_cosines.extend(feature_cos.cpu().numpy())
            velocity_cosines.extend(vel_cos.cpu().numpy())

            # Per-horizon
            for j, h in enumerate(all_horizon[i:end_i]):
                h = int(h)
                cosines_by_horizon[h].append(feature_cos[j].item())
                vel_by_horizon[h].append(vel_cos[j].item())

        # Report results
        mean_feature_cos = np.mean(feature_cosines)
        mean_vel_cos = np.mean(velocity_cosines)
        improvement = mean_feature_cos - np.mean(copy_cosines)

        logger.info(f"\nResults for {num_steps} ODE steps:")
        logger.info(f"  Feature Cosine (z1' vs z1): {mean_feature_cos:.4f} ± {np.std(feature_cosines):.4f}")
        logger.info(f"  Velocity Cosine (v' vs v): {mean_vel_cos:.4f} ± {np.std(velocity_cosines):.4f}")
        logger.info(f"  Improvement over copy: {improvement:+.4f}")

        logger.info(f"\nPer-horizon:")
        for h in [1, 2, 3]:
            if cosines_by_horizon[h]:
                logger.info(f"  H{h}: feature_cos={np.mean(cosines_by_horizon[h]):.4f}, "
                           f"vel_cos={np.mean(vel_by_horizon[h]):.4f}")

        results[f"steps_{num_steps}"] = {
            "feature_cosine": float(mean_feature_cos),
            "feature_cosine_std": float(np.std(feature_cosines)),
            "velocity_cosine": float(mean_vel_cos),
            "velocity_cosine_std": float(np.std(velocity_cosines)),
            "improvement": float(improvement),
            "per_horizon": {
                h: {
                    "feature_cosine": float(np.mean(cosines_by_horizon[h])) if cosines_by_horizon[h] else 0,
                    "velocity_cosine": float(np.mean(vel_by_horizon[h])) if vel_by_horizon[h] else 0,
                }
                for h in [1, 2, 3]
            }
        }

    return results


class AsyncGr00tPolicy:
    """Asynchronous GR00T Policy that caches VLM (S2) outputs."""

    def __init__(self, policy, vlm_update_freq: int = 1, use_anchor: bool = True, anchor_noise_level: float = 0.05):
        self.policy = policy
        self.model = policy.model
        self.processor = policy.processor
        self.embodiment_tag = policy.embodiment_tag
        self.modality_configs = policy.modality_configs
        self.collate_fn = policy.collate_fn
        self.language_key = policy.language_key

        self.vlm_update_freq = vlm_update_freq
        self.use_anchor = use_anchor
        self.anchor_noise_level = anchor_noise_level

        self._cached_backbone_outputs = None
        self._cached_action_inputs = None
        self._step_count = 0
        self._anchor = None
        self.s2_times = []
        self.s1_times = []

    def reset(self, options=None):
        self._cached_backbone_outputs = None
        self._cached_action_inputs = None
        self._step_count = 0
        self._anchor = None
        self.s2_times = []
        self.s1_times = []
        return {}

    def _unbatch_observation(self, value):
        unbatched_obs = []
        batch_size = value["video"][list(value["video"].keys())[0]].shape[0]
        for i in range(batch_size):
            unbatched_value = {
                "video": {k: v[i] for k, v in value["video"].items()},
                "state": {k: v[i] for k, v in value["state"].items()},
                "language": {k: v[i] for k, v in value["language"].items()},
            }
            unbatched_obs.append(unbatched_value)
        return unbatched_obs

    def _to_vla_step_data(self, observation):
        from gr00t.data.types import VLAStepData
        return VLAStepData(
            images=observation["video"],
            states=observation["state"],
            actions={},
            text=observation["language"][self.language_key][0],
            embodiment=self.embodiment_tag,
        )

    def _rec_to_dtype(self, x, dtype):
        if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
            return x.to(dtype)
        elif isinstance(x, dict):
            return {k: self._rec_to_dtype(v, dtype) for k, v in x.items()}
        elif isinstance(x, (list, tuple)):
            return type(x)(self._rec_to_dtype(v, dtype) for v in x)
        else:
            return x

    def _prepare_inputs(self, observation):
        """Prepare inputs from observation."""
        from gr00t.data.types import MessageType

        new_obs = {}
        for modality in ["video", "state", "language"]:
            new_obs[modality] = {}
            for key in self.modality_configs[modality].modality_keys:
                if modality == "language":
                    if key == "task" and "annotation.human.coarse_action" in observation:
                        parsed_key = "annotation.human.coarse_action"
                    else:
                        parsed_key = key
                else:
                    parsed_key = f"{modality}.{key}"

                arr = observation[parsed_key]
                if modality == "language":
                    new_obs[modality][key] = [[str(item)] for item in arr]
                else:
                    new_obs[modality][key] = arr

        unbatched_observations = self._unbatch_observation(new_obs)
        processed_inputs = []
        states = []

        for obs in unbatched_observations:
            vla_step_data = self._to_vla_step_data(obs)
            states.append(vla_step_data.states)
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(self.processor(messages))

        collated_inputs = self.collate_fn(processed_inputs)
        collated_inputs = self._rec_to_dtype(collated_inputs, dtype=torch.bfloat16)

        actual_inputs = collated_inputs["inputs"] if "inputs" in collated_inputs else collated_inputs
        backbone_inputs, action_inputs = self.model.prepare_input(actual_inputs)

        return backbone_inputs, action_inputs, states

    def get_action(self, observation, options=None):
        backbone_inputs, action_inputs, states = self._prepare_inputs(observation)

        should_update_s2 = (
            self._cached_backbone_outputs is None or
            self._step_count % self.vlm_update_freq == 0
        )

        with torch.inference_mode():
            if should_update_s2:
                torch.cuda.synchronize()
                s2_start = time.perf_counter()
                backbone_outputs = self.model.backbone(backbone_inputs)
                torch.cuda.synchronize()
                s2_time = (time.perf_counter() - s2_start) * 1000
                self.s2_times.append(s2_time)

                self._cached_backbone_outputs = backbone_outputs
                self._cached_action_inputs = action_inputs
            else:
                backbone_outputs = self._cached_backbone_outputs
                self._cached_action_inputs.state = action_inputs.state
                action_inputs = self._cached_action_inputs

            torch.cuda.synchronize()
            s1_start = time.perf_counter()

            if self.use_anchor and self._anchor is not None:
                action_outputs = self.model.action_head.get_action(
                    backbone_outputs, action_inputs,
                    anchor=self._anchor,
                    anchor_noise_level=self.anchor_noise_level,
                )
            else:
                action_outputs = self.model.action_head.get_action(backbone_outputs, action_inputs)

            torch.cuda.synchronize()
            s1_time = (time.perf_counter() - s1_start) * 1000
            self.s1_times.append(s1_time)

        if self.use_anchor:
            self._anchor = action_outputs.action_pred.clone()

        self._step_count += 1

        normalized_action = action_outputs.action_pred.float()
        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)

        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )

        casted_action = {
            f"action.{key}": value.astype(np.float32)
            for key, value in unnormalized_action.items()
        }
        return casted_action, {}


class BridgeGr00tPolicy(AsyncGr00tPolicy):
    """
    GR00T Policy using Stable-Dynamic Flow Bridge for feature prediction.

    Uses the DiT rectified flow model to predict layer 16 features
    instead of using stale cached features.

    IMPORTANT: Uses layer 10 as stable context (matches training data)
    """

    def __init__(
        self,
        policy,
        bridge_model,
        flow,
        vlm_update_freq: int = 3,
        use_anchor: bool = True,
        anchor_noise_level: float = 0.05,
        bridge_seq_len: int = 204,
        bridge_state_dim: int = 8,
        bridge_action_dim: int = 7,
        stable_layer_idx: int = 10,  # Layer 10 for stable context (matches training data)
        num_ode_steps: int = 1,  # Number of ODE integration steps (1=one-step, >1=multi-step)
    ):
        super().__init__(policy, vlm_update_freq, use_anchor, anchor_noise_level)
        self.bridge_model = bridge_model
        self.flow = flow
        self.bridge_seq_len = bridge_seq_len
        self.bridge_state_dim = bridge_state_dim
        self.bridge_action_dim = bridge_action_dim
        self.stable_layer_idx = stable_layer_idx
        self.num_ode_steps = num_ode_steps
        self._steps_since_vlm = 0
        self._cached_state = None
        self._cached_action = None
        self._cached_stable_features = None  # Stable layer features

        # Register hook to capture stable layer features
        self._stable_layer_features = None
        self._setup_stable_layer_hook()

    def _setup_stable_layer_hook(self):
        """Register forward hook to capture stable layer hidden states."""
        # Access the language model layers
        backbone = self.model.backbone
        if hasattr(backbone, 'model') and hasattr(backbone.model, 'language_model'):
            layers = backbone.model.language_model.model.layers
            if len(layers) > self.stable_layer_idx:
                def hook_fn(module, input, output):
                    # output is typically (hidden_states, ...) or just hidden_states
                    if isinstance(output, tuple):
                        self._stable_layer_features = output[0].detach()
                    else:
                        self._stable_layer_features = output.detach()
                layers[self.stable_layer_idx].register_forward_hook(hook_fn)
                logger.info(f"Registered hook for layer {self.stable_layer_idx} (stable context)")
            else:
                logger.warning(f"Layer {self.stable_layer_idx} not found, using fallback")

    def reset(self, options=None):
        result = super().reset(options)
        self._steps_since_vlm = 0
        self._cached_state = None
        self._cached_action = None
        self._cached_stable_features = None
        self._stable_layer_features = None
        return result

    def get_action(self, observation, options=None):
        backbone_inputs, action_inputs, states = self._prepare_inputs(observation)

        should_update_s2 = (
            self._cached_backbone_outputs is None or
            self._step_count % self.vlm_update_freq == 0
        )

        with torch.inference_mode():
            if should_update_s2:
                # Run S2 (backbone) - expensive VLM forward
                torch.cuda.synchronize()
                s2_start = time.perf_counter()
                backbone_outputs = self.model.backbone(backbone_inputs)
                torch.cuda.synchronize()
                s2_time = (time.perf_counter() - s2_start) * 1000
                self.s2_times.append(s2_time)

                self._cached_backbone_outputs = backbone_outputs
                self._cached_action_inputs = action_inputs
                self._steps_since_vlm = 0

                # Cache layer 8 features captured by hook (stable context)
                if self._stable_layer_features is not None:
                    self._cached_stable_features = self._stable_layer_features.clone()
                else:
                    # Fallback: use last layer features
                    self._cached_stable_features = backbone_outputs["backbone_features"].clone()

                # Cache state for bridge conditioning
                if "state" in action_inputs:
                    self._cached_state = action_inputs["state"].float()
            else:
                # Use Bridge to PREDICT features instead of using stale ones
                self._steps_since_vlm += 1

                # Get current state for conditioning
                current_state = action_inputs["state"].float()

                # Get cached features (layer 16)
                features = self._cached_backbone_outputs["backbone_features"].float()

                # Prepare features for bridge (handle dimensions)
                while features.ndim > 3:
                    features = features.squeeze(0)

                B, actual_seq_len, feat_dim = features.shape

                # Pad/truncate to bridge seq_len
                if actual_seq_len > self.bridge_seq_len:
                    features_for_bridge = features[:, :self.bridge_seq_len, :]
                elif actual_seq_len < self.bridge_seq_len:
                    pad_len = self.bridge_seq_len - actual_seq_len
                    features_for_bridge = torch.cat([
                        features,
                        torch.zeros(B, pad_len, feat_dim, device=features.device, dtype=features.dtype)
                    ], dim=1)
                else:
                    features_for_bridge = features

                # Prepare state (truncate/pad to match bridge expected dim)
                while current_state.ndim > 2:
                    current_state = current_state.squeeze(0)
                state_dim = current_state.shape[-1]
                if state_dim < self.bridge_state_dim:
                    current_state = torch.cat([
                        current_state,
                        torch.zeros(B, self.bridge_state_dim - state_dim, device=current_state.device, dtype=current_state.dtype)
                    ], dim=-1)
                elif state_dim > self.bridge_state_dim:
                    current_state = current_state[:, :self.bridge_state_dim]

                # Prepare action conditioning (from cached or zeros)
                if self._cached_action is not None:
                    action_cond = self._cached_action.float()
                    # GR00T action output is [B, T, action_dim] where T=50 timesteps
                    # Take first action step and ensure [B, action_dim] shape
                    if action_cond.ndim == 3:
                        action_cond = action_cond[:, 0, :]  # [B, action_dim]
                    elif action_cond.ndim == 2 and action_cond.shape[0] > B:
                        # Shape is [T, action_dim], take first step
                        action_cond = action_cond[0:1, :]  # [1, action_dim]
                    # Ensure batch dimension matches
                    if action_cond.shape[0] != B:
                        action_cond = action_cond[:B] if action_cond.shape[0] > B else action_cond.expand(B, -1)
                    # Truncate/pad action dim
                    if action_cond.shape[-1] < self.bridge_action_dim:
                        action_cond = torch.cat([
                            action_cond,
                            torch.zeros(B, self.bridge_action_dim - action_cond.shape[-1], device=action_cond.device)
                        ], dim=-1)
                    elif action_cond.shape[-1] > self.bridge_action_dim:
                        action_cond = action_cond[:, :self.bridge_action_dim]
                else:
                    action_cond = torch.zeros(B, self.bridge_action_dim, device=features.device)

                # Horizon embedding
                horizon = torch.tensor([self._steps_since_vlm], device=features.device).expand(B)
                horizon = horizon.clamp(0, 10)

                # Prepare stable features (layer 8) for bridge conditioning
                stable_features = self._cached_stable_features.float()
                while stable_features.ndim > 3:
                    stable_features = stable_features.squeeze(0)
                # Pad/truncate stable features to bridge seq_len
                if self.image_only and self._image_mask is not None:
                    stable_for_bridge = self._extract_image_tokens(stable_features)
                elif stable_features.shape[1] > self.bridge_seq_len:
                    stable_for_bridge = stable_features[:, :self.bridge_seq_len, :]
                elif stable_features.shape[1] < self.bridge_seq_len:
                    pad_len = self.bridge_seq_len - stable_features.shape[1]
                    stable_for_bridge = torch.cat([
                        stable_features,
                        torch.zeros(B, pad_len, stable_features.shape[2], device=stable_features.device, dtype=stable_features.dtype)
                    ], dim=1)
                else:
                    stable_for_bridge = stable_features

                # Use bridge model to predict features
                if self.num_ode_steps == 1:
                    # One-step generation: z_1 = z_0 + v(z_0, t=0)
                    t = torch.zeros(B, device=features.device)
                    velocity = self.bridge_model(
                        features_for_bridge,
                        t * 999,  # Scale to match training
                        state=current_state,
                        action=action_cond,
                        horizon=horizon,
                        stable_features=stable_for_bridge,
                    )
                    predicted_features = features_for_bridge + velocity
                else:
                    # Multi-step ODE: Euler integration from t=0 to t=1
                    dt = 1.0 / self.num_ode_steps
                    z_t = features_for_bridge.clone()
                    for step_i in range(self.num_ode_steps):
                        t = torch.full((B,), step_i * dt, device=features.device)
                        velocity = self.bridge_model(
                            z_t,
                            t * 999,  # Scale to match training
                            state=current_state,
                            action=action_cond,
                            horizon=horizon,
                            stable_features=stable_for_bridge,
                        )
                        z_t = z_t + dt * velocity
                    predicted_features = z_t

                # Restore original sequence length
                if actual_seq_len > self.bridge_seq_len:
                    # Concatenate with remainder
                    predicted_features = torch.cat([
                        predicted_features,
                        features[:, self.bridge_seq_len:, :]
                    ], dim=1)
                elif actual_seq_len < self.bridge_seq_len:
                    predicted_features = predicted_features[:, :actual_seq_len, :]

                # Restore original batch dimensions
                target_shape = self._cached_backbone_outputs["backbone_features"].shape
                while predicted_features.ndim < len(target_shape):
                    predicted_features = predicted_features.unsqueeze(0)

                # Create backbone outputs with predicted features
                backbone_outputs = BatchFeature({
                    "backbone_features": predicted_features.to(torch.bfloat16),
                    "backbone_attention_mask": self._cached_backbone_outputs["backbone_attention_mask"],
                    "image_mask": self._cached_backbone_outputs["image_mask"],
                })

                # Update state in action_inputs
                self._cached_action_inputs.state = action_inputs.state
                action_inputs = self._cached_action_inputs

            # Run S1 (action_head) - always runs
            torch.cuda.synchronize()
            s1_start = time.perf_counter()

            if self.use_anchor and self._anchor is not None:
                action_outputs = self.model.action_head.get_action(
                    backbone_outputs, action_inputs,
                    anchor=self._anchor,
                    anchor_noise_level=self.anchor_noise_level,
                )
            else:
                action_outputs = self.model.action_head.get_action(backbone_outputs, action_inputs)

            torch.cuda.synchronize()
            s1_time = (time.perf_counter() - s1_start) * 1000
            self.s1_times.append(s1_time)

        # Update anchor for next step
        if self.use_anchor:
            self._anchor = action_outputs.action_pred.clone()

        # Cache action for conditioning
        self._cached_action = action_outputs.action_pred.clone()

        self._step_count += 1

        # Decode actions
        normalized_action = action_outputs.action_pred.float()
        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)

        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )

        casted_action = {
            f"action.{key}": value.astype(np.float32)
            for key, value in unnormalized_action.items()
        }
        return casted_action, {}


class AutoregressiveBridgeGr00tPolicy(AsyncGr00tPolicy):
    """
    Single-horizon bridge that chains its own predictions.

    Uses SingleStepDiT (no time/horizon) to predict feature deltas.
    Key difference from BridgeGr00tPolicy: updates self._current_features
    with predicted features (chains predictions) instead of always reading from VLM cache.
    """

    def __init__(
        self,
        policy,
        bridge_model,
        vlm_update_freq: int = 3,
        use_anchor: bool = True,
        anchor_noise_level: float = 0.05,
        bridge_seq_len: int = 204,
        bridge_state_dim: int = 8,
        bridge_action_dim: int = 7,
        stable_layer_idx: int = 10,
        diagnose_steps: int = 0,
        diagnose_all: bool = False,
        image_only: bool = False,
    ):
        super().__init__(policy, vlm_update_freq, use_anchor, anchor_noise_level)
        self.bridge_model = bridge_model
        self.bridge_seq_len = bridge_seq_len
        self.bridge_state_dim = bridge_state_dim
        self.bridge_action_dim = bridge_action_dim
        self.stable_layer_idx = stable_layer_idx
        self.diagnose_steps = diagnose_steps
        self.diagnose_all = diagnose_all
        self.image_only = image_only
        self._image_mask = None  # cached image_mask from backbone outputs

        # Current features: may be VLM-fresh or bridge-predicted (chained)
        self._current_features = None
        self._current_stable_features = None
        self._last_action = None
        self._cached_masks = None  # (attention_mask, image_mask) from VLM

        # Per-step diagnostics: {step_idx: {cosine, delta_mag_ratio, ...}}
        self.step_diagnostics = []  # list of per-episode lists

        # Hook for stable layer
        self._stable_layer_features = None
        self._setup_stable_layer_hook()

    def _extract_image_tokens(self, features):
        """Extract image tokens from full feature sequence using cached image_mask."""
        if not self.image_only or self._image_mask is None:
            return features
        mask = self._image_mask  # [seq_len] bool
        while features.ndim > 3:
            features = features.squeeze(0)
        B = features.shape[0]
        seq_len = features.shape[1]
        # Handle variable sequence lengths across tasks
        if mask.shape[0] != seq_len:
            mask = mask[:seq_len] if mask.shape[0] > seq_len else \
                torch.cat([mask, torch.zeros(seq_len - mask.shape[0], dtype=mask.dtype, device=mask.device)])
        return features[:, mask, :]  # [B, n_img, dim]

    def _reconstruct_full_features(self, image_features, full_features):
        """Place image token predictions back into full feature sequence, copy text tokens."""
        if not self.image_only or self._image_mask is None:
            return image_features
        mask = self._image_mask
        result = full_features.clone()
        while result.ndim > 3:
            result = result.squeeze(0)
        seq_len = result.shape[1]
        if mask.shape[0] != seq_len:
            mask = mask[:seq_len] if mask.shape[0] > seq_len else \
                torch.cat([mask, torch.zeros(seq_len - mask.shape[0], dtype=mask.dtype, device=mask.device)])
        result[:, mask, :] = image_features.to(result.dtype)
        return result

    def _setup_stable_layer_hook(self):
        """Register forward hook to capture stable layer hidden states."""
        backbone = self.model.backbone
        if hasattr(backbone, 'model') and hasattr(backbone.model, 'language_model'):
            layers = backbone.model.language_model.model.layers
            if len(layers) > self.stable_layer_idx:
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        self._stable_layer_features = output[0].detach()
                    else:
                        self._stable_layer_features = output.detach()
                layers[self.stable_layer_idx].register_forward_hook(hook_fn)
                logger.info(f"AutoregressiveBridge: hook on layer {self.stable_layer_idx}")
            else:
                logger.warning(f"Layer {self.stable_layer_idx} not found, using fallback")

    def reset(self, options=None):
        # Save previous episode diagnostics before clearing
        if hasattr(self, '_ep_diagnostics') and self._ep_diagnostics:
            self.step_diagnostics.append(self._ep_diagnostics)
        result = super().reset(options)
        self._current_features = None
        self._current_stable_features = None
        self._last_action = None
        self._cached_masks = None
        self._stable_layer_features = None
        self._ep_diagnostics = []
        self._diag_prev_gt = None
        return result

    def get_step_diagnostics(self):
        """Return all collected diagnostics (flush current episode too)."""
        all_diags = list(self.step_diagnostics)
        if self._ep_diagnostics:
            all_diags.append(self._ep_diagnostics)
        return all_diags

    def get_action(self, observation, options=None):
        backbone_inputs, action_inputs, states = self._prepare_inputs(observation)

        should_update_vlm = (
            self._current_features is None or
            self._step_count % self.vlm_update_freq == 0
        )

        with torch.inference_mode():
            if should_update_vlm:
                # Run VLM → fresh features
                torch.cuda.synchronize()
                s2_start = time.perf_counter()
                backbone_outputs = self.model.backbone(backbone_inputs)
                torch.cuda.synchronize()
                s2_time = (time.perf_counter() - s2_start) * 1000
                self.s2_times.append(s2_time)

                # Update current features with fresh VLM output
                self._current_features = backbone_outputs["backbone_features"].clone()
                self._cached_action_inputs = action_inputs
                self._cached_masks = (
                    backbone_outputs["backbone_attention_mask"],
                    backbone_outputs["image_mask"],
                )
                # Cache image_mask for image-only bridge (update if seq_len changes)
                if self.image_only:
                    im = backbone_outputs["image_mask"]
                    while im.ndim > 1:
                        im = im[0]
                    new_mask = im.bool().cpu()
                    if self._image_mask is None or self._image_mask.shape[0] != new_mask.shape[0]:
                        self._image_mask = new_mask

                # Cache stable layer features
                if self._stable_layer_features is not None:
                    self._current_stable_features = self._stable_layer_features.clone()
                else:
                    self._current_stable_features = backbone_outputs["backbone_features"].clone()

                # Diagnostic: save GT features at VLM step
                if self.diagnose_steps > 0 and self._step_count < self.diagnose_steps or self.diagnose_all:
                    gt_feat = backbone_outputs["backbone_features"].float()
                    while gt_feat.ndim > 3:
                        gt_feat = gt_feat.squeeze(0)
                    self._diag_prev_gt = gt_feat.clone()
                    self._ep_diagnostics.append({
                        "step": self._step_count, "type": "vlm",
                    })

                # Use fresh backbone outputs for action head
                # Match action head dtype if needed
                ah_dtype = next(self.model.action_head.parameters()).dtype
                if ah_dtype != backbone_outputs["backbone_features"].dtype:
                    backbone_outputs_for_action = BatchFeature({
                        "backbone_features": backbone_outputs["backbone_features"].to(ah_dtype),
                        "backbone_attention_mask": backbone_outputs["backbone_attention_mask"],
                        "image_mask": backbone_outputs["image_mask"],
                    })
                else:
                    backbone_outputs_for_action = backbone_outputs
            else:
                # Bridge predicts one step from PREVIOUS features (possibly own prediction)
                features = self._current_features.float()
                while features.ndim > 3:
                    features = features.squeeze(0)

                B, actual_seq_len, feat_dim = features.shape

                # Extract image tokens or pad/truncate to bridge seq_len
                if self.image_only and self._image_mask is not None:
                    features_for_bridge = self._extract_image_tokens(features)
                elif actual_seq_len > self.bridge_seq_len:
                    features_for_bridge = features[:, :self.bridge_seq_len, :]
                elif actual_seq_len < self.bridge_seq_len:
                    pad_len = self.bridge_seq_len - actual_seq_len
                    features_for_bridge = torch.cat([
                        features,
                        torch.zeros(B, pad_len, feat_dim, device=features.device, dtype=features.dtype)
                    ], dim=1)
                else:
                    features_for_bridge = features

                # Prepare state
                current_state = action_inputs["state"].float()
                while current_state.ndim > 2:
                    current_state = current_state.squeeze(0)
                state_dim = current_state.shape[-1]
                if state_dim < self.bridge_state_dim:
                    current_state = torch.cat([
                        current_state,
                        torch.zeros(B, self.bridge_state_dim - state_dim, device=current_state.device, dtype=current_state.dtype)
                    ], dim=-1)
                elif state_dim > self.bridge_state_dim:
                    current_state = current_state[:, :self.bridge_state_dim]

                # Prepare action (shifted action from bridge-controlled policy)
                if self._last_action is not None:
                    action_cond = self._last_action.float()
                    if action_cond.ndim == 3:
                        action_cond = action_cond[:, 0, :]
                    elif action_cond.ndim == 2 and action_cond.shape[0] > B:
                        action_cond = action_cond[0:1, :]
                    if action_cond.shape[0] != B:
                        action_cond = action_cond[:B] if action_cond.shape[0] > B else action_cond.expand(B, -1)
                    if action_cond.shape[-1] < self.bridge_action_dim:
                        action_cond = torch.cat([
                            action_cond,
                            torch.zeros(B, self.bridge_action_dim - action_cond.shape[-1], device=action_cond.device)
                        ], dim=-1)
                    elif action_cond.shape[-1] > self.bridge_action_dim:
                        action_cond = action_cond[:, :self.bridge_action_dim]
                else:
                    action_cond = torch.zeros(B, self.bridge_action_dim, device=features.device)

                # Prepare stable features
                stable_features = self._current_stable_features.float()
                while stable_features.ndim > 3:
                    stable_features = stable_features.squeeze(0)
                if self.image_only and self._image_mask is not None:
                    stable_for_bridge = self._extract_image_tokens(stable_features)
                elif stable_features.shape[1] > self.bridge_seq_len:
                    stable_for_bridge = stable_features[:, :self.bridge_seq_len, :]
                elif stable_features.shape[1] < self.bridge_seq_len:
                    pad_len = self.bridge_seq_len - stable_features.shape[1]
                    stable_for_bridge = torch.cat([
                        stable_features,
                        torch.zeros(B, pad_len, stable_features.shape[2], device=stable_features.device, dtype=stable_features.dtype)
                    ], dim=1)
                else:
                    stable_for_bridge = stable_features

                # SingleStepDiT: forward(z0, stable_features, state, action) → delta
                # No time embedding, no horizon embedding
                delta = self.bridge_model(
                    features_for_bridge,
                    stable_for_bridge,
                    current_state,
                    action_cond,
                )
                predicted_image_features = features_for_bridge + delta

                # Restore full sequence
                if self.image_only and self._image_mask is not None:
                    predicted_features = self._reconstruct_full_features(
                        predicted_image_features, features)
                elif actual_seq_len > self.bridge_seq_len:
                    predicted_features = torch.cat([
                        predicted_image_features,
                        features[:, self.bridge_seq_len:, :]
                    ], dim=1)
                elif actual_seq_len < self.bridge_seq_len:
                    predicted_features = predicted_image_features[:, :actual_seq_len, :]
                else:
                    predicted_features = predicted_image_features

                # Chain: update current features with own prediction
                target_shape = self._current_features.shape
                while predicted_features.ndim < len(target_shape):
                    predicted_features = predicted_features.unsqueeze(0)
                self._current_features = predicted_features.to(self._current_features.dtype)

                # Diagnostic: run VLM ground truth and compare
                if self.diagnose_steps > 0 and self._step_count < self.diagnose_steps or self.diagnose_all:
                    # Save/restore stable hook state so GT call doesn't contaminate bridge
                    _saved_stable = self._stable_layer_features
                    gt_backbone_outputs = self.model.backbone(backbone_inputs)
                    self._stable_layer_features = _saved_stable
                    gt_feat = gt_backbone_outputs["backbone_features"].float()
                    pred_feat = predicted_features.float()
                    while gt_feat.ndim > 3:
                        gt_feat = gt_feat.squeeze(0)
                    while pred_feat.ndim > 3:
                        pred_feat = pred_feat.squeeze(0)
                    # Image mask
                    img_mask = self._cached_masks[1] if self._cached_masks else None
                    if img_mask is not None:
                        m = img_mask.squeeze()[:gt_feat.shape[1]].bool()
                        gt_img = gt_feat[:, m, :]
                        pred_img = pred_feat[:, m, :]
                    else:
                        gt_img = gt_feat
                        pred_img = pred_feat

                    steps_since_vlm = self._step_count % self.vlm_update_freq

                    # 1. cos(pred_z1, gt_z1) — feature accuracy
                    feat_cos = F.cosine_similarity(pred_img, gt_img, dim=-1).mean().item()

                    # 2. cos(z_prev, z_current) — copy baseline
                    copy_cos = -1.0
                    delta_cos = -1.0
                    delta_mag_ratio = -1.0
                    if hasattr(self, '_diag_prev_gt') and self._diag_prev_gt is not None:
                        prev_gt = self._diag_prev_gt
                        if img_mask is not None:
                            prev_img = prev_gt[:, m, :]
                        else:
                            prev_img = prev_gt
                        copy_cos = F.cosine_similarity(prev_img, gt_img, dim=-1).mean().item()

                        # 3. cos(pred_delta, gt_delta)
                        gt_delta = gt_img - prev_img
                        pred_delta = pred_img - prev_img
                        delta_cos = F.cosine_similarity(pred_delta, gt_delta, dim=-1).mean().item()
                        gt_dn = gt_delta.norm(dim=-1).mean().item()
                        pred_dn = pred_delta.norm(dim=-1).mean().item()
                        delta_mag_ratio = pred_dn / (gt_dn + 1e-8)

                    # Save current GT for next step
                    self._diag_prev_gt = gt_feat.clone()

                    diag = {
                        "step": self._step_count,
                        "type": "bridge",
                        "steps_since_vlm": steps_since_vlm,
                        "feat_cos": feat_cos,
                        "copy_cos": copy_cos,
                        "delta_cos": delta_cos,
                        "delta_mag_ratio": delta_mag_ratio,
                    }
                    self._ep_diagnostics.append(diag)
                    logger.info(f"  [diag] step={self._step_count} offset={steps_since_vlm} "
                                f"feat_cos={feat_cos:.4f} copy_cos={copy_cos:.4f} "
                                f"delta_cos={delta_cos:.4f} Δ_ratio={delta_mag_ratio:.3f}")

                # Create backbone outputs with predicted features
                # Match action head dtype (bfloat16 for original, float32 for fine-tuned)
                ah_dtype = next(self.model.action_head.parameters()).dtype
                backbone_outputs_for_action = BatchFeature({
                    "backbone_features": predicted_features.to(ah_dtype),
                    "backbone_attention_mask": self._cached_masks[0],
                    "image_mask": self._cached_masks[1],
                })

                # Update state in action_inputs
                self._cached_action_inputs.state = action_inputs.state
                action_inputs = self._cached_action_inputs

            # Run action head — cast inputs to match head dtype if needed
            _ah_dtype = next(self.model.action_head.parameters()).dtype
            if hasattr(action_inputs, 'state') and action_inputs.state.dtype != _ah_dtype:
                action_inputs.state = action_inputs.state.to(_ah_dtype)

            torch.cuda.synchronize()
            s1_start = time.perf_counter()

            if self.use_anchor and self._anchor is not None:
                action_outputs = self.model.action_head.get_action(
                    backbone_outputs_for_action, action_inputs,
                    anchor=self._anchor,
                    anchor_noise_level=self.anchor_noise_level,
                )
            else:
                action_outputs = self.model.action_head.get_action(
                    backbone_outputs_for_action, action_inputs
                )

            torch.cuda.synchronize()
            s1_time = (time.perf_counter() - s1_start) * 1000
            self.s1_times.append(s1_time)

        # Update anchor
        if self.use_anchor:
            self._anchor = action_outputs.action_pred.clone()

        # Save action for next bridge call (shifted action)
        self._last_action = action_outputs.action_pred.clone()

        self._step_count += 1

        # Decode actions
        normalized_action = action_outputs.action_pred.float()
        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)

        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )

        casted_action = {
            f"action.{key}": value.astype(np.float32)
            for key, value in unnormalized_action.items()
        }
        return casted_action, {}


class PhaseAwareAsyncGr00tPolicy(AsyncGr00tPolicy):
    """
    Phase-aware async policy with dynamic VLM frequency (no bridge).

    Uses stale cached features during non-VLM steps (same as AsyncGr00tPolicy),
    but adjusts VLM call frequency based on detected task phase:
    - Navigation (high action mag): fewer VLM calls
    - Manipulation (low action mag): sync or near-sync
    """

    def __init__(
        self,
        policy,
        vlm_update_freq: int = 3,
        use_anchor: bool = True,
        anchor_noise_level: float = 0.05,
        # Phase-aware params
        nav_vlm_freq: int = 5,
        trans_vlm_freq: int = 3,
        manip_vlm_freq: int = 1,
        nav_threshold: float = 0.55,
        manip_threshold: float = 0.35,
        hysteresis_margin: float = 0.03,
        smoothing_window: int = 3,
        force_vlm_on_phase_change: bool = True,
    ):
        super().__init__(policy, vlm_update_freq, use_anchor, anchor_noise_level)

        self.nav_vlm_freq = nav_vlm_freq
        self.trans_vlm_freq = trans_vlm_freq
        self.manip_vlm_freq = manip_vlm_freq
        self.nav_threshold = nav_threshold
        self.manip_threshold = manip_threshold
        self.hysteresis_margin = hysteresis_margin
        self.smoothing_window = smoothing_window
        self.force_vlm_on_phase_change = force_vlm_on_phase_change

        self._current_phase = "navigation"
        self._action_mag_history = deque(maxlen=smoothing_window)
        self._steps_since_vlm_pa = 0
        self._current_vlm_freq = nav_vlm_freq
        self._last_action = None

        self._phase_counts = {"navigation": 0, "transition": 0, "manipulation": 0}
        self._vlm_call_count = 0
        self._phase_transitions = 0
        self._total_steps = 0

    def reset(self, options=None):
        result = super().reset(options)
        self._current_phase = "navigation"
        self._action_mag_history.clear()
        self._steps_since_vlm_pa = 0
        self._current_vlm_freq = self.nav_vlm_freq
        self._last_action = None
        self._phase_counts = {"navigation": 0, "transition": 0, "manipulation": 0}
        self._vlm_call_count = 0
        self._phase_transitions = 0
        self._total_steps = 0
        return result

    def _compute_action_magnitude(self, action_pred):
        """Compute L2 norm of position+rotation dims (exclude gripper)."""
        if action_pred is None:
            return 0.0
        act = action_pred.float()
        if act.ndim == 3:
            act = act[:, 0, :6]
        elif act.ndim == 2:
            act = act[:, :6]
        else:
            return 0.0
        return act.norm(dim=-1).mean().item()

    def _detect_phase(self, action_mag):
        """State machine with hysteresis for phase detection."""
        self._action_mag_history.append(action_mag)
        smoothed_mag = np.mean(self._action_mag_history)

        old_phase = self._current_phase
        margin = self.hysteresis_margin

        if self._current_phase == "navigation":
            if smoothed_mag < self.nav_threshold - margin:
                if smoothed_mag < self.manip_threshold - margin:
                    self._current_phase = "manipulation"
                else:
                    self._current_phase = "transition"
        elif self._current_phase == "transition":
            if smoothed_mag > self.nav_threshold + margin:
                self._current_phase = "navigation"
            elif smoothed_mag < self.manip_threshold - margin:
                self._current_phase = "manipulation"
        elif self._current_phase == "manipulation":
            if smoothed_mag > self.manip_threshold + margin:
                if smoothed_mag > self.nav_threshold + margin:
                    self._current_phase = "navigation"
                else:
                    self._current_phase = "transition"

        freq_map = {
            "navigation": self.nav_vlm_freq,
            "transition": self.trans_vlm_freq,
            "manipulation": self.manip_vlm_freq,
        }
        self._current_vlm_freq = freq_map[self._current_phase]

        phase_changed = (self._current_phase != old_phase)
        if phase_changed:
            self._phase_transitions += 1
        return phase_changed

    def get_phase_statistics(self):
        """Return phase distribution and VLM savings diagnostics."""
        total = sum(self._phase_counts.values())
        if total == 0:
            return {}
        phase_pcts = {k: v / total * 100 for k, v in self._phase_counts.items()}
        effective_freq = self._total_steps / max(self._vlm_call_count, 1)
        vlm_savings = 1.0 - (self._vlm_call_count / max(self._total_steps, 1))
        return {
            "phase_distribution": phase_pcts,
            "effective_vlm_freq": effective_freq,
            "vlm_savings_vs_sync": vlm_savings * 100,
            "vlm_calls": self._vlm_call_count,
            "total_steps": self._total_steps,
            "phase_transitions": self._phase_transitions,
        }

    def get_action(self, observation, options=None):
        backbone_inputs, action_inputs, states = self._prepare_inputs(observation)

        # Phase detection from previous action
        if self._last_action is not None:
            action_mag = self._compute_action_magnitude(self._last_action)
            if self._total_steps % 10 == 0:
                smoothed = np.mean(self._action_mag_history) if self._action_mag_history else action_mag
                logger.debug(f"  step={self._total_steps} action_mag={action_mag:.4f} smoothed={smoothed:.4f} phase={self._current_phase} freq={self._current_vlm_freq}")
            phase_changed = self._detect_phase(action_mag)
        else:
            phase_changed = False

        self._phase_counts[self._current_phase] += 1
        self._total_steps += 1

        # Dynamic VLM decision
        should_update_vlm = (
            self._cached_backbone_outputs is None or
            self._steps_since_vlm_pa >= self._current_vlm_freq or
            (phase_changed and self.force_vlm_on_phase_change
             and self._current_phase == "navigation")
        )

        with torch.inference_mode():
            if should_update_vlm:
                torch.cuda.synchronize()
                s2_start = time.perf_counter()
                backbone_outputs = self.model.backbone(backbone_inputs)
                torch.cuda.synchronize()
                s2_time = (time.perf_counter() - s2_start) * 1000
                self.s2_times.append(s2_time)

                self._cached_backbone_outputs = backbone_outputs
                self._cached_action_inputs = action_inputs

                self._steps_since_vlm_pa = 0
                self._vlm_call_count += 1
            else:
                # Use stale cached features (no bridge prediction)
                backbone_outputs = self._cached_backbone_outputs
                self._cached_action_inputs.state = action_inputs.state
                action_inputs = self._cached_action_inputs

            torch.cuda.synchronize()
            s1_start = time.perf_counter()

            if self.use_anchor and self._anchor is not None:
                action_outputs = self.model.action_head.get_action(
                    backbone_outputs, action_inputs,
                    anchor=self._anchor,
                    anchor_noise_level=self.anchor_noise_level,
                )
            else:
                action_outputs = self.model.action_head.get_action(
                    backbone_outputs, action_inputs
                )

            torch.cuda.synchronize()
            s1_time = (time.perf_counter() - s1_start) * 1000
            self.s1_times.append(s1_time)

        if self.use_anchor:
            self._anchor = action_outputs.action_pred.clone()

        self._last_action = action_outputs.action_pred.clone()
        self._step_count += 1
        self._steps_since_vlm_pa += 1

        normalized_action = action_outputs.action_pred.float()
        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )
        casted_action = {
            f"action.{key}": value.astype(np.float32)
            for key, value in unnormalized_action.items()
        }
        return casted_action, {}


class PhaseAwareBridgeGr00tPolicy(AutoregressiveBridgeGr00tPolicy):
    """
    Phase-aware AR bridge with dynamic VLM frequency based on task phase.

    Uses PREVIOUS ACTION TRANSLATION MAGNITUDE (dims 0-2) as phase signal.
    Large translation → large observation change → large feature delta → bridge struggles.
    Feature delta fully mediates the action→bridge_quality relationship (partial r≈0).

    Phase mapping (based on prev_trans × offset analysis):
    - High translation (>0.5): freq=2 — cos=0.949@off1, 0.919@off2, don't chain
    - Medium translation (0.1-0.5): freq=3 — cos=0.935-0.964, standard
    - Low translation (<0.1): freq=4 — cos=0.950@off2, safe for longer chains

    Uses smoothed translation magnitude with hysteresis to prevent phase flickering.
    """

    def __init__(
        self,
        policy,
        bridge_model,
        vlm_update_freq: int = 3,  # fallback / initial freq
        use_anchor: bool = True,
        anchor_noise_level: float = 0.05,
        bridge_seq_len: int = 204,
        bridge_state_dim: int = 8,
        bridge_action_dim: int = 7,
        stable_layer_idx: int = 10,
        # Phase-aware params: high translation = more VLM
        nav_vlm_freq: int = 2,      # high trans (>0.5) → freq=2
        trans_vlm_freq: int = 3,     # medium trans → freq=3
        manip_vlm_freq: int = 4,     # low trans (<0.1) → freq=4
        nav_threshold: float = 0.5,  # prev_trans above this → high phase
        manip_threshold: float = 0.1,  # prev_trans below this → low phase
        hysteresis_margin: float = 0.03,
        smoothing_window: int = 3,
        force_vlm_on_phase_change: bool = True,
        image_only: bool = False,
    ):
        super().__init__(
            policy, bridge_model,
            vlm_update_freq=vlm_update_freq,
            use_anchor=use_anchor,
            anchor_noise_level=anchor_noise_level,
            bridge_seq_len=bridge_seq_len,
            bridge_state_dim=bridge_state_dim,
            bridge_action_dim=bridge_action_dim,
            stable_layer_idx=stable_layer_idx,
            image_only=image_only,
        )

        # Phase frequency mapping
        self.nav_vlm_freq = nav_vlm_freq
        self.trans_vlm_freq = trans_vlm_freq
        self.manip_vlm_freq = manip_vlm_freq

        # Phase detection thresholds
        self.nav_threshold = nav_threshold
        self.manip_threshold = manip_threshold
        self.hysteresis_margin = hysteresis_margin
        self.smoothing_window = smoothing_window
        self.force_vlm_on_phase_change = force_vlm_on_phase_change

        # Phase state
        self._current_phase = "navigation"
        self._action_mag_history = deque(maxlen=smoothing_window)
        self._steps_since_vlm_pa = 0  # replaces modulo-based check
        self._current_vlm_freq = nav_vlm_freq

        # Diagnostics
        self._phase_counts = {"navigation": 0, "transition": 0, "manipulation": 0}
        self._vlm_call_count = 0
        self._phase_transitions = 0
        self._total_steps = 0

    def reset(self, options=None):
        result = super().reset(options)
        self._current_phase = "navigation"
        self._action_mag_history.clear()
        self._steps_since_vlm_pa = 0
        self._current_vlm_freq = self.nav_vlm_freq
        self._phase_counts = {"navigation": 0, "transition": 0, "manipulation": 0}
        self._vlm_call_count = 0
        self._phase_transitions = 0
        self._total_steps = 0
        return result

    def _compute_action_magnitude(self, action_pred):
        """Compute L2 norm of translation dims only (0-2), not rotation/gripper.

        Translation magnitude is the best available proxy for feature delta (r=0.49),
        which is the true predictor of bridge quality (r=-0.66).
        """
        # action_pred: [B, action_horizon, action_dim]
        if action_pred is None:
            return 0.0
        act = action_pred.float()
        if act.ndim == 3:
            act = act[:, 0, :3]  # first timestep, translation only (3 dims)
        elif act.ndim == 2:
            act = act[:, :3]
        else:
            return 0.0
        return act.norm(dim=-1).mean().item()

    def _detect_phase(self, action_mag):
        """State machine with hysteresis for phase detection."""
        self._action_mag_history.append(action_mag)
        smoothed_mag = np.mean(self._action_mag_history)

        old_phase = self._current_phase
        margin = self.hysteresis_margin

        if self._current_phase == "navigation":
            # Drop to transition if below nav_threshold - margin
            if smoothed_mag < self.nav_threshold - margin:
                if smoothed_mag < self.manip_threshold - margin:
                    self._current_phase = "manipulation"
                else:
                    self._current_phase = "transition"
        elif self._current_phase == "transition":
            # Rise to navigation if above nav_threshold + margin
            if smoothed_mag > self.nav_threshold + margin:
                self._current_phase = "navigation"
            # Drop to manipulation if below manip_threshold - margin
            elif smoothed_mag < self.manip_threshold - margin:
                self._current_phase = "manipulation"
        elif self._current_phase == "manipulation":
            # Rise to transition if above manip_threshold + margin
            if smoothed_mag > self.manip_threshold + margin:
                if smoothed_mag > self.nav_threshold + margin:
                    self._current_phase = "navigation"
                else:
                    self._current_phase = "transition"

        # Update freq based on phase
        freq_map = {
            "navigation": self.nav_vlm_freq,
            "transition": self.trans_vlm_freq,
            "manipulation": self.manip_vlm_freq,
        }
        self._current_vlm_freq = freq_map[self._current_phase]

        phase_changed = (self._current_phase != old_phase)
        if phase_changed:
            self._phase_transitions += 1

        return phase_changed

    def get_phase_statistics(self):
        """Return phase distribution and VLM savings diagnostics."""
        total = sum(self._phase_counts.values())
        if total == 0:
            return {}

        phase_pcts = {k: v / total * 100 for k, v in self._phase_counts.items()}

        # Effective vlm freq = total_steps / vlm_calls
        effective_freq = self._total_steps / max(self._vlm_call_count, 1)

        # VLM savings vs sync (sync = 1 call per step)
        vlm_savings = 1.0 - (self._vlm_call_count / max(self._total_steps, 1))

        return {
            "phase_distribution": phase_pcts,
            "effective_vlm_freq": effective_freq,
            "vlm_savings_vs_sync": vlm_savings * 100,
            "vlm_calls": self._vlm_call_count,
            "total_steps": self._total_steps,
            "phase_transitions": self._phase_transitions,
        }

    def get_action(self, observation, options=None):
        backbone_inputs, action_inputs, states = self._prepare_inputs(observation)

        # Phase detection from previous action
        if self._last_action is not None:
            action_mag = self._compute_action_magnitude(self._last_action)
            phase_changed = self._detect_phase(action_mag)
        else:
            phase_changed = False

        # Track phase
        self._phase_counts[self._current_phase] += 1
        self._total_steps += 1

        # VLM decision: first step, freq exceeded, or forced on phase change
        should_update_vlm = (
            self._current_features is None or
            self._steps_since_vlm_pa >= self._current_vlm_freq or
            (phase_changed and self.force_vlm_on_phase_change
             and self._current_phase == "navigation")
        )

        with torch.inference_mode():
            if should_update_vlm:
                # Run VLM → fresh features
                torch.cuda.synchronize()
                s2_start = time.perf_counter()
                backbone_outputs = self.model.backbone(backbone_inputs)
                torch.cuda.synchronize()
                s2_time = (time.perf_counter() - s2_start) * 1000
                self.s2_times.append(s2_time)

                self._current_features = backbone_outputs["backbone_features"].clone()
                self._cached_action_inputs = action_inputs
                self._cached_masks = (
                    backbone_outputs["backbone_attention_mask"],
                    backbone_outputs["image_mask"],
                )

                if self._stable_layer_features is not None:
                    self._current_stable_features = self._stable_layer_features.clone()
                else:
                    self._current_stable_features = backbone_outputs["backbone_features"].clone()

                backbone_outputs_for_action = backbone_outputs

                self._steps_since_vlm_pa = 0
                self._vlm_call_count += 1
            else:
                # Bridge predicts one step from PREVIOUS features (chained)
                features = self._current_features.float()
                while features.ndim > 3:
                    features = features.squeeze(0)

                B, actual_seq_len, feat_dim = features.shape

                # Extract image tokens or pad/truncate features to bridge seq_len
                if self.image_only and self._image_mask is not None:
                    features_for_bridge = self._extract_image_tokens(features)
                elif actual_seq_len > self.bridge_seq_len:
                    features_for_bridge = features[:, :self.bridge_seq_len, :]
                elif actual_seq_len < self.bridge_seq_len:
                    pad_len = self.bridge_seq_len - actual_seq_len
                    features_for_bridge = torch.cat([
                        features,
                        torch.zeros(B, pad_len, feat_dim, device=features.device, dtype=features.dtype)
                    ], dim=1)
                else:
                    features_for_bridge = features

                # Prepare state
                current_state = action_inputs["state"].float()
                while current_state.ndim > 2:
                    current_state = current_state.squeeze(0)
                state_dim = current_state.shape[-1]
                if state_dim < self.bridge_state_dim:
                    current_state = torch.cat([
                        current_state,
                        torch.zeros(B, self.bridge_state_dim - state_dim, device=current_state.device, dtype=current_state.dtype)
                    ], dim=-1)
                elif state_dim > self.bridge_state_dim:
                    current_state = current_state[:, :self.bridge_state_dim]

                # Prepare action
                if self._last_action is not None:
                    action_cond = self._last_action.float()
                    if action_cond.ndim == 3:
                        action_cond = action_cond[:, 0, :]
                    elif action_cond.ndim == 2 and action_cond.shape[0] > B:
                        action_cond = action_cond[0:1, :]
                    if action_cond.shape[0] != B:
                        action_cond = action_cond[:B] if action_cond.shape[0] > B else action_cond.expand(B, -1)
                    if action_cond.shape[-1] < self.bridge_action_dim:
                        action_cond = torch.cat([
                            action_cond,
                            torch.zeros(B, self.bridge_action_dim - action_cond.shape[-1], device=action_cond.device)
                        ], dim=-1)
                    elif action_cond.shape[-1] > self.bridge_action_dim:
                        action_cond = action_cond[:, :self.bridge_action_dim]
                else:
                    action_cond = torch.zeros(B, self.bridge_action_dim, device=features.device)

                # Prepare stable features
                stable_features = self._current_stable_features.float()
                while stable_features.ndim > 3:
                    stable_features = stable_features.squeeze(0)
                if self.image_only and self._image_mask is not None:
                    stable_for_bridge = self._extract_image_tokens(stable_features)
                elif stable_features.shape[1] > self.bridge_seq_len:
                    stable_for_bridge = stable_features[:, :self.bridge_seq_len, :]
                elif stable_features.shape[1] < self.bridge_seq_len:
                    pad_len = self.bridge_seq_len - stable_features.shape[1]
                    stable_for_bridge = torch.cat([
                        stable_features,
                        torch.zeros(B, pad_len, stable_features.shape[2], device=stable_features.device, dtype=stable_features.dtype)
                    ], dim=1)
                else:
                    stable_for_bridge = stable_features

                # SingleStepDiT: forward(z0, stable_features, state, action) → delta
                delta = self.bridge_model(
                    features_for_bridge,
                    stable_for_bridge,
                    current_state,
                    action_cond,
                )
                predicted_image_features = features_for_bridge + delta

                # Restore full sequence
                if self.image_only and self._image_mask is not None:
                    predicted_features = self._reconstruct_full_features(
                        predicted_image_features, features)
                elif actual_seq_len > self.bridge_seq_len:
                    predicted_features = torch.cat([
                        predicted_image_features,
                        features[:, self.bridge_seq_len:, :]
                    ], dim=1)
                elif actual_seq_len < self.bridge_seq_len:
                    predicted_features = predicted_image_features[:, :actual_seq_len, :]
                else:
                    predicted_features = predicted_image_features

                # Chain: update current features with own prediction
                target_shape = self._current_features.shape
                while predicted_features.ndim < len(target_shape):
                    predicted_features = predicted_features.unsqueeze(0)
                self._current_features = predicted_features.to(self._current_features.dtype)

                backbone_outputs_for_action = BatchFeature({
                    "backbone_features": predicted_features.to(torch.bfloat16),
                    "backbone_attention_mask": self._cached_masks[0],
                    "image_mask": self._cached_masks[1],
                })

                self._cached_action_inputs.state = action_inputs.state
                action_inputs = self._cached_action_inputs

            # Run action head — cast inputs to match head dtype if needed
            _ah_dtype = next(self.model.action_head.parameters()).dtype
            if hasattr(action_inputs, 'state') and action_inputs.state.dtype != _ah_dtype:
                action_inputs.state = action_inputs.state.to(_ah_dtype)

            torch.cuda.synchronize()
            s1_start = time.perf_counter()

            if self.use_anchor and self._anchor is not None:
                action_outputs = self.model.action_head.get_action(
                    backbone_outputs_for_action, action_inputs,
                    anchor=self._anchor,
                    anchor_noise_level=self.anchor_noise_level,
                )
            else:
                action_outputs = self.model.action_head.get_action(
                    backbone_outputs_for_action, action_inputs
                )

            torch.cuda.synchronize()
            s1_time = (time.perf_counter() - s1_start) * 1000
            self.s1_times.append(s1_time)

        # Update anchor
        if self.use_anchor:
            self._anchor = action_outputs.action_pred.clone()

        # Save action for next bridge call and phase detection
        self._last_action = action_outputs.action_pred.clone()

        self._step_count += 1
        self._steps_since_vlm_pa += 1

        # Decode actions
        normalized_action = action_outputs.action_pred.float()
        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )
        casted_action = {
            f"action.{key}": value.astype(np.float32)
            for key, value in unnormalized_action.items()
        }
        return casted_action, {}


class TransitionTriggeredBridgeGr00tPolicy(AutoregressiveBridgeGr00tPolicy):
    """
    AR bridge with transition-triggered VLM calls.

    Base AR bridge with vlm_freq scheduling, PLUS: forces a VLM call whenever
    the action magnitude crosses a threshold (default 0.2) in either direction.
    This directly targets phase transitions (idle↔active), the #1 predictor
    of bridge degradation (r=-0.74).

    Extra VLM calls only happen at transition moments (~6-7 per hard episode),
    keeping overall VLM usage moderate.
    """

    def __init__(
        self,
        policy,
        bridge_model,
        vlm_update_freq: int = 3,
        use_anchor: bool = True,
        anchor_noise_level: float = 0.05,
        bridge_seq_len: int = 204,
        bridge_state_dim: int = 8,
        bridge_action_dim: int = 7,
        stable_layer_idx: int = 10,
        transition_threshold: float = 0.2,
        smoothing_window: int = 2,
    ):
        super().__init__(
            policy, bridge_model,
            vlm_update_freq=vlm_update_freq,
            use_anchor=use_anchor,
            anchor_noise_level=anchor_noise_level,
            bridge_seq_len=bridge_seq_len,
            bridge_state_dim=bridge_state_dim,
            bridge_action_dim=bridge_action_dim,
            stable_layer_idx=stable_layer_idx,
        )
        self.transition_threshold = transition_threshold
        self.smoothing_window = smoothing_window
        self._action_mag_history = deque(maxlen=smoothing_window)
        self._prev_above_threshold = None  # None = unknown at start
        self._steps_since_vlm_tt = 0
        # Diagnostics
        self._transition_vlm_calls = 0
        self._scheduled_vlm_calls = 0
        self._total_vlm_calls = 0
        self._total_steps_tt = 0

    def reset(self, options=None):
        result = super().reset(options)
        self._action_mag_history.clear()
        self._prev_above_threshold = None
        self._steps_since_vlm_tt = 0
        self._transition_vlm_calls = 0
        self._scheduled_vlm_calls = 0
        self._total_vlm_calls = 0
        self._total_steps_tt = 0
        return result

    def _compute_action_magnitude(self, action_pred):
        """Compute L2 norm of position+rotation dims (exclude gripper)."""
        if action_pred is None:
            return 0.0
        act = action_pred.float()
        if act.ndim == 3:
            act = act[:, 0, :6]
        elif act.ndim == 2:
            act = act[:, :6]
        else:
            return 0.0
        return act.norm(dim=-1).mean().item()

    def _detect_transition(self, action_mag):
        """Check if action magnitude crossed threshold. Returns True on crossing."""
        self._action_mag_history.append(action_mag)
        smoothed = np.mean(self._action_mag_history)
        above = smoothed >= self.transition_threshold

        if self._prev_above_threshold is None:
            self._prev_above_threshold = above
            return False

        crossed = (above != self._prev_above_threshold)
        self._prev_above_threshold = above
        return crossed

    def get_action(self, observation, options=None):
        backbone_inputs, action_inputs, states = self._prepare_inputs(observation)

        self._total_steps_tt += 1

        # Detect transition from previous action
        transition_triggered = False
        if self._last_action is not None:
            action_mag = self._compute_action_magnitude(self._last_action)
            transition_triggered = self._detect_transition(action_mag)

        # VLM decision: first step, scheduled, or transition-triggered
        scheduled = (
            self._current_features is None or
            self._steps_since_vlm_tt >= self.vlm_update_freq
        )
        should_update_vlm = scheduled or transition_triggered

        with torch.inference_mode():
            if should_update_vlm:
                # Run VLM
                torch.cuda.synchronize()
                s2_start = time.perf_counter()
                backbone_outputs = self.model.backbone(backbone_inputs)
                torch.cuda.synchronize()
                s2_time = (time.perf_counter() - s2_start) * 1000
                self.s2_times.append(s2_time)

                self._current_features = backbone_outputs["backbone_features"].clone()
                self._cached_action_inputs = action_inputs
                self._cached_masks = (
                    backbone_outputs["backbone_attention_mask"],
                    backbone_outputs["image_mask"],
                )

                if self._stable_layer_features is not None:
                    self._current_stable_features = self._stable_layer_features.clone()
                else:
                    self._current_stable_features = backbone_outputs["backbone_features"].clone()

                backbone_outputs_for_action = backbone_outputs

                self._steps_since_vlm_tt = 0
                self._total_vlm_calls += 1
                if transition_triggered and not scheduled:
                    self._transition_vlm_calls += 1
                else:
                    self._scheduled_vlm_calls += 1
            else:
                # Bridge step (identical to parent)
                features = self._current_features.float()
                while features.ndim > 3:
                    features = features.squeeze(0)

                B, actual_seq_len, feat_dim = features.shape

                if actual_seq_len > self.bridge_seq_len:
                    features_for_bridge = features[:, :self.bridge_seq_len, :]
                elif actual_seq_len < self.bridge_seq_len:
                    pad_len = self.bridge_seq_len - actual_seq_len
                    features_for_bridge = torch.cat([
                        features,
                        torch.zeros(B, pad_len, feat_dim, device=features.device, dtype=features.dtype)
                    ], dim=1)
                else:
                    features_for_bridge = features

                current_state = action_inputs["state"].float()
                while current_state.ndim > 2:
                    current_state = current_state.squeeze(0)
                state_dim = current_state.shape[-1]
                if state_dim < self.bridge_state_dim:
                    current_state = torch.cat([
                        current_state,
                        torch.zeros(B, self.bridge_state_dim - state_dim, device=current_state.device, dtype=current_state.dtype)
                    ], dim=-1)
                elif state_dim > self.bridge_state_dim:
                    current_state = current_state[:, :self.bridge_state_dim]

                if self._last_action is not None:
                    action_cond = self._last_action.float()
                    if action_cond.ndim == 3:
                        action_cond = action_cond[:, 0, :]
                    elif action_cond.ndim == 2 and action_cond.shape[0] > B:
                        action_cond = action_cond[0:1, :]
                    if action_cond.shape[0] != B:
                        action_cond = action_cond[:B] if action_cond.shape[0] > B else action_cond.expand(B, -1)
                    if action_cond.shape[-1] < self.bridge_action_dim:
                        action_cond = torch.cat([
                            action_cond,
                            torch.zeros(B, self.bridge_action_dim - action_cond.shape[-1], device=action_cond.device)
                        ], dim=-1)
                    elif action_cond.shape[-1] > self.bridge_action_dim:
                        action_cond = action_cond[:, :self.bridge_action_dim]
                else:
                    action_cond = torch.zeros(B, self.bridge_action_dim, device=features.device)

                stable_features = self._current_stable_features.float()
                while stable_features.ndim > 3:
                    stable_features = stable_features.squeeze(0)
                if self.image_only and self._image_mask is not None:
                    stable_for_bridge = self._extract_image_tokens(stable_features)
                elif stable_features.shape[1] > self.bridge_seq_len:
                    stable_for_bridge = stable_features[:, :self.bridge_seq_len, :]
                elif stable_features.shape[1] < self.bridge_seq_len:
                    pad_len = self.bridge_seq_len - stable_features.shape[1]
                    stable_for_bridge = torch.cat([
                        stable_features,
                        torch.zeros(B, pad_len, stable_features.shape[2], device=stable_features.device, dtype=stable_features.dtype)
                    ], dim=1)
                else:
                    stable_for_bridge = stable_features

                delta = self.bridge_model(
                    features_for_bridge,
                    stable_for_bridge,
                    current_state,
                    action_cond,
                )
                predicted_features = features_for_bridge + delta

                if actual_seq_len > self.bridge_seq_len:
                    predicted_features = torch.cat([
                        predicted_features,
                        features[:, self.bridge_seq_len:, :]
                    ], dim=1)
                elif actual_seq_len < self.bridge_seq_len:
                    predicted_features = predicted_features[:, :actual_seq_len, :]

                target_shape = self._current_features.shape
                while predicted_features.ndim < len(target_shape):
                    predicted_features = predicted_features.unsqueeze(0)
                self._current_features = predicted_features.to(self._current_features.dtype)

                backbone_outputs_for_action = BatchFeature({
                    "backbone_features": predicted_features.to(torch.bfloat16),
                    "backbone_attention_mask": self._cached_masks[0],
                    "image_mask": self._cached_masks[1],
                })

                self._cached_action_inputs.state = action_inputs.state
                action_inputs = self._cached_action_inputs

            # Run action head
            _ah_dtype = next(self.model.action_head.parameters()).dtype
            if hasattr(action_inputs, 'state') and action_inputs.state.dtype != _ah_dtype:
                action_inputs.state = action_inputs.state.to(_ah_dtype)

            torch.cuda.synchronize()
            s1_start = time.perf_counter()

            if self.use_anchor and self._anchor is not None:
                action_outputs = self.model.action_head.get_action(
                    backbone_outputs_for_action, action_inputs,
                    anchor=self._anchor,
                    anchor_noise_level=self.anchor_noise_level,
                )
            else:
                action_outputs = self.model.action_head.get_action(
                    backbone_outputs_for_action, action_inputs
                )

            torch.cuda.synchronize()
            s1_time = (time.perf_counter() - s1_start) * 1000
            self.s1_times.append(s1_time)

        # Update anchor
        if self.use_anchor:
            self._anchor = action_outputs.action_pred.clone()

        self._last_action = action_outputs.action_pred.clone()
        self._step_count += 1
        self._steps_since_vlm_tt += 1

        # Decode actions
        normalized_action = action_outputs.action_pred.float()
        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )
        casted_action = {
            f"action.{key}": value.astype(np.float32)
            for key, value in unnormalized_action.items()
        }
        return casted_action, {}

    def get_transition_statistics(self):
        """Return transition-triggered VLM diagnostics."""
        total = self._total_steps_tt
        if total == 0:
            return {}
        effective_freq = total / max(self._total_vlm_calls, 1)
        vlm_savings = 1.0 - (self._total_vlm_calls / max(total, 1))
        return {
            "total_vlm_calls": self._total_vlm_calls,
            "scheduled_vlm_calls": self._scheduled_vlm_calls,
            "transition_vlm_calls": self._transition_vlm_calls,
            "total_steps": total,
            "effective_vlm_freq": effective_freq,
            "vlm_savings_vs_sync": vlm_savings * 100,
        }


class LinearInterpGr00tPolicy(AsyncGr00tPolicy):
    """
    Oracle linear interpolation policy for closed-loop evaluation.

    At each VLM refresh interval (every vlm_freq steps):
    1. Save env state
    2. Run sync rollout for vlm_freq steps → collect z_0_sync, z_1_sync, z_2_sync
    3. Restore env state
    4. Run interpolation rollout:
       - t=0: use z_0 (fresh), execute action
       - t=k: use alpha*z_0 + (1-alpha)*z_k_sync, execute action
    5. Next interval starts from interp rollout's env state

    This tests whether linear interpolation of hidden states works in closed-loop
    where actions from interpolated features affect the actual environment.
    """

    def __init__(self, policy, vlm_update_freq: int = 3, use_anchor: bool = True,
                 anchor_noise_level: float = 0.05, alpha: float = 0.5):
        super().__init__(policy, vlm_update_freq, use_anchor, anchor_noise_level)
        self.alpha = alpha
        self._sync_backbone_outputs = []
        self._sync_action_inputs = []
        self._env = None
        self._interval_z0_backbone = None

    def set_env(self, env):
        """Set env reference for state save/restore.

        Traverses wrapper chain: MultiStepWrapper → LiberoEnv → OffScreenRenderEnv (ControlEnv)
        Gymnasium wrappers use .env, LiberoEnv uses ._env internally.
        """
        self._env = env

        # Find the LiberoEnv (for single-step sync rollout)
        self._libero_env = None
        cursor = env
        while cursor is not None:
            if type(cursor).__name__ == 'LiberoEnv':
                self._libero_env = cursor
                break
            cursor = getattr(cursor, 'env', None)

        # Find the base ControlEnv/OffScreenRenderEnv with sim state access
        self._base_env = env
        while True:
            if hasattr(self._base_env, 'get_sim_state'):
                break
            if hasattr(self._base_env, 'env'):
                self._base_env = self._base_env.env
            elif hasattr(self._base_env, '_env'):
                self._base_env = self._base_env._env
            else:
                raise RuntimeError("Could not find base env with get_sim_state()")
        logger.info(f"LinearInterp: base_env={type(self._base_env).__name__}, "
                     f"libero_env={'found' if self._libero_env else 'not found'}")

    def reset(self, options=None):
        super().reset(options)
        self._sync_backbone_outputs = []
        self._sync_action_inputs = []
        self._interval_z0_backbone = None
        return {}

    def _run_sync_rollout(self, first_obs):
        """Run sync rollout from saved state to collect ground truth features.

        Steps through LiberoEnv directly (single-step) to avoid MultiStepWrapper
        state corruption and terminated-episode errors. For each VLM interval step:
        1. Run VLM backbone on current obs → collect features
        2. Get action from features
        3. Execute n_action_steps single env steps (matching MultiStepWrapper behavior)
        """
        sync_backbone_list = []
        sync_action_inputs_list = []
        # n_action_steps passed from caller (8 for LIBERO, 1 for SimplerEnv)

        obs = first_obs
        for i in range(self.vlm_update_freq):
            # Build observation matching MultiStepWrapper format (history_dim + batch_dim)
            # LiberoEnv gives: video (H,W,C), state (dim,)
            # MultiStepWrapper gives: video (1,H,W,C), state (1,dim,) [history=1]
            # After external batching: video (1,1,H,W,C), state (1,1,dim,)
            batched_obs = {}
            for k, v in obs.items():
                if isinstance(v, np.ndarray):
                    # Add history dim + batch dim
                    batched_obs[k] = v[np.newaxis, np.newaxis, ...]
                elif isinstance(v, str):
                    batched_obs[k] = (v,)
                elif isinstance(v, list):
                    batched_obs[k] = np.array(v)[np.newaxis, np.newaxis, ...]
                else:
                    batched_obs[k] = v

            backbone_inputs, action_inputs, states = self._prepare_inputs(batched_obs)

            with torch.inference_mode():
                backbone_outputs = self.model.backbone(backbone_inputs)
                sync_backbone_list.append(backbone_outputs)
                sync_action_inputs_list.append(action_inputs)

                action_outputs = self.model.action_head.get_action(
                    backbone_outputs, action_inputs
                )

            # Decode action → (1, n_action_steps, action_dim) per key
            normalized_action = action_outputs.action_pred.float()
            batched_states = {}
            for k in self.modality_configs["state"].modality_keys:
                batched_states[k] = np.stack([s[k] for s in states], axis=0)
            unnormalized_action = self.processor.decode_action(
                normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
            )
            casted_action = {
                f"action.{key}": value.astype(np.float32)
                for key, value in unnormalized_action.items()
            }
            # unbatch: (1, n_action_steps, dim) → (n_action_steps, dim)
            unbatched_action = {k: v[0] for k, v in casted_action.items()}

            # Step env n_action_steps times through LiberoEnv directly
            if i < self.vlm_update_freq - 1:
                episode_done = False
                for step_j in range(n_action_steps):
                    single_act = {k: v[step_j] for k, v in unbatched_action.items()}
                    try:
                        obs, _, terminated, truncated, _ = self._libero_env.step(single_act)
                    except ValueError:
                        # "executing action in terminated episode"
                        episode_done = True
                        break
                    if terminated or truncated:
                        episode_done = True
                        break
                if episode_done:
                    # Pad remaining with last features
                    for j in range(i + 1, self.vlm_update_freq):
                        sync_backbone_list.append(backbone_outputs)
                        sync_action_inputs_list.append(action_inputs)
                    break

        return sync_backbone_list, sync_action_inputs_list

    def _blend_features(self, z0_outputs, zk_outputs, alpha):
        """Blend backbone features: alpha*z0 + (1-alpha)*zk."""
        from transformers.feature_extraction_utils import BatchFeature
        blended_data = {}
        for key in z0_outputs:
            if isinstance(z0_outputs[key], torch.Tensor) and z0_outputs[key].is_floating_point():
                blended_data[key] = alpha * z0_outputs[key] + (1 - alpha) * zk_outputs[key]
            else:
                blended_data[key] = zk_outputs[key]
        return BatchFeature(data=blended_data)

    def get_action(self, observation, options=None):
        step_in_interval = self._step_count % self.vlm_update_freq
        is_refresh = (self._cached_backbone_outputs is None or step_in_interval == 0)

        if is_refresh:
            assert self._env is not None, "Must call set_env() before evaluation"

            # 1. Save current env state + robosuite internal state
            saved_sim_state = self._base_env.get_sim_state()
            # Save robosuite timestep/done from all envs in the chain
            saved_robosuite_state = {}
            env_cursor = self._base_env
            idx = 0
            while env_cursor is not None:
                state = {}
                if hasattr(env_cursor, 'done'):
                    state['done'] = env_cursor.done
                if hasattr(env_cursor, 'timestep'):
                    state['timestep'] = env_cursor.timestep
                if state:
                    saved_robosuite_state[idx] = state
                idx += 1
                env_cursor = getattr(env_cursor, 'env', None)

            # 2. Run sync rollout to collect ground truth features
            # Convert observation to LiberoEnv format (strip batch + history dims)
            # Normal obs: batch(1) x history(1) x data → strip 2 leading dims
            raw_obs = {}
            for k, v in observation.items():
                if isinstance(v, np.ndarray) and v.ndim >= 2:
                    raw_obs[k] = v[0, 0]  # Strip batch + history dims
                elif isinstance(v, (tuple, list)):
                    raw_obs[k] = v[0]
                else:
                    raw_obs[k] = v

            self._sync_backbone_outputs, self._sync_action_inputs = \
                self._run_sync_rollout(raw_obs)

            # 3. Restore env state + robosuite internal state
            self._base_env.set_state(saved_sim_state)
            self._base_env.sim.forward()
            if hasattr(self._base_env, '_update_observables'):
                self._base_env._update_observables(force=True)
            # Restore robosuite timestep/done
            env_cursor = self._base_env
            idx = 0
            while env_cursor is not None:
                if idx in saved_robosuite_state:
                    for attr, val in saved_robosuite_state[idx].items():
                        setattr(env_cursor, attr, val)
                idx += 1
                env_cursor = getattr(env_cursor, 'env', None)

            # 4. Use z_0 (first sync feature) for this step
            backbone_outputs = self._sync_backbone_outputs[0]
            self._interval_z0_backbone = backbone_outputs
            self._cached_backbone_outputs = backbone_outputs

            backbone_inputs, action_inputs, states = self._prepare_inputs(observation)
            self._cached_action_inputs = action_inputs
            self.s2_times.append(0)  # Placeholder

        else:
            # Use interpolated features
            backbone_inputs, action_inputs, states = self._prepare_inputs(observation)
            z0 = self._interval_z0_backbone
            zk = self._sync_backbone_outputs[step_in_interval]
            backbone_outputs = self._blend_features(z0, zk, self.alpha)
            self._cached_action_inputs = action_inputs

        with torch.inference_mode():
            if self.use_anchor and self._anchor is not None:
                action_outputs = self.model.action_head.get_action(
                    backbone_outputs, action_inputs,
                    anchor=self._anchor,
                    anchor_noise_level=self.anchor_noise_level,
                )
            else:
                action_outputs = self.model.action_head.get_action(
                    backbone_outputs, action_inputs
                )

        if self.use_anchor:
            self._anchor = action_outputs.action_pred.clone()

        self._step_count += 1

        normalized_action = action_outputs.action_pred.float()
        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )
        casted_action = {
            f"action.{key}": value.astype(np.float32)
            for key, value in unnormalized_action.items()
        }
        return casted_action, {}


def evaluate_task(policy_wrapper, env, task_name, n_episodes, max_steps, init_states=None, n_action_steps=8):
    """Evaluate a single task."""
    successes = []
    latencies = []
    vlm_calls_list = []
    phase_stats_list = []

    ep_step_data = []  # Per-episode step-level data for failure analysis

    for ep in range(n_episodes):
        obs, _ = env.reset()
        # Set deterministic init state if available
        if init_states is not None and ep < len(init_states):
            env.unwrapped._env.set_init_state(init_states[ep])
            obs, _ = env.reset()
        policy_wrapper.reset()

        done = False
        step = 0
        ep_latencies = []
        ep_action_mags = []
        ep_rewards = []
        ep_feat_norms = []

        while not done and step < max_steps:
            start_time = time.time()

            batched_obs = {}
            for k, v in obs.items():
                if isinstance(v, np.ndarray):
                    batched_obs[k] = v[np.newaxis, ...]
                elif isinstance(v, str):
                    batched_obs[k] = (v,)
                else:
                    batched_obs[k] = v

            action, _ = policy_wrapper.get_action(batched_obs)
            unbatched_action = {k: v[0] for k, v in action.items()}

            # Track action magnitude (pos+rot, exclude gripper)
            act_arr = None
            for k, v in unbatched_action.items():
                if "action" in k and isinstance(v, np.ndarray) and v.ndim >= 1:
                    act_arr = v.flatten()[:6]
                    break
            act_mag = float(np.linalg.norm(act_arr)) if act_arr is not None else 0.0
            ep_action_mags.append(act_mag)

            # Record feature norm from policy internal state
            feat_norm = 0.0
            if hasattr(policy_wrapper, '_current_features') and policy_wrapper._current_features is not None:
                feat_norm = float(policy_wrapper._current_features.float().norm(dim=-1).mean())
            elif hasattr(policy_wrapper, '_cached_backbone_outputs') and policy_wrapper._cached_backbone_outputs is not None:
                feat_norm = float(policy_wrapper._cached_backbone_outputs["backbone_features"].float().norm(dim=-1).mean())
            ep_feat_norms.append(feat_norm)

            latency = (time.time() - start_time) * 1000
            ep_latencies.append(latency)

            obs, reward, terminated, truncated, info = env.step(unbatched_action)
            ep_rewards.append(float(reward) if reward is not None else 0.0)
            done = terminated or truncated
            step += 1

        success = False
        if "success" in info:
            if isinstance(info["success"], (list, np.ndarray)):
                success = any(info["success"])
            else:
                success = bool(info["success"])

        successes.append(success)
        latencies.extend(ep_latencies)
        vlm_calls_list.append(len(policy_wrapper.s2_times))
        ep_step_data.append({
            "success": success,
            "n_steps": step,
            "action_mags": ep_action_mags,
            "feat_norms": ep_feat_norms,
        })

        # Collect phase statistics if available
        if hasattr(policy_wrapper, 'get_phase_statistics'):
            ep_phase_stats = policy_wrapper.get_phase_statistics()
            if ep_phase_stats:
                phase_stats_list.append(ep_phase_stats)
                logger.info(f"  Ep {ep}: phase={ep_phase_stats['phase_distribution']}, "
                           f"eff_freq={ep_phase_stats['effective_vlm_freq']:.2f}, "
                           f"VLM savings={ep_phase_stats['vlm_savings_vs_sync']:.1f}%, "
                           f"transitions={ep_phase_stats['phase_transitions']}")

    # Collect transition-triggered statistics if available
    if hasattr(policy_wrapper, 'get_transition_statistics'):
        tt_stats = policy_wrapper.get_transition_statistics()
        if tt_stats:
            logger.info(f"  Transition-triggered stats: scheduled={tt_stats['scheduled_vlm_calls']}, "
                       f"transition={tt_stats['transition_vlm_calls']}, "
                       f"eff_freq={tt_stats['effective_vlm_freq']:.2f}, "
                       f"VLM savings={tt_stats['vlm_savings_vs_sync']:.1f}%")

    # Collect bridge step diagnostics if available
    if hasattr(policy_wrapper, 'get_step_diagnostics'):
        all_diags = policy_wrapper.get_step_diagnostics()
        if all_diags:
            from collections import defaultdict
            by_offset = defaultdict(list)
            for ep_diags in all_diags:
                for d in ep_diags:
                    if d["type"] == "bridge":
                        by_offset[d["steps_since_vlm"]].append(d)
            if by_offset:
                logger.info(f"  Bridge diagnostics for {task_name.split('/')[-1][:40]}:")
                for offset in sorted(by_offset.keys()):
                    entries = by_offset[offset]
                    avg_feat = np.mean([e["feat_cos"] for e in entries])
                    avg_copy = np.mean([e["copy_cos"] for e in entries if e["copy_cos"] >= 0])
                    avg_dcosine = np.mean([e["delta_cos"] for e in entries if e["delta_cos"] >= 0])
                    avg_dratio = np.mean([e["delta_mag_ratio"] for e in entries if e["delta_mag_ratio"] >= 0])
                    logger.info(f"    offset={offset}: feat_cos={avg_feat:.4f} copy_cos={avg_copy:.4f} "
                                f"delta_cos={avg_dcosine:.4f} Δ_ratio={avg_dratio:.3f} (n={len(entries)})")

            # diagnose_all: show quality by episode phase (early/mid/late)
            all_bridge = []
            for ep_diags in all_diags:
                for d in ep_diags:
                    if d["type"] == "bridge":
                        all_bridge.append(d)
            if all_bridge and all_bridge[0].get("step", -1) >= 0:
                max_step = max(d["step"] for d in all_bridge)
                if max_step > 6:  # Only show phase breakdown for full-episode diagnostics
                    third = max(1, (max_step + 1) // 3)
                    phases = [
                        ("early (0-{})".format(third - 1), [d for d in all_bridge if d["step"] < third]),
                        ("mid ({}-{})".format(third, 2 * third - 1), [d for d in all_bridge if third <= d["step"] < 2 * third]),
                        ("late ({}-{})".format(2 * third, max_step), [d for d in all_bridge if d["step"] >= 2 * third]),
                    ]
                    logger.info(f"  Bridge quality by episode phase:")
                    for phase_name, entries in phases:
                        if entries:
                            f_cos = np.mean([e["feat_cos"] for e in entries])
                            c_cos = np.mean([e["copy_cos"] for e in entries if e["copy_cos"] >= 0])
                            d_cos = np.mean([e["delta_cos"] for e in entries if e["delta_cos"] >= 0])
                            # Split by offset within each phase
                            off1 = [e for e in entries if e["steps_since_vlm"] == 1]
                            off2 = [e for e in entries if e["steps_since_vlm"] == 2]
                            o1_str = "off1={:.4f}".format(np.mean([e["feat_cos"] for e in off1])) if off1 else "off1=N/A"
                            o2_str = "off2={:.4f}".format(np.mean([e["feat_cos"] for e in off2])) if off2 else "off2=N/A"
                            logger.info(f"    {phase_name}: feat_cos={f_cos:.4f} delta_cos={d_cos:.4f} "
                                        f"{o1_str} {o2_str} (n={len(entries)})")

    # Failure analysis: action magnitude profile for success vs failure episodes
    if ep_step_data:
        succ_eps = [e for e in ep_step_data if e["success"]]
        fail_eps = [e for e in ep_step_data if not e["success"]]
        if fail_eps:
            fail_lengths = [e["n_steps"] for e in fail_eps]
            succ_lengths = [e["n_steps"] for e in succ_eps] if succ_eps else []
            logger.info(f"  Failure analysis for {task_name.split('/')[-1][:40]}:")
            logger.info(f"    Success eps: {len(succ_eps)}, avg_len={np.mean(succ_lengths):.0f}" if succ_eps else f"    Success eps: 0")
            logger.info(f"    Failed eps:  {len(fail_eps)}, avg_len={np.mean(fail_lengths):.0f}")
            # Bin action magnitudes into phases and show where failures diverge
            n_bins = 5
            max_len = max(e["n_steps"] for e in ep_step_data)
            bin_size = max(1, max_len // n_bins)
            for b in range(n_bins):
                s, e_idx = b * bin_size, min((b + 1) * bin_size, max_len)
                pct = f"steps {s:2d}-{e_idx:2d}"
                succ_mags = [np.mean(ep["action_mags"][s:e_idx]) for ep in succ_eps if len(ep["action_mags"]) > s]
                fail_mags = [np.mean(ep["action_mags"][s:e_idx]) for ep in fail_eps if len(ep["action_mags"]) > s]
                succ_str = f"succ_mag={np.mean(succ_mags):.3f}" if succ_mags else "succ_mag=N/A"
                fail_str = f"fail_mag={np.mean(fail_mags):.3f}" if fail_mags else "fail_mag=N/A"
                still_running = sum(1 for ep in fail_eps if len(ep["action_mags"]) > s)
                logger.info(f"    {pct}: {succ_str}  {fail_str}  (fail_eps_active={still_running})")

    # Feature norm summary
    all_feat_norms = [fn for ep in ep_step_data for fn in ep["feat_norms"] if fn > 0]
    if all_feat_norms:
        logger.info(f"  Feature norms: mean={np.mean(all_feat_norms):.2f} std={np.std(all_feat_norms):.2f}")
        if ep_step_data[0]["feat_norms"]:
            succ_norms = [fn for ep in ep_step_data if ep["success"] for fn in ep["feat_norms"] if fn > 0]
            fail_norms = [fn for ep in ep_step_data if not ep["success"] for fn in ep["feat_norms"] if fn > 0]
            if succ_norms:
                logger.info(f"    Success eps: feat_norm={np.mean(succ_norms):.2f}")
            if fail_norms:
                logger.info(f"    Failure eps: feat_norm={np.mean(fail_norms):.2f}")

    # Print component latency
    if hasattr(policy_wrapper, 's2_times') and policy_wrapper.s2_times:
        s2 = policy_wrapper.s2_times[2:]  # skip warmup
        s1 = policy_wrapper.s1_times[2:] if hasattr(policy_wrapper, 's1_times') else []
        if s2:
            logger.info(f"  Backbone (s2) latency: median={np.median(s2):.1f}ms mean={np.mean(s2):.1f}ms (n={len(s2)})")
        if s1:
            logger.info(f"  Action head (s1) latency: median={np.median(s1):.1f}ms mean={np.mean(s1):.1f}ms (n={len(s1)})")
        if s2 and s1:
            logger.info(f"  Total per-step: {np.median(s2)+np.median(s1):.1f}ms")

    result = {
        "success_rate": np.mean(successes),
        "avg_latency_ms": np.mean(latencies),
        "avg_vlm_calls": np.mean(vlm_calls_list),
        "n_episodes": n_episodes,
    }

    # Aggregate phase statistics across episodes
    if phase_stats_list:
        avg_phase_dist = {}
        for phase in ["navigation", "transition", "manipulation"]:
            avg_phase_dist[phase] = np.mean([s["phase_distribution"].get(phase, 0) for s in phase_stats_list])
        result["phase_distribution"] = avg_phase_dist
        result["effective_vlm_freq"] = np.mean([s["effective_vlm_freq"] for s in phase_stats_list])
        result["vlm_savings_vs_sync"] = np.mean([s["vlm_savings_vs_sync"] for s in phase_stats_list])
        result["avg_phase_transitions"] = np.mean([s["phase_transitions"] for s in phase_stats_list])

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stable-Dynamic Flow Bridge")
    parser.add_argument("--model_path", type=str,
                        default="./outputs/groot_model")
    parser.add_argument("--bridge_path", type=str,
                        default="./outputs/single_step_bridge/best_model_dit.pt")
    parser.add_argument("--output_dir", type=str,
                        default="./outputs/eval_results")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=720)
    parser.add_argument("--vlm_freq", type=int, default=3, help="VLM update frequency for async/bridge")
    parser.add_argument("--modes", type=str, nargs="+", default=["sync", "async", "bridge"],
                        help="Modes: sync, async, bridge, autoregressive_bridge, phase_aware_async, phase_aware_bridge, linear_interp")
    parser.add_argument("--ar_bridge_path", type=str,
                        default="./outputs/single_step_bridge_v2/best_model_dit.pt",
                        help="Path to SingleStepDiT checkpoint for autoregressive_bridge mode")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Fixed alpha for linear_interp: features = alpha*z_0 + (1-alpha)*z_k_sync")
    parser.add_argument("--use_anchor", action="store_true", default=False)
    parser.add_argument("--num_ode_steps", type=int, default=1,
                        help="Number of ODE steps for bridge (1=one-step, >1=multi-step Euler)")
    # Offline evaluation mode
    parser.add_argument("--offline_eval", action="store_true",
                        help="Run offline evaluation on dataset (measure velocity cosine)")
    parser.add_argument("--data_path", type=str,
                        default="./outputs/latent_bridge_data/multilayer_train_data.h5",
                        help="Path to HDF5 data for offline evaluation")
    parser.add_argument("--offline_steps", type=int, nargs="+", default=[1, 5, 10],
                        help="ODE steps to evaluate in offline mode")
    parser.add_argument("--offline_max_samples", type=int, default=500,
                        help="Max samples for offline evaluation")
    parser.add_argument("--task_filter", type=str, default=None,
                        help="Substring filter for task names (e.g., 'KITCHEN_SCENE3' to run one task)")
    parser.add_argument("--action_head_path", type=str, default=None,
                        help="Path to fine-tuned action head checkpoint (Stage 2)")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA adapter directory for action head")
    # Phase-aware bridge arguments
    parser.add_argument("--nav_vlm_freq", type=int, default=5,
                        help="VLM freq during navigation phase (high action mag)")
    parser.add_argument("--trans_vlm_freq", type=int, default=3,
                        help="VLM freq during transition phase (medium action mag)")
    parser.add_argument("--manip_vlm_freq", type=int, default=1,
                        help="VLM freq during manipulation phase (low action mag, sync)")
    parser.add_argument("--nav_threshold", type=float, default=0.55,
                        help="Action magnitude above this = navigation phase")
    parser.add_argument("--manip_threshold", type=float, default=0.35,
                        help="Action magnitude below this = manipulation phase")
    parser.add_argument("--hysteresis_margin", type=float, default=0.03,
                        help="Hysteresis margin to prevent phase flickering")
    parser.add_argument("--smoothing_window", type=int, default=3,
                        help="Moving average window for action magnitude smoothing")
    parser.add_argument("--force_vlm_on_phase_change", action="store_true", default=True,
                        help="Force VLM refresh when entering manipulation phase")
    parser.add_argument("--no_force_vlm_on_phase_change", dest="force_vlm_on_phase_change",
                        action="store_false",
                        help="Disable forced VLM refresh on phase change")
    parser.add_argument("--diagnose_steps", type=int, default=0,
                        help="Run VLM ground-truth comparison on first N steps per episode (0=disabled)")
    parser.add_argument("--diagnose_all", action="store_true",
                        help="Run VLM ground-truth at every bridge step (full episode). Slower but shows error propagation.")
    parser.add_argument("--num_inference_timesteps", type=int, default=1,
                        help="Number of ODE denoising steps for action head (default=1, GR00T default=4)")
    # Transition-triggered bridge arguments
    parser.add_argument("--transition_threshold", type=float, default=0.2,
                        help="Action magnitude threshold for transition detection (crossing triggers VLM)")
    parser.add_argument("--transition_smoothing", type=int, default=2,
                        help="Smoothing window for transition detection")
    parser.add_argument("--task_suite", type=str, default="libero_10",
                        choices=["libero_10", "libero_spatial", "libero_object", "libero_goal", "libero_90"],
                        help="LIBERO task suite to evaluate on")
    parser.add_argument("--env_names", type=str, nargs="+", default=None,
                        help="Direct env names (bypass suite lookup). E.g., robocasa_panda_omron/OpenDrawer_PandaOmron_Env")
    parser.add_argument("--embodiment_tag", type=str, default="LIBERO_PANDA",
                        help="Embodiment tag (LIBERO_PANDA, ROBOCASA_PANDA_OMRON)")
    parser.add_argument("--n_action_steps", type=int, default=8,
                        help="Number of action steps per inference (1 for SimplerEnv, 8 for LIBERO)")
    parser.add_argument("--use_init_states", action="store_true",
                        help="Use fixed init states from LIBERO benchmark for deterministic eval")
    args = parser.parse_args()

    # Import bridge model (only if needed)
    if "bridge" in args.modes or args.offline_eval:
        from qcvla.model.rectified_flow_bridge import DiTStableDynamicFlowModel, RectifiedFlowBridge

    # Import SingleStepDiT for autoregressive_bridge, phase_aware_bridge, or transition_triggered_bridge mode
    if any(m in args.modes for m in ["autoregressive_bridge", "phase_aware_bridge", "transition_triggered_bridge"]):
        sys.path.insert(0, str(Path(__file__).parent))
        from train_single_step_dit import SingleStepDiT

    # Imports for simulation eval (only if not offline)
    if not args.offline_eval:
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        from gr00t.eval.rollout_policy import WrapperConfigs, VideoConfig, MultiStepConfig, create_eval_env
        from gr00t.data.embodiment_tags import EmbodimentTag
        if "ROBOCASA" not in args.embodiment_tag.upper() and "OXE" not in args.embodiment_tag.upper():
            from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs

    # Setup
    torch.cuda.set_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    if not args.offline_eval:
        if "ROBOCASA" in args.embodiment_tag.upper():
            import robocasa  # noqa
        elif "OXE" in args.embodiment_tag.upper():
            from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs
            register_simpler_envs()
        else:
            register_libero_envs()

    # Get task list
    if args.env_names:
        task_list = args.env_names
        logger.info(f"Direct env names: {len(task_list)} tasks")
    else:
        from libero.libero import benchmark as libero_benchmark
        benchmark_dict = libero_benchmark.get_benchmark_dict()
        suite = benchmark_dict[args.task_suite]()
        task_list = [f"libero_sim/{suite.get_task(i).name}" for i in range(suite.get_num_tasks())]
        logger.info(f"Task suite: {args.task_suite} ({len(task_list)} tasks)")

    if args.task_filter:
        task_list = [t for t in task_list if args.task_filter in t]
        logger.info(f"Task filter '{args.task_filter}': {len(task_list)} tasks")

    # Load base policy (skip if only doing offline eval)
    base_policy = None
    if not args.offline_eval:
        logger.info(f"Loading GR00T model from {args.model_path}...")
        base_policy = Gr00tPolicy(
            embodiment_tag=getattr(EmbodimentTag, args.embodiment_tag),
            model_path=args.model_path,
            device=args.device,
            strict=False,
        )
        base_policy.model.action_head.num_inference_timesteps = args.num_inference_timesteps
        logger.info(f"Model loaded with {args.num_inference_timesteps}-step inference.")

        # Load fine-tuned action head (Stage 2) if provided
        if args.action_head_path and os.path.exists(args.action_head_path):
            logger.info(f"Loading fine-tuned action head from {args.action_head_path}...")
            ah_ckpt = torch.load(args.action_head_path, map_location=args.device, weights_only=False)
            # Load only the action head state dict
            ah_state = ah_ckpt.get("action_head_state_dict", ah_ckpt.get("model_state_dict", {}))
            base_policy.model.action_head.load_state_dict(ah_state, strict=False)
            # Keep action head in float32 to match training dtype
            base_policy.model.action_head.float()
            logger.info(f"  Fine-tuned action head loaded (step {ah_ckpt.get('step', '?')}, "
                         f"val_loss {ah_ckpt.get('val_loss', '?')}) — converted to float32")

        # Load LoRA adapter for action head if provided
        if args.lora_path and os.path.exists(args.lora_path):
            from peft import PeftModel
            logger.info(f"Loading LoRA adapter from {args.lora_path}...")
            base_policy.model.action_head.model = PeftModel.from_pretrained(
                base_policy.model.action_head.model, args.lora_path
            )
            # Load VLLN if saved separately
            vlln_path = os.path.join(args.lora_path, "vlln_state_dict.pt")
            if os.path.exists(vlln_path):
                vlln_state = torch.load(vlln_path, map_location=args.device, weights_only=False)
                base_policy.model.action_head.vlln.load_state_dict(
                    vlln_state["vlln_state_dict"]
                )
                logger.info(f"  VLLN loaded (step {vlln_state.get('step', '?')}, "
                            f"sync_val_loss {vlln_state.get('sync_val_loss', '?')})")
            base_policy.model.action_head.eval()
            logger.info(f"  LoRA adapter loaded from {args.lora_path}")

    # Load bridge model
    bridge_model = None
    flow = None
    bridge_state_dim = 8
    bridge_action_dim = 7
    if "bridge" in args.modes or args.offline_eval:
        logger.info(f"Loading Stable-Dynamic Flow Bridge from {args.bridge_path}...")
        checkpoint = torch.load(args.bridge_path, map_location=args.device, weights_only=False)
        config = checkpoint.get("config", {})

        # Channel importance is only used for training loss weighting, not needed for inference
        # Use actual dims from training config
        bridge_model = DiTStableDynamicFlowModel(
            feature_dim=config.get("feature_dim", 2048),
            seq_len=204,
            hidden_dim=config.get("hidden_dim", 512),
            num_heads=config.get("num_heads", 8),
            num_blocks=config.get("num_blocks", 4),
            state_dim=config.get("state_dim", 8),
            action_dim=config.get("action_dim", 7),
            use_channel_importance=False,  # Not needed for inference
        ).to(args.device)

        # Load with strict=False to ignore channel_importance weights
        bridge_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        bridge_model.eval()
        flow = RectifiedFlowBridge()
        bridge_state_dim = config.get("state_dim", 8)
        bridge_action_dim = config.get("action_dim", 7)
        logger.info(f"Bridge loaded. Val Cosine: {checkpoint.get('val_cosine', 'N/A'):.4f}")
        logger.info(f"  state_dim={bridge_state_dim}, action_dim={bridge_action_dim}")
        logger.info(f"  num_ode_steps={args.num_ode_steps} ({'one-step' if args.num_ode_steps == 1 else 'multi-step Euler'})")

    # Load autoregressive bridge model (SingleStepDiT)
    ar_bridge_model = None
    ar_bridge_state_dim = 8
    ar_bridge_action_dim = 7
    if any(m in args.modes for m in ["autoregressive_bridge", "phase_aware_bridge", "transition_triggered_bridge"]):
        logger.info(f"Loading SingleStepDiT from {args.ar_bridge_path}...")
        ar_checkpoint = torch.load(args.ar_bridge_path, map_location=args.device, weights_only=False)
        ar_config = ar_checkpoint.get("config", {})

        ar_bridge_model = SingleStepDiT(
            feature_dim=ar_config.get("feature_dim", 2048),
            seq_len=ar_config.get("seq_len", 204),
            hidden_dim=ar_config.get("hidden_dim", 768),
            num_heads=ar_config.get("num_heads", 12),
            num_blocks=ar_config.get("num_blocks", 12),
            state_dim=ar_config.get("state_dim", 8),
            action_dim=ar_config.get("action_dim", 7),
            low_rank=ar_config.get("low_rank", 0),
            num_image_tokens=ar_config.get("num_image_tokens", 162),
        ).to(args.device)

        ar_bridge_model.load_state_dict(ar_checkpoint["model_state_dict"], strict=False)
        ar_bridge_model.eval()
        ar_bridge_state_dim = ar_config.get("state_dim", 8)
        ar_bridge_action_dim = ar_config.get("action_dim", 7)
        ar_bridge_image_only = ar_config.get("image_only", False)
        ar_bridge_seq_len = ar_config.get("seq_len", 204)
        logger.info(f"SingleStepDiT loaded. Val Cosine: {ar_checkpoint.get('val_cosine', 'N/A')}, "
                     f"seq_len={ar_bridge_seq_len}, image_only={ar_bridge_image_only}")

    # Offline evaluation mode
    if args.offline_eval:
        if bridge_model is None:
            logger.error("Bridge model required for offline evaluation. Add 'bridge' to --modes or load separately.")
            return

        logger.info("\n" + "="*60)
        logger.info("OFFLINE EVALUATION MODE")
        logger.info("="*60)
        logger.info(f"Data: {args.data_path}")
        logger.info(f"ODE steps to test: {args.offline_steps}")
        logger.info(f"Max samples: {args.offline_max_samples}")

        offline_results = offline_eval_ode(
            model=bridge_model,
            data_path=args.data_path,
            device=args.device,
            num_steps_list=args.offline_steps,
            max_samples=args.offline_max_samples,
            batch_size=32,
        )

        # Save offline results
        offline_output = os.path.join(args.output_dir, "offline_eval_results.json")
        with open(offline_output, "w") as f:
            json.dump(offline_results, f, indent=2)
        logger.info(f"\nOffline results saved to: {offline_output}")

        # Summary
        print("\n" + "="*60)
        print("OFFLINE EVALUATION SUMMARY")
        print("="*60)
        print(f"Copy baseline: {offline_results['copy_baseline']:.4f}")
        for num_steps in args.offline_steps:
            key = f"steps_{num_steps}"
            if key in offline_results:
                r = offline_results[key]
                print(f"\n{num_steps}-step ODE:")
                print(f"  Feature Cosine: {r['feature_cosine']:.4f} ± {r['feature_cosine_std']:.4f}")
                print(f"  Velocity Cosine: {r['velocity_cosine']:.4f} ± {r['velocity_cosine_std']:.4f}")
                print(f"  Improvement over copy: {r['improvement']:+.4f}")

        return  # Exit after offline eval

    # Wrapper configs
    wrapper_configs = WrapperConfigs(
        video=VideoConfig(video_dir=None),
        multistep=MultiStepConfig(n_action_steps=args.n_action_steps, max_episode_steps=args.max_steps, terminate_on_success=True),
    )

    all_results = {}

    # Evaluate each mode
    for mode in args.modes:
        if mode == "sync":
            vlm_freq = 1
        else:
            vlm_freq = args.vlm_freq

        if mode == "sync":
            mode_name = "sync"
        elif mode == "bridge":
            mode_name = f"bridge-{vlm_freq}-{args.num_ode_steps}step"
        elif mode == "autoregressive_bridge":
            mode_name = f"ar_bridge-{vlm_freq}"
        elif mode == "phase_aware_async":
            mode_name = f"pa_async-n{args.nav_vlm_freq}-t{args.trans_vlm_freq}-m{args.manip_vlm_freq}"
        elif mode == "phase_aware_bridge":
            mode_name = f"pa_bridge-n{args.nav_vlm_freq}-t{args.trans_vlm_freq}-m{args.manip_vlm_freq}"
        elif mode == "transition_triggered_bridge":
            mode_name = f"tt_bridge-{vlm_freq}-thr{args.transition_threshold}"
        elif mode == "linear_interp":
            mode_name = f"linear_interp-{vlm_freq}-alpha{args.alpha}"
        else:
            mode_name = f"{mode}-{vlm_freq}"
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {mode_name}")
        logger.info(f"{'='*60}")

        mode_results = []

        # Load fixed init states if requested
        task_init_states = {}
        if args.use_init_states:
            import torch as _torch
            import glob as _glob
            init_dir = os.path.join(os.path.dirname(__file__), '..', 'benchmarks', 'Isaac-GR00T',
                                     'external_dependencies', 'LIBERO', 'libero', 'libero', 'init_files',
                                     args.task_suite)
            for init_file in _glob.glob(os.path.join(init_dir, '*.pruned_init')):
                task_short = os.path.basename(init_file).replace('.pruned_init', '')
                task_init_states[task_short] = _torch.load(init_file, weights_only=False)
            logger.info(f"Loaded init states for {len(task_init_states)} tasks")

        for task_name in tqdm(task_list, desc=mode_name):
            env = create_eval_env(task_name, env_idx=0, total_n_envs=1, wrapper_configs=wrapper_configs)
            # Get init states for this task
            task_short = task_name.split('/')[-1]
            init_states = task_init_states.get(task_short, None)

            if mode == "linear_interp":
                policy_wrapper = LinearInterpGr00tPolicy(
                    base_policy, vlm_update_freq=vlm_freq,
                    use_anchor=args.use_anchor,
                    alpha=args.alpha,
                )
                # Give env reference for state save/restore
                policy_wrapper.set_env(env)
            elif mode == "phase_aware_async":
                policy_wrapper = PhaseAwareAsyncGr00tPolicy(
                    base_policy, vlm_update_freq=vlm_freq,
                    use_anchor=args.use_anchor,
                    nav_vlm_freq=args.nav_vlm_freq,
                    trans_vlm_freq=args.trans_vlm_freq,
                    manip_vlm_freq=args.manip_vlm_freq,
                    nav_threshold=args.nav_threshold,
                    manip_threshold=args.manip_threshold,
                    hysteresis_margin=args.hysteresis_margin,
                    smoothing_window=args.smoothing_window,
                    force_vlm_on_phase_change=args.force_vlm_on_phase_change,
                )
            elif mode == "phase_aware_bridge" and ar_bridge_model is not None:
                policy_wrapper = PhaseAwareBridgeGr00tPolicy(
                    base_policy, ar_bridge_model,
                    vlm_update_freq=vlm_freq, use_anchor=args.use_anchor,
                    bridge_state_dim=ar_bridge_state_dim,
                    bridge_action_dim=ar_bridge_action_dim,
                    bridge_seq_len=ar_bridge_seq_len,
                    nav_vlm_freq=args.nav_vlm_freq,
                    trans_vlm_freq=args.trans_vlm_freq,
                    manip_vlm_freq=args.manip_vlm_freq,
                    nav_threshold=args.nav_threshold,
                    manip_threshold=args.manip_threshold,
                    hysteresis_margin=args.hysteresis_margin,
                    smoothing_window=args.smoothing_window,
                    force_vlm_on_phase_change=args.force_vlm_on_phase_change,
                    image_only=ar_bridge_image_only,
                )
            elif mode == "transition_triggered_bridge" and ar_bridge_model is not None:
                policy_wrapper = TransitionTriggeredBridgeGr00tPolicy(
                    base_policy, ar_bridge_model,
                    vlm_update_freq=vlm_freq, use_anchor=args.use_anchor,
                    bridge_state_dim=ar_bridge_state_dim,
                    bridge_action_dim=ar_bridge_action_dim,
                    transition_threshold=args.transition_threshold,
                    smoothing_window=args.transition_smoothing,
                )
            elif mode == "autoregressive_bridge" and ar_bridge_model is not None:
                policy_wrapper = AutoregressiveBridgeGr00tPolicy(
                    base_policy, ar_bridge_model,
                    vlm_update_freq=vlm_freq, use_anchor=args.use_anchor,
                    bridge_state_dim=ar_bridge_state_dim,
                    bridge_action_dim=ar_bridge_action_dim,
                    bridge_seq_len=ar_bridge_seq_len,
                    diagnose_steps=args.diagnose_steps,
                    diagnose_all=args.diagnose_all,
                    image_only=ar_bridge_image_only,
                )
            elif mode == "bridge" and bridge_model is not None:
                policy_wrapper = BridgeGr00tPolicy(
                    base_policy, bridge_model, flow,
                    vlm_update_freq=vlm_freq, use_anchor=args.use_anchor,
                    bridge_state_dim=bridge_state_dim,
                    bridge_action_dim=bridge_action_dim,
                    num_ode_steps=args.num_ode_steps,
                )
            else:
                policy_wrapper = AsyncGr00tPolicy(
                    base_policy, vlm_update_freq=vlm_freq, use_anchor=args.use_anchor
                )

            task_result = evaluate_task(policy_wrapper, env, task_name, args.n_episodes, args.max_steps,
                                       init_states=init_states, n_action_steps=args.n_action_steps)
            mode_results.append({
                "task": task_name.split("/")[-1],
                **task_result
            })

            logger.info(f"  {task_name.split('/')[-1][:40]}: {task_result['success_rate']*100:.1f}%")

            env.close()

        # Aggregate results
        agg = {
            "success_rate": np.mean([r["success_rate"] for r in mode_results]) * 100,
            "avg_latency_ms": np.mean([r["avg_latency_ms"] for r in mode_results]),
            "avg_vlm_calls": np.mean([r["avg_vlm_calls"] for r in mode_results]),
            "n_episodes": sum([r["n_episodes"] for r in mode_results]),
            "per_task": mode_results,
        }

        # Aggregate phase statistics if present
        phase_results = [r for r in mode_results if "phase_distribution" in r]
        if phase_results:
            avg_phase_dist = {}
            for phase in ["navigation", "transition", "manipulation"]:
                avg_phase_dist[phase] = np.mean([r["phase_distribution"].get(phase, 0) for r in phase_results])
            agg["phase_distribution"] = avg_phase_dist
            agg["effective_vlm_freq"] = np.mean([r["effective_vlm_freq"] for r in phase_results])
            agg["vlm_savings_vs_sync"] = np.mean([r["vlm_savings_vs_sync"] for r in phase_results])
            agg["avg_phase_transitions"] = np.mean([r["avg_phase_transitions"] for r in phase_results])

        all_results[mode_name] = agg

    # Print summary
    print("\n" + "="*60)
    print("STABLE-DYNAMIC FLOW BRIDGE EVALUATION RESULTS")
    print("="*60)

    for mode_name, data in sorted(all_results.items()):
        print(f"\n{mode_name.upper()}")
        print(f"  Success Rate: {data['success_rate']:.1f}%")
        print(f"  Avg Latency: {data['avg_latency_ms']:.1f} ms")
        print(f"  Avg VLM Calls: {data['avg_vlm_calls']:.1f}")

    # Key comparison
    print("\n" + "-"*60)
    print("KEY COMPARISON")
    print("-"*60)

    sync_key = "sync"
    async_key = f"async-{args.vlm_freq}"
    bridge_key = f"bridge-{args.vlm_freq}-{args.num_ode_steps}step"

    if sync_key in all_results:
        print(f"Sync (baseline):     {all_results[sync_key]['success_rate']:.1f}%")

    if async_key in all_results:
        async_sr = all_results[async_key]['success_rate']
        print(f"Async-{args.vlm_freq} (stale):    {async_sr:.1f}%")
        if sync_key in all_results:
            drop = all_results[sync_key]['success_rate'] - async_sr
            print(f"  -> Drop from sync: -{drop:.1f}%")

    if bridge_key in all_results:
        bridge_sr = all_results[bridge_key]['success_rate']
        print(f"Bridge-{args.vlm_freq} (ours):   {bridge_sr:.1f}%")
        if async_key in all_results:
            improvement = bridge_sr - all_results[async_key]['success_rate']
            print(f"  -> Improvement over async: +{improvement:.1f}%")
        if sync_key in all_results:
            recovery = (bridge_sr - all_results[async_key]['success_rate']) / (all_results[sync_key]['success_rate'] - all_results[async_key]['success_rate']) * 100
            print(f"  -> Recovery of sync performance: {recovery:.1f}%")

    ar_bridge_key = f"ar_bridge-{args.vlm_freq}"
    if ar_bridge_key in all_results:
        ar_sr = all_results[ar_bridge_key]['success_rate']
        print(f"AR Bridge-{args.vlm_freq} (DAgger): {ar_sr:.1f}%")
        if async_key in all_results:
            improvement = ar_sr - all_results[async_key]['success_rate']
            print(f"  -> Improvement over async: +{improvement:.1f}%")
        if sync_key in all_results and async_key in all_results:
            denom = all_results[sync_key]['success_rate'] - all_results[async_key]['success_rate']
            if denom > 0:
                recovery = (ar_sr - all_results[async_key]['success_rate']) / denom * 100
                print(f"  -> Recovery of sync performance: {recovery:.1f}%")

    pa_async_key = f"pa_async-n{args.nav_vlm_freq}-t{args.trans_vlm_freq}-m{args.manip_vlm_freq}"
    if pa_async_key in all_results:
        pa_async_data = all_results[pa_async_key]
        pa_async_sr = pa_async_data['success_rate']
        print(f"Phase-Aware Async (n={args.nav_vlm_freq}/t={args.trans_vlm_freq}/m={args.manip_vlm_freq}): {pa_async_sr:.1f}%")
        if sync_key in all_results:
            drop = all_results[sync_key]['success_rate'] - pa_async_sr
            print(f"  -> Drop from sync: -{drop:.1f}%")
        if async_key in all_results:
            improvement = pa_async_sr - all_results[async_key]['success_rate']
            print(f"  -> Improvement over fixed async-{args.vlm_freq}: +{improvement:.1f}%")
        if "phase_distribution" in pa_async_data:
            print(f"  Phase distribution:")
            for phase, pct in pa_async_data["phase_distribution"].items():
                print(f"    {phase}: {pct:.1f}%")
        if "effective_vlm_freq" in pa_async_data:
            print(f"  Effective VLM freq: {pa_async_data['effective_vlm_freq']:.2f}")
        if "vlm_savings_vs_sync" in pa_async_data:
            print(f"  VLM savings vs sync: {pa_async_data['vlm_savings_vs_sync']:.1f}%")
        if "avg_phase_transitions" in pa_async_data:
            print(f"  Avg phase transitions: {pa_async_data['avg_phase_transitions']:.1f}")

    pa_bridge_key = f"pa_bridge-n{args.nav_vlm_freq}-t{args.trans_vlm_freq}-m{args.manip_vlm_freq}"
    if pa_bridge_key in all_results:
        pa_data = all_results[pa_bridge_key]
        pa_sr = pa_data['success_rate']
        print(f"Phase-Aware Bridge (n={args.nav_vlm_freq}/t={args.trans_vlm_freq}/m={args.manip_vlm_freq}): {pa_sr:.1f}%")
        if sync_key in all_results:
            drop = all_results[sync_key]['success_rate'] - pa_sr
            print(f"  -> Drop from sync: -{drop:.1f}%")
        if ar_bridge_key in all_results:
            improvement = pa_sr - all_results[ar_bridge_key]['success_rate']
            print(f"  -> Improvement over fixed AR bridge: +{improvement:.1f}%")
        # Phase statistics
        if "phase_distribution" in pa_data:
            print(f"  Phase distribution:")
            for phase, pct in pa_data["phase_distribution"].items():
                print(f"    {phase}: {pct:.1f}%")
        if "effective_vlm_freq" in pa_data:
            print(f"  Effective VLM freq: {pa_data['effective_vlm_freq']:.2f}")
        if "vlm_savings_vs_sync" in pa_data:
            print(f"  VLM savings vs sync: {pa_data['vlm_savings_vs_sync']:.1f}%")
        if "avg_phase_transitions" in pa_data:
            print(f"  Avg phase transitions: {pa_data['avg_phase_transitions']:.1f}")

    interp_key = f"linear_interp-{args.vlm_freq}-alpha{args.alpha}"
    if interp_key in all_results:
        interp_sr = all_results[interp_key]['success_rate']
        print(f"Linear Interp-{args.vlm_freq} (alpha={args.alpha}): {interp_sr:.1f}%")
        if sync_key in all_results:
            drop = all_results[sync_key]['success_rate'] - interp_sr
            print(f"  -> Drop from sync: -{drop:.1f}%")
        if async_key in all_results:
            improvement = interp_sr - all_results[async_key]['success_rate']
            print(f"  -> Improvement over async: +{improvement:.1f}%")

    # Save results
    output_file = os.path.join(args.output_dir, f"eval_results_freq{args.vlm_freq}_steps{args.num_ode_steps}.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
