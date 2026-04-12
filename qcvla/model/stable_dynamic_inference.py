"""
Stable-Dynamic Inference with Layer 8 Caching and Drift Logging.

Updates stable context at VLM forward frequency and logs drift.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, List
from dataclasses import dataclass, field
import numpy as np


@dataclass
class DriftMetrics:
    """Track layer 8 drift across delays."""
    delays: List[int] = field(default_factory=lambda: [1, 3, 5])
    layer8_cosines: Dict[int, List[float]] = field(default_factory=dict)
    layer16_cosines: Dict[int, List[float]] = field(default_factory=dict)

    def __post_init__(self):
        for d in self.delays:
            self.layer8_cosines[d] = []
            self.layer16_cosines[d] = []

    def log_drift(self, delay: int, layer8_cos: float, layer16_cos: float):
        """Log cosine similarity for a specific delay."""
        if delay in self.layer8_cosines:
            self.layer8_cosines[delay].append(layer8_cos)
            self.layer16_cosines[delay].append(layer16_cos)

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        summary = {}
        for d in self.delays:
            if self.layer8_cosines[d]:
                summary[f'layer8_cos_delay{d}'] = {
                    'mean': np.mean(self.layer8_cosines[d]),
                    'std': np.std(self.layer8_cosines[d]),
                    'change': 1 - np.mean(self.layer8_cosines[d]),
                }
                summary[f'layer16_cos_delay{d}'] = {
                    'mean': np.mean(self.layer16_cosines[d]),
                    'std': np.std(self.layer16_cosines[d]),
                    'change': 1 - np.mean(self.layer16_cosines[d]),
                }
        return summary

    def print_summary(self):
        """Print drift summary."""
        print("\n" + "="*60)
        print("LAYER DRIFT ANALYSIS")
        print("="*60)
        print(f"{'Delay':<8} {'Layer 8 (stable)':<25} {'Layer 16 (dynamic)':<25}")
        print("-"*60)

        for d in self.delays:
            if self.layer8_cosines[d]:
                l8_mean = np.mean(self.layer8_cosines[d])
                l8_change = 1 - l8_mean
                l16_mean = np.mean(self.layer16_cosines[d])
                l16_change = 1 - l16_mean

                print(f"{d:<8} cos={l8_mean:.4f} (Δ={l8_change:.4f})   "
                      f"cos={l16_mean:.4f} (Δ={l16_change:.4f})")

        print("="*60)


class StableDynamicInference:
    """
    Inference wrapper for Stable-Dynamic Flow Bridge.

    Features:
    - Updates stable context (layer 8) at VLM forward frequency
    - Logs drift metrics for different delays
    - Supports async-style inference with latent bridge
    """

    def __init__(
        self,
        bridge_model: torch.nn.Module,
        vlm_forward_interval: int = 3,  # Run VLM every N steps
        device: str = "cuda",
        log_drift: bool = True,
    ):
        self.bridge = bridge_model
        self.vlm_interval = vlm_forward_interval
        self.device = device
        self.log_drift = log_drift

        # Cached features
        self.cached_layer8 = None   # Stable context
        self.cached_layer16 = None  # Last VLM output
        self.prev_layer8 = None     # For drift logging
        self.prev_layer16 = None

        # Tracking
        self.steps_since_vlm = 0
        self.total_steps = 0

        # Drift metrics
        self.drift_metrics = DriftMetrics() if log_drift else None

    def reset(self):
        """Reset for new episode."""
        self.cached_layer8 = None
        self.cached_layer16 = None
        self.prev_layer8 = None
        self.prev_layer16 = None
        self.steps_since_vlm = 0
        self.total_steps = 0

    def update_from_vlm(
        self,
        layer8_features: torch.Tensor,   # [1, seq, feat] from VLM
        layer16_features: torch.Tensor,  # [1, seq, feat] from VLM
    ):
        """
        Update cached features from VLM forward pass.
        Called at VLM forward frequency.
        """
        # Log drift if we have previous features
        if self.log_drift and self.cached_layer8 is not None:
            delay = self.steps_since_vlm

            # Compute cosine similarity
            layer8_cos = F.cosine_similarity(
                self.cached_layer8.flatten(),
                layer8_features.flatten(),
                dim=0
            ).item()

            layer16_cos = F.cosine_similarity(
                self.cached_layer16.flatten(),
                layer16_features.flatten(),
                dim=0
            ).item()

            self.drift_metrics.log_drift(delay, layer8_cos, layer16_cos)

        # Update cache
        self.prev_layer8 = self.cached_layer8
        self.prev_layer16 = self.cached_layer16
        self.cached_layer8 = layer8_features.clone()
        self.cached_layer16 = layer16_features.clone()
        self.steps_since_vlm = 0

    def predict_layer16(
        self,
        current_layer16: torch.Tensor,  # [1, seq, feat]
        state: torch.Tensor,             # [1, state_dim]
        action: torch.Tensor,            # [1, action_dim]
        horizon: int = 1,
    ) -> torch.Tensor:
        """
        Predict next layer 16 features using cached layer 8 as stable context.

        Returns: predicted layer 16 features [1, seq, feat]
        """
        self.bridge.eval()

        with torch.no_grad():
            # Use cached layer 8 as stable context
            # (updated at VLM frequency, but reused between updates)

            t = torch.zeros(1, device=self.device)  # t=0 for one-step generation
            horizon_tensor = torch.tensor([horizon], device=self.device)

            velocity = self.bridge(
                x=current_layer16,
                t=t * 999,
                state=state,
                action=action,
                horizon=horizon_tensor,
                stable_features=self.cached_layer8,  # Cached, updated at VLM freq
            )

            predicted = current_layer16 + velocity

        self.steps_since_vlm += 1
        self.total_steps += 1

        return predicted

    def should_run_vlm(self) -> bool:
        """Check if VLM should be run this step."""
        return self.steps_since_vlm >= self.vlm_interval or self.cached_layer8 is None

    def get_drift_summary(self) -> Dict:
        """Get drift metrics summary."""
        if self.drift_metrics:
            return self.drift_metrics.get_summary()
        return {}

    def print_drift_summary(self):
        """Print drift analysis."""
        if self.drift_metrics:
            self.drift_metrics.print_summary()


class StableDynamicPolicyWrapper:
    """
    Full policy wrapper that integrates VLM + Bridge.

    Manages:
    - When to run VLM (every N steps)
    - Layer extraction and caching
    - Bridge prediction
    - Drift logging
    """

    def __init__(
        self,
        policy,  # Gr00tPolicy
        bridge_model: torch.nn.Module,
        vlm_forward_interval: int = 3,
        layer_indices: List[int] = [1, 8, 15, 16],
        device: str = "cuda",
        log_drift: bool = True,
    ):
        self.policy = policy
        self.bridge = bridge_model
        self.vlm_interval = vlm_forward_interval
        self.layer_indices = layer_indices
        self.device = device

        # Layer index mapping
        self.layer_map = {
            'early': layer_indices[0],     # 1
            'stable': layer_indices[1],    # 8
            'pre_final': layer_indices[2], # 15
            'target': layer_indices[3],    # 16
        }

        # Inference wrapper
        self.inference = StableDynamicInference(
            bridge_model=bridge_model,
            vlm_forward_interval=vlm_forward_interval,
            device=device,
            log_drift=log_drift,
        )

        # Cache for layers
        self.cached_layers = {}

    def reset(self):
        """Reset for new episode."""
        self.inference.reset()
        self.cached_layers = {}

    def extract_layers(self, backbone_inputs: Dict) -> Dict[str, torch.Tensor]:
        """Extract specific layers from VLM."""
        backbone = self.policy.model.backbone
        backbone.set_frozen_modules_to_eval_mode()

        keys_to_use = ["input_ids", "attention_mask", "pixel_values"]
        vl_input = {k: backbone_inputs[k] for k in keys_to_use}

        with torch.no_grad():
            outputs = backbone.model(**vl_input, output_hidden_states=True)
            hidden_states = outputs["hidden_states"]

        layers = {
            'early': hidden_states[self.layer_map['early']],
            'stable': hidden_states[self.layer_map['stable']],
            'pre_final': hidden_states[self.layer_map['pre_final']],
            'target': hidden_states[self.layer_map['target']],
        }

        return layers

    def step(
        self,
        observation: Dict,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get action-relevant features for this step.

        Runs VLM if needed, otherwise uses bridge prediction.

        Returns: layer 16 features for action head
        """
        if self.inference.should_run_vlm():
            # Run VLM and update cache
            backbone_inputs = self.prepare_backbone_inputs(observation)
            layers = self.extract_layers(backbone_inputs)

            # Update inference cache
            self.inference.update_from_vlm(
                layer8_features=layers['stable'],
                layer16_features=layers['target'],
            )

            # Cache all layers
            self.cached_layers = layers

            return layers['target']
        else:
            # Use bridge to predict
            predicted_layer16 = self.inference.predict_layer16(
                current_layer16=self.cached_layers['target'],
                state=state,
                action=action,
            )

            # Update cached target
            self.cached_layers['target'] = predicted_layer16

            return predicted_layer16

    def prepare_backbone_inputs(self, observation: Dict) -> Dict:
        """Prepare inputs for backbone (implement based on policy)."""
        # This should be implemented based on the actual policy interface
        raise NotImplementedError("Override in subclass")

    def get_drift_analysis(self):
        """Get and print drift analysis."""
        self.inference.print_drift_summary()
        return self.inference.get_drift_summary()


# ============================================================================
# Utility: Analyze drift from collected data
# ============================================================================

def analyze_layer_drift_from_data(h5_path: str, delays: List[int] = [1, 3, 5]):
    """
    Analyze layer drift from collected multi-layer data.

    Computes cosine similarity for layer 8 and layer 16 at different delays.
    """
    import h5py

    drift_metrics = DriftMetrics(delays=delays)

    with h5py.File(h5_path, 'r') as f:
        episode_keys = sorted([k for k in f.keys() if k.startswith('episode_')])

        for ep_key in episode_keys:
            ep = f[ep_key]

            if 'multilayer_features' not in ep:
                continue

            features = np.array(ep['multilayer_features'])  # [T, num_layers, seq, feat]
            T = features.shape[0]

            # Assuming layer order: [1, 8, 15, 16] -> indices [0, 1, 2, 3]
            layer8_idx = 1
            layer16_idx = 3

            for delay in delays:
                for t in range(T - delay):
                    # Layer 8 drift
                    l8_t0 = features[t, layer8_idx].flatten()
                    l8_t1 = features[t + delay, layer8_idx].flatten()
                    l8_cos = np.dot(l8_t0, l8_t1) / (np.linalg.norm(l8_t0) * np.linalg.norm(l8_t1) + 1e-8)

                    # Layer 16 drift
                    l16_t0 = features[t, layer16_idx].flatten()
                    l16_t1 = features[t + delay, layer16_idx].flatten()
                    l16_cos = np.dot(l16_t0, l16_t1) / (np.linalg.norm(l16_t0) * np.linalg.norm(l16_t1) + 1e-8)

                    drift_metrics.log_drift(delay, float(l8_cos), float(l16_cos))

    drift_metrics.print_summary()
    return drift_metrics.get_summary()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--delays', type=int, nargs='+', default=[1, 3, 5])
    args = parser.parse_args()

    print(f"Analyzing drift from {args.data_path}...")
    analyze_layer_drift_from_data(args.data_path, args.delays)
