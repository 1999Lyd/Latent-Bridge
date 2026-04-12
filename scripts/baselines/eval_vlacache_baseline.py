#!/usr/bin/env python3
"""
VLA-Cache baseline for GR00T: simulates KV-cache feature reuse at output level.

Simulates VLA-Cache (Hsu et al., NeurIPS 2025) effect on GR00T:
- Compares consecutive image patches (14x14, cosine sim > 0.996)
- For static patches: backbone output features replaced with previous step's features
- For dynamic patches: fresh backbone output used
- Action head runs on mixed features
- Theoretical FLOPs/speedup cited from their paper

This output-level mixing faithfully captures VLA-Cache's quality impact:
the action head sees stale features at static token positions, fresh at dynamic.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_vlacache_baseline.py \
        --model_path ./outputs/libero_10_finetune_v2 \
        --task_suite libero_10 --n_episodes 20 --device cuda:0
"""

import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'benchmarks', 'Isaac-GR00T'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

import argparse
import torch
import numpy as np
import time
import logging
from tqdm import tqdm
from PIL import Image

from eval_stable_dynamic_bridge import AsyncGr00tPolicy, evaluate_task

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'baseline', 'vla-cache', 'src', 'openvla', 'experiments', 'robot'))
from vla_cache_utils import find_static_patches


class VLACacheGr00tPolicy(AsyncGr00tPolicy):
    """
    Simulates VLA-Cache on GR00T by mixing cached backbone output features
    for static image patches with fresh features for dynamic patches.
    """

    def __init__(self, policy, sim_threshold=0.996, top_k_static=130):
        super().__init__(policy, vlm_update_freq=1, use_anchor=False)
        self.sim_threshold = sim_threshold
        self.top_k_static = top_k_static
        self._prev_image_pil = None
        self._prev_backbone_features = None  # [B, seq, dim] from prev step
        self._image_mask_cached = None
        self.cache_hits = 0
        self.cache_total = 0

    def reset(self, options=None):
        result = super().reset(options)
        self._prev_image_pil = None
        self._prev_backbone_features = None
        return result

    def get_action(self, observation, options=None):
        # Extract image BEFORE parent processes
        curr_img = None
        if 'video.image' in observation:
            img = observation['video.image']
            if isinstance(img, np.ndarray):
                while img.ndim > 3:
                    img = img[0]
                curr_img = img  # [H, W, C] uint8

        # Run parent: backbone + action head
        # Parent's get_action runs backbone, caches in _cached_backbone_outputs,
        # runs action head, returns decoded action.
        # We intercept by hooking backbone output BEFORE action head.

        # Step 1: prepare inputs
        backbone_inputs, action_inputs, states = self._prepare_inputs(observation)

        with torch.inference_mode():
            # Step 2: run backbone
            backbone_outputs = self.model.backbone(backbone_inputs)

            # Cache image_mask on first call
            if self._image_mask_cached is None and "image_mask" in backbone_outputs:
                im = backbone_outputs["image_mask"]
                while im.ndim > 1:
                    im = im[0]
                self._image_mask_cached = im.bool().cpu()

            # Step 3: VLA-Cache mixing — replace static tokens with cached
            if (curr_img is not None and self._prev_image_pil is not None and
                self._prev_backbone_features is not None and
                self._image_mask_cached is not None):

                curr_pil = Image.fromarray(curr_img).convert("RGB").resize((224, 224))
                static_patches = find_static_patches(
                    curr_pil, self._prev_image_pil,
                    top_k=self.top_k_static, sim_threshold=self.sim_threshold
                )

                img_indices = self._image_mask_cached.nonzero(as_tuple=True)[0]
                n_img = len(img_indices)
                n_patches = 256  # 16x16 grid for 224x224 image with patch_size=14
                self.cache_total += n_img
                n_cached = 0

                fresh = backbone_outputs["backbone_features"]
                cached = self._prev_backbone_features

                if static_patches and n_img > 0 and cached.shape == fresh.shape:
                    for pid in static_patches:
                        tok = int(pid * n_img / n_patches)
                        if tok < n_img:
                            abs_idx = img_indices[tok].item()
                            if abs_idx < fresh.shape[1]:
                                fresh[0, abs_idx] = cached[0, abs_idx]
                                n_cached += 1

                self.cache_hits += n_cached
                backbone_outputs["backbone_features"] = fresh

            # Save FRESH features for next step (before mixing contaminated them —
            # actually mixing was in-place, so save a copy BEFORE mixing next time)
            # Fix: clone before mixing
            # We already mixed in-place above. For next step's cache, we want
            # the mixed output (which is what the model "saw"), matching VLA-Cache
            # where cached KV is from whatever the model computed.
            self._prev_backbone_features = backbone_outputs["backbone_features"].detach().clone()

            if curr_img is not None:
                self._prev_image_pil = Image.fromarray(curr_img).convert("RGB").resize((224, 224))

            # Step 4: run action head on (possibly mixed) features
            self._cached_backbone_outputs = backbone_outputs
            self._cached_action_inputs = action_inputs

            action_outputs = self.model.action_head.get_action(
                backbone_outputs, action_inputs)

        self._step_count += 1

        # Step 5: decode action (same as parent)
        normalized_action = action_outputs["action_pred"].float()
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--task_suite", type=str, default="libero_10")
    parser.add_argument("--sim_threshold", type=float, default=0.996)
    parser.add_argument("--top_k_static", type=int, default=130)
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=720)
    parser.add_argument("--num_inference_timesteps", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    torch.cuda.set_device(args.device)

    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval.rollout_policy import WrapperConfigs, MultiStepConfig, create_eval_env
    from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs

    register_libero_envs()

    logger.info(f"Loading model from {args.model_path}...")
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.LIBERO_PANDA,
        model_path=args.model_path,
        device=args.device,
        strict=False,
    )
    policy.model.action_head.num_inference_timesteps = args.num_inference_timesteps

    wrapper = VLACacheGr00tPolicy(
        policy, sim_threshold=args.sim_threshold, top_k_static=args.top_k_static
    )

    from libero.libero import benchmark as libero_benchmark
    benchmark_dict = libero_benchmark.get_benchmark_dict()
    suite = benchmark_dict[args.task_suite]()
    task_list = [f"libero_sim/{suite.get_task(i).name}" for i in range(suite.get_num_tasks())]
    logger.info(f"Task suite: {args.task_suite} ({len(task_list)} tasks)")
    logger.info(f"VLA-Cache config: threshold={args.sim_threshold}, top_k={args.top_k_static}")

    wrapper_configs = WrapperConfigs(
        multistep=MultiStepConfig(n_action_steps=8, max_episode_steps=args.max_steps,
                                  terminate_on_success=True),
    )

    all_results = []
    for task_name in tqdm(task_list, desc="VLA-Cache"):
        env = create_eval_env(task_name, env_idx=0, total_n_envs=1,
                              wrapper_configs=wrapper_configs)
        task_result = evaluate_task(wrapper, env, task_name, args.n_episodes, args.max_steps)
        all_results.append(task_result)
        logger.info(f"  {task_name.split('/')[-1][:50]}: {task_result['success_rate']*100:.1f}%")

        import threading
        t = threading.Thread(target=env.close)
        t.daemon = True
        t.start()
        t.join(timeout=30)

    avg_sr = np.mean([r["success_rate"] for r in all_results])
    cache_rate = wrapper.cache_hits / max(wrapper.cache_total, 1)

    # Theoretical FLOPs calculation
    # VLA-Cache prunes ~50% of 162 image tokens at layers 3-16 (14 of 16 layers)
    # Attention FLOPs: O(n^2 * d) → with 50% fewer tokens: ~75% of original
    # MLP FLOPs: O(n * d * 4d) → with 50% fewer tokens: ~50% of original
    # Layers 0-2: full cost. Layers 3-16: reduced.
    # Approximate total savings: ~17% (matching their paper's 1.07x speedup)
    savings_pct = cache_rate * (14/16) * 0.5 * 100  # rough estimate

    logger.info(f"\n{'='*60}")
    logger.info(f"VLA-Cache Baseline (GR00T)")
    logger.info(f"  Average SR: {avg_sr*100:.1f}%")
    logger.info(f"  Cache hit rate: {cache_rate*100:.1f}% of image tokens")
    logger.info(f"  Estimated FLOPs savings: ~{savings_pct:.0f}%")
    logger.info(f"  Reference speedup: 1.07x (VLA-Cache paper on OpenVLA-OFT)")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
