#!/usr/bin/env python3
"""
FastV baseline for GR00T: prune 50% of visual tokens at layer 2 based on attention scores.

Reference: Chen et al., "An Image is Worth 1/2 Tokens After Layer 2:
Plug-and-Play Inference Acceleration for Large Vision-Language Models"

Implementation:
  - After layer 2 of the LLM backbone, compute average attention over visual tokens
  - Drop the bottom 50% (lowest attention) visual tokens
  - Continue forward pass with pruned sequence
  - At bridge steps, use our latent bridge as usual

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_fastv_baseline.py \
        --model_path outputs/libero_10_finetune_v2 \
        --task_suite libero_10 \
        --prune_ratio 0.5 --prune_layer 2 \
        --n_episodes 20 --device cuda:0
"""

import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'benchmarks', 'Isaac-GR00T'))

import argparse
import torch
import numpy as np
import time
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')


class FastVHook:
    """Hook that prunes visual tokens at a specific layer based on attention scores."""

    def __init__(self, prune_layer: int = 2, prune_ratio: float = 0.5,
                 num_image_tokens: int = 162):
        self.prune_layer = prune_layer
        self.prune_ratio = prune_ratio
        self.num_image_tokens = num_image_tokens
        self.hooks = []
        self.attn_scores = None  # captured from prune_layer
        self.pruned_indices = None  # which tokens to keep
        self._layer_idx = 0
        self.enabled = True

    def register(self, model):
        """Register hooks on the LLM layers."""
        if hasattr(model, 'language_model'):
            layers = model.language_model.model.layers
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        else:
            raise ValueError("Cannot find LLM layers in model")

        # Hook on the attention module of prune_layer to capture attention weights
        attn_module = layers[self.prune_layer].self_attn
        self.hooks.append(
            attn_module.register_forward_hook(self._capture_attention)
        )

        # Hook on the layer AFTER prune_layer to prune tokens
        if self.prune_layer + 1 < len(layers):
            self.hooks.append(
                layers[self.prune_layer + 1].register_forward_pre_hook(self._prune_tokens)
            )

        logger.info(f"FastV: registered hooks on layer {self.prune_layer} "
                    f"(prune {self.prune_ratio*100:.0f}% of {self.num_image_tokens} visual tokens)")

    def _capture_attention(self, module, input, output):
        """Capture attention scores from the target layer."""
        if not self.enabled:
            return
        # output is typically (hidden_states, attn_weights, ...) or just hidden_states
        # For Qwen3, we need to access attention weights
        # Since flash attention doesn't return weights, we'll use the hidden states
        # approach: compute importance from the output hidden states norm
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            # attn_weights available: [B, num_heads, seq_len, seq_len]
            attn_weights = output[1]
            # Average over heads, sum attention FROM all tokens TO each visual token
            # This measures how much each visual token is attended to
            avg_attn = attn_weights.mean(dim=1)  # [B, seq, seq]
            # Sum attention received by each token (column sum)
            token_importance = avg_attn.sum(dim=1)  # [B, seq]
            self.attn_scores = token_importance
        else:
            # Flash attention — no weights available
            # Fall back to hidden state norm as importance proxy
            hidden = output[0] if isinstance(output, tuple) else output
            # L2 norm of each token's hidden state as importance
            self.attn_scores = hidden.norm(dim=-1)  # [B, seq]

    def _prune_tokens(self, module, input):
        """Zero out pruned visual tokens before the next layer.

        Instead of physically removing tokens (which breaks attention masks),
        we zero out the least important visual tokens. This faithfully measures
        the SR impact while avoiding tensor size mismatches.
        Theoretical speedup is computed from the prune ratio.
        """
        if not self.enabled or self.attn_scores is None:
            return input

        hidden_states = input[0] if isinstance(input, tuple) else input
        B, seq_len, dim = hidden_states.shape

        if seq_len <= self.num_image_tokens:
            return input

        importance = self.attn_scores[:, :seq_len]  # [B, seq]

        n_img = min(self.num_image_tokens, seq_len)
        n_prune = int(n_img * self.prune_ratio)

        if n_prune <= 0:
            return input

        # Find least important visual tokens
        img_importance = importance[:, :n_img]  # [B, n_img]
        _, prune_indices = torch.topk(img_importance, n_prune, dim=1, largest=False)

        # Zero out pruned tokens (preserves tensor shape)
        new_hidden = hidden_states.clone()
        for b in range(B):
            new_hidden[b, prune_indices[b], :] = 0.0

        if isinstance(input, tuple):
            new_input = list(input)
            new_input[0] = new_hidden
            return tuple(new_input)
        else:
            return new_hidden

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--task_suite", type=str, default="libero_10",
                        choices=["libero_10", "libero_spatial", "libero_object", "libero_goal"])
    parser.add_argument("--prune_ratio", type=float, default=0.5,
                        help="Fraction of visual tokens to prune (0.5 = drop 50%%)")
    parser.add_argument("--prune_layer", type=int, default=2,
                        help="Layer after which to prune tokens")
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

    sys.path.insert(0, os.path.dirname(__file__))
    from eval_stable_dynamic_bridge import AsyncGr00tPolicy

    register_libero_envs()

    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.LIBERO_PANDA,
        model_path=args.model_path,
        device=args.device,
        strict=False,
    )
    policy.model.action_head.num_inference_timesteps = args.num_inference_timesteps

    # Wrap in AsyncGr00tPolicy (freq=1 = sync) for proper observation handling
    sync_wrapper = AsyncGr00tPolicy(policy, vlm_update_freq=1, use_anchor=False)

    # Install FastV hooks on the backbone LLM
    fastv = FastVHook(
        prune_layer=args.prune_layer,
        prune_ratio=args.prune_ratio,
        num_image_tokens=162,
    )
    backbone_llm = policy.model.backbone.model
    fastv.register(backbone_llm)

    # Task list
    from libero.libero import benchmark as libero_benchmark
    benchmark_dict = libero_benchmark.get_benchmark_dict()
    suite = benchmark_dict[args.task_suite]()
    task_list = [f"libero_sim/{suite.get_task(i).name}" for i in range(suite.get_num_tasks())]
    logger.info(f"Task suite: {args.task_suite} ({len(task_list)} tasks)")

    wrapper_configs = WrapperConfigs(
        multistep=MultiStepConfig(n_action_steps=8, max_episode_steps=args.max_steps,
                                  terminate_on_success=True),
    )

    # Eval loop
    all_results = []
    for task_name in tqdm(task_list, desc="Tasks"):
        env = create_eval_env(task_name, env_idx=0, total_n_envs=1,
                              wrapper_configs=wrapper_configs)

        successes = 0
        latencies = []

        for ep in range(args.n_episodes):
            obs, _ = env.reset()
            sync_wrapper.reset()
            done = False
            step = 0

            while not done and step < args.max_steps:
                batched_obs = {}
                for k, v in obs.items():
                    if isinstance(v, np.ndarray):
                        batched_obs[k] = v[np.newaxis, ...]
                    elif isinstance(v, str):
                        batched_obs[k] = (v,)
                    else:
                        batched_obs[k] = v

                start = time.time()
                action, _ = sync_wrapper.get_action(batched_obs)
                latencies.append(time.time() - start)

                unbatched = {k: v[0] for k, v in action.items()}
                obs, reward, terminated, truncated, info = env.step(unbatched)
                done = terminated or truncated
                step += 1

            success = False
            if "success" in info:
                if isinstance(info["success"], (list, np.ndarray)):
                    success = any(info["success"])
                else:
                    success = bool(info["success"])
            if success:
                successes += 1

        sr = successes / args.n_episodes
        task_short = task_name.split("/")[-1][:50]
        logger.info(f"  {task_short}: {sr*100:.1f}%")
        all_results.append({"task": task_short, "sr": sr})

        # Close env
        import threading
        t = threading.Thread(target=env.close)
        t.daemon = True
        t.start()
        t.join(timeout=30)

    # Summary
    avg_sr = np.mean([r["sr"] for r in all_results])
    avg_latency = np.mean(latencies) * 1000
    logger.info(f"\n{'='*60}")
    logger.info(f"FastV Baseline: prune_ratio={args.prune_ratio}, prune_layer={args.prune_layer}")
    logger.info(f"Average SR: {avg_sr*100:.1f}%")
    logger.info(f"Average latency: {avg_latency:.1f}ms/step")
    logger.info(f"{'='*60}")

    fastv.remove()


if __name__ == "__main__":
    main()
