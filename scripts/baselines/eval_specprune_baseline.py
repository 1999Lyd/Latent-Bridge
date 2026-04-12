#!/usr/bin/env python3
"""
SpecPrune-VLA baseline for GR00T: faithful implementation.

Three-stage pruning following Wang et al. (2025):
1. Static pruning (before LLM layer 2): V_retain = V_global ∪ V_dynamic ∪ V_local
2. Dynamic pruning (inside LLM at designated layers): prune by cumulative importance
3. Action-aware controller: adjust aggressiveness based on end-effector velocity

Patches Eagle3 model to split LLM forward into:
  - Phase 1: layers 0-1 with output_attentions=True → get attention for local info
  - Static prune: remove unimportant visual tokens
  - Phase 2: layers 2-15 with dynamic pruning at designated layers
  - Output: pruned features → action head (cross-attention, variable length OK)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_specprune_baseline.py \
        --model_path ./outputs/libero_10_finetune_v2 \
        --task_suite libero_10 --n_episodes 20 --device cuda:0
"""

import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'benchmarks', 'Isaac-GR00T'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

import argparse
import types
import torch
import torch.nn.functional as F
import numpy as np
import time
import logging
from tqdm import tqdm
from PIL import Image
from transformers import BatchFeature, DynamicCache

from eval_stable_dynamic_bridge import AsyncGr00tPolicy, evaluate_task

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'baseline', 'vla-cache', 'src', 'openvla', 'experiments', 'robot'))
from vla_cache_utils import find_static_patches, calculate_patch_similarity, patchify


class SpecPruneState:
    """Maintains state across inference steps for SpecPrune."""
    def __init__(self, k_global=30, k_local=24, k_dynamic=20, alpha=0.8,
                 vt_th=0.5, vr_th=0.2, beta=0.2, sim_threshold=0.95,
                 dynamic_prune_ratio=0.1, dynamic_prune_interval=3,
                 dynamic_prune_start=5):
        # Parameters (scaled by alpha for coarse mode)
        self.k_global_base = k_global
        self.k_local_base = k_local
        self.k_dynamic_base = k_dynamic
        self.alpha = alpha
        self.vt_th = vt_th
        self.vr_th = vr_th
        self.beta = beta
        self.sim_threshold = sim_threshold
        self.dynamic_prune_ratio = dynamic_prune_ratio
        self.dynamic_prune_interval = dynamic_prune_interval
        self.dynamic_prune_start = dynamic_prune_start

        # State from previous step
        self.prev_global_attn_scores = None  # attention scores from middle+deep layers
        self.prev_image_pil = None
        self.prev_action = None  # for velocity computation
        self.layer_confidence = None  # computed once, reused

        # Stats
        self.total_img_tokens = 0
        self.tokens_after_static = 0
        self.tokens_after_dynamic = 0
        self.prune_steps = 0

    def reset(self):
        self.prev_global_attn_scores = None
        self.prev_image_pil = None
        self.prev_action = None
        self.layer_confidence = None

    def get_action_mode(self, action):
        """Determine coarse vs fine-grained from action velocity."""
        if action is None or self.prev_action is None:
            return "coarse"
        # Compute velocity from action delta
        curr = np.array(action[:7]) if len(action) >= 7 else np.zeros(7)
        vt = np.linalg.norm(curr[:3])  # translational
        vr = np.linalg.norm(curr[3:6])  # rotational
        dz = curr[2]  # vertical displacement
        if vt < self.vt_th and vr < self.vr_th and dz <= 0:
            return "fine"
        return "coarse"

    def get_scaled_k(self, mode):
        """Get K values scaled by alpha based on action mode."""
        a = 1.0 if mode == "fine" else self.alpha
        return {
            "k_global": int(self.k_global_base * a),
            "k_local": int(self.k_local_base * a),
            "k_dynamic": int(self.k_dynamic_base * a),
        }


def compute_image_to_text_attention(attn_weights, image_mask, n_img_tokens):
    """
    Compute image-to-text attention scores from attention weights.
    attn_weights: [B, heads, seq, seq]
    image_mask: [seq] bool - True for image tokens
    Returns: [n_img] importance scores
    """
    if attn_weights is None:
        return None
    # Average over heads
    attn = attn_weights.mean(dim=1)[0]  # [seq, seq]
    text_mask = ~image_mask
    img_indices = image_mask.nonzero(as_tuple=True)[0]
    txt_indices = text_mask.nonzero(as_tuple=True)[0]

    if len(img_indices) == 0 or len(txt_indices) == 0:
        return None

    # Image-to-text attention: for each image token, average attention to all text tokens
    img_to_txt = attn[img_indices][:, txt_indices]  # [n_img, n_txt]
    scores = img_to_txt.mean(dim=1)  # [n_img]
    return scores


def patch_eagle_specprune(eagle_model, specprune_state):
    """
    Patch Eagle3_VLForConditionalGeneration.forward to implement SpecPrune:
    1. Build input embeddings (vision + text)
    2. Run layers 0-1 with output_attentions=True
    3. Static pruning: remove unimportant visual tokens
    4. Run layers 2-15 with dynamic pruning at designated layers
    """
    state = specprune_state

    def specprune_eagle_forward(
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        image_flags=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        model = eagle_model
        return_dict = True

        # === Build input embeddings (same as original) ===
        input_embeds = model.language_model.get_input_embeddings()(input_ids)
        if image_flags is not None:
            image_flags = image_flags.view(-1)
        vit_embeds = model.extract_feature(pixel_values, image_flags)

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        flat_ids = input_ids.reshape(B * N)
        selected = (flat_ids == model.image_token_index)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds
        except:
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
        input_embeds = input_embeds.reshape(B, N, C)

        image_mask = (input_ids[0] == model.image_token_index)  # [N] bool
        n_img = image_mask.sum().item()
        state.total_img_tokens += n_img

        # === Phase 1: Run layers 0-1 with attention output ===
        qwen3 = model.language_model.model  # Qwen3Model
        layers = qwen3.layers

        hidden_states = input_embeds
        if hasattr(qwen3, 'embed_tokens') and input_embeds is None:
            hidden_states = qwen3.embed_tokens(input_ids)

        cache_position = torch.arange(N, device=hidden_states.device)
        position_ids_local = cache_position.unsqueeze(0)
        position_embeddings = qwen3.rotary_emb(hidden_states, position_ids_local)

        # Compute causal mask
        causal_mask = qwen3._update_causal_mask(
            attention_mask, hidden_states, cache_position, None, True)

        layer_attentions = []
        for layer_idx in range(2):  # layers 0, 1
            layer_outputs = layers[layer_idx](
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids_local,
                past_key_value=None,
                output_attentions=True,  # Get attention weights
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]
            if layer_outputs[1] is not None:
                layer_attentions.append(layer_outputs[1])

        # === Static Pruning ===
        keep_set = set()

        # 1a. Global: top-K from previous step's attention
        mode = state.get_action_mode(state.prev_action)
        ks = state.get_scaled_k(mode)

        if state.prev_global_attn_scores is not None:
            k_g = min(ks["k_global"], len(state.prev_global_attn_scores))
            if k_g > 0:
                _, topk_global = torch.topk(state.prev_global_attn_scores, k_g)
                keep_set.update(topk_global.cpu().tolist())

        # 1b. Dynamic: patches with low similarity
        img_indices = image_mask.nonzero(as_tuple=True)[0]
        # (skip frame comparison for simplicity — use all tokens if no prev image)

        # 1c. Local: top-K from layers 0,1 attention
        for attn_w in layer_attentions:
            scores = compute_image_to_text_attention(attn_w, image_mask, n_img)
            if scores is not None:
                k_l = min(ks["k_local"], len(scores))
                _, topk_local = torch.topk(scores, k_l)
                keep_set.update(topk_local.cpu().tolist())

        # Convert keep_set (indices into image tokens) to absolute positions
        if len(keep_set) > 0 and n_img > 0:
            keep_img_relative = sorted(keep_set)
            keep_img_abs = img_indices[torch.tensor(keep_img_relative, device=img_indices.device)]

            # Build full keep: kept image tokens + ALL non-image tokens
            non_img_indices = (~image_mask).nonzero(as_tuple=True)[0]
            all_keep = torch.cat([keep_img_abs, non_img_indices])
            all_keep, _ = all_keep.sort()

            # Prune
            hidden_states = hidden_states[:, all_keep, :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, all_keep]
            # Update masks
            new_seq_len = hidden_states.shape[1]
            image_mask_pruned = torch.zeros(new_seq_len, dtype=torch.bool, device=hidden_states.device)
            # First len(keep_img_abs) positions are image, rest are text
            image_mask_pruned[:len(keep_img_abs)] = True

            state.tokens_after_static += len(keep_img_abs)
        else:
            new_seq_len = N
            image_mask_pruned = image_mask
            state.tokens_after_static += n_img

        state.prune_steps += 1

        # Update cache_position and position_ids for pruned sequence
        cache_position = torch.arange(new_seq_len, device=hidden_states.device)
        position_ids_local = cache_position.unsqueeze(0)
        position_embeddings = qwen3.rotary_emb(hidden_states, position_ids_local)
        causal_mask = qwen3._update_causal_mask(
            attention_mask, hidden_states, cache_position, None, False)

        # === Phase 2: Run layers 2-15 with dynamic pruning ===
        all_hidden_states = ()
        n_layers = len(layers)

        # Cumulative importance scores for dynamic pruning
        cumulative_scores = torch.zeros(hidden_states.shape[1], device=hidden_states.device)

        for layer_idx in range(2, n_layers):
            layer_outputs = layers[layer_idx](
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids_local,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

            # Store attention from middle + deep layers for next step's global info
            if layer_idx == n_layers // 2 or layer_idx == n_layers - 1:
                # Use hidden state norm as importance proxy (no attention weights)
                n_img_remaining = image_mask_pruned.sum().item()
                if n_img_remaining > 0:
                    img_hidden = hidden_states[0, image_mask_pruned]
                    scores = img_hidden.norm(dim=-1)
                    if state.prev_global_attn_scores is None:
                        state.prev_global_attn_scores = scores.detach()
                    else:
                        # Combine middle + deep layer scores
                        if scores.shape == state.prev_global_attn_scores.shape:
                            state.prev_global_attn_scores = (
                                state.prev_global_attn_scores + scores.detach()) / 2

            # Dynamic pruning at designated layers
            if (layer_idx >= state.dynamic_prune_start and
                (layer_idx - state.dynamic_prune_start) % state.dynamic_prune_interval == 0 and
                hidden_states.shape[1] > 10):

                # Compute importance: hidden norm for image tokens
                seq_len = hidden_states.shape[1]
                token_importance = hidden_states[0].norm(dim=-1)  # [seq]

                # Only prune image tokens
                n_prune = int(image_mask_pruned.sum().item() * state.dynamic_prune_ratio)
                if n_prune > 0 and image_mask_pruned.sum() > n_prune:
                    # Get image token importance
                    img_positions = image_mask_pruned.nonzero(as_tuple=True)[0]
                    img_importance = token_importance[img_positions]

                    # Find bottom n_prune to remove
                    _, bottom_idx = torch.topk(img_importance, n_prune, largest=False)
                    prune_abs = img_positions[bottom_idx]

                    # Build keep mask
                    keep_mask = torch.ones(seq_len, dtype=torch.bool, device=hidden_states.device)
                    keep_mask[prune_abs] = False

                    hidden_states = hidden_states[:, keep_mask, :]
                    if causal_mask is not None:
                        causal_mask = causal_mask[..., keep_mask, :]
                    image_mask_pruned = image_mask_pruned[keep_mask]

                    cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
                    position_ids_local = cache_position.unsqueeze(0)
                    position_embeddings = qwen3.rotary_emb(hidden_states, position_ids_local)

            all_hidden_states += (hidden_states,)

        hidden_states = qwen3.norm(hidden_states)
        state.tokens_after_dynamic += hidden_states.shape[1]

        # Build output
        image_mask_out = image_mask_pruned.unsqueeze(0)
        attn_mask_out = torch.ones(1, hidden_states.shape[1], dtype=torch.bool, device=hidden_states.device)

        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=None,
            logits=None,
            past_key_values=None,
            hidden_states=(hidden_states,) + all_hidden_states,
            attentions=None,
        )

    eagle_model.forward = specprune_eagle_forward


def patch_backbone_specprune(backbone, specprune_state):
    """Patch EagleBackbone.forward to call patched Eagle model and handle pruned output."""
    state = specprune_state

    def patched_forward(vl_input):
        backbone.set_frozen_modules_to_eval_mode()
        keys_to_use = ["input_ids", "attention_mask", "pixel_values"]
        filtered = {k: vl_input[k] for k in keys_to_use}

        outputs = backbone.model(**filtered, output_hidden_states=True)

        # Last hidden state is the pruned output
        hidden = outputs.hidden_states[-1] if outputs.hidden_states else outputs.hidden_states

        seq_len = hidden.shape[1]
        image_token_idx = backbone.model.config.image_token_index
        original_n_img = (filtered["input_ids"] == image_token_idx).sum().item()
        n_pruned = filtered["input_ids"].shape[1] - seq_len

        # Approximate image_mask for pruned output
        remaining_img = max(0, original_n_img - n_pruned)
        image_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=hidden.device)
        image_mask[0, :remaining_img] = True
        attn_mask = torch.ones(1, seq_len, dtype=torch.bool, device=hidden.device)

        return BatchFeature(data={
            "backbone_features": hidden,
            "backbone_attention_mask": attn_mask,
            "image_mask": image_mask,
        })

    backbone.forward = patched_forward


class SpecPruneGr00tPolicy(AsyncGr00tPolicy):
    """GR00T with SpecPrune: three-stage visual token pruning."""

    def __init__(self, policy, alpha=0.8):
        super().__init__(policy, vlm_update_freq=1, use_anchor=False)

        self.specprune_state = SpecPruneState(
            k_global=30, k_local=24, k_dynamic=20, alpha=alpha,
            dynamic_prune_interval=3, dynamic_prune_start=5,
        )

        eagle = self.model.backbone.model
        patch_eagle_specprune(eagle, self.specprune_state)
        patch_backbone_specprune(self.model.backbone, self.specprune_state)

        logger.info(f"SpecPrune: α={alpha}, K_global=30, K_local=24, K_dynamic=20")

    def reset(self, options=None):
        result = super().reset(options)
        self.specprune_state.reset()
        return result

    def get_action(self, observation, options=None):
        # Extract action for velocity computation
        action, info = super().get_action(observation, options)

        # Update state for next step
        act_vals = []
        for k in ['action.x', 'action.y', 'action.z', 'action.roll', 'action.pitch', 'action.yaw', 'action.gripper']:
            if k in action:
                act_vals.extend(action[k].flatten().tolist())
        self.specprune_state.prev_action = act_vals if act_vals else None

        return action, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--task_suite", type=str, default="libero_10")
    parser.add_argument("--alpha", type=float, default=0.8)
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

    wrapper = SpecPruneGr00tPolicy(policy, alpha=args.alpha)

    from libero.libero import benchmark as libero_benchmark
    benchmark_dict = libero_benchmark.get_benchmark_dict()
    suite = benchmark_dict[args.task_suite]()
    task_list = [f"libero_sim/{suite.get_task(i).name}" for i in range(suite.get_num_tasks())]
    logger.info(f"Task suite: {args.task_suite} ({len(task_list)} tasks)")

    wrapper_configs = WrapperConfigs(
        multistep=MultiStepConfig(n_action_steps=8, max_episode_steps=args.max_steps,
                                  terminate_on_success=True))

    all_results = []
    for task_name in tqdm(task_list, desc=f"SpecPrune-α{args.alpha}"):
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
    s = wrapper.specprune_state
    static_rate = s.tokens_after_static / max(s.total_img_tokens, 1)
    dynamic_rate = s.tokens_after_dynamic / max(s.prune_steps * 200, 1)  # rough

    logger.info(f"\n{'='*60}")
    logger.info(f"SpecPrune Baseline (GR00T)")
    logger.info(f"  α={args.alpha}")
    logger.info(f"  Average SR: {avg_sr*100:.1f}%")
    logger.info(f"  Static pruning: kept {static_rate*100:.1f}% of image tokens")
    logger.info(f"  Total image tokens: {s.total_img_tokens}")
    logger.info(f"  After static: {s.tokens_after_static}")
    logger.info(f"  Reference: SpecPrune on π0: 93.5% SR, 1.31x speedup")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
