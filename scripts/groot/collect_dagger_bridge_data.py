#!/usr/bin/env python3
"""
DAgger data collection for autoregressive bridge.

Runs bridge policy in LIBERO env. At every step (both VLM and bridge),
records (z_input, z_gt, stable, state, action, image_mask) pairs where:
  - z_input: what the bridge actually used as input (own prediction or VLM)
  - z_gt: VLM ground truth on the same observation
  - stable: stable layer features from VLM
  - state/action: robot state and previous action

This teaches the bridge to correct from its own noisy predictions.

Usage:
    CUDA_VISIBLE_DEVICES=6,7 python scripts/collect_dagger_bridge_data.py \
        --model_path outputs/libero_10_finetune \
        --bridge_path outputs/bridge_v3/best_model_dit.pt \
        --output_path outputs/bridge_v3/dagger_data.h5 \
        --n_episodes_per_task 30 --device cuda:0
"""

import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'benchmarks', 'Isaac-GR00T'))

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import h5py
import time
import logging
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')


def batch_obs(obs):
    batched = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            batched[k] = v[np.newaxis, ...]
        elif isinstance(v, str):
            batched[k] = (v,)
        else:
            batched[k] = v
    return batched


def get_translation_mag(action_7d):
    """Compute translation magnitude (dims 0-2) from 7d action."""
    return float(np.linalg.norm(action_7d[:3]))


def collect_dagger_episode(policy_wrapper, env, max_steps, phase_aware=False,
                           high_thresh=0.5, low_thresh=0.1,
                           high_freq=2, mid_freq=3, low_freq=4):
    """
    Run one episode with bridge policy. At each step, record:
    - z_input: features the bridge used as input (bridge pred or fresh VLM)
    - z_gt: VLM ground truth features on same observation
    - stable: stable layer features
    - state, action, image_mask

    If phase_aware=True, uses PhaseAwareBridgeGr00tPolicy which dynamically
    adjusts VLM frequency based on previous action TRANSLATION magnitude:
    - High trans (>0.5): freq=2 — large feature deltas, need frequent VLM
    - Medium trans (0.1-0.5): freq=3 — standard bridge
    - Low trans (<0.1): freq=4 — small deltas, safe for longer chains

    Returns list of step dicts and episode metadata.
    """
    obs, _ = env.reset()
    policy_wrapper.reset()

    steps = []
    done = False
    step = 0
    steps_since_vlm = 0
    current_freq = policy_wrapper.vlm_update_freq  # initial freq
    prev_action_mag = 0.0

    while not done and step < max_steps:
        batched_obs = batch_obs(obs)

        # Before get_action: capture what z_input will be
        # (the bridge's _current_features before this step)
        z_input_before = None
        if policy_wrapper._current_features is not None:
            z_input_before = policy_wrapper._current_features.clone().detach().float()

        action, _ = policy_wrapper.get_action(batched_obs)
        unbatched_action = {k: v[0] for k, v in action.items()}

        # Determine what happened this step
        if phase_aware:
            # After get_action, _steps_since_vlm_pa == 1 means VLM just ran
            # (it was set to 0 then incremented by 1)
            if hasattr(policy_wrapper, '_steps_since_vlm_pa'):
                is_vlm_step = (policy_wrapper._steps_since_vlm_pa == 1)
            else:
                is_vlm_step = (step == 0) or (steps_since_vlm >= current_freq)
        else:
            is_vlm_step = (step % policy_wrapper.vlm_update_freq == 0) or step == 0

        # Track steps since VLM
        if is_vlm_step:
            steps_since_vlm = 0
        else:
            steps_since_vlm += 1

        if is_vlm_step:
            # VLM step: z_input = z_gt (both are fresh VLM output)
            # Still useful as training data (clean input → clean output)
            z_gt = policy_wrapper._current_features.clone().detach().float()
            z_input = z_gt.clone()
            stable = policy_wrapper._current_stable_features.clone().detach().float()
        else:
            # Bridge step: z_input was the previous prediction, z_gt needs VLM
            z_input = z_input_before  # what bridge used as input

            # Run VLM to get ground truth on same observation
            # IMPORTANT: save/restore hook state so the GT VLM call doesn't
            # contaminate the bridge's stable features for the next step
            saved_stable = policy_wrapper._stable_layer_features
            saved_current_stable = policy_wrapper._current_stable_features.clone() if policy_wrapper._current_stable_features is not None else None

            backbone_inputs, action_inputs, _ = policy_wrapper._prepare_inputs(batched_obs)
            with torch.inference_mode():
                gt_backbone_outputs = policy_wrapper.model.backbone(backbone_inputs)
                z_gt = gt_backbone_outputs["backbone_features"].clone().detach().float()

            # Restore stable features (hook overwrote them during GT call)
            policy_wrapper._stable_layer_features = saved_stable
            policy_wrapper._current_stable_features = saved_current_stable

            stable = saved_current_stable.clone().detach().float() if saved_current_stable is not None else z_gt.clone()

        # Get masks
        if policy_wrapper._cached_masks is not None:
            image_mask = policy_wrapper._cached_masks[1].clone().detach()
        else:
            image_mask = None

        # Get state and action
        if policy_wrapper._cached_action_inputs is not None:
            state = policy_wrapper._cached_action_inputs["state"].detach().float().cpu()
            while state.ndim > 2:
                state = state.squeeze(0)
            state = state[0, :8].numpy().astype(np.float32)
        else:
            state = np.zeros(8, dtype=np.float32)

        # Previous action
        if policy_wrapper._last_action is not None:
            prev_act = policy_wrapper._last_action.detach().float().cpu()
            if prev_act.ndim == 3:
                prev_act = prev_act[0, 0, :]
            elif prev_act.ndim == 2:
                prev_act = prev_act[0, :]
            prev_act = prev_act[:7].numpy().astype(np.float32)
        else:
            prev_act = np.zeros(7, dtype=np.float32)

        # Squeeze features to [seq, dim]
        def squeeze_feat(f):
            while f.ndim > 3:
                f = f.squeeze(0)
            if f.ndim == 3:
                f = f.squeeze(0)  # [seq, dim]
            return f.cpu().numpy().astype(np.float16)

        step_data = {
            "z_input": squeeze_feat(z_input) if z_input is not None else None,
            "z_gt": squeeze_feat(z_gt),
            "stable": squeeze_feat(stable),
            "state": state,
            "action": prev_act,
            "is_vlm_step": is_vlm_step,
        }
        if image_mask is not None:
            step_data["image_mask"] = image_mask.cpu().numpy().squeeze()

        steps.append(step_data)

        obs, reward, terminated, truncated, info = env.step(unbatched_action)
        done = terminated or truncated
        step += 1

        # Track translation magnitude for phase-aware scheduling
        if phase_aware:
            prev_action_mag = get_translation_mag(prev_act)

    success = False
    if "success" in info:
        if isinstance(info["success"], (list, np.ndarray)):
            success = any(info["success"])
        else:
            success = bool(info["success"])

    return steps, success, step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="./outputs/libero_10_finetune")
    parser.add_argument("--bridge_path", type=str, required=True,
                        help="Path to SingleStepDiT checkpoint")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--n_episodes_per_task", type=int, default=30)
    parser.add_argument("--vlm_freq", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=720)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--task_filter", type=str, default=None)
    parser.add_argument("--task_suite", type=str, default="libero_10",
                        choices=["libero_10", "libero_spatial", "libero_object", "libero_goal", "libero_90"])
    parser.add_argument("--env_names", type=str, nargs="+", default=None,
                        help="Direct env names (bypass suite lookup)")
    parser.add_argument("--embodiment_tag", type=str, default="LIBERO_PANDA",
                        help="Embodiment tag (LIBERO_PANDA, ROBOCASA_PANDA_OMRON)")
    # Phase-aware args
    parser.add_argument("--phase_aware", action="store_true",
                        help="Use adaptive VLM freq based on action magnitude")
    parser.add_argument("--high_thresh", type=float, default=0.5,
                        help="Translation mag above this → high freq (more VLM)")
    parser.add_argument("--low_thresh", type=float, default=0.1,
                        help="Translation mag below this → low freq (fewer VLM)")
    parser.add_argument("--high_freq", type=int, default=2,
                        help="VLM freq for high action magnitude")
    parser.add_argument("--mid_freq", type=int, default=3,
                        help="VLM freq for medium action magnitude")
    parser.add_argument("--low_freq", type=int, default=4,
                        help="VLM freq for low action magnitude")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA adapter for action head")
    parser.add_argument("--num_inference_timesteps", type=int, default=1,
                        help="Number of ODE steps for action head (default: 1)")
    args = parser.parse_args()

    torch.cuda.set_device(args.device)

    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval.rollout_policy import WrapperConfigs, VideoConfig, MultiStepConfig, create_eval_env
    if "ROBOCASA" in args.embodiment_tag.upper():
        import robocasa  # noqa
    else:
        from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs
        register_libero_envs()

    embodiment = getattr(EmbodimentTag, args.embodiment_tag)
    logger.info(f"Loading GR00T model from {args.model_path}...")
    base_policy = Gr00tPolicy(
        embodiment_tag=embodiment,
        model_path=args.model_path,
        device=args.device,
        strict=False,
    )
    base_policy.model.action_head.num_inference_timesteps = args.num_inference_timesteps

    # Load LoRA adapter if provided
    if args.lora_path:
        from peft import PeftModel
        logger.info(f"Loading LoRA adapter from {args.lora_path}...")
        base_policy.model.action_head = PeftModel.from_pretrained(
            base_policy.model.action_head, args.lora_path
        )
        base_policy.model.action_head.eval()
        logger.info("LoRA adapter loaded successfully")

    # Load bridge
    from scripts.train_single_step_dit import SingleStepDiT
    checkpoint = torch.load(args.bridge_path, map_location=args.device, weights_only=False)
    config = checkpoint.get("config", {})
    bridge_model = SingleStepDiT(
        feature_dim=config.get("feature_dim", 2048),
        seq_len=config.get("seq_len", 204),
        hidden_dim=config.get("hidden_dim", 768),
        num_blocks=config.get("num_blocks", 12),
        num_heads=config.get("num_heads", 12),
        state_dim=config.get("state_dim", 8),
        action_dim=config.get("action_dim", 7),
        low_rank=config.get("low_rank", 0),
        num_image_tokens=config.get("num_image_tokens", 162),
    ).to(args.device)
    bridge_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    bridge_model.eval()
    bridge_image_only = config.get("image_only", False)
    bridge_seq_len = config.get("seq_len", 204)
    logger.info(f"Bridge loaded: {sum(p.numel() for p in bridge_model.parameters())/1e6:.1f}M params, "
                f"seq_len={bridge_seq_len}, image_only={bridge_image_only}")

    # Import policy wrapper
    from scripts.eval_stable_dynamic_bridge import AutoregressiveBridgeGr00tPolicy
    if args.phase_aware:
        from scripts.eval_stable_dynamic_bridge import PhaseAwareBridgeGr00tPolicy

    wrapper_configs = WrapperConfigs(
        video=VideoConfig(video_dir=None),
        multistep=MultiStepConfig(n_action_steps=8, max_episode_steps=args.max_steps,
                                  terminate_on_success=True),
    )

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
    logger.info(f"Tasks: {len(task_list)} (filter={args.task_filter})")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    total_episodes = 0
    total_steps = 0
    total_bridge_steps = 0
    total_vlm_steps = 0
    successes = 0

    with h5py.File(args.output_path, 'w') as hf:
        # Store metadata
        hf.attrs["vlm_freq"] = args.vlm_freq
        hf.attrs["bridge_path"] = args.bridge_path
        hf.attrs["model_path"] = args.model_path
        hf.attrs["phase_aware"] = args.phase_aware
        if args.phase_aware:
            hf.attrs["high_thresh"] = args.high_thresh
            hf.attrs["low_thresh"] = args.low_thresh
            hf.attrs["high_freq"] = args.high_freq
            hf.attrs["mid_freq"] = args.mid_freq
            hf.attrs["low_freq"] = args.low_freq

        for task_name in task_list:
            short_name = task_name.split('/')[-1][:50]
            logger.info(f"\n{'='*60}")
            logger.info(f"Task: {short_name}")

            env = create_eval_env(task_name, env_idx=0, total_n_envs=1,
                                  wrapper_configs=wrapper_configs)

            if args.phase_aware:
                policy_wrapper = PhaseAwareBridgeGr00tPolicy(
                    base_policy, bridge_model,
                    vlm_update_freq=args.vlm_freq,
                    use_anchor=False,
                    nav_vlm_freq=args.high_freq,
                    trans_vlm_freq=args.mid_freq,
                    manip_vlm_freq=args.low_freq,
                    nav_threshold=args.high_thresh,
                    manip_threshold=args.low_thresh,
                )
            else:
                policy_wrapper = AutoregressiveBridgeGr00tPolicy(
                    base_policy, bridge_model,
                    vlm_update_freq=args.vlm_freq,
                    use_anchor=False,
                    bridge_seq_len=bridge_seq_len,
                    image_only=bridge_image_only,
                )

            task_successes = 0
            for ep in tqdm(range(args.n_episodes_per_task), desc=short_name):
                steps, success, n_steps = collect_dagger_episode(
                    policy_wrapper, env, max_steps=90,  # policy calls
                    phase_aware=args.phase_aware,
                    high_thresh=args.high_thresh, low_thresh=args.low_thresh,
                    high_freq=args.high_freq, mid_freq=args.mid_freq, low_freq=args.low_freq,
                )

                if not steps:
                    continue

                # Filter out steps with None z_input (shouldn't happen but safety)
                valid_steps = [s for s in steps if s["z_input"] is not None]
                if not valid_steps:
                    continue

                ep_name = f"episode_{total_episodes:04d}"
                ep_grp = hf.create_group(ep_name)
                ep_grp.attrs["task_name"] = task_name
                ep_grp.attrs["success"] = success
                ep_grp.attrs["n_steps"] = n_steps

                # Stack and save
                z_input = np.stack([s["z_input"] for s in valid_steps], axis=0)
                z_gt = np.stack([s["z_gt"] for s in valid_steps], axis=0)
                stable = np.stack([s["stable"] for s in valid_steps], axis=0)
                states = np.stack([s["state"] for s in valid_steps], axis=0)
                actions = np.stack([s["action"] for s in valid_steps], axis=0)
                is_vlm = np.array([s["is_vlm_step"] for s in valid_steps], dtype=bool)

                ep_grp.create_dataset("z_input", data=z_input, compression="gzip", compression_opts=4)
                ep_grp.create_dataset("z_gt", data=z_gt, compression="gzip", compression_opts=4)
                ep_grp.create_dataset("stable", data=stable, compression="gzip", compression_opts=4)
                ep_grp.create_dataset("state", data=states, compression="gzip", compression_opts=4)
                ep_grp.create_dataset("action", data=actions, compression="gzip", compression_opts=4)
                ep_grp.create_dataset("is_vlm_step", data=is_vlm)

                if "image_mask" in valid_steps[0]:
                    img_mask = np.stack([s["image_mask"] for s in valid_steps], axis=0)
                    ep_grp.create_dataset("image_mask", data=img_mask, compression="gzip", compression_opts=4)

                n_bridge = sum(1 for s in valid_steps if not s["is_vlm_step"])
                n_vlm = sum(1 for s in valid_steps if s["is_vlm_step"])
                total_bridge_steps += n_bridge
                total_vlm_steps += n_vlm
                total_steps += len(valid_steps)
                total_episodes += 1
                if success:
                    successes += 1
                    task_successes += 1

                if (ep + 1) % 10 == 0:
                    logger.info(f"  Ep {ep+1}: {task_successes}/{ep+1} success, "
                                f"{total_steps} total steps")

            env.close()
            logger.info(f"  Task SR: {task_successes}/{args.n_episodes_per_task}")

        hf.attrs["total_episodes"] = total_episodes
        hf.attrs["total_steps"] = total_steps

    logger.info(f"\n{'='*60}")
    logger.info(f"DAgger collection complete!")
    logger.info(f"  Episodes: {total_episodes} ({successes} success)")
    logger.info(f"  Total steps: {total_steps} (bridge={total_bridge_steps}, vlm={total_vlm_steps})")
    logger.info(f"  Saved to: {args.output_path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
