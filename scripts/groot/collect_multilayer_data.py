#!/usr/bin/env python3
"""
Collect multi-layer VLM hidden states for training the Latent Bridge.

This script collects the last N layers of hidden states from the VLM backbone,
enabling experiments with multi-layer input for the latent bridge.
"""

import argparse
import numpy as np
import torch
import h5py
import os
from pathlib import Path
from tqdm import tqdm
import sys
from typing import Dict, List
from dataclasses import dataclass
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks" / "Isaac-GR00T"))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_libero_task_names(suite_name: str) -> List[str]:
    """Get task names from LIBERO benchmark API."""
    from libero.libero import benchmark
    benchmark_dict = benchmark.get_benchmark_dict()

    if suite_name not in benchmark_dict:
        logger.warning(f"Unknown suite: {suite_name}")
        return []

    task_suite = benchmark_dict[suite_name]()
    env_names = []
    for task_id in range(task_suite.get_num_tasks()):
        task = task_suite.get_task(task_id)
        env_names.append(f"libero_sim/{task.name}")

    return env_names


SUPPORTED_SUITES = ["libero_10", "libero_spatial", "libero_object", "libero_goal", "libero_90"]


@dataclass
class EpisodeDataMultiLayer:
    """Container for episode data with multi-layer hidden states."""
    episode_id: int
    task_name: str
    task_suite: str
    steps: List[Dict]
    success: bool
    total_reward: float


class VisionFeatureCapture:
    """Captures vision encoder features (pre-LLM) during backbone forward pass.

    Hooks into the Eagle model's mlp1 layer, which projects vision tokens
    from Siglip2 into the LLM embedding space (2048-dim). This fires during
    the normal backbone forward — no extra computation needed.

    Vision features represent what the vision encoder sees (spatial layout,
    object positions) WITHOUT the LLM's language-conditioned reasoning.
    ~430M params to compute vs ~2.1B for full VLM.
    """

    def __init__(self, backbone):
        self.features_list = []
        self._hook = None
        eagle_model = backbone.model
        if hasattr(eagle_model, 'mlp1'):
            self._hook = eagle_model.mlp1.register_forward_hook(self._capture)
            logger.info("VisionFeatureCapture: hooked into Eagle mlp1")
        else:
            logger.warning("VisionFeatureCapture: mlp1 not found, vision features unavailable")

    def _capture(self, module, inp, out):
        self.features_list.append(out.detach())

    def get_and_reset(self):
        """Get concatenated vision features and reset buffer."""
        if self.features_list:
            # Concatenate along sequence dim (one call per image)
            result = torch.cat(self.features_list, dim=1)  # [B, N_total, dim]
            self.features_list = []
            return result
        return None

    def remove(self):
        if self._hook:
            self._hook.remove()


class MultiLayerBackboneWrapper:
    """Wrapper to extract multi-layer hidden states from backbone."""

    def __init__(self, backbone, layer_indices: List[int] = None, num_layers: int = 4):
        """
        Args:
            backbone: VLM backbone model
            layer_indices: Specific layer indices to extract (e.g., [7, 14, 21, 28])
                          If None, uses last `num_layers` layers
            num_layers: Number of layers from end (only used if layer_indices is None)
        """
        self.backbone = backbone
        self.layer_indices = layer_indices
        self.num_layers = num_layers
        self._original_forward = backbone.forward

    def forward_with_all_layers(self, vl_input):
        """Forward pass that returns multiple layers of hidden states."""
        self.backbone.set_frozen_modules_to_eval_mode()

        keys_to_use = ["input_ids", "attention_mask", "pixel_values"]
        vl_input_filtered = {k: vl_input[k] for k in keys_to_use}

        # Get all hidden states
        outputs = self.backbone.model(**vl_input_filtered, output_hidden_states=True)
        all_hidden_states = outputs["hidden_states"]

        total_layers = len(all_hidden_states)

        # Select layers based on indices.
        # Qwen2 hidden_states layout (N transformer layers, 0-indexed):
        #   hidden_states[0]   = embedding output (input to layer 0)
        #   hidden_states[k]   = output of layer k-1 (input to layer k)
        #   hidden_states[N]   = output of layer N-1 + final LayerNorm
        #   hidden_states[-1]  = same as hidden_states[N] (post-norm)
        #
        # To match eval hooks on layers[K] (which capture OUTPUT of layer K):
        #   Use index K+1 (e.g., layers[10] output → hidden_states[11])
        # To match eval backbone_features (= hidden_states[-1]):
        #   Use index -1
        if self.layer_indices is not None:
            selected_layers = tuple(all_hidden_states[i] for i in self.layer_indices)
        else:
            selected_layers = all_hidden_states[-self.num_layers:]

        # Get masks
        image_mask = vl_input_filtered["input_ids"] == self.backbone.model.config.image_token_index
        attention_mask = vl_input_filtered["attention_mask"] == 1

        return {
            "all_layer_features": selected_layers,  # Tuple of [B, seq, hidden]
            "backbone_features": all_hidden_states[-1],  # Post-norm (= eval backbone_features)
            "backbone_attention_mask": attention_mask,
            "image_mask": image_mask,
            "total_layers": total_layers,
        }


def collect_episode_multilayer(
    policy,
    env,
    env_name: str,
    task_suite: str,
    episode_id: int,
    layer_indices: List[int] = None,
    num_layers: int = 4,
    max_steps: int = 600,
    device: str = "cuda:0",
    vision_capture: 'VisionFeatureCapture' = None,
) -> EpisodeDataMultiLayer:
    """Run one episode and collect multi-layer backbone features + vision features.

    Args:
        layer_indices: Indices into hidden_states tuple. Use K+1 for output of
                      transformer layer K (e.g., 11 for layer 10 output).
                      Use -1 for post-norm output (= backbone_features in eval).
        vision_capture: VisionFeatureCapture instance (reused across episodes)
    """
    from gr00t.data.types import MessageType, VLAStepData

    steps_data = []
    prev_action = None

    # Create multi-layer wrapper
    ml_wrapper = MultiLayerBackboneWrapper(
        policy.model.backbone,
        layer_indices=layer_indices,
        num_layers=num_layers
    )

    def unbatch_observation(value):
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

    def to_vla_step_data(observation):
        return VLAStepData(
            images=observation["video"],
            states=observation["state"],
            actions={},
            text=observation["language"][policy.language_key][0],
            embodiment=policy.embodiment_tag,
        )

    def rec_to_dtype(x, dtype):
        if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
            return x.to(dtype=dtype)
        elif isinstance(x, dict) or hasattr(x, "items"):
            return {k: rec_to_dtype(v, dtype) for k, v in x.items()}
        elif isinstance(x, list):
            return [rec_to_dtype(v, dtype) for v in x]
        else:
            return x

    # Run episode
    obs, _ = env.reset()
    done = False
    step = 0
    total_reward = 0.0
    success = False

    while not done and step < max_steps:
        # Prepare observation
        batched_obs = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                batched_obs[k] = v[np.newaxis, ...]
            elif isinstance(v, str):
                batched_obs[k] = (v,)
            else:
                batched_obs[k] = v

        new_obs = {}
        for modality in ["video", "state", "language"]:
            new_obs[modality] = {}
            for key in policy.modality_configs[modality].modality_keys:
                if modality == "language":
                    if key == "task" and "annotation.human.coarse_action" in batched_obs:
                        parsed_key = "annotation.human.coarse_action"
                    else:
                        parsed_key = key
                else:
                    parsed_key = f"{modality}.{key}"

                arr = batched_obs[parsed_key]
                if modality == "language":
                    new_obs[modality][key] = [[str(item)] for item in arr]
                else:
                    new_obs[modality][key] = arr

        unbatched_observations = unbatch_observation(new_obs)
        processed_inputs = []

        for obs_item in unbatched_observations:
            vla_step_data = to_vla_step_data(obs_item)
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(policy.processor(messages))

        collated_inputs = policy.collate_fn(processed_inputs)
        collated_inputs = rec_to_dtype(collated_inputs, dtype=torch.bfloat16)
        actual_inputs = collated_inputs["inputs"] if "inputs" in collated_inputs else collated_inputs

        # Run model and collect multi-layer features
        with torch.inference_mode():
            backbone_inputs, action_inputs = policy.model.prepare_input(actual_inputs)

            # Get multi-layer hidden states
            ml_outputs = ml_wrapper.forward_with_all_layers(backbone_inputs)

            # Get action using standard backbone output
            from transformers.feature_extraction_utils import BatchFeature
            backbone_outputs = BatchFeature(data={
                "backbone_features": ml_outputs["backbone_features"],
                "backbone_attention_mask": ml_outputs["backbone_attention_mask"],
                "image_mask": ml_outputs["image_mask"],
            })
            action_outputs = policy.model.action_head.get_action(backbone_outputs, action_inputs)

            action_pred = action_outputs.action_pred.detach().float().cpu().numpy().astype(np.float32)

            # Stack multi-layer features: [num_layers, seq, hidden]
            all_layer_feats = [layer.detach().float() for layer in ml_outputs["all_layer_features"]]
            multi_layer_features = torch.stack(all_layer_feats, dim=1
            ).squeeze(0).cpu().numpy().astype(np.float16)  # [num_layers, seq, hidden]

            # Verify: target (last selected layer) must match backbone_features
            if step == 0 and episode_id == 0:
                bb = ml_outputs["backbone_features"].detach().float().squeeze(0)
                tgt = all_layer_feats[-1].squeeze(0)
                cos = torch.nn.functional.cosine_similarity(
                    bb.reshape(1, -1), tgt.reshape(1, -1)).item()
                logger.info(f"[VERIFY] target layer vs backbone_features: cos={cos:.6f}"
                            f" norms: target={tgt.norm():.1f} bb={bb.norm():.1f}")
                assert cos > 0.999, (
                    f"Target layer does NOT match backbone_features (cos={cos:.4f})! "
                    f"Use layer_indices [-1] for target to get post-norm hidden states.")

            # Squeeze state: [B, padded_dim] → [state_dim]
            state_np = action_inputs["state"].detach().float().cpu().numpy().astype(np.float32)
            while state_np.ndim > 1:
                state_np = state_np[0]
            state_np = state_np[:8]

            # Squeeze action: [B, T, action_dim] → [action_dim]
            action_np = action_pred.copy()
            if action_np.ndim == 3:
                action_np = action_np[0, 0, :]
            elif action_np.ndim == 2:
                action_np = action_np[0, :]
            action_np = action_np[:7]

            step_data = {
                "step": step,
                "multi_layer_features": multi_layer_features,
                "backbone_features": ml_outputs["backbone_features"].detach().float().cpu().numpy().astype(np.float16),
                "backbone_attention_mask": ml_outputs["backbone_attention_mask"].detach().cpu().numpy(),
                "image_mask": ml_outputs["image_mask"].detach().cpu().numpy(),
                "state": state_np,
                "action_pred": action_np,
                "prev_action": (prev_action if prev_action is not None
                                else np.zeros_like(action_np)),
            }

            # Always capture vision encoder features (pre-LLM)
            if vision_capture is not None:
                vis_feats = vision_capture.get_and_reset()
                if vis_feats is not None:
                    step_data["vision_features"] = vis_feats.squeeze(0).float().cpu().numpy().astype(np.float16)

            steps_data.append(step_data)
            prev_action = action_np.copy()

        # Execute action
        normalized_action = action_outputs.action_pred.float()
        states = [obs_item["state"] for obs_item in unbatched_observations]
        batched_states = {}
        for k in policy.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)

        unnormalized_action = policy.processor.decode_action(
            normalized_action.cpu().numpy(), policy.embodiment_tag, batched_states
        )

        casted_action = {
            f"action.{key}": value.astype(np.float32)[0]
            for key, value in unnormalized_action.items()
        }

        obs, reward, terminated, truncated, info = env.step(casted_action)
        total_reward += reward
        done = terminated or truncated

        if terminated and reward > 0:
            success = True

        step += 1

    return EpisodeDataMultiLayer(
        episode_id=episode_id,
        task_name=env_name,
        task_suite=task_suite,
        steps=steps_data,
        success=success,
        total_reward=total_reward,
    )


def save_episode_multilayer(episode_data: EpisodeDataMultiLayer, hdf5_file: h5py.File):
    """Save episode data with multi-layer features to HDF5."""
    ep_grp = hdf5_file.create_group(f"episode_{episode_data.episode_id:04d}")

    ep_grp.attrs["task_name"] = episode_data.task_name
    ep_grp.attrs["task_suite"] = episode_data.task_suite
    ep_grp.attrs["success"] = episode_data.success
    ep_grp.attrs["total_reward"] = episode_data.total_reward
    ep_grp.attrs["num_steps"] = len(episode_data.steps)

    num_steps = len(episode_data.steps)
    if num_steps == 0:
        return

    # Stack step data
    multi_layer_features = np.stack([s["multi_layer_features"] for s in episode_data.steps], axis=0)
    backbone_features = np.stack([s["backbone_features"] for s in episode_data.steps], axis=0)
    backbone_attention_mask = np.stack([s["backbone_attention_mask"] for s in episode_data.steps], axis=0)
    image_mask = np.stack([s["image_mask"] for s in episode_data.steps], axis=0)
    states = np.stack([s["state"] for s in episode_data.steps], axis=0)
    actions = np.stack([s["action_pred"] for s in episode_data.steps], axis=0)
    prev_actions = np.stack([s["prev_action"] for s in episode_data.steps], axis=0)

    # Save with compression
    ep_grp.create_dataset("multi_layer_features", data=multi_layer_features, compression="gzip", compression_opts=4)
    ep_grp.create_dataset("backbone_features", data=backbone_features, compression="gzip", compression_opts=4)
    ep_grp.create_dataset("backbone_attention_mask", data=backbone_attention_mask, compression="gzip", compression_opts=4)
    ep_grp.create_dataset("image_mask", data=image_mask, compression="gzip", compression_opts=4)
    ep_grp.create_dataset("state", data=states, compression="gzip", compression_opts=4)
    ep_grp.create_dataset("action", data=actions, compression="gzip", compression_opts=4)
    ep_grp.create_dataset("prev_action", data=prev_actions, compression="gzip", compression_opts=4)

    # Vision encoder features (optional, collected with --save_vision)
    if "vision_features" in episode_data.steps[0]:
        vision_features = np.stack([s["vision_features"] for s in episode_data.steps], axis=0)
        ep_grp.create_dataset("vision_features", data=vision_features, compression="gzip", compression_opts=4)


def main():
    parser = argparse.ArgumentParser(description="Collect multi-layer VLM features")
    parser.add_argument("--model_path", type=str,
                        default="./outputs/libero_10_finetune")
    parser.add_argument("--output_path", type=str,
                        default="./outputs/latent_bridge_data/multilayer_train_data.h5")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of layers to collect (from last layer backwards, ignored if --layer_indices set)")
    parser.add_argument("--layer_indices", type=int, nargs="+", default=None,
                        help="Specific layer indices to collect (e.g., 7 14 21 28 for spread layers). "
                             "If not set, uses last --num_layers layers.")
    parser.add_argument("--n_episodes_per_task", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=720)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--task_suites", type=str, nargs="+", default=["libero_10"])
    parser.add_argument("--env_names", type=str, nargs="+", default=None,
                        help="Direct env names (bypass suite lookup). E.g., robocasa_panda_omron/OpenDrawer_PandaOmron_Env")
    parser.add_argument("--embodiment_tag", type=str, default="LIBERO_PANDA",
                        help="Embodiment tag for the model (LIBERO_PANDA, ROBOCASA_PANDA_OMRON, etc.)")
    parser.add_argument("--task_filter", type=str, default=None,
                        help="Substring filter for task names (e.g., 'KITCHEN_SCENE3' to run one task)")
    parser.add_argument("--save_vision", action="store_true",
                        help="Also save vision encoder features (pre-LLM, ~430M params output). "
                             "Required for training VisionConditionedBridge.")
    args = parser.parse_args()

    # Determine number of layers being collected
    if args.layer_indices is not None:
        actual_num_layers = len(args.layer_indices)
        layer_desc = f"layers {args.layer_indices}"
    else:
        actual_num_layers = args.num_layers
        layer_desc = f"last {args.num_layers} layers"

    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.eval.rollout_policy import WrapperConfigs, VideoConfig, MultiStepConfig, create_eval_env
    from gr00t.data.embodiment_tags import EmbodimentTag

    torch.cuda.set_device(args.device)

    # Register envs based on embodiment
    if "ROBOCASA" in args.embodiment_tag.upper():
        import robocasa  # noqa
    elif "OXE" in args.embodiment_tag.upper():
        from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs
        register_simpler_envs()
    else:
        from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs
        register_libero_envs()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    logger.info(f"Loading model from {args.model_path}...")
    embodiment = getattr(EmbodimentTag, args.embodiment_tag)
    policy = Gr00tPolicy(
        embodiment_tag=embodiment,
        model_path=args.model_path,
        device=args.device,
        strict=False,
    )
    logger.info(f"Model loaded. Collecting {layer_desc}.")

    # Vision feature capture (hooks into mlp1, fires during backbone forward)
    # Always enabled — needed for VisionConditionedBridge
    vision_capture = VisionFeatureCapture(policy.model.backbone)

    wrapper_configs = WrapperConfigs(
        video=VideoConfig(video_dir=None),
        multistep=MultiStepConfig(
            n_action_steps=args.n_action_steps,
            max_episode_steps=args.max_steps,
            terminate_on_success=True,
        ),
    )

    total_episodes = 0
    total_steps = 0
    successful_episodes = 0
    episode_id = 0

    with h5py.File(args.output_path, "w") as hdf5_file:
        hdf5_file.attrs["model_path"] = args.model_path
        hdf5_file.attrs["num_layers"] = actual_num_layers
        hdf5_file.attrs["layer_indices"] = json.dumps(args.layer_indices) if args.layer_indices else "last_n"
        hdf5_file.attrs["n_episodes_per_task"] = args.n_episodes_per_task
        hdf5_file.attrs["task_suites"] = json.dumps(args.task_suites)

        if args.env_names:
            all_env_names = args.env_names
            task_suite = "custom"
            if args.task_filter:
                all_env_names = [n for n in all_env_names if args.task_filter in n]
            logger.info(f"\n{'='*60}")
            logger.info(f"Collecting from {len(all_env_names)} envs (direct)")
            logger.info(f"{'='*60}")
        else:
            all_env_names = []
            task_suite = args.task_suites[0] if args.task_suites else "unknown"
            for ts in args.task_suites:
                if ts not in SUPPORTED_SUITES:
                    logger.warning(f"Unknown task suite: {ts}, skipping...")
                    continue
                task_suite = ts
                env_names = get_libero_task_names(ts)
                if args.task_filter:
                    env_names = [n for n in env_names if args.task_filter in n]
                all_env_names.extend(env_names)
            logger.info(f"\n{'='*60}")
            logger.info(f"Collecting from {len(all_env_names)} tasks")
            logger.info(f"{'='*60}")

        for env_name in all_env_names:
                logger.info(f"\nCollecting data for: {env_name}")

                try:
                    env = create_eval_env(env_name, env_idx=0, total_n_envs=1, wrapper_configs=wrapper_configs)
                except Exception as e:
                    logger.error(f"Failed to create env {env_name}: {e}")
                    continue

                task_success = 0
                for ep in tqdm(range(args.n_episodes_per_task), desc=f"Episodes"):
                    try:
                        episode_data = collect_episode_multilayer(
                            policy=policy,
                            env=env,
                            env_name=env_name,
                            task_suite=task_suite,
                            episode_id=episode_id,
                            layer_indices=args.layer_indices,
                            num_layers=args.num_layers,
                            max_steps=args.max_steps,
                            device=args.device,
                            vision_capture=vision_capture,
                        )

                        save_episode_multilayer(episode_data, hdf5_file)
                        hdf5_file.flush()  # Flush after each episode to prevent corruption

                        total_episodes += 1
                        total_steps += len(episode_data.steps)
                        if episode_data.success:
                            successful_episodes += 1
                            task_success += 1

                        episode_id += 1

                    except Exception as e:
                        logger.error(f"Episode {episode_id} failed: {e}")
                        import traceback
                        traceback.print_exc()
                        episode_id += 1
                        continue

                task_short = env_name.split("/")[-1][:50]
                logger.info(f"  Task SR: {task_success}/{args.n_episodes_per_task} "
                           f"({100*task_success/args.n_episodes_per_task:.0f}%) - {task_short}")

                # Close env with timeout to prevent hangs
                import threading
                close_thread = threading.Thread(target=env.close)
                close_thread.daemon = True
                close_thread.start()
                close_thread.join(timeout=30)
                if close_thread.is_alive():
                    logger.warning(f"env.close() timed out for {env_name}, continuing...")

        hdf5_file.attrs["total_episodes"] = total_episodes
        hdf5_file.attrs["total_steps"] = total_steps
        hdf5_file.attrs["successful_episodes"] = successful_episodes
        hdf5_file.flush()  # Final flush before context manager close

    print("\n" + "="*60)
    print("MULTI-LAYER DATA COLLECTION COMPLETE")
    print("="*60)
    print(f"Output file: {args.output_path}")
    print(f"Layers collected: {layer_desc}")
    print(f"Total episodes: {total_episodes}")
    print(f"Total steps: {total_steps}")
    print(f"Success rate: {100*successful_episodes/max(1,total_episodes):.1f}%")

    file_size = os.path.getsize(args.output_path) / (1024**3)
    print(f"File size: {file_size:.2f} GB")


if __name__ == "__main__":
    main()
