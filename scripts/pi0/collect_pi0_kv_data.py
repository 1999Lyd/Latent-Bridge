#!/usr/bin/env python3
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Collect KV cache data from π0 on LIBERO (saves pre-RoPE K+V directly).

Runs π0 in sync mode, captures hidden states via hook, computes KV
using frozen Gemma weights, saves KV + embedding per inference step.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/collect_pi0_kv_data.py \
        --output_path ./outputs/pi0_bridge_data/kv_object.h5 \
        --task_suite_name libero_object --num_trials_per_task 30
"""
import collections, dataclasses, logging, math, pathlib, sys, os

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'baseline', 'openpi'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'baseline', 'openpi', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'baseline', 'openpi', 'third_party', 'libero'))

import h5py, numpy as np, torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
import tqdm, tyro

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

@dataclasses.dataclass
class Args:
    checkpoint_dir: str = "./checkpoints/pi05_base"
    output_path: str = "./outputs/pi0_bridge_data/kv_object.h5"
    device: str = "cuda:0"
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_object"
    num_steps_wait: int = 10
    num_trials_per_task: int = 30
    seed: int = 7

def _quat2axisangle(quat):
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def main(args: Args):
    np.random.seed(args.seed)
    torch.cuda.set_device(args.device)
    S = 768  # image tokens

    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    cfg = _config.get_config("pi05_libero")
    policy = _policy_config.create_trained_policy(cfg, args.checkpoint_dir, pytorch_device=args.device,
        sample_kwargs={"num_steps": 10})

    model = policy._model
    lm = model.paligemma_with_expert.paligemma.language_model
    lm.config._attn_implementation = "eager"

    # Hook to capture hidden states
    captured_hs = [None]
    orig_fwd = lm.forward
    def hook(*a, **kw):
        kw['output_hidden_states'] = True
        out = orig_fwd(*a, **kw)
        captured_hs[0] = out.hidden_states
        return out
    lm.forward = hook

    # Setup LIBERO
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    max_steps_map = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
    max_steps = max_steps_map.get(args.task_suite_name, 400)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    total_episodes, total_successes = 0, 0
    episode_id = 0

    with h5py.File(args.output_path, 'w') as hf:
        for task_id in tqdm.tqdm(range(task_suite.n_tasks), desc="Tasks"):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            task_description = task.language
            task_bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(bddl_file_name=task_bddl, camera_heights=256, camera_widths=256)
            env.seed(args.seed)

            task_successes = 0
            for ep_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=task_description[:40]):
                env.reset()
                action_plan = collections.deque()
                obs = env.set_init_state(initial_states[ep_idx])
                t = 0

                ep_kv = []      # [n_infer, 18, S, 512]
                ep_emb = []     # [n_infer, S, 2048]
                ep_states = []  # [n_steps, 8]
                ep_actions = [] # [n_steps, 7]

                while t < max_steps + args.num_steps_wait:
                    try:
                        if t < args.num_steps_wait:
                            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                            t += 1; continue

                        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
                        wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))

                        raw_st = np.concatenate((obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"])).astype(np.float32)

                        if not action_plan:
                            element = {"observation/image": img, "observation/wrist_image": wrist,
                                       "observation/state": raw_st, "prompt": str(task_description)}

                            captured_hs[0] = None
                            result = policy.infer(element)
                            action_chunk = result["actions"]
                            action_plan.extend(action_chunk[:args.replan_steps])

                            # Compute KV from captured hidden states
                            if captured_hs[0] is not None:
                                hs = captured_hs[0]
                                emb = hs[0][0, :S].cpu().float().numpy().astype(np.float16)
                                ep_emb.append(emb)

                                kv_layer = np.zeros((18, S, 512), dtype=np.float16)
                                with torch.no_grad():
                                    for l in range(18):
                                        h = hs[l][:, :S, :].to(lm.layers[0].self_attn.k_proj.weight.dtype)
                                        normed = lm.layers[l].input_layernorm(h)
                                        if isinstance(normed, tuple): normed = normed[0]
                                        k = lm.layers[l].self_attn.k_proj(normed)
                                        v = lm.layers[l].self_attn.v_proj(normed)
                                        kv_layer[l] = torch.cat([k, v], dim=-1)[0].cpu().float().numpy().astype(np.float16)
                                ep_kv.append(kv_layer)

                        action = action_plan.popleft()
                        ep_states.append(raw_st)
                        ep_actions.append(action[:7].astype(np.float32))

                        obs, reward, done, info = env.step(action.tolist())
                        if done:
                            task_successes += 1; total_successes += 1; break
                        t += 1
                    except Exception as e:
                        logger.error("Exception: %s", e)
                        break

                total_episodes += 1
                success = done if isinstance(done, bool) else False

                if ep_kv and ep_states:
                    g = hf.create_group(f"episode_{episode_id:04d}")
                    g.create_dataset('kv', data=np.stack(ep_kv), compression='gzip', compression_opts=1)
                    g.create_dataset('embedding', data=np.stack(ep_emb), compression='gzip', compression_opts=1)
                    g.create_dataset('state', data=np.stack(ep_states))
                    g.create_dataset('action', data=np.stack(ep_actions))
                    g.attrs['success'] = success
                    g.attrs['task_name'] = task.name
                    g.attrs['task_description'] = task_description
                    g.attrs['n_steps'] = len(ep_states)
                    g.attrs['n_inferences'] = len(ep_kv)
                    episode_id += 1
                    hf.flush()

            env.close()
            logger.info("  %s: %d/%d", task_description[:40], task_successes, args.num_trials_per_task)

        hf.attrs['total_episodes'] = total_episodes
        hf.attrs['total_successes'] = total_successes
        hf.attrs['success_rate'] = total_successes / max(total_episodes, 1)
        hf.attrs['task_suite'] = args.task_suite_name

    logger.info("\nTotal: %d/%d = %.1f%%", total_successes, total_episodes,
                total_successes / max(total_episodes, 1) * 100)
    logger.info("Saved to: %s", args.output_path)

if __name__ == "__main__":
    main(tyro.cli(Args))
