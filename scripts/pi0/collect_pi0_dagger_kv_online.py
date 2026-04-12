#!/usr/bin/env python3
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Online DAgger for π0 KV bridge. Uses KVBridgePolicy (same as eval) to run
bridge in LIBERO env. After each inference, also runs oracle Gemma for GT KV.

Robot follows BRIDGE actions (bridge distribution). Oracle provides KV labels.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/collect_pi0_dagger_kv_online.py \
        --bridge_path ./outputs/pi0_bridge_kv_libero10/best_model.pt \
        --output_path ./outputs/pi0_bridge_data/dagger_kv_libero10_online.h5 \
        --task_suite_name libero_10 --num_trials_per_task 30 --vlm_freq 3
"""
import collections, dataclasses, logging, math, pathlib, sys, os

sys.path.insert(0, PROJECT_ROOT)
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
    bridge_path: str = ""
    output_path: str = ""
    device: str = "cuda:0"
    num_denoise_steps: int = 2
    replan_steps: int = 5
    vlm_freq: int = 3
    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    num_trials_per_task: int = 30
    seed: int = 7

def _quat2axisangle(quat):
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def compute_preRoPE_kv(hs, lm, S, device):
    """Compute pre-RoPE K+V from hidden states."""
    kv = np.zeros((18, S, 512), dtype=np.float16)
    with torch.no_grad():
        for l in range(18):
            h = hs[l][:, :S, :].to(lm.layers[0].self_attn.k_proj.weight.dtype)
            normed = lm.layers[l].input_layernorm(h)
            if isinstance(normed, tuple): normed = normed[0]
            k = lm.layers[l].self_attn.k_proj(normed)
            v = lm.layers[l].self_attn.v_proj(normed)
            kv[l] = torch.cat([k, v], dim=-1)[0].cpu().float().numpy().astype(np.float16)
    return kv

def main(args: Args):
    np.random.seed(args.seed)
    torch.cuda.set_device(args.device)
    S = 768

    # Load policy
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    cfg = _config.get_config("pi05_libero")
    policy = _policy_config.create_trained_policy(cfg, args.checkpoint_dir, pytorch_device=args.device,
        sample_kwargs={"num_steps": args.num_denoise_steps})
    model = policy._model
    lm = model.paligemma_with_expert.paligemma.language_model
    lm.config._attn_implementation = "eager"

    # Load bridge and wrap with KVBridgePolicy (same as eval)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
    from pi0_bridge_kv import Pi0BridgeKV
    from eval_pi0_bridge_kv import KVBridgePolicy

    ckpt = torch.load(args.bridge_path, map_location=args.device, weights_only=False)
    bcfg = ckpt['config']
    bridge = Pi0BridgeKV(kv_dim=256, num_layers=18, seq_len=768, emb_dim=2048,
        hidden_dim=bcfg['hidden_dim'], num_heads=12, num_blocks=bcfg['num_blocks'],
        state_dim=8, action_dim=7).to(args.device).eval()
    bridge.load_state_dict(ckpt['model_state_dict'])
    bridge = bridge.to(torch.bfloat16)
    logger.info("Bridge loaded: val_cos=%.4f", ckpt['val_cos'])

    # Wrap policy with bridge (handles freq, KV reconstruction, denoising)
    bkv = KVBridgePolicy(policy, bridge, args.num_denoise_steps, args.device,
                          vlm_freq=args.vlm_freq)

    # For oracle calls: run Gemma forward directly (bypass bridge patching)
    # The bridge's LM hook already captures hidden states on VLM steps.
    # For bridge steps, we run a separate oracle Gemma forward.
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
    import jax
    from openpi.models import model as _model

    # LIBERO setup
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    max_steps_map = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
    max_steps = max_steps_map.get(args.task_suite_name, 400)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    total_ep, total_ok, total_samples = 0, 0, 0
    episode_id = 0

    with h5py.File(args.output_path, 'w') as hf:
        for task_id in tqdm.tqdm(range(task_suite.n_tasks), desc="Tasks"):
            task = task_suite.get_task(task_id)
            init_states = task_suite.get_task_init_states(task_id)
            desc = task.language
            bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
            env.seed(args.seed)

            task_ok = 0
            for ei in tqdm.tqdm(range(args.num_trials_per_task), desc=desc[:40]):
                env.reset()
                action_plan = collections.deque()
                obs = env.set_init_state(init_states[ei])
                t = 0
                bkv.reset()
                ep_kv_inputs = []
                ep_kv_oracles = []
                ep_embs = []
                ep_states = []
                ep_actions = []
                ep_is_vlm = []

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
                            # Record what bridge state is BEFORE this inference
                            was_vlm = bkv._should_use_vlm()
                            # Reset the counter since _should_use_vlm incremented it
                            if not was_vlm:
                                bkv.steps_since_vlm -= 1  # undo the increment

                            # Capture bridge's current KV (before inference)
                            if bkv.prev_kv_preRoPE is not None:
                                input_kv = bkv.prev_kv_preRoPE.cpu().numpy().astype(np.float16)
                            else:
                                input_kv = None

                            # Run bridge policy (handles VLM/bridge internally)
                            element = {"observation/image": img, "observation/wrist_image": wrist,
                                       "observation/state": raw_st, "prompt": str(desc)}
                            chunk = policy.infer(element)["actions"]
                            action_plan.extend(chunk[:args.replan_steps])

                            # Run oracle: full Gemma forward on same observation
                            # Build observation and run prefix directly
                            inputs = policy._input_transform({**element})
                            inputs_t = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(args.device)[None, ...], inputs)
                            obs_t = _model.Observation.from_dict(inputs_t)
                            o_images, o_img_masks, o_lang_tokens, o_lang_masks, o_state = \
                                model._preprocess_observation(obs_t, train=False)
                            o_prefix_embs, o_prefix_pad_masks, o_prefix_att_masks = \
                                model.embed_prefix(o_images, o_img_masks, o_lang_tokens, o_lang_masks)
                            o_att_2d = make_att_2d_masks(o_prefix_pad_masks, o_prefix_att_masks)
                            o_pos = torch.cumsum(o_prefix_pad_masks, dim=1) - 1
                            o_att_4d = model._prepare_attention_masks_4d(o_att_2d)

                            # Run full Gemma (oracle) — this triggers the hidden state hook
                            bkv._captured = None
                            with torch.no_grad():
                                _, _ = model.paligemma_with_expert.forward(
                                    attention_mask=o_att_4d, position_ids=o_pos,
                                    past_key_values=None, inputs_embeds=[o_prefix_embs, None],
                                    use_cache=True)

                            if bkv._captured is not None:
                                oracle_kv = compute_preRoPE_kv(bkv._captured, lm, S, args.device)
                                oracle_emb = bkv._captured[0][0, :S].clone().cpu().float().numpy().astype(np.float16)
                            else:
                                oracle_kv = bkv.prev_kv_preRoPE.cpu().numpy().astype(np.float16) if bkv.prev_kv_preRoPE is not None else np.zeros((18, S, 512), dtype=np.float16)
                                oracle_emb = np.zeros((S, 2048), dtype=np.float16)

                            # Determine if this was a VLM or bridge step
                            is_vlm = (input_kv is None)  # first step is always VLM

                            # For VLM steps, input_kv = oracle_kv (clean pair)
                            if input_kv is None:
                                input_kv = oracle_kv

                            ep_kv_inputs.append(input_kv)
                            ep_kv_oracles.append(oracle_kv)
                            ep_embs.append(oracle_emb)
                            ep_states.append(raw_st)
                            ep_actions.append(chunk[0][:7].astype(np.float32))
                            ep_is_vlm.append(is_vlm)

                            bkv.raw_state = torch.from_numpy(raw_st).unsqueeze(0).to(args.device)
                            bkv.raw_action = torch.from_numpy(chunk[0][:7].astype(np.float32)).unsqueeze(0).to(args.device)

                        action = action_plan.popleft()
                        obs, reward, done, info = env.step(action.tolist())
                        if done:
                            task_ok += 1; total_ok += 1; break
                        t += 1
                    except Exception as e:
                        logger.error("Exception: %s", e)
                        import traceback; traceback.print_exc()
                        break

                total_ep += 1

                if ep_kv_inputs and len(ep_kv_inputs) > 1:
                    g = hf.create_group(f"episode_{episode_id:04d}")
                    g.create_dataset('kv', data=np.stack(ep_kv_inputs), compression='gzip', compression_opts=1)
                    g.create_dataset('oracle_kv', data=np.stack(ep_kv_oracles), compression='gzip', compression_opts=1)
                    g.create_dataset('embedding', data=np.stack(ep_embs), compression='gzip', compression_opts=1)
                    g.create_dataset('state', data=np.stack(ep_states))
                    g.create_dataset('action', data=np.stack(ep_actions))
                    g.create_dataset('is_vlm', data=np.array(ep_is_vlm))
                    g.attrs['success'] = done if isinstance(done, bool) else False
                    g.attrs['task_name'] = task.name
                    episode_id += 1
                    total_samples += len(ep_kv_inputs)
                    hf.flush()

            env.close()
            logger.info("  %s: %d/%d", desc[:40], task_ok, args.num_trials_per_task)

        hf.attrs['total_episodes'] = total_ep
        hf.attrs['total_successes'] = total_ok
        hf.attrs['total_samples'] = total_samples
        hf.attrs['vlm_freq'] = args.vlm_freq

    logger.info("\nTotal: %d/%d = %.1f%%, %d samples", total_ok, total_ep,
                total_ok / max(total_ep, 1) * 100, total_samples)

if __name__ == "__main__":
    main(tyro.cli(Args))
