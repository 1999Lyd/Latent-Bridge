#!/usr/bin/env python3
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Eval π0 with KV Bridge: predict KV deltas directly, skip Gemma.

First step: full fresh Gemma → capture KV cache + embedding
Subsequent: SigLIP only → bridge predicts KV deltas → apply RoPE → denoise
"""
import collections, dataclasses, logging, math, pathlib, sys

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'baseline', 'openpi'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'baseline', 'openpi', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'baseline', 'openpi', 'third_party', 'libero'))

import numpy as np, torch
from transformers import DynamicCache
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
    bridge_path: str = "./outputs/pi0_bridge_kv/best_model.pt"
    device: str = "cuda:0"
    num_denoise_steps: int = 10
    replan_steps: int = 5
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 20
    seed: int = 7
    vlm_freq: int = 3
    phase_aware: bool = False
    nav_threshold: float = 0.3
    manip_threshold: float = 0.05
    nav_freq: int = 2
    trans_freq: int = 3
    manip_freq: int = 4

def _quat2axisangle(quat):
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


class KVBridgePolicy:
    def __init__(self, policy, bridge, num_denoise_steps=10, device='cuda:0',
                 vlm_freq=3, phase_aware=False,
                 nav_threshold=0.3, manip_threshold=0.05,
                 nav_freq=2, trans_freq=3, manip_freq=4,
                 no_delta=False):
        self.policy = policy
        self.model = policy._model
        self.bridge = bridge
        self.no_delta = no_delta
        self.num_denoise_steps = num_denoise_steps
        self.device = device
        self.lm = self.model.paligemma_with_expert.paligemma.language_model
        self.lm.config._attn_implementation = "eager"

        # Frequency control
        self.vlm_freq = vlm_freq
        self.phase_aware = phase_aware
        self.nav_threshold = nav_threshold
        self.manip_threshold = manip_threshold
        self.nav_freq = nav_freq
        self.trans_freq = trans_freq
        self.manip_freq = manip_freq
        self.infer_count = 0
        self.steps_since_vlm = 0

        # State
        self.prev_kv_preRoPE = None  # [18, S, 512] pre-RoPE K+V from prev step
        self.prev_embedding = None    # [S, 2048]
        self.raw_state = None
        self.raw_action = None
        self.n_fresh = 0
        self.n_bridge = 0
        self.fresh_times = []   # ms per fresh VLM step
        self.bridge_times = []  # ms per bridge step
        self.denoise_times = [] # ms per denoise (shared)

        # Monkey-patch
        self.model.sample_actions = self._sample
        policy._sample_actions = self._sample

    def reset(self):
        self.prev_kv_preRoPE = None
        self.prev_embedding = None
        self.raw_state = None
        self.raw_action = None
        self.infer_count = 0
        self.steps_since_vlm = 0

    def _should_use_vlm(self):
        """Decide whether to run fresh VLM based on freq or phase-aware scheduling."""
        if self.prev_kv_preRoPE is None:
            return True  # first step always fresh
        if self.phase_aware and self.raw_action is not None:
            trans_mag = self.raw_action[0, :3].norm().item()
            if trans_mag > self.nav_threshold:
                freq = self.nav_freq
            elif trans_mag < self.manip_threshold:
                freq = self.manip_freq
            else:
                freq = self.trans_freq
        else:
            freq = self.vlm_freq
        self.steps_since_vlm += 1
        return self.steps_since_vlm >= freq

    def _compute_preRoPE_kv(self, past_kv, prefix_pad_masks):
        """Extract pre-RoPE K and V from real KV cache by un-applying RoPE to K."""
        S = 768
        pos_ids = torch.cumsum(prefix_pad_masks, dim=1)[:, :S] - 1
        cos, sin = self.lm.rotary_emb(torch.zeros(1, S, 2048, device=self.device), pos_ids)
        c, s_ = cos.unsqueeze(1), sin.unsqueeze(1)

        preRoPE_kv = []
        for l in range(18):
            k_post = past_kv.key_cache[l][:, :, :S, :]   # [1, 1, S, 256] post-RoPE
            v = past_kv.value_cache[l][:, :, :S, :]       # [1, 1, S, 256]
            # Un-apply RoPE: k_pre = k_post * cos - rotate_half(k_post) * sin
            # Since RoPE: k_post = k_pre * cos + rotate_half(k_pre) * sin
            # Inverse: k_pre = k_post * cos - rotate_half(k_post) * sin
            hd = k_post.shape[-1]
            k1, k2 = k_post[..., :hd//2], k_post[..., hd//2:]
            k_pre = k_post * c - torch.cat((-k2, k1), dim=-1) * s_
            kv = torch.cat([k_pre.squeeze(0).squeeze(0), v.squeeze(0).squeeze(0)], dim=-1)  # [S, 512]
            preRoPE_kv.append(kv)
        return torch.stack(preRoPE_kv)  # [18, S, 512]

    def _build_kv_cache(self, preRoPE_kv, prefix_pad_masks, full_seq):
        """Build DynamicCache from pre-RoPE KV by applying RoPE to K."""
        S = 768
        pos_ids = torch.cumsum(prefix_pad_masks, dim=1)[:, :S] - 1
        cos, sin = self.lm.rotary_emb(torch.zeros(1, S, 2048, device=self.device), pos_ids)
        c, s_ = cos.unsqueeze(1), sin.unsqueeze(1)

        cache = DynamicCache()
        for l in range(18):
            kv = preRoPE_kv[l]  # [S, 512]
            k_pre = kv[:, :256].unsqueeze(0).unsqueeze(0)  # [1, 1, S, 256]
            v = kv[:, 256:].unsqueeze(0).unsqueeze(0)

            # Apply RoPE to K
            hd = 256
            k1, k2 = k_pre[..., :hd//2], k_pre[..., hd//2:]
            k_post = k_pre * c + torch.cat((-k2, k1), dim=-1) * s_

            # Cast to bfloat16 to match Gemma's expected dtype
            k_post = k_post.to(torch.bfloat16)
            v = v.to(torch.bfloat16)

            # Pad to full_seq if needed
            if S < full_seq:
                k_pad = torch.zeros(1, 1, full_seq - S, hd, dtype=k_post.dtype, device=self.device)
                v_pad = torch.zeros(1, 1, full_seq - S, hd, dtype=v.dtype, device=self.device)
                k_post = torch.cat([k_post, k_pad], dim=2)
                v = torch.cat([v, v_pad], dim=2)

            cache.key_cache.append(k_post)
            cache.value_cache.append(v)
        return cache

    def _sample(self, device, observation, noise=None, num_steps=None):
        if num_steps is None:
            num_steps = self.num_denoise_steps

        bsize = observation.state.shape[0]
        if noise is None:
            shape = (bsize, self.model.config.action_horizon, self.model.config.action_dim)
            noise = self.model.sample_noise(shape, device)

        images, img_masks, lang_tokens, lang_masks, state = \
            self.model._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = \
            self.model.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self.model._prepare_attention_masks_4d(prefix_att_2d_masks)

        S = 768
        full_seq = prefix_embs.shape[1]

        use_vlm = self._should_use_vlm()
        import time as _time

        if use_vlm:
            # Fresh VLM step
            self.steps_since_vlm = 0
            torch.cuda.synchronize()
            _t0 = _time.perf_counter()
            _, past_kv = self.model.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
            # Extract pre-RoPE KV for next bridge step
            self.prev_kv_preRoPE = self._compute_preRoPE_kv(past_kv, prefix_pad_masks).detach()
            self.prev_embedding = prefix_embs[0, :S].detach()
            torch.cuda.synchronize()
            self.fresh_times.append((_time.perf_counter() - _t0) * 1000)
            self.n_fresh += 1
        else:
            # Bridge step: predict KV deltas
            torch.cuda.synchronize()
            _t0 = _time.perf_counter()
            curr_emb = prefix_embs[0, :S].detach()
            bdtype = torch.bfloat16  # match compiled bridge dtype
            emb_delta = (curr_emb - self.prev_embedding).unsqueeze(0).to(bdtype)
            curr_emb_f = curr_emb.unsqueeze(0).to(bdtype)
            B_S = self.prev_kv_preRoPE.shape[1]
            prev_kv_flat = self.prev_kv_preRoPE.permute(1, 0, 2).reshape(1, B_S, -1).to(bdtype)
            s = (self.raw_state if self.raw_state is not None else torch.zeros(1, 8, device=device)).to(bdtype)
            a = (self.raw_action if self.raw_action is not None else torch.zeros(1, 7, device=device)).to(bdtype)

            with torch.no_grad():
                kv_deltas = self.bridge(emb_delta, curr_emb_f, prev_kv_flat, s, a)

            # Apply deltas (or use directly if no_delta mode)
            new_preRoPE = self.prev_kv_preRoPE.clone()
            for l in range(18):
                pred = kv_deltas[l][0].to(new_preRoPE.dtype)  # [S, 512]
                if self.no_delta:
                    new_preRoPE[l] = pred  # full KV prediction, no residual
                else:
                    new_preRoPE[l] = new_preRoPE[l] + pred  # delta prediction

            # Build KV cache from predicted pre-RoPE KV
            past_kv = self._build_kv_cache(new_preRoPE, prefix_pad_masks, full_seq)

            # Update state
            self.prev_kv_preRoPE = new_preRoPE.detach()
            self.prev_embedding = curr_emb.detach()
            torch.cuda.synchronize()
            self.bridge_times.append((_time.perf_counter() - _t0) * 1000)
            self.n_bridge += 1

        # Denoising
        torch.cuda.synchronize()
        _td0 = _time.perf_counter()
        dt = -1.0 / num_steps
        dt_t = torch.tensor(dt, dtype=torch.float32, device=device)
        x_t = noise
        time_val = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time_val >= -dt_t / 2:
            v_t = self.model.denoise_step(
                state, prefix_pad_masks, past_kv, x_t, time_val.expand(bsize))
            x_t = x_t + dt_t * v_t
            time_val += dt_t
        torch.cuda.synchronize()
        self.denoise_times.append((_time.perf_counter() - _td0) * 1000)
        return x_t


def main(args: Args):
    np.random.seed(args.seed)
    torch.cuda.set_device(args.device)

    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    cfg = _config.get_config("pi05_libero")
    policy = _policy_config.create_trained_policy(cfg, args.checkpoint_dir, pytorch_device=args.device,
        sample_kwargs={"num_steps": args.num_denoise_steps})

    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
    from pi0_bridge_kv import Pi0BridgeKV
    ckpt = torch.load(args.bridge_path, map_location=args.device, weights_only=False)
    bcfg = ckpt['config']
    no_vision = bcfg.get('no_vision', False) or getattr(args, 'no_vision', False)
    no_state = bcfg.get('no_state', False)
    no_action = bcfg.get('no_action', False)
    bridge = Pi0BridgeKV(
        kv_dim=256, num_layers=18, seq_len=768, emb_dim=2048,
        hidden_dim=bcfg['hidden_dim'], num_heads=12, num_blocks=bcfg['num_blocks'],
        state_dim=8, action_dim=7, no_vision=no_vision,
        no_state=no_state, no_action=no_action,
    ).to(args.device).eval()
    bridge.load_state_dict(ckpt['model_state_dict'])
    # Optimize: bfloat16 + torch.compile for flash attention speedup (3.6ms vs 16.6ms)
    bridge = bridge.to(torch.bfloat16)
    bridge = torch.compile(bridge, mode="max-autotune")
    logger.info("KV Bridge: val_cos=%.4f epoch=%d (compiled, bf16)", ckpt['val_cos'], ckpt['epoch'])

    no_delta = bcfg.get('no_delta', False)
    bkv = KVBridgePolicy(policy, bridge, args.num_denoise_steps, args.device,
                          vlm_freq=args.vlm_freq, phase_aware=args.phase_aware,
                          nav_threshold=args.nav_threshold, manip_threshold=args.manip_threshold,
                          nav_freq=args.nav_freq, trans_freq=args.trans_freq, manip_freq=args.manip_freq,
                          no_delta=no_delta)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    max_steps_map = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
    max_steps = max_steps_map.get(args.task_suite_name, 400)

    total_ep, total_ok = 0, 0
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

            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1; continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
                    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))
                    raw_st = np.concatenate((obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]),
                                             obs["robot0_gripper_qpos"])).astype(np.float32)

                    if not action_plan:
                        element = {"observation/image": img, "observation/wrist_image": wrist,
                                   "observation/state": raw_st, "prompt": str(desc)}
                        chunk = policy.infer(element)["actions"]
                        action_plan.extend(chunk[:args.replan_steps])
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

        env.close()
        logger.info("  %s: %d/%d", desc[:40], task_ok, args.num_trials_per_task)

    logger.info("\nTotal: %d/%d = %.1f%%", total_ok, total_ep, total_ok / max(total_ep, 1) * 100)
    logger.info("Fresh: %d, Bridge: %d", bkv.n_fresh, bkv.n_bridge)

    # Latency report (skip first few for JIT warmup)
    skip = 5
    if len(bkv.fresh_times) > skip:
        ft = bkv.fresh_times[skip:]
        logger.info("\nLatency (ms):")
        logger.info("  VLM fresh:  mean=%.1f  median=%.1f  p95=%.1f  (n=%d)",
                    np.mean(ft), np.median(ft), np.percentile(ft, 95), len(ft))
    if len(bkv.bridge_times) > skip:
        bt = bkv.bridge_times[skip:]
        logger.info("  Bridge:     mean=%.1f  median=%.1f  p95=%.1f  (n=%d)",
                    np.mean(bt), np.median(bt), np.percentile(bt, 95), len(bt))
    if len(bkv.denoise_times) > skip:
        dt = bkv.denoise_times[skip:]
        logger.info("  Denoise:    mean=%.1f  median=%.1f  p95=%.1f  (n=%d)",
                    np.mean(dt), np.median(dt), np.percentile(dt, 95), len(dt))
    if len(bkv.fresh_times) > skip and len(bkv.bridge_times) > skip:
        ft_med = np.median(bkv.fresh_times[skip:])
        bt_med = np.median(bkv.bridge_times[skip:])
        dt_med = np.median(bkv.denoise_times[skip:]) if len(bkv.denoise_times) > skip else 0
        logger.info("  VLM total (prefix+denoise): %.1f ms", ft_med + dt_med)
        logger.info("  Bridge total (bridge+denoise): %.1f ms", bt_med + dt_med)
        logger.info("  Speedup: %.2fx", (ft_med + dt_med) / max(bt_med + dt_med, 0.1))

if __name__ == "__main__":
    main(tyro.cli(Args))
