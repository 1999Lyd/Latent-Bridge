"""Custom KV-bridge closed-loop server (openpi venv). Implements skip-VLM inference
with a switchable FILL mode: sync (VLM every step), cache (reuse stale KV),
taylor (causal finite-difference extrapolation of past VLM KVs), bridge (learned).

Client selects mode + f + episode reset via sentinel keys in the observation:
  {"__reset__": True, "__mode__": "taylor", "__f__": 4}   -> reset state, set mode
Adapted from Latent-Bridge eval_pi0_bridge_kv.py (KVBridgePolicy)."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import argparse, pathlib, sys
OPENPI = "/path/to/study/std_openpi"
sys.path.insert(0, os.path.join(OPENPI, "src"))
sys.path.insert(0, "/path/to/Latent-Bridge/scripts/pi0")
import types as _types
import lerobot.datasets.lerobot_dataset as _ld
_c = _types.ModuleType("lerobot.common"); _d = _types.ModuleType("lerobot.common.datasets")
_c.datasets = _d
sys.modules["lerobot.common"] = _c
sys.modules["lerobot.common.datasets"] = _d
sys.modules["lerobot.common.datasets.lerobot_dataset"] = _ld
sys.path.insert(0, "/path/to/Latent-Bridge/scripts/pi0")

import numpy as np
import torch
from transformers import DynamicCache

S = 768
N_LAYERS = 18


class KVBridgeInference:
    def __init__(self, policy, bridge, device, num_denoise_steps=10):
        self.policy = policy
        self.model = policy._model
        self.bridge = bridge
        self.device = device
        self.num_denoise_steps = num_denoise_steps
        self.lm = self.model.paligemma_with_expert.paligemma.language_model
        self.lm.config._attn_implementation = "eager"
        self.mode = "sync"
        self.f = 1
        self.fixed_noise_seed = None  # per-request seed for reproducible flow noise
        self.reset()
        self.model.sample_actions = self._sample
        policy._sample_actions = self._sample

    def reset(self):
        self.prev_kv_preRoPE = None
        self.prev_embedding = None
        self.raw_state = None
        self.raw_action = None
        self.steps_since_vlm = 0
        self.kv_history = []  # past VLM-step pre-RoPE KVs (most recent last)

    def _should_use_vlm(self):
        if self.prev_kv_preRoPE is None:
            return True
        if self.mode == "sync":
            return True
        self.steps_since_vlm += 1
        return self.steps_since_vlm >= self.f

    def _compute_preRoPE_kv(self, past_kv, prefix_pad_masks):
        pos_ids = torch.cumsum(prefix_pad_masks, dim=1)[:, :S] - 1
        cos, sin = self.lm.rotary_emb(torch.zeros(1, S, 2048, device=self.device), pos_ids)
        c, s_ = cos.unsqueeze(1), sin.unsqueeze(1)
        out = []
        for l in range(N_LAYERS):
            k_post = past_kv.key_cache[l][:, :, :S, :]
            v = past_kv.value_cache[l][:, :, :S, :]
            hd = k_post.shape[-1]
            k1, k2 = k_post[..., :hd // 2], k_post[..., hd // 2:]
            k_pre = k_post * c - torch.cat((-k2, k1), dim=-1) * s_
            out.append(torch.cat([k_pre.squeeze(0).squeeze(0), v.squeeze(0).squeeze(0)], dim=-1))
        return torch.stack(out)  # [18,S,512]

    def _build_kv_cache(self, preRoPE_kv, prefix_pad_masks, full_seq):
        pos_ids = torch.cumsum(prefix_pad_masks, dim=1)[:, :S] - 1
        cos, sin = self.lm.rotary_emb(torch.zeros(1, S, 2048, device=self.device), pos_ids)
        c, s_ = cos.unsqueeze(1), sin.unsqueeze(1)
        cache = DynamicCache()
        for l in range(N_LAYERS):
            kv = preRoPE_kv[l]
            k_pre = kv[:, :256].unsqueeze(0).unsqueeze(0)
            v = kv[:, 256:].unsqueeze(0).unsqueeze(0)
            hd = 256
            k1, k2 = k_pre[..., :hd // 2], k_pre[..., hd // 2:]
            k_post = (k_pre * c + torch.cat((-k2, k1), dim=-1) * s_).to(torch.bfloat16)
            v = v.to(torch.bfloat16)
            if S < full_seq:
                kp = torch.zeros(1, 1, full_seq - S, hd, dtype=k_post.dtype, device=self.device)
                vp = torch.zeros(1, 1, full_seq - S, hd, dtype=v.dtype, device=self.device)
                k_post = torch.cat([k_post, kp], dim=2)
                v = torch.cat([v, vp], dim=2)
            cache.key_cache.append(k_post)
            cache.value_cache.append(v)
        return cache

    def _taylor_fill(self, k):
        """Causal Taylor extrapolation of pre-RoPE KV at offset k from past VLM KVs."""
        H = self.kv_history
        F0 = H[-1]
        if len(H) < 2:
            return F0.clone()
        d1 = F0 - H[-2]
        out = F0 + (k / self.f) * d1
        if len(H) >= 3:
            d2 = F0 - 2 * H[-2] + H[-3]
            out = out + (k * k / (2.0 * self.f * self.f)) * d2
        return out

    @torch.no_grad()
    def _sample(self, device, observation, noise=None, num_steps=None):
        if num_steps is None:
            num_steps = self.num_denoise_steps
        bsize = observation.state.shape[0]
        if noise is None:
            shape = (bsize, self.model.config.action_horizon, self.model.config.action_dim)
            if self.fixed_noise_seed is not None:
                g = torch.Generator().manual_seed(int(self.fixed_noise_seed))
                noise = torch.randn(shape, generator=g, dtype=torch.float32).to(device)
                self.fixed_noise_seed = None
            else:
                noise = self.model.sample_noise(shape, device)
        images, img_masks, lang_tokens, lang_masks, state = \
            self.model._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = \
            self.model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        a2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
        a4d = self.model._prepare_attention_masks_4d(a2d)
        full_seq = prefix_embs.shape[1]

        if self._should_use_vlm():
            self.steps_since_vlm = 0
            _, past_kv = self.model.paligemma_with_expert.forward(
                attention_mask=a4d, position_ids=pos, past_key_values=None,
                inputs_embeds=[prefix_embs, None], use_cache=True)
            self.prev_kv_preRoPE = self._compute_preRoPE_kv(past_kv, prefix_pad_masks).detach()
            self.prev_embedding = prefix_embs[0, :S].detach()
            self.kv_history.append(self.prev_kv_preRoPE.clone())
            if len(self.kv_history) > 3:
                self.kv_history.pop(0)
        else:
            k = self.steps_since_vlm
            if self.mode == "cache":
                new_preRoPE = self.prev_kv_preRoPE.clone()
            elif self.mode == "taylor":
                new_preRoPE = self._taylor_fill(k)
            else:  # bridge
                curr_emb = prefix_embs[0, :S].detach()
                bd = torch.bfloat16
                emb_delta = (curr_emb - self.prev_embedding).unsqueeze(0).to(bd)
                curr_emb_f = curr_emb.unsqueeze(0).to(bd)
                flat = self.prev_kv_preRoPE.permute(1, 0, 2).reshape(1, S, -1).to(bd)
                st = (self.raw_state if self.raw_state is not None else torch.zeros(1, 8, device=device)).to(bd)
                ac = (self.raw_action if self.raw_action is not None else torch.zeros(1, 7, device=device)).to(bd)
                deltas = self.bridge(emb_delta, curr_emb_f, flat, st, ac)
                new_preRoPE = self.prev_kv_preRoPE.clone()
                for l in range(N_LAYERS):
                    new_preRoPE[l] = new_preRoPE[l] + deltas[l][0].to(new_preRoPE.dtype)
                self.prev_embedding = curr_emb.detach()
            self.prev_kv_preRoPE = new_preRoPE.detach()
            past_kv = self._build_kv_cache(new_preRoPE, prefix_pad_masks, full_seq)

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        tv = torch.tensor(1.0, dtype=torch.float32, device=device)
        while tv >= -dt / 2:
            v_t = self.model.denoise_step(state, prefix_pad_masks, past_kv, x_t, tv.expand(bsize))
            x_t = x_t + dt * v_t
            tv += dt
        return x_t


class ResetablePolicy:
    """Wraps the openpi Policy; handles reset/mode sentinels then delegates infer."""
    def __init__(self, policy, kvb):
        self.policy = policy
        self.kvb = kvb
        self.metadata = policy.metadata

    def infer(self, obs, **kw):
        if isinstance(obs, dict) and obs.get("__reset__"):
            self.kvb.reset()
            self.kvb.mode = obs.get("__mode__", "sync")
            self.kvb.f = int(obs.get("__f__", 1))
            return {"actions": np.zeros((1, 7), np.float32), "reset_ok": np.array([1])}
        if "__seed__" in obs:
            self.kvb.fixed_noise_seed = int(np.asarray(obs["__seed__"]))
        out = self.policy.infer({k: v for k, v in obs.items()
                                 if not str(k).startswith("__")})
        # stash state/action for the bridge conditioning
        try:
            st = np.asarray(obs["observation/state"], np.float32)[:8]
            self.kvb.raw_state = torch.from_numpy(st).unsqueeze(0).to(self.kvb.device)
            a0 = np.asarray(out["actions"])[0, :7].astype(np.float32)
            self.kvb.raw_action = torch.from_numpy(a0).unsqueeze(0).to(self.kvb.device)
        except Exception:
            pass
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_dir", default="/path/to/study/serve_base")
    ap.add_argument("--norm", default="/path/to/study/assets/libero")
    ap.add_argument("--bridge", default="/path/to/study/out/bridge_sim.pt")
    ap.add_argument("--num_denoise_steps", type=int, default=10)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.serving import websocket_policy_server
    import openpi.shared.normalize as _normalize
    from pi0_bridge_kv import Pi0BridgeKV

    # The server's infer() is synchronous and blocks the asyncio loop; under heavy node
    # load the first inference exceeds the 20s keepalive ping timeout and the connection
    # drops (1011). Disable pings so long blocking inferences don't kill the connection.
    import websockets.asyncio.server as _wss
    _orig_serve = _wss.serve
    def _serve_noping(*a, **k):
        k.setdefault("ping_interval", None)
        k.setdefault("ping_timeout", None)
        return _orig_serve(*a, **k)
    _wss.serve = _serve_noping

    cfg = _config.get_config("pi05_libero")
    norm_stats = _normalize.load(pathlib.Path(args.norm))
    print("loading base policy...", flush=True)
    policy = _policy_config.create_trained_policy(
        cfg, args.policy_dir, norm_stats=norm_stats, pytorch_device=args.device,
        sample_kwargs={"num_steps": args.num_denoise_steps})

    print("loading bridge...", flush=True)
    ckpt = torch.load(args.bridge, map_location=args.device, weights_only=False)
    sd = ckpt.get("model", ckpt.get("model_state_dict"))
    bridge = Pi0BridgeKV(kv_dim=256, num_layers=18, seq_len=768, emb_dim=2048,
                         hidden_dim=768, num_heads=12, num_blocks=10,
                         state_dim=8, action_dim=7).to(args.device).eval()
    bridge.load_state_dict(sd)
    bridge = bridge.to(torch.bfloat16)  # match bfloat16 inputs (as in the paper's eval)

    kvb = KVBridgeInference(policy, bridge, args.device, args.num_denoise_steps)
    wrapped = ResetablePolicy(policy, kvb)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped, host="0.0.0.0", port=args.port, metadata=policy.metadata)
    print(f"KVBRIDGE_SERVER_READY port {args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
