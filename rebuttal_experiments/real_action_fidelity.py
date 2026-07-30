"""Per-step ACTION fidelity under each KV-fill mode, on BOTH sim (LIBERO) and real
(DROID) episode streams, with the identical env-free protocol (pi0.5 base, DROID input
transform; the only variable between arms is the pixel content -- same design as
kv_common / the KV-fidelity numbers).

For each held-out episode (same val split as delta_fidelity: seed 42, 25%):
  sync pass  : full VLM every step, flow noise seeded per step -> reference actions
  mode passes: cache / taylor / bridge at f=4, SAME per-step noise seeds
Metric: RMSE between the mode's action chunk (first 5 x 7 dims) and the sync
reference, split into VLM steps (sanity, expect 0) and skip steps (feature-induced).
"""
import argparse, json, pathlib, sys
import h5py
import numpy as np
import torch
from transformers import DynamicCache

sys.path.insert(0, "/path/to/study/scripts")
import kv_common as K  # noqa: E402

sys.path.insert(0, "/path/to/Latent-Bridge/scripts/pi0")
from pi0_bridge_kv import Pi0BridgeKV  # noqa: E402

S, N_LAYERS = K.S, K.N_LAYERS


class KVFill:
    """KV-fill inference: sync/cache/taylor/bridge with per-request seeded flow noise.
    Same logic as kv_bridge_server_std.KVBridgeInference (copied to avoid mixing the
    std_openpi and robocasa source trees on sys.path)."""

    def __init__(self, policy, device, num_denoise_steps=10):
        self.policy = policy
        self.model = policy._model
        self.bridge = None
        self.device = device
        self.num_denoise_steps = num_denoise_steps
        self.lm = self.model.paligemma_with_expert.paligemma.language_model
        self.lm.config._attn_implementation = "eager"
        self.mode = "sync"
        self.f = 1
        self.fixed_noise_seed = None
        self.reset()
        self.model.sample_actions = self._sample
        policy._sample_actions = self._sample

    def reset(self):
        self.prev_kv_preRoPE = None
        self.prev_embedding = None
        self.raw_state = None
        self.raw_action = None
        self.steps_since_vlm = 0
        self.kv_history = []

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
        return torch.stack(out)

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
                st = (self.raw_state if self.raw_state is not None
                      else torch.zeros(1, 8, device=device)).to(bd)
                ac = (self.raw_action if self.raw_action is not None
                      else torch.zeros(1, 7, device=device)).to(bd)
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


def load_bridge(path, device):
    m = Pi0BridgeKV(kv_dim=256, num_layers=18, seq_len=768, emb_dim=2048,
                    hidden_dim=768, num_heads=12, num_blocks=10,
                    state_dim=8, action_dim=7)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck.get("model", ck.get("model_state_dict")))
    return m.to(device).to(torch.bfloat16).eval()


def val_source_eps(h5_path, val_frac=0.25, seed=42):
    """Same split as delta_fidelity: sorted keys, RandomState(seed) permutation."""
    with h5py.File(h5_path, "r") as hf:
        eps = sorted(hf.keys())
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(eps))
        n_val = max(1, int(len(eps) * val_frac))
        val = [eps[i] for i in perm[:n_val]]
        out = [(int(hf[e].attrs["source_episode"]), int(hf[e].attrs["n_steps"])) for e in val]
        stride = int(hf.attrs["stride"])
    return out, stride


def stash(kvb, element, out):
    st = np.concatenate([element["observation/joint_position"],
                         element["observation/gripper_position"]]).astype(np.float32)[:8]
    s8 = np.zeros(8, np.float32); s8[:st.shape[0]] = st
    kvb.raw_state = torch.from_numpy(s8).unsqueeze(0).to(kvb.device)
    a = np.asarray(out["actions"])[0, :7].astype(np.float32)
    kvb.raw_action = torch.from_numpy(a).unsqueeze(0).to(kvb.device)


def run_pass(policy, kvb, elems, mode, f, seed0):
    kvb.mode = mode; kvb.f = f; kvb.reset()
    chunks = []
    for i, el in enumerate(elems):
        kvb.fixed_noise_seed = seed0 + i
        out = policy.infer(el)
        chunks.append(np.asarray(out["actions"])[:5, :7].astype(np.float32))
        stash(kvb, el, out)
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="out/action_fidelity_simreal.json")
    args = ap.parse_args()

    policy = K.load_policy(args.device, num_steps=10)
    kvb = KVFill(policy, args.device, num_denoise_steps=10)
    print("policy loaded", flush=True)

    sources = {
        "sim": dict(h5="out/kv_libero.h5", bridge="out/bridge_sim.pt"),
        "real": dict(h5="out/kv_droid.h5", bridge="out/bridge_real.pt"),
    }
    res = {}
    for src, cfg in sources.items():
        kvb.bridge = load_bridge(cfg["bridge"], args.device)
        vals, stride = val_source_eps(cfg["h5"])
        print(f"[{src}] val source episodes {vals} stride={stride}", flush=True)
        if src == "real":
            meta, data, _ = K.droid_episodes()
        acc = {m: {"vlm": [], "skip": []} for m in ["cache", "taylor", "bridge"]}
        for src_ep, n_steps in vals:
            if src == "real":
                row = meta[meta["episode_index"] == src_ep].iloc[0]
                imgs, state, action, ep = K.droid_episode_frames(row, data)
                task = row["tasks"]
                task = task[0] if isinstance(task, (list, np.ndarray)) else task
                task = str(task) if str(task).strip() else "do something"
            else:
                found = False
                for fpath in K.libero_episode_files():
                    imgs, state, action, ep, task = K.libero_episode_frames(fpath)
                    if ep == src_ep:
                        found = True
                        break
                if not found:
                    print(f"  [{src}] source ep {src_ep} not found, skip", flush=True)
                    continue
            n_fr = min(len(imgs["ext1"]), len(imgs["wrist"]), len(state))
            idxs = list(range(0, n_fr, stride))[: min(args.max_steps, n_steps)]
            elems = [K.make_element(imgs["ext1"][i], imgs["wrist"][i], state[i], task)
                     for i in idxs]
            seed0 = 100000 + src_ep * 1000
            ref = run_pass(policy, kvb, elems, "sync", 1, seed0)
            for mode in ["cache", "taylor", "bridge"]:
                ch = run_pass(policy, kvb, elems, mode, args.f, seed0)
                for i, (a, b) in enumerate(zip(ch, ref)):
                    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
                    key = "vlm" if i % args.f == 0 else "skip"
                    acc[mode][key].append(rmse)
            print(f"  [{src}] ep{src_ep}: {len(elems)} steps done", flush=True)

        res[src] = {}
        for mode in ["cache", "taylor", "bridge"]:
            sk, vl = acc[mode]["skip"], acc[mode]["vlm"]
            res[src][mode] = {
                "skip_rmse": float(np.mean(sk)) if sk else None,
                "skip_std": float(np.std(sk)) if sk else None,
                "n_skip": len(sk),
                "vlm_sanity_rmse": float(np.mean(vl)) if vl else None,
            }
        print(f"=== [{src}] per-step action RMSE vs sync-conditioned reference ===", flush=True)
        for mode in ["cache", "taylor", "bridge"]:
            r = res[src][mode]
            print(f"  {mode:7s} skip={r['skip_rmse']:.4f} (n={r['n_skip']})  "
                  f"[vlm sanity {r['vlm_sanity_rmse']:.5f}]", flush=True)

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print("wrote", args.out, flush=True)
    print("DONE_REAL_ACTION_FIDELITY", flush=True)


if __name__ == "__main__":
    main()
