"""Autoregressive chaining eval: how well does the bridge (vs stale caching) recover
the true VLM KV as a function of skip period f? Produces the feature-recovery-quality
axis that we pair with the paper's existing SR-vs-f (Table 10) to make the
quality -> SR relationship explicit.

Bridge step: input is the bridge's OWN previous recovered KV (chained); SigLIP embedding
is fresh every step (as in real pi0.5). Cache: reuse the last VLM-step KV (stale).
"""
import argparse, json, sys
import h5py
import numpy as np
import torch

sys.path.insert(0, "/path/to/Latent-Bridge/scripts/pi0")
from pi0_bridge_kv import Pi0BridgeKV  # noqa: E402

N_LAYERS, S = 18, 768


def load_bridge(ckpt, device):
    m = Pi0BridgeKV(kv_dim=256, num_layers=18, seq_len=768, emb_dim=2048,
                    hidden_dim=768, num_heads=12, num_blocks=10,
                    state_dim=8, action_dim=7).to(device)
    sd = torch.load(ckpt, map_location=device)["model"]
    m.load_state_dict(sd)
    m.eval()
    return m


def per_layer_cos(a, b):
    c = torch.nn.functional.cosine_similarity(a, b, dim=-1)  # [18,S]
    return c.mean().item()


@torch.no_grad()
def chain_episode(bridge, g, f, device):
    """Returns per-step cosine to ground truth for four recovery strategies at period f:
    learned bridge, stale reuse (=Taylor order 0), Taylor order 1, Taylor order 2.
    Taylor uses the true VLM features at the last VLM steps (spacing f), matching
    TaylorSeer's finite-difference extrapolation ported to control steps.
    """
    kv = torch.from_numpy(g["kv"][:].astype(np.float32)).to(device)      # [T,18,768,512]
    emb = torch.from_numpy(g["embedding"][:].astype(np.float32)).to(device)  # [T,768,2048]
    st = torch.from_numpy(g["state"][:].astype(np.float32)).to(device)
    ac = torch.from_numpy(g["action"][:].astype(np.float32)).to(device)
    T = kv.shape[0]
    out = {"bridge": [], "cache": [], "taylor1": [], "taylor2": []}
    recovered = kv[0].clone()
    hist = [kv[0].clone()]  # history of VLM-step features (most-recent last)
    last_vlm_t = 0
    for t in range(1, T):
        if t % f == 0:
            recovered = kv[t].clone()
            hist.append(kv[t].clone())
            if len(hist) > 3:
                hist.pop(0)
            last_vlm_t = t
            continue
        k = t - last_vlm_t  # bridge offset
        # learned bridge (chained)
        flat = recovered.permute(1, 0, 2).reshape(1, S, N_LAYERS * 512)
        ed = (emb[t] - emb[t - 1]).unsqueeze(0)
        ce = emb[t].unsqueeze(0)
        deltas = bridge(ed, ce, flat, st[t].unsqueeze(0), ac[t - 1].unsqueeze(0))
        recovered = recovered + torch.stack(deltas, dim=1)[0]
        out["bridge"].append(per_layer_cos(recovered, kv[t]))
        # Taylor / stale reuse from true VLM features (spacing f)
        F0 = hist[-1]
        out["cache"].append(per_layer_cos(F0, kv[t]))                      # order 0
        if len(hist) >= 2:
            d1 = F0 - hist[-2]
            t1 = F0 + (k / f) * d1
            out["taylor1"].append(per_layer_cos(t1, kv[t]))
            if len(hist) >= 3:
                d2 = F0 - 2 * hist[-2] + hist[-3]
                t2 = t1 + (k * k / (2.0 * f * f)) * d2
                out["taylor2"].append(per_layer_cos(t2, kv[t]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--bridge", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--f_list", default="2,3,4,6,8")
    ap.add_argument("--val_frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    bridge = load_bridge(args.bridge, args.device)
    with h5py.File(args.data, "r") as hf:
        eps = sorted(hf.keys())
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(len(eps))
        n_val = max(1, int(len(eps) * args.val_frac))
        val_eps = [eps[i] for i in perm[:n_val]]  # held-out only

        res = {"data": args.data, "bridge": args.bridge, "n_val_eps": len(val_eps), "f": {}}
        for f in [int(x) for x in args.f_list.split(",")]:
            agg = {"bridge": [], "cache": [], "taylor1": [], "taylor2": []}
            for name in val_eps:
                o = chain_episode(bridge, hf[name], f, args.device)
                for k in agg:
                    agg[k] += o[k]
            m = {k: (float(np.mean(v)) if v else None) for k, v in agg.items()}
            res["f"][str(f)] = {
                "bridge_recovery": m["bridge"], "cache_recovery": m["cache"],
                "taylor1_recovery": m["taylor1"], "taylor2_recovery": m["taylor2"],
                "n_bridge_steps": len(agg["bridge"]),
            }
            print(f"f={f}: bridge {m['bridge']:.5f}  cache {m['cache']:.5f}  "
                  f"taylor1 {m['taylor1']:.5f}  "
                  f"taylor2 {m['taylor2'] if m['taylor2'] else float('nan'):.5f}  "
                  f"(bridge-taylor1 {m['bridge']-m['taylor1']:+.5f}, {len(agg['bridge'])} steps)",
                  flush=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
