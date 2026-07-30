"""Directional-fidelity metric: how well does each method predict the DIRECTION of the
per-step KV change (the paper's actual thesis), rather than raw feature cosine.

For each consecutive pair (t-1, t):
  true_delta   = kv[t] - kv[t-1]
  bridge_delta = learned bridge prediction
  taylor_delta = kv[t-1] - kv[t-2]      (order-1 extrapolation of the previous change)
  cache_delta  = 0                       (predicts no change -> zero directional info)
Metric = per-token cosine(pred_delta, true_delta), averaged over tokens that ACTUALLY
changed (||true_delta|| above the per-step median) -> isolates prediction quality from
the static/unchanging tokens.

Expectation: bridge > taylor > cache(=0), monotone with closed-loop SR
(sync/bridge >> taylor > cache), unlike raw feature cosine which mis-ranks Taylor.
"""
import argparse, json, sys
import h5py
import numpy as np
import torch

sys.path.insert(0, "/path/to/Latent-Bridge/scripts/pi0")
from pi0_bridge_kv import Pi0BridgeKV  # noqa: E402

N_LAYERS, S = 18, 768


def dir_cos_masked(pred, true, frac=0.5):
    """Mean cosine(pred, true) over the top-(1-frac) tokens by ||true|| (changed tokens).
    pred/true: [18, S, 512]."""
    tnorm = true.norm(dim=-1)                       # [18,S]
    thr = tnorm.flatten().median()
    mask = tnorm > thr
    c = torch.nn.functional.cosine_similarity(pred, true, dim=-1)  # [18,S]
    if mask.sum() == 0:
        return float("nan")
    return c[mask].mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/kv_sim.h5")
    ap.add_argument("--bridge", default="out/bridge_sim.pt")
    ap.add_argument("--out", default="out/delta_fidelity_sim.json")
    ap.add_argument("--val_frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    dev = args.device
    m = Pi0BridgeKV(kv_dim=256, num_layers=18, seq_len=768, emb_dim=2048,
                    hidden_dim=768, num_heads=12, num_blocks=10,
                    state_dim=8, action_dim=7).to(dev).eval()
    ck = torch.load(args.bridge, map_location=dev, weights_only=False)
    m.load_state_dict(ck.get("model", ck.get("model_state_dict")))

    with h5py.File(args.data, "r") as hf:
        eps = sorted(hf.keys())
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(len(eps))
        n_val = max(1, int(len(eps) * args.val_frac))
        val = [eps[i] for i in perm[:n_val]]

        b_dc, t_dc, raw_bridge, raw_cache, raw_taylor = [], [], [], [], []
        for name in val:
            g = hf[name]
            kv = torch.from_numpy(g["kv"][:].astype(np.float32)).to(dev)   # [T,18,S,512]
            emb = torch.from_numpy(g["embedding"][:].astype(np.float32)).to(dev)
            st = torch.from_numpy(g["state"][:].astype(np.float32)).to(dev)
            ac = torch.from_numpy(g["action"][:].astype(np.float32)).to(dev)
            T = kv.shape[0]
            for t in range(2, T):
                true_delta = kv[t] - kv[t - 1]
                taylor_delta = kv[t - 1] - kv[t - 2]
                flat = kv[t - 1].permute(1, 0, 2).reshape(1, S, N_LAYERS * 512)
                with torch.no_grad():
                    d = m((emb[t] - emb[t - 1]).unsqueeze(0), emb[t].unsqueeze(0),
                          flat, st[t].unsqueeze(0), ac[t - 1].unsqueeze(0))
                bridge_delta = torch.stack(d, dim=1)[0]
                b_dc.append(dir_cos_masked(bridge_delta, true_delta))
                t_dc.append(dir_cos_masked(taylor_delta, true_delta))
                # raw feature cosine to the TRUE next feature (for reference)
                raw_bridge.append(torch.nn.functional.cosine_similarity(
                    kv[t - 1] + bridge_delta, kv[t], dim=-1).mean().item())
                raw_cache.append(torch.nn.functional.cosine_similarity(
                    kv[t - 1], kv[t], dim=-1).mean().item())
                raw_taylor.append(torch.nn.functional.cosine_similarity(
                    kv[t - 1] + taylor_delta, kv[t], dim=-1).mean().item())

    def mean(x):
        x = [v for v in x if v == v]
        return float(np.mean(x)) if x else float("nan")

    res = {
        "data": args.data, "n_val": len(val), "n_pairs": len(b_dc),
        "delta_direction_cos": {"cache": 0.0, "taylor": mean(t_dc), "bridge": mean(b_dc)},
        "raw_feature_cos": {"cache": mean(raw_cache), "taylor": mean(raw_taylor),
                            "bridge": mean(raw_bridge)},
    }
    print("=== directional-fidelity (cosine of predicted delta vs true delta, changed tokens) ===")
    print(f"  cache  = 0.000 (predicts no change)")
    print(f"  taylor = {res['delta_direction_cos']['taylor']:.4f}")
    print(f"  bridge = {res['delta_direction_cos']['bridge']:.4f}")
    print("=== raw feature cosine (for contrast; mis-ranks taylor) ===")
    print(f"  cache={res['raw_feature_cos']['cache']:.4f}  "
          f"taylor={res['raw_feature_cos']['taylor']:.4f}  "
          f"bridge={res['raw_feature_cos']['bridge']:.4f}")
    json.dump(res, open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
