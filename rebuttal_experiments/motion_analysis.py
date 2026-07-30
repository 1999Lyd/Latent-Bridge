"""Answer Reviewer CHT2's fast-motion / motion-blur concern with real DROID data.
Bin consecutive-step pairs by end-effector translation magnitude (the phase-aware
signal; high motion ~ motion blur / fast motion), and report:
  - cache (stale-reuse) feature-recovery cosine per bin  [drops with motion]
  - bridge feature-recovery cosine per bin                [stays high]
Shows the bridge's advantage over stale reuse is LARGEST exactly in high-motion frames.
"""
import argparse, json, sys
import h5py
import numpy as np
import torch

sys.path.insert(0, "/path/to/Latent-Bridge/scripts/pi0")
from pi0_bridge_kv import Pi0BridgeKV  # noqa: E402

N_LAYERS, S = 18, 768


def per_layer_cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/kv_droid.h5")
    ap.add_argument("--bridge", default="out/bridge_real.pt")
    ap.add_argument("--out", default="out/motion_analysis.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    dev = args.device
    m = Pi0BridgeKV(kv_dim=256, num_layers=18, seq_len=768, emb_dim=2048,
                    hidden_dim=768, num_heads=12, num_blocks=10,
                    state_dim=8, action_dim=7).to(dev).eval()
    ck = torch.load(args.bridge, map_location=dev, weights_only=False)
    m.load_state_dict(ck.get("model", ck.get("model_state_dict")))

    rows = []  # (motion, cache_cos, bridge_cos)
    with h5py.File(args.data, "r") as hf:
        for name in sorted(hf.keys()):
            g = hf[name]
            kv = torch.from_numpy(g["kv"][:].astype(np.float32)).to(dev)
            emb = torch.from_numpy(g["embedding"][:].astype(np.float32)).to(dev)
            st = torch.from_numpy(g["state"][:].astype(np.float32)).to(dev)
            ac = torch.from_numpy(g["action"][:].astype(np.float32)).to(dev)
            T = kv.shape[0]
            for t in range(1, T):
                motion = float(np.linalg.norm(ac[t - 1][:3].cpu().numpy()))
                cache = per_layer_cos(kv[t - 1], kv[t])
                flat = kv[t - 1].permute(1, 0, 2).reshape(1, S, N_LAYERS * 512)
                with torch.no_grad():
                    d = m((emb[t] - emb[t - 1]).unsqueeze(0), emb[t].unsqueeze(0),
                          flat, st[t].unsqueeze(0), ac[t - 1].unsqueeze(0))
                pred = kv[t - 1] + torch.stack(d, dim=1)[0]
                rows.append((motion, cache, per_layer_cos(pred, kv[t])))

    rows = np.array(rows)  # [N,3]
    motions = rows[:, 0]
    qs = np.quantile(motions, [0.0, 0.33, 0.66, 1.0])
    labels = ["low", "medium", "high"]
    res = {"data": args.data, "n_pairs": len(rows), "bins": {}}
    print(f"{len(rows)} pairs; motion (translation norm) tertiles:", flush=True)
    for i, lab in enumerate(labels):
        lo, hi = qs[i], qs[i + 1]
        sel = (motions >= lo) & (motions <= hi if i == 2 else motions < hi)
        r = rows[sel]
        cache_m, bridge_m = r[:, 1].mean(), r[:, 2].mean()
        res["bins"][lab] = {"motion_range": [float(lo), float(hi)], "n": int(sel.sum()),
                            "cache_cos": float(cache_m), "bridge_cos": float(bridge_m),
                            "bridge_gain": float(bridge_m - cache_m)}
        print(f"  {lab:6s} motion[{lo:.3f},{hi:.3f}] n={sel.sum():4d}  "
              f"cache={cache_m:.5f} bridge={bridge_m:.5f} gain={bridge_m-cache_m:+.5f}", flush=True)
    # correlation motion vs cache recovery
    cc = np.corrcoef(motions, rows[:, 1])[0, 1]
    res["corr_motion_vs_cache"] = float(cc)
    print(f"corr(motion, cache_recovery) = {cc:.3f} (negative => fast motion hurts stale reuse)",
          flush=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
