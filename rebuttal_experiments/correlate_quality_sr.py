"""Make the case-study insight quantitative: feature-recovery quality -> SR.

Pairs measured chained feature-recovery cosine (chain_eval on held-out sim data)
with the paper's EXISTING success rates:
  - bridge at f=2..8  <- Table 10 frequency sweep SR (per-suite mean)
  - cache at default f <- Table 1 Feature-Caching SR
  - sync (quality=1.0) <- Table 1 Sync SR
and reports Spearman rank correlation + a scatter figure.

This is the missing link for the no-real-arm argument: (1) quality governs SR
(shown here + case study + matched-f contrast), (2) the bridge attains high quality
on REAL data (Phase A/B), => (3) SR should transfer.
"""
import argparse, json, sys
import numpy as np

# Existing paper SR (pi0.5), averaged across the 4 LIBERO suites.
# Table 10 frequency sweep (bridge SR):
SWEEP = {  # f -> [Spatial, Object, Goal, Long]
    2: [98.5, 98.0, 97.0, 93.5],
    3: [99.0, 100.0, 99.0, 92.5],
    4: [99.0, 97.5, 97.5, 93.5],
    6: [98.0, 96.5, 98.5, 87.5],
    8: [95.5, 96.5, 94.5, 84.5],
}
SYNC_AVG = 96.96      # Table 1 pi0.5 sync avg
CACHE_AVG = 56.38     # Table 1 pi0.5 feature-caching avg
CACHE_F = 4           # caching reported at the default operating f


def mean(x):
    return sum(x) / len(x)


def spearman(x, y):
    def rank(v):
        order = np.argsort(v)
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        return r
    rx, ry = rank(np.array(x)), rank(np.array(y))
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx @ ry) / (np.sqrt((rx @ rx) * (ry @ ry)) + 1e-12))


def pearson(x, y):
    x, y = np.array(x), np.array(y)
    x = x - x.mean(); y = y - y.mean()
    return float((x @ y) / (np.sqrt((x @ x) * (y @ y)) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True, help="chain_eval sim output json")
    ap.add_argument("--out", default="out/quality_sr.json")
    ap.add_argument("--fig", default="out/quality_sr.png")
    args = ap.parse_args()

    ch = json.load(open(args.chain))["f"]

    xs, ys, labels, series = [], [], [], []
    # sync anchor
    xs.append(1.000); ys.append(SYNC_AVG); labels.append("sync"); series.append("sync")
    # bridge points across f
    for f, srs in SWEEP.items():
        if str(f) in ch and ch[str(f)]["bridge_recovery"] is not None:
            xs.append(ch[str(f)]["bridge_recovery"]); ys.append(mean(srs))
            labels.append(f"bridge f={f}"); series.append("bridge")
    # cache anchor
    if str(CACHE_F) in ch and ch[str(CACHE_F)]["cache_recovery"] is not None:
        xs.append(ch[str(CACHE_F)]["cache_recovery"]); ys.append(CACHE_AVG)
        labels.append(f"cache f={CACHE_F}"); series.append("cache")

    rho = spearman(xs, ys)
    r = pearson(xs, ys)
    res = {"spearman": rho, "pearson": r,
           "points": [{"quality": x, "sr": y, "label": l} for x, y, l in zip(xs, ys, labels)]}
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)

    print(f"Spearman rho = {rho:.3f}   Pearson r = {r:.3f}   ({len(xs)} operating points)")
    for x, y, l in sorted(zip(xs, ys, labels)):
        print(f"  {l:14s} recovery {x:.4f} -> SR {y:.2f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {"sync": "#2ca02c", "bridge": "#1f77b4", "cache": "#d62728"}
        fig, ax = plt.subplots(figsize=(6, 4.6))
        for s in ["sync", "bridge", "cache"]:
            xi = [x for x, se in zip(xs, series) if se == s]
            yi = [y for y, se in zip(ys, series) if se == s]
            ax.scatter(xi, yi, s=70, color=colors[s], label=s, zorder=3,
                       edgecolor="k", linewidth=0.5)
        for x, y, l in zip(xs, ys, labels):
            ax.annotate(l, (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")
        ax.set_xlabel("chained feature-recovery cosine to ground-truth KV (measured)")
        ax.set_ylabel("success rate (%) — existing paper results")
        ax.set_title(f"Feature-recovery quality governs SR (Spearman ρ={rho:.2f})")
        ax.grid(alpha=0.3); ax.legend()
        plt.tight_layout(); plt.savefig(args.fig, dpi=150, bbox_inches="tight")
        print("wrote", args.fig)
    except Exception as e:
        print("fig skipped:", e)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
