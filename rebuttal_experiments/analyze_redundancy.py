"""Phase A - temporal redundancy of the pi0.5 prefix KV cache on REAL vs SIM streams.

Same checkpoint (pi0.5 base), same input transform, same metric. Only the pixels differ.
For each episode we extract prefix KV at consecutive frames and report per-layer
per-token cosine between frames separated by stride s (i.e. what the "copy baseline"
would achieve if the VLM were skipped for s frames).

DROID is 15 fps, LIBERO is 10 fps, so strides are also reported in wall-clock seconds:
DROID stride 3 == LIBERO stride 2 == 0.200 s.
"""
import argparse, json, sys, time
import numpy as np

sys.path.insert(0, "/path/to/study/scripts")
import kv_common as K

STRIDES = [1, 2, 3, 4, 5, 6, 8, 10]
FPS = {"droid": 15.0, "libero": 10.0}


def episode_iter(source, n_episodes):
    if source == "droid":
        meta, data, _ = K.droid_episodes()
        for i in range(min(n_episodes, len(meta))):
            row = meta.iloc[i]
            imgs, state, action, ep = K.droid_episode_frames(row, data)
            task = row["tasks"]
            task = task[0] if isinstance(task, (list, np.ndarray)) else task
            if not str(task).strip():
                task = "do something"
            yield imgs, state, action, ep, str(task)
    else:
        for f in K.libero_episode_files(n_episodes):
            imgs, state, action, ep, task = K.libero_episode_frames(f)
            yield imgs, state, action, ep, task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["droid", "libero"], required=True)
    ap.add_argument("--n_episodes", type=int, default=40)
    ap.add_argument("--max_frames", type=int, default=120)
    ap.add_argument("--min_frames", type=int, default=24)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    policy = K.load_policy(args.device, num_steps=1)
    lm, captured = K.attach_hook(policy)
    print("policy loaded", flush=True)

    # accumulators: per stride -> list over pairs of per-layer cosine [18]
    acc = {s: [] for s in STRIDES}
    n_eps = 0
    n_inf = 0
    t0 = time.time()

    for imgs, state, action, ep, task in episode_iter(args.source, args.n_episodes):
        n = min(len(imgs["ext1"]), len(imgs["wrist"]), len(state), args.max_frames)
        if n < args.min_frames:
            print(f"  skip ep{ep} (only {n} frames)", flush=True)
            continue
        kvs = np.zeros((n, K.N_LAYERS, K.S, K.KV_DIM), dtype=np.float16)
        for i in range(n):
            el = K.make_element(imgs["ext1"][i], imgs["wrist"][i], state[i], task)
            kv, _ = K.run_step(policy, captured, el)
            kvs[i] = kv
            n_inf += 1
        for s in STRIDES:
            for i in range(0, n - s):
                acc[s].append(K.per_layer_cosine(kvs[i], kvs[i + s]))
        n_eps += 1
        el_t = time.time() - t0
        print(f"  ep{ep} n={n} done ({n_eps} eps, {n_inf} inf, {el_t/60:.1f} min, "
              f"{el_t/max(n_inf,1):.2f}s/inf)", flush=True)
        del kvs

    res = {"source": args.source, "fps": FPS[args.source], "n_episodes": n_eps,
           "n_inferences": n_inf, "strides": {}}
    for s in STRIDES:
        if not acc[s]:
            continue
        a = np.stack(acc[s])                     # [pairs, 18]
        res["strides"][str(s)] = {
            "dt_seconds": s / FPS[args.source],
            "n_pairs": int(a.shape[0]),
            "per_layer_mean": a.mean(0).round(6).tolist(),
            "per_layer_std": a.std(0).round(6).tolist(),
            "mean": float(a.mean()),
            "std_over_pairs": float(a.mean(1).std()),
            "p05": float(np.percentile(a.mean(1), 5)),
        }
        print(f"stride {s} (dt={s/FPS[args.source]:.3f}s): mean cos {a.mean():.5f}", flush=True)

    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
