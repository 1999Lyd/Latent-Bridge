"""Phase B - store prefix KV sequences on REAL episodes so an R0 bridge can be trained
offline, exactly the regime that a real robot deployment would use (no simulator, so
no DAgger). Sampling stride is chosen so consecutive stored steps are ~0.2s apart,
matching the paper's pi0.5 replan interval.
"""
import argparse, sys, time
import h5py
import numpy as np

sys.path.insert(0, "/path/to/study/scripts")
import kv_common as K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["droid", "libero"], default="droid")
    ap.add_argument("--n_episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=3, help="frames between stored inferences")
    ap.add_argument("--max_steps", type=int, default=60, help="max stored steps per episode")
    ap.add_argument("--min_steps", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    policy = K.load_policy(args.device, num_steps=1)
    lm, captured = K.attach_hook(policy)
    print("policy loaded", flush=True)

    if args.source == "droid":
        meta, data, _ = K.droid_episodes()
        n_avail = len(meta)
    else:
        files = K.libero_episode_files(args.n_episodes)
        n_avail = len(files)

    t0 = time.time()
    kept = 0
    n_inf = 0
    with h5py.File(args.out, "w") as hf:
        for i in range(min(args.n_episodes, n_avail)):
            if args.source == "droid":
                row = meta.iloc[i]
                imgs, state, action, ep = K.droid_episode_frames(row, data)
                task = row["tasks"]
                task = task[0] if isinstance(task, (list, np.ndarray)) else task
                if not str(task).strip():
                    task = "do something"
                task = str(task)
            else:
                imgs, state, action, ep, task = K.libero_episode_frames(files[i])

            n_fr = min(len(imgs["ext1"]), len(imgs["wrist"]), len(state))
            idxs = list(range(0, n_fr, args.stride))[: args.max_steps]
            if len(idxs) < args.min_steps:
                print(f"  skip ep{ep} ({len(idxs)} steps)", flush=True)
                continue

            kv_seq = np.zeros((len(idxs), K.N_LAYERS, K.S, K.KV_DIM), dtype=np.float16)
            emb_seq = np.zeros((len(idxs), K.S, 2048), dtype=np.float16)
            st_seq = np.zeros((len(idxs), 8), dtype=np.float32)
            ac_seq = np.zeros((len(idxs), 7), dtype=np.float32)

            for j, fi in enumerate(idxs):
                el = K.make_element(imgs["ext1"][fi], imgs["wrist"][fi], state[fi], task)
                kv, emb = K.run_step(policy, captured, el)
                kv_seq[j] = kv
                emb_seq[j] = emb
                s = state[fi]
                st_seq[j, : min(8, s.shape[0])] = s[:8]
                a = action[fi]
                ac_seq[j, : min(7, a.shape[0])] = a[:7]
                n_inf += 1

            g = hf.create_group(f"episode_{kept:04d}")
            g.create_dataset("kv", data=kv_seq, compression="gzip", compression_opts=1)
            g.create_dataset("embedding", data=emb_seq, compression="gzip", compression_opts=1)
            g.create_dataset("state", data=st_seq)
            g.create_dataset("action", data=ac_seq)
            g.attrs["task"] = task
            g.attrs["source_episode"] = ep
            g.attrs["stride"] = args.stride
            g.attrs["n_steps"] = len(idxs)
            hf.flush()
            kept += 1
            el_t = time.time() - t0
            print(f"  ep{ep}: {len(idxs)} steps  (kept {kept}, {n_inf} inf, "
                  f"{el_t/60:.1f} min, {el_t/max(n_inf,1):.2f}s/inf)", flush=True)
            del kv_seq, emb_seq

        hf.attrs["source"] = args.source
        hf.attrs["n_episodes"] = kept
        hf.attrs["stride"] = args.stride
        hf.attrs["fps"] = 15.0 if args.source == "droid" else 10.0
    print(f"wrote {args.out}: {kept} episodes, {n_inf} inferences", flush=True)


if __name__ == "__main__":
    main()
