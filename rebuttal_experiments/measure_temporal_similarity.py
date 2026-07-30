"""Precise answer to Reviewer 2 Q1/Q5: per-layer consecutive-step cosine of the VLM
hidden states, separately for IMAGE tokens and TEXT (prompt) tokens, on real (DROID)
and sim (LIBERO). Averaged over tokens, over all consecutive step-pairs, over episodes.

Directly tests: (a) which layers are 'stable' vs 'dynamic', (b) whether text tokens
are ~invariant, and (c) the reviewer's point that DEEP text tokens may drift via
bidirectional attention to changing visual tokens.
"""
import argparse, json, sys
import numpy as np
import torch

sys.path.insert(0, "/path/to/study/scripts")
import kv_common as K

S_IMG = 768  # image tokens


@torch.no_grad()
def hs_step(policy, captured, element):
    """Prefix forward; return list of 19 hidden states [1, L_total, 2048] (fp32 cpu-free)."""
    captured[0] = None
    K._prefix_forward(policy, element)
    hs = captured[0]
    return hs  # tuple of tensors on GPU


def cos_tokens(a, b):
    """mean per-token cosine between [T,D] and [T,D]."""
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()


def run_source(policy, captured, source, n_ep, max_frames, device):
    if source == "droid":
        meta, data, _ = K.droid_episodes()
        gen = []
        for i in range(min(n_ep, len(meta))):
            row = meta.iloc[i]
            imgs, state, action, ep = K.droid_episode_frames(row, data)
            task = row["tasks"]; task = task[0] if isinstance(task, (list, np.ndarray)) else task
            task = str(task) if str(task).strip() else "do something"
            gen.append((imgs, state, task))
    else:
        gen = []
        for f in K.libero_episode_files(n_ep):
            imgs, state, action, ep, task = K.libero_episode_frames(f)
            gen.append((imgs, state, task))

    n_layers = 19  # embeddings + 18
    img_acc = [[] for _ in range(n_layers)]
    txt_acc = [[] for _ in range(n_layers)]

    for imgs, state, task in gen:
        n = min(len(imgs["ext1"]), len(imgs["wrist"]), len(state), max_frames)
        if n < 4:
            continue
        prev = None
        for i in range(n):
            el = K.make_element(imgs["ext1"][i], imgs["wrist"][i], state[i], task)
            hs = hs_step(policy, captured, el)
            cur = [h[0].float() for h in hs]  # each [L_total, 2048]
            if prev is not None:
                Ltot = cur[0].shape[0]
                for l in range(n_layers):
                    img_acc[l].append(cos_tokens(prev[l][:S_IMG], cur[l][:S_IMG]))
                    if Ltot > S_IMG:
                        txt_acc[l].append(cos_tokens(prev[l][S_IMG:], cur[l][S_IMG:]))
            prev = cur

    res = {"source": source, "n_episodes": len(gen),
           "image_cos_per_layer": [float(np.mean(x)) if x else None for x in img_acc],
           "text_cos_per_layer": [float(np.mean(x)) if x else None for x in txt_acc],
           "n_pairs": len(img_acc[0])}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=15)
    ap.add_argument("--max_frames", type=int, default=48)
    ap.add_argument("--out", default="out/temporal_similarity.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    policy = K.load_policy(args.device, num_steps=1)
    lm, captured = K.attach_hook(policy)
    print("policy loaded", flush=True)

    out = {}
    for source in ["droid", "libero"]:
        print(f"=== {source} ===", flush=True)
        r = run_source(policy, captured, source, args.n_episodes, args.max_frames, args.device)
        out[source] = r
        img = r["image_cos_per_layer"]; txt = r["text_cos_per_layer"]
        print(f"  {r['n_pairs']} pairs", flush=True)
        print(f"  image L0={img[0]:.5f} L6={img[6]:.5f} L12={img[12]:.5f} L18={img[18]:.5f}", flush=True)
        tv = [t for t in txt if t is not None]
        if tv:
            print(f"  text  L0={txt[0]:.6f} L6={txt[6]:.6f} L12={txt[12]:.6f} L18={txt[18]:.6f}", flush=True)
            print(f"  text min-over-layers = {min(tv):.6f}", flush=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", args.out, flush=True)
    print("DONE_TEMPORAL_SIM", flush=True)


if __name__ == "__main__":
    main()
