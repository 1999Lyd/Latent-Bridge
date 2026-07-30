"""Sync-policy SR across the four LIBERO suites at a given action-chunk length, to show
that chunking to length 10 does not degrade the base (full-VLM) policy.

Sync mode only (VLM every inference). For each suite we cover all 10 tasks; the requested
n (default 60/suite) is split into 3 seed groups x 20 episodes, with DISTINCT init states
per seed (seed s -> init states [2s, 2s+1] of each task), so the three groups are genuinely
independent rollouts (paper-style 3-seed reporting) rather than repeats.

Outputs, per (suite, chunk): overall SR + Wilson 95% CI, and the per-seed SR triple.
"""
import argparse, collections, json, math, pathlib, sys, time
import numpy as np

import torch as _torch
_ol = _torch.load
_torch.load = lambda *a, **k: _ol(*a, **{**k, "weights_only": False})

sys.path.insert(0, "/path/to/envs/libero_mem/data_repo")
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from openpi_client import image_tools  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402

DUMMY = [0.0] * 6 + [-1.0]
MAXSTEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
SUITE_ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def quat2axisangle(q):
    q = np.array(q, dtype=np.float64)
    q[3] = max(-1.0, min(1.0, q[3]))
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def elem(obs, task):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wr = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
    wr = image_tools.convert_to_uint8(image_tools.resize_with_pad(wr, 224, 224))
    st = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                         obs["robot0_gripper_qpos"])).astype(np.float32)
    return {"observation/image": img, "observation/wrist_image": wr,
            "observation/state": st, "prompt": str(task)}


def run_ep(client, env, init_state, task, replan, budget, wait=10):
    client.infer({"__reset__": True, "__mode__": "sync", "__f__": 1})
    env.reset()
    obs = env.set_init_state(init_state)
    plan = collections.deque()
    t = 0
    while t < budget + wait:
        if t < wait:
            obs, _, done, _ = env.step(DUMMY)
            t += 1
            continue
        if not plan:
            chunk = np.asarray(client.infer(elem(obs, task))["actions"])
            for i in range(min(replan, chunk.shape[0])):
                plan.append(chunk[i, :7])
        obs, _, done, _ = env.step(np.asarray(plan.popleft()).tolist())
        if done:
            return True, t
        t += 1
    return False, t


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", default=",".join(SUITE_ORDER))
    ap.add_argument("--chunks", default="10")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n_tasks", type=int, default=10)
    ap.add_argument("--eps_per_seed_task", type=int, default=2)  # 10 tasks x 3 seeds x 2 = 60/suite
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--out", default="out/sync_chunk_eval.json")
    args = ap.parse_args()

    suites = args.suites.split(",")
    chunks = [int(x) for x in args.chunks.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print("connected to server", flush=True)

    bd = benchmark.get_benchmark_dict()
    t0 = time.time()
    records = []
    for suite_name in suites:
        suite = bd[suite_name]()
        budget = MAXSTEPS.get(suite_name, 300)
        n_tasks = min(args.n_tasks, suite.n_tasks)
        for tid in range(n_tasks):
            task = suite.get_task(tid)
            inits = suite.get_task_init_states(tid)
            bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
            for seed in seeds:
                env.seed(seed)
                for j in range(args.eps_per_seed_task):
                    idx = (seed * args.eps_per_seed_task + j) % len(inits)
                    for chunk in chunks:
                        succ, steps = run_ep(client, env, inits[idx], task.language, chunk, budget)
                        records.append({"suite": suite_name, "task_id": tid, "seed": seed,
                                        "init_idx": idx, "chunk": chunk,
                                        "success": int(succ), "steps": steps})
            env.close()
            done_n = sum(1 for r in records if r["suite"] == suite_name)
            print(f"  [{suite_name} task{tid}] cum {done_n} eps  ({(time.time()-t0)/60:.1f} min)", flush=True)
        # per-suite summary
        for chunk in chunks:
            rs = [r["success"] for r in records if r["suite"] == suite_name and r["chunk"] == chunk]
            k, n = int(sum(rs)), len(rs)
            lo, hi = wilson(k, n)
            per_seed = {}
            for seed in seeds:
                sr = [r["success"] for r in records
                      if r["suite"] == suite_name and r["chunk"] == chunk and r["seed"] == seed]
                per_seed[seed] = (int(sum(sr)), len(sr))
            ps = "  ".join(f"s{seed}={100*a/b:.1f}({a}/{b})" for seed, (a, b) in per_seed.items())
            print(f"  [SR {suite_name} chunk={chunk}] {100*k/n:.2f}% ({k}/{n})  "
                  f"95%CI[{100*lo:.1f},{100*hi:.1f}]  | {ps}", flush=True)

    # aggregate
    summ = {}
    for suite_name in suites:
        for chunk in chunks:
            rs = [r["success"] for r in records if r["suite"] == suite_name and r["chunk"] == chunk]
            k, n = int(sum(rs)), len(rs)
            lo, hi = wilson(k, n)
            summ[f"{suite_name}|chunk{chunk}"] = {
                "n": n, "succ": k, "sr": 100 * k / n if n else float("nan"),
                "ci95": [100 * lo, 100 * hi]}
    # mean-of-suites per chunk
    means = {}
    for chunk in chunks:
        srs = [summ[f"{s}|chunk{chunk}"]["sr"] for s in suites if f"{s}|chunk{chunk}" in summ]
        means[chunk] = float(np.mean(srs)) if srs else float("nan")

    out = {"suites": suites, "chunks": chunks, "seeds": seeds,
           "per_suite": summ, "mean_of_suites": means, "records": records}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("\n=== SYNC SR by suite x chunk ===", flush=True)
    for chunk in chunks:
        row = "  ".join(f"{s.replace('libero_','')}={summ[f'{s}|chunk{chunk}']['sr']:.2f}" for s in suites)
        print(f"  chunk={chunk}:  {row}   mean={means[chunk]:.2f}", flush=True)
    print("wrote", args.out, flush=True)
    print("DONE_SYNC_CHUNK", flush=True)


if __name__ == "__main__":
    main()
