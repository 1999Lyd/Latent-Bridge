"""Open-loop ACTION fidelity under each KV-fill mode (safety evidence for R1).

Pass 1 (rollout): closed-loop sync episode in the env; save every inference-step
observation element and the sync action chunk. Flow-matching noise is seeded per
step (via the server's __seed__ sentinel).

Replay passes (no env): feed the SAME observation stream with the SAME per-step
seeds through each mode at f=4. Any action difference vs the sync chunk is then
attributable to the injected features alone (observations and noise identical):
  sync2  : sync mode, DIFFERENT seeds  -> the policy's own sampling-noise scale
  cache  : stale reuse                 -> divergence from stale features
  taylor : order-1 extrapolation
  bridge : learned delta prediction

Metric per inference step: RMSE between the executed part of the chunk (replan x 7)
and the sync reference; also relative to the RMS magnitude of the sync actions.
Reported split by VLM steps (i % f == 0; sanity: cache/taylor/bridge run a fresh
VLM there, so with matched seeds divergence should be ~0) and skip steps.
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


def rollout_sync(client, env, init_state, task, replan, budget, seed0, wait=10):
    """Closed-loop sync rollout; returns saved elements + sync chunks (+ success)."""
    client.infer({"__reset__": True, "__mode__": "sync", "__f__": 1})
    env.reset()
    obs = env.set_init_state(init_state)
    elems, chunks = [], []
    plan = collections.deque()
    t, i = 0, 0
    done = False
    while t < budget + wait:
        if t < wait:
            obs, _, done, _ = env.step(DUMMY)
            t += 1
            continue
        if not plan:
            e = elem(obs, task)
            out = client.infer({**e, "__seed__": seed0 + i})
            chunk = np.asarray(out["actions"])[:replan, :7].astype(np.float32)
            elems.append(e)
            chunks.append(chunk)
            i += 1
            for a in chunk:
                plan.append(a)
        obs, _, done, _ = env.step(np.asarray(plan.popleft()).tolist())
        if done:
            break
        t += 1
    return elems, chunks, bool(done)


def replay(client, elems, mode, f, replan, seed0):
    """Feed the saved observation stream through `mode`; return chunks."""
    client.infer({"__reset__": True, "__mode__": mode, "__f__": f})
    chunks = []
    for i, e in enumerate(elems):
        out = client.infer({**e, "__seed__": seed0 + i})
        chunks.append(np.asarray(out["actions"])[:replan, :7].astype(np.float32))
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--f", type=int, default=4)
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--n_tasks", type=int, default=2)
    ap.add_argument("--n_eps", type=int, default=3)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--out", default="out/action_fidelity.json")
    args = ap.parse_args()

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print("connected", flush=True)

    bd = benchmark.get_benchmark_dict()
    suite = bd[args.suite]()
    budget = MAXSTEPS.get(args.suite, 300)
    modes = ["sync2", "cache", "taylor", "bridge"]
    # per (mode, steptype) lists of (rmse, rel)
    acc = {m: {"vlm": [], "skip": []} for m in modes}
    t0 = time.time()

    for tid in range(min(args.n_tasks, suite.n_tasks)):
        task = suite.get_task(tid)
        inits = suite.get_task_init_states(tid)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        env.seed(0)
        for ep in range(min(args.n_eps, len(inits))):
            seed0 = 1000 * (tid * 100 + ep)
            elems, sync_chunks, succ = rollout_sync(
                client, env, inits[ep], task.language, args.replan, budget, seed0)
            if len(elems) < args.f + 1:
                continue
            for mode in modes:
                srv_mode = "sync" if mode == "sync2" else mode
                f = 1 if mode == "sync2" else args.f
                s0 = seed0 + 500000 if mode == "sync2" else seed0  # sync2: fresh noise
                ch = replay(client, elems, srv_mode, f, args.replan, s0)
                for i, (a, b) in enumerate(zip(ch, sync_chunks)):
                    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
                    rel = rmse / (float(np.sqrt(np.mean(b ** 2))) + 1e-8)
                    key = "vlm" if (mode == "sync2" or i % args.f == 0) else "skip"
                    acc[mode][key].append((rmse, rel))
            print(f"  task{tid} ep{ep}: {len(elems)} steps, sync_succ={succ} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
        env.close()

    def agg(pairs):
        if not pairs:
            return {"n": 0}
        r = np.array([p[0] for p in pairs]); q = np.array([p[1] for p in pairs])
        return {"n": len(pairs), "rmse_mean": float(r.mean()), "rmse_std": float(r.std()),
                "rel_mean": float(q.mean()), "rel_std": float(q.std())}

    res = {"suite": args.suite, "f": args.f, "replan": args.replan,
           "modes": {m: {k: agg(v) for k, v in acc[m].items()} for m in modes}}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)

    print("\n=== open-loop ACTION divergence from sync-conditioned actions ===", flush=True)
    print("  (identical observations + identical flow noise; only KV fill differs)", flush=True)
    nf = res["modes"]["sync2"]["vlm"]
    print(f"  sampling-noise floor (sync, new noise): rmse={nf.get('rmse_mean', float('nan')):.4f} "
          f"rel={nf.get('rel_mean', float('nan')):.3f} (n={nf['n']})", flush=True)
    for m in ["cache", "taylor", "bridge"]:
        s = res["modes"][m]["skip"]; v = res["modes"][m]["vlm"]
        print(f"  {m:7s} skip-steps: rmse={s.get('rmse_mean', float('nan')):.4f} "
              f"rel={s.get('rel_mean', float('nan')):.3f} (n={s['n']})   "
              f"[vlm-step sanity rmse={v.get('rmse_mean', float('nan')):.5f}]", flush=True)
    print("wrote", args.out, flush=True)
    print("DONE_ACTION_FIDELITY", flush=True)


if __name__ == "__main__":
    main()
