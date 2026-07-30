"""KV-bridge closed-loop eval CLIENT (libero venv). For each LIBERO task and init
state, runs the SAME episode under each fill mode (sync/cache/taylor/bridge) via the
KV-bridge server, recording success. Paired across modes (identical init state) for a
variance-controlled comparison. Shows closed-loop SR degradation of stale reuse and
Taylor vs the learned bridge."""
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


def run_ep(client, env, init_state, task, mode, f, replan, budget, wait=10):
    # Matches the official openpi LIBERO client: success = env.step() done flag,
    # default env (no ignore_done).
    client.infer({"__reset__": True, "__mode__": mode, "__f__": f})
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--modes", default="sync,cache,taylor,bridge")
    ap.add_argument("--f", type=int, default=4)
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--n_tasks", type=int, default=3)
    ap.add_argument("--n_eps", type=int, default=10)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--out", default="out/bridge_closedloop.json")
    args = ap.parse_args()

    modes = args.modes.split(",")
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print("connected to KV-bridge server", flush=True)

    bd = benchmark.get_benchmark_dict()
    suite = bd[args.suite]()
    budget = MAXSTEPS.get(args.suite, 300)
    n_tasks = min(args.n_tasks, suite.n_tasks)
    t0 = time.time()
    records = []
    for tid in range(n_tasks):
        task = suite.get_task(tid)
        inits = suite.get_task_init_states(tid)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        env.seed(0)
        for ep in range(min(args.n_eps, len(inits))):
            for mode in modes:
                succ, steps = run_ep(client, env, inits[ep], task.language,
                                     mode, args.f, args.replan, budget)
                records.append({"task_id": tid, "ep": ep, "mode": mode,
                                "success": bool(succ), "steps": steps})
            row = {m: sum(r["success"] for r in records if r["task_id"] == tid
                          and r["ep"] == ep and r["mode"] == m) for m in modes}
            print(f"  task{tid} ep{ep}: {row}  ({(time.time()-t0)/60:.1f} min)", flush=True)
        env.close()
        srs = {m: np.mean([r["success"] for r in records
                           if r["task_id"] == tid and r["mode"] == m]) for m in modes}
        print(f"  [task{tid} SR] " + "  ".join(f"{m}={srs[m]:.2f}" for m in modes), flush=True)

    summ = {m: {"n": sum(1 for r in records if r["mode"] == m),
                "succ": int(sum(r["success"] for r in records if r["mode"] == m)),
                "sr": float(np.mean([r["success"] for r in records if r["mode"] == m]))}
            for m in modes}
    out = {"suite": args.suite, "f": args.f, "replan": args.replan,
           "n_tasks": n_tasks, "n_eps": args.n_eps, "summary": summ, "records": records}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("\n=== closed-loop SR by fill mode (f=%d) ===" % args.f, flush=True)
    for m in modes:
        print(f"  {m:8s} SR={summ[m]['sr']:.3f} ({summ[m]['succ']}/{summ[m]['n']})", flush=True)
    print("wrote", args.out, flush=True)
    print("DONE_BRIDGE_CLOSEDLOOP", flush=True)


if __name__ == "__main__":
    main()
