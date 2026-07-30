"""Action-chunk length sweep (baseline for R1 Q5(iii) / R2).

Shows that simply lengthening the open-loop action chunk of the BASE policy (execute
`replan` actions per VLM inference, no bridge) degrades closed-loop SR monotonically —
the competing way to cut VLM calls, and why the bridge (which keeps the action head
reactive every step) is preferable.

All episodes run in sync mode (full VLM every inference); only the number of executed
actions per inference (`replan` = effective chunk length) varies. Paired across replan
values by init state, so we can run a within-episode trend test.

Outputs per-replan SR + Wilson 95% CI, a Cochran-Armitage/logistic trend p-value, and a
paired McNemar test between the shortest and longest chunk.
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


def run_ep(client, env, init_state, task, replan, budget, wait=10):
    # sync mode, VLM every inference; execute `replan` actions per inference.
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
            return True, t, chunk.shape[0]
        t += 1
    return False, t, chunk.shape[0]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h, c + h)


def cochran_armitage(x, y):
    """Cochran-Armitage trend test: is success probability monotone in replan?
    A SCORE test (not Wald), so — unlike logistic regression — it stays valid and
    significant under quasi/complete separation (e.g. replan5=100%, replan9=0%),
    which is exactly the pattern we expect. Scores = replan values. Returns (z, p).
    Treats replan groups as independent (ignores the init-state pairing, so it is
    CONSERVATIVE here — pairing is positively correlated); McNemar covers the pair."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    N = len(y)
    if N == 0:
        return (float("nan"), float("nan"))
    pbar = y.mean()
    tbar = x.mean()
    T = float(np.sum(y * (x - tbar)))                 # Σ x_i(t_i - t̄) over successes
    var = pbar * (1 - pbar) * float(np.sum((x - tbar) ** 2))
    if var <= 0:
        return (0.0, 1.0)
    z = T / math.sqrt(var)
    from math import erf
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / math.sqrt(2))))
    return (z, p)


def mcnemar(pairs_a, pairs_b):
    """Paired McNemar between two replan settings on the same init states.
    a=success@short, b=success@long. Returns (b01, b10, p) two-sided."""
    b01 = sum(1 for a, b in zip(pairs_a, pairs_b) if a == 0 and b == 1)  # short fail, long ok
    b10 = sum(1 for a, b in zip(pairs_a, pairs_b) if a == 1 and b == 0)  # short ok, long fail
    n = b01 + b10
    if n == 0:
        return (b01, b10, 1.0)
    # exact binomial two-sided
    from math import comb
    k = min(b01, b10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n))
    return (b01, b10, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--replans", default="5,6,7,8,9")
    ap.add_argument("--n_tasks", type=int, default=2)
    ap.add_argument("--n_eps", type=int, default=10)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--out", default="out/chunk_sweep.json")
    args = ap.parse_args()

    replans = [int(x) for x in args.replans.split(",")]
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print("connected to KV-bridge server", flush=True)

    bd = benchmark.get_benchmark_dict()
    suite = bd[args.suite]()
    budget = MAXSTEPS.get(args.suite, 300)
    n_tasks = min(args.n_tasks, suite.n_tasks)
    t0 = time.time()
    records = []
    chunk_len_seen = set()
    for tid in range(n_tasks):
        task = suite.get_task(tid)
        inits = suite.get_task_init_states(tid)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        env.seed(0)
        for ep in range(min(args.n_eps, len(inits))):
            for rp in replans:
                succ, steps, clen = run_ep(client, env, inits[ep], task.language, rp, budget)
                chunk_len_seen.add(clen)
                records.append({"task_id": tid, "ep": ep, "replan": rp,
                                "success": int(succ), "steps": steps})
            row = {rp: next(r["success"] for r in records if r["task_id"] == tid
                            and r["ep"] == ep and r["replan"] == rp) for rp in replans}
            print(f"  task{tid} ep{ep}: {row}  ({(time.time()-t0)/60:.1f} min)", flush=True)
        env.close()

    # defensive: persist raw records before stats so a stats bug never loses the run
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"suite": args.suite, "replans": replans, "records": records,
               "_note": "records-only checkpoint; stats added on completion"},
              open(args.out, "w"), indent=2)

    # per-replan SR + Wilson CI
    summ = {}
    for rp in replans:
        rs = [r["success"] for r in records if r["replan"] == rp]
        k, n = int(sum(rs)), len(rs)
        lo, hi = wilson(k, n)
        summ[rp] = {"n": n, "succ": k, "sr": k / n if n else float("nan"),
                    "ci95": [lo, hi]}

    # Cochran-Armitage trend test over all episodes (robust to separation)
    xs = [r["replan"] for r in records]
    ys = [r["success"] for r in records]
    z, p_trend = cochran_armitage(xs, ys)
    lo_rp0, hi_rp0 = min(replans), max(replans)
    sr_drop = summ[lo_rp0]["sr"] - summ[hi_rp0]["sr"]  # effect size: SR(short) - SR(long)

    # paired McNemar shortest vs longest
    lo_rp, hi_rp = min(replans), max(replans)
    keyed = lambda rp: {(r["task_id"], r["ep"]): r["success"]
                        for r in records if r["replan"] == rp}
    a_map, b_map = keyed(lo_rp), keyed(hi_rp)
    keys = sorted(set(a_map) & set(b_map))
    b01, b10, p_mc = mcnemar([a_map[k] for k in keys], [b_map[k] for k in keys])

    out = {"suite": args.suite, "mode": "sync", "replans": replans,
           "n_tasks": n_tasks, "n_eps": args.n_eps, "chunk_len": sorted(chunk_len_seen),
           "summary": summ,
           "trend": {"test": "cochran_armitage", "z": z, "p": p_trend,
                     "sr_drop_short_minus_long": sr_drop,
                     "short_replan": lo_rp0, "long_replan": hi_rp0},
           "mcnemar": {"short": lo_rp, "long": hi_rp, "n_pairs": len(keys),
                       "short_fail_long_ok": b01, "short_ok_long_fail": b10, "p": p_mc},
           "records": records}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    print("\n=== action-chunk length sweep (sync base policy, %s) ===" % args.suite, flush=True)
    print("  chunk produced by policy = %s actions" % sorted(chunk_len_seen), flush=True)
    for rp in replans:
        s = summ[rp]
        print(f"  replan={rp}: SR={s['sr']:.3f} ({s['succ']}/{s['n']})  "
              f"95%CI[{s['ci95'][0]:.3f},{s['ci95'][1]:.3f}]", flush=True)
    print(f"  trend (Cochran-Armitage): z={z:.2f}  p={p_trend:.2e}  "
          f"| SR drop replan{lo_rp0}->{hi_rp0} = {sr_drop*100:+.1f}pp", flush=True)
    print(f"  McNemar replan{lo_rp} vs replan{hi_rp}: "
          f"+{b01}/-{b10} over {len(keys)} pairs  p={p_mc:.3e}", flush=True)
    print("wrote", args.out, flush=True)
    print("DONE_CHUNK_SWEEP", flush=True)


if __name__ == "__main__":
    main()
