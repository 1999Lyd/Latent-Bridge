# Rebuttal Experiments: Real-Data and Action-Level Evidence

Scripts behind the additional results in the paper's revision. Each script has a small
CONFIG block of paths at the top (marked `/path/to/...`); point them at your checkpoints,
data, and virtual environments before running. All KV-space experiments use the pi0.5
base checkpoint under the DROID input transform, so the only variable between the real
and sim arms is the pixel content.

## Scripts

| Script | Purpose |
|---|---|
| `kv_common.py` | Shared: load pi0.5, decode DROID/LIBERO episodes, extract pre-RoPE prefix KV |
| `collect_kv_dataset.py` | Store per-step KV/embedding/state/action sequences (real or sim) |
| `analyze_redundancy.py` | Prefix-KV cosine vs. wall-clock skip interval, real vs. sim |
| `measure_temporal_similarity.py` | Per-layer consecutive-step cosine, image and text tokens |
| `delta_fidelity.py` | Per-step recovered-KV cosine for cache / Taylor / bridge |
| `chain_eval.py` | Chained recovery at skip period f (cache / Taylor order-1,2 / bridge) |
| `motion_analysis.py` | Recovery cosine binned by end-effector translation magnitude |
| `correlate_quality_sr.py` | Feature-recovery quality vs. closed-loop SR (Spearman) |
| `kv_bridge_server_std.py` | Serving policy with switchable KV fill (sync/cache/taylor/bridge) + seeded flow noise |
| `bridge_eval_client.py` | Closed-loop LIBERO eval client, paired init states across modes |
| `action_fidelity.py` | Per-step action RMSE vs. sync-conditioned reference (obs replay, seeded noise) |
| `real_action_fidelity.py` | Same protocol env-free over real DROID + sim LIBERO streams |
| `sync_chunk_eval.py` | Sync SR at chunk length 5 vs. 10 (4 suites, 3 seeds) |
| `chunk_sweep.py` | Chunk-length sweep with Cochran-Armitage trend + McNemar tests |

## Headline numbers (A100)

Temporal redundancy at matched skip interval (dt ~ 0.2 s): prefix-KV cosine **0.973 real
(DROID) vs 0.950 sim (LIBERO)**, real >= sim at every one of the 18 layers.

Monotone fidelity -> action -> SR chain (pi0.5, f=4, 3 seeds):

| Method | KV cos (sim/real) | action RMSE (sim/real) | mean closed-loop SR |
|---|---|---|---|
| sync | 1.00 / 1.00 | 0 / 0 (ref.) | 96.96 |
| bridge | 0.976 / 0.981 | 0.136 / 0.098 | 96.92 |
| stale reuse | 0.951 / 0.974 | 0.279 / 0.176 | 56.38 |
| Taylor | 0.910 / 0.951 | 0.298 / 0.189 | 46.67 |

The policy's own sampling-noise floor (sync replayed with re-sampled flow noise) is
0.037 on real streams. R0-only bridge trained fully offline on real trajectories beats
stale reuse at every skip interval (0.982 vs 0.975 next-step KV cosine at f=2). Under
fast motion the bridge's recovery advantage over stale reuse is 4x its slow-motion value
(+0.019 vs +0.005). Chunk-length study: sync 96.96 -> 97.50 (chunk 5 -> 10, n=240 direct
measurement), stale reuse 56.38 -> 38.92, bridge 96.92 -> 95.83.
