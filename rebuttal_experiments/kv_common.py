"""Shared utilities: load pi0.5 base, decode real (DROID) and sim (LIBERO) episodes,
extract pre-RoPE prefix KV exactly as Latent-Bridge's collect_pi0_kv_data.py does.

The SAME checkpoint and the SAME input transform are used for both sources, so the only
variable between the real and sim arms is the pixel content.
"""
import io
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

OPENPI = "/path/to/envs/openpi_robocasa"
sys.path.insert(0, os.path.join(OPENPI, "src"))

# H200 fp32 matmuls are ~15x slower without TF32; the vision tower keeps a few fp32
# ops, so enable TF32 globally. This does not touch the bf16 language-model path.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

STUDY = pathlib.Path("/path/to/study")
CKPT = f"{OPENPI}/checkpoints/pi05_base_pytorch"
DROID_DIR = STUDY / "data/droid_100"
LIBERO_DIR = pathlib.Path(
    "/path/to/envs/lerobot_datasets/local/libero_std"
)

S = 768        # image tokens in the pi0.5 prefix (3 slots x 256)
N_LAYERS = 18  # Gemma-2B layers
KV_DIM = 512   # 256 K + 256 V


# --------------------------------------------------------------------------- model
def load_policy(device="cuda:0", num_steps=1):
    """pi0.5 base (real-data pretrained) under the DROID input transform."""
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    import openpi.shared.normalize as _normalize

    cfg = _config.get_config("pi05_droid")
    norm_stats = _normalize.load(STUDY / "assets/droid")
    policy = _policy_config.create_trained_policy(
        cfg, CKPT, norm_stats=norm_stats,
        pytorch_device=device, sample_kwargs={"num_steps": num_steps},
    )
    return policy


def _get_layers(lm):
    if hasattr(lm, "layers"):
        return lm.layers
    return lm.model.layers


def attach_hook(policy):
    """Capture the language model's hidden states on every forward."""
    model = policy._model
    lm = model.paligemma_with_expert.paligemma.language_model
    lm.config._attn_implementation = "eager"
    captured = [None]
    orig = lm.forward

    def hook(*a, **kw):
        kw["output_hidden_states"] = True
        out = orig(*a, **kw)
        captured[0] = out.hidden_states
        return out

    lm.forward = hook
    return lm, captured


@torch.no_grad()
def prefix_kv_from_hidden(lm, hs):
    """Pre-RoPE K,V for all 18 layers over the 768 image tokens -> [18, 768, 512] fp16.

    hs[l] is the *input* to layer l, so applying layer l's input_layernorm + k_proj/v_proj
    reproduces exactly the K,V that layer l writes to the cache (before RoPE).
    """
    layers = _get_layers(lm)
    dtype = layers[0].self_attn.k_proj.weight.dtype
    kv = np.zeros((N_LAYERS, S, KV_DIM), dtype=np.float16)
    for l in range(N_LAYERS):
        h = hs[l][:, :S, :].to(dtype)
        normed = layers[l].input_layernorm(h)
        if isinstance(normed, tuple):
            normed = normed[0]
        k = layers[l].self_attn.k_proj(normed)
        v = layers[l].self_attn.v_proj(normed)
        kv[l] = torch.cat([k, v], dim=-1)[0].float().cpu().numpy().astype(np.float16)
    return kv


@torch.no_grad()
def _prefix_forward(policy, element):
    """Run ONLY the PaliGemma prefix (SigLIP + Gemma-2B), skipping the action expert
    and the denoising loop. The prefix KV is computed identically to a full
    policy.infer(), so the captured hidden states match the release's collector
    exactly -- but this avoids the ~minutes-long fp32 denoise path.
    """
    import jax
    from openpi.models import model as _model
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    model = policy._model
    dev = policy._pytorch_device
    inputs = policy._input_transform(dict(element))
    tin = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(dev)[None, ...], inputs)
    obs = _model.Observation.from_dict(tin)

    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(obs, train=False)
    pe, ppm, pam = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    a2d = make_att_2d_masks(ppm, pam)
    pos = torch.cumsum(ppm, dim=1) - 1
    a4d = model._prepare_attention_masks_4d(a2d)
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    model.paligemma_with_expert.forward(
        attention_mask=a4d, position_ids=pos, past_key_values=None,
        inputs_embeds=[pe, None], use_cache=True)


@torch.no_grad()
def run_step(policy, captured, element):
    """One prefix forward; returns (kv [18,768,512] fp16, emb [768,2048] fp16)."""
    captured[0] = None
    _prefix_forward(policy, element)
    hs = captured[0]
    if hs is None:
        raise RuntimeError("hidden states were not captured")
    lm = policy._model.paligemma_with_expert.paligemma.language_model
    kv = prefix_kv_from_hidden(lm, hs)
    emb = hs[0][0, :S].float().cpu().numpy().astype(np.float16)
    return kv, emb


# ----------------------------------------------------------------------- real DROID
def droid_episodes():
    meta = pd.read_parquet(DROID_DIR / "meta/episodes/chunk-000/file-000.parquet")
    tasks = pd.read_parquet(DROID_DIR / "meta/tasks.parquet")
    idx2task = {int(v): str(k) for k, v in zip(tasks.index, tasks["task_index"])}
    data = pd.read_parquet(DROID_DIR / "data/chunk-000/file-000.parquet")
    return meta, data, idx2task


def decode_video_range(path, from_ts, to_ts):
    """Decode frames whose pts fall in [from_ts, to_ts). Returns list of HWC uint8."""
    import av

    frames = []
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        # seek slightly before the segment, then filter by timestamp
        c.seek(max(int((from_ts - 1.0) / stream.time_base), 0), stream=stream)
        for frame in c.decode(video=0):
            t = float(frame.pts * stream.time_base)
            if t >= to_ts - 1e-6:
                break
            if t >= from_ts - 1e-6:
                frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def droid_episode_frames(meta_row, data):
    ep = int(meta_row["episode_index"])
    lo, hi = int(meta_row["dataset_from_index"]), int(meta_row["dataset_to_index"])
    sub = data.iloc[lo:hi]
    out = {}
    for cam, key in [("exterior_image_1_left", "ext1"), ("wrist_image_left", "wrist")]:
        p = DROID_DIR / f"videos/observation.images.{cam}/chunk-000/file-000.mp4"
        f_ts = float(meta_row[f"videos/observation.images.{cam}/from_timestamp"])
        t_ts = float(meta_row[f"videos/observation.images.{cam}/to_timestamp"])
        out[key] = decode_video_range(p, f_ts, t_ts)
    state = np.stack(sub["observation.state"].values).astype(np.float32)
    action = np.stack(sub["action"].values).astype(np.float32)
    return out, state, action, ep


# ------------------------------------------------------------------------ sim LIBERO
def libero_episode_files(n=None):
    fs = sorted((LIBERO_DIR / "data/chunk-000").glob("*.parquet"))
    return fs if n is None else fs[:n]


_LIBERO_TASKS = None


def libero_task_map():
    global _LIBERO_TASKS
    if _LIBERO_TASKS is None:
        _LIBERO_TASKS = {}
        p = STUDY / "assets/libero_tasks.jsonl"
        if p.exists():
            for line in open(p):
                line = line.strip()
                if line:
                    d = json.loads(line)
                    _LIBERO_TASKS[int(d["task_index"])] = d["task"]
    return _LIBERO_TASKS


def libero_episode_frames(path):
    from PIL import Image

    df = pd.read_parquet(path)

    def dec(col):
        return [np.array(Image.open(io.BytesIO(v["bytes"])).convert("RGB"))
                for v in df[col].values]

    imgs = {"ext1": dec("image"), "wrist": dec("wrist_image")}
    state = np.stack(df["state"].values).astype(np.float32)
    action = np.stack(df["actions"].values).astype(np.float32)
    ti = int(df["task_index"].iloc[0])
    task = libero_task_map().get(ti, "complete the task")
    return imgs, state, action, int(df["episode_index"].iloc[0]), task


# ------------------------------------------------------------------------- elements
def make_element(ext1, wrist, state, prompt):
    """Pack one observation into the DroidInputs schema (identical for real and sim)."""
    st = np.asarray(state, dtype=np.float32)
    joint = st[:7] if st.shape[0] >= 7 else np.pad(st, (0, 7 - st.shape[0]))
    grip = np.array([st[7]] if st.shape[0] > 7 else [st[-1]], dtype=np.float32)
    return {
        "observation/exterior_image_1_left": np.ascontiguousarray(ext1),
        "observation/wrist_image_left": np.ascontiguousarray(wrist),
        "observation/joint_position": joint.astype(np.float32),
        "observation/gripper_position": grip,
        "prompt": str(prompt),
    }


# --------------------------------------------------------------------------- metrics
def per_layer_cosine(a, b):
    """Mean per-token cosine between two [18,768,512] arrays -> [18]."""
    x = a.astype(np.float32)
    y = b.astype(np.float32)
    num = (x * y).sum(-1)
    den = np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1) + 1e-8
    return (num / den).mean(-1)
