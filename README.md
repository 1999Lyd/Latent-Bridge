# Latent Bridge

**Latent Bridge: Feature Delta Prediction for Efficient Dual-System Vision-Language-Action Model Inference**

[Paper](docs/neurips_draft/neurips_2026.pdf) | [Checkpoints](https://drive.google.com/drive/folders/1oRtVNAnVUvV52AFCYJ6jwn8qUkAx7Fz_?usp=sharing)

## Overview

Dual-system Vision-Language-Action (VLA) models achieve state-of-the-art robotic manipulation but are bottlenecked by the VLM backbone at every control step. **Latent Bridge** is a lightweight model that predicts VLM feature deltas, enabling the VLM to run at reduced frequency while maintaining policy performance.

Key results:
- **95--100% task success retention** across 4 LIBERO suites and 24 RoboCasa tasks
- **1.65--1.73x net per-episode inference speedup** on A100 GPU
- **50--75% VLM compute savings** 
- Validated on two VLA architectures: **GR00T-N1** (feature-space) and **pi0.5** (KV-cache)

## Architecture

```
VLM step (every k steps):     obs -> VLM backbone -> z_t (fresh features) -> action head -> action
Bridge step (in between):     z_t + state + action -> Bridge -> delta -> z_{t+1} = z_t + delta -> action head -> action
```

The bridge is a lightweight DiT (~148M params for pi0.5, ~186M for GR00T) with:
- AdaLN conditioning on robot state and previous action
- Cross-attention to stable context features
- Zero-initialized output (starts as copy baseline)
- DAgger refinement for distribution shift correction

## Project Structure

```
Latent-Bridge/
├── qcvla/                          # Core model components
│   ├── model/
│   │   ├── rectified_flow_bridge.py  # DiT blocks (DiTCrossBlock, DiTFinalLayer)
│   │   └── common.py                # Shared utilities
│   └── __init__.py
├── scripts/
│   ├── groot/                      # GR00T-N1 pipeline
│   │   ├── collect_multilayer_data.py   # Step 1: Collect sync training data
│   │   ├── train_single_step_dit.py     # Step 2: Train bridge (R0 + DAgger R1)
│   │   ├── collect_dagger_bridge_data.py # Step 3: Collect DAgger data
│   │   ├── eval_stable_dynamic_bridge.py # Step 4: Evaluate
│   │   └── run_pipeline.sh              # End-to-end pipeline
│   ├── pi0/                        # pi0.5 pipeline
│   │   ├── pi0_bridge_kv.py             # KV bridge model (Pi0BridgeKV)
│   │   ├── collect_pi0_kv_data.py       # Step 1: Collect KV + embedding data
│   │   ├── train_pi0_bridge_kv.py       # Step 2: Train KV bridge
│   │   ├── eval_pi0_bridge_kv.py        # Step 3: Evaluate
│   │   ├── generate_pi0_dagger_kv.py    # Optional: Generate DAgger data offline
│   │   ├── collect_pi0_dagger_kv_online.py  # Optional: Online DAgger collection
│   │   └── run_pipeline.sh              # End-to-end pipeline
│   └── baselines/                  # Baseline comparisons
│       ├── eval_fastv_baseline.py       # FastV token pruning
│       ├── eval_specprune_baseline.py   # SpecPrune-VLA
│       └── eval_vlacache_baseline.py    # VLA-Cache
└── docs/                           # Paper and documentation
```

## Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+

### For GR00T-N1 experiments
```bash
# Clone this repo
git clone https://github.com/1999Lyd/Latent-Bridge.git
cd Latent-Bridge

# Install GR00T dependencies (Isaac-GR00T)
# Follow: https://github.com/NVIDIA/Isaac-GR00T
pip install -e .

# Install LIBERO
pip install libero
```

### For pi0.5 experiments
```bash
# Install OpenPI (pi0.5 inference framework)
# Follow: https://github.com/Physical-Intelligence/openpi
cd baseline/openpi
pip install -e .
```

## Checkpoints

Download pre-trained bridge checkpoints:

| Model | Suite | Bridge Type | SR (Sync) | SR (Bridge) | Retention | Download |
|-------|-------|-------------|-----------|-------------|-----------|----------|
| pi0.5 | LIBERO-Spatial | KV Bridge | 98.7% | 99.0% | 100.3% | [link](https://drive.google.com/drive/folders/1oRtVNAnVUvV52AFCYJ6jwn8qUkAx7Fz_?usp=sharing) |
| pi0.5 | LIBERO-Object | KV Bridge | 98.3% | 99.0% | 100.7% | [link](https://drive.google.com/drive/folders/1oRtVNAnVUvV52AFCYJ6jwn8qUkAx7Fz_?usp=sharing) |
| pi0.5 | LIBERO-Goal | KV Bridge | 97.0% | 97.0% | 100.0% | [link](https://drive.google.com/drive/folders/1oRtVNAnVUvV52AFCYJ6jwn8qUkAx7Fz_?usp=sharing) |
| pi0.5 | LIBERO-10 | KV Bridge | 94.0% | 92.0% | 97.9% | [link](https://drive.google.com/drive/folders/1oRtVNAnVUvV52AFCYJ6jwn8qUkAx7Fz_?usp=sharing) |
| GR00T | LIBERO-Goal | Feature Bridge | 97.5% | 95.0% | 97.4% | [link](https://drive.google.com/drive/folders/1oRtVNAnVUvV52AFCYJ6jwn8qUkAx7Fz_?usp=sharing) |

All checkpoints: [Google Drive](https://drive.google.com/drive/folders/1oRtVNAnVUvV52AFCYJ6jwn8qUkAx7Fz_?usp=sharing) (4.0 GB)

## Quick Start

### pi0.5 KV Bridge (Recommended)

**1. Collect training data**
```bash
python scripts/pi0/collect_pi0_kv_data.py \
    --checkpoint_dir <path_to_pi05_pytorch> \
    --task_suite libero_spatial \
    --output_path outputs/pi0_bridge_data/kv_spatial.h5
```

**2. Train bridge**
```bash
python scripts/pi0/train_pi0_bridge_kv.py \
    --data_path outputs/pi0_bridge_data/kv_spatial.h5 \
    --output_dir outputs/pi0_bridge_kv_spatial \
    --epochs 50 --batch_size 4 --lr 3e-4 \
    --hidden_dim 768 --num_blocks 10
```

**3. Evaluate**
```bash
python scripts/pi0/eval_pi0_bridge_kv.py \
    --bridge_path outputs/pi0_bridge_kv_spatial/best_model.pt \
    --task_suite_name libero_spatial \
    --vlm_freq 3 --num_denoise_steps 1
```

### GR00T-N1 Feature Bridge

See `scripts/groot/run_pipeline.sh` for the full pipeline, or run individual steps:

```bash
# Step 1: Collect sync data
python scripts/groot/collect_multilayer_data.py \
    --model_path <path_to_groot_model> \
    --task_suite libero_goal \
    --output_path outputs/sync_data.h5

# Step 2: Train R0 bridge
python scripts/groot/train_single_step_dit.py \
    --data_path outputs/sync_data.h5 \
    --output_dir outputs/bridge_r0 \
    --epochs 200 --lr 3e-4

# Step 3: Collect DAgger data
python scripts/groot/collect_dagger_bridge_data.py \
    --model_path <path_to_groot_model> \
    --bridge_path outputs/bridge_r0/best_model_dit.pt \
    --task_suite libero_goal \
    --output_path outputs/dagger_data.h5

# Step 4: Train R1 bridge (DAgger fine-tuning)
python scripts/groot/train_single_step_dit.py \
    --data_path outputs/sync_data.h5 \
    --dagger_data_path outputs/dagger_data.h5 \
    --resume outputs/bridge_r0/best_model_dit.pt \
    --reset_best --epochs 216 --lr 3e-4

# Step 5: Evaluate
python scripts/groot/eval_stable_dynamic_bridge.py \
    --model_path <path_to_groot_model> \
    --ar_bridge_path outputs/bridge_r1/best_model_dit.pt \
    --task_suite libero_goal \
    --modes sync autoregressive_bridge --vlm_freq 3
```

## Main Results

### pi0.5 (bf16+compile, 1.47x avg speedup)

| Suite | Sync SR | Bridge SR | Retention | VLM Savings |
|-------|---------|-----------|-----------|-------------|
| LIBERO-Spatial | 98.7% | 99.0% | 100.3% | 65% |
| LIBERO-Object | 98.3% | 99.0% | 100.7% | 65% |
| LIBERO-Goal | 97.0% | 97.0% | 100.0% | 65% |
| LIBERO-10 | 94.0% | 92.0% | 97.9% | 65% |

### GR00T-N1 (1.73x speedup)

| Suite | Sync SR | Bridge SR | Retention | VLM Savings |
|-------|---------|-----------|-----------|-------------|
| LIBERO-Spatial | 96.0% | 96.0% | 100.0% | 67% |
| LIBERO-Object | 100.0% | 98.0% | 98.0% | 67% |
| LIBERO-Goal | 97.5% | 95.0% | 97.4% | 50% |
| LIBERO-10 | 93.0% | 89.0% | 95.7% | 57.6% |

## Citation

```bibtex
@inproceedings{liu2026latentbridge,
  title={Latent Bridge: Feature Delta Prediction for Efficient Dual-System Vision-Language-Action Model Inference},
  author={Liu, Yudong and Lin, Yueqian and Wang, Qinsi and Tang, Zijia and Zheng, Yuxi and Li, Yuan and Li, Yi and Liu, Shuangjun and Zhang, Shuai and Jing, Taotao and Gao, Dashan and Bi, Ning and Sun, Jingwei and Chen, Yiran and Li, Hai},
  year={2026}
}
```

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgments

This work was supported by Duke University, Qualcomm AI Research, and University of Florida. We thank the teams behind [GR00T](https://github.com/NVIDIA/Isaac-GR00T), [OpenPI](https://github.com/Physical-Intelligence/openpi), and [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) for their open-source frameworks.
