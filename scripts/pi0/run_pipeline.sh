#!/bin/bash
# π0 KV Bridge Pipeline
# Reproduces results on LIBERO benchmarks
#
# Prerequisites:
#   - π0 PyTorch checkpoint (e.g., ./checkpoints/pi05_base)
#   - OpenPI installed with LIBERO support
#   - Run from openpi directory: cd baseline/openpi
#
# Usage:
#   bash scripts/pi0/run_pipeline.sh <CHECKPOINT_DIR> <TASK_SUITE> <OUTPUT_DIR> [GPU_ID]
#   Example:
#   bash scripts/pi0/run_pipeline.sh /data/openpi_checkpoints/pi05_libero_pytorch libero_spatial /data/outputs/pi0_spatial "0"

set -e

CHECKPOINT_DIR=${1:?Usage: $0 CHECKPOINT_DIR TASK_SUITE OUTPUT_DIR [GPU_ID]}
TASK_SUITE=${2:?Usage: $0 CHECKPOINT_DIR TASK_SUITE OUTPUT_DIR [GPU_ID]}
OUTPUT_DIR=${3:?Usage: $0 CHECKPOINT_DIR TASK_SUITE OUTPUT_DIR [GPU_ID]}
GPU_ID=${4:-"0"}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QCVLA_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OPENPI_DIR="$QCVLA_DIR/baseline/openpi"
PYTHON="${OPENPI_DIR}/.venv/bin/python"

echo "============================================"
echo "π0 KV Bridge Pipeline"
echo "  Checkpoint: $CHECKPOINT_DIR"
echo "  Suite: $TASK_SUITE"
echo "  Output: $OUTPUT_DIR"
echo "  GPU: $GPU_ID"
echo "============================================"

mkdir -p $OUTPUT_DIR

# Step 1: Collect sync KV data (runs π0 in sync mode, captures KV + embedding)
echo "[Step 1/4] Collecting sync KV data..."
cd $OPENPI_DIR
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON $QCVLA_DIR/scripts/pi0/collect_pi0_kv_data.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --output_path $OUTPUT_DIR/kv_${TASK_SUITE}.h5 \
    --task_suite_name $TASK_SUITE \
    --num_trials_per_task 30

# Step 2: Train KV bridge
echo "[Step 2/4] Training KV bridge..."
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON $QCVLA_DIR/scripts/pi0/train_pi0_bridge_kv.py \
    --data_path $OUTPUT_DIR/kv_${TASK_SUITE}.h5 \
    --output_dir $OUTPUT_DIR/bridge_r0 \
    --epochs 100 --batch_size 4 --lr 3e-4 \
    --hidden_dim 768 --num_blocks 10

# Step 3: Evaluate bridge (freq=3, 2 denoise steps)
echo "[Step 3/4] Evaluating bridge (freq=3, 2 denoise)..."
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON $QCVLA_DIR/scripts/pi0/eval_pi0_bridge_kv.py \
    --bridge_path $OUTPUT_DIR/bridge_r0/best_model.pt \
    --task_suite_name $TASK_SUITE \
    --num_trials_per_task 20 \
    --vlm_freq 3 --num_denoise_steps 2

# Step 4 (optional): DAgger R1 for harder suites
echo "[Step 4/4] DAgger R1 (optional, for LIBERO-10)..."
if [ "$TASK_SUITE" = "libero_10" ]; then
    # Collect DAgger data online
    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON $QCVLA_DIR/scripts/pi0/collect_pi0_dagger_kv_online.py \
        --bridge_path $OUTPUT_DIR/bridge_r0/best_model.pt \
        --output_path $OUTPUT_DIR/dagger_kv_${TASK_SUITE}.h5 \
        --task_suite_name $TASK_SUITE \
        --num_trials_per_task 30 --vlm_freq 3

    # Train R1
    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON $QCVLA_DIR/scripts/pi0/train_pi0_bridge_kv.py \
        --data_path $OUTPUT_DIR/kv_${TASK_SUITE}.h5 \
        --dagger_path $OUTPUT_DIR/dagger_kv_${TASK_SUITE}.h5 \
        --resume $OUTPUT_DIR/bridge_r0/best_model.pt \
        --output_dir $OUTPUT_DIR/bridge_r1 \
        --epochs 100 --batch_size 4 --lr 3e-5 \
        --hidden_dim 768 --num_blocks 10

    # Eval R1
    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON $QCVLA_DIR/scripts/pi0/eval_pi0_bridge_kv.py \
        --bridge_path $OUTPUT_DIR/bridge_r1/best_model.pt \
        --task_suite_name $TASK_SUITE \
        --num_trials_per_task 20 \
        --vlm_freq 3 --num_denoise_steps 2
fi

echo "============================================"
echo "Pipeline complete! Results in $OUTPUT_DIR"
echo "============================================"
