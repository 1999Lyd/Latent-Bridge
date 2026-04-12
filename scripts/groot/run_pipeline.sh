#!/bin/bash
# GR00T Latent Bridge Pipeline
# Reproduces results on LIBERO benchmarks (Spatial, Object, Goal, LIBERO-10)
#
# Prerequisites:
#   - GR00T finetuned checkpoint (e.g., outputs/libero_spatial_finetune/checkpoint-20000)
#   - LIBERO benchmark installed
#   - CUDA GPUs available
#
# Usage:
#   bash scripts/groot/run_pipeline.sh <MODEL_PATH> <TASK_SUITE> <OUTPUT_DIR> [GPU_IDS]
#   Example:
#   bash scripts/groot/run_pipeline.sh outputs/libero_spatial_finetune/checkpoint-20000 libero_spatial outputs/bridge_spatial "0,1"

set -e

MODEL_PATH=${1:?Usage: $0 MODEL_PATH TASK_SUITE OUTPUT_DIR [GPU_IDS]}
TASK_SUITE=${2:?Usage: $0 MODEL_PATH TASK_SUITE OUTPUT_DIR [GPU_IDS]}
OUTPUT_DIR=${3:?Usage: $0 MODEL_PATH TASK_SUITE OUTPUT_DIR [GPU_IDS]}
GPU_IDS=${4:-"0,1"}
PYTHON=${PYTHON:-$(which python)}

echo "============================================"
echo "GR00T Latent Bridge Pipeline"
echo "  Model: $MODEL_PATH"
echo "  Suite: $TASK_SUITE"
echo "  Output: $OUTPUT_DIR"
echo "  GPUs: $GPU_IDS"
echo "============================================"

mkdir -p $OUTPUT_DIR

# Step 1: Collect sync data (R0 training data + sync SR)
echo "[Step 1/6] Collecting sync data..."
CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON scripts/groot/collect_multilayer_data.py \
    --model_path $MODEL_PATH \
    --output_path $OUTPUT_DIR/sync_data.h5 \
    --task_suites $TASK_SUITE \
    --n_episodes_per_task 30

# Step 2: Train R0 bridge
echo "[Step 2/6] Training R0 bridge..."
CUDA_VISIBLE_DEVICES=${GPU_IDS%%,*} $PYTHON scripts/groot/train_single_step_dit.py \
    --data_path $OUTPUT_DIR/sync_data.h5 \
    --output_dir $OUTPUT_DIR/r0 \
    --epochs 100 --batch_size 64 --lr 3e-4 --seq_len 0 --num_gpus 1

# Step 3: Collect DAgger R1 data
echo "[Step 3/6] Collecting DAgger R1 data..."
CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON scripts/groot/collect_dagger_bridge_data.py \
    --model_path $MODEL_PATH \
    --bridge_path $OUTPUT_DIR/r0/best_model_dit.pt \
    --output_path $OUTPUT_DIR/dagger_r1_data.h5 \
    --task_suite $TASK_SUITE \
    --n_episodes_per_task 30

# Step 4: Train R1 bridge (DAgger)
echo "[Step 4/6] Training R1 bridge..."
CUDA_VISIBLE_DEVICES=${GPU_IDS%%,*} $PYTHON scripts/groot/train_single_step_dit.py \
    --data_path $OUTPUT_DIR/sync_data.h5 \
    --dagger_data_path $OUTPUT_DIR/dagger_r1_data.h5 \
    --output_dir $OUTPUT_DIR/r1 \
    --epochs 100 --batch_size 64 --lr 3e-5 --seq_len 0 --num_gpus 1 \
    --resume $OUTPUT_DIR/r0/best_model_dit.pt --reset_best

# Step 5: Evaluate R1 bridge
echo "[Step 5/6] Evaluating R1 bridge..."
CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON scripts/groot/eval_stable_dynamic_bridge.py \
    --model_path $MODEL_PATH \
    --ar_bridge_path $OUTPUT_DIR/r1/best_model_dit.pt \
    --task_suite $TASK_SUITE \
    --n_episodes 20 \
    --modes sync autoregressive_bridge \
    --use_init_states

# Step 6 (optional): LoRA + Phase-Aware for LIBERO-10
if [ "$TASK_SUITE" = "libero_10" ]; then
    echo "[Step 6/6] LoRA + Phase-Aware (LIBERO-10 only)..."

    # Collect LoRA data
    CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON scripts/groot/collect_online_action_data.py \
        --model_path $MODEL_PATH \
        --bridge_path $OUTPUT_DIR/r1/best_model_dit.pt \
        --output_path $OUTPUT_DIR/lora_data.h5 \
        --task_suite $TASK_SUITE --n_episodes_per_task 30

    # Train LoRA
    CUDA_VISIBLE_DEVICES=${GPU_IDS%%,*} $PYTHON scripts/groot/train_action_head_lora.py \
        --data_path $OUTPUT_DIR/lora_data.h5 \
        --model_path $MODEL_PATH \
        --output_dir $OUTPUT_DIR/lora \
        --max_steps 3000 --lr 1e-4 --batch_size 32

    # Eval with LoRA + Phase-Aware
    CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON scripts/groot/eval_stable_dynamic_bridge.py \
        --model_path $MODEL_PATH \
        --ar_bridge_path $OUTPUT_DIR/r1/best_model_dit.pt \
        --lora_path $OUTPUT_DIR/lora/best_adapter \
        --task_suite $TASK_SUITE \
        --n_episodes 20 \
        --modes phase_aware_bridge \
        --nav_vlm_freq 2 --trans_vlm_freq 3 --manip_vlm_freq 4 \
        --nav_threshold 0.5 --manip_threshold 0.1 \
        --use_init_states
fi

echo "============================================"
echo "Pipeline complete! Results in $OUTPUT_DIR"
echo "============================================"
