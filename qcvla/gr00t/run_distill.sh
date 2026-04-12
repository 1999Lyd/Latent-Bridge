#!/bin/bash
#SBATCH --job-name=distill_data
#SBATCH --output=distill_data.out
#SBATCH --error=distill_data.err
#SBATCH --partition=athena-genai
#remove nodelist to use any available node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

echo "Generating distillation data"
echo "Start: $(date)"

cd /zpool-00/home/yl817/QCVLA
source benchmarks/Isaac-GR00T/.venv/bin/activate

python -m qcvla.gr00t.generate_distill_data \
    --teacher_path outputs/stable_dynamic_bridge/best_model.pt \
    --data_path outputs/latent_bridge_data/multilayer_train_data.h5 \
    --output_path outputs/latent_bridge_data/distill_data.h5 \
    --num_steps 10 \
    --batch_size 128 \
    --num_gpus 1

echo "Done: $(date)"
