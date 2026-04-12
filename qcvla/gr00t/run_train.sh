#!/bin/bash
#SBATCH --job-name=gr00t_bridge
#SBATCH --output=gr00t_bridge.out
#SBATCH --error=gr00t_bridge.err
#SBATCH --time=24:00:00
#SBATCH --partition=athena-genai
#SBATCH --nodelist=node5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:7
#SBATCH --mem=200G

echo "Training GR00T FlowBridge with DDP (8 GPUs)"
echo "Start: $(date)"
echo "t=0 focused sampling (50%)"

cd /zpool-00/home/yl817/QCVLA
source benchmarks/Isaac-GR00T/.venv/bin/activate

torchrun --nproc_per_node=7 --master_port=29500 \
    -m qcvla.gr00t.train_ddp \
    --data_path outputs/latent_bridge_data/multilayer_train_data.h5 \
    --output_dir outputs/gr00t_bridge \
    --hidden_dim 512 \
    --num_layers 4 \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --t0_prob 0.5 \
    --num_workers 4

echo "Done: $(date)"
