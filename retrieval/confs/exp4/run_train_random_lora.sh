#!/bin/bash

# Sbatch script for exp4 LoRA training with BAAI/bge-m3 on 1 GPU

#SBATCH -J exp4_lora                               # Job name
#SBATCH --ntasks-per-node=1                        # 1 task
#SBATCH --nodes=1                                  # Single node
#SBATCH --partition=a100-galvani                   # A100 partition
#SBATCH --time=3-00:00                             # 3 days (max for a100-galvani)
#SBATCH --gres=gpu:1                               # 1 GPU
#SBATCH --output=/mnt/lustre/work/oh/arubinstein17/math_prover_project/ReProver/out/exp4/train_random_lora-%j.out
#SBATCH --error=/mnt/lustre/work/oh/arubinstein17/math_prover_project/ReProver/out/exp4/train_random_lora-%j.err

# Diagnostic Phase
echo "========== Job Info =========="
scontrol show job $SLURM_JOB_ID
echo ""
echo "========== Node Info =========="
hostname
pwd
nvidia-smi
echo ""

# Setup Phase
cd /mnt/lustre/work/oh/arubinstein17/math_prover_project/ReProver

# Set torch extensions dir to avoid permission issues
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions

# Compute Phase
echo "========== Starting Training (exp4, bge-m3, 1 GPU, LoRA) =========="
srun python -m retrieval.main fit \
    --config retrieval/confs/exp4/train_random_lora.yaml

echo ""
echo "========== Training Complete =========="
