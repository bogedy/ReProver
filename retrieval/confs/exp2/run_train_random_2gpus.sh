#!/bin/bash

# Sbatch script for exp2 training with google/byt5-small on 2 GPUs (DDP)

#SBATCH -J exp2_2gpus                              # Job name
#SBATCH --ntasks-per-node=2                        # 2 tasks (1 per GPU for DDP)
#SBATCH --nodes=1                                  # Single node
#SBATCH --partition=a100-galvani                   # A100 partition
#SBATCH --time=3-00:00                             # 3 days (max for a100-galvani)
#SBATCH --gres=gpu:2                               # 2 GPUs
#SBATCH --output=/mnt/lustre/work/oh/arubinstein17/math_prover_project/ReProver/out/exp2/train_random_2gpus-%j.out
#SBATCH --error=/mnt/lustre/work/oh/arubinstein17/math_prover_project/ReProver/out/exp2/train_random_2gpus-%j.err

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
echo "========== Starting Training (exp2, byt5-small, 2 GPUs DDP) =========="
srun python -m retrieval.main fit \
    --config retrieval/confs/exp2/train_random_2gpus.yaml

echo ""
echo "========== Training Complete =========="
