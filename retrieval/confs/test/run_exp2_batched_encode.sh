#!/bin/bash

# Sbatch script for testing batched encoding (single _encode pass)
# Runs 1000 training steps with all inputs encoded in one forward pass

#SBATCH -J batched_encode                          # Job name
#SBATCH --ntasks=1                                 # Number of tasks
#SBATCH --nodes=1                                  # Single node
#SBATCH --partition=a100-galvani                   # A100 partition
#SBATCH --time=0-02:00                             # 2 hours
#SBATCH --gres=gpu:1                               # 1 GPU
#SBATCH --output=/mnt/lustre/work/oh/arubinstein17/math_prover_project/ReProver/out/test/batched_encode-%j.out
#SBATCH --error=/mnt/lustre/work/oh/arubinstein17/math_prover_project/ReProver/out/test/batched_encode-%j.err

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
echo "========== Starting Training (batched encode, 1000 steps) =========="
python -m retrieval.main fit \
    --config retrieval/confs/test/exp2_batched_encode.yaml

echo ""
echo "========== Training Complete =========="
