#!/bin/bash

# Sbatch script for testing prefetch configuration with exp2
# Uses byt5-small model for faster iteration

#SBATCH -J test_prefetch_exp2                      # Job name
#SBATCH --ntasks=1                                 # Number of tasks
#SBATCH --nodes=1                                  # Single node
#SBATCH --partition=a100-galvani                   # A100 partition
#SBATCH --time=0-06:00                             # 6 hours should be plenty for testing
#SBATCH --gres=gpu:1                               # 1 GPU
#SBATCH --output=/mnt/lustre/work/oh/arubinstein17/math_prover_project/ReProver/retrieval/out/test/exp2_train_random-%j.out
#SBATCH --error=/mnt/lustre/work/oh/arubinstein17/math_prover_project/ReProver/retrieval/out/test/exp2_train_random-%j.err

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

# Activate conda environment if needed (uncomment and modify as needed)
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate your_env

# Create output directory for profile
mkdir -p out/test/profiles

# Compute Phase - Run with cProfile for SnakeViz analysis
# Profile will be saved to a .prof file that can be viewed with snakeviz
echo "========== Starting Training with Profiler =========="
python -m cProfile -o out/test/profiles/exp2_train_random_${SLURM_JOB_ID}.prof \
    retrieval/main.py fit --config retrieval/confs/test/exp2_train_random.yaml

echo ""
echo "========== Training Complete =========="
echo "Profile saved to: out/test/profiles/exp2_train_random_${SLURM_JOB_ID}.prof"
echo "View with: snakeviz out/test/profiles/exp2_train_random_${SLURM_JOB_ID}.prof"
