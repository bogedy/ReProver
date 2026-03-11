#!/bin/bash

# Sbatch script for profiling training with cProfile + snakeviz
# Profiles only the trainer.fit() call, excluding setup

#SBATCH -J profile_exp2                            # Job name
#SBATCH --ntasks=1                                 # Number of tasks
#SBATCH --nodes=1                                  # Single node
#SBATCH --partition=a100-galvani                   # A100 partition
#SBATCH --time=0-00:10                             # 10 minutes
#SBATCH --gres=gpu:1                               # 1 GPU
#SBATCH --output=/mnt/lustre/work/oh/arubinstein17/math_prover_project/ReProver/out/test/profile_exp2-%j.out
#SBATCH --error=/mnt/lustre/work/oh/arubinstein17/math_prover_project/ReProver/out/test/profile_exp2-%j.err

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

# ── Profiler Configuration ──
export PROFILE_STEPS=150       # ~1 minute of training, no validation
export PROFILE_OUTPUT=out/test/profiles/training_${SLURM_JOB_ID}.prof

# Note: no "fit" subcommand — run=False mode doesn't use subcommands
echo "========== Starting Training with cProfile =========="
python -m retrieval.profile_training \
    --config retrieval/confs/test/exp2_train_random.yaml

echo ""
echo "========== Training Complete =========="
echo "Profile saved to: ${PROFILE_OUTPUT}"
echo "View with: snakeviz ${PROFILE_OUTPUT}"
