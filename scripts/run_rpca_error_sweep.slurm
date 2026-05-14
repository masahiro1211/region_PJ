#!/bin/bash
#SBATCH -N 1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH -t 96:00:00
#SBATCH -J rpca_error_sweep
#SBATCH -o rpca_error_sweep_%j.out

cd "$SLURM_SUBMIT_DIR"
ulimit -s unlimited
scontrol show job "$SLURM_JOB_ID"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python scripts/run_rpca_error_sweep.py
