#!/bin/bash
#SBATCH -N 1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH -t 96:00:00
#SBATCH -J exp_sum_fit
#SBATCH -o results/logs/exp_sum_fit_%j.out

cd "$SLURM_SUBMIT_DIR"
ulimit -s unlimited
scontrol show job "$SLURM_JOB_ID"
source scripts/experiment_config.sh

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python scripts/run_exp_sum_fitting.py \
  --compute \
  --L "$EXP_L" \
  --ranks "$EXP_SUM_RANKS" \
  "$EXP_SUM_FIT_NONNEG_OPT" \
  --max-iter "$EXP_SUM_MAX_ITER" \
  --n-points "$EXP_SUM_N_POINTS" \
  --r-min "$EXP_SUM_R_MIN"
