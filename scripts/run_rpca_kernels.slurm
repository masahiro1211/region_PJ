#!/bin/bash
#SBATCH -N 1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH -t 96:00:00
#SBATCH -J rpca_kernels
#SBATCH -o results/logs/rpca_kernels_%j.out

cd "$SLURM_SUBMIT_DIR"
ulimit -s unlimited
scontrol show job "$SLURM_JOB_ID"
source scripts/experiment_config.sh

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python scripts/run_rpca_kernels.py \
  --compute \
  --N "$EXP_N" \
  --L "$EXP_L" \
  --exp-sum-rank "$EXP_SUM_RANK" \
  --rpca-max-iter "$RPCA_MAX_ITER" \
  --rpca-tol "$RPCA_TOL" \
  --exp-sum-ranks "$EXP_SUM_RANKS" \
  "$EXP_SUM_NONNEG_OPT" \
  --exp-sum-max-iter "$EXP_SUM_MAX_ITER" \
  --exp-sum-n-points "$EXP_SUM_N_POINTS" \
  --exp-sum-r-min "$EXP_SUM_R_MIN"
