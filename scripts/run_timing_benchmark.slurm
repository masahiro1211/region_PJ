#!/bin/bash
#SBATCH -N 1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH -t 96:00:00
#SBATCH -J timing_bench
#SBATCH -o results/logs/timing_bench_%j.out

cd "$SLURM_SUBMIT_DIR"
ulimit -s unlimited
scontrol show job "$SLURM_JOB_ID"
source scripts/experiment_config.sh

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python scripts/run_timing_benchmark.py \
  --N "$EXP_N" \
  --L "$EXP_L" \
  --density-alpha "$EXP_DENSITY_ALPHA" \
  --exp-sum-rank "$EXP_SUM_RANK" \
  --r-bench "$TIMING_R_BENCH" \
  --tau-bench "$TIMING_TAU_BENCH" \
  --n-warmup "$TIMING_N_WARMUP" \
  --n-inner "$TIMING_N_INNER" \
  --n-repeat "$TIMING_N_REPEAT"
