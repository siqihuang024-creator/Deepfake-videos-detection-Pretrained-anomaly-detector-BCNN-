#!/bin/bash
# Primary experiment. GenD is not loaded: train the small Bayesian head from caches.
#SBATCH --job-name=gend-bayesian-heads
#SBATCH --partition=batch
#SBATCH --nodelist=hefest
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=work/logs/gend_bayesian_heads_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
protocol="${1:?Supply the registered experiment directory}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
for name in official finetuned; do
    work/.venv-gend/bin/python -u GenD_anomaly_detector/scripts/evaluate_bayesian.py \
        --plan "$protocol/diagnostic_plan.json" --lists "$protocol/lists/frame_lists.json" \
        --development-cache "$protocol/cache_receipts/$name.json" \
        --out "$protocol/bayesian/${name}_${SLURM_JOB_ID}" \
        --freeze-to "$protocol/locks/bayesian/${name}_${SLURM_JOB_ID}" --device cpu
done
