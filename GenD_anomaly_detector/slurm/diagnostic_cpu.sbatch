#!/bin/bash
#SBATCH --job-name=gend-dev-heads
#SBATCH --partition=batch
#SBATCH --nodelist=hefest
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=work/logs/gend_dev_heads_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
protocol="${1:?Supply registered diagnostic directory}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
for name in official finetuned; do
    work/.venv-gend/bin/python -u GenD_anomaly_detector/scripts/evaluate_diagnostic.py \
        --plan "$protocol/diagnostic_plan.json" --lists "$protocol/lists/frame_lists.json" \
        --development-cache "$protocol/cache_receipts/$name.json" \
        --out "$protocol/diagnosis/${name}_${SLURM_JOB_ID}"
done
