#!/bin/bash
# Paired validation inference only; uses the locked fine-tuned checkpoint too.
#SBATCH --job-name=gend-official-val
#SBATCH --partition=batch
#SBATCH --nodelist=hefest
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=work/logs/gend_official_val_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
experiment="${1:-work/celeb_identity_seed42}"
out="$experiment/eval/official_val_$SLURM_JOB_ID"
py="$PWD/work/.venv-gend/bin/python"
exec "$py" -u GenD_anomaly_detector/scripts/compare_validation.py \
    --experiment "$experiment" --out "$out" \
    --batch-size "${GEND_PREDICT_BATCH_SIZE:-128}" --num-workers "${GEND_NUM_WORKERS:-6}" \
    --precision bf16
