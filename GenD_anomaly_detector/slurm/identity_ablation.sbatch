#!/bin/bash
# WORKSPACE {fit|probe|evaluate}; use --array=6 for the 20-step full-model probe.
# Default array is the baseline only. Use identity_ablation.py --action submit
# for all nine fits in three dependent serial arrays. Select before evaluate.
#SBATCH --job-name=gend-ablation
#SBATCH --partition=batch
#SBATCH --nodelist=hefest
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0
#SBATCH --array=0-2%1
#SBATCH --output=work/logs/gend_ablation_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export GEND_NUM_WORKERS="${GEND_NUM_WORKERS:-6}"
export GEND_PREDICT_BATCH_SIZE="${GEND_PREDICT_BATCH_SIZE:-256}"
extra=()
if [[ "${2:-}" == probe && "${GEND_GRAD_CHECKPOINTING:-0}" == 1 ]]; then
    extra+=(--gradient-checkpointing)
fi
exec work/.venv-gend/bin/python -u GenD_anomaly_detector/scripts/identity_ablation.py \
    --experiment "${1:?Supply the experiment directory}" --action "${2:?Supply fit, probe or evaluate}" \
    --index "${SLURM_ARRAY_TASK_ID:?Submit as an array}" "${extra[@]}"
