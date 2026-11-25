#!/bin/bash
# Registered development roles only. This script never starts training or test evaluation.
#SBATCH --job-name=gend-dev-cache
#SBATCH --partition=batch
#SBATCH --nodelist=hefest
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0
#SBATCH --output=work/logs/gend_dev_cache_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
protocol="${1:?Supply registered diagnostic directory}"
preprocessed="${2:?Supply diagnostic preprocessing directory}"
selection="${3:?Supply step-4 ablation_selection.json}"
py="$PWD/work/.venv-gend/bin/python"
checkpoint=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$selection")
for name in official finetuned; do
    model=official
    if [[ "$name" == finetuned ]]; then model="$checkpoint"; fi
    "$py" -u GenD_anomaly_detector/scripts/extract_frame_cache.py \
        --model "$model" --selection "$selection" \
        --frame-list "$protocol/lists/development.txt" \
        --list-registration "$protocol/lists/frame_lists.json" \
        --records "$preprocessed/records" --cache-root work/feature_cache \
        --receipt "$protocol/cache_receipts/$name.json" \
        --batch-size "${GEND_PREDICT_BATCH_SIZE:-128}" \
        --num-workers "${GEND_NUM_WORKERS:-6}"
done
