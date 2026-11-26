#!/bin/bash
# Cache already cropped candidates. Does not train either GenD or the heads.
#SBATCH --job-name=gend-segment-cache
#SBATCH --partition=batch
#SBATCH --nodelist=hefest
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=work/logs/gend_segment_cache_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
data="${1:?Supply completed segment-data directory}"
out="${2:?Supply a new candidate-cache registration directory}"
selection="${3:?Supply locked step-4 ablation_selection.json}"
py="$PWD/work/.venv-gend/bin/python"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
cuda_libs="$("$py" -c 'import sysconfig; from pathlib import Path; print(":".join(str(p) for p in sorted((Path(sysconfig.get_path("purelib")) / "nvidia").glob("*/lib")) if p.is_dir()))')"
export LD_LIBRARY_PATH="$cuda_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
checkpoint="$("$py" -c 'import json,sys; from pathlib import Path; import hashlib; p=json.load(open(sys.argv[1])); assert p["ablation_run"]=="ln_b128_s42" and p["best_epoch"]==21; f=Path(p["checkpoint"]); assert hashlib.file_digest(f.open("rb"),"sha256").hexdigest()==p["checkpoint_sha256"]; print(f)' "$selection")"
"$py" -u GenD_anomaly_detector/scripts/prepare_segment_cache.py \
    --plan "$data/segment_data_plan.json" --faces "$data/faces" --out "$out"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
for name in official finetuned; do
    model=official
    if [[ "$name" == finetuned ]]; then model="$checkpoint"; fi
    printf '\n=== Frozen encoder: %s ===\n' "$name"
    "$py" -u GenD_anomaly_detector/scripts/extract_frame_cache.py \
        --model "$model" --selection "$selection" \
        --frame-list "$out/candidates.txt" --list-registration "$out/frame_lists.json" \
        --records "$data/faces/records" --cache-root work/feature_cache \
        --receipt "$out/cache_receipts/$name.json" \
        --batch-size "${GEND_PREDICT_BATCH_SIZE:-256}" \
        --num-workers "${GEND_NUM_WORKERS:-8}" --precision bf16 --device cuda
done
echo 'Both frozen encoder caches completed. No head training or final testing performed.'
