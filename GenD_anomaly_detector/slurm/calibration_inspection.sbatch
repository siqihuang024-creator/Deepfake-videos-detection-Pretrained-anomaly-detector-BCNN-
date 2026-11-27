#!/bin/bash
# Original videos reside on hefest; CPU decoding only, no GPU allocation.
#SBATCH --job-name=gend-cal-inspect
#SBATCH --partition=batch
#SBATCH --nodelist=hefest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0
#SBATCH --output=work/logs/gend_cal_inspect_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
exec work/.venv-gend/bin/python -u GenD_anomaly_detector/scripts/export_calibration_inspection.py \
    --diagnostics "${1:?Supply completed covariance diagnostics}" \
    --data "${2:?Supply candidate data 301}" \
    --cache-run "${3:?Supply feature registration 311}" \
    --video-root "${GEND_VIDEO_ROOT:-/data/datasets/shuang}" \
    --out "${4:?Supply a new inspection output directory}"
