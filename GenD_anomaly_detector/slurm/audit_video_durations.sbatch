#!/bin/bash
# CPU metadata audit only; submit from the repository root.
#SBATCH --job-name=gend-duration-audit
#SBATCH --partition=batch
#SBATCH --nodelist=hefest
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=work/logs/gend_duration_audit_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
inventory="${1:-$HOME/gend_steps34_metadata.tar.gz}"
out="${2:-work/video_duration_audit_$SLURM_JOB_ID}"
extra=()
if [[ "${3:-}" == resume ]]; then extra=(--resume); elif [[ -n "${3:-}" ]]; then exit 2; fi
"$PWD/work/.venv-gend/bin/python" -u GenD_anomaly_detector/scripts/audit_video_durations.py \
    --inventory "$inventory" --video-root /data/datasets/shuang --out "$out" \
    --workers "$SLURM_CPUS_PER_TASK" --timeout 30 "${extra[@]}"
