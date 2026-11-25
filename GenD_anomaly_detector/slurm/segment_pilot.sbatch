#!/bin/bash
# Small face-preprocessing timing pilot. No GenD inference or head training.
#SBATCH --job-name=gend-segment-pilot
#SBATCH --partition=batch
#SBATCH --nodelist=hefest
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=work/logs/gend_segment_pilot_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
py="$PWD/work/.venv-gend/bin/python"
plan="${1:?Supply segment_pilot.json}"
work="${2:?Supply a NEW pilot preprocessing directory}"
extra=()
if [[ "${3:-}" == resume ]]; then extra=(--resume); elif [[ -n "${3:-}" ]]; then exit 2; fi
cuda_libs="$("$py" -c 'import sysconfig; from pathlib import Path; print(":".join(str(p) for p in sorted((Path(sysconfig.get_path("purelib")) / "nvidia").glob("*/lib")) if p.is_dir()))')"
export LD_LIBRARY_PATH="$cuda_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
echo 'Node-wide GPU snapshot; entries may include other allocations:'
nvidia-smi --query-gpu=index,uuid,power.limit,utilization.gpu,memory.used --format=csv
"$py" -u GenD_anomaly_detector/scripts/preprocess_segment_pilot.py \
    --pilot-plan "$plan" --video-root /data/datasets/shuang --work "$work" \
    --retinaface work/weights/det_10g.onnx "${extra[@]}"
