#!/bin/bash
# Face preprocessing of the training videos of groups 02-30, split over a Slurm array.
# Submit from the repository root (the directory that contains GenD_anomaly_detector/ and work/):
#   mkdir -p work/logs && sbatch GenD_anomaly_detector/slurm/preprocess_train.sbatch
# Videos already done (ok / no_face) are skipped, so resubmitting retries only failures.
#SBATCH --job-name=gend-prep
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0
#SBATCH --array=0-7%4
#SBATCH --output=work/logs/prep_%A_%a.out

set -euo pipefail
P="$SLURM_SUBMIT_DIR"
W="$P/work"
V="${V:-/data/datasets/shuang/CelebDFv3/CelebDFv3}"
PY="$W/.venv-gend/bin/python"
NUM_SHARDS=8

# CUDA 12 / cuDNN 9 libraries installed in the venv for onnxruntime-gpu.
CUDA_LIBS="$("$PY" -c 'import sysconfig; from pathlib import Path; print(":".join(str(p) for p in sorted((Path(sysconfig.get_path("purelib")) / "nvidia").glob("*/lib")) if p.is_dir()))')"
export LD_LIBRARY_PATH="$CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=2

echo "host $(hostname)  shard $SLURM_ARRAY_TASK_ID/$NUM_SHARDS  commit $(git -C "$P" rev-parse --short HEAD)"
"$PY" -u "$P/GenD_anomaly_detector/scripts/preprocess_videos.py" \
  --videos "$W/tables/videos.csv" --membership "$W/tables/membership.csv" \
  --groups $(seq 2 30) --splits train \
  --video-root "$V" --work "$W/faces" --retinaface "$W/weights/det_10g.onnx" \
  --num-shards "$NUM_SHARDS" --shard-index "$SLURM_ARRAY_TASK_ID"
