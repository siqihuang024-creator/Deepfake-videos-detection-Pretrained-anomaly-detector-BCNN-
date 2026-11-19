#!/bin/bash
# Submit from the project root with: WORKSPACE VIDEO_ROOT {train-val|test}
# Override the array size with --array=0-7%2 to use eight shards, two at a time.
#SBATCH --job-name=gend-protocol-prep
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0
#SBATCH --array=0
#SBATCH --output=work/logs/gend_protocol_prep_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
py="$PWD/work/.venv-gend/bin/python"
experiment="$(realpath "${1:?Supply the experiment directory}")"
video_root="$(realpath "${2:?Supply the video root}")"
case "${3:?Supply train-val or test}" in
    train-val) splits=(train val) ;;
    test) splits=(test) ;;
    *) echo "Expected train-val or test" >&2; exit 2 ;;
esac

if (( SLURM_ARRAY_TASK_MIN != 0 || SLURM_ARRAY_TASK_MAX + 1 != SLURM_ARRAY_TASK_COUNT )); then
    echo "Use a contiguous array starting at 0" >&2
    exit 2
fi
cuda_libs="$("$py" -c 'import sysconfig; from pathlib import Path; print(":".join(str(p) for p in sorted((Path(sysconfig.get_path("purelib")) / "nvidia").glob("*/lib")) if p.is_dir()))')"
export LD_LIBRARY_PATH="$cuda_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=2

"$py" -u GenD_anomaly_detector/scripts/preprocess_videos.py \
    --videos "$experiment/tables/videos.csv" --splits "${splits[@]}" \
    --video-root "$video_root" --work "$experiment/faces" --retinaface work/weights/det_10g.onnx \
    --num-shards "$SLURM_ARRAY_TASK_COUNT" --shard-index "$SLURM_ARRAY_TASK_ID"
