#!/bin/bash
# Shared-home feature caches only; tarasca avoids the busy GPU node.
#SBATCH --job-name=gend-cov-controls
#SBATCH --partition=batch
#SBATCH --nodelist=tarasca
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0
#SBATCH --output=work/logs/gend_cov_controls_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
py="$PWD/work/.venv-gend/bin/python"
hostname
git log -1 --oneline
"$py" -m unittest discover -s GenD_anomaly_detector/tests -p test_covariance_controls.py -q
exec "$py" -u GenD_anomaly_detector/scripts/run_covariance_controls.py \
    --reference "${1:?Supply completed reference 313}" \
    --controls "${2:?Supply completed controls 316}" \
    --geometry-run "${3:?Supply completed geometry 342}" \
    --data "${4:?Supply candidate data 301}" \
    --cache-run "${5:?Supply feature registration 311}" \
    --selection "${6:?Supply fine-tuned selection}" \
    --out "${7:?Supply a new covariance output directory}"
