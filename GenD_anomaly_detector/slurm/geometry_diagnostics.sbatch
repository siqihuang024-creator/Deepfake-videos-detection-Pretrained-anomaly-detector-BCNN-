#!/bin/bash
# Existing features only; CPU model loading verifies frozen classifier directions.
#SBATCH --job-name=gend-geometry
#SBATCH --partition=batch
#SBATCH --nodelist=hefest
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0
#SBATCH --output=work/logs/gend_geometry_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
git log -1 --oneline
work/.venv-gend/bin/python -m unittest discover \
    -s GenD_anomaly_detector/tests -p test_feature_geometry.py -q
exec work/.venv-gend/bin/python -u GenD_anomaly_detector/scripts/run_geometry_diagnostics.py \
    --reference "${1:?Supply completed Bayesian reference}" \
    --controls "${2:?Supply completed controls}" \
    --data "${3:?Supply candidate-data directory}" \
    --cache-run "${4:?Supply candidate-cache registration}" \
    --selection "${5:?Supply selected fine-tuned checkpoint registration}" \
    --out "${6:?Supply a new geometry output directory}"
