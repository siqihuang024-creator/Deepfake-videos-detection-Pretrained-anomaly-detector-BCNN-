#!/bin/bash
# Train only the Bayesian heads on CPU; never load or train a GenD model.
#SBATCH --job-name=gend-segment-bayes
#SBATCH --partition=batch
#SBATCH --nodelist=hefest
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0
#SBATCH --output=work/logs/gend_segment_bayes_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
echo '=== Code version and CPU regressions ==='
git log -1 --oneline
work/.venv-gend/bin/python -m unittest discover \
    -s GenD_anomaly_detector/tests -p test_grouped_bayesian.py -q
work/.venv-gend/bin/python -m unittest discover \
    -s GenD_anomaly_detector/tests -p test_segment_training.py -q
echo '=== Register units and train both Bayesian heads ==='
exec work/.venv-gend/bin/python -u GenD_anomaly_detector/scripts/train_segment_bayesian.py \
    --data "${1:?Supply completed segment-data directory}" \
    --cache-run "${2:?Supply completed candidate-cache run directory}" \
    --selection "${3:?Supply locked fine-tuned selection}" \
    --out "${4:?Supply a new Bayesian development output directory}"
