#!/bin/bash
# Source-balanced CPU controls, using the frozen job-313 development units.
#SBATCH --job-name=gend-svdd-controls
#SBATCH --partition=batch
#SBATCH --nodelist=hefest
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0
#SBATCH --output=work/logs/gend_svdd_controls_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
git log -1 --oneline
work/.venv-gend/bin/python -m unittest discover \
    -s GenD_anomaly_detector/tests -p test_svdd_controls.py -q
exec work/.venv-gend/bin/python -u GenD_anomaly_detector/scripts/run_segment_controls.py \
    --reference "${1:?Supply completed Bayesian reference directory}" \
    --data "${2:?Supply registered candidate-data directory}" \
    --cache-run "${3:?Supply candidate-cache registration directory}" \
    --selection "${4:?Supply frozen fine-tuned selection}" \
    --out "${5:?Supply a new control output directory}"
