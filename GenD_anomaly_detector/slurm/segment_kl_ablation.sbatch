#!/bin/bash
# Prepared follow-up; submit only after inspecting the completed controls.
#SBATCH --job-name=gend-kl-ablation
#SBATCH --partition=batch
#SBATCH --nodelist=hefest
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0
#SBATCH --output=work/logs/gend_kl_ablation_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
git log -1 --oneline
work/.venv-gend/bin/python -m unittest discover \
    -s GenD_anomaly_detector/tests -p test_bayesian_kl_ablation.py -q
exec work/.venv-gend/bin/python -u GenD_anomaly_detector/scripts/run_segment_kl_ablation.py \
    --reference "${1:?Supply the original completed Bayesian run}" \
    --controls "${2:?Supply the completed CPU-control run}" \
    --data "${3:?Supply candidate-data directory}" \
    --cache-run "${4:?Supply candidate-cache registration directory}" \
    --selection "${5:?Supply frozen fine-tuned selection}" \
    --out "${6:?Supply a new KL-ablation output directory}"
