#!/bin/bash
# Fixed source-composition sensitivity on existing shared-home features.
#SBATCH --job-name=gend-source-check
#SBATCH --partition=batch
#SBATCH --nodelist=tarasca
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0
#SBATCH --output=work/logs/gend_source_check_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
py="$PWD/work/.venv-gend/bin/python"
hostname
git log -1 --oneline
"$py" -m unittest discover -s GenD_anomaly_detector/tests -p test_source_sensitivity.py -q
exec "$py" -u GenD_anomaly_detector/scripts/run_source_sensitivity.py \
    --reference "${1:?Supply reference 313}" \
    --controls "${2:?Supply controls 316}" \
    --covariance-run "${3:?Supply covariance run 344}" \
    --data "${4:?Supply data 301}" \
    --cache-run "${5:?Supply feature registration 311}" \
    --selection "${6:?Supply selected checkpoint registration}" \
    --out "${7:?Supply a new output directory}"
