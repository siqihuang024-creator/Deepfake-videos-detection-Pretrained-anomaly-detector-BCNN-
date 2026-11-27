#!/bin/bash
# One fixed CPU experiment on existing shared-home caches.
#SBATCH --job-name=gend-hdtf-followup
#SBATCH --partition=batch
#SBATCH --nodelist=tarasca
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0
#SBATCH --output=work/logs/gend_hdtf_followup_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
py="$PWD/work/.venv-gend/bin/python"
hostname
git log -1 --oneline
"$py" -m unittest discover -s GenD_anomaly_detector/tests -p test_hdtf_followup.py -q
exec "$py" -u GenD_anomaly_detector/scripts/run_hdtf_followup.py \
    --reference "${1:?Supply reference 313}" \
    --controls "${2:?Supply controls 316}" \
    --kl-run "${3:?Supply KL run 317}" \
    --geometry-run "${4:?Supply geometry 342}" \
    --covariance-run "${5:?Supply covariance run 344}" \
    --sensitivity-run "${6:?Supply source sensitivity 347}" \
    --data "${7:?Supply data 301}" \
    --cache-run "${8:?Supply cache registration 311}" \
    --selection "${9:?Supply selected step2 registration}" \
    --out "${10:?Supply a new output directory}"
