#!/bin/bash
# Step 2: fine-tune groups sequentially from the official weights (seed = 41 + group).
# Submit from the repository root, passing the groups of this job:
#   sbatch --export=ALL,STEP2_GROUPS="2 3 4 5 6 7 8 9 10 11" GenD_anomaly_detector/slurm/train_groups.sbatch
# Finished groups are skipped; an unfinished group directory stops the job for inspection.
#SBATCH --job-name=gend-step2
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0
#SBATCH --output=work/logs/step2_%j.out

set -euo pipefail
: "${STEP2_GROUPS:?set STEP2_GROUPS, e.g. --export=ALL,STEP2_GROUPS=\"2 3 4\"}"
P="$SLURM_SUBMIT_DIR"
W="$P/work"
PY="$W/.venv-gend/bin/python"

echo "host $(hostname)  groups $STEP2_GROUPS  commit $(git -C "$P" rev-parse --short HEAD)"
# num-workers 2 = same as group 01; keep it identical across all 30 groups.
"$PY" -u "$P/GenD_anomaly_detector/scripts/run_groups.py" \
  --lists "$W/lists" --runs-dir "$W/runs/step2" \
  --groups $STEP2_GROUPS --num-workers 2
