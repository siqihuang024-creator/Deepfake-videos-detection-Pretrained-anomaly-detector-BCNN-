#!/bin/bash
# Submit from the project root with: WORKSPACE [IDENTITY_SELECTION]
# Omit the selection to run the official encoder only.
#SBATCH --job-name=gend-anomaly
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0
#SBATCH --output=work/logs/gend_anomaly_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
py="$PWD/work/.venv-gend/bin/python"
scripts="$PWD/GenD_anomaly_detector/scripts"
experiment="$(realpath "${1:?Supply the external anomaly experiment directory}")"
lists="$experiment/lists"
out="$experiment/eval/external_$SLURM_JOB_ID"
export OMP_NUM_THREADS=2
predict_batch="${GEND_PREDICT_BATCH_SIZE:-256}"
workers="${GEND_NUM_WORKERS:-2}"

encoders=(official)
if [[ -n "${2:-}" ]]; then
    identity_selection="$(realpath "$2")"
    checkpoint="$("$py" -c '
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    selection = json.load(f)
if selection["step"] != 2 or selection.get("protocol") != "celeb_identity":
    sys.exit("Expected the selection from the identity-disjoint fine-tuning run.")
print(selection["checkpoint"])
' "$identity_selection")"
    encoders+=(finetuned)
fi
official_selection="$out/official_selection.json"
"$py" "$scripts/select_run.py" --official --out "$official_selection"

for encoder in "${encoders[@]}"; do
    model=official
    selection="$official_selection"
    if [[ "$encoder" == "finetuned" ]]; then
        model="$checkpoint"
        selection="$identity_selection"
    fi
    cache="$out/features/$encoder"
    for list in train val test/test; do
        "$py" -u "$scripts/predict_frames.py" --features --model "$model" --selection "$selection" \
            --frame-list "$lists/$list.txt" --precision bf16 --batch-size "$predict_batch" --num-workers "$workers" \
            --out "$cache/${list##*/}.npz"
    done
    "$py" -u "$scripts/evaluate_anomaly.py" --videos "$experiment/tables/videos.csv" \
        --train-features "$cache/train.npz" --val-features "$cache/val.npz" \
        --test-features "$cache/test.npz" --out "$out/$encoder"
done
