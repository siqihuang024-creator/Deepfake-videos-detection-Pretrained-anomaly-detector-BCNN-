#!/bin/bash
# Submit from the project root with: WORKSPACE {fit|official|fine-tuned}
# WORKSPACE contains tables/videos.csv, lists/ and separate face preprocessing.
#SBATCH --job-name=gend-identity
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0
#SBATCH --output=work/logs/gend_identity_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
py="$PWD/work/.venv-gend/bin/python"
scripts="$PWD/GenD_anomaly_detector/scripts"
experiment="$(realpath "${1:?Supply the identity-disjoint experiment directory}")"
action="${2:?Supply fit, official, or fine-tuned}"
run="$experiment/runs/finetuned"
lists="$experiment/lists"
export OMP_NUM_THREADS=2

case "$action" in
    fit)
        "$py" -u "$scripts/train_gend.py" --init official --seed 42 \
            --train-list "$lists/train.txt" --val-list "$lists/val.txt" \
            --run-dir "$experiment/runs" --run-name finetuned --num-workers 2
        "$py" "$scripts/select_run.py" --identity-run "$run" --out "$run/selection.json"
        exit
        ;;
    official)
        model=official
        out="$experiment/eval/official_$SLURM_JOB_ID"
        selection="$out/selection.json"
        "$py" "$scripts/select_run.py" --official --out "$selection"
        ;;
    fine-tuned)
        out="$experiment/eval/finetuned_$SLURM_JOB_ID"
        selection="$run/selection.json"
        model="$("$py" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["checkpoint"])' "$selection")"
        ;;
    *)
        echo "Expected fit, official, or fine-tuned" >&2
        exit 2
        ;;
esac

"$py" -u "$scripts/predict_frames.py" --model "$model" --frame-list "$lists/val.txt" \
    --precision bf16 --batch-size 64 --num-workers 2 --out "$out/val.csv"
"$py" -u "$scripts/predict_frames.py" --model "$model" --frame-list "$lists/test/test.txt" \
    --selection "$selection" --precision bf16 --batch-size 64 --num-workers 2 --out "$out/test.csv"
"$py" -u "$scripts/evaluate.py" --protocol celeb_identity --videos "$experiment/tables/videos.csv" \
    --val-pred "$out/val.csv" --test-pred "$out/test.csv" --selection "$selection" --out "$out/report.json"
