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
run_name="${GEND_RUN_NAME:-finetuned}"
run="$experiment/runs/$run_name"
lists="$experiment/lists"
train_batch="${GEND_TRAIN_BATCH_SIZE:-128}"
micro_batch="${GEND_TRAIN_MICRO_BATCH_SIZE:-$train_batch}"
workers="${GEND_NUM_WORKERS:-2}"
predict_batch="${GEND_PREDICT_BATCH_SIZE:-256}"
trainable="${GEND_TRAINABLE:-layernorm}"
seed="${GEND_TRAIN_SEED:-42}"
extra=()
if [[ "${GEND_GRAD_CHECKPOINTING:-0}" == 1 ]]; then
    extra+=(--gradient-checkpointing)
fi
if [[ -n "${GEND_PROBE_STEPS:-}" ]]; then
    extra+=(--probe-steps "$GEND_PROBE_STEPS")
fi
export OMP_NUM_THREADS=2

case "$action" in
    fit)
        "$py" -u "$scripts/train_gend.py" --init official --seed "$seed" \
            --train-list "$lists/train.txt" --val-list "$lists/val.txt" \
            --run-dir "$experiment/runs" --run-name "$run_name" \
            --batch-size "$train_batch" --mini-batch-size "$micro_batch" --num-workers "$workers" \
            --trainable "$trainable" --lr "${GEND_LR:-0.0003}" \
            --backbone-lr "${GEND_BACKBONE_LR:-0.00001}" "${extra[@]}"
        if [[ -n "${GEND_PROBE_STEPS:-}" ]]; then
            exit
        fi
        "$py" "$scripts/select_run.py" --identity-run "$run" --out "$run/selection.json"
        exit
        ;;
    official)
        model=official
        out="$experiment/eval/official"
        test -s "$lists/test/test.txt"
        if [[ -e "$out" ]]; then echo "Evaluation directory exists: $out" >&2; exit 2; fi
        mkdir -p "$out"
        selection="$out/selection.json"
        "$py" "$scripts/select_run.py" --official --out "$selection"
        ;;
    fine-tuned)
        out="${GEND_EVAL_DIR:-$experiment/eval/${run_name}_$SLURM_JOB_ID}"
        selection="$run/selection.json"
        model="$("$py" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["checkpoint"])' "$selection")"
        test -s "$lists/test/test.txt"
        if [[ -e "$out" ]]; then echo "Evaluation directory exists: $out" >&2; exit 2; fi
        mkdir -p "$out"
        ;;
    *)
        echo "Expected fit, official, or fine-tuned" >&2
        exit 2
        ;;
esac

"$py" -u "$scripts/predict_frames.py" --model "$model" --frame-list "$lists/val.txt" \
    --precision bf16 --batch-size "$predict_batch" --num-workers "$workers" --out "$out/val.csv"
"$py" -u "$scripts/predict_frames.py" --model "$model" --frame-list "$lists/test/test.txt" \
    --selection "$selection" --precision bf16 --batch-size "$predict_batch" --num-workers "$workers" --out "$out/test.csv"
"$py" -u "$scripts/evaluate.py" --protocol celeb_identity --videos "$experiment/tables/videos.csv" \
    --val-pred "$out/val.csv" --test-pred "$out/test.csv" --selection "$selection" --out "$out/report.json"
