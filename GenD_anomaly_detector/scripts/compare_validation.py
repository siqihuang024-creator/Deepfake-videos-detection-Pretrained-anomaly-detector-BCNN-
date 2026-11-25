"""Validation-only paired inference with identical scripts, frames and precision.

Re-score the already selected LN-b128 checkpoint; never reselect it or load test
predictions. Its training-time validation metric is retained only for reference.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gad.feature_cache import sha256_file


def compare(experiment, out, batch_size=128, num_workers=6, precision="bf16", bootstrap_draws=2000):
    experiment, out = Path(experiment), Path(out)
    if out.exists():
        raise FileExistsError(out)
    if batch_size < 1 or num_workers < 0 or precision not in ("bf16", "fp32") or bootstrap_draws < 2:
        raise ValueError("Invalid inference settings")
    selection_path = experiment / "runs/ablation_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    frame_list = experiment / "lists/val.txt"
    table = experiment / "tables/videos.csv"
    ablation_plan_path = experiment / "ablation_plan.json"
    ablation_plan = json.loads(ablation_plan_path.read_text(encoding="utf-8"))
    if (selection.get("step") != 2 or selection.get("protocol") != "celeb_identity"
            or not selection.get("ablation_run", "").startswith("ln_b128_s")
            or not selection.get("ablation_plan_sha256")
            or sha256_file(ablation_plan_path) != selection["ablation_plan_sha256"]
            or sha256_file(table) != ablation_plan["input_sha256"]["tables/videos.csv"]
            or sha256_file(Path(selection["checkpoint"])) != selection["checkpoint_sha256"]
            or sha256_file(frame_list) != selection["val_frame_list_sha256"]):
        raise ValueError("Use the locked LN-b128 winner and its unchanged validation frame list")
    input_hashes = {"frame_list_sha256": sha256_file(frame_list), "video_table_sha256": sha256_file(table),
                    "selection_sha256": sha256_file(selection_path)}
    scripts = Path(__file__).resolve().parent
    out.mkdir(parents=True)
    reports = {}
    for name, model in (("official", "official"), ("finetuned", selection["checkpoint"])):
        folder = out / name
        folder.mkdir()
        prediction = folder / "val.csv"
        subprocess.run([sys.executable, str(scripts / "predict_frames.py"), "--model", model,
                        "--selection", str(selection_path), "--frame-list", str(frame_list),
                        "--out", str(prediction), "--batch-size", str(batch_size),
                        "--num-workers", str(num_workers), "--precision", precision], check=True)
        subprocess.run([sys.executable, str(scripts / "evaluate.py"), "--protocol", "celeb_identity",
                        "--videos", str(table), "--val-pred", str(prediction),
                        "--bootstrap-draws", str(bootstrap_draws), "--out", str(folder / "report.json")], check=True)
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        meta = report["val_pred_meta"]
        if (meta["model"] != model or meta["frame_list_sha256"] != input_hashes["frame_list_sha256"]
                or meta["precision"] != precision or meta["batch_size"] != batch_size
                or meta["num_workers"] != num_workers
                or (name == "finetuned" and meta["checkpoint_sha256"] != selection["checkpoint_sha256"])):
            raise ValueError("Paired inference metadata differ")
        reports[name] = report
    for field, path in (("frame_list_sha256", frame_list), ("video_table_sha256", table),
                        ("selection_sha256", selection_path)):
        if sha256_file(path) != input_hashes[field]:
            raise ValueError("Validation inputs changed during paired inference")
    if sha256_file(Path(selection["checkpoint"])) != selection["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed during paired inference")
    if reports["official"]["val_coverage"] != reports["finetuned"]["val_coverage"]:
        raise ValueError("Official and fine-tuned validation coverage differs")
    official = reports["official"]["val"]["auroc"]
    finetuned = reports["finetuned"]["val"]["auroc"]
    result = {**input_hashes, "precision": precision, "batch_size": batch_size, "num_workers": num_workers,
              "selection": selection, "official_val_auroc": official, "finetuned_val_auroc": finetuned,
              "finetuned_minus_official_auroc": finetuned - official,
              "training_time_selected_val_auroc_reference": selection["best_val_auroc_video"],
              "training_vs_inference_auroc_difference": finetuned - selection["best_val_auroc_video"],
              "official_report": "official/report.json", "finetuned_report": "finetuned/report.json",
              "scope": "Validation diagnosis only. Both use predict_frames.py and evaluate.py. No test evaluation or reselection.",
              "limitation": "Fine-tuned checkpoint was selected on these validation data; its apparent improvement is not an unbiased held-out-test gain. Official GenD development also used CDFv3."}
    with (out / "comparison.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(compare(args.experiment, args.out, args.batch_size, args.num_workers,
                             args.precision, args.bootstrap_draws), indent=2))
