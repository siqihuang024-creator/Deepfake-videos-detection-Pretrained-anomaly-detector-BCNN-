"""Prepare, run and summarize the single-GPU 3 x 3 step-2 ablation matrix.

Selection happens before test evaluation. Step 4 uses the validation winner
among the three ln_b128 runs; test outcomes never enter this decision.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gad.ablations import EXECUTION, FULL_PROBE_INDEX, SEEDS, SETTINGS, TASK_COUNT, run_spec


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)


def input_hashes(experiment):
    paths = ("tables/videos.csv", "lists/train.txt", "lists/val.txt")
    return {path: digest(experiment / path) for path in paths}


def prepare(experiment, gradient_checkpointing=False):
    plan = {"version": 2, "split_seed": 42, "input_sha256": input_hashes(experiment),
            "execution": EXECUTION,
            "full_gradient_checkpointing": gradient_checkpointing,
            "runs": [run_spec(i) for i in range(TASK_COUNT)],
            "within_run_selection": "best validation video AUROC; earliest epoch wins ties",
            "test_reporting": "all three seeds per setting, mean and sample standard deviation",
            "step4_selection": "highest validation AUROC among ln_b128 seeds; lowest seed wins ties"}
    write_new(experiment / "ablation_plan.json", plan)
    return plan


def load_plan(experiment):
    plan = json.loads((experiment / "ablation_plan.json").read_text(encoding="utf-8"))
    if (plan.get("version") != 2 or plan.get("execution") != EXECUTION
            or plan["runs"] != [run_spec(i) for i in range(TASK_COUNT)]
            or plan["input_sha256"] != input_hashes(experiment)):
        raise ValueError("Matrix, execution policy or input files differ from the registered plan; keep the old plan and logs")
    return plan


def submit(experiment):
    """Chain three serial arrays: %1 alone does not guarantee setting order."""
    script = Path(__file__).resolve().parents[1] / "slurm/identity_ablation.sbatch"
    previous = None
    submitted = []
    for setting_index, setting in enumerate(SETTINGS):
        start = setting_index * len(SEEDS)
        end = start + len(SEEDS) - 1
        command = ["sbatch", "--parsable", f"--array={start}-{end}%1"]
        if previous is not None:
            command += [f"--dependency=afterok:{previous}"]
        command += [str(script), str(experiment), "fit"]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        previous = result.stdout.strip().split(";", 1)[0]
        if not previous.isdigit():
            raise ValueError(f"Unexpected sbatch response for {setting}: {result.stdout!r}; inspect the queue before retrying")
        submitted.append({"setting": setting, "array_job_id": previous})
        # Keep already submitted IDs visible if a later submission fails.
        print(f"Submitted {setting}: {previous}", flush=True)
    return submitted


def select(experiment, plan):
    selections = {}
    for spec in plan["runs"]:
        directory = experiment / "runs" / spec["run_name"]
        meta = json.loads((directory / "run_meta.json").read_text(encoding="utf-8"))
        record = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
        if (not meta.get("finished") or meta.get("debug") or record.get("protocol") != "celeb_identity"
                or meta["args"]["seed"] != spec["seed"] or meta["trainable"] != spec["trainable"]
                or meta["batching"]["forward_batch_size"] != spec["batch_size"]
                or meta["batching"]["gradient_accumulation"] != 1
                or meta["args"]["lr"] != spec["lr"] or meta["args"]["backbone_lr"] != spec["backbone_lr"]
                or meta["gradient_checkpointing"] != (plan["full_gradient_checkpointing"] and spec["trainable"] == "full")
                or meta["ablation_plan_sha256"] != digest(experiment / "ablation_plan.json")
                or record["checkpoint_sha256"] != meta["best_checkpoint_sha256"]):
            raise ValueError(f"Incomplete or unexpected run: {spec['run_name']}")
        for split in ("train", "val"):
            path = (experiment / f"lists/{split}.txt").resolve().as_posix()
            if meta["list_sha256"][path] != plan["input_sha256"][f"lists/{split}.txt"]:
                raise ValueError(f"Different {split} list: {spec['run_name']}")
        if digest(record["checkpoint"]) != record["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed: {spec['run_name']}")
        selections[spec["run_name"]] = {"spec": spec, "selection": record}
    baseline = [item for item in selections.values() if item["spec"]["setting"] == "ln_b128"]
    chosen = max(baseline, key=lambda item: (item["selection"]["best_val_auroc_video"], -item["spec"]["seed"]))
    record = {**chosen["selection"], "ablation_run": chosen["spec"]["run_name"],
              "rule": plan["step4_selection"], "ablation_plan_sha256": digest(experiment / "ablation_plan.json")}
    write_new(experiment / "runs/ablation_selection.json", record)
    write_new(experiment / "matrix_selections.json", selections)
    return record


def launch(experiment, plan, index, action):
    spec = run_spec(index)
    checkpointing = plan["full_gradient_checkpointing"] and spec["trainable"] == "full"
    if action == "probe":
        if spec["setting"] != "full_b128":
            raise ValueError("The resource probe is for full_b128 only (index 6..8)")
        spec = {**spec, "run_name": spec["run_name"] + ("_probe_gc" if checkpointing else "_probe")}
    env = dict(os.environ, GEND_RUN_NAME=spec["run_name"], GEND_TRAIN_SEED=str(spec["seed"]),
               GEND_TRAIN_BATCH_SIZE=str(spec["batch_size"]), GEND_TRAIN_MICRO_BATCH_SIZE=str(spec["batch_size"]),
               GEND_TRAINABLE=spec["trainable"], GEND_LR=str(spec["lr"]), GEND_BACKBONE_LR=str(spec["backbone_lr"]))
    env["GEND_GRAD_CHECKPOINTING"] = "1" if checkpointing else "0"
    env["GEND_ABLATION_PLAN_SHA256"] = digest(experiment / "ablation_plan.json") if action != "probe" else ""
    env.pop("GEND_PROBE_STEPS", None)
    if action == "probe":
        env["GEND_PROBE_STEPS"] = "20"
    if action == "fit" and spec["trainable"] == "full":
        probe_name = "full_b128_s42" + ("_probe_gc" if checkpointing else "_probe")
        probe = json.loads((experiment / "runs" / probe_name / "run_meta.json").read_text(encoding="utf-8"))
        if (not probe.get("probe_completed") or probe.get("probe_steps_completed", 0) < 20
                or probe["trainable"] != "full" or probe["batching"]["forward_batch_size"] != 128
                or probe["gradient_checkpointing"] != checkpointing or not probe["performance"]["gpu"]):
            raise ValueError("A successful 20-step full-model resource probe is required first")
        for split in ("train", "val"):
            path = (experiment / f"lists/{split}.txt").resolve().as_posix()
            if probe["list_sha256"][path] != plan["input_sha256"][f"lists/{split}.txt"]:
                raise ValueError("Resource probe used different input lists")
    if action == "evaluate":
        locked = json.loads((experiment / "matrix_selections.json").read_text(encoding="utf-8"))
        current = json.loads((experiment / "runs" / spec["run_name"] / "selection.json").read_text(encoding="utf-8"))
        if locked[spec["run_name"]] != {"spec": spec, "selection": current}:
            raise ValueError("Selection changed after validation-only matrix selection")
        if not (experiment / "lists/test/test.txt").is_file():
            raise ValueError("Prepare the audited test frame list before evaluation")
        env["GEND_EVAL_DIR"] = str(experiment / "eval/ablations" / spec["run_name"])
    script = Path(__file__).resolve().parents[1] / "slurm/identity_supervised.sbatch"
    subprocess.run(["bash", str(script), str(experiment), "fine-tuned" if action == "evaluate" else "fit"],
                   env=env, check=True)


def summarize(experiment, plan):
    results = {}
    locked = json.loads((experiment / "matrix_selections.json").read_text(encoding="utf-8"))
    evaluation_lists = set()
    for setting in SETTINGS:
        reports = [json.loads((experiment / "eval/ablations" / f"{setting}_s{seed}" / "report.json").read_text(
            encoding="utf-8")) for seed in SEEDS]
        if any(report["video_table_sha256"] != plan["input_sha256"]["tables/videos.csv"] for report in reports):
            raise ValueError("Evaluation table differs from the registered plan")
        for seed, report in zip(SEEDS, reports):
            if report["selection"]["checkpoint_sha256"] != locked[f"{setting}_s{seed}"]["selection"]["checkpoint_sha256"]:
                raise ValueError("Evaluation did not use the locked checkpoint")
            evaluation_lists.add((report["val_pred_meta"]["frame_list_sha256"],
                                  report["test_pred_meta"]["frame_list_sha256"]))
        values = [report["test"]["auroc"] for report in reports]
        methods = set(reports[0]["test"]["per_method"])
        if any(set(report["test"]["per_method"]) != methods for report in reports):
            raise ValueError("Different scorable methods between seeds")
        per_method = {}
        for method in sorted(methods):
            numbers = [report["test"]["per_method"][method]["auroc"] for report in reports]
            per_method[method] = {"mean": statistics.mean(numbers), "stdev": statistics.stdev(numbers)}
        results[setting] = {"seeds": list(SEEDS), "test_auroc": values, "mean": statistics.mean(values),
                            "stdev": statistics.stdev(values), "per_method_auroc": per_method}
    if len(evaluation_lists) != 1:
        raise ValueError("Runs did not evaluate the same validation/test frames")
    write_new(experiment / "eval/ablation_summary.json", results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--action", choices=("prepare", "submit", "fit", "probe", "select", "evaluate", "summarize"), required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable for a resource probe, or register it for all full-model training runs")
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    if args.action == "prepare":
        result = prepare(experiment, args.gradient_checkpointing)
    elif args.action == "probe":
        launch(experiment, {"full_gradient_checkpointing": args.gradient_checkpointing},
               FULL_PROBE_INDEX if args.index is None else args.index, args.action)
        return
    else:
        if args.gradient_checkpointing:
            parser.error("Register gradient checkpointing with --action prepare before training")
        plan = load_plan(experiment)
        if args.action == "submit":
            result = submit(experiment)
            print(json.dumps(result, indent=2))
            return
        if args.action in ("fit", "probe", "evaluate"):
            if args.index is None:
                parser.error("--index is required for this action")
            launch(experiment, plan, args.index, args.action)
            return
        result = select(experiment, plan) if args.action == "select" else summarize(experiment, plan)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
