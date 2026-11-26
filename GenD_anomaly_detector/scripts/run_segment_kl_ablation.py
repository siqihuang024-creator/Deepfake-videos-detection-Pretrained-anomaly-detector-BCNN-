"""Prepared follow-up: two lower KL weights after completion of the CPU controls."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gad import bayesian_kl_ablation as ablation, bayesian_svdd as bayes, segment_training as original, old_metrics as metrics
from gad.diagnostic_protocol import sha256_file, false_positive_groups
from run_segment_controls import load_reference


def verify_controls(controls, reference, config):
    controls, reference = Path(controls), Path(reference)
    if original.read_json(controls / "completion.json") != {
            "completed": True, "phase": "development_controls", "encoders": ["official", "finetuned"]}:
        raise ValueError("Complete the existing CPU controls before starting KL ablations")
    registration_path = controls / "control_registration.json"
    registration = original.read_json(registration_path)
    if (registration["reference_registration_sha256"] != sha256_file(reference / "training_registration.json")
            or registration["reference_configuration"] != config
            or sha256_file(controls / "training_units.csv") != registration["training_units_sha256"]
            or registration["training_units_sha256"] != config["training_units_sha256"]):
        raise ValueError("Controls use different reference data/units")
    comparison = pd.read_csv(controls / "comparison.csv")
    if len(comparison) != 20 or comparison.duplicated(["encoder", "head", "nominal_fpr"]).any():
        raise ValueError("Invalid control comparison table")
    for encoder in ("official", "finetuned"):
        folder = controls / encoder
        lock = original.read_json(folder / "control_lock.json")
        if (lock["registration_sha256"] != sha256_file(registration_path)
                or lock["heads_sha256"] != sha256_file(folder / "control_heads.npz")
                or lock["report_sha256"] != sha256_file(folder / "report.json")):
            raise ValueError("Control results changed since locking")
        report = original.read_json(folder / "report.json")
        subset = comparison.loc[comparison.encoder.eq(encoder)]
        if set(subset["head"]) != set(report["heads"]):
            raise ValueError("Control comparison/report head coverage differs")
        for row in subset.itertuples(index=False):
            point = report["heads"][row.head][str(row.nominal_fpr)]
            for field in ("auroc", "threshold", "tn", "fp", "fn", "tp"):
                if not np.isclose(getattr(row, field), point[field], rtol=1e-7, atol=1e-8):
                    raise ValueError("Control comparison differs from its locked reports")
    return sha256_file(registration_path)


def run(reference, controls, data, cache_run, selection, out):
    out, reference, controls = Path(out), Path(reference), Path(controls)
    if out.exists():
        raise FileExistsError("Use a new KL-ablation output directory")
    units, caches, config, old_heads, old_reports, old_locks = load_reference(reference, data, cache_run, selection)
    controls_hash = verify_controls(controls, reference, config)
    out.mkdir(parents=True)
    units.to_csv(out / "training_units.csv", index=False)
    registration = {"purpose": "Fixed two-weight Bayesian development follow-up; no final-test mode",
        "variants": ablation.VARIANTS, "reference_registration_sha256": sha256_file(reference / "training_registration.json"),
        "controls_registration_sha256": controls_hash, "control_comparison_sha256": sha256_file(controls / "comparison.csv"),
        "reference_configuration": config, "training_units_sha256": sha256_file(out / "training_units.csv"),
        "loss_logging": "Sample-count-weighted averages of total, distance, raw KL and scaled KL from identical pre-update minibatch states",
        "recipe": "Same initial distributions, center, training RNG, source weights and optimization as reference; only KL coefficient changes",
        "libraries": {"torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "algorithm_sha256": {"run_segment_kl_ablation.py": sha256_file(__file__),
                              "bayesian_kl_ablation.py": sha256_file(ablation.__file__),
                              "run_segment_controls.py": sha256_file(Path(__file__).with_name("run_segment_controls.py"))},
        "limitations": config["limitations"] + ["Exploratory diagnosis after development results; improvement here is not final-test evidence."]}
    original.write_json(out / "kl_registration.json", registration)
    masks = {role: units.head_role.eq(role).to_numpy() for role in ("gradient_fit", "early_stop", "calibration", "dev")}
    summary = []
    for encoder in ("official", "finetuned"):
        values = caches[encoder]
        lookup = {key: i for i, key in enumerate(values["video_keys"])}
        x = values["features"][[lookup[key] for key in units.video_key]]
        dev_rows = units.loc[masks["dev"]].reset_index(drop=True)
        cal_rows = units.loc[masks["calibration"]].reset_index(drop=True)
        groups = units.loc[masks["gradient_fit"], "source_clip"].tolist()
        early_groups = units.loc[masks["early_stop"], "source_clip"].tolist()
        # No retraining of the beta=1e-3 reference.
        for role in ("calibration", "dev"):
            scores, _ = bayes.score(x[masks[role]], old_heads[encoder], "cpu")
            filename = "development_scores.csv" if role == "dev" else "calibration_scores.csv"
            frame = pd.read_csv(reference / encoder / filename, keep_default_na=False).set_index("video_key")
            keys = units.loc[masks[role], "video_key"]
            np.testing.assert_allclose(scores, frame.loc[keys, "bayesian_deep_svdd_score"], rtol=1e-5, atol=1e-7)
        for name, beta in ablation.VARIANTS.items():
            print("\n=== " + encoder + " / " + name + " ===", flush=True)
            folder = out / encoder / name
            folder.mkdir(parents=True)

            def progress(event):
                with (folder / "training_progress.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event) + "\n")
                print("Epoch {epoch}: data={training_distance_loss:.6g}, raw_KL={training_kl_raw:.6g}, scaled_KL={training_weighted_kl:.6g}, early={early_stop_distance:.6g}, best={best_epoch}".format(**event), flush=True)

            head = ablation.fit(x[masks["gradient_fit"]], x[masks["early_stop"]], groups, early_groups, beta, progress)
            np.savez(folder / "head.npz", **head)
            np.savetxt(folder / "training_history.csv", head["training_history"], delimiter=",",
                       header=",".join(ablation.HISTORY_COLUMNS), comments="")
            calibration_scores, _ = bayes.score(x[masks["calibration"]], head, "cpu")
            scores, extras = bayes.score(x[masks["dev"]], head, "cpu")
            if not np.isfinite(calibration_scores).all() or not np.isfinite(scores).all():
                raise ValueError("Nonfinite KL-ablation scores")
            scored, calibrated = dev_rows.copy(), cal_rows.copy()
            scored["score"], scored["posterior_distance_variance"] = scores, extras["posterior_distance_variance"]
            calibrated["score"] = calibration_scores
            scored.to_csv(folder / "development_scores.csv", index=False)
            calibrated.to_csv(folder / "calibration_scores.csv", index=False)
            points, thresholds, false_frames = {}, {}, []
            for alpha in original.RECIPE["target_fprs"]:
                threshold = float(np.quantile(calibration_scores, 1-alpha))
                thresholds[str(alpha)] = threshold
                point = metrics.detection_metrics((dev_rows.label == 0).astype(int), scores, threshold, alpha)
                point["roc_interpolated_tpr_at_nominal_fpr"] = point.pop("tpr_at_target_fpr")
                point["calibration_observed_fpr"] = float((calibration_scores >= threshold).mean())
                point["source_real_fpr"] = false_positive_groups(dev_rows, scores, threshold, "source_only")
                points[str(alpha)] = point
                summary.append({"encoder": encoder, "head": name, "kl_weight": beta, "nominal_fpr": alpha,
                    **{key: value for key, value in point.items() if np.isscalar(value) or value is None}})
                false = scored.loc[dev_rows.label.eq(0) & (scores >= threshold)].copy()
                false["nominal_fpr"] = alpha
                false_frames.append(false)
            pd.concat(false_frames, ignore_index=True).to_csv(folder / "false_positives.csv", index=False)
            diagnostics = {key: np.asarray(head[key]).item() for key in ("best_epoch", "epochs_run", "training_seconds",
                "validation_distance", "train_score_relative_std", "near_constant_train_scores", "train_embedding_variance_ratio",
                "low_train_embedding_variance", "best_state_kl_raw", "best_state_weighted_kl",
                "first_mean_abs", "first_std_mean", "first_median_abs_mean_over_std",
                "second_mean_abs", "second_std_mean", "second_median_abs_mean_over_std")}
            original.write_json(folder / "report.json", {"phase": "development_kl_ablation", "encoder": encoder,
                "kl_weight": beta, "points": points, "training": diagnostics, "limitations": registration["limitations"]})
            original.write_json(folder / "kl_lock.json", {"registration_sha256": sha256_file(out / "kl_registration.json"),
                "head_sha256": sha256_file(folder / "head.npz"), "report_sha256": sha256_file(folder / "report.json"), "thresholds": thresholds})
    prior_comparison = pd.read_csv(controls / "comparison.csv")
    prior_comparison["kl_weight"] = np.where(prior_comparison["head"].eq("bayesian_reference"), bayes.KL_WEIGHT, np.nan)
    results = pd.concat([prior_comparison, pd.DataFrame(summary)], ignore_index=True)
    results.to_csv(out / "comparison.csv", index=False)
    original.write_json(out / "completion.json", {"completed": True, "phase": "development_kl_ablation",
        "encoders": ["official", "finetuned"], "kl_weights": list(ablation.VARIANTS.values())})
    print("Both lower-KL variants completed for both encoders. Final test remains unopened.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "controls", "data", "cache-run", "selection", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.reference, args.controls, args.data, args.cache_run, args.selection, args.out)
