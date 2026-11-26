"""CPU controls on the exact registered units and cached inputs of job 313."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gad import segment_training as original, bayesian_svdd as bayes, svdd_controls as controls
from gad import old_metrics as metrics
from gad.diagnostic_protocol import sha256_file, false_positive_groups

CONTROL_RECIPE = {
    "centroid_l2": "Squared Euclidean distance to the source-balanced fitting centroid; no learning",
    "deterministic_svdd": "Same 1024-512-128 bias-free network, initial mean matrices and center, source weights, Adam and early stopping; point weights, no KL or weight decay",
    "deterministic_prior_matched_svdd": "Same point network; add KL_WEIGHT / fitting_sources times sum(weight^2 / prior_variance) / 2",
    "comparison_scope": "Controls reuse exact old units/inputs; new deterministic minibatch RNG stream differs from the Bayesian sampling stream. Differences do not uniquely identify posterior uncertainty as the cause.",
    "final_test": "forbidden", "seed": 42,
}


def load_reference(reference, data, cache_run, selection):
    reference = Path(reference)
    config_path = reference / "training_registration.json"
    config = original.read_json(config_path)
    completion = original.read_json(reference / "completion.json")
    if completion != {"completed": True, "encoders": ["official", "finetuned"], "phase": "development"}:
        raise ValueError("Use the completed paired Bayesian development run")
    if config["recipe"] != original.RECIPE:
        raise ValueError("Reference training recipe differs")
    constants = {"widths": list(bayes.WIDTHS), "seed": bayes.SEED, "max_epochs": bayes.MAX_EPOCHS,
        "patience": bayes.PATIENCE, "batch_size": bayes.BATCH_SIZE, "learning_rate": bayes.LEARNING_RATE,
        "kl_weight": bayes.KL_WEIGHT, "center_epsilon": bayes.CENTER_EPS,
        "validation_draws": bayes.VALIDATION_DRAWS, "posterior_draws": bayes.POSTERIOR_DRAWS}
    if any(config["recipe"][key] != value for key, value in constants.items()):
        raise ValueError("Runtime hyperparameters differ from the Bayesian reference")
    path = reference / "training_units.csv"
    if sha256_file(path) != config["training_units_sha256"]:
        raise ValueError("Reference unit table changed")
    units, caches, provenance = original.load_inputs(data, cache_run, selection)
    frozen = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not frozen.equals(units.astype(str)):
        raise ValueError("Reference units/roles/order differ from the verified inputs")
    for field in ("data_plan_sha256", "candidate_list_registration_sha256", "selection_sha256", "cache_fingerprints"):
        if config[field] != provenance[field]:
            raise ValueError("Reference input fingerprint changed: " + field)
    for name, expected in config["algorithm_sha256"].items():
        if sha256_file(Path(original.__file__).parent / name) != expected:
            raise ValueError("Original algorithm changed: " + name)
    heads, reports, locks = {}, {}, {}
    for name in ("official", "finetuned"):
        folder = reference / name
        lock = original.read_json(folder / "development_lock.json")
        if (lock["registration_sha256"] != sha256_file(config_path)
                or lock["head_sha256"] != sha256_file(folder / "head.npz")
                or lock["report_sha256"] != sha256_file(folder / "report.json")):
            raise ValueError("Reference head/report lock changed")
        with np.load(folder / "head.npz", allow_pickle=False) as stored:
            heads[name] = dict(stored)
        reports[name], locks[name] = original.read_json(folder / "report.json"), lock
    return units, caches, config, heads, reports, locks


def run(reference, data, cache_run, selection, out):
    out, reference = Path(out), Path(reference)
    if out.exists():
        raise FileExistsError("Use a new control output directory")
    units, caches, config, old_heads, old_reports, old_locks = load_reference(reference, data, cache_run, selection)
    out.mkdir(parents=True)
    units.to_csv(out / "training_units.csv", index=False)
    registration = {"purpose": "Small development-only controls; no final testing", "recipe": CONTROL_RECIPE,
        "reference_registration_sha256": sha256_file(reference / "training_registration.json"),
        "reference": str(reference.resolve()), "training_units_sha256": sha256_file(out / "training_units.csv"),
        "reference_configuration": config, "libraries": {"torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "algorithm_sha256": {"run_segment_controls.py": sha256_file(__file__), "svdd_controls.py": sha256_file(controls.__file__)}}
    original.write_json(out / "control_registration.json", registration)
    masks = {name: units.head_role.eq(name).to_numpy() for name in ("gradient_fit", "early_stop", "calibration", "dev")}
    summary = []
    for encoder in ("official", "finetuned"):
        print("\n=== Controls: " + encoder + " ===", flush=True)
        folder = out / encoder
        folder.mkdir()
        values = caches[encoder]
        lookup = {key: i for i, key in enumerate(values["video_keys"])}
        order = [lookup[k] for k in units.video_key]
        x, probs = values["features"][order], values["fake_prob"][order]
        dev_rows = units.loc[masks["dev"]].reset_index(drop=True)
        cal_rows = units.loc[masks["calibration"]].reset_index(drop=True)
        groups = units.loc[masks["gradient_fit"], "source_clip"].tolist()
        early_groups = units.loc[masks["early_stop"], "source_clip"].tolist()
        old_scores, old_extras = bayes.score(x[masks["dev"]], old_heads[encoder], "cpu")
        old_cal, _ = bayes.score(x[masks["calibration"]], old_heads[encoder], "cpu")
        # An explicit replay verifies that cache aggregation/input ordering still
        # reproduces the saved network scores, without retraining Bayesian heads.
        for role, scores, column in (("development", old_scores, "bayesian_deep_svdd_score"),
                                      ("calibration", old_cal, "bayesian_deep_svdd_score")):
            frame = pd.read_csv(reference / encoder / (role + "_scores.csv"), keep_default_na=False)
            keys = units.loc[masks["dev" if role == "development" else "calibration"], "video_key"]
            np.testing.assert_allclose(scores, frame.set_index("video_key").loc[keys, column], rtol=1e-5, atol=1e-7)
            np.testing.assert_allclose(probs[masks["dev" if role == "development" else "calibration"]],
                frame.set_index("video_key").loc[keys, "classifier_score"], rtol=1e-6, atol=1e-7)
        scores_by_head = {"bayesian_reference": (old_cal, old_scores),
                          "classifier_reference": (probs[masks["calibration"]], probs[masks["dev"]])}
        diagnostics = {}
        trained = {}
        for name in ("centroid_l2", "deterministic_svdd", "deterministic_prior_matched_svdd"):
            print("--- " + name + " ---", flush=True)
            if name == "centroid_l2":
                head = controls.fit_centroid(x[masks["gradient_fit"]], groups)
                score = controls.score_centroid
            else:
                def progress(event):
                    with (folder / (name + "_progress.jsonl")).open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(event) + "\n")
                    print("Epoch {epoch}: data={data_loss:.6g}, regularizer={regularizer:.6g}, early={early_stop_distance:.6g}, best={best_epoch}".format(**event), flush=True)
                head = controls.fit_deterministic(x[masks["gradient_fit"]], x[masks["early_stop"]], groups, early_groups,
                    prior_matched=name == "deterministic_prior_matched_svdd", progress=progress)
                score = controls.score_deterministic
                np.savetxt(folder / (name + "_history.csv"), head["training_history"], delimiter=",",
                    header="epoch,total_loss,data_loss,regularizer,early_stop_distance,elapsed_seconds", comments="")
                diagnostics[name] = {key: np.asarray(head[key]).item() for key in ("best_epoch", "epochs_run", "training_seconds",
                    "validation_distance", "train_score_relative_std", "train_embedding_variance_ratio", "near_constant_train_scores", "low_train_embedding_variance")}
            trained.update({name + "/" + k: v for k, v in head.items()})
            scores_by_head[name] = (score(x[masks["calibration"]], head), score(x[masks["dev"]], head))
        np.savez(folder / "control_heads.npz", **trained)
        scored, calibrated = dev_rows.copy(), cal_rows.copy()
        scored["reference_posterior_distance_variance"] = old_extras["posterior_distance_variance"]
        reports, thresholds, false_frames = {}, {}, []
        for name, (cal_values, scores) in scores_by_head.items():
            if not np.isfinite(cal_values).all() or not np.isfinite(scores).all():
                raise ValueError("Nonfinite control scores")
            scored[name], calibrated[name] = scores, cal_values
            reports[name], thresholds[name] = {}, {}
            for alpha in original.RECIPE["target_fprs"]:
                if name.endswith("_reference"):
                    old_name = "bayesian_deep_svdd" if name == "bayesian_reference" else "classifier"
                    threshold = old_locks[encoder]["thresholds"][old_name][str(alpha)]
                else:
                    threshold = float(np.quantile(cal_values, 1 - alpha))
                thresholds[name][str(alpha)] = threshold
                point = metrics.detection_metrics((dev_rows.label == 0).astype(int), scores, threshold, alpha)
                point["roc_interpolated_tpr_at_nominal_fpr"] = point.pop("tpr_at_target_fpr")
                point["calibration_observed_fpr"] = float((cal_values >= threshold).mean())
                point["source_real_fpr"] = false_positive_groups(dev_rows, scores, threshold, "source_only")
                if name.endswith("_reference"):
                    for key in ("auroc", "tn", "fp", "fn", "tp"):
                        if not np.isclose(point[key], old_reports[encoder]["heads"][old_name][str(alpha)][key]):
                            raise ValueError("Replayed Bayesian/classifier metrics differ from the reference")
                reports[name][str(alpha)] = point
                summary.append({"encoder": encoder, "head": name, "nominal_fpr": alpha,
                    **{key: value for key, value in point.items() if np.isscalar(value) or value is None}})
                false = dev_rows.loc[dev_rows.label.eq(0) & (scores >= threshold)].copy()
                false["head"], false["score"], false["nominal_fpr"] = name, scores[dev_rows.label.eq(0) & (scores >= threshold)], alpha
                false_frames.append(false)
        scored.to_csv(folder / "development_scores.csv", index=False)
        calibrated.to_csv(folder / "calibration_scores.csv", index=False)
        pd.concat(false_frames, ignore_index=True).to_csv(folder / "false_positives.csv", index=False)
        original.write_json(folder / "report.json", {"phase": "development_controls", "encoder": encoder,
            "heads": reports, "training": diagnostics, "coverage": config["coverage"], "limitations": config["limitations"] + [CONTROL_RECIPE["comparison_scope"]]})
        original.write_json(folder / "control_lock.json", {"registration_sha256": sha256_file(out / "control_registration.json"),
            "heads_sha256": sha256_file(folder / "control_heads.npz"), "report_sha256": sha256_file(folder / "report.json"),
            "thresholds": thresholds, "scope": "Development snapshot only; no final test performed"})
    pd.DataFrame(summary).to_csv(out / "comparison.csv", index=False)
    original.write_json(out / "completion.json", {"completed": True, "phase": "development_controls", "encoders": ["official", "finetuned"]})
    print("All CPU controls completed. Final test remains unopened.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "data", "cache-run", "selection", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.reference, args.data, args.cache_run, args.selection, args.out)
