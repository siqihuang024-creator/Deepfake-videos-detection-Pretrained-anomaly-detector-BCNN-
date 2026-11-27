"""CPU covariance, real-source shift and frozen-classifier-direction diagnosis."""
import argparse
from contextlib import redirect_stdout
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gad import feature_geometry as geometry, classifier_direction as direction_reader
from gad import segment_training as original, old_metrics as metrics
from gad.diagnostic_protocol import sha256_file
from run_segment_controls import load_reference
from run_segment_kl_ablation import verify_controls

RECIPE = {
    "covariance": "LedoitWolf on one mean vector per gradient-fitting source; not 679 independent clips",
    "covariance_scope": "Between-source mean variation; within-source clip covariance is not separately modeled",
    "centroid": "Equal total fitting weight per source; identical centroid to the existing L2 baseline",
    "direction": "Unit vector of frozen classifier weight_fake - weight_real, matched to actual cached model state",
    "residual_space": "d-1 Householder coordinates orthogonal to classifier direction; no norm rescaling",
    "fit_roles": ["gradient_fit"], "calibration_role": "calibration", "target_fprs": [.05, .01],
    "source_diagnosis": "Source-mean descriptive MMD and score summaries on genuine samples only; no fitting on development",
    "final_test": "forbidden", "NIW": "deferred", "bootstrap_draws": 500,
}


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    scale = 1 + z*z/n
    middle = (p + z*z/(2*n)) / scale
    radius = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / scale
    return {"low": float(max(0., middle-radius)), "high": float(min(1., middle+radius)),
            "limitation": "Binomial interval assumes independent real sources; unknown shared actors are not accounted for."}


def run(reference, controls, data, cache_run, selection, out, loader=None):
    out, controls = Path(out), Path(controls)
    if out.exists():
        raise FileExistsError("Use a new geometry output directory")
    units, caches, config, _, _, _ = load_reference(reference, data, cache_run, selection)
    control_hash = verify_controls(controls, reference, config)
    masks = {role: units.head_role.eq(role).to_numpy() for role in ("gradient_fit", "early_stop", "calibration", "dev")}
    out.mkdir(parents=True)
    units.to_csv(out / "training_units.csv", index=False)
    direction_values, direction_info = {}, {}
    for encoder in ("official", "finetuned"):
        print("Reading cached-model classifier on CPU: " + encoder, flush=True)
        with (out / (encoder + "_model_load.log")).open("w", encoding="utf-8") as log, redirect_stdout(log):
            values, info = direction_reader.read_direction(caches[encoder]["metadata"]["signature"], loader=loader)
        direction_values.update({encoder + "/" + key: value for key, value in values.items()})
        direction_info[encoder] = info
    np.savez(out / "classifier_directions.npz", **direction_values)
    registration = {"purpose": "Frozen-feature development geometry diagnosis; no network training or final testing",
        "recipe": RECIPE, "reference_configuration": config, "controls_registration_sha256": control_hash,
        "training_units_sha256": sha256_file(out / "training_units.csv"),
        "directions_sha256": sha256_file(out / "classifier_directions.npz"), "direction_provenance": direction_info,
        "algorithm_sha256": {"run_geometry_diagnostics.py": sha256_file(__file__),
            "feature_geometry.py": sha256_file(geometry.__file__), "classifier_direction.py": sha256_file(direction_reader.__file__),
            "run_segment_controls.py": sha256_file(Path(__file__).with_name("run_segment_controls.py")),
            "run_segment_kl_ablation.py": sha256_file(Path(__file__).with_name("run_segment_kl_ablation.py"))},
        "libraries": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__},
        "limitations": config["limitations"] + [
            "Real-only refers to downstream fitting/calibration; GenD itself used real/fake supervision.",
            "Residual-direction results measure dependence on one frozen linear direction, not all supervised information.",
            "Score/domain distribution comparisons are descriptive, not causal proof. Sample-size and source-composition effects remain."]}
    original.write_json(out / "geometry_registration.json", registration)
    summary = []
    for encoder in ("official", "finetuned"):
        print("\n=== Geometry: " + encoder + " ===", flush=True)
        folder = out / encoder
        folder.mkdir()
        cache = caches[encoder]
        lookup = {key: i for i, key in enumerate(cache["video_keys"])}
        order = [lookup[key] for key in units.video_key]
        x, probabilities = cache["features"][order], cache["fake_prob"][order]
        u = direction_values[encoder + "/direction"]
        residual = geometry.remove_direction(x, u)
        sources = units.loc[masks["gradient_fit"], "source_clip"].tolist()
        heads = {"original": geometry.fit_distances(x[masks["gradient_fit"]], sources),
                 "residual": geometry.fit_distances(residual[masks["gradient_fit"]], sources)}
        l2, mahal = geometry.score_distances(x, heads["original"])
        residual_l2, residual_mahal = geometry.score_distances(residual, heads["residual"])
        direction_l2 = ((np.asarray(x, float) - heads["original"]["mean"]) @ u) ** 2
        np.testing.assert_allclose(l2, direction_l2 + residual_l2, rtol=1e-8, atol=1e-10)
        np.savez(folder / "distance_heads.npz", **{space + "/" + key: value for space, head in heads.items() for key, value in head.items()})
        scores = {"centroid_l2": l2, "mahalanobis_source_mean": mahal,
            "centroid_without_classifier_direction": residual_l2,
            "mahalanobis_without_classifier_direction": residual_mahal,
            "classifier_direction_only_l2": direction_l2, "classifier_reference": probabilities}
        # Original centroid and classifier must reproduce the completed controls.
        for role, filename in (("calibration", "calibration_scores.csv"), ("dev", "development_scores.csv")):
            previous = pd.read_csv(controls / encoder / filename, keep_default_na=False).set_index("video_key")
            keys = units.loc[masks[role], "video_key"]
            for name in ("centroid_l2", "classifier_reference"):
                np.testing.assert_allclose(scores[name][masks[role]], previous.loc[keys, name], rtol=1e-6, atol=1e-7)
        scored = units.copy()
        for name, values in scores.items():
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite geometry score")
            scored[name] = values
        scored["signed_classifier_coordinate"] = np.asarray(x, float) @ u
        scored.to_csv(folder / "all_unit_scores.csv", index=False)
        scored.loc[masks["dev"]].to_csv(folder / "development_scores.csv", index=False)
        scored.loc[masks["calibration"]].to_csv(folder / "calibration_scores.csv", index=False)
        dev_rows = units.loc[masks["dev"]].reset_index(drop=True)
        reports, thresholds, false_rows, real_summaries, domain_scores = {}, {}, [], [], []
        for name, values in scores.items():
            calibration, development = values[masks["calibration"]], values[masks["dev"]]
            thresholds[name], reports[name] = {}, {}
            interval = metrics.clustered_auroc_interval(dev_rows.label, development, dev_rows.source_clip,
                                                        draws=RECIPE["bootstrap_draws"])
            for alpha in RECIPE["target_fprs"]:
                threshold = float(np.quantile(calibration, 1-alpha))
                thresholds[name][str(alpha)] = threshold
                point = metrics.detection_metrics((dev_rows.label == 0).astype(int), development, threshold, alpha)
                point["roc_interpolated_tpr_at_nominal_fpr"] = point.pop("tpr_at_target_fpr")
                point["calibration_observed_fpr"] = float((calibration >= threshold).mean())
                real = dev_rows.label.eq(0).to_numpy()
                point["dev_real_fpr_wilson95"] = wilson(int((development[real] >= threshold).sum()), int(real.sum()))
                reports[name][str(alpha)] = point
                summary.append({"encoder": encoder, "head": name, "nominal_fpr": alpha,
                    "auc_ci_low": interval["low"], "auc_ci_high": interval["high"],
                    **{key: value for key, value in point.items() if np.isscalar(value) or value is None}})
                false = dev_rows.loc[real & (development >= threshold)].copy()
                false["head"], false["nominal_fpr"], false["score"] = name, alpha, development[real & (development >= threshold)]
                false_rows.append(false)
            # Summaries average scores inside source, so each source contributes once.
            for (role, dataset), part in scored.loc[scored.label.eq(0)].groupby(["head_role", "dataset"]):
                means = part.groupby("source_clip")[name].mean().to_numpy()
                real_summaries.append({"head": name, "role": role, "dataset": dataset,
                    "clips": len(part), "sources": len(means),
                    **dict(zip(("q05", "q25", "median", "q75", "q95"), np.quantile(means, [.05, .25, .5, .75, .95]))),
                    "fraction_source_mean_above_cal95": float((means >= thresholds[name]["0.05"]).mean())})
            real_development = development[dev_rows.label.eq(0)]
            domain_scores.append({"head": name, "calibration_real_sources": len(calibration), "DFDC_real_sources": len(real_development),
                "domain_score_auc_DFDC_real_higher": float(roc_auc_score(
                    np.r_[np.zeros(len(calibration)), np.ones(len(real_development))], np.r_[calibration, real_development])),
                "limitation": "Domain ranking using a pre-existing score; not fake detection and not a learned domain classifier."})
        pd.concat(false_rows, ignore_index=True).to_csv(folder / "false_positives.csv", index=False)
        pd.DataFrame(real_summaries).to_csv(folder / "real_source_score_summary.csv", index=False)
        original.write_json(folder / "real_source_shift.json", {"original_features": geometry.real_feature_shift(x, units),
            "residual_features": geometry.real_feature_shift(residual, units), "score_domain_rankings": domain_scores})
        correlations = {}
        dev_scores = scored.loc[masks["dev"]]
        for subset, mask in (("all", np.ones(len(dev_scores), bool)), ("real", dev_scores.label.eq(0)), ("fake", dev_scores.label.eq(1))):
            ranks = dev_scores.loc[mask, list(scores)].rank()
            correlations[subset] = ranks.corr().to_dict()
        report = {"phase": "development_geometry", "encoder": encoder, "heads": reports,
            "source_covariance": {space: {"source_count": int(head["source_count"]), "dimension": len(head["mean"]),
                "shrinkage": head["shrinkage"], "covariance_trace": head["covariance_trace"]} for space, head in heads.items()},
            "spearman_correlations": correlations, "limitations": registration["limitations"]}
        original.write_json(folder / "report.json", report)
        original.write_json(folder / "geometry_lock.json", {"registration_sha256": sha256_file(out / "geometry_registration.json"),
            "heads_sha256": sha256_file(folder / "distance_heads.npz"), "report_sha256": sha256_file(folder / "report.json"), "thresholds": thresholds})
        print("Completed geometry and real-source summaries: " + encoder, flush=True)
    pd.DataFrame(summary).to_csv(out / "comparison.csv", index=False)
    original.write_json(out / "completion.json", {"completed": True, "phase": "development_geometry", "encoders": ["official", "finetuned"]})
    print("All geometry diagnostics completed. Final test remains unopened.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "controls", "data", "cache-run", "selection", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.reference, args.controls, args.data, args.cache_run, args.selection, args.out)
