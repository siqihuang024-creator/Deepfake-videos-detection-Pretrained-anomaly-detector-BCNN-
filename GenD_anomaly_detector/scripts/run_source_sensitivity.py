"""HDTF-only fitting/calibration sensitivity on the original cached development set."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gad import source_sensitivity as sensitivity, segment_training as original, old_metrics as metrics
from gad import feature_geometry, covariance_controls
from gad.diagnostic_protocol import sha256_file
from run_segment_controls import load_reference
from run_segment_kl_ablation import verify_controls
from run_geometry_diagnostics import wilson

RECIPE = {"fit_profiles": list(sensitivity.FIT_PROFILES), "calibration_profiles": list(sensitivity.CALIBRATION_PROFILES),
    "selection": "Whole-dataset ablation: remove all TalkingHead A units in each HDTF-only profile, never rank scores or images",
    "fit_roles": ["gradient_fit"], "early_stop": "unused for these statistical fits",
    "weights": "Same source-balanced clip weights, renormalized over retained fitting sources",
    "methods": ["centroid_l2", "mahalanobis_source_mean", "mahalanobis_clip_original_a30",
                "mahalanobis_clip_original_a50", "classifier_reference"],
    "thresholds": "Separate real-calibration linear 95%/99% quantiles; score >= threshold",
    "development": "Identical original videos and cached scores/inputs; no new development sample or final test",
    "bootstrap_draws": 500, "final_test": "forbidden", "neural_heads_retrained": False,
    "interpretation": "Source-composition sensitivity confounds source domain, sample size and image quality; not a causal cleaning test",
    "decision": "Report all four cells; no AUROC>0.80 pass criterion or best-cell selection"}


def verify_completed_covariance(directory, config):
    directory = Path(directory)
    completion = original.read_json(directory / "completion.json")
    reg = original.read_json(directory / "covariance_registration.json")
    if (not completion.get("completed") or completion.get("phase") != "development_clip_covariance"
            or completion["registration_sha256"] != sha256_file(directory / "covariance_registration.json")
            or reg["reference_configuration"] != config
            or reg["training_units_sha256"] != config["training_units_sha256"]):
        raise ValueError("Use the matching completed covariance run 344")
    for name, expected in completion["artifacts_sha256"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError("Covariance result changed: " + name)
    for name, expected in reg["algorithm_sha256"].items():
        base = Path(__file__).parent if name.startswith("run_") else Path(covariance_controls.__file__).parent
        if sha256_file(base / name) != expected:
            raise ValueError("Covariance algorithm changed: " + name)
    return sha256_file(directory / "covariance_registration.json")


def run(reference, controls, covariance_run, data, cache_run, selection, out):
    out, covariance_run = Path(out), Path(covariance_run)
    if out.exists():
        raise FileExistsError("Use a new source-sensitivity output directory")
    units, caches, config, _, _, _ = load_reference(reference, data, cache_run, selection)
    controls_hash = verify_controls(controls, reference, config)
    covariance_hash = verify_completed_covariance(covariance_run, config)
    masks = sensitivity.selection_masks(units)
    out.mkdir(parents=True)
    units.to_csv(out / "training_units.csv", index=False)
    selected = []
    for name, mask in masks.items():
        if name != "dev":
            part = units.loc[mask].copy()
            part["sensitivity_role"], part["source_profile"] = name.split("/")
            selected.append(part)
    pd.concat(selected, ignore_index=True).to_csv(out / "selected_units.csv", index=False)
    counts = {name: {"clips": int(mask.sum()), "sources": units.loc[mask, "source_clip"].nunique()}
              for name, mask in masks.items()}
    registration = {"phase": "development_source_sensitivity", "recipe": RECIPE, "reference_configuration": config,
        "controls_registration_sha256": controls_hash, "covariance_registration_sha256": covariance_hash,
        "training_units_sha256": sha256_file(out / "training_units.csv"),
        "selected_units_sha256": sha256_file(out / "selected_units.csv"), "counts": counts,
        "algorithm_sha256": {"run_source_sensitivity.py": sha256_file(__file__),
            "source_sensitivity.py": sha256_file(sensitivity.__file__),
            "feature_geometry.py": sha256_file(feature_geometry.__file__),
            "covariance_controls.py": sha256_file(covariance_controls.__file__)},
        "libraries": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "limitations": config["limitations"] + [RECIPE["interpretation"],
            "Stable results do not certify natural-human-face eligibility or validate unknown contamination prevalence.",
            "This check does not retrain Bayesian neural heads and cannot establish robustness of their negative result.",
            "HDTF-only calibration has fewer observations; nominal FPR is not guaranteed on DFDC."]}
    original.write_json(out / "sensitivity_registration.json", registration)
    comparison, artifacts = [], {}
    dev_rows = units.loc[masks["dev"]].reset_index(drop=True)
    for enc in ("official", "finetuned"):
        print("=== Source sensitivity: " + enc + " ===", flush=True)
        folder = out / enc
        folder.mkdir()
        cache = caches[enc]
        lookup = {key: i for i, key in enumerate(cache["video_keys"])}
        order = [lookup[key] for key in units.video_key]
        x, probabilities = cache["features"][order], cache["fake_prob"][order]
        old = pd.read_csv(covariance_run / enc / "all_unit_scores.csv", keep_default_na=False)
        if old.video_key.tolist() != units.video_key.tolist():
            raise ValueError("Covariance unit order changed")
        old_thresholds = original.read_json(covariance_run / enc / "thresholds.json")
        reports, saved_heads = {}, {}
        for fit_profile in sensitivity.FIT_PROFILES:
            fit = masks["gradient_fit/"+fit_profile]
            heads, _ = sensitivity.fit_family(x[fit], units.loc[fit, "source_clip"].tolist())
            saved_heads.update({fit_profile+"/"+name+"/"+key: value
                                for name, head in heads.items() for key, value in head.items()})
            scores = sensitivity.score_family(x, heads, probabilities)
            if fit_profile == "mixed":
                for name, values in scores.items():
                    np.testing.assert_allclose(values, old[name], rtol=1e-6, atol=1e-7)
            scored = units.copy()
            for name, values in scores.items(): scored[name] = values
            scored.to_csv(folder / (fit_profile+"_all_unit_scores.csv"), index=False)
            for name, values in scores.items():
                development = values[masks["dev"]]
                interval = metrics.clustered_auroc_interval(dev_rows.label, development, dev_rows.source_clip,
                                                            draws=RECIPE["bootstrap_draws"])
                for calibration_profile in sensitivity.CALIBRATION_PROFILES:
                    cal = values[masks["calibration/"+calibration_profile]]
                    cell = fit_profile+"__"+calibration_profile
                    reports.setdefault(cell, {}).setdefault(name, {})
                    for alpha in (.05, .01):
                        threshold = float(np.quantile(cal, 1-alpha))
                        if fit_profile == calibration_profile == "mixed":
                            np.testing.assert_allclose(threshold, old_thresholds[name][str(alpha)], rtol=1e-6, atol=1e-7)
                        point = metrics.detection_metrics((dev_rows.label == 0).astype(int), development, threshold, alpha)
                        point["roc_interpolated_tpr_at_nominal_fpr"] = point.pop("tpr_at_target_fpr")
                        point["calibration_observed_fpr"] = float((cal >= threshold).mean())
                        real = dev_rows.label.eq(0).to_numpy()
                        point["dev_real_fpr_wilson95"] = wilson(int((development[real] >= threshold).sum()), int(real.sum()))
                        reports[cell][name][str(alpha)] = point
                        comparison.append({"encoder": enc, "fit_profile": fit_profile, "calibration_profile": calibration_profile,
                            "head": name, "nominal_fpr": alpha, "calibration_n": len(cal),
                            "fit_sources": counts["gradient_fit/"+fit_profile]["sources"],
                            "auc_ci_low": interval["low"], "auc_ci_high": interval["high"],
                            **{key: value for key, value in point.items() if np.isscalar(value) or value is None}})
            print("Completed fitting profile: " + fit_profile, flush=True)
        np.savez_compressed(folder / "distance_heads.npz", **saved_heads)
        original.write_json(folder / "report.json", {"encoder": enc, "cells": reports, "limitations": registration["limitations"]})
        for path in folder.iterdir(): artifacts[path.relative_to(out).as_posix()] = sha256_file(path)
    table = pd.DataFrame(comparison)
    if (table.groupby(["encoder", "fit_profile", "head", "nominal_fpr"]).auroc.nunique() != 1).any():
        raise ValueError("Calibration unexpectedly changed AUROC")
    if table.loc[table["head"].eq("classifier_reference")].groupby("encoder").auroc.nunique().ne(1).any():
        raise ValueError("Source restriction unexpectedly changed frozen classifier AUROC")
    table.to_csv(out / "comparison.csv", index=False)
    for name in ("training_units.csv", "selected_units.csv", "comparison.csv"):
        artifacts[name] = sha256_file(out / name)
    original.write_json(out / "completion.json", {"completed": True, "phase": "development_source_sensitivity",
        "registration_sha256": sha256_file(out / "sensitivity_registration.json"), "artifacts_sha256": artifacts})
    print("All four fitting/calibration cells completed. Final test remains unopened.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "controls", "covariance-run", "data", "cache-run", "selection", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.reference, args.controls, args.covariance_run, args.data, args.cache_run, args.selection, args.out)
