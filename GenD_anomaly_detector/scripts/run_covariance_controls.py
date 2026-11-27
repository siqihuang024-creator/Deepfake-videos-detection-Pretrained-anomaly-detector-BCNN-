"""Fixed clip-covariance follow-up to completed geometry diagnostics; CPU only."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gad import covariance_controls as covariance, feature_geometry as geometry
from gad import segment_training as original, old_metrics as metrics
from gad.diagnostic_protocol import sha256_file
from run_segment_controls import load_reference
from run_segment_kl_ablation import verify_controls
from run_geometry_diagnostics import wilson

RECIPE = {
    "shrinkages": list(covariance.SHRINKAGES), "spaces": ["original", "residual"],
    "fit": "Only gradient_fit clips; each source total weight 1/G, clips within source equal",
    "covariance": "Biased weighted population covariance = between-source means + within-source variation",
    "regularization": "(1-alpha)*weighted_covariance + alpha*trace(weighted_covariance)/dimension*I",
    "independence": "Correlated clips are not treated as independent observations; no iid correction or inferred sample-size prior",
    "reference": "Replay all six geometry-342 scores on exact original units and caches",
    "direction": "Reuse hash-verified frozen geometry classifier direction; d-1 orthogonal coordinates, no renormalization",
    "calibration": "Same real calibration units, separate linear 95% and 99% quantiles for each score; score >= threshold",
    "decomposition": "Own empirical-null spectral energy; fixed OLD-null self/support/cross terms. Old-null self/total is not a probability",
    "selection": "Report both fixed alphas and both spaces; no best-variant selection",
    "bootstrap_draws": 500, "final_test": "forbidden", "NIW": "not implemented in this follow-up",
}


def verify_geometry(directory, config, units, controls_hash):
    directory = Path(directory)
    reg = original.read_json(directory / "geometry_registration.json")
    if (original.read_json(directory / "completion.json") != {
            "completed": True, "phase": "development_geometry", "encoders": ["official", "finetuned"]}
            or reg["reference_configuration"] != config
            or reg["controls_registration_sha256"] != controls_hash
            or reg["training_units_sha256"] != config["training_units_sha256"]
            or sha256_file(directory / "training_units.csv") != reg["training_units_sha256"]
            or not pd.read_csv(directory / "training_units.csv", dtype=str, keep_default_na=False).equals(units.astype(str))
            or sha256_file(directory / "classifier_directions.npz") != reg["directions_sha256"]):
        raise ValueError("Geometry reference data/direction/registration changed")
    for name, expected in reg["algorithm_sha256"].items():
        base = Path(__file__).parent if name.startswith("run_") else Path(geometry.__file__).parent
        if sha256_file(base / name) != expected:
            raise ValueError("Geometry reference algorithm changed: " + name)
    with np.load(directory / "classifier_directions.npz", allow_pickle=False) as values:
        directions = dict(values)
    heads = {}
    for enc in ("official", "finetuned"):
        folder = directory / enc
        lock = original.read_json(folder / "geometry_lock.json")
        if (lock["registration_sha256"] != sha256_file(directory / "geometry_registration.json")
                or lock["heads_sha256"] != sha256_file(folder / "distance_heads.npz")
                or lock["report_sha256"] != sha256_file(folder / "report.json")
                or reg["direction_provenance"][enc]["weights_sha256"] != config["cache_fingerprints"][enc]["signature"]["weights_sha256"]):
            raise ValueError("Geometry head/report/model lock changed")
        np.testing.assert_allclose(directions[enc + "/direction"],
            geometry.classifier_direction(directions[enc + "/linear_weights"]), rtol=1e-10, atol=1e-12)
        with np.load(folder / "distance_heads.npz", allow_pickle=False) as values:
            heads[enc] = {space: {k.split("/", 1)[1]: values[k] for k in values.files if k.startswith(space + "/")}
                          for space in ("original", "residual")}
    return reg, directions, heads


def choose_inspection_targets(scored, threshold, maximum=6):
    selected = scored.loc[scored.head_role.eq("calibration") & scored.label.eq(0)
                          & (scored.centroid_l2 >= threshold)].copy().sort_values("video_key")
    if selected.empty or len(selected) > maximum:
        raise ValueError("Unexpected inspection target count; do not expand score-selected inspection silently")
    selected["selection_score"] = selected.centroid_l2
    selected["selection_threshold"] = threshold
    # Keep only original metadata plus an explicit inspection decision, not all scores.
    columns = [c for c in selected if c not in SCORE_NAMES and not c.startswith("mahalanobis_clip_")]
    return selected[columns].reset_index(drop=True)


SCORE_NAMES = ("centroid_l2", "mahalanobis_source_mean", "centroid_without_classifier_direction",
               "mahalanobis_without_classifier_direction", "classifier_direction_only_l2", "classifier_reference")


def run(reference, controls, geometry_run, data, cache_run, selection, out):
    out, geometry_run = Path(out), Path(geometry_run)
    if out.exists():
        raise FileExistsError("Use a new covariance-control output directory")
    units, caches, config, _, _, _ = load_reference(reference, data, cache_run, selection)
    controls_hash = verify_controls(controls, reference, config)
    _, directions, old_heads = verify_geometry(geometry_run, config, units, controls_hash)
    masks = {role: units.head_role.eq(role).to_numpy() for role in ("gradient_fit", "early_stop", "calibration", "dev")}
    out.mkdir(parents=True)
    units.to_csv(out / "training_units.csv", index=False)
    registration = {"phase": "development_clip_covariance", "recipe": RECIPE,
        "reference_configuration": config, "controls_registration_sha256": controls_hash,
        "geometry_registration_sha256": sha256_file(geometry_run / "geometry_registration.json"),
        "geometry": str(geometry_run.resolve()), "training_units_sha256": sha256_file(out / "training_units.csv"),
        "algorithm_sha256": {"run_covariance_controls.py": sha256_file(__file__),
                              "covariance_controls.py": sha256_file(covariance.__file__)},
        "limitations": config["limitations"] + [
            "Exploratory development follow-up; no independent test evidence or corrected multiple-comparison inference.",
            "A changed low-variance contribution is diagnostic, not proof of the cause of AUROC changes.",
            "New variants change covariance scope and shrinkage together; they do not uniquely isolate either factor.",
            "Frozen GenD and classifier directions used supervised real/fake training."]}
    original.write_json(out / "covariance_registration.json", registration)
    comparison, artifacts = [], {}
    for enc in ("official", "finetuned"):
        print("=== Clip covariance: " + enc + " ===", flush=True)
        folder = out / enc
        folder.mkdir()
        cache = caches[enc]
        lookup = {key: i for i, key in enumerate(cache["video_keys"])}
        order = [lookup[key] for key in units.video_key]
        x = cache["features"][order]
        spaces = {"original": x, "residual": geometry.remove_direction(x, directions[enc + "/direction"])}
        l2, mahal = geometry.score_distances(x, old_heads[enc]["original"])
        rl2, rmahal = geometry.score_distances(spaces["residual"], old_heads[enc]["residual"])
        scores = dict(zip(SCORE_NAMES, (l2, mahal, rl2, rmahal,
            ((x.astype(float)-old_heads[enc]["original"]["mean"]) @ directions[enc + "/direction"])**2,
            cache["fake_prob"][order])))
        previous = pd.read_csv(geometry_run / enc / "all_unit_scores.csv", keep_default_na=False)
        if (previous.video_key.tolist() != units.video_key.tolist()
                or previous.head_role.tolist() != units.head_role.tolist()):
            raise ValueError("Geometry score ordering/roles changed")
        old_lock = original.read_json(geometry_run / enc / "geometry_lock.json")
        for name in SCORE_NAMES:
            np.testing.assert_allclose(scores[name], previous[name], rtol=1e-6, atol=1e-7)
            for alpha in (.05, .01):
                np.testing.assert_allclose(np.quantile(scores[name][masks["calibration"]], 1-alpha),
                                           old_lock["thresholds"][name][str(alpha)], rtol=1e-6, atol=1e-7)
        groups = units.loc[masks["gradient_fit"], "source_clip"].tolist()
        heads, decomposition, diagnostics = {}, [], {}
        for space, features in spaces.items():
            old_name = "mahalanobis_source_mean" if space == "original" else "mahalanobis_without_classifier_direction"
            old = covariance.recover_source_mean_head(old_heads[enc][space])
            mean, empirical, traces = covariance.clip_covariance(features[masks["gradient_fit"]], groups)
            np.testing.assert_allclose(mean, old["mean"], rtol=1e-8, atol=1e-12)
            eigensystem = covariance.spectrum(empirical)
            variants = {old_name: old}
            for alpha in covariance.SHRINKAGES:
                name = "mahalanobis_clip_" + space + "_a" + str(int(alpha*100))
                variants[name] = covariance.regularize(mean, empirical, alpha, eigensystem)
            diagnostics[space] = traces
            for name, head in variants.items():
                parts = covariance.distance_parts(features, head, old["empirical_null_basis"])
                if name == old_name:
                    np.testing.assert_allclose(parts["score"], scores[name], rtol=1e-6, atol=1e-7)
                scores[name] = parts["score"]
                heads.update({name + "/" + k: v for k, v in head.items()})
                part = units[["video_key", "head_role", "dataset", "source_clip", "label"]].copy()
                part["head"], part["space"] = name, space
                for key, values in parts.items():
                    part[key] = values
                decomposition.append(part)
                diagnostics[name] = {"shrinkage": head["shrinkage"], "floor": head["floor"],
                    "dimension": len(mean), "empirical_null_dimensions": head["empirical_null_basis"].shape[1],
                    "old_null_dimensions": old["empirical_null_basis"].shape[1],
                    "condition_number": head["condition_number"]}
                print(name + ": own-null dimensions=" + str(head["empirical_null_basis"].shape[1]), flush=True)
        np.savez_compressed(folder / "distance_heads.npz", **heads)
        decomposition_table = pd.concat(decomposition, ignore_index=True)
        decomposition_table.to_csv(folder / "distance_decomposition.csv", index=False)
        decomposition_summary = []
        for keys, part in decomposition_table.groupby(["head", "space", "head_role", "dataset", "label"]):
            decomposition_summary.append(dict(zip(("head", "space", "head_role", "dataset", "label"), keys),
                clips=len(part), sources=part.source_clip.nunique(),
                median_score=float(part.score.median()), median_own_null_share=float(part.own_null_share.median()),
                mean_old_null_self=float(part.old_null_self.mean()),
                mean_old_support_self=float(part.old_support_self.mean()), mean_old_cross=float(part.old_cross.mean())))
        pd.DataFrame(decomposition_summary).to_csv(folder / "decomposition_summary.csv", index=False)
        scored = units.copy()
        for name, values in scores.items():
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite covariance score")
            scored[name] = values
        scored.to_csv(folder / "all_unit_scores.csv", index=False)
        for role in ("calibration", "dev"):
            scored.loc[masks[role]].to_csv(folder / (role + "_scores.csv"), index=False)
        dev = units.loc[masks["dev"]].reset_index(drop=True)
        reports, thresholds = {}, {}
        for name, values in scores.items():
            reports[name], thresholds[name] = {}, {}
            cal, development = values[masks["calibration"]], values[masks["dev"]]
            interval = metrics.clustered_auroc_interval(dev.label, development, dev.source_clip, draws=RECIPE["bootstrap_draws"])
            for alpha in (.05, .01):
                threshold = float(np.quantile(cal, 1-alpha))
                thresholds[name][str(alpha)] = threshold
                point = metrics.detection_metrics((dev.label == 0).astype(int), development, threshold, alpha)
                point["roc_interpolated_tpr_at_nominal_fpr"] = point.pop("tpr_at_target_fpr")
                point["calibration_observed_fpr"] = float((cal >= threshold).mean())
                real = dev.label.eq(0).to_numpy()
                point["dev_real_fpr_wilson95"] = wilson(int((development[real] >= threshold).sum()), int(real.sum()))
                reports[name][str(alpha)] = point
                comparison.append({"encoder": enc, "head": name, "nominal_fpr": alpha,
                    "auc_ci_low": interval["low"], "auc_ci_high": interval["high"],
                    **{k: v for k, v in point.items() if np.isscalar(v) or v is None}})
        original.write_json(folder / "report.json", {"encoder": enc, "heads": reports,
            "covariance_diagnostics": diagnostics, "limitations": registration["limitations"]})
        original.write_json(folder / "thresholds.json", thresholds)
        if enc == "official":
            targets = choose_inspection_targets(scored, thresholds["centroid_l2"]["0.05"])
            targets.to_csv(out / "inspection_targets.csv", index=False)
        for path in sorted(folder.iterdir()):
            artifacts[path.relative_to(out).as_posix()] = sha256_file(path)
    pd.DataFrame(comparison).to_csv(out / "comparison.csv", index=False)
    original.write_json(out / "inspection_request.json", {
        "target_sha256": sha256_file(out / "inspection_targets.csv"),
        "covariance_registration_sha256": sha256_file(out / "covariance_registration.json"),
        "data_plan_sha256": config["data_plan_sha256"],
        "candidate_list_registration_sha256": config["candidate_list_registration_sha256"],
        "selection": "All original official centroid calibration scores >= original cal95 threshold, maximum six",
        "frames_per_clip": 5, "policy": "Quality review only; do not delete or reassign units based on scores"})
    for name in ("training_units.csv", "comparison.csv", "inspection_targets.csv", "inspection_request.json"):
        artifacts[name] = sha256_file(out / name)
    original.write_json(out / "completion.json", {"completed": True, "phase": "development_clip_covariance",
        "encoders": ["official", "finetuned"], "registration_sha256": sha256_file(out / "covariance_registration.json"),
        "artifacts_sha256": artifacts})
    print("Both fixed covariance variants and decompositions completed. Final test remains unopened.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "controls", "geometry-run", "data", "cache-run", "selection", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.reference, args.controls, args.geometry_run, args.data, args.cache_run, args.selection, args.out)
