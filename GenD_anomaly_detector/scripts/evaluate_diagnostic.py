"""Parallel statistical diagnosis with independent calibration and a test lock.

The same evaluation engine supports the primary Bayesian Deep SVDD entry point
in evaluate_bayesian.py. Neither path trains a GenD encoder.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gad import anomaly, old_metrics as M
from gad.diagnostic_protocol import (EXPERIMENT_HEADS, false_positive_groups,
                                     read_registered_plan, sha256_file, split_head_training)
from gad.feature_cache import load_video_cache, signature_hash, video_key
from gad.scoring import nan_to_none


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(nan_to_none(value), stream, indent=2, allow_nan=False)


def resolve_cache(path):
    path = Path(path)
    if path.is_file():
        receipt = json.loads(path.read_text(encoding="utf-8"))
        path = Path(receipt["cache"])
        if sha256_file(path / "metadata.json") != receipt["metadata_sha256"]:
            raise ValueError("Cache metadata changed since its receipt")
    return path


def encoder_identity(signature):
    fields = ("weights_sha256", "checkpoint_sha256", "precision", "gend_commit", "preprocess",
              "extractor_source_sha256", "custom_preprocessing", "head_configuration", "cache_algorithm")
    if any(key not in signature for key in fields):
        raise ValueError("Cache lacks an encoder/preprocessing fingerprint")
    return {key: signature[key] for key in fields}


def registered_cache(cache_path, registration_path, plan_path, plan, videos, test=False):
    cache_path = resolve_cache(cache_path)
    cache = load_video_cache(cache_path)
    signature = cache["metadata"]["signature"]
    registration = json.loads(Path(registration_path).read_text(encoding="utf-8"))
    plan_hash = sha256_file(plan_path)
    list_name = "test/test.txt" if test else "development.txt"
    list_path = Path(registration_path).parent / list_name
    if (registration["plan_sha256"] != plan_hash
            or signature.get("diagnostic_plan_sha256") != plan_hash
            or registration["files"].get(list_name) != sha256_file(list_path)
            or signature.get("frame_list_sha256") != registration["files"][list_name]):
        raise ValueError("Cache/list registration does not belong to this locked protocol")
    roles = ("test",) if test else ("train", "calibration", "dev")
    expected = {key for role in roles for key in registration["coverage"][role]["usable_keys"]}
    if set(cache["video_keys"]) != expected:
        raise ValueError("Cache coverage differs from registered scorable videos")
    listed = videos.loc[videos["split"].isin(roles)].copy()
    if not expected.issubset(set(listed["video_key"])):
        raise ValueError("Cache contains videos outside its allowed roles")
    identity = encoder_identity(signature)
    if signature.get("model") != "official":
        if identity["checkpoint_sha256"] != plan["finetuned_selection"]["checkpoint_sha256"]:
            raise ValueError("Use only the registered LN-b128 validation winner")
    index = {key: i for i, key in enumerate(cache["video_keys"])}
    rows = listed.loc[listed["video_key"].isin(expected)].reset_index(drop=True)
    order = [index[key] for key in rows["video_key"]]
    return cache_path, cache, rows, cache["features"][order], cache["fake_prob"][order], {
        role: registration["coverage"][role] for role in roles}


def algorithm_hash():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root / "gad/anomaly.py", root / "gad/diagnostic_protocol.py",
             root / "gad/old_metrics.py", root / "gad/feature_cache.py", root / "gad/bayesian_svdd.py"]
    return signature_hash({p.name: sha256_file(p) for p in paths})


def detection_metrics(rows, scores, threshold, alpha):
    result = M.detection_metrics((rows["label"] == 0).astype(int), scores, threshold, alpha)
    result["roc_interpolated_tpr_at_nominal_fpr"] = result.pop("tpr_at_target_fpr")
    result["num_real"] = int(rows["label"].eq(0).sum())
    result["num_fake"] = int(rows["label"].eq(1).sum())
    return result


def contact_sheets(out, false_positives, cache_path, limit):
    """Bounded development-only face-crop examples, not a background-style analysis."""
    if not limit or false_positives.empty:
        return
    from PIL import Image, ImageDraw, ImageOps
    paths = (cache_path / "frame_paths.txt").read_text(encoding="utf-8").splitlines()
    grouped = {}
    for path in paths:
        grouped.setdefault(video_key(path), []).append(path)
    folder = out / "contact_sheets"
    folder.mkdir()
    items = []
    for index, row in enumerate(false_positives.drop_duplicates("video_key").head(limit).itertuples()):
        choices = grouped[row.video_key]
        canvas = Image.new("RGB", (640, 190), "white")
        ImageDraw.Draw(canvas).text((5, 3), f"Case {index + 1}: selected face crops", fill="black")
        for column, position in enumerate(np.linspace(0, len(choices) - 1, 4, dtype=int)):
            with Image.open(choices[position]) as picture:
                canvas.paste(ImageOps.contain(picture.convert("RGB"), (156, 156)), (column * 160, 27))
        name = f"case_{index + 1:03d}.jpg"
        canvas.save(folder / name)
        items.append({"image": name, "video_key": row.video_key, "rel_path": row.rel_path})
    write_json(folder / "index.json", items)


def run(plan_path, lists, development_cache, out, phase="dev", test_cache=None,
        lock=None, freeze_to=None, posterior_draws=0, bootstrap_draws=500, examples=20,
        profile="baseline", device="cpu"):
    plan_path, out = Path(plan_path), Path(out)
    if out.exists():
        raise FileExistsError(out)
    if posterior_draws < 0 or posterior_draws == 1 or bootstrap_draws < 2 or examples < 0:
        raise ValueError("Invalid draw count or example limit")
    if phase not in ("dev", "test") or (phase == "test" and (not lock or not test_cache or freeze_to)):
        raise ValueError("Final test requires a prior lock and a separate test cache")
    if freeze_to and Path(freeze_to).exists():
        raise FileExistsError(freeze_to)
    if profile not in EXPERIMENT_HEADS or device not in ("cpu", "cuda"):
        raise ValueError("Choose a registered profile and CPU/CUDA device")
    if profile == "baseline" and device != "cpu":
        raise ValueError("The statistical parallel experiment uses CPU only")
    if freeze_to and any((plan_path.parent / "final_test_receipts").glob("*.json")):
        raise ValueError("Freeze all main and parallel configurations before any final testing")
    plan, videos = read_registered_plan(plan_path)
    if plan["experiments"] != EXPERIMENT_HEADS or plan["target_fprs"] != [.05, .01]:
        raise ValueError("Initial diagnostic baseline recipe changed; register a new protocol")
    names = tuple(EXPERIMENT_HEADS[profile])
    recipe = None
    if profile == "bayesian":
        import torch
        from gad import bayesian_svdd
        if posterior_draws != 0:
            raise ValueError("--posterior-draws configures the NIW parallel experiment; Bayesian Deep SVDD uses its fixed 50-draw recipe")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; allocate a GPU or explicitly use CPU")
        recipe = {"widths": list(bayesian_svdd.WIDTHS), "bias": False, "seed": anomaly.SEED,
                  "max_epochs": bayesian_svdd.MAX_EPOCHS, "patience": bayesian_svdd.PATIENCE,
                  "learning_rate": bayesian_svdd.LEARNING_RATE, "batch_size": bayesian_svdd.BATCH_SIZE,
                  "kl_weight": bayesian_svdd.KL_WEIGHT, "center_epsilon": bayesian_svdd.CENTER_EPS,
                  "validation_draws": bayesian_svdd.VALIDATION_DRAWS,
                  "posterior_draws": bayesian_svdd.POSTERIOR_DRAWS, "device": device,
                  "input_centering": "gradient-fitting genuine videos only"}

    def score_head(name, features, head, calibration=False):
        if name == "bayesian_deep_svdd":
            return bayesian_svdd.score(features, head, torch.device(device))
        return anomaly.score_statistical_head(name, features, head, 0 if calibration else posterior_draws)
    dev_path, dev_cache, dev_rows, x, probabilities, coverage = registered_cache(
        development_cache, lists, plan_path, plan, videos)
    signature = dev_cache["metadata"]["signature"]
    import scipy
    import sklearn
    config = {"plan_sha256": sha256_file(plan_path), "algorithm_sha256": algorithm_hash(),
              "development_metadata_sha256": sha256_file(dev_path / "metadata.json"),
              "encoder": encoder_identity(signature), "heads": list(names), "experiment_profile": profile,
              "bayesian_recipe": recipe,
              "target_fprs": plan["target_fprs"], "posterior_draws": posterior_draws,
              "bootstrap_draws": bootstrap_draws, "threshold_comparison": "score >= threshold",
              "libraries": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
              "threshold_rule": "linear quantile of held-out genuine calibration scores"}
    if profile == "bayesian":
        config["libraries"]["torch"] = torch.__version__
    started = time.perf_counter()
    masks = {role: dev_rows["split"].eq(role).to_numpy() for role in ("train", "calibration", "dev")}
    for role, mask in masks.items():
        if not mask.any():
            raise ValueError(f"No scorable {role} videos")
    receipt = None
    head_training = None
    if phase == "dev":
        if profile == "bayesian":
            fit_mask, early_mask, head_training = split_head_training(dev_rows.loc[masks["train"]])
            out.mkdir(parents=True)

            def progress(event):
                with (out / "training_progress.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event) + "\n")
                print(f"Bayesian head epoch {event['epoch']}: loss={event['training_loss']:.6g}, "
                      f"early-stop distance={event['early_stop_distance']:.6g}, "
                      f"best={event['best_epoch']}", flush=True)

            head = bayesian_svdd.fit(x[masks["train"]][fit_mask], x[masks["train"]][early_mask], torch.device(device), progress)
            heads = {"bayesian_deep_svdd": head}
            head_training.update({key: np.asarray(head[key]).item() for key in (
                "best_epoch", "epochs_run", "training_seconds", "validation_distance", "train_score_mean",
                "train_score_variance", "train_score_relative_std", "near_constant_train_scores",
                "initial_train_embedding_variance", "train_embedding_variance", "train_embedding_variance_ratio",
                "low_train_embedding_variance")})
            np.savetxt(out / "training_history.csv", head["training_history"], delimiter=",",
                       header="epoch,training_loss,early_stop_distance,elapsed_seconds", comments="")
        else:
            heads = anomaly.fit_statistical_heads(x[masks["train"]], head_names=names)
        thresholds = {}
        calibration_points = {}
        for name, head in heads.items():
            values, _ = score_head(name, x[masks["calibration"]], head, calibration=True)
            thresholds[name] = {str(alpha): float(np.quantile(values, 1 - alpha)) for alpha in plan["target_fprs"]}
            calibration_points[name] = {str(alpha): float((values >= thresholds[name][str(alpha)]).mean())
                                        for alpha in plan["target_fprs"]}
        thresholds["classifier"] = {str(alpha): float(np.quantile(probabilities[masks["calibration"]], 1 - alpha))
                                     for alpha in plan["target_fprs"]}
        calibration_points["classifier"] = {str(alpha): float((probabilities[masks["calibration"]] >= thresholds["classifier"][str(alpha)]).mean())
                                             for alpha in plan["target_fprs"]}
        rows, features, probs = dev_rows.loc[masks["dev"]].reset_index(drop=True), x[masks["dev"]], probabilities[masks["dev"]]
        scored_cache = dev_path
    else:
        lock = Path(lock)
        frozen = json.loads((lock / "method_lock.json").read_text(encoding="utf-8"))
        if frozen["configuration"] != config or sha256_file(lock / "heads.npz") != frozen["heads_sha256"]:
            raise ValueError("Configuration, development cache, or locked heads changed")
        with np.load(lock / "heads.npz", allow_pickle=False) as saved:
            heads = {name: {key.split("/", 1)[1]: saved[key] for key in saved.files if key.startswith(name + "/")}
                     for name in names}
        thresholds = frozen["thresholds"]
        calibration_points = frozen["calibration_observed_fpr"]
        head_training = frozen.get("head_training")
        scored_cache, test_values, rows, features, probs, coverage = registered_cache(
            test_cache, lists, plan_path, plan, videos, test=True)
        if encoder_identity(test_values["metadata"]["signature"]) != config["encoder"]:
            raise ValueError("Development and test caches use different encoders or preprocessing")
        # One receipt per registered profile/encoder, regardless of output path.
        directory = plan_path.parent / "final_test_receipts"
        directory.mkdir(exist_ok=True)
        receipt = directory / (profile + "_" + signature_hash(config["encoder"]) + ".json")
        write_json(receipt, {"status": "started", "lock_sha256": sha256_file(lock / "method_lock.json"),
                             "out": str(out.resolve())})
    if not out.exists():
        out.mkdir(parents=True)
    report = {"phase": phase, "configuration": config, "coverage": coverage,
              "head_training": head_training,
              "calibration_videos": int(masks["calibration"].sum()),
              "calibration_fpr_resolution": 1 / int(masks["calibration"].sum()),
              "calibration_observed_fpr": calibration_points,
              "calibration_note": "Nominal FPR is not a guaranteed target-domain FPR. Quantiles and ties are discrete.",
              "uncertainty_note": ("Bayesian network: 50 sampled weight sets; score is posterior mean squared distance, uncertainty is its variance, not a calibrated error probability"
                                   if profile == "bayesian" else
                                   "Predictive scores only; no Monte Carlo uncertainty claim" if posterior_draws == 0
                                   else "Posterior distance dispersion is not a calibrated error probability"),
              "official_provenance": plan["official_provenance"], "heads": {}}
    false_frames = []
    for name in (*names, "classifier"):
        scores, extras = ((probs, {}) if name == "classifier" else
                          score_head(name, features, heads[name]))
        scored = rows.copy()
        scored["score"] = scores
        for key, values in extras.items():
            scored[key] = values
        scored.to_csv(out / (name + "_scores.csv"), index=False)
        points = {}
        for alpha in plan["target_fprs"]:
            threshold = thresholds[name][str(alpha)]
            result = detection_metrics(rows, scores, threshold, alpha)
            result["per_dataset"] = {}
            for dataset, subset in rows.groupby("dataset"):
                values = scores[subset.index.to_numpy()]
                metrics = detection_metrics(subset, values, threshold, alpha)
                metrics["per_fake_method"] = {}
                for method in sorted(subset.loc[subset["label"].eq(1), "method"].unique()):
                    keep = subset["label"].eq(0) | subset["method"].eq(method)
                    metrics["per_fake_method"][method] = detection_metrics(subset.loc[keep], values[keep.to_numpy()], threshold, alpha)
                if name == "classifier":
                    metrics["fixed_probability_0.5"] = M.operating_point_metrics((subset["label"] == 0).astype(int), values, .5)
                result["per_dataset"][dataset] = metrics
            result["source_real_fpr"] = false_positive_groups(rows, scores, threshold, plan["identity_policy"], bootstrap_draws)
            # This operating point uses only held-out calibration videos, not dev/test ROC interpolation.
            result["threshold_source"] = "calibration role, genuine videos only"
            if "posterior_distance_variance" in extras:
                from evaluate_anomaly import uncertainty_summary
                result["uncertainty"] = uncertainty_summary(scores, rows["label"].to_numpy(), threshold,
                                                            extras["posterior_distance_variance"])
            points[str(alpha)] = result
            false = scored.loc[rows["label"].eq(0) & (scores >= threshold)].copy()
            false["head"], false["nominal_fpr"], false["threshold"] = name, alpha, threshold
            false_frames.append(false)
        if name == "classifier":
            points["fixed_probability_0.5"] = M.operating_point_metrics((rows["label"] == 0).astype(int), scores, .5)
            points["real_fake_probability_quantiles"] = dict(zip(
                ("min", "q25", "median", "q75", "max"), np.quantile(scores[rows["label"].eq(0)], [0, .25, .5, .75, 1]).tolist()))
        report["heads"][name] = points
    import pandas as pd
    false = pd.concat(false_frames, ignore_index=True)
    false.to_csv(out / "false_positives.csv", index=False)
    if phase == "dev":
        contact_sheets(out, false, scored_cache, examples)
    report["head_pipeline_seconds"] = time.perf_counter() - started
    report["cpu_diagnostic_seconds"] = report["head_pipeline_seconds"] if device == "cpu" else None
    report["cpu_diagnostic_seconds_note"] = "Wall time on the requested CPU process, excluding cache loading and GPU extraction"
    if phase == "dev":
        np.savez(out / "heads.npz", **{name + "/" + key: value for name, head in heads.items() for key, value in head.items()})
        report["head_parameters_sha256"] = sha256_file(out / "heads.npz")
    write_json(out / "report.json", report)
    if freeze_to:
        freeze_to = Path(freeze_to)
        freeze_to.mkdir(parents=True, exist_ok=False)
        import shutil
        shutil.copyfile(out / "heads.npz", freeze_to / "heads.npz")
        write_json(freeze_to / "method_lock.json", {"configuration": config, "thresholds": thresholds,
                   "head_training": head_training,
                   "calibration_observed_fpr": calibration_points,
                   "heads_sha256": sha256_file(freeze_to / "heads.npz"),
                   "development_report_sha256": sha256_file(out / "report.json")})
    if receipt:
        # Preserve the original claim even if reporting fails; no silent re-testing.
        record = json.loads(receipt.read_text(encoding="utf-8"))
        record.update(status="complete", report_sha256=sha256_file(out / "report.json"))
        receipt.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return report


def main(profile="baseline"):
    parser = argparse.ArgumentParser(description=(__doc__ if profile == "baseline" else
        "Primary Bayesian Deep SVDD head on frozen GenD features; independent early stopping and calibration."))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--lists", type=Path, required=True, help="Registered frame_lists.json")
    parser.add_argument("--development-cache", type=Path, required=True, help="Cache directory or extraction receipt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("dev", "test"), default="dev")
    parser.add_argument("--test-cache", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--freeze-to", type=Path, help="Freeze this baseline configuration before final testing")
    if profile == "baseline":
        parser.add_argument("--posterior-draws", type=int, default=0, help="Optional NIW uncertainty draws")
    else:
        parser.set_defaults(posterior_draws=0)
    parser.add_argument("--bootstrap-draws", type=int, default=500)
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="Bayesian head only; CUDA requires a GPU allocation, CPU trains from cached features")
    args = parser.parse_args()
    run(args.plan, args.lists, args.development_cache, args.out, args.phase, args.test_cache,
        args.lock, args.freeze_to, args.posterior_draws, args.bootstrap_draws, args.examples, profile, args.device)
    print(f"Wrote {args.phase} report: {args.out / 'report.json'}")


if __name__ == "__main__":
    main()
