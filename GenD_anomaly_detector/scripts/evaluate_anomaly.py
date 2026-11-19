"""Fit anomaly heads on external genuine training features and evaluate on B.

Pass the same audited video table and frame lists for the official and
fine-tuned encoder runs. Validation is genuine-only and supplies the
threshold. Test labels enter metrics only. See build_anomaly_table.py.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build_anomaly_table import audit_table  # noqa: E402
from gad import anomaly, bayesian_svdd, manifest as MF, scoring  # noqa: E402


def load_split(cache, videos):
    with np.load(cache, allow_pickle=False) as stored:
        keys = stored["video_keys"]
        values = stored["features"]
        frame_counts = stored["num_frames"]
        metadata = json.loads(stored["metadata"].item())
    if (keys.ndim != 1 or values.ndim != 2 or len(values) != len(keys) or not len(keys)
            or not np.isfinite(values).all() or np.shape(frame_counts) != keys.shape
            or np.any(frame_counts <= 0) or len(set(keys)) != len(keys)):
        raise ValueError(f"Invalid features in {cache}")
    indexed = videos.set_index("video_key")
    unexpected = set(keys) - set(indexed.index)
    if unexpected:
        raise ValueError(f"Features from outside the requested split in {cache}: {sorted(unexpected)[:3]}")
    scored = indexed.loc[keys].reset_index()
    scored["num_frames_scored"] = frame_counts
    missing = indexed.loc[~indexed.index.isin(keys), "rel_path"].tolist()
    if set(scored["label"]) != set(videos["label"]):
        raise ValueError(f"No scorable videos for a required class in {cache}")
    return values, scored, metadata, missing


def uncertainty_summary(scores, labels, threshold, uncertainty):
    errors = (scores >= threshold) != (labels == MF.GEND_FAKE)
    correct, incorrect = uncertainty[~errors], uncertainty[errors]
    varies = len(correct) and len(incorrect) and np.ptp(uncertainty) > 0
    return {
        "spearman_with_error": float(pd.Series(uncertainty).rank().corr(pd.Series(errors.astype(int))))
        if varies else None,
        "correct_count": len(correct),
        "error_count": len(incorrect),
        "mean_uncertainty_correct": float(correct.mean()) if len(correct) else None,
        "mean_uncertainty_error": float(incorrect.mean()) if len(incorrect) else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--val-features", type=Path, required=True)
    parser.add_argument("--test-features", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--out", type=Path, required=True, help="New results directory")
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"Output directory already exists: {args.out}")
    videos = pd.read_csv(args.videos, keep_default_na=False, dtype={"target_id": str, "donor_id": str, "source_clip": str})
    audit_table(videos)
    split_names = ("train", "val", "test")
    cache_paths = (args.train_features, args.val_features, args.test_features)
    data = {split: load_split(cache, videos[videos["split"] == split])
            for split, cache in zip(split_names, cache_paths)}
    model_fields = ("model", "model_repo", "checkpoint_sha256", "precision", "feature_type", "gend_commit")
    signature = {tuple(metadata.get(field) for field in model_fields) for _, _, metadata, _ in data.values()}
    if len(signature) != 1 or next(iter(signature))[4] != "mean_of_frame_l2_embeddings":
        raise SystemExit("The feature caches use different extractors or settings.")
    if any(item[0].shape[1] != data["train"][0].shape[1] for item in data.values()):
        raise SystemExit("Feature dimensions differ across splits.")

    training_features, _, _, _ = data["train"]
    val_features, val, _, _ = data["val"]
    test_features, test, _, _ = data["test"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heads = anomaly.fit_statistical_heads(training_features, val_features)
    heads["bayesian_deep_svdd"] = bayesian_svdd.fit(training_features, val_features, device)
    gaussian = heads["bayesian_gaussian"]
    mixture = heads["bayesian_gaussian_mixture"]
    svdd = heads["bayesian_deep_svdd"]
    svdd_diagnostics = {
        key: svdd[key] for key in (
            "train_score_mean", "train_score_variance", "train_score_relative_std",
            "near_constant_train_scores", "initial_train_embedding_variance", "train_embedding_variance",
            "train_embedding_variance_ratio", "low_train_embedding_variance",
        )
    }
    print(f"Bayesian Deep SVDD training score variance={svdd['train_score_variance']:.6g}, "
          f"embedding variance ratio={svdd['train_embedding_variance_ratio']:.6g}")
    if svdd["near_constant_train_scores"] or svdd["low_train_embedding_variance"]:
        print("Warning: Bayesian Deep SVDD training scores or embeddings show low dispersion. "
              "See training_diagnostics in report.json.", file=sys.stderr)
    report = {
        "threshold_source": "genuine A validation videos at the 95th percentile",
        "video_table_sha256": MF.sha256(args.videos),
        "head_settings": {
            "seed": anomaly.SEED,
            "mahalanobis": {"covariance": "LedoitWolf", "shrinkage": heads["mahalanobis"]["shrinkage"]},
            "knn_cosine": {"neighbors": anomaly.KNN_NEIGHBORS},
            "bayesian_gaussian": {"kappa0": anomaly.BAYESIAN_PRIOR_MEAN_WEIGHT, "nu0": gaussian["nu0"],
                                  "predictive_df": gaussian["nu_n"] - training_features.shape[1] + 1,
                                  "posterior_draws": anomaly.NIW_DRAWS},
            "bayesian_gaussian_mixture": {"pca_dim": len(mixture["pca_components"]),
                                          "components_max": mixture["n_components"],
                                          "covariance_type": mixture["covariance_type"],
                                          "validation_nll_diag": mixture["validation_nll_diag"],
                                          "validation_nll_full": mixture["validation_nll_full"],
                                          "converged": bool(mixture["converged"])},
            "bayesian_deep_svdd": {"widths": list(bayesian_svdd.WIDTHS), "device": str(device),
                                   "center_epsilon": bayesian_svdd.CENTER_EPS, "kl_weight": bayesian_svdd.KL_WEIGHT,
                                   "input_preprocessing": "subtract genuine training feature mean",
                                   "epochs_run": svdd["epochs_run"], "best_epoch": svdd["best_epoch"],
                                   "patience": bayesian_svdd.PATIENCE, "max_epochs": bayesian_svdd.MAX_EPOCHS,
                                   "learning_rate": bayesian_svdd.LEARNING_RATE,
                                   "batch_size": bayesian_svdd.BATCH_SIZE,
                                   "validation_draws": bayesian_svdd.VALIDATION_DRAWS,
                                   "posterior_draws": bayesian_svdd.POSTERIOR_DRAWS,
                                   "training_diagnostics": {
                                       **svdd_diagnostics,
                                       "min_score_relative_std": bayesian_svdd.MIN_TRAIN_SCORE_RELATIVE_STD,
                                       "min_embedding_variance_ratio": bayesian_svdd.MIN_TRAIN_EMBEDDING_VARIANCE_RATIO,
                                   }},
        },
        "feature_caches": {split: {"metadata": data[split][2], "sha256": MF.sha256(cache)}
                           for split, cache in zip(split_names, cache_paths)},
        "coverage": {split: {"scored": len(rows), "listed": len(rows) + len(missing), "unscorable": missing}
                     for split, (_, rows, _, missing) in data.items()},
        "results": {},
    }
    predictions = test.copy()
    validation_predictions = val.copy()
    eval_features = np.concatenate([val_features, test_features])
    cut = len(val_features)
    labels = test["label"].to_numpy()
    for name, head in heads.items():
        if name == "bayesian_deep_svdd":
            scores, extras = bayesian_svdd.score(eval_features, head, device)
        else:
            scores, extras = anomaly.score_statistical_head(name, eval_features, head)
        val_scores, test_scores = scores[:cut], scores[cut:]
        threshold = scoring.calibrate_on_validation(val.assign(score=val_scores))
        result = scoring.evaluate_split(test.assign(score=test_scores), threshold, args.bootstrap_draws,
                                        probability_scores=False)
        entry = {
            "threshold": threshold,
            "validation_observed_fpr": float((val_scores >= threshold).mean()),
            "test": result,
        }
        for field, values in extras.items():
            validation_predictions[f"{name}_{field}"] = values[:cut]
            predictions[f"{name}_{field}"] = values[cut:]
            if field in ("posterior_distance_std", "posterior_distance_variance"):
                entry["uncertainty"] = {"measure": field,
                                        **uncertainty_summary(test_scores, labels, threshold, values[cut:])}
        report["results"][name] = entry
        predictions[name + "_score"] = test_scores
        validation_predictions[name + "_score"] = val_scores
        print(f"{name}: AUROC={result['auroc']:.4f}, test FPR={result['observed_false_positive_rate']:.4f}, "
              f"test TPR={result['observed_true_positive_rate']:.4f}")

    args.out.mkdir(parents=True)
    validation_predictions.to_csv(args.out / "val_video_scores.csv", index=False)
    predictions.to_csv(args.out / "test_video_scores.csv", index=False)
    np.savez_compressed(args.out / "heads.npz", **{
        f"{name}_{field}": value for name, head in heads.items() for field, value in head.items()
    })
    MF.write_json(args.out / "report.json", scoring.nan_to_none(report))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
