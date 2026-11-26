"""Source-balanced Bayesian head development on existing candidate caches.

This entry has no final-test mode. Every selection and training recipe is
written before fitting either head; development labels only enter reports.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .diagnostic_protocol import split_head_training, sha256_file, false_positive_groups
from .feature_cache import load_video_cache

RECIPE = {
    "version": 1, "seed": 42, "training_clip_cap_per_source": 2,
    "A_eligibility": "32 selected frames and registered sustained-face diagnostic flag",
    "selection": "ascending SHA256 of 42|segment-head-v1|video_key; never rank model scores",
    "calibration_unit": "one eligible 10-second window per calibration source, chosen by the same hash",
    "development_unit": "one registered window per video; keep low coverage if scorable",
    "unscorable_policy": "retain no-face and excluded sources in coverage denominators; no replacement across roles",
    "frame_aggregation": "mean of up to 32 frame L2 embeddings before fp16 storage quantization",
    "training_weights": "equal total per source; equal weight per retained clip within source",
    "early_stopping": "20 percent connected train groups, seed 42; no calibration or development labels",
    "kl_denominator": "number of gradient-fitting source groups; source-balanced surrogate, not independent clips",
    "widths": [512, 128], "bias": False, "max_epochs": 100, "patience": 12,
    "batch_size": 128, "learning_rate": .001, "kl_weight": .001,
    "center_epsilon": .1, "validation_draws": 8, "posterior_draws": 50,
    "target_fprs": [.05, .01], "threshold": "linear quantile of independent genuine calibration scores; score >= threshold",
    "device": "cpu", "final_test": "forbidden in this development entry",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    # JSON deliberately refuses NaN. Metrics with undefined values use null.
    from .scoring import nan_to_none
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(nan_to_none(value), stream, indent=2, allow_nan=False)


def choose_units(rows, scorable, low_coverage):
    rows = rows.copy()
    if (rows.empty or rows.video_key.duplicated().any()
            or not rows.split.isin(("train", "calibration", "dev")).all()
            or not rows.label.isin((0, 1)).all()
            or not rows.loc[rows.split.ne("dev"), "label"].eq(0).all()
            or rows.source_clip.eq("").any()
            or (rows.groupby("source_clip").split.nunique() > 1).any()):
        raise ValueError("Expected genuine source-isolated A and registered development candidates only")
    if not (rows.loc[rows.split.isin(("train", "calibration")), "dataset"]
            .isin(("HDTF", "TalkingHead-1KH")).all()):
        raise ValueError("Unexpected A dataset")
    # B candidate preparation registered exactly one window per dev video.
    dev = rows.loc[rows.split.eq("dev")]
    if dev.parent_video_key.duplicated().any():
        raise ValueError("Development must have one pre-registered window per video")
    if not set(scorable).issubset(set(rows.video_key)) or not set(low_coverage).issubset(set(scorable)):
        raise ValueError("Unknown cached or low-coverage candidate")
    rows["scorable"] = rows.video_key.isin(scorable)
    rows["sustained"] = rows.scorable & ~rows.video_key.isin(low_coverage)
    rows["rank"] = rows.video_key.map(lambda key: hashlib.sha256(
        ("42|segment-head-v1|" + key).encode()).hexdigest())
    chosen = []
    for role, cap in (("train", 2), ("calibration", 1)):
        eligible = rows.loc[rows.split.eq(role) & rows.sustained].sort_values(["rank", "video_key"])
        chosen.extend(eligible.groupby("source_clip", sort=True).head(cap).video_key)
    chosen.extend(dev.loc[dev.video_key.isin(scorable), "video_key"])
    units = rows.loc[rows.video_key.isin(chosen)].sort_values(["split", "source_clip", "rank"]).reset_index(drop=True)
    if any(not units.split.eq(role).any() for role in ("train", "calibration", "dev")):
        raise ValueError("No eligible units for a required role")
    if set(units.loc[units.split.eq("dev"), "label"]) != {0, 1}:
        raise ValueError("Development requires scorable real and fake videos")
    train = units.loc[units.split.eq("train")].reset_index(drop=True)
    fit, early, split_report = split_head_training(train)
    units["head_role"] = units["split"]
    units.loc[units.video_key.isin(train.loc[fit, "video_key"]), "head_role"] = "gradient_fit"
    units.loc[units.video_key.isin(train.loc[early, "video_key"]), "head_role"] = "early_stop"
    coverage = []
    for (role, dataset, label), group in rows.groupby(["split", "dataset", "label"]):
        used = units.loc[units.video_key.isin(group.video_key)]
        coverage.append({"split": role, "dataset": dataset, "label": int(label),
                         "candidate_windows": len(group), "original_videos": group.parent_video_key.nunique(),
                         "source_groups": group.source_clip.nunique(), "scorable_windows": int(group.scorable.sum()),
                         "sustained_windows": int(group.sustained.sum()), "retained_windows": len(used),
                         "retained_sources": used.source_clip.nunique(),
                         "sources_without_sustained_window": group.source_clip.nunique() - group.loc[group.sustained, "source_clip"].nunique()})
    return units, coverage, split_report


def load_inputs(data, cache_run, selection_path):
    data, cache_run, selection_path = Path(data).resolve(), Path(cache_run).resolve(), Path(selection_path).resolve()
    plan_path = data / "segment_data_plan.json"
    plan, lists = read_json(plan_path), read_json(cache_run / "frame_lists.json")
    plan_hash = sha256_file(plan_path)
    if (plan.get("purpose") != "registered bounded candidate preprocessing; no training or final testing"
            or plan.get("version") != 1 or plan.get("table") != "candidate_videos.csv"
            or lists["plan_sha256"] != plan_hash):
        raise ValueError("Not the completed bounded development candidate registration")
    for name in ("parents.csv", "selected_originals.csv", "reserved_test.csv", "unused_development.csv", "candidate_videos.csv"):
        if sha256_file(data / name) != plan["registered_files"][name]:
            raise ValueError("Changed registered candidate table: " + name)
    list_hash = sha256_file(cache_run / "candidates.txt")
    if list_hash != lists["files"]["candidates.txt"]:
        raise ValueError("Candidate frame list changed")
    if sha256_file(data / "faces/pilot_run_contract.json") != lists["preprocessing_contract_sha256"]:
        raise ValueError("Preprocessing contract changed")
    rows = pd.read_csv(data / plan["table"], dtype=str, keep_default_na=False)
    rows["label"] = rows.label.astype(int)
    if set(rows.video_key) != set(lists["records_sha256"]) or len(rows) != plan["candidate_clips"]:
        raise ValueError("Candidate/record coverage changed")
    low, scorable = set(), set()
    counts = {}
    for row in rows.itertuples(index=False):
        path = data / "faces/records" / (row.video_key + ".json")
        if sha256_file(path) != lists["records_sha256"][row.video_key]:
            raise ValueError("Preprocessing record changed")
        record = read_json(path)
        if record["status"] == "ok" and record["selected"]:
            scorable.add(row.video_key)
            counts[row.video_key] = len(record["selected"])
            if not record["sustained_face_candidate"]:
                low.add(row.video_key)
            elif len(record["selected"]) != 32:
                raise ValueError("Sustained flag requires 32 selected frames")
        elif record["status"] != "no_face" or record["selected"]:
            raise ValueError("Invalid or failed preprocessing record")
    if (low != set(lists["low_coverage_included_keys"])
            or set(rows.video_key) - scorable != set(lists["unscorable_no_face_keys"])):
        raise ValueError("Cache registration coverage changed")
    selection = read_json(selection_path)
    if (selection.get("ablation_run") != "ln_b128_s42" or selection.get("best_epoch") != 21
            or sha256_file(selection["checkpoint"]) != selection["checkpoint_sha256"]):
        raise ValueError("Use the already locked step-2 LN-b128 checkpoint")
    caches, fingerprints = {}, {}
    for name in ("official", "finetuned"):
        receipt_path = cache_run / "cache_receipts" / (name + ".json")
        receipt = read_json(receipt_path)
        path = Path(receipt["cache"])
        if sha256_file(path / "metadata.json") != receipt["metadata_sha256"]:
            raise ValueError("Cache metadata no longer matches receipt")
        values = load_video_cache(path)
        signature = values["metadata"]["signature"]
        if (signature.get("diagnostic_plan_sha256") != plan_hash
                or signature["frame_list_sha256"] != list_hash
                or values["metadata"]["dimension"] != 1024 or set(values["video_keys"]) != scorable
                or signature["precision"] != "bf16"
                or any(int(n) != counts[key] for key, n in zip(values["video_keys"], values["num_frames"]))):
            raise ValueError("Cache does not match the registered candidate features")
        if name == "official":
            if signature["model"] != "official":
                raise ValueError("Official cache contains a different encoder")
        elif (signature.get("checkpoint_sha256") != selection["checkpoint_sha256"]
              or Path(signature["model"]).resolve() != Path(selection["checkpoint"]).resolve()):
            raise ValueError("Fine-tuned cache uses a different checkpoint")
        caches[name] = values
        fingerprints[name] = {"cache": str(path), "metadata_sha256": receipt["metadata_sha256"],
                              "signature": signature}
    for field in ("frame_times_sha256", "crop_stat_sha256", "gend_commit", "cache_algorithm"):
        if fingerprints["official"]["signature"][field] != fingerprints["finetuned"]["signature"][field]:
            raise ValueError("Encoder inputs/feature recipes differ: " + field)
    units, coverage, split_report = choose_units(rows, scorable, low)
    provenance = {"data_plan_sha256": plan_hash, "candidate_list_registration_sha256": sha256_file(cache_run / "frame_lists.json"),
                  "selection_sha256": sha256_file(selection_path), "selection": selection,
                  "cache_fingerprints": fingerprints, "coverage": coverage, "head_split": split_report}
    return units, caches, provenance


def run(data, cache_run, selection, out):
    import torch
    import scipy
    import sklearn
    from . import bayesian_svdd as svdd, old_metrics as metrics
    out = Path(out)
    if out.exists():
        raise FileExistsError("Use a new head-training directory; never overwrite a run")
    constants = {"widths": list(svdd.WIDTHS), "max_epochs": svdd.MAX_EPOCHS, "patience": svdd.PATIENCE,
                 "batch_size": svdd.BATCH_SIZE, "learning_rate": svdd.LEARNING_RATE, "kl_weight": svdd.KL_WEIGHT,
                 "center_epsilon": svdd.CENTER_EPS, "validation_draws": svdd.VALIDATION_DRAWS,
                 "posterior_draws": svdd.POSTERIOR_DRAWS, "seed": svdd.SEED}
    if any(RECIPE[k] != v for k, v in constants.items()):
        raise ValueError("Bayesian training constants differ from the registered recipe")
    units, caches, provenance = load_inputs(data, cache_run, selection)
    out.mkdir(parents=True)
    units.to_csv(out / "training_units.csv", index=False)
    config = {"purpose": "paired source-balanced Bayesian development; no final-test evaluation",
              "recipe": RECIPE, **provenance, "training_units_sha256": sha256_file(out / "training_units.csv"),
              "algorithm_sha256": {p.name: sha256_file(p) for p in (Path(__file__), Path(svdd.__file__),
                  Path(__file__).with_name("diagnostic_protocol.py"), Path(metrics.__file__))},
              "libraries": {"numpy": np.__version__, "torch": torch.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
              "limitations": ["Source-only separation does not establish person independence.",
                              "Coverage-based A eligibility conditions the real training/calibration distribution; target FPR is not guaranteed.",
                              "Only 100 paired DFDC development families; development metrics are not final-test evidence.",
                              "Official GenD development used CDFv3/DeepSpeak; no fully unseen provenance claim.",
                              "Posterior distance variance is not a calibrated probability of error."]}
    # Commit BOTH encoders' exact units/recipe before fitting the first network.
    write_json(out / "training_registration.json", config)
    print("Registered units: " + str(units.groupby("head_role").size().to_dict()), flush=True)
    masks = {role: units.head_role.eq(role).to_numpy() for role in ("gradient_fit", "early_stop", "calibration", "dev")}
    summary = []
    for name in ("official", "finetuned"):
        print("\n=== Bayesian head: " + name + " ===", flush=True)
        folder = out / name
        folder.mkdir()
        cache = caches[name]
        lookup = {key: i for i, key in enumerate(cache["video_keys"])}
        indices = [lookup[key] for key in units.video_key]
        features, probs = cache["features"][indices], cache["fake_prob"][indices]

        def progress(event):
            with (folder / "training_progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event) + "\n")
            print("Epoch {epoch}/100 | loss={training_loss:.6g} | early={early_stop_distance:.6g} | best={best_epoch}".format(**event), flush=True)

        head = svdd.fit_source_balanced(features[masks["gradient_fit"]], features[masks["early_stop"]],
            units.loc[masks["gradient_fit"], "source_clip"].tolist(),
            units.loc[masks["early_stop"], "source_clip"].tolist(), "cpu", progress)
        np.savez(folder / "head.npz", **head)
        np.savetxt(folder / "training_history.csv", head["training_history"], delimiter=",",
                   header="epoch,training_loss,early_stop_distance,elapsed_seconds", comments="")
        cal_scores, _ = svdd.score(features[masks["calibration"]], head, "cpu")
        dev_scores, extras = svdd.score(features[masks["dev"]], head, "cpu")
        dev_rows = units.loc[masks["dev"]].reset_index(drop=True)
        thresholds, reports = {}, {}
        scored = dev_rows.copy()
        scored["posterior_distance_variance"] = extras["posterior_distance_variance"]
        scored["bayesian_deep_svdd_score"] = dev_scores
        scored["classifier_score"] = probs[masks["dev"]]
        scored.to_csv(folder / "development_scores.csv", index=False)
        cal_rows = units.loc[masks["calibration"]].copy()
        cal_rows["bayesian_deep_svdd_score"] = cal_scores
        cal_rows["classifier_score"] = probs[masks["calibration"]]
        cal_rows.to_csv(folder / "calibration_scores.csv", index=False)
        false_positives = []
        for head_name, cal_values, scores in (("bayesian_deep_svdd", cal_scores, dev_scores),
                                              ("classifier", probs[masks["calibration"]], probs[masks["dev"]])):
            thresholds[head_name], reports[head_name] = {}, {}
            for alpha in RECIPE["target_fprs"]:
                threshold = float(np.quantile(cal_values, 1 - alpha))
                thresholds[head_name][str(alpha)] = threshold
                point = metrics.detection_metrics((dev_rows.label == 0).astype(int), scores, threshold, alpha)
                point["roc_interpolated_tpr_at_nominal_fpr"] = point.pop("tpr_at_target_fpr")
                point["calibration_observed_fpr"] = float((cal_values >= threshold).mean())
                point["source_real_fpr"] = false_positive_groups(dev_rows, scores, threshold, "source_only")
                reports[head_name][str(alpha)] = point
                false = scored.loc[dev_rows.label.eq(0) & (scores >= threshold)].copy()
                false["head"], false["nominal_fpr"], false["threshold"] = head_name, alpha, threshold
                false_positives.append(false)
                summary.append({"encoder": name, "head": head_name, "nominal_fpr": alpha,
                                **{k: v for k, v in point.items() if np.isscalar(v) or v is None}})
        pd.concat(false_positives, ignore_index=True).to_csv(folder / "false_positives.csv", index=False)
        diagnostics = {key: np.asarray(head[key]).item() for key in (
            "best_epoch", "epochs_run", "training_seconds", "validation_distance", "train_score_relative_std",
            "near_constant_train_scores", "train_embedding_variance_ratio", "low_train_embedding_variance",
            "effective_train_groups", "early_stop_source_groups")}
        report = {"phase": "development", "encoder": name, "heads": reports, "head_training": diagnostics,
                  "calibration_units": int(masks["calibration"].sum()),
                  "calibration_fpr_resolution": 1 / int(masks["calibration"].sum()),
                  "development_units": len(dev_rows), "limitations": config["limitations"]}
        write_json(folder / "report.json", report)
        write_json(folder / "development_lock.json", {"registration_sha256": sha256_file(out / "training_registration.json"),
            "head_sha256": sha256_file(folder / "head.npz"), "thresholds": thresholds,
            "report_sha256": sha256_file(folder / "report.json"),
            "scope": "Snapshot for later locked evaluation; this script cannot run final tests"})
        print("Head completed: " + name + "; diagnostics=" + str(diagnostics), flush=True)
    pd.DataFrame(summary).to_csv(out / "comparison.csv", index=False)
    write_json(out / "completion.json", {"completed": True, "encoders": ["official", "finetuned"], "phase": "development"})
    print("Both Bayesian head development runs completed. Final test remains unopened.", flush=True)
