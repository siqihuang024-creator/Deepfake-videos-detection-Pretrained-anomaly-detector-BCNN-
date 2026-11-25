"""Explicit fitting, calibration, development and withheld-test roles."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np
import pandas as pd

ROLES = ("train", "calibration", "dev", "test")
VERSION = 3
FRAME_SUBSET = "uniform32"
BASELINE_HEADS = ("mahalanobis", "bayesian_gaussian")
EXPERIMENT_HEADS = {"bayesian": ["bayesian_deep_svdd"], "baseline": list(BASELINE_HEADS)}
HEAD_VALIDATION = {"fraction_of_training_groups": .2, "seed": 42,
                   "grouping": "connected original-source families and all supplied identities"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_dataset(value):
    name = re.sub("[^a-z0-9]", "", value.lower().replace("+", "p"))
    return {"cdfv3": "celebdfv3", "cdfpp": "celebdfv3", "celebdfpp": "celebdfv3",
            "faceforensicspp": "ffpp"}.get(name, name)


def audit_diagnostic_table(videos, identity_policy):
    required = {"split", "dataset", "rel_path", "label", "target_id", "donor_id",
                "source_clip", "method", "video_key", "frame_subset"}
    if not required.issubset(videos):
        raise ValueError(f"Missing fields: {sorted(required - set(videos))}")
    if identity_policy not in ("provided_ids", "source_only"):
        raise ValueError("Choose provided_ids or explicitly acknowledge source_only separation")
    if not videos["split"].isin(ROLES).all() or set(videos["split"]) != set(ROLES):
        raise ValueError("Register nonempty train, calibration, dev and test roles")
    if not videos["label"].isin((0, 1)).all():
        raise ValueError("Labels must be 0=visually real and 1=visual fake")
    if not videos["frame_subset"].eq(FRAME_SUBSET).all():
        raise ValueError("All diagnostic roles must use the same uniform32 sampling rule")
    if videos[["dataset", "rel_path", "source_clip", "method", "video_key"]].eq("").any().any():
        raise ValueError("Dataset, path, globally audited source group, method and key are required")
    if videos["video_key"].duplicated().any() or videos["rel_path"].duplicated().any():
        raise ValueError("Video keys and paths must be unique")
    for path in videos["rel_path"]:
        p = PurePosixPath(path)
        if p.is_absolute() or ".." in p.parts or "\\" in path or ":" in path or p.as_posix() != path:
            raise ValueError(f"Not a normalized relative video path: {path}")
    for role in ROLES:
        labels = set(videos.loc[videos["split"].eq(role), "label"])
        if labels != ({0} if role in ("train", "calibration") else {0, 1}):
            raise ValueError(f"Unexpected labels in {role}")
    if (videos.groupby("source_clip")["split"].nunique() > 1).any():
        raise ValueError("An original source family crosses roles; keep originals and all derived fakes together")
    identity_fields = [k for k in ("target_id", "donor_id", "driver_id") if k in videos]
    if identity_policy == "provided_ids" and videos["target_id"].eq("").any():
        raise ValueError("provided_ids requires an audited target identity for every video")
    known = videos.melt(id_vars="split", value_vars=identity_fields, value_name="identity")
    known = known[known["identity"].ne("")]
    if (known.groupby("identity")["split"].nunique() > 1).any():
        raise ValueError("A supplied target/donor/driver identity crosses roles")
    names = videos["dataset"].map(canonical_dataset)
    deferred = names.isin(("deepspeakv1", "deepspeakv11", "deepspeakv2", "cocovideo26k"))
    if deferred.any():
        raise ValueError("DeepSpeak and CoCoVideo are deferred in this protocol; audit provenance before a new registration")
    fitting = videos["split"].isin(("train", "calibration"))
    if not names[fitting].isin(("hdtf", "talkinghead1kh")).all():
        raise ValueError("This initial protocol fits/calibrates only on HDTF and TalkingHead-1KH")
    forbidden_final = names.eq("celebdfv3") | names.eq("celebdfpp") | names.eq("ffpp") | names.eq("faceforensicspp")
    if (forbidden_final & videos["split"].eq("test")).any():
        raise ValueError("CelebDF/FF++ are not the held-out external B test in this protocol")
    return {
        "identity_policy": identity_policy,
        "identity_claim": ("no overlap among supplied IDs; not an automatic face-identity verification"
                           if identity_policy == "provided_ids" else
                           "source families isolated; person independence NOT established"),
        "source_family_overlap": 0,
        "counts": {role: {"videos": int(videos["split"].eq(role).sum()),
                         "real": int((videos["split"].eq(role) & videos["label"].eq(0)).sum()),
                         "fake": int((videos["split"].eq(role) & videos["label"].eq(1)).sum())}
                   for role in ROLES},
    }


def read_registered_plan(path):
    path = Path(path)
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("version") != VERSION:
        raise ValueError("Registered experiment protocol changed; register a new plan and preprocessing directory, without overwriting old results")
    if plan.get("experiments") != EXPERIMENT_HEADS or plan.get("head_validation") != HEAD_VALIDATION:
        raise ValueError("Registered Bayesian/parallel experiment recipe changed")
    table_path = path.parent / plan["table"]
    if sha256_file(table_path) != plan["table_sha256"]:
        raise ValueError("Registered diagnostic table changed")
    videos = pd.read_csv(table_path, dtype=str, keep_default_na=False)
    videos["label"] = videos["label"].astype(int)
    actual = audit_diagnostic_table(videos, plan["identity_policy"])
    if actual != plan["audit"]:
        raise ValueError("Registered split audit changed")
    selection = plan["finetuned_selection"]
    if selection.get("ablation_run", "").split("_s")[0] != "ln_b128":
        raise ValueError("Step 4 must use the registered LN-b128 validation winner")
    return plan, videos


def split_head_training(rows):
    """Deterministic early-stop holdout INSIDE A train; never read calibration/dev.

    Use connected components, so known actors spanning multiple originals do
    not split between gradient fitting and early stopping either.
    """
    if (rows.empty or set(rows["split"]) != {"train"} or set(rows["label"]) != {0}
            or rows["video_key"].duplicated().any()):
        raise ValueError("Head early-stop split requires distinct genuine training-role videos")
    rows = rows.reset_index(drop=True)
    parent = list(range(len(rows)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owners = {}
    for index, row in rows.iterrows():
        identifiers = [("source", row["source_clip"])]
        identifiers += [("person", row[field]) for field in ("target_id", "donor_id", "driver_id")
                        if field in rows and row[field]]
        for identifier in identifiers:
            if identifier in owners:
                a, b = root(index), root(owners[identifier])
                parent[max(a, b)] = min(a, b)
            else:
                owners[identifier] = index
    members = {}
    for index in range(len(rows)):
        members.setdefault(root(index), []).append(index)
    # Stable under manifest row reordering.
    components = sorted(members.values(), key=lambda indices: tuple(sorted(rows.loc[indices, "video_key"])))
    if len(components) < 2:
        raise ValueError("Need at least two independent training groups for Bayesian early stopping")
    count = min(len(components) - 1, max(1, int(np.ceil(len(components) * HEAD_VALIDATION["fraction_of_training_groups"]))))
    rng = np.random.default_rng(HEAD_VALIDATION["seed"])
    selected = rng.permutation(len(components))[:count]
    validation = np.zeros(len(rows), dtype=bool)
    for index in selected:
        validation[components[index]] = True
    fitting = ~validation
    if fitting.sum() < 2:
        raise ValueError("Bayesian gradient fitting requires at least two genuine videos after group holdout")
    return fitting, validation, {
        **HEAD_VALIDATION, "training_groups": len(components) - count, "validation_groups": count,
        "gradient_fit_video_keys": sorted(rows.loc[fitting, "video_key"]),
        "early_stop_video_keys": sorted(rows.loc[validation, "video_key"]),
        "note": "Both subsets come only from train; source-only metadata do not establish unknown-person separation"}


def bootstrap_rate(values, clusters, draws=500, seed=42):
    values, clusters = np.asarray(values, float), np.asarray(clusters)
    unique = np.unique(clusters)
    if len(unique) < 2 or draws < 2:
        return {"low": None, "high": None, "clusters": len(unique), "draws": 0}
    members = [np.flatnonzero(clusters == key) for key in unique]
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(draws):
        rows = np.concatenate([members[i] for i in rng.integers(0, len(unique), len(unique))])
        scores.append(float(values[rows].mean()))
    return {"low": float(np.quantile(scores, .025)), "high": float(np.quantile(scores, .975)),
            "clusters": len(unique), "draws": draws}


def false_positive_groups(videos, scores, threshold, identity_policy, draws=500):
    real = videos.loc[videos["label"].eq(0)].copy()
    real["flag"] = np.asarray(scores)[videos["label"].to_numpy() == 0] >= threshold
    groups = []
    columns = ["dataset"] + (["language"] if "language" in real and real["language"].ne("").any() else [])
    cluster_column = "target_id" if identity_policy == "provided_ids" else "source_clip"
    for name, rows in real.groupby(columns[0]):
        for language, subset in ([("", rows)] + [(k or "unknown", v) for k, v in rows.groupby("language")]
                                  if len(columns) == 2 else [("", rows)]):
            interval = bootstrap_rate(subset["flag"], subset[cluster_column], draws)
            groups.append({"dataset": name, "language": language or None,
                           "real_videos": len(subset), "false_positives": int(subset["flag"].sum()),
                           "observed_fpr": float(subset["flag"].mean()),
                           "interval_95": interval, "cluster_unit": cluster_column,
                           "interval_limitation": (
                               "Source-family bootstrap does not account for actors appearing in multiple originals; intervals may be too narrow. No person-independent claim."
                               if identity_policy == "source_only" else
                               "Conditional on supplied target identities being complete and correct; no automatic identity verification.")})
    return groups
