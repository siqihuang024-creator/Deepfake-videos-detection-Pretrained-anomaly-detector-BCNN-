"""Register bounded candidate preprocessing on A plus a paired DFDC dev subset.

This records the data-preparation scope before decoding. It does not select
training vectors, fit heads, set score thresholds or open final-test videos.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gad.diagnostic_protocol import audit_diagnostic_table, canonical_dataset
from gad.face_detection import policy_spec
from gad.segment_pilot import (VERSION, PILOT_COVERAGE_RULE, make_pilot, parent_key, stable_hash)
from audit_video_durations import atomic_text, digest, fingerprint, write_csv

PURPOSE = "registered bounded candidate preprocessing; no training or final testing"


def select_originals(parents, dev_source_groups=100, seed=42):
    if dev_source_groups < 1 or int(dev_source_groups) != dev_source_groups:
        raise ValueError("Positive integer development source-group budget required")
    dev_source_groups = int(dev_source_groups)
    selected = []
    # One original per globally audited A source family. Multiple copies or
    # very long originals must not buy that source a larger candidate budget.
    fitting = parents.loc[parents["split"].isin(("train", "calibration"))]
    for _, group in fitting.groupby(["split", "source_clip"], sort=True):
        chosen = min(group.to_dict("records"), key=lambda row: stable_hash(f"{seed}|{row['rel_path']}"))
        selected.append(chosen)
    development = parents.loc[parents["split"].eq("dev") &
                              parents["dataset"].map(canonical_dataset).isin(("dfdc", "dfdckaggle"))]
    groups = {key: group for key, group in development.groupby("source_clip", sort=True)
              if set(group["label"]) == {0, 1}}
    if len(groups) < dev_source_groups:
        raise ValueError(f"Need {dev_source_groups} paired dev groups; only {len(groups)} available")
    chosen_groups = sorted(groups, key=lambda key: stable_hash(f"{seed}|paired-dev|{key}"))[:dev_source_groups]
    for key in chosen_groups:
        group = groups[key]
        for label in (0, 1):
            candidates = group.loc[group["label"].eq(label)].to_dict("records")
            selected.append(min(candidates, key=lambda row: stable_hash(f"{seed}|{row['rel_path']}")))
    return pd.DataFrame(selected).sort_values(["split", "dataset", "rel_path"]).reset_index(drop=True)


def prepare(manifest, duration_audit, reference_pilot, video_root, out,
            dev_source_groups=100, max_candidate_clips=3000, seed=42):
    manifest, duration_audit = Path(manifest), Path(duration_audit)
    reference_pilot, video_root, out = Path(reference_pilot), Path(video_root), Path(out)
    if out.exists():
        raise FileExistsError("Use a new registered data-preparation directory")
    if max_candidate_clips < 1 or int(max_candidate_clips) != max_candidate_clips:
        raise ValueError("Positive integer clip budget required")
    reference = json.loads(reference_pilot.read_text(encoding="utf-8"))
    if (reference.get("purpose") != "preprocessing timing and face-coverage pilot only"
            or reference.get("window_placement") != "interior_v1"
            or reference.get("pilot_coverage_rule") != PILOT_COVERAGE_RULE):
        raise ValueError("Supply the reviewed interior pilot registration")
    if digest(reference_pilot.parent / "pilot_videos.csv") != reference["pilot_table_sha256"]:
        raise ValueError("Reference pilot table changed")
    if digest(manifest) != reference["manifest_sha256"]:
        raise ValueError("Original role manifest differs from the reviewed pilot")
    sums = {}
    for line in (duration_audit / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        sha, name = line.split(None, 1)
        sums[name.strip()] = sha
    for name in ("durations.csv", "summary.json", "audit_contract.json", "errors.csv"):
        if sums.get(name) != digest(duration_audit / name):
            raise ValueError("Duration audit output changed: " + name)
    if (digest(duration_audit / "durations.csv") != reference["duration_table_sha256"]
            or digest(duration_audit / "audit_contract.json") != reference["duration_audit_contract_sha256"]):
        raise ValueError("Duration audit differs from the reviewed pilot")
    if json.loads((duration_audit / "summary.json").read_text(encoding="utf-8"))["errors"]:
        raise ValueError("Resolve duration audit errors first")
    audit_contract = json.loads((duration_audit / "audit_contract.json").read_text(encoding="utf-8"))
    if Path(audit_contract["video_root"]).resolve() != video_root.resolve():
        raise ValueError("Video root differs from duration audit")
    parents = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    parents["label"] = parents["label"].astype(int)
    for field in ("target_id", "donor_id", "driver_id", "language"):
        if field not in parents:
            parents[field] = ""
    if any(field in parents for field in ("sample_unit", "clip_id", "segment_start_seconds")):
        raise ValueError("Supply original whole-video roles, not an existing segment table")
    parents["video_key"] = [parent_key(row.dataset, row.rel_path, row.label) for row in parents.itertuples(index=False)]
    parents["frame_subset"] = "uniform32"
    role_audit = audit_diagnostic_table(parents, "source_only")
    if role_audit != reference["parent_role_audit"]:
        raise ValueError("Original source-role audit changed")
    originals = select_originals(parents, dev_source_groups, seed)
    with (duration_audit / "durations.csv").open(encoding="utf-8", newline="") as stream:
        durations = list(csv.DictReader(stream))
    candidates, excluded = make_pilot(originals.to_dict("records"), durations, seconds=10,
                                      clips_per_parent=4, groups_per_stratum=len(originals) + 1,
                                      minimum_seconds=2, seed=seed, placement="interior_v1")
    if excluded:
        raise ValueError("Selected originals have no eligible >=2s window; preserve and review before registration")
    if len(candidates) > max_candidate_clips:
        raise ValueError(f"Candidate budget exceeded: {len(candidates)} > {max_candidate_clips}")
    lookup = {(row["dataset"], row["rel_path"]): row for row in durations}
    for row in originals.to_dict("records"):
        _, stat = fingerprint(video_root, row)
        recorded = lookup[(row["dataset"], row["rel_path"])]
        if stat != {"bytes": int(recorded["bytes"]), "mtime_ns": int(recorded["mtime_ns"])}:
            raise ValueError("Selected video changed since the duration audit: " + row["rel_path"])
    table = pd.DataFrame(candidates)
    # The global A group budget also covers groups whose metadata spans datasets.
    a_rows = table.loc[table["split"].isin(("train", "calibration"))]
    if (a_rows.groupby("source_clip")["clip_id"].count() > 4).any():
        raise ValueError("A source family exceeds its global four-window budget")
    counts = []
    for (role, dataset, label), group in table.groupby(["split", "dataset", "label"], sort=True):
        counts.append({"split": role, "dataset": dataset, "label": int(label), "candidate_clips": len(group),
                       "original_videos": group["parent_video_key"].nunique(),
                       "source_groups": group["source_clip"].nunique(),
                       "selected_video_seconds": float((group.segment_end_seconds - group.segment_start_seconds).sum())})
    out.mkdir(parents=True)
    parents.to_csv(out / "parents.csv", index=False)
    originals.to_csv(out / "selected_originals.csv", index=False)
    parents.loc[parents["split"].eq("test")].to_csv(out / "reserved_test.csv", index=False)
    parents.loc[parents["split"].eq("dev") & ~parents["rel_path"].isin(originals["rel_path"])].to_csv(
        out / "unused_development.csv", index=False)
    write_csv(out / "candidate_videos.csv", candidates)
    files = {name: digest(out / name) for name in ("parents.csv", "selected_originals.csv", "reserved_test.csv",
                                                   "unused_development.csv", "candidate_videos.csv")}
    plan = {
        "version": VERSION, "purpose": PURPOSE, "table": "candidate_videos.csv",
        "table_sha256": files["candidate_videos.csv"], "registered_files": files,
        "manifest_sha256": digest(manifest), "reference_pilot_sha256": digest(reference_pilot),
        "duration_table_sha256": digest(duration_audit / "durations.csv"),
        "generator_code_sha256": digest(Path(__file__)), "parent_role_audit": role_audit,
        "source_grouping": "Existing globally audited source_clip; not proven person isolation",
        "seed": seed, "seconds": 10, "minimum_seconds": 2, "window_placement": "interior_v1",
        "a_scope": "All registered A source groups; one deterministic original and <=4 candidates per group",
        "dev_source_groups": dev_source_groups, "dev_videos": int(originals["split"].eq("dev").sum()),
        "dev_scope": "Paired DFDC dev subset: one real and one fake per selected source group; diagnostic only",
        "unused_dev_videos": int((parents["split"].eq("dev") & ~parents["rel_path"].isin(originals["rel_path"])).sum()),
        "reserved_test_videos": int(parents["split"].eq("test").sum()),
        "test_policy": "Metadata reserved; no test candidate decoding, visual review, features or scoring",
        "detection_policy": policy_spec("pad_on_empty_v1"), "pilot_coverage_rule": PILOT_COVERAGE_RULE,
        "max_candidate_clips": max_candidate_clips, "candidate_clips": len(table),
        "maximum_selected_images": len(table) * 32, "counts": counts,
        "next_stage": "Keep all preprocessing records, audit coverage, then finalize source-balanced training "
                      "and independent calibration/evaluation units. This is not a head-training registration.",
    }
    atomic_text(out / "segment_data_plan.json", json.dumps(plan, indent=2))
    print(json.dumps(plan, indent=2), flush=True)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--duration-audit", type=Path, required=True)
    parser.add_argument("--reference-pilot", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dev-source-groups", type=int, default=100)
    parser.add_argument("--max-candidate-clips", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.manifest, args.duration_audit, args.reference_pilot, args.video_root, args.out,
            args.dev_source_groups, args.max_candidate_clips, args.seed)


if __name__ == "__main__":
    main()
