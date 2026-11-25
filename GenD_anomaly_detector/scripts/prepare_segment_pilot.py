"""Create a separate, bounded face-preprocessing pilot from audited video durations.

This is NOT a formal anomaly-head registration. Test files are never included.
The candidate window length and caps are explicit and recorded before decoding.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gad.diagnostic_protocol import audit_diagnostic_table
from gad.segment_pilot import VERSION, make_pilot, parent_key
from audit_video_durations import atomic_text, digest, write_csv


def prepare(manifest, duration_audit, out, seconds=10, clips_per_parent=2,
            groups_per_stratum=2, minimum_seconds=2, seed=42):
    manifest, duration_audit, out = Path(manifest), Path(duration_audit), Path(out)
    if out.exists():
        raise FileExistsError("Preserve the old draft and use a new pilot directory")
    sums = {}
    for line in (duration_audit / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        sha, name = line.split(None, 1)
        sums[name.strip()] = sha
    for name in ("durations.csv", "summary.json", "audit_contract.json", "errors.csv"):
        if sums.get(name) != digest(duration_audit / name):
            raise ValueError("Duration audit output changed: " + name)
    summary = json.loads((duration_audit / "summary.json").read_text(encoding="utf-8"))
    if summary["errors"]:
        raise ValueError("Resolve duration audit errors before making a pilot")
    parents = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    if "sample_unit" in parents or "clip_id" in parents:
        raise ValueError("Supply the original whole-video manifest, not an existing segment manifest")
    parents["label"] = parents["label"].astype(int)
    for name in ("target_id", "donor_id", "driver_id", "language"):
        if name not in parents:
            parents[name] = ""
    parents["video_key"] = [parent_key(row.dataset, row.rel_path, row.label)
                            for row in parents.itertuples(index=False)]
    parents["frame_subset"] = "uniform32"
    audit = audit_diagnostic_table(parents, "source_only")
    with (duration_audit / "durations.csv").open(encoding="utf-8", newline="") as stream:
        durations = list(csv.DictReader(stream))
    rows, excluded = make_pilot(parents.to_dict("records"), durations, seconds, clips_per_parent,
                                groups_per_stratum, minimum_seconds, seed)
    out.mkdir(parents=True)
    write_csv(out / "pilot_videos.csv", rows)
    write_csv(out / "excluded_short_originals.csv", excluded)
    counts = []
    table = pd.DataFrame(rows)
    for (role, dataset, label), subset in table.groupby(["split", "dataset", "label"]):
        counts.append({"split": role, "dataset": dataset, "label": int(label), "clips": len(subset),
                       "original_videos": subset["parent_video_key"].nunique(),
                       "source_groups": subset["source_clip"].nunique(),
                       "selected_seconds": float((subset.segment_end_seconds - subset.segment_start_seconds).sum())})
    plan = {"version": VERSION, "purpose": "preprocessing timing and face-coverage pilot only",
            "manifest_sha256": digest(manifest), "parent_role_audit": audit,
            "duration_audit_contract_sha256": digest(duration_audit / "audit_contract.json"),
            "duration_table_sha256": digest(duration_audit / "durations.csv"),
            "pilot_table_sha256": digest(out / "pilot_videos.csv"),
            "nominal_seconds": seconds, "minimum_seconds": minimum_seconds,
            "max_clips_per_original": clips_per_parent, "max_groups_per_role_dataset_label": groups_per_stratum,
            "originals_per_group_per_stratum": 1, "seed": seed,
            "sampling": "source-group hash selection, one original per chosen group, timeline-spread nonoverlapping windows",
            "time_interpretation": "seconds from audited stream start, half-open bounds; actual PTS used during decode",
            "test_policy": "test originals reserved; no test pixels decoded or scores produced",
            "tube_policy": "official TalkingHead frame intervals are not applied without alignment verification",
            "counts": counts,
            "limitation": "Clips from one original/source are correlated. This pilot cannot be passed to "
                          "prepare_diagnostic or used to train/calibrate/report a final detector."}
    atomic_text(out / "segment_pilot.json", json.dumps(plan, indent=2))
    print(json.dumps(plan, indent=2))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--duration-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--clips-per-original", type=int, default=2)
    parser.add_argument("--groups-per-stratum", type=int, default=2)
    parser.add_argument("--minimum-seconds", type=float, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.manifest, args.duration_audit, args.out, args.seconds, args.clips_per_original,
            args.groups_per_stratum, args.minimum_seconds, args.seed)


if __name__ == "__main__":
    main()
