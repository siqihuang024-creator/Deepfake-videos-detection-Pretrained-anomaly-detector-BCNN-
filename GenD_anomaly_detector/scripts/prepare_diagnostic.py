"""Register audited roles before looking at external development/test results.

The manifest must assign roles explicitly. This tool never treats DFDC shard
numbers as identities or invents actor IDs. No dataset is downloaded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gad.diagnostic_protocol import (VERSION, FRAME_SUBSET, EXPERIMENT_HEADS, HEAD_VALIDATION,
                                     audit_diagnostic_table, sha256_file)


def prepare(manifest, video_root, selection_path, out, identity_policy, audit_note):
    manifest, video_root, out = Path(manifest), Path(video_root), Path(out)
    if out.exists():
        raise FileExistsError(out)
    if not audit_note.strip():
        raise ValueError("Describe where source/identity assignments were audited")
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    if (selection.get("step") != 2 or selection.get("protocol") != "celeb_identity"
            or not selection.get("ablation_run", "").startswith("ln_b128_s")
            or not selection.get("checkpoint_sha256") or not selection.get("ablation_plan_sha256")):
        raise ValueError("Run identity_ablation.py --action select first; supply its LN-b128 matrix selection")
    videos = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    required = {"split", "dataset", "rel_path", "label", "source_clip", "method"}
    if not required.issubset(videos):
        raise ValueError(f"Manifest missing: {sorted(required - set(videos))}")
    if not videos["label"].isin(("0", "1")).all():
        raise ValueError("Expected 0=visual real, 1=visual fake")
    videos["label"] = videos["label"].astype(int)
    for name in ("target_id", "donor_id", "driver_id", "language"):
        if name not in videos:
            videos[name] = ""
    videos["gend_source"] = videos["label"].map({0: "real_external", 1: "fake_external"})
    videos["video_key"] = ["external/" + source + "/" + hashlib.sha256(
        (dataset + "|" + path).encode()).hexdigest()
        for source, dataset, path in zip(videos["gend_source"], videos["dataset"], videos["rel_path"])]
    videos["frame_subset"] = FRAME_SUBSET
    audit = audit_diagnostic_table(videos, identity_policy)
    missing = [p for p in videos["rel_path"] if not (video_root / p).is_file()]
    if missing:
        raise ValueError(f"Video files missing; first entries: {missing[:5]}")
    out.mkdir(parents=True)
    (out / "tables").mkdir()
    videos.sort_values(["split", "dataset", "rel_path"]).to_csv(out / "tables/videos.csv", index=False)
    plan = {
        "version": VERSION, "table": "tables/videos.csv",
        "table_sha256": sha256_file(out / "tables/videos.csv"),
        "manifest_sha256": sha256_file(manifest),
        "identity_policy": identity_policy, "audit": audit, "audit_note": audit_note,
        "frame_sampling": "uniform at most 32 detected-face frames in every role; no image augmentation",
        "fitting_sources": ["HDTF", "TalkingHead-1KH"],
        "finetuned_selection": selection,
        "main_experiment": "bayesian", "parallel_experiment": "baseline",
        "experiments": EXPERIMENT_HEADS, "head_validation": HEAD_VALIDATION,
        "target_fprs": [.05, .01],
        "calibration": "held-out genuine calibration role, never the fitting or development videos",
        "test_policy": "lock configuration before final test; do not use test failures to design methods",
        "official_provenance": "GenD development used CDFv3/DeepSpeak; no fully-unseen-source claim",
        "deferred_datasets": ["DeepSpeak", "CoCoVideo-26K"],
        "checkpoint": "2026-12-03 diagnostic review, not a promised positive result",
    }
    (out / "diagnostic_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--identity-policy", choices=("provided_ids", "source_only"), required=True)
    parser.add_argument("--audit-note", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.video_root, args.selection, args.out,
                             args.identity_policy, args.audit_note), indent=2))


if __name__ == "__main__":
    main()
