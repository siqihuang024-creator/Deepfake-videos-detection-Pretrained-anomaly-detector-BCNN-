"""Prepare a GenD video table for real-only anomaly fitting.

Input CSV columns: dataset, rel_path, split, label (0 real, 1 visual fake),
target_id, donor_id (empty when absent), source_clip, method. Paths are
relative to a common video root. Supply audited identity and source IDs in
a common namespace across datasets. Train/val come from A and contain only
reals. Test comes from a different dataset B and contains both classes.
"""

import argparse
from hashlib import sha256
from pathlib import Path, PurePosixPath
import re
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gad import manifest as MF  # noqa: E402


def audit_table(videos):
    if not videos["split"].isin(("train", "val", "test")).all():
        raise ValueError("Expected train, val and test splits.")
    if videos["rel_path"].duplicated().any() or videos["video_key"].duplicated().any():
        raise ValueError("Paths and video keys must be unique.")
    if videos[["dataset", "rel_path", "target_id", "source_clip", "method"]].eq("").any().any():
        raise ValueError("Dataset, video path, face identity, source clip and method are required.")
    fit = videos[videos["split"].isin(("train", "val"))]
    test = videos[videos["split"] == "test"]
    if set(fit["split"]) != {"train", "val"} or set(fit["label"]) != {MF.GEND_REAL}:
        raise ValueError("Train and validation must both contain genuine videos only.")
    if set(test["label"]) != {MF.GEND_REAL, MF.GEND_FAKE}:
        raise ValueError("Test must contain both genuine and visually manipulated videos.")
    excluded = {"ffpp", "faceforensicspp", "celebdfv3", "celebdfpp"}
    def canonical_names(rows):
        return {re.sub(r"[^a-z0-9]", "", name.lower().replace("+", "p")) for name in rows["dataset"]}

    fit_names, test_names = canonical_names(fit), canonical_names(test)
    if fit_names & test_names or (fit_names | test_names) & excluded or "" in fit_names | test_names:
        raise ValueError("Use different external datasets for fit and test. Exclude FF++ and CelebDFv3.")
    roles = videos.melt(id_vars="split", value_vars=["target_id", "donor_id"], value_name="identity")
    identities = roles[roles["identity"] != ""].groupby("identity")["split"].nunique()
    sources = videos.groupby("source_clip")["split"].nunique()
    if (identities > 1).any() or (sources > 1).any():
        raise ValueError("Face identity or source-clip IDs overlap across splits.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"Output directory already exists: {args.out}")
    columns = ["dataset", "rel_path", "split", "label", "target_id", "donor_id", "source_clip", "method"]
    videos = pd.read_csv(args.manifest, usecols=columns, dtype=str, keep_default_na=False)[columns]
    if not videos["label"].isin(("0", "1")).all():
        parser.error("Expected GenD labels 0=real, 1=fake.")
    videos["label"] = videos["label"].astype(int)
    for value in videos["rel_path"]:
        path = PurePosixPath(value)
        if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value or path.as_posix() != value:
            parser.error(f"Not a normalized path below the common video root: {value}")
        if not (args.video_root / path).is_file():
            parser.error(f"Video file missing: {args.video_root / path}")
    videos["gend_source"] = videos["label"].map({MF.GEND_REAL: "real_external", MF.GEND_FAKE: "fake_external"})
    videos["video_key"] = "external/" + videos["gend_source"] + "/" + videos["rel_path"].map(
        lambda path: sha256(path.encode("utf-8")).hexdigest()
    )
    videos["frame_subset"] = videos["split"].map(MF.frame_subset_for_split)
    audit_table(videos)
    args.out.mkdir(parents=True)
    videos.sort_values(["split", "rel_path"]).to_csv(args.out / "videos.csv", index=False)
    MF.write_json(args.out / "summary.json", {
        "input_manifest_sha256": MF.sha256(args.manifest),
        "overlap_check": "assigned face IDs and supplied source IDs",
        "videos": len(videos),
    })
    print("wrote", args.out / "videos.csv")


if __name__ == "__main__":
    main()
