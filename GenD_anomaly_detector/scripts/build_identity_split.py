"""Build and audit the Celeb-DF++ face-identity-disjoint protocol.

Input is the full inventory (combined_manifest_p05.csv), with the old 1=real
labels. The existing split column is ignored. The output videos.csv uses the
same conventions as build_video_table.py (0=real), and can be passed directly
to preprocess_videos.py without a membership file.
"""

import argparse
import hashlib
import re
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gad import manifest as MF  # noqa: E402

SPLITS = ("train", "val", "test")
FACE_COUNTS = (41, 9, 9)
DATASET = "CelebDFv3"
TWO_FACE_FAMILIES = ("FaceSwap", "FaceReenact")
SUPPORTED_FAKE_FAMILIES = TWO_FACE_FAMILIES + ("TalkingFace",)
REQUIRED_COLUMNS = (
    "dataset", "path", "label", "method", "target_id", "donor_id", "source_clip", "source_id", "driver_id",
)


def load_inventory(path):
    original = pd.read_csv(path, usecols=REQUIRED_COLUMNS, dtype=str, keep_default_na=False)
    inventory = original.loc[original["dataset"] == DATASET].sort_values("path").reset_index(drop=True)
    if inventory.empty or inventory["path"].duplicated().any():
        raise ValueError("Expected nonempty, unique CelebDFv3 video paths.")
    if not inventory["label"].isin(["0", "1"]).all():
        raise ValueError("Expected old manifest labels 1=real, 0=fake.")
    inventory["label"] = inventory["label"].astype(int)
    return inventory


def split_identities(real_celebs, seed):
    ids = set(real_celebs["target_id"])
    if len(ids) != sum(FACE_COUNTS) or any(re.fullmatch(r"id\d+", identity) is None for identity in ids):
        raise ValueError("Expected 59 confirmed face identities in REAL/Real_WEB.")
    # Rank by a seeded hash to get the same assignment across Python versions
    # and manifest row orders. No performance results enter the selection.
    ordered = sorted(ids, key=lambda identity: hashlib.sha256(f"{seed}:{identity}".encode("ascii")).hexdigest())
    return dict(zip(ordered, ["train"] * FACE_COUNTS[0] + ["val"] * FACE_COUNTS[1] + ["test"] * FACE_COUNTS[2]))


def assign_videos(inventory, seed):
    real = inventory.loc[inventory["label"] == MF.OLD_REAL].copy()
    fake = inventory.loc[inventory["label"] == MF.OLD_FAKE].copy()
    real_celebs = real["path"].str.startswith("REAL/Real_WEB/")
    real_youtube = real["path"].str.startswith("REAL/Real_YouTube/")
    if not (real_celebs | real_youtube).all() or not fake["path"].str.startswith("FAKE/").all():
        raise ValueError("Expected REAL/Real_WEB or REAL/Real_YouTube for reals and FAKE for fakes.")

    family = fake["path"].str.split("/").str[1]
    if not family.isin(SUPPORTED_FAKE_FAMILIES).all():
        raise ValueError("Unknown forgery family in full inventory.")
    two_faces = family.isin(TWO_FACE_FAMILIES)
    assignments = split_identities(real.loc[real_celebs], seed)
    target_split = fake["target_id"].map(assignments)
    donor_split = fake["donor_id"].map(assignments)
    if target_split.isna().any() or donor_split[two_faces].isna().any():
        raise ValueError("Every fake needs a known target, and swap/reenactment fakes also need a known donor.")
    if fake.loc[~two_faces, "donor_id"].ne("").any():
        raise ValueError("Talking-face entries must identify the audio source via driver_id, not donor_id.")

    source_rows = real.loc[real_celebs]
    if source_rows["source_clip"].eq("").any() or source_rows["source_clip"].duplicated().any():
        raise ValueError("Expected a unique nonempty source_clip per Real_WEB video.")
    source_targets = source_rows.set_index("source_clip")["target_id"]
    if not fake["source_clip"].map(source_targets).eq(fake["target_id"]).all():
        raise ValueError("A fake has no matching genuine source clip for its target identity.")

    real["split"] = real["target_id"].map(assignments).where(real_celebs, "excluded")
    real["reason"] = ""
    real.loc[real_youtube, "reason"] = "unverified_youtube_face_identity"
    cross_split = two_faces & donor_split.ne(target_split)
    fake["split"] = target_split.where(~cross_split, "excluded")
    fake["reason"] = ""
    fake.loc[cross_split, "reason"] = "face_identities_in_different_splits"
    assigned = pd.concat([real, fake]).sort_values("path")
    kept = assigned.loc[assigned["split"] != "excluded"]
    excluded = assigned.loc[assigned["split"] == "excluded"]
    return kept, excluded, assignments


def audit_splits(kept):
    if set(kept["split"]) != set(SPLITS) or kept["target_id"].eq("").any() or kept["source_clip"].eq("").any():
        raise ValueError("Expected train, val and test with a target face and source clip for every row.")
    paths = kept["rel_path"] if "rel_path" in kept else kept["path"]
    if paths.duplicated().any():
        raise ValueError("A video path occurs more than once.")
    needs_donor = paths.str.startswith(("FAKE/FaceSwap/", "FAKE/FaceReenact/"))
    if kept.loc[needs_donor, "donor_id"].eq("").any():
        raise ValueError("Swap and reenactment rows require both face identities.")
    identity_sets, source_sets, driver_sets = {}, {}, {}
    for split, expected_count in zip(SPLITS, FACE_COUNTS):
        rows = kept.loc[kept["split"] == split]
        if set(rows["label"]) != {MF.OLD_REAL, MF.OLD_FAKE}:
            raise ValueError(f"{split} needs both real and fake videos.")
        identity_sets[split] = (set(rows["target_id"]) | set(rows["donor_id"])) - {""}
        if len(identity_sets[split]) != expected_count:
            raise ValueError(f"{split} must contain exactly {expected_count} face identities.")
        source_sets[split] = set(rows["source_clip"])
        driver_sets[split] = set(rows["driver_id"]) - {""}

    overlaps = {}
    for left, right in combinations(SPLITS, 2):
        face_overlap = identity_sets[left] & identity_sets[right]
        clip_overlap = source_sets[left] & source_sets[right]
        if face_overlap or clip_overlap:
            raise ValueError(f"{left}/{right} overlap: faces={sorted(face_overlap)}, source clips={sorted(clip_overlap)}")
        overlaps[f"{left}/{right}"] = {
            "face_identities": len(face_overlap),
            "source_clips": len(clip_overlap),
            "talking_face_audio_driver_clips": len(driver_sets[left] & driver_sets[right]),
        }
    return overlaps


def sample_training_fakes(kept, methods, seed, per_method, total_cap):
    in_train = kept["split"].eq("train")
    candidates = kept[in_train & kept["label"].eq(MF.OLD_FAKE)].copy()
    candidates["rank_key"] = candidates["path"].map(
        lambda path: hashlib.sha256(f"{seed}:{path}".encode("utf-8")).hexdigest()
    )
    candidates = candidates.sort_values("rank_key")
    by_method = candidates.groupby("method")
    if per_method:
        quota = int((in_train & kept["label"].eq(MF.OLD_REAL)).sum())
        available = by_method.size().reindex(methods, fill_value=0)
        shortage = available[available < quota]
        if not shortage.empty:
            details = ", ".join(f"{method}={count}" for method, count in shortage.items())
            raise ValueError(f"Each method needs {quota} training fakes. Insufficient: {details}")
        return candidates.index.difference(by_method.head(quota).index), "training_fake_per_method_quota"

    if total_cap < by_method.ngroups:
        raise ValueError("--max-train-fakes must allow at least one video per available training method")
    candidates["method_rank"] = by_method.cumcount()
    removed = candidates.sort_values(["method_rank", "rank_key"]).index[total_cap:]
    return removed, "training_fake_cap"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Full combined_manifest_p05.csv inventory")
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic 41/9/9 face-identity assignment")
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--train-fakes-per-method-equal-to-real", action="store_true",
                          help="Sample as many training fakes per method as there are genuine training videos")
    sampling.add_argument("--max-train-fakes", type=int, help="Cap training fakes, allocating evenly across methods")
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"Output directory already exists: {args.out}")

    inventory = load_inventory(args.manifest)
    kept, excluded, assignments = assign_videos(inventory, args.seed)
    if args.train_fakes_per_method_equal_to_real or args.max_train_fakes is not None:
        methods = sorted(inventory.loc[inventory["label"].eq(MF.OLD_FAKE), "method"].unique())
        try:
            removed, reason = sample_training_fakes(
                kept, methods, args.seed, args.train_fakes_per_method_equal_to_real, args.max_train_fakes
            )
        except ValueError as error:
            parser.error(str(error))
        excluded = pd.concat([excluded, kept.loc[removed].assign(reason=reason)]).sort_values("path")
        kept = kept.drop(removed)
    overlaps = audit_splits(kept)
    videos = MF.encode_videos(kept)
    videos["donor_id"] = kept["donor_id"]
    videos["driver_id"] = kept["driver_id"]
    if videos["video_key"].duplicated().any():
        raise ValueError("Duplicate GenD video key after conversion.")
    summary = {
        "protocol": "CelebDFv3 face-identity-disjoint 41/9/9",
        "seed": args.seed,
        "train_fakes_per_method_equal_to_real": args.train_fakes_per_method_equal_to_real,
        "max_train_fakes": args.max_train_fakes,
        "input_manifest_sha256": MF.sha256(args.manifest),
        "identity_counts": dict(zip(SPLITS, FACE_COUNTS)),
        "overlap": overlaps,
        "label_convention": "GenD: 0=real, 1=fake in videos.csv",
        "excluded_by_reason": excluded["reason"].value_counts().to_dict(),
        "splits": {},
    }
    for split in SPLITS:
        rows = videos.loc[videos["split"] == split]
        reals = rows["label"] == MF.GEND_REAL
        summary["splits"][split] = {
            "real": int(reals.sum()),
            "fake": int((~reals).sum()),
            "fake_by_method": rows.loc[~reals, "method"].value_counts().to_dict(),
            "video_list_sha256": MF.list_hash(rows["rel_path"]),
        }

    args.out.mkdir(parents=True)
    videos.to_csv(args.out / "videos.csv", index=False, lineterminator="\n")
    pd.DataFrame(sorted(assignments.items()), columns=["identity", "split"]).to_csv(
        args.out / "identities.csv", index=False, lineterminator="\n"
    )
    excluded[["path", "method", "target_id", "donor_id", "source_clip", "driver_id", "reason"]].to_csv(
        args.out / "excluded.csv", index=False, lineterminator="\n"
    )
    MF.write_json(args.out / "summary.json", summary)
    for split, counts in summary["splits"].items():
        print(f"{split}: {counts['real']} real / {counts['fake']} fake")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
