"""Write GenD frame-list files from preprocessing records.

  <out>/val.txt                 fixed 268 validation videos (old protocol)
  <out>/group_XX/train.txt      that group's 616 real + 616 fake training videos
  <out>/test/test.txt           fixed 266 test videos, only with --with-test;
                                kept in its own directory and never passed to training
  <out>/coverage.json           per split: videos without any usable face crop

With --identity-split, read the table from build_identity_split.py and write
one train.txt and one val.txt. Test remains opt-in via --with-test. Preprocess
the identity split in a separate work directory because video roles may change.

Each line is an absolute crop path <crops>/<dataset>/<source>/<video>/frame_XXXXX.png,
the layout GenD's DeepfakeDataset expects. Its label rule ("real" in the
source directory -> 0) is re-checked on every path against videos.csv.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gad import frames as FR  # noqa: E402
from gad import manifest as MF  # noqa: E402
from build_anomaly_table import audit_table  # noqa: E402
from build_identity_split import audit_splits  # noqa: E402
from gad.diagnostic_protocol import read_registered_plan, sha256_file


def frame_paths(video_rows, records_dir: Path, crops_dir: Path, verify_split=False):
    lines, unusable = [], []
    for row in video_rows.itertuples(index=False):
        record_path = records_dir / (row.video_key + ".json")
        if not record_path.exists():
            raise SystemExit("Not preprocessed yet: {} ({})".format(row.rel_path, record_path))
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if verify_split and (record.get("split"), record.get("frame_subset")) != (row.split, row.frame_subset):
            raise SystemExit(f"Preprocessing record does not match this split: {record_path}")
        if verify_split and record["status"] not in ("ok", "no_face"):
            raise SystemExit(f"Retry preprocessing for {row.rel_path}: {record['status']}")
        if record["status"] != "ok":
            unusable.append({"rel_path": row.rel_path, "status": record["status"]})
            continue
        if verify_split and not record["selected"]:
            raise SystemExit(f"No frames in successful preprocessing record: {record_path}")
        for index in record["selected"]:
            path = crops_dir / row.video_key / FR.frame_file_name(index)
            if not path.is_file():
                raise SystemExit("Missing crop {}".format(path))
            if MF.gend_label_from_source(path.parent.parent.name) != row.label:
                raise AssertionError("GenD label rule disagrees for {}".format(path))
            lines.append(path.as_posix())
    return lines, unusable


def write_list(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_single_lists(videos, out, records, crops, with_test, anomaly=False):
    if anomaly:
        audit_table(videos)
    else:
        if not {"donor_id", "driver_id"}.issubset(videos.columns):
            raise SystemExit("--identity-split expects videos.csv from build_identity_split.py.")
        audit_splits(videos)
    files = {"train": "train.txt", "val": "val.txt"}
    if with_test:
        files["test"] = "test/test.txt"
    coverage = {}
    for split, filename in files.items():
        rows = videos[videos["split"] == split].sort_values("rel_path")
        lines, unusable = frame_paths(rows, records, crops, verify_split=True)
        usable = rows.loc[~rows["rel_path"].isin([item["rel_path"] for item in unusable])]
        expected_labels = {MF.GEND_REAL} if anomaly and split != "test" else {MF.GEND_REAL, MF.GEND_FAKE}
        if set(rows["label"]) != expected_labels or set(usable["label"]) != expected_labels:
            raise SystemExit(f"Unexpected labels or missing scorable classes in the {split} split.")
        listed_fakes = rows.loc[rows["label"].eq(MF.GEND_FAKE), "method"].value_counts()
        usable_fakes = usable.loc[usable["label"].eq(MF.GEND_FAKE), "method"].value_counts()
        write_list(out / filename, lines)
        coverage[split] = {"videos": len(rows), "unusable": unusable, "frames": len(lines),
                           "real_videos_usable": int(usable["label"].eq(MF.GEND_REAL).sum()),
                           "fake_videos_usable_by_method": usable_fakes.reindex(listed_fakes.index, fill_value=0).to_dict(),
                           "list_sha256": MF.list_hash(rows["rel_path"])}
    return coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", required=True, help="Directory with videos.csv (and membership.csv for groups)")
    parser.add_argument("--work", required=True, help="Preprocessing work dir (crops/, records/)")
    parser.add_argument("--out", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--groups", type=int, nargs="*", default=list(MF.FORMAL_GROUPS))
    mode.add_argument("--identity-split", action="store_true")
    mode.add_argument("--anomaly", action="store_true", help="Read real-only train/val from build_anomaly_table.py")
    mode.add_argument("--diagnostic-plan", type=Path,
                      help="Registered train/calibration/dev/test protocol; test stays opt-in")
    parser.add_argument("--with-test", action="store_true")
    args = parser.parse_args()

    tables, work, out = Path(args.tables), Path(args.work), Path(args.out)
    if args.diagnostic_plan:
        if tables.resolve() != (args.diagnostic_plan.parent / "tables").resolve():
            parser.error("--tables must belong to --diagnostic-plan")
        coverage = write_diagnostic_lists(args.diagnostic_plan, work, out, args.with_test)
        print(json.dumps(coverage, indent=2))
        return
    videos = pd.read_csv(tables / "videos.csv", keep_default_na=not (args.identity_split or args.anomaly),
                         dtype={"target_id": str, "donor_id": str, "source_clip": str}
                         if args.identity_split or args.anomaly else None)
    records, crops = work / "records", work.resolve() / "crops"
    if args.identity_split or args.anomaly:
        coverage = write_single_lists(videos, out, records, crops, args.with_test, anomaly=args.anomaly)
    else:
        member = pd.read_csv(tables / "membership.csv")
        coverage = write_group_lists(videos, member, out, records, crops, args.groups, args.with_test)
    MF.write_json(out / "coverage.json", coverage)
    for name, item in coverage.items():
        print("{}: {} videos, {} frames, {} without usable face".format(
            name, item["videos"], item["frames"], len(item["unusable"])))


def write_diagnostic_lists(plan_path, work, out, with_test=False):
    _, videos = read_registered_plan(plan_path)
    out = Path(out)
    if out.exists():
        raise FileExistsError("Use a new list directory; registered lists must not be overwritten")
    files, coverage, combined = {}, {}, []
    roles = ("train", "calibration", "dev", "test") if with_test else ("train", "calibration", "dev")
    for role in roles:
        rows = videos.loc[videos["split"].eq(role)].sort_values("rel_path")
        lines, unusable = frame_paths(rows, Path(work) / "records", Path(work).resolve() / "crops", True)
        usable = rows.loc[~rows["rel_path"].isin([r["rel_path"] for r in unusable])]
        if set(usable["label"]) != ({0} if role in ("train", "calibration") else {0, 1}):
            raise ValueError(f"Missing scorable classes in {role}")
        files["test/test.txt" if role == "test" else role + ".txt"] = lines
        if role != "test":
            combined.extend(lines)
        coverage[role] = {"videos": len(rows), "frames": len(lines), "unusable": unusable,
                          "usable_keys": usable["video_key"].tolist()}
    files["development.txt"] = combined
    for filename, lines in files.items():
        write_list(out / filename, lines)
    MF.write_json(out / "frame_lists.json", {
        "plan_sha256": sha256_file(plan_path), "coverage": coverage,
        "files": {name: sha256_file(out / name) for name in files}})
    return coverage


def write_group_lists(videos, member, out, records, crops, groups, with_test):
    coverage = {}
    val = videos[videos["split"] == "val"].sort_values("rel_path")
    lines, unusable = frame_paths(val, records, crops)
    write_list(out / "val.txt", lines)
    coverage["val"] = {"videos": len(val), "unusable": unusable, "frames": len(lines),
                       "list_sha256": MF.list_hash(val["rel_path"])}

    for group in groups:
        keep = set(member[(member["group"] == group) & (member["split"] == "train")]["rel_path"])
        train = videos[videos["rel_path"].isin(keep)].sort_values("rel_path")
        assert len(train) == 1232, (group, len(train))
        lines, unusable = frame_paths(train, records, crops)
        write_list(out / "group_{:02d}".format(group) / "train.txt", lines)
        coverage["group_{:02d}_train".format(group)] = {"videos": len(train), "unusable": unusable, "frames": len(lines)}

    if with_test:
        test = videos[videos["split"] == "test"].sort_values("rel_path")
        lines, unusable = frame_paths(test, records, crops)
        write_list(out / "test" / "test.txt", lines)
        coverage["test"] = {"videos": len(test), "unusable": unusable, "frames": len(lines),
                            "list_sha256": MF.list_hash(test["rel_path"])}

    return coverage


if __name__ == "__main__":
    main()
