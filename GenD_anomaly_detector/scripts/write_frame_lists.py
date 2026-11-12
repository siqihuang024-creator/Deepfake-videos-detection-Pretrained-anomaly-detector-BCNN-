"""Write GenD frame-list files from preprocessing records.

  <out>/val.txt                 fixed 268 validation videos (old protocol)
  <out>/group_XX/train.txt      that group's 616 real + 616 fake training videos
  <out>/test/test.txt           fixed 266 test videos, only with --with-test;
                                kept in its own directory and never passed to training
  <out>/coverage.json           per split: videos without any usable face crop

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


def frame_paths(video_rows, records_dir: Path, crops_dir: Path):
    lines, unusable = [], []
    for row in video_rows.itertuples(index=False):
        record_path = records_dir / (row.video_key + ".json")
        if not record_path.exists():
            raise SystemExit("Not preprocessed yet: {} ({})".format(row.rel_path, record_path))
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["status"] != "ok":
            unusable.append({"rel_path": row.rel_path, "status": record["status"]})
            continue
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", required=True, help="Directory with videos.csv and membership.csv")
    parser.add_argument("--work", required=True, help="Preprocessing work dir (crops/, records/)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--groups", type=int, nargs="*", default=list(MF.FORMAL_GROUPS))
    parser.add_argument("--with-test", action="store_true")
    args = parser.parse_args()

    tables, work, out = Path(args.tables), Path(args.work), Path(args.out)
    videos = pd.read_csv(tables / "videos.csv")
    member = pd.read_csv(tables / "membership.csv")
    records, crops = work / "records", work.resolve() / "crops"
    coverage = {}

    val = videos[videos["split"] == "val"].sort_values("rel_path")
    lines, unusable = frame_paths(val, records, crops)
    write_list(out / "val.txt", lines)
    coverage["val"] = {"videos": len(val), "unusable": unusable, "frames": len(lines),
                       "list_sha256": MF.list_hash(val["rel_path"])}

    for group in args.groups:
        keep = set(member[(member["group"] == group) & (member["split"] == "train")]["rel_path"])
        train = videos[videos["rel_path"].isin(keep)].sort_values("rel_path")
        assert len(train) == 1232, (group, len(train))
        lines, unusable = frame_paths(train, records, crops)
        write_list(out / "group_{:02d}".format(group) / "train.txt", lines)
        coverage["group_{:02d}_train".format(group)] = {"videos": len(train), "unusable": unusable, "frames": len(lines)}

    if args.with_test:
        test = videos[videos["split"] == "test"].sort_values("rel_path")
        lines, unusable = frame_paths(test, records, crops)
        write_list(out / "test" / "test.txt", lines)
        coverage["test"] = {"videos": len(test), "unusable": unusable, "frames": len(lines),
                            "list_sha256": MF.list_hash(test["rel_path"])}

    MF.write_json(out / "coverage.json", coverage)
    for name, item in coverage.items():
        print("{}: {} videos, {} frames, {} without usable face".format(
            name, item["videos"], item["frames"], len(item["unusable"])))


if __name__ == "__main__":
    main()
