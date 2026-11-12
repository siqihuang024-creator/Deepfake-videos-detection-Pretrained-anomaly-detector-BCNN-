"""Old 30-group manifests -> explicit video tables for the GenD pipeline.

Writes (under --out):
  videos.csv       one row per distinct video: split, mapped GenD label,
                   output directory key, frame subset rule, cluster ids
  membership.csv   (group, split, rel_path, video_key) for groups 01-30
  summary.json     counts, protocol hashes, list hashes

Splits and groups follow the old protocol; nothing is re-drawn here.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gad import manifest as MF  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="Output directory, e.g. <work>/tables/cdfv3_old_protocol")
    parser.add_argument("--video-root", default=None,
                        help="CelebDFv3 root (contains REAL/ and FAKE/); if given, every video must exist")
    args = parser.parse_args()

    table = MF.build_all_groups()
    videos = MF.unique_videos(table)

    if args.video_root:
        root = Path(args.video_root)
        missing = [p for p in videos["rel_path"] if not (root / p).is_file()]
        if missing:
            raise SystemExit("{} videos missing under {}, e.g. {}".format(len(missing), root, missing[:5]))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    videos.to_csv(out / "videos.csv", index=False)
    table[["group", "split", "rel_path", "video_key", "label"]].to_csv(out / "membership.csv", index=False)

    counts = videos.groupby(["split", "label"]).size()
    summary = {
        "protocol": "old 30-group video-disjoint (celeb_video_disjoint_seed42), groups 01-30",
        "label_convention": "label: GenD (0 real, 1 fake); old_label: old manifest (1 real, 0 fake)",
        "distinct_videos_by_split_label": {"{}/{}".format(s, l): int(n) for (s, l), n in counts.items()},
        "val_list_sha256": MF.list_hash(videos[videos.split == "val"]["rel_path"]),
        "test_list_sha256": MF.list_hash(videos[videos.split == "test"]["rel_path"]),
        "protocol_sha256sums": (MF.PROTOCOL_DIR / "SHA256SUMS").read_text().splitlines(),
        "gend_sources": sorted(videos["gend_source"].unique()),
    }
    MF.write_json(out / "summary.json", summary)
    print("videos: {} distinct; by split/label: {}".format(len(videos), summary["distinct_videos_by_split_label"]))
    print("wrote", out)


if __name__ == "__main__":
    main()
