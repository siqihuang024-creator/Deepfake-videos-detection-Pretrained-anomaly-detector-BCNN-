"""Old 30-group CelebDFv3 manifests -> one explicit video table.

Label conventions (the main trap):
  old manifest : label 1 = real, 0 = fake
  GenD         : label 0 = real, 1 = fake, derived from the *source directory
                 name*: "real" (lowercase substring, case-sensitive) -> 0.

We therefore never rely on the original directory names (``REAL/Real_WEB``
contains no lowercase "real"). Each video gets an output source directory
whose name is chosen from its manifest label, and the GenD rule is re-applied
to that name and asserted to agree with the mapped label.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

OLD_REAL, OLD_FAKE = 1, 0
GEND_REAL, GEND_FAKE = 0, 1

PROTOCOL_DIR = Path(__file__).resolve().parent.parent / "protocols" / "celeb_video_disjoint_seed42"
FORMAL_GROUPS = tuple(range(1, 31))

EXPECTED_COUNTS = {
    ("train", GEND_REAL): 616, ("train", GEND_FAKE): 616,
    ("val", GEND_REAL): 134, ("val", GEND_FAKE): 134,
    ("test", GEND_REAL): 133, ("test", GEND_FAKE): 133,
}


def gend_label_from_source(source: str) -> int:
    """Exactly the rule in upstream src/dataset/dataset.py (binary=True)."""
    return GEND_REAL if "real" in source else GEND_FAKE


def map_label(old_label: int) -> int:
    if old_label == OLD_REAL:
        return GEND_REAL
    if old_label == OLD_FAKE:
        return GEND_FAKE
    raise ValueError("Unexpected old label {!r}".format(old_label))


def gend_source_name(rel_path: str, gend_label: int) -> str:
    """Output source directory: ``real_<subset>`` or ``<family>-<method>``."""
    parts = rel_path.split("/")
    if gend_label == GEND_REAL:
        assert parts[0] == "REAL" and len(parts) == 3, rel_path
        name = "real_" + parts[1]
    else:
        assert parts[0] == "FAKE" and len(parts) == 4, rel_path
        name = parts[1] + "-" + parts[2]
    if gend_label_from_source(name) != gend_label:
        raise AssertionError("GenD would mislabel {} via source dir {!r}".format(rel_path, name))
    return name


def video_dir_name(rel_path: str) -> str:
    stem = rel_path.rsplit("/", 1)[1]
    assert stem.endswith(".mp4"), rel_path
    return stem[:-4]


def frame_subset_for_split(split: str) -> str:
    """GenD: training videos get random-32, validation/test get uniform-32."""
    return {"train": "random32", "val": "uniform32", "test": "uniform32"}[split]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_protocol_hashes(protocol_dir: Path = PROTOCOL_DIR) -> None:
    for line in (protocol_dir / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        name = name.lstrip("*")
        if sha256(protocol_dir / name) != digest:
            raise AssertionError("Protocol file changed: {}".format(name))


def load_group(group: int, protocol_dir: Path = PROTOCOL_DIR) -> pd.DataFrame:
    path = protocol_dir / "group_{:02d}.csv".format(group)
    frame = pd.read_csv(path)
    frame.insert(0, "group", group)
    return frame


def encode_videos(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "split": frame["split"],
        "dataset": frame["dataset"],
        "rel_path": frame["path"],
        "old_label": frame["label"].astype(int),
        "method": frame["method"],
        "target_id": frame["target_id"].astype(str),
        "source_clip": frame["source_clip"].astype(str),
        "source_id": frame["source_id"].astype(str),
    })
    out["label"] = out["old_label"].map(map_label)
    out["gend_source"] = [gend_source_name(p, l) for p, l in zip(out["rel_path"], out["label"])]
    out["video_dir"] = out["rel_path"].map(video_dir_name)
    out["video_key"] = out["dataset"] + "/" + out["gend_source"] + "/" + out["video_dir"]
    out["frame_subset"] = out["split"].map(frame_subset_for_split)
    return out


def to_video_table(frame: pd.DataFrame) -> pd.DataFrame:
    out = encode_videos(frame)
    out.insert(0, "group", frame["group"])
    check_video_table(out)
    return out


def check_video_table(table: pd.DataFrame) -> None:
    # Cross-check the mapping in both directions on every row.
    for old, new, cls in zip(table["old_label"], table["label"], table["rel_path"]):
        top = cls.split("/")[0]
        if (old == OLD_REAL) != (top == "REAL") or (new == GEND_REAL) != (top == "REAL"):
            raise AssertionError("Label/path disagreement: {}".format(cls))
    for key, label in zip(table["video_key"], table["label"]):
        if gend_label_from_source(key.split("/")[-2]) != label:
            raise AssertionError("GenD path rule disagrees for {}".format(key))
    for group, part in table.groupby("group"):
        if part["rel_path"].duplicated().any() or part["video_key"].duplicated().any():
            raise AssertionError("Duplicate video in group {}".format(group))
        splits = {s: set(p["rel_path"]) for s, p in part.groupby("split")}
        for a in splits:
            for b in splits:
                if a < b and splits[a] & splits[b]:
                    raise AssertionError("Split overlap {}/{} in group {}".format(a, b, group))
        counts = part.groupby(["split", "label"]).size().to_dict()
        if counts != EXPECTED_COUNTS:
            raise AssertionError("Group {} counts {} != {}".format(group, counts, EXPECTED_COUNTS))


def build_all_groups(groups=FORMAL_GROUPS, protocol_dir: Path = PROTOCOL_DIR) -> pd.DataFrame:
    verify_protocol_hashes(protocol_dir)
    table = pd.concat([to_video_table(load_group(g, protocol_dir)) for g in groups], ignore_index=True)
    # Validation and test lists must be identical across groups (old protocol).
    for split in ("val", "test"):
        reference = None
        for _, part in table[table["split"] == split].groupby("group"):
            paths = tuple(sorted(part["rel_path"]))
            if reference is None:
                reference = paths
            elif paths != reference:
                raise AssertionError("{} list differs between groups".format(split))
    # A video keeps one split across all groups.
    roles = table.groupby("rel_path")["split"].nunique()
    if (roles > 1).any():
        raise AssertionError("Videos with several splits: {}".format(list(roles[roles > 1].index[:5])))
    return table


def unique_videos(table: pd.DataFrame) -> pd.DataFrame:
    cols = ["split", "dataset", "rel_path", "old_label", "label", "method", "target_id",
            "source_clip", "source_id", "gend_source", "video_dir", "video_key", "frame_subset"]
    videos = table[cols].drop_duplicates().sort_values("rel_path").reset_index(drop=True)
    if videos["rel_path"].duplicated().any():
        raise AssertionError("Inconsistent per-video attributes across groups")
    return videos


def list_hash(paths) -> str:
    return hashlib.sha256("\n".join(sorted(paths)).encode("utf-8")).hexdigest()


def write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
