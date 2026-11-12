"""CPU-only checks of the protocol layer. Run: python tests/test_protocol.py"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gad import frames as FR  # noqa: E402
from gad import manifest as MF  # noqa: E402
from gad import scoring as S  # noqa: E402


def test_label_mapping_and_sources():
    assert MF.map_label(1) == MF.GEND_REAL and MF.map_label(0) == MF.GEND_FAKE
    # Original directory would be mislabelled by GenD's rule: the reason for renaming.
    assert MF.gend_label_from_source("Real_WEB") == MF.GEND_FAKE
    assert MF.gend_source_name("REAL/Real_WEB/id1_0001.mp4", MF.GEND_REAL) == "real_Real_WEB"
    assert MF.gend_source_name("REAL/Real_YouTube/00001.mp4", MF.GEND_REAL) == "real_Real_YouTube"
    name = MF.gend_source_name("FAKE/TalkingFace/Real3DPortrait/a.mp4", MF.GEND_FAKE)
    assert name == "TalkingFace-Real3DPortrait" and MF.gend_label_from_source(name) == MF.GEND_FAKE


def test_all_groups():
    table = MF.build_all_groups()
    videos = MF.unique_videos(table)
    assert set(table["group"]) == set(range(1, 31))
    counts = videos.groupby(["split", "label"]).size().to_dict()
    assert counts[("val", 0)] == 134 and counts[("val", 1)] == 134
    assert counts[("test", 0)] == 133 and counts[("test", 1)] == 133
    assert counts[("train", 0)] <= 622 and counts[("train", 1)] == 30 * 616
    assert "REAL/Real_WEB/id27_0005.mp4" not in set(videos["rel_path"])
    assert (videos["old_label"] == 1 - videos["label"]).all()
    assert not videos["video_key"].duplicated().any()
    assert set(videos[videos.split == "train"]["frame_subset"]) == {"random32"}
    assert set(videos[videos.split != "train"]["frame_subset"]) == {"uniform32"}
    return videos


def test_frame_rules():
    detected = list(range(0, 300, 2))
    u = FR.uniform_subset(detected)
    assert len(u) == 32 and u[0] == 0 and u[-1] == 298 and set(u) <= set(detected)
    r1 = FR.random_subset(detected, "FAKE/x/y/a.mp4")
    r2 = FR.random_subset(detected, "FAKE/x/y/a.mp4")
    r3 = FR.random_subset(detected, "FAKE/x/y/b.mp4")
    assert r1 == r2 and r1 != r3 and len(set(r1)) == 32 and set(r1) <= set(detected)
    assert FR.uniform_subset([3, 5, 9]) == [3, 5, 9] and FR.random_subset([3, 5], "p") == [3, 5]
    assert FR.uniform_subset([]) == []


def test_scoring(videos):
    val = videos[videos.split == "val"].copy()
    rng = np.random.default_rng(0)
    rows = []
    for key, label in zip(val["video_key"], val["label"]):
        for _ in range(4):
            rows.append((key, float(np.clip(rng.normal(0.3 + 0.4 * label, 0.15), 0, 1))))
    preds = pd.DataFrame(rows, columns=["video_key", "fake_prob"])
    merged, missing = S.join_videos(val, S.video_scores(preds.iloc[4:]))  # drop one video's frames
    assert len(missing) == 1 and len(merged) == 267
    threshold = S.calibrate_on_validation(merged)
    result = S.evaluate_split(merged, threshold, draws=200)
    assert 0.8 < result["auroc"] <= 1.0
    assert abs(result["observed_false_positive_rate"] - 0.05) < 0.02
    print("synthetic:", S.short_summary(result))


if __name__ == "__main__":
    test_label_mapping_and_sources()
    v = test_all_groups()
    test_frame_rules()
    test_scoring(v)
    print("all protocol checks passed")
