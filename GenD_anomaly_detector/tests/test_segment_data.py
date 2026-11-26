"""Source budgets, independent roles and registration checks for candidate preparation."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from audit_video_durations import atomic_text, digest, write_csv
from prepare_segment_pilot import prepare as reference_prepare
from prepare_segment_data import prepare, select_originals
from preprocess_segment_pilot import load_pilot


class SegmentDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "tests")
        self.root = Path(self.temporary.name)
        self.videos = self.root / "videos"
        self.videos.mkdir()
        rows = []
        for role in ("train", "calibration"):
            for dataset in ("HDTF", "TalkingHead-1KH"):
                for group in range(2):
                    for copy in range(2):
                        rows.append(dict(dataset=dataset, rel_path=f"{dataset}/{role}_{group}_{copy}.mp4",
                                         label=0, split=role, source_clip=f"{dataset}_{role}_{group}", method="real"))
        for group in range(3):
            for label, copy in ((0, 0), (1, 0), (1, 1)):
                rows.append(dict(dataset="DFDC-Kaggle", rel_path=f"DFDC-Kaggle/{group}_{label}_{copy}.mp4",
                                 label=label, split="dev", source_clip=f"dfdc_{group}", method="real" if not label else "fake"))
        for label in (0, 1):
            rows.append(dict(dataset="PolyGlotFake", rel_path=f"PolyGlotFake/{label}.mp4", label=label,
                             split="test", source_clip="poly_original", method="real" if not label else "fake"))
        durations = []
        for row in rows:
            path = self.videos / row["rel_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"metadata-stage-placeholder")
            stat = path.stat()
            row["bytes"] = stat.st_size
            durations.append(dict(dataset=row["dataset"], rel_path=row["rel_path"], status="ok",
                                  bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, stream_start_seconds=0,
                                  duration_source="video_stream",
                                  duration_seconds=10.01 if row["split"] == "dev" else 100))
        self.parents = pd.DataFrame(rows)
        self.manifest = self.root / "parents.csv"
        write_csv(self.manifest, rows)
        self.audit = self.root / "duration_audit"
        self.audit.mkdir()
        write_csv(self.audit / "durations.csv", durations)
        write_csv(self.audit / "errors.csv", [])
        atomic_text(self.audit / "summary.json", json.dumps({"errors": 0}))
        atomic_text(self.audit / "audit_contract.json", json.dumps({"video_root": str(self.videos.resolve())}))
        atomic_text(self.audit / "SHA256SUMS.txt", "".join(
            f"{digest(self.audit / name)}  {name}\n" for name in
            ("durations.csv", "errors.csv", "summary.json", "audit_contract.json")))
        with redirect_stdout(io.StringIO()):
            reference_prepare(self.manifest, self.audit, self.root / "reference", placement="interior_v1")
        self.reference = self.root / "reference/segment_pilot.json"

    def tearDown(self):
        self.temporary.cleanup()

    def register(self, name="registered", **kwargs):
        out = self.root / name
        with redirect_stdout(io.StringIO()):
            plan = prepare(self.manifest, self.audit, self.reference, self.videos, out,
                           dev_source_groups=2, **kwargs)
        return out, plan

    def test_global_a_budget_and_paired_development_subset(self):
        selected = select_originals(self.parents, dev_source_groups=2)
        a = selected[selected["split"].isin(("train", "calibration"))]
        self.assertEqual(len(a), 8)
        self.assertTrue(a.groupby("source_clip").size().eq(1).all())
        dev = selected[selected["split"].eq("dev")]
        self.assertEqual(len(dev), 4)
        self.assertTrue(all(set(group.label) == {0, 1} for _, group in dev.groupby("source_clip")))
        self.assertNotIn("test", set(selected.split))
        reordered = select_originals(self.parents.sample(frac=1, random_state=3), dev_source_groups=2)
        self.assertEqual(selected.rel_path.tolist(), reordered.rel_path.tolist())

    def test_registration_preserves_tests_and_all_candidate_records(self):
        out, plan = self.register()
        self.assertEqual(plan["candidate_clips"], 36)
        self.assertEqual(plan["maximum_selected_images"], 1152)
        self.assertEqual(plan["reserved_test_videos"], 2)
        self.assertEqual(plan["dev_videos"], 4)
        self.assertEqual(plan["unused_dev_videos"], 5)
        loaded, rows = load_pilot(out / "segment_data_plan.json")
        self.assertEqual(loaded, plan)
        self.assertNotIn("test", set(rows.split))
        self.assertEqual(len(rows), 36)
        parents = pd.read_csv(out / "parents.csv")
        self.assertEqual(len(parents), len(self.parents))
        self.assertEqual(len(pd.read_csv(out / "reserved_test.csv")), 2)
        self.assertEqual(plan["detection_policy"]["name"], "pad_on_empty_v1")

    def test_clip_budget_failure_happens_before_creating_output(self):
        with self.assertRaisesRegex(ValueError, "budget exceeded"):
            self.register(max_candidate_clips=1)
        self.assertFalse((self.root / "registered").exists())

    def test_changed_original_manifest_is_rejected(self):
        with self.manifest.open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            self.register()

    def test_changed_selected_video_is_rejected_before_registration(self):
        selected = select_originals(self.parents, dev_source_groups=2)
        path = self.videos / selected.iloc[0].rel_path
        path.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.register()
        self.assertFalse((self.root / "registered").exists())

    def test_modified_reserved_metadata_invalidates_registered_preprocessing(self):
        out, _ = self.register()
        with (out / "reserved_test.csv").open("a", encoding="utf-8") as stream:
            stream.write("changed")
        with self.assertRaisesRegex(ValueError, "Registered data-preparation file changed"):
            load_pilot(out / "segment_data_plan.json")

    def test_missing_paired_dev_groups_cannot_silently_expand_scope(self):
        with self.assertRaisesRegex(ValueError, "paired dev groups"):
            select_originals(self.parents, dev_source_groups=4)


if __name__ == "__main__":
    unittest.main()
