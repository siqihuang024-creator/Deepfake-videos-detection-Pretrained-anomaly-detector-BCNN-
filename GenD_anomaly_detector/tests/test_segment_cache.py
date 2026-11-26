import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_segment_cache import prepare, digest


class SegmentCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "tests")
        self.root = Path(self.temp.name)
        self.faces = self.root / "faces"
        self.faces.mkdir()
        self.plan = self.root / "segment_data_plan.json"
        self.rows = []
        self.records = []
        # Include a low-coverage successful clip and an unscorable clip.
        for i, (role, label, n) in enumerate((("train", 0, 32), ("train", 0, 1),
                                             ("calibration", 0, 32), ("dev", 0, 32),
                                             ("dev", 1, 32), ("train", 0, 0))):
            key = "external/%s/%s" % ("fake_external" if label else "real_external", i)
            row = dict(video_key=key, split=role, label=label, source_clip=role + str(i),
                       rel_path="dataset/%s.mp4" % i, frame_subset="uniform32",
                       parent_video_key="parent%s" % i, clip_id=str(i), source_bytes=10,
                       source_mtime_ns=1700000000000000000, segment_start_seconds=0,
                       segment_end_seconds=10, stream_start_seconds=0)
            record = dict(row, label=label, detection_policy_details={"name": "pad_on_empty_v1"},
                          target_size=[224, 224], selected=list(range(n)), total_frames=250,
                          selected_timestamps_seconds=[j / 25 for j in range(n)],
                          sustained_face_candidate=n == 32, status="ok" if n else "no_face")
            self.rows.append(row)
            self.records.append(record)
            for j in range(n):
                crop = self.faces / "crops" / key / ("frame_%05d.png" % j)
                crop.parent.mkdir(parents=True, exist_ok=True)
                crop.write_bytes(b"presence-only fixture")
        self.publish()

    def publish(self):
        table = self.root / "candidate_videos.csv"
        with table.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        names = ("parents.csv", "selected_originals.csv", "reserved_test.csv", "unused_development.csv")
        for name in names:
            (self.root / name).write_text("metadata\n", encoding="utf-8")
        plan = dict(version=1, purpose="registered bounded candidate preprocessing; no training or final testing",
                    table=table.name, table_sha256=digest(table), candidate_clips=len(self.rows),
                    registered_files={name: digest(self.root / name) for name in (*names, table.name)})
        self.plan.write_text(json.dumps(plan), encoding="utf-8")
        contract = self.faces / "pilot_run_contract.json"
        contract.write_text(json.dumps({"pilot_plan_sha256": digest(self.plan),
                                       "detection_policy": {"name": "pad_on_empty_v1"}}), encoding="utf-8")
        for record in self.records:
            record["pilot_contract_sha256"] = digest(contract)
            path = self.faces / "records" / (record["video_key"] + ".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record), encoding="utf-8")
        (self.faces / "pilot_summary.json").write_text(json.dumps({
            "all_clips_attempted": True, "clips_attempted": len(self.rows), "clips_planned": len(self.rows),
            "groups": [{"failed_clips": 0, "selected_frames": sum(len(r["selected"]) for r in self.records)}]}), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def run_prepare(self, name="out"):
        with redirect_stdout(io.StringIO()):
            return prepare(self.plan, self.faces, self.root / name)

    def test_include_low_coverage_keep_no_face_denominator_and_refuse_overwrite(self):
        result = self.run_prepare()
        self.assertEqual(result["frame_count"], 129)
        self.assertEqual(result["scorable_candidates"], 5)
        self.assertEqual(len(result["unscorable_no_face_keys"]), 1)
        self.assertEqual(result["low_coverage_included_keys"], [self.rows[1]["video_key"]])
        self.assertEqual(len((self.root / "out/candidates.txt").read_text(encoding="utf-8").splitlines()), 129)
        self.assertEqual(result["files"]["candidates.txt"], digest(self.root / "out/candidates.txt"))
        with self.assertRaises(FileExistsError):
            self.run_prepare()

    def test_table_tampering_and_missing_crop_fail_before_publication(self):
        path = self.root / "reserved_test.csv"
        path.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "table changed"):
            self.run_prepare()
        self.publish()
        next((self.faces / "crops").rglob("*.png")).unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_prepare()
        self.assertFalse((self.root / "out").exists())

    def test_test_role_and_cross_role_source_rejected(self):
        self.rows[0]["split"] = "test"
        self.publish()
        with self.assertRaises(ValueError):
            self.run_prepare()
        self.rows[0]["split"] = "train"
        self.rows[2]["source_clip"] = self.rows[0]["source_clip"]
        self.publish()
        with self.assertRaisesRegex(ValueError, "crosses roles"):
            self.run_prepare()

    def test_stale_record_contract_invalid_times_and_nanosecond_drift_rejected(self):
        self.records[0]["selected_timestamps_seconds"][0] = -1
        self.publish()
        with self.assertRaisesRegex(ValueError, "presentation times"):
            self.run_prepare()
        self.records[0]["selected_timestamps_seconds"][0] = 0
        self.records[0]["source_mtime_ns"] += 1
        self.publish()
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.run_prepare()
        self.records[0]["source_mtime_ns"] -= 1
        self.publish()
        path = self.faces / "records" / (self.records[0]["video_key"] + ".json")
        record = json.loads(path.read_text())
        record["pilot_contract_sha256"] = "changed"
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "contract changed"):
            self.run_prepare()


if __name__ == "__main__":
    unittest.main()
