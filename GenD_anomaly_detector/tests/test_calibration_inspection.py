"""Small export scope, raw/crop timing identity, and no test-role access."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from export_calibration_inspection import run, validate_request, inside
from gad.diagnostic_protocol import sha256_file
from gad.segment_pilot import window_spec


class CalibrationInspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "tests")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.diag, self.data, self.cache, self.videos = [self.root / n for n in ("diag", "data", "cache", "videos")]
        for p in (self.diag, self.data / "faces", self.cache, self.videos):
            p.mkdir(parents=True)
        video = self.videos / "TalkingHead-1KH/raw/example.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"Only a decoder fixture, not a real video")
        self.row = dict(video_key="external/real_external/example", head_role="calibration", split="calibration",
            label=0, dataset="TalkingHead-1KH", rel_path="TalkingHead-1KH/raw/example.mp4", source_clip="one-source",
            sample_unit="segment_pilot", parent_video_key="parent", clip_id="clip", segment_start_seconds=10.,
            segment_end_seconds=12., stream_start_seconds=2., source_bytes=video.stat().st_size,
            source_mtime_ns=video.stat().st_mtime_ns)
        record = dict(window_spec(self.row), status="ok", video_key=self.row["video_key"], rel_path=self.row["rel_path"],
            selected=list(range(32)), selected_timestamps_seconds=[12+i*.05 for i in range(32)])
        self.record = self.data / "faces/records" / (self.row["video_key"] + ".json")
        self.record.parent.mkdir(parents=True)
        self.record.write_text(json.dumps(record), encoding="utf-8")
        crops = self.data / "faces/crops" / self.row["video_key"]
        crops.mkdir(parents=True)
        for i in range(32):
            Image.new("RGB", (224, 224), (i, 10, 20)).save(crops / ("frame_%05d.png" % i))
        (self.data / "segment_data_plan.json").write_text("{}", encoding="utf-8")
        (self.data / "faces/pilot_run_contract.json").write_text("{}", encoding="utf-8")
        lists = {"records_sha256": {self.row["video_key"]: sha256_file(self.record)},
                 "preprocessing_contract_sha256": sha256_file(self.data / "faces/pilot_run_contract.json")}
        (self.cache / "frame_lists.json").write_text(json.dumps(lists), encoding="utf-8")
        (self.diag / "covariance_registration.json").write_text("{}", encoding="utf-8")
        self.publish(self.row)

    def publish(self, row):
        pd.DataFrame([row]).to_csv(self.diag / "training_units.csv", index=False)
        target = dict(row, selection_score=2., selection_threshold=1.)
        pd.DataFrame([target]).to_csv(self.diag / "inspection_targets.csv", index=False)
        request = dict(frames_per_clip=5, target_sha256=sha256_file(self.diag / "inspection_targets.csv"),
            covariance_registration_sha256=sha256_file(self.diag / "covariance_registration.json"),
            data_plan_sha256=sha256_file(self.data / "segment_data_plan.json"),
            candidate_list_registration_sha256=sha256_file(self.cache / "frame_lists.json"))
        (self.diag / "inspection_request.json").write_text(json.dumps(request), encoding="utf-8")
        completion = dict(completed=True, phase="development_clip_covariance",
            registration_sha256=sha256_file(self.diag / "covariance_registration.json"),
            artifacts_sha256={n: sha256_file(self.diag / n) for n in
                ("inspection_request.json", "inspection_targets.csv", "training_units.csv")})
        (self.diag / "completion.json").write_text(json.dumps(completion), encoding="utf-8")

    @staticmethod
    def decoder(path, start, end, origin):
        for i in range(32):
            yield i, np.zeros((120, 160, 3), np.uint8), origin+start+i*.05

    def test_export_is_five_pairs_per_clip_and_archive_is_checksummed(self):
        out = self.root / "export"
        run(self.diag, self.data, self.cache, self.videos, out, decoder=self.decoder)
        self.assertEqual(len(list(out.rglob("frame_*.png"))), 5)
        self.assertEqual(len(list(out.rglob("raw_*.jpg"))), 5)
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(len(manifest["clips"]), 1)
        self.assertEqual([e["frame_index"] for e in manifest["clips"][0]["frames"]], [0, 8, 16, 23, 31])
        archive = Path(str(out)+".tar.gz")
        self.assertEqual(Path(str(archive)+".sha256").read_text().split()[0], sha256_file(archive))
        with self.assertRaises(FileExistsError):
            run(self.diag, self.data, self.cache, self.videos, out, decoder=self.decoder)

    def test_test_role_tampering_and_escape_are_rejected_before_export(self):
        self.publish(dict(self.row, head_role="test", split="test"))
        with self.assertRaisesRegex(ValueError, "calibration"):
            validate_request(self.diag, self.data, self.cache)
        self.publish(self.row)
        self.record.write_text("{}", encoding="utf-8")
        out = self.root / "reject"
        with self.assertRaisesRegex(ValueError, "record changed"):
            run(self.diag, self.data, self.cache, self.videos, out, decoder=self.decoder)
        self.assertFalse(out.exists())
        with self.assertRaises(ValueError):
            inside(self.videos, "../outside.mp4")

    def test_raw_pts_drift_never_publishes_completion_or_archive(self):
        def drift(path, start, end, origin):
            for index, image, pts in self.decoder(path, start, end, origin):
                yield index, image, pts+.01
        out = self.root / "drift"
        with self.assertRaisesRegex(ValueError, "PTS"):
            run(self.diag, self.data, self.cache, self.videos, out, decoder=drift)
        self.assertFalse((out / "completion.json").exists())
        self.assertFalse(Path(str(out)+".tar.gz").exists())
