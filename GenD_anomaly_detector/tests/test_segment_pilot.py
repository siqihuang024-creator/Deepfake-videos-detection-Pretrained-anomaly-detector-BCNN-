"""CPU-only metadata, window-boundary and resume tests; no real models needed."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from gad.video_probe import ffprobe_metadata, finite_number, pyav_metadata
from gad.segment_pilot import (bounded_windows, iter_window_frames, make_pilot, window_spec,
                               interior_windows, temporal_face_coverage, sustained_candidate)
from audit_video_durations import audit, atomic_text, digest, inventory_rows, probe_row, write_csv
from prepare_segment_pilot import prepare as prepare_pilot
from preprocess_segment_pilot import load_pilot, summarize
from prepare_diagnostic import prepare


class ProbeTests(unittest.TestCase):
    def test_fractional_fps_and_explicit_container_fallback(self):
        result = ffprobe_metadata({"streams": [{"avg_frame_rate": "30000/1001", "r_frame_rate": "60/1",
                                               "time_base": "1/90000", "start_time": "2.5"}],
                                   "format": {"duration": "12.75"}})
        self.assertAlmostEqual(result["fps_average"], 29.97002997002997)
        self.assertEqual(result["stream_start_seconds"], 2.5)
        self.assertEqual(result["duration_source"], "container_fallback")
        self.assertEqual(result["duration_seconds"], 12.75)
        self.assertIsNone(finite_number("0/0"))
        self.assertIsNone(finite_number("nan"))
        with self.assertRaises(ValueError):
            ffprobe_metadata({"streams": [{"nb_frames": "300", "avg_frame_rate": "30/1"}]})

    def test_pyav_header_probe_reads_only_first_frame_if_origin_unknown(self):
        from fractions import Fraction
        stream = SimpleNamespace(duration=None, time_base=Fraction(1, 1000), start_time=None,
                                 average_rate=Fraction(24000, 1001), base_rate=Fraction(24),
                                 frames=0, codec_context=SimpleNamespace(width=640, height=480, name="h264"))
        visited = []
        def decode(s):
            visited.append(1)
            yield SimpleNamespace(time=2.5)
            raise AssertionError("Header probe must not decode a second frame")
        container = SimpleNamespace(streams=SimpleNamespace(video=[stream]), duration=12000000, decode=decode)
        class Context:
            def __enter__(self): return container
            def __exit__(self, *args): return False
        with patch.dict(sys.modules, {"av": SimpleNamespace(open=lambda path: Context(), time_base=1000000)}):
            result = pyav_metadata("video.mp4")
        self.assertEqual(result["duration_seconds"], 12)
        self.assertEqual(result["start_source"], "first_decoded_frame")
        self.assertEqual(result["stream_start_seconds"], 2.5)
        self.assertEqual(visited, [1])

    def test_inventory_bundle_is_read_without_extracting_members(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as name:
            path = Path(name) / "metadata.tar.gz"
            with tarfile.open(path, "w:gz") as archive:
                content = b"dataset,rel_path,bytes\nHDTF,HDTF/a.mp4,3\n"
                info = tarfile.TarInfo("nested/video_inventory.csv")
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
                info = tarfile.TarInfo("../../never_extract.txt")
                info.size = 3
                archive.addfile(info, io.BytesIO(b"bad"))
            self.assertEqual(inventory_rows(path)[0]["rel_path"], "HDTF/a.mp4")
            self.assertEqual(list(Path(name).iterdir()), [path])
            for rel in ("../a.mp4", "C:/a.mp4", "a\\b.mp4", "a//b.mp4"):
                csv_path = Path(name) / "invalid.csv"
                csv_path.write_text("dataset,rel_path\nHDTF," + rel + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    inventory_rows(csv_path)

    def test_probe_errors_are_recorded_and_resume_reprobes_changed_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as name:
            root = Path(name)
            (root / "a.mp4").write_bytes(b"a")
            (root / "b.mp4").write_bytes(b"b")
            inventory = root / "inventory.csv"
            inventory.write_text("dataset,rel_path\nHDTF,a.mp4\nHDTF,b.mp4\n", encoding="utf-8")
            out = root / "audit"
            def fake_probe(path, backend, timeout):
                if path.name == "b.mp4" and not getattr(fake_probe, "fixed", False):
                    raise subprocess.TimeoutExpired("fake", timeout)
                return {"duration_seconds": 12., "duration_source": "video_stream",
                        "stream_start_seconds": 0., "fps_average": 25.}
            with redirect_stdout(io.StringIO()), \
                    patch("audit_video_durations.subprocess.run", return_value=SimpleNamespace(stdout=b"ffprobe fake\n")), \
                    patch("audit_video_durations.probe", side_effect=fake_probe) as probing:
                first = audit(inventory, root, out, backend="ffprobe")
                self.assertEqual(first["errors"], 1)
                fake_probe.fixed = True
                second = audit(inventory, root, out, backend="ffprobe", resume=True)
                self.assertEqual(second["reused"], 1)
                self.assertEqual(second["errors"], 0)
                self.assertEqual(probing.call_count, 3)
                # File content/stat changed: stale cached metadata cannot be reused.
                (root / "a.mp4").write_bytes(b"changed")
                third = audit(inventory, root, out, backend="ffprobe", resume=True)
                self.assertEqual(third["reused"], 1)
                self.assertEqual(probing.call_count, 4)
                # Interrupted final journal line is retried safely, not joined to new JSON.
                with (out / "progress.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write('{"partial":')
                audit(inventory, root, out, backend="ffprobe", resume=True)
                for line in (out / "progress.jsonl").read_text(encoding="utf-8").splitlines():
                    json.loads(line)
                inventory.write_text("dataset,rel_path\nHDTF,a.mp4\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    audit(inventory, root, out, backend="ffprobe", resume=True)

    def test_inventory_size_mismatch_is_an_error_not_a_missing_success(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as name:
            root = Path(name)
            (root / "a.mp4").write_bytes(b"a")
            result = probe_row(root, {"dataset": "HDTF", "rel_path": "a.mp4", "bytes": "2"}, "pyav", 1)
            self.assertEqual(result["status"], "error")
            self.assertIn("size differs", result["error"])


class WindowTests(unittest.TestCase):
    def test_interior_windows_are_centered_nonoverlapping_and_avoid_endpoints_when_possible(self):
        self.assertEqual(interior_windows(100, 10, 4), [(15000000, 25000000), (35000000, 45000000),
                                                       (55000000, 65000000), (75000000, 85000000)])
        self.assertEqual(interior_windows(20, 10, 4), [(5000000, 15000000)])
        self.assertEqual(interior_windows(15, 10, 4), [(2500000, 12500000)])
        self.assertEqual(interior_windows(5, 10, 4), [(0, 5000000)])
        self.assertEqual(interior_windows(1, 10, 4), [])
        with self.assertRaises(ValueError):
            interior_windows(100, 10, 0)
        for duration in (2, 9.999999, 10, 10.01001, 20, 29.999, 50, 1234.56789):
            windows = interior_windows(duration, 10, 4)
            self.assertLessEqual(len(windows), 4)
            self.assertTrue(all(0 <= a < b <= int(duration * 1000000 + .01) for a, b in windows))
            self.assertTrue(all(a[1] <= b[0] for a, b in zip(windows, windows[1:])))

    def test_changing_placement_preserves_parent_selection_and_roles(self):
        parents, durations = [], []
        for role in ("train", "calibration", "dev", "test"):
            for index in range(6):
                path = f"HDTF/{role}_{index}.mp4"
                parents.append(dict(dataset="HDTF", rel_path=path, split=role, label=0, source_clip=f"{role}_{index}"))
                durations.append(dict(dataset="HDTF", rel_path=path, status="ok", duration_seconds=100,
                                      stream_start_seconds=0, bytes=1, mtime_ns=1, duration_source="video_stream"))
        old = make_pilot(parents, durations)[0]
        new = make_pilot(parents, durations, clips_per_parent=4, placement="interior_v1")[0]
        self.assertEqual({(r["parent_video_key"], r["split"]) for r in old},
                         {(r["parent_video_key"], r["split"]) for r in new})
        self.assertEqual(len(new), 24)
        self.assertNotIn("test", {r["split"] for r in new})
        self.assertTrue(all(0 < r["segment_start_seconds"] < r["segment_end_seconds"] < 100 for r in new))
        self.assertEqual(new, make_pilot(list(reversed(parents)), durations,
                                        clips_per_parent=4, placement="interior_v1")[0])

    def test_coverage_distinguishes_sustained_faces_from_short_concentrated_detection(self):
        timestamps = [.05 + index * .1 for index in range(100)]
        concentrated = temporal_face_coverage(timestamps, [i < 50 for i in range(100)], 0, 10)
        spread = temporal_face_coverage(timestamps, [i % 2 == 0 for i in range(100)], 0, 10)
        self.assertEqual(concentrated["detected_fraction"], .5)
        self.assertEqual(spread["detected_fraction"], .5)
        self.assertEqual(spread["good_bins"], 5)
        self.assertFalse(sustained_candidate(32, concentrated))
        self.assertTrue(sustained_candidate(32, spread))
        self.assertFalse(sustained_candidate(1, spread))
        empty = temporal_face_coverage([], [], 0, 10)
        self.assertFalse(sustained_candidate(0, empty))

    def test_coverage_uses_actual_nonzero_pts_and_rejects_unknown_or_outside_timing(self):
        result = temporal_face_coverage([5.1, 6.1, 7.1, 8.1, 9.1], [True] * 5, 5, 10)
        self.assertEqual(result["detected_frames_by_bin"], [1] * 5)
        self.assertEqual(result["good_bins"], 5)
        for times in ([float("nan")], [10], [4.9]):
            with self.assertRaises(ValueError):
                temporal_face_coverage(times, [True], 5, 10)
        with self.assertRaises(ValueError):
            temporal_face_coverage([5.1], [], 5, 10)

    def test_capped_timeline_windows_short_tails_and_invalid_lengths(self):
        self.assertEqual(bounded_windows(102, 10, 3), [(0, 10000000), (50000000, 60000000), (100000000, 102000000)])
        self.assertEqual(bounded_windows(21, 10, 20), [(0, 10000000), (10000000, 20000000)])
        self.assertEqual(bounded_windows(5, 10, 2), [(0, 5000000)])
        self.assertEqual(bounded_windows(1, 10, 2), [])
        for args in ((10, 0, 2), (10, 10, 0), (float("nan"), 10, 2), (10, 1, 2)):
            with self.assertRaises(ValueError):
                bounded_windows(*args)

    def test_actual_pts_nonzero_origin_and_half_open_boundaries(self):
        from fractions import Fraction
        frames = [SimpleNamespace(time=timestamp, to_ndarray=lambda format: "image")
                  for timestamp in (6.8, 7., 7.04, 7.19, 7.5, 8.)]
        stream = SimpleNamespace(time_base=Fraction(1, 1000), thread_count=0)
        container = SimpleNamespace(streams=SimpleNamespace(video=[stream]), decode=lambda s: iter(frames))
        calls = []
        container.seek = lambda *args, **kwargs: calls.append((args, kwargs))
        class Context:
            def __enter__(self): return container
            def __exit__(self, *args): return False
        fake_av = SimpleNamespace(open=lambda path: Context())
        with patch.dict(sys.modules, {"av": fake_av}):
            actual = list(iter_window_frames("video.mp4", 2, 3, 5))
            self.assertEqual([row[0] for row in actual], [0, 1, 2, 3])
            self.assertEqual([row[2] for row in actual], [7., 7.04, 7.19, 7.5])
            self.assertEqual(calls[0][0], (7000,))
            self.assertFalse(calls[0][1]["any_frame"])
            self.assertTrue(calls[0][1]["backward"])
            frames[2].time = None
            with self.assertRaises(ValueError):
                list(iter_window_frames("video.mp4", 2, 3, 5))

    def test_source_caps_role_inheritance_and_order_independent_ids(self):
        parents, durations = [], []
        for role in ("train", "calibration", "dev", "test"):
            for group in range(5):
                for original in range(3):
                    path = f"HDTF/{role}_{group}_{original}.mp4"
                    parents.append({"dataset": "HDTF", "rel_path": path, "split": role,
                                    "label": 0, "source_clip": f"{role}_{group}"})
                    durations.append({"dataset": "HDTF", "rel_path": path, "status": "ok",
                                      "duration_seconds": 120, "duration_source": "video_stream",
                                      "stream_start_seconds": 1.5, "bytes": 10, "mtime_ns": 99})
        rows, _ = make_pilot(parents, durations, clips_per_parent=2, groups_per_stratum=2)
        self.assertEqual(rows, make_pilot(list(reversed(parents)), list(reversed(durations)))[0])
        self.assertEqual(len(rows), 12)  # 3 development roles x 2 groups x 2 windows.
        self.assertNotIn("test", {r["split"] for r in rows})
        self.assertEqual(len({r["parent_video_key"] for r in rows}), 6)
        self.assertEqual(len({r["video_key"] for r in rows}), len(rows))
        self.assertEqual(len({r["source_clip"] for r in rows}), 6)
        self.assertEqual(window_spec(rows[0])["source_mtime_ns"], 99)
        unknown = [dict(r, stream_start_seconds=None) for r in durations]
        with self.assertRaises(ValueError):
            make_pilot(parents, unknown)

    def test_reused_pilot_records_require_identical_bounds_contract_and_source_stat(self):
        from preprocess_videos import process
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as name:
            root = Path(name)
            (root / "a.mp4").write_bytes(b"a")
            stat = (root / "a.mp4").stat()
            row = SimpleNamespace(sample_unit="segment_pilot", clip_id="a", parent_video_key="parent",
                                  segment_start_seconds=0, segment_end_seconds=10, stream_start_seconds=0,
                                  source_bytes=stat.st_size, source_mtime_ns=stat.st_mtime_ns,
                                  video_key="external/real_external/a", rel_path="a.mp4")
            args = SimpleNamespace(video_root=str(root), records=str(root / "records"), pilot_contract_sha256="contract")
            record = root / "records/external/real_external/a.json"
            record.parent.mkdir(parents=True)
            record.write_text(json.dumps({**window_spec(row), "pilot_contract_sha256": "contract", "status": "ok"}), encoding="utf-8")
            with patch.dict(sys.modules, {"av": SimpleNamespace()}):
                self.assertEqual(process(row, args, None), "skip")
                row.segment_end_seconds = 9
                with self.assertRaises(ValueError):
                    process(row, args, None)
                row.segment_end_seconds = 10
                (root / "a.mp4").write_bytes(b"different")
                with self.assertRaises(ValueError):
                    process(row, args, None)

    def test_pilot_summary_counts_originals_and_excludes_reused_timing(self):
        import pandas as pd
        rows = pd.DataFrame([{"video_key": "a", "parent_video_key": "parent", "source_clip": "group",
                              "split": "train", "dataset": "HDTF", "label": 0},
                             {"video_key": "b", "parent_video_key": "parent", "source_clip": "group",
                              "split": "train", "dataset": "HDTF", "label": 0}])
        measurements = [{"video_key": "a", "status": "ok", "seconds": 9, "window_seconds": 10, "reused": True},
                        {"video_key": "b", "status": "no_face", "seconds": 7, "window_seconds": 10, "reused": False}]
        group = summarize(rows, measurements, 1, 9)["groups"][0]
        self.assertEqual(group["original_videos"], 1)
        self.assertEqual(group["source_groups"], 1)
        self.assertEqual(group["fresh_processing_seconds"], 7)
        self.assertEqual(group["no_face_clips"], 1)

    def test_pilot_cannot_silently_enter_whole_video_training(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as name:
            root = Path(name)
            manifest = root / "clips.csv"
            manifest.write_text("sample_unit,clip_id\nsegment_pilot,a\n", encoding="utf-8")
            selection = root / "selection.json"
            selection.write_text(json.dumps({"step": 2, "protocol": "celeb_identity", "ablation_run": "ln_b128_s42",
                                             "checkpoint_sha256": "a", "ablation_plan_sha256": "b"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a registered"):
                prepare(manifest, root, selection, root / "registered", "source_only", "audit")
            self.assertFalse((root / "registered").exists())

    def test_duration_to_registered_pilot_and_no_test_decode(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as name:
            root = Path(name)
            parents = [dict(dataset=dataset, rel_path=f"{dataset}/{role}_{label}.mp4", split=role,
                            label=label, source_clip=f"{dataset}_{role}_{label}", method="real" if not label else "fake")
                       for role, dataset, labels in (("train", "HDTF", [0]), ("calibration", "HDTF", [0]),
                                                     ("dev", "DFDC-Kaggle", [0, 1]),
                                                     ("test", "PolyGlotFake", [0, 1])) for label in labels]
            write_csv(root / "manifest.csv", parents)
            durations = [dict(dataset=r["dataset"], rel_path=r["rel_path"], status="ok", duration_seconds=21,
                              stream_start_seconds=0, duration_source="video_stream", bytes=1, mtime_ns=99)
                         for r in parents]
            audit_dir = root / "audit"
            audit_dir.mkdir()
            write_csv(audit_dir / "durations.csv", durations)
            write_csv(audit_dir / "errors.csv", [])
            atomic_text(audit_dir / "summary.json", json.dumps({"errors": 0}))
            atomic_text(audit_dir / "audit_contract.json", "{}")
            atomic_text(audit_dir / "SHA256SUMS.txt", "".join(
                f"{digest(audit_dir / p)}  {p}\n" for p in
                ("durations.csv", "summary.json", "audit_contract.json", "errors.csv")))
            with redirect_stdout(io.StringIO()):
                prepare_pilot(root / "manifest.csv", audit_dir, root / "pilot")
            plan, rows = load_pilot(root / "pilot/segment_pilot.json")
            self.assertEqual(len(rows), 8)
            self.assertNotIn("test", set(rows["split"]))
            self.assertEqual(sum(r["original_videos"] for r in plan["counts"]), 4)
            self.assertTrue(all(r["clips"] <= 2 * r["source_groups"] for r in plan["counts"]))
            with (root / "pilot/pilot_videos.csv").open("a", encoding="utf-8") as file:
                file.write("changed")
            with self.assertRaisesRegex(ValueError, "table changed"):
                load_pilot(root / "pilot/segment_pilot.json")

    def test_window_detection_crop_and_record_use_the_same_pts(self):
        import numpy as np
        import preprocess_videos as prep
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as name:
            root = Path(name)
            (root / "a.mp4").write_bytes(b"a")
            stat = (root / "a.mp4").stat()
            row = SimpleNamespace(sample_unit="segment_pilot", clip_id="a", parent_video_key="parent",
                                  segment_start_seconds=0, segment_end_seconds=10, stream_start_seconds=3,
                                  source_bytes=stat.st_size, source_mtime_ns=stat.st_mtime_ns,
                                  video_key="external/real_external/a", rel_path="a.mp4", split="train",
                                  label=0, frame_subset="uniform32")
            args = SimpleNamespace(video_root=str(root), records=str(root / "records"),
                                   crops=str(root / "crops"), detections=str(root / "detections"),
                                   pilot_contract_sha256="contract")
            def timed(path, window=None):
                self.assertEqual(window["segment_end_seconds"], 10)
                for index in range(40):
                    yield index, np.zeros((2, 2, 3), np.uint8), 3 + index * .2
            model = SimpleNamespace(detect=lambda frame: (np.array([[0, 0, 1, 1, .9]]), np.zeros((1, 5, 2))))
            def imwrite(path, face):
                Path(path).write_bytes(b"png")
                return True
            fake_av = SimpleNamespace(error=SimpleNamespace(FFmpegError=type("DecodeError", (Exception,), {})))
            with patch.dict(sys.modules, {"av": fake_av, "cv2": SimpleNamespace(imwrite=imwrite),
                                         "detector": SimpleNamespace(align_face=lambda image, *args, **kw: (image, None))}), \
                    patch.object(prep, "iter_timed_frames", side_effect=timed), \
                    patch.object(prep, "add_gend_to_path"), patch.object(prep, "upstream_commit", return_value="fake"):
                self.assertEqual(prep.process(row, args, model), "ok")
                record = json.loads((root / "records/external/real_external/a.json").read_text(encoding="utf-8"))
                self.assertEqual(record["total_frames"], 40)
                self.assertEqual(len(record["selected"]), 32)
                self.assertEqual(record["selected_timestamps_seconds"], [3 + i * .2 for i in record["selected"]])
                self.assertIn("not a global", record["frame_index_semantics"])
                self.assertEqual(len(list((root / "crops/external/real_external/a").glob("*.png"))), 32)


if __name__ == "__main__":
    unittest.main()
