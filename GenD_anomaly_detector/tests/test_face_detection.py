"""Fallback geometry, provenance and original-image crop checks; no models required."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from gad.face_detection import detect_with_policy, policy_spec

EMPTY = (np.empty((0, 5), np.float32), np.empty((0, 5, 2), np.float32))
POINTS = np.array([[[2, 2], [8, 2], [5, 5], [2, 7], [8, 7]]], np.float32)
BOX = np.array([[1, 1, 10, 9, .9]], np.float32)


class DetectionTests(unittest.TestCase):
    def test_default_none_keeps_existing_no_face_behavior(self):
        frame = np.ones((11, 13, 3), np.uint8)
        model = SimpleNamespace(detect=Mock(return_value=EMPTY))
        boxes, points, audit = detect_with_policy(model, frame)
        self.assertIs(boxes, EMPTY[0])
        self.assertIs(points, EMPTY[1])
        self.assertEqual(model.detect.call_count, 1)
        self.assertFalse(audit["fallback_attempted"])

    def test_existing_positive_is_returned_unchanged_without_second_pass(self):
        frame = np.ones((11, 13, 3), np.uint8)
        model = SimpleNamespace(detect=Mock(return_value=(BOX, POINTS)))
        boxes, points, audit = detect_with_policy(model, frame, "pad_on_empty_v1")
        self.assertIs(boxes, BOX)
        self.assertIs(points, POINTS)
        model.detect.assert_called_once_with(frame)
        self.assertFalse(audit["fallback_used"])

    def test_odd_size_padding_maps_boxes_and_landmarks_back_without_mutation(self):
        frame = np.full((11, 13, 3), 7, np.uint8)
        padded_box, padded_points = BOX.copy(), POINTS.copy()
        padded_box[:, [0, 2]] += 6
        padded_box[:, [1, 3]] += 5
        padded_points += [6, 5]
        def detect(image):
            if image.shape == frame.shape:
                return EMPTY
            self.assertEqual(image.shape, (21, 25, 3))
            np.testing.assert_array_equal(image[5:16, 6:19], frame)
            self.assertFalse(image[:5].any())
            self.assertFalse(image[:, :6].any())
            return padded_box, padded_points
        boxes, points, audit = detect_with_policy(SimpleNamespace(detect=detect), frame, "pad_on_empty_v1")
        np.testing.assert_array_equal(boxes, BOX)
        np.testing.assert_array_equal(points, POINTS)
        np.testing.assert_array_equal(padded_points, POINTS + [6, 5])
        self.assertTrue(audit["fallback_attempted"])
        self.assertTrue(audit["fallback_used"])
        self.assertEqual(audit["fallback_rejected"], 0)

    def test_candidates_on_padding_or_with_bad_geometry_are_rejected(self):
        boxes = np.repeat(BOX, 5, axis=0)
        points = np.repeat(POINTS, 5, axis=0)
        boxes[:, [0, 2]] += 6
        boxes[:, [1, 3]] += 5
        points += [6, 5]
        boxes[1] = [0, 0, 2, 2, .99]  # Center lies entirely on padding.
        points[2, 0, 0] = 5  # One landmark outside original image after mapping.
        points[3, 1, 1] = np.nan
        boxes[4, 2] = boxes[4, 0]  # Zero width.
        model = SimpleNamespace(detect=Mock(side_effect=[EMPTY, (boxes, points)]))
        actual, kps, audit = detect_with_policy(model, np.ones((11, 13, 3)), "pad_on_empty_v1")
        np.testing.assert_array_equal(actual, BOX)
        np.testing.assert_array_equal(kps, POINTS)
        self.assertEqual(audit["fallback_rejected"], 4)

    def test_no_padding_detection_remains_no_face(self):
        model = SimpleNamespace(detect=Mock(side_effect=[EMPTY, EMPTY]))
        boxes, points, audit = detect_with_policy(model, np.ones((11, 13, 3)), "pad_on_empty_v1")
        self.assertEqual(len(boxes), 0)
        self.assertEqual(points.shape, (0, 5, 2))
        self.assertTrue(audit["fallback_attempted"])
        self.assertFalse(audit["fallback_used"])

    def test_fallback_requires_five_point_landmarks(self):
        model = SimpleNamespace(detect=Mock(side_effect=[EMPTY, (BOX, None)]))
        with self.assertRaisesRegex(ValueError, "five-point"):
            detect_with_policy(model, np.ones((11, 13, 3)), "pad_on_empty_v1")

    def test_unknown_policy_does_not_run_detector(self):
        model = SimpleNamespace(detect=Mock())
        with self.assertRaises(ValueError):
            detect_with_policy(model, np.ones((11, 13, 3)), "unregistered")
        model.detect.assert_not_called()

    def test_crops_use_original_images_and_remapped_points_and_record_recovery(self):
        import preprocess_videos as prep
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as name:
            root = Path(name)
            (root / "video.mp4").write_bytes(b"video")
            stat = (root / "video.mp4").stat()
            row = SimpleNamespace(sample_unit="segment_pilot", clip_id="a", parent_video_key="parent",
                                  segment_start_seconds=0, segment_end_seconds=1, stream_start_seconds=0,
                                  source_bytes=stat.st_size, source_mtime_ns=stat.st_mtime_ns,
                                  video_key="external/real_external/a", rel_path="video.mp4", split="train",
                                  label=0, frame_subset="uniform32")
            args = SimpleNamespace(video_root=str(root), records=str(root / "records"),
                                   crops=str(root / "crops"), detections=str(root / "detections"),
                                   pilot_contract_sha256="same-plan-new-run", detection_policy="pad_on_empty_v1")
            padded_box = BOX + [6, 5, 6, 5, 0]
            def detect(image):
                return EMPTY if image.shape == (11, 13, 3) else (padded_box.copy(), POINTS + [6, 5])
            def frames(*args, **kwargs):
                for index in range(4):
                    yield index, np.full((11, 13, 3), 7, np.uint8), .1 + index * .2
            aligned = []
            def align(image, points, **kwargs):
                self.assertEqual(image.shape, (11, 13, 3))
                self.assertTrue((image == 7).all())
                np.testing.assert_array_equal(points, POINTS[0])
                aligned.append(True)
                return image, None
            def write(path, image):
                Path(path).write_bytes(b"crop")
                return True
            fake_av = SimpleNamespace(error=SimpleNamespace(FFmpegError=type("DecodeError", (Exception,), {})))
            with patch.dict(sys.modules, {"av": fake_av, "cv2": SimpleNamespace(imwrite=write),
                                         "detector": SimpleNamespace(align_face=align)}), \
                    patch.object(prep, "iter_timed_frames", side_effect=frames), \
                    patch.object(prep, "add_gend_to_path"), patch.object(prep, "upstream_commit", return_value="fake"):
                self.assertEqual(prep.process(row, args, SimpleNamespace(detect=detect)), "ok")
                record = json.loads((root / "records/external/real_external/a.json").read_text(encoding="utf-8"))
                self.assertEqual(record["fallback_attempted_frames"], 4)
                self.assertEqual(record["fallback_recovered_frames"], 4)
                self.assertEqual(record["selected_fallback_frames"], 4)
                self.assertEqual(record["detected_fraction"], 1.)
                self.assertEqual(len(aligned), 4)
                with np.load(root / "detections/external/real_external/a.npz") as cache:
                    np.testing.assert_array_equal(cache["landmarks"], np.repeat(POINTS, 4, axis=0))
                    self.assertTrue(cache["fallback_used"].all())
                args.detection_policy = "none"
                with self.assertRaisesRegex(ValueError, "new preprocessing directory"):
                    prep.process(row, args, SimpleNamespace(detect=detect))


if __name__ == "__main__":
    unittest.main()
