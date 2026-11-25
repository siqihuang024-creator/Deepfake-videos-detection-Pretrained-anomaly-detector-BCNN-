"""Paired validation orchestration checks without loading models or test data."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from compare_validation import compare
from gad.feature_cache import sha256_file


class PairedValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "tests")
        self.experiment = Path(self.temp.name)
        for folder in ("runs", "lists", "tables"):
            (self.experiment / folder).mkdir()
        self.checkpoint = self.experiment / "runs/locked.ckpt"
        self.checkpoint.write_bytes(b"previously-selected-weights")
        (self.experiment / "tables/videos.csv").write_text("unchanged table", encoding="utf-8")
        (self.experiment / "lists/val.txt").write_text("unchanged validation crops", encoding="utf-8")
        plan = self.experiment / "ablation_plan.json"
        plan.write_text(json.dumps({"input_sha256": {"tables/videos.csv": sha256_file(self.experiment / "tables/videos.csv")}}), encoding="utf-8")
        self.selection = self.experiment / "runs/ablation_selection.json"
        self.selection.write_text(json.dumps({"step": 2, "protocol": "celeb_identity", "ablation_run": "ln_b128_s42",
            "checkpoint": str(self.checkpoint), "checkpoint_sha256": sha256_file(self.checkpoint),
            "val_frame_list_sha256": sha256_file(self.experiment / "lists/val.txt"),
            "ablation_plan_sha256": sha256_file(plan), "best_val_auroc_video": .9744}), encoding="utf-8")
        self.commands, self.predictions = [], {}
        self.out = self.experiment / "paired"

    def tearDown(self):
        self.temp.cleanup()

    def fake_run(self, command, check):
        self.assertTrue(check)
        self.commands.append(command)
        self.assertNotIn("--test-pred", command)
        options = dict(zip(command[2::2], command[3::2]))
        if Path(command[1]).name == "predict_frames.py":
            self.predictions[options["--out"]] = options
        elif Path(command[1]).name == "evaluate.py":
            settings = self.predictions[options["--val-pred"]]
            meta = {"model": settings["--model"], "precision": settings["--precision"],
                    "batch_size": int(settings["--batch-size"]), "num_workers": int(settings["--num-workers"]),
                    "frame_list_sha256": sha256_file(Path(settings["--frame-list"])),
                    "checkpoint_sha256": sha256_file(self.checkpoint) if settings["--model"] != "official" else None}
            result = {"val_pred_meta": meta, "val": {"auroc": .96 if meta["model"] == "official" else .975},
                      "val_coverage": {"listed": 4, "scored": 4, "unscorable": []}}
            Path(options["--out"]).write_text(json.dumps(result), encoding="utf-8")
        else:
            self.fail("Unexpected command: " + str(command))

    def test_same_inference_path_reports_paired_difference_without_reselection(self):
        original = sha256_file(self.selection)
        with patch("compare_validation.subprocess.run", side_effect=self.fake_run):
            result = compare(self.experiment, self.out, batch_size=128, num_workers=6, precision="bf16")
        self.assertAlmostEqual(result["finetuned_minus_official_auroc"], .015)
        self.assertAlmostEqual(result["training_time_selected_val_auroc_reference"], .9744)
        self.assertEqual(sha256_file(self.selection), original)
        self.assertEqual([Path(c[1]).name for c in self.commands],
                         ["predict_frames.py", "evaluate.py", "predict_frames.py", "evaluate.py"])
        self.assertEqual(self.commands[0][self.commands[0].index("--frame-list") + 1],
                         self.commands[2][self.commands[2].index("--frame-list") + 1])
        with self.assertRaises(FileExistsError):
            compare(self.experiment, self.out)

    def test_changed_validation_or_checkpoint_rejected_before_inference(self):
        for path in (self.experiment / "lists/val.txt", self.checkpoint):
            original = path.read_bytes()
            path.write_bytes(b"changed")
            with patch("compare_validation.subprocess.run") as runner:
                with self.assertRaises(ValueError):
                    compare(self.experiment, self.out)
                runner.assert_not_called()
            self.assertFalse(self.out.exists())
            path.write_bytes(original)

    def test_mismatched_report_precision_does_not_publish_a_comparison(self):
        def wrong_precision(command, check):
            self.fake_run(command, check)
            if Path(command[1]).name == "evaluate.py":
                path = Path(command[command.index("--out") + 1])
                report = json.loads(path.read_text(encoding="utf-8"))
                report["val_pred_meta"]["precision"] = "fp32"
                path.write_text(json.dumps(report), encoding="utf-8")
        with patch("compare_validation.subprocess.run", side_effect=wrong_precision):
            with self.assertRaises(ValueError):
                compare(self.experiment, self.out)
        self.assertFalse((self.out / "comparison.json").exists())


if __name__ == "__main__":
    unittest.main()
