"""Synthetic checkpoints verify the pre-test matrix lock and step-4 rule."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("identity_ablation", ROOT / "scripts/identity_ablation.py")
ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation)


class AblationSelectionChecks(unittest.TestCase):
    def make_experiment(self, root):
        root = Path(root)
        for relative in ("tables/videos.csv", "lists/train.txt", "lists/val.txt"):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative, encoding="utf-8")
        plan = ablation.prepare(root)
        for run in plan["runs"]:
            directory = root / "runs" / run["run_name"]
            directory.mkdir(parents=True)
            checkpoint = directory / "model.ckpt"
            checkpoint.write_bytes(run["run_name"].encode("ascii"))
            # Other settings deliberately have higher validation scores.
            score = 0.9 if run["setting"] == "ln_b128" else 0.99
            selection = {"step": 2, "protocol": "celeb_identity", "checkpoint": str(checkpoint),
                         "checkpoint_sha256": ablation.digest(checkpoint), "best_val_auroc_video": score}
            meta = {"finished": True, "debug": False, "args": run, "trainable": run["trainable"],
                    "batching": {"forward_batch_size": run["batch_size"], "gradient_accumulation": 1},
                    "gradient_checkpointing": False,
                    "ablation_plan_sha256": ablation.digest(root / "ablation_plan.json"),
                    "best_checkpoint_sha256": selection["checkpoint_sha256"],
                    "list_sha256": {str((root / f"lists/{split}.txt").resolve().as_posix()):
                                    plan["input_sha256"][f"lists/{split}.txt"] for split in ("train", "val")}}
            (directory / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
            (directory / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
        return root, plan

    def test_step4_chooses_only_baseline_validation_and_lowest_seed_on_ties(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan = self.make_experiment(directory)
            chosen = ablation.select(root, plan)
            self.assertEqual(chosen["ablation_run"], "ln_b128_s42")
            self.assertEqual(chosen["protocol"], "celeb_identity")
            self.assertEqual(len(json.loads((root / "matrix_selections.json").read_text())), 9)
            self.assertFalse((root / "eval").exists())

    def test_changed_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.make_experiment(directory)
            (root / "lists/train.txt").write_text("different")
            with self.assertRaises(ValueError):
                ablation.load_plan(root)

    def test_summary_reports_all_registered_seeds_without_test_based_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan = self.make_experiment(directory)
            selected = ablation.select(root, plan)
            for run in plan["runs"]:
                selection = json.loads((root / "runs" / run["run_name"] / "selection.json").read_text())
                score = 0.8 + (run["seed"] - 42) * 0.01
                report = {"video_table_sha256": plan["input_sha256"]["tables/videos.csv"],
                          "selection": selection,
                          "val_pred_meta": {"frame_list_sha256": "same-val"},
                          "test_pred_meta": {"frame_list_sha256": "same-test"},
                          "test": {"auroc": score, "per_method": {"swap": {"auroc": score}}}}
                path = root / "eval/ablations" / run["run_name"] / "report.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(report))
            summary = ablation.summarize(root, plan)
            for value in summary.values():
                self.assertEqual(len(value["test_auroc"]), 3)
                self.assertAlmostEqual(value["mean"], 0.81)
                self.assertAlmostEqual(value["stdev"], 0.01)
            self.assertEqual(json.loads((root / "runs/ablation_selection.json").read_text()), selected)

    def test_incomplete_matrix_cannot_select(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan = self.make_experiment(directory)
            path = root / "runs/full_b128_s44/run_meta.json"
            value = json.loads(path.read_text())
            value["finished"] = False
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                ablation.select(root, plan)
            self.assertFalse((root / "matrix_selections.json").exists())

    def test_modified_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan = self.make_experiment(directory)
            (root / "runs/ln_b128_s42/model.ckpt").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                ablation.select(root, plan)

    def test_submission_serializes_settings_using_success_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.make_experiment(directory)
            with patch.object(ablation.subprocess, "run", side_effect=[
                SimpleNamespace(stdout="201\n"), SimpleNamespace(stdout="202;cluster\n"),
                SimpleNamespace(stdout="203\n")
            ]) as run:
                submitted = ablation.submit(root)
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual([item["setting"] for item in submitted], ["ln_b128", "ln_b256", "full_b128"])
            self.assertIn("--array=0-2%1", commands[0])
            self.assertFalse(any("--dependency=" in arg for arg in commands[0]))
            self.assertIn("--array=3-5%1", commands[1])
            self.assertIn("--dependency=afterok:201", commands[1])
            self.assertIn("--array=6-8%1", commands[2])
            self.assertIn("--dependency=afterok:202", commands[2])

    def test_old_plan_is_rejected_without_overwriting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan = self.make_experiment(directory)
            plan["version"] = 1
            path = root / "ablation_plan.json"
            original = json.dumps(plan)
            path.write_text(original)
            with self.assertRaisesRegex(ValueError, "keep the old plan"):
                ablation.load_plan(root)
            self.assertEqual(path.read_text(), original)


if __name__ == "__main__":
    unittest.main()
