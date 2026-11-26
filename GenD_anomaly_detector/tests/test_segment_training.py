"""Development selection, provenance rejection and a real CPU head round trip."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gad import segment_training as pipeline
from gad.feature_cache import FrameCacheWriter, sha256_file


def candidate_rows():
    values = []
    for role, groups, clips in (("train", 6, 3), ("calibration", 3, 3), ("dev", 2, 2)):
        for group in range(groups):
            for clip in range(clips):
                label = clip if role == "dev" else 0
                key = "external/%s/%s_%s_%s" % ("fake_external" if label else "real_external", role, group, clip)
                values.append(dict(split=role, source_clip=role + str(group), video_key=key,
                                   parent_video_key=role + str(group) + (str(clip) if role == "dev" else ""),
                                   dataset="DFDC-Kaggle" if role == "dev" else "HDTF",
                                   label=label, method="fake" if label else "real", rel_path=key + ".mp4",
                                   target_id="", donor_id="", driver_id=""))
    return pd.DataFrame(values)


class SegmentTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "tests")
        self.root = Path(self.temp.name)
        self.rows = candidate_rows()

    def tearDown(self):
        self.temp.cleanup()

    def choose(self, rows=None, low=None):
        return pipeline.choose_units(self.rows if rows is None else rows, set(self.rows.video_key), set(low or []))

    def test_capped_selection_is_reorder_stable_and_disjoint(self):
        a, coverage, split = self.choose()
        b, _, _ = self.choose(self.rows.sample(frac=1, random_state=5))
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(len(a.loc[a.split.eq("train")]), 12)
        self.assertEqual(len(a.loc[a.split.eq("calibration")]), 3)
        self.assertEqual(len(a.loc[a.split.eq("dev")]), 4)
        fit = set(a.loc[a.head_role.eq("gradient_fit"), "source_clip"])
        early = set(a.loc[a.head_role.eq("early_stop"), "source_clip"])
        self.assertFalse(fit & early)
        self.assertEqual((len(fit), len(early)), (4, 2))
        self.assertEqual(split["training_groups"], 4)

    def test_low_coverage_excludes_A_only_and_dropped_sources_stay_in_coverage(self):
        excluded_source = self.rows.loc[self.rows.source_clip.eq("train0"), "video_key"].tolist()
        low_dev = self.rows.loc[self.rows.split.eq("dev"), "video_key"].iloc[0]
        units, coverage, _ = self.choose(low=excluded_source + [low_dev])
        self.assertNotIn("train0", set(units.source_clip))
        self.assertIn(low_dev, set(units.video_key))
        group = next(g for g in coverage if g["split"] == "train")
        self.assertEqual(group["sources_without_sustained_window"], 1)
        self.assertEqual(group["source_groups"], 6)

    def test_dev_labels_do_not_change_training_and_invalid_roles_rejected(self):
        units, _, _ = self.choose()
        rows = self.rows.copy()
        rows.loc[rows.split.eq("dev"), "label"] = 1 - rows.loc[rows.split.eq("dev"), "label"]
        changed, _, _ = self.choose(rows)
        self.assertEqual(units.loc[units.split.ne("dev"), "video_key"].tolist(),
                         changed.loc[changed.split.ne("dev"), "video_key"].tolist())
        rows.loc[0, "split"] = "test"
        with self.assertRaises(ValueError):
            self.choose(rows)
        rows = self.rows.copy()
        rows.loc[rows.split.eq("calibration"), "source_clip"] = "train0"
        with self.assertRaises(ValueError):
            self.choose(rows)
        rows = self.rows.copy()
        rows.loc[0, "dataset"] = "CelebDFv3"
        with self.assertRaisesRegex(ValueError, "A dataset"):
            self.choose(rows)

    def make_inputs(self):
        data, cache_run = self.root / "data", self.root / "caches"
        data.mkdir()
        cache_run.mkdir()
        self.rows.to_csv(data / "candidate_videos.csv", index=False)
        names = ("parents.csv", "selected_originals.csv", "reserved_test.csv", "unused_development.csv", "candidate_videos.csv")
        for name in names[:-1]:
            (data / name).write_text("registered metadata", encoding="utf-8")
        plan = dict(version=1, purpose="registered bounded candidate preprocessing; no training or final testing",
                    table="candidate_videos.csv", candidate_clips=len(self.rows),
                    registered_files={name: sha256_file(data / name) for name in names})
        plan_path = data / "segment_data_plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        contract = data / "faces/pilot_run_contract.json"
        contract.parent.mkdir()
        contract.write_text("{}", encoding="utf-8")
        records = {}
        for row in self.rows.itertuples(index=False):
            path = data / "faces/records" / (row.video_key + ".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(status="ok", selected=list(range(32)), sustained_face_candidate=True)), encoding="utf-8")
            records[row.video_key] = sha256_file(path)
        paths = [str(self.root / "crops" / key / ("frame_%05d.png" % i)) for key in self.rows.video_key for i in range(32)]
        listing = cache_run / "candidates.txt"
        listing.write_text("\n".join(paths) + "\n", encoding="utf-8")
        (cache_run / "frame_lists.json").write_text(json.dumps(dict(
            plan_sha256=sha256_file(plan_path), files={"candidates.txt": sha256_file(listing)},
            preprocessing_contract_sha256=sha256_file(contract), records_sha256=records,
            low_coverage_included_keys=[], unscorable_no_face_keys=[])), encoding="utf-8")
        checkpoint = self.root / "checkpoint"
        checkpoint.write_bytes(b"synthetic checkpoint")
        selection = self.root / "selection.json"
        selection.write_text(json.dumps(dict(ablation_run="ln_b128_s42", best_epoch=21,
                              checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint))), encoding="utf-8")
        rng = np.random.default_rng(2)
        raw = rng.normal(size=(len(paths), 1024)).astype(np.float32)
        raw /= np.linalg.norm(raw, axis=1, keepdims=True)
        probs = rng.uniform(size=len(paths)).astype(np.float32)
        for name in ("official", "finetuned"):
            destination = self.root / (name + "_features")
            signature = dict(model="official" if name == "official" else str(checkpoint),
                checkpoint_sha256=None if name == "official" else sha256_file(checkpoint),
                diagnostic_plan_sha256=sha256_file(plan_path), frame_list_sha256=sha256_file(listing),
                precision="bf16", frame_times_sha256="same", crop_stat_sha256="same", gend_commit="same", cache_algorithm="same")
            writer = FrameCacheWriter(destination, paths, signature)
            writer.write(np.arange(len(paths)), raw, np.ones(len(paths), np.float32), probs)
            writer.finish()
            receipts = cache_run / "cache_receipts"
            receipts.mkdir(exist_ok=True)
            (receipts / (name + ".json")).write_text(json.dumps(dict(cache=str(destination),
                metadata_sha256=sha256_file(destination / "metadata.json"))), encoding="utf-8")
        return data, cache_run, selection

    def test_real_cache_provenance_and_tampering_checks(self):
        inputs = self.make_inputs()
        units, caches, _ = pipeline.load_inputs(*inputs)
        self.assertEqual(len(units), 19)
        self.assertEqual(set(caches), {"official", "finetuned"})
        record = next((inputs[0] / "faces/records").rglob("*.json"))
        record.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "record changed"):
            pipeline.load_inputs(*inputs)

    def test_cpu_training_scores_and_locked_artifacts_round_trip(self):
        import torch
        from gad import bayesian_svdd as svdd
        torch.set_num_threads(1)
        inputs = self.make_inputs()
        out = self.root / "heads"
        with ExitStack() as stack:
            small = {"WIDTHS": (4, 2), "MAX_EPOCHS": 2, "PATIENCE": 2,
                     "BATCH_SIZE": 4, "VALIDATION_DRAWS": 3, "POSTERIOR_DRAWS": 4}
            recipe = dict(pipeline.RECIPE)
            for constant, value in small.items():
                stack.enter_context(patch.object(svdd, constant, value))
                recipe[constant.lower()] = list(value) if constant == "WIDTHS" else value
            stack.enter_context(patch.dict(pipeline.RECIPE, recipe))
            with redirect_stdout(io.StringIO()):
                pipeline.run(*inputs, out)
            self.assertTrue(json.loads((out / "completion.json").read_text())["completed"])
            config = json.loads((out / "training_registration.json").read_text())
            self.assertEqual(config["recipe"]["final_test"], "forbidden in this development entry")
            for name in ("official", "finetuned"):
                folder = out / name
                lock = json.loads((folder / "development_lock.json").read_text())
                self.assertEqual(lock["head_sha256"], sha256_file(folder / "head.npz"))
                with np.load(folder / "head.npz", allow_pickle=False) as saved:
                    self.assertEqual(int(saved["effective_train_groups"]), 4)
                    head = dict(saved)
                cache = pipeline.load_inputs(*inputs)[1][name]
                units = pd.read_csv(out / "training_units.csv", keep_default_na=False)
                dev = units.loc[units.head_role.eq("dev")]
                lookup = {k: i for i, k in enumerate(cache["video_keys"])}
                scores, _ = svdd.score(cache["features"][[lookup[k] for k in dev.video_key]], head, "cpu")
                written = pd.read_csv(folder / "development_scores.csv")
                np.testing.assert_allclose(scores, written.bayesian_deep_svdd_score, rtol=1e-5)
            with self.assertRaises(FileExistsError):
                pipeline.run(*inputs, out)


if __name__ == "__main__":
    unittest.main()
