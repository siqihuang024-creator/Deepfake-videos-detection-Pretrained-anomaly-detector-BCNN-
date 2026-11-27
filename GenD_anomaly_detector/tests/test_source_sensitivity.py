"""Source restriction, unchanged development ranking, replay and output locks."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from gad import source_sensitivity as sensitivity, segment_training, bayesian_svdd as bayes
from gad.feature_cache import sha256_file, model_state_hash, signature_hash
import run_source_sensitivity as followup
import run_geometry_diagnostics as geometry_run
import run_covariance_controls as covariance_run
from run_segment_controls import run as controls_run
from test_feature_geometry import FrozenModel
import test_segment_training as fixtures


class SourceSensitivityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    @staticmethod
    def rows():
        rows = pd.DataFrame({"video_key": [str(i) for i in range(10)], "source_clip": ["g"+str(i) for i in range(10)],
            "dataset": ["HDTF"]*3+["TalkingHead-1KH"]+["HDTF"]+["HDTF","TalkingHead-1KH"]+["DFDC-Kaggle"]*3,
            "head_role": ["gradient_fit"]*4+["early_stop"]+["calibration"]*2+["dev"]*3,
            "label": [0]*8+[1]*2})
        return rows

    def test_masks_restrict_whole_sources_and_do_not_use_development_labels(self):
        rows = self.rows()
        masks = sensitivity.selection_masks(rows)
        self.assertEqual(int(masks["gradient_fit/mixed"].sum()), 4)
        self.assertEqual(int(masks["gradient_fit/hdtf_only"].sum()), 3)
        self.assertEqual(int(masks["calibration/mixed"].sum()), 2)
        self.assertEqual(int(masks["calibration/hdtf_only"].sum()), 1)
        changed = rows.copy()
        changed.loc[changed.head_role.eq("dev"), "label"] = 1-changed.loc[changed.head_role.eq("dev"), "label"]
        for name, values in sensitivity.selection_masks(changed).items():
            np.testing.assert_array_equal(values, masks[name])
        changed.loc[0, "head_role"] = "test"
        with self.assertRaises(ValueError): sensitivity.selection_masks(changed)
        changed = rows.copy(); changed.loc[5, "dataset"] = "TalkingHead-1KH"
        with self.assertRaisesRegex(ValueError, "No eligible"): sensitivity.selection_masks(changed)
        changed = rows.copy(); changed.loc[5, "source_clip"] = "g0"
        with self.assertRaises(ValueError): sensitivity.selection_masks(changed)

    def test_equal_source_family_and_shrinkages_match_direct_quadratic_scores(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(9, 5)); groups = ["a"]*3+["b"]*2+["c"]*4
        heads, counts = sensitivity.fit_family(x, groups)
        mean = np.stack([x[:3].mean(0), x[3:5].mean(0), x[5:].mean(0)]).mean(0)
        np.testing.assert_allclose(heads["centroid_l2"]["mean"], mean)
        self.assertEqual(counts["sources"], 3)
        query, probabilities = rng.normal(size=(7, 5)), np.linspace(.1, .9, 7)
        scores = sensitivity.score_family(query, heads, probabilities)
        np.testing.assert_array_equal(scores["classifier_reference"], probabilities)
        delta = query-mean
        np.testing.assert_allclose(scores["centroid_l2"], (delta*delta).sum(1))
        for alpha in (30, 50):
            name = "mahalanobis_clip_original_a"+str(alpha)
            head = heads[name]; c = head["cholesky"]@head["cholesky"].T
            np.testing.assert_allclose(scores[name], np.einsum("ij,jk,ik->i", delta, np.linalg.inv(c), delta))
            self.assertEqual(head["shrinkage"], alpha/100)

    def test_heldout_and_talkinghead_changes_do_not_change_hdtf_fit(self):
        rows = self.rows(); masks = sensitivity.selection_masks(rows)
        x = np.random.default_rng(8).normal(size=(len(rows), 5))
        fit = masks["gradient_fit/hdtf_only"]
        a, _ = sensitivity.fit_family(x[fit], rows.loc[fit, "source_clip"].tolist())
        x[~fit] += 1000
        b, _ = sensitivity.fit_family(x[fit], rows.loc[fit, "source_clip"].tolist())
        for name, head in a.items():
            for key, value in head.items(): np.testing.assert_array_equal(value, b[name][key])

    def test_full_four_cell_pipeline_replay_unchanged_auc_and_tamper_rejection(self):
        fixture = fixtures.SegmentTrainingTests(); fixture.setUp()
        self.addCleanup(fixture.tearDown)
        fixture.rows.loc[fixture.rows.source_clip.isin(("train0", "calibration0")), "dataset"] = "TalkingHead-1KH"
        inputs = fixture.make_inputs()
        cache = inputs[1]
        models = {"official": FrozenModel(7), "finetuned": FrozenModel(8)}
        for enc, model in models.items():
            receipt_path = cache / "cache_receipts" / (enc+".json")
            receipt = json.loads(receipt_path.read_text())
            path = Path(receipt["cache"])/"metadata.json"
            metadata = json.loads(path.read_text())
            metadata["signature"].update(weights_sha256=model_state_hash(model), head_configuration="LinearNorm",
                official_revision="pinned" if enc == "official" else None)
            metadata["signature_sha256"] = signature_hash(metadata["signature"])
            path.write_text(json.dumps(metadata), encoding="utf-8")
            receipt["metadata_sha256"] = sha256_file(path)
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        reference, controls, geometry, covariance, out = [fixture.root/n for n in ("reference", "controls", "geometry", "covariance", "sensitivity")]
        with ExitStack() as stack:
            recipe = dict(segment_training.RECIPE)
            for name, value in {"WIDTHS": (4, 2), "MAX_EPOCHS": 2, "PATIENCE": 2,
                    "BATCH_SIZE": 4, "VALIDATION_DRAWS": 3, "POSTERIOR_DRAWS": 4}.items():
                stack.enter_context(patch.object(bayes, name, value))
                recipe[name.lower()] = list(value) if name == "WIDTHS" else value
            stack.enter_context(patch.dict(segment_training.RECIPE, recipe))
            for module in (geometry_run, covariance_run, followup):
                stack.enter_context(patch.dict(module.RECIPE, bootstrap_draws=10))
            def loader(model, device, revision):
                return models["official" if model == "official" else "finetuned"], None
            with redirect_stdout(io.StringIO()):
                segment_training.run(*inputs, reference)
                controls_run(reference, *inputs, controls)
                geometry_run.run(reference, controls, *inputs, geometry, loader=loader)
                covariance_run.run(reference, controls, geometry, *inputs, covariance)
                followup.run(reference, controls, covariance, *inputs, out)
            table = pd.read_csv(out / "comparison.csv")
            self.assertEqual(len(table), 80)
            self.assertTrue(table.groupby(["encoder", "fit_profile", "head", "nominal_fpr"]).auroc.nunique().eq(1).all())
            self.assertTrue(table.loc[table["head"].eq("classifier_reference")].groupby("encoder").auroc.nunique().eq(1).all())
            completion = json.loads((out / "completion.json").read_text())
            for name, value in completion["artifacts_sha256"].items(): self.assertEqual(sha256_file(out/name), value)
            reg = json.loads((out / "sensitivity_registration.json").read_text())
            self.assertLess(reg["counts"]["gradient_fit/hdtf_only"]["sources"], reg["counts"]["gradient_fit/mixed"]["sources"])
            selected = pd.read_csv(out / "selected_units.csv")
            self.assertTrue(selected.loc[selected.source_profile.eq("hdtf_only"), "dataset"].eq("HDTF").all())
            with self.assertRaises(FileExistsError): followup.run(reference, controls, covariance, *inputs, out)
            (covariance / "comparison.csv").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Covariance result changed"):
                followup.run(reference, controls, covariance, *inputs, fixture.root/"rejected")
            self.assertFalse((fixture.root/"rejected").exists())
