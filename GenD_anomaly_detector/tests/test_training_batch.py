"""CPU-only checks of explicit batch overrides; no models or downloads.

Run: python tests/test_training_batch.py -v
"""

import sys
import unittest
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gad import recipe
from gad.ablations import run_spec, reserve_run_directory


def configuration(**kwargs):
    return recipe.gend_pe_config(
        trn_files=["train.txt"], val_files=["val.txt"],
        run_dir="unused", run_name="check", seed=42, **kwargs,
    )


class TrainingBatchChecks(unittest.TestCase):
    def test_default_configuration_is_preserved(self):
        default = configuration()
        explicit = configuration(batch_size=128, mini_batch_size=128)
        self.assertEqual(default.model_dump(), explicit.model_dump())
        self.assertEqual(default.batch_size, 128)
        self.assertEqual(default.mini_batch_size, 128)

    def test_256_uses_one_forward_without_other_recipe_changes(self):
        baseline = configuration().model_dump()
        larger = configuration(batch_size=256)
        self.assertEqual(larger.batch_size, 256)
        self.assertEqual(larger.mini_batch_size, 256)
        settings = recipe.effective_settings(larger)
        self.assertEqual(settings["gradient_accumulation"], 1)
        self.assertEqual(settings["metric_loss_batch_size"], 256)
        changed = larger.model_dump()
        for field in ("batch_size", "mini_batch_size"):
            baseline.pop(field)
            changed.pop(field)
        self.assertEqual(baseline, changed)

    def test_accumulation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "gradient accumulation is disabled"):
            configuration(batch_size=256, mini_batch_size=128)

    def test_full_configuration_unfreezes_the_encoder_only_as_requested(self):
        baseline = configuration().model_dump()
        full = configuration(trainable="full").model_dump()
        self.assertTrue(baseline.pop("freeze_feature_extractor"))
        self.assertFalse(full.pop("freeze_feature_extractor"))
        self.assertEqual(baseline, full)

    def test_matrix_keeps_split_seed_fixed_and_separates_training_seeds(self):
        specs = [run_spec(i) for i in range(9)]
        self.assertEqual(len({s["run_name"] for s in specs}), 9)
        self.assertEqual({s["split_seed"] for s in specs}, {42})
        for setting in ("ln_b128", "ln_b256", "full_b128"):
            self.assertEqual({s["seed"] for s in specs if s["setting"] == setting}, {42, 43, 44})
        self.assertEqual([s["setting"] for s in specs], ["ln_b128"] * 3 + ["ln_b256"] * 3 + ["full_b128"] * 3)
        for invalid in (-1, 9):
            with self.assertRaises(ValueError):
                run_spec(invalid)

    def test_run_reservation_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "runs" / "ln_b128_s42"
            reserve_run_directory(target)
            with self.assertRaises(FileExistsError):
                reserve_run_directory(target)

    def test_invalid_batch_settings_are_rejected(self):
        for total, micro in ((0, None), (128, 0), (128, 1), (128, 256), (256, 100), (-1, 128)):
            with self.subTest(total=total, micro=micro):
                with self.assertRaises(ValueError):
                    configuration(batch_size=total, mini_batch_size=micro)


if __name__ == "__main__":
    unittest.main()
