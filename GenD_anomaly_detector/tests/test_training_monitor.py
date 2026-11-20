"""CPU callback lifecycle tests; Lightning base is stubbed, no GPU needed."""

import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import lightning
except ImportError:
    stub = ModuleType("lightning")
    stub.Callback = object
    sys.modules["lightning"] = stub
from gad.training_monitor import TrainingMonitor


class MonitorChecks(unittest.TestCase):
    def test_throughput_excludes_validation_and_preserves_image_count(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = TrainingMonitor(directory)
            trainer = SimpleNamespace(current_epoch=0, num_training_batches=1)
            model = SimpleNamespace(device=torch.device("cpu"))
            monitor.on_fit_start(trainer, model)
            with patch("gad.training_monitor.time.perf_counter", side_effect=[10, 20, 25, 40]):
                monitor.on_train_epoch_start(trainer, model)
                monitor.on_train_batch_end(trainer, model, None, {"image": torch.zeros(128, 3)}, 0)
                monitor.on_validation_start(trainer, model)
                monitor.on_validation_end(trainer, model)
                monitor.on_train_epoch_end(trainer, model)
            monitor.on_fit_end(trainer, model)
            result = json.loads((Path(directory) / "performance.json").read_text())
            self.assertEqual(result["training_seconds"], 25)
            self.assertEqual(result["training_images"], 128)
            self.assertAlmostEqual(result["images_per_second"], 128 / 25)
            self.assertTrue(result["epochs"][0]["complete"])

    def test_step_limited_probe_does_not_claim_a_full_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = TrainingMonitor(directory)
            trainer = SimpleNamespace(current_epoch=0, num_training_batches=2299)
            model = SimpleNamespace(device=torch.device("cpu"))
            monitor.on_fit_start(trainer, model)
            with patch("gad.training_monitor.time.perf_counter", side_effect=[0, 10]):
                monitor.on_train_epoch_start(trainer, model)
                for batch_index in range(20):
                    monitor.on_train_batch_end(trainer, model, None, {"image": torch.zeros(128, 3)}, batch_index)
                monitor.on_train_epoch_end(trainer, model)
            epoch = monitor.report()["epochs"][0]
            self.assertFalse(epoch["complete"])
            self.assertEqual(epoch["batches"], 20)
            self.assertEqual(epoch["images"], 2560)

    def test_bare_torch_uuid_filters_the_assigned_gpu_in_nvidia_smi(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = TrainingMonitor(directory)
            raw_uuid = "4048158a-8cc7-d532-33f5-144cb78bf4f9"
            props = SimpleNamespace(name="H200", total_memory=150156607488, uuid=raw_uuid)
            process = Mock()
            process.poll.return_value = 0
            process.returncode = 0
            with patch("gad.training_monitor.torch.cuda.reset_peak_memory_stats"), \
                    patch("gad.training_monitor.torch.cuda.get_device_properties", return_value=props), \
                    patch("gad.training_monitor.shutil.which", return_value="nvidia-smi"), \
                    patch("gad.training_monitor.subprocess.Popen", return_value=process) as popen:
                monitor.on_fit_start(SimpleNamespace(), SimpleNamespace(device=torch.device("cuda")))
                command = popen.call_args[0][0]
                self.assertEqual(command[command.index("--id") + 1], "GPU-" + raw_uuid)
                self.assertEqual(monitor.gpu["uuid"], "GPU-" + raw_uuid)
                monitor.stop()

    def test_monitor_process_and_files_are_cleaned_once(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = TrainingMonitor(directory)
            process = Mock()
            process.poll.return_value = None
            process.returncode = 0
            stream = (Path(directory) / "gpu.csv").open("w")
            monitor.process, monitor.handles = process, [stream]
            monitor.stop()
            monitor.stop()
            process.terminate.assert_called_once()
            process.wait.assert_called_once()
            self.assertTrue(stream.closed)


if __name__ == "__main__":
    unittest.main()
