"""Measure actual training throughput and GPU memory; observe power only."""

import json
from pathlib import Path
import shutil
import subprocess
import time
from uuid import UUID

import torch
from lightning import Callback


class TrainingMonitor(Callback):
    def __init__(self, directory, interval=5):
        self.directory = Path(directory)
        self.interval = interval
        self.epochs = []
        self.process = None
        self.handles = []
        self.started = None
        self.images = 0
        self.batches = 0
        self.validation_started = None
        self.validation_seconds = 0.0
        self.gpu = {}
        self.monitor_status = "not_started"

    def on_fit_start(self, trainer, pl_module):
        self.device = pl_module.device
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            props = torch.cuda.get_device_properties(self.device)
            gpu_uuid = str(getattr(props, "uuid", ""))
            if gpu_uuid and not gpu_uuid.startswith(("GPU-", "MIG-")):
                try:
                    gpu_uuid = "GPU-" + str(UUID(gpu_uuid))
                except ValueError:
                    pass
            self.gpu = {"name": props.name, "total_memory_bytes": props.total_memory,
                        "uuid": gpu_uuid, "torch": torch.__version__,
                        "cuda": torch.version.cuda}
            executable = shutil.which("nvidia-smi")
            if executable:
                command = [executable, "--query-gpu=timestamp,index,uuid,power.draw,power.limit,utilization.gpu,memory.used",
                           "--format=csv", "-l", str(self.interval)]
                if self.gpu["uuid"].startswith(("GPU-", "MIG-")):
                    command += ["--id", self.gpu["uuid"]]
                out = (self.directory / "gpu_log.csv").open("w", encoding="utf-8")
                error = (self.directory / "gpu_monitor_errors.log").open("w", encoding="utf-8")
                self.handles = [out, error]
                try:
                    self.process = subprocess.Popen(command, stdout=out, stderr=error, stdin=subprocess.DEVNULL)
                    self.monitor_status = "started"
                except OSError as exc:
                    self.monitor_status = str(exc)
            else:
                self.monitor_status = "nvidia-smi unavailable"
        else:
            self.monitor_status = "CPU run"

    def on_train_epoch_start(self, trainer, pl_module):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.started = time.perf_counter()
        self.images = 0
        self.batches = 0
        self.validation_seconds = 0.0

    def on_validation_start(self, trainer, pl_module):
        if self.started is not None:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.validation_started = time.perf_counter()

    def on_validation_end(self, trainer, pl_module):
        if self.validation_started is not None:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.validation_seconds += time.perf_counter() - self.validation_started
            self.validation_started = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.images += len(batch["image"])
        self.batches += 1

    def finish_epoch(self, epoch, complete=True):
        if self.started is None:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - self.started - self.validation_seconds
        self.epochs.append({"epoch": epoch, "images": self.images, "batches": self.batches, "seconds": seconds,
                            "images_per_second": self.images / seconds, "complete": complete})
        self.started = None
        self.write_report()

    def on_train_epoch_end(self, trainer, pl_module):
        # Lightning calls this hook even when max_steps stops a partial epoch.
        expected = getattr(trainer, "num_training_batches", float("inf"))
        self.finish_epoch(trainer.current_epoch, complete=self.batches >= expected)

    def report(self):
        seconds = sum(item["seconds"] for item in self.epochs)
        images = sum(item["images"] for item in self.epochs)
        return {"gpu": self.gpu, "gpu_monitor": self.monitor_status,
                "throughput_definition": "training epoch wall time including data waits, excluding validation",
                "epochs": self.epochs, "training_images": images, "training_seconds": seconds,
                "images_per_second": images / seconds if seconds else None,
                "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(self.device)
                if hasattr(self, "device") and self.device.type == "cuda" else None,
                "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(self.device)
                if hasattr(self, "device") and self.device.type == "cuda" else None}

    def write_report(self):
        (self.directory / "performance.json").write_text(json.dumps(self.report(), indent=2), encoding="utf-8")

    def stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.monitor_status = f"exit_code={self.process.returncode}"
            self.process = None
        for stream in self.handles:
            stream.close()
        self.handles = []

    def on_fit_end(self, trainer, pl_module):
        self.stop()
        self.write_report()

    def on_exception(self, trainer, pl_module, exception):
        self.finish_epoch(trainer.current_epoch, complete=False)
        self.stop()
        self.write_report()
