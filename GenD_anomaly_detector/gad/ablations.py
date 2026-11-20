"""Preregistered step-2 matrix; split seed stays 42 for every training seed."""

from pathlib import Path

SEEDS = (42, 43, 44)
SETTINGS = ("ln_b128", "ln_b256", "full_b128")
TASK_COUNT = len(SEEDS) * len(SETTINGS)
FULL_PROBE_INDEX = len(SEEDS) * SETTINGS.index("full_b128")
EXECUTION = {"gpu_per_task": 1, "max_concurrent_tasks": 1, "setting_order": list(SETTINGS)}


def run_spec(index):
    if not 0 <= index < TASK_COUNT:
        raise ValueError(f"Ablation task index must be 0..{TASK_COUNT - 1}")
    setting = SETTINGS[index // len(SEEDS)]
    seed = SEEDS[index % len(SEEDS)]
    return {"setting": setting, "seed": seed, "split_seed": 42,
            "run_name": f"{setting}_s{seed}", "trainable": "full" if setting.startswith("full") else "layernorm",
            "batch_size": 256 if setting.endswith("256") else 128,
            "lr": 3e-4, "backbone_lr": 1e-5}


def reserve_run_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path
