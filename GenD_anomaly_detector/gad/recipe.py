"""GenD-PE training recipe, written out field by field.

Values are what upstream actually runs for ``wacv-PE_L-LN+L2+UnAl-seed*``
through ``run_exp.py``: ``get_default_train_config()`` (batch 128, 1 warm-up
epoch, lr 3e-4, min lr 1e-5, wd 0) merged with ``get_common()`` (cyclic
cosine, 10-epoch cycles, 30 epochs, patience 30) and the PE experiment entry.
``compare_with_upstream()`` rebuilds that config from upstream code and
reports any difference, so the copy here cannot silently drift.

Deliberate protocol deviations (EXPERIMENT_PLAN.md section 0):
  - training/validation files come from our lists, no test files at all;
  - checkpoint monitor ``val/auroc_video`` instead of ``val/mAP_video``.
"""

from __future__ import annotations

from .upstream import add_gend_to_path

UPSTREAM_EXPERIMENT = "wacv-PE_L-LN+L2+UnAl"

# Fields that define the GenD method. Everything else (files, run dir,
# logging, workers, monitor) is protocol/infrastructure.
RECIPE_FIELDS = (
    "num_classes", "backbone", "backbone_args", "freeze_feature_extractor", "unfreeze_layers", "head",
    "inference_strategy", "peft_v2", "binary_labels", "custom_preprocessing", "augmentations",
    "test_augmentations", "load_pairs", "lr", "min_lr", "lr_scheduler", "warmup_epochs",
    "num_epochs_in_cycle", "optimizer", "weight_decay", "betas", "loss", "max_epochs", "batch_size",
    "precision", "early_stopping_patience", "monitor_metric_mode",
    "make_binary_before_video_aggregation", "reduce_video_predictions",
)

MONITOR_METRIC = "val/auroc_video"
CHECKPOINT_NAME = "best_auroc_video"


def gend_pe_config(*, trn_files, val_files, run_dir, run_name, seed, mini_batch_size=128, num_workers=8):
    add_gend_to_path()
    from src import config as C

    cfg = C.Config()
    # --- get_default_train_config() ---
    cfg.num_classes = 2
    cfg.freeze_feature_extractor = True
    cfg.batch_size = 128
    cfg.lr_scheduler = "cosine"  # replaced by get_common() below
    cfg.lr = 3e-4
    cfg.min_lr = 1e-5
    cfg.weight_decay = 0
    cfg.warmup_epochs = 1
    # --- get_common() ---
    cfg.lr_scheduler = "cyclic"
    cfg.num_epochs_in_cycle = 10
    cfg.early_stopping_patience = 30
    cfg.max_epochs = 30
    # --- experiment "wacv-PE_L-LN+L2+UnAl" ---
    cfg.backbone = C.Backbone.PerceptionEncoder_L_p14_336
    cfg.backbone_args = C.BackboneArgs(img_size=224)
    cfg.head = C.Head.NLinear
    cfg.unfreeze_layers = ["norm_pre", "norm1", "norm2", "norm"]
    cfg.loss = C.Loss(ce_labels=1.0, uniformity=0.5, alignment_labels=0.1)
    # --- protocol / infrastructure (ours) ---
    if 128 % mini_batch_size:
        raise ValueError("mini_batch_size must divide 128 (gradient accumulation keeps batch 128)")
    cfg.mini_batch_size = mini_batch_size
    cfg.seed = seed
    cfg.trn_files = list(trn_files)
    cfg.val_files = list(val_files)
    cfg.tst_files = []
    cfg.monitor_metric = MONITOR_METRIC
    cfg.monitor_metric_mode = "max"
    cfg.checkpoint_name = CHECKPOINT_NAME
    cfg.checkpoint = None
    cfg.run_dir = str(run_dir)
    cfg.run_name = run_name
    cfg.wandb = False
    cfg.num_workers = num_workers
    cfg.devices = 1
    return cfg


def _dump(value):
    return value.model_dump() if hasattr(value, "model_dump") else value


def upstream_reference_config(seed: int = 0):
    """Rebuild the config exactly as upstream ``run_exp.entry`` would."""
    add_gend_to_path()
    from copy import deepcopy

    from run_exp import experiments, get_default_train_config
    from src.config import Config

    config = deepcopy(get_default_train_config())
    for modify in experiments["{}-seed{}".format(UPSTREAM_EXPERIMENT, seed)]:
        difference = modify.model_dump(exclude_unset=True)
        config = Config(**config.model_copy(update=difference).model_dump())
    return config


def compare_with_upstream(ours) -> dict:
    """Return {field: (ours, upstream)} for every recipe field that differs."""
    reference = upstream_reference_config()
    diff = {}
    for name in RECIPE_FIELDS:
        a, b = _dump(getattr(ours, name)), _dump(getattr(reference, name))
        if a != b:
            diff[name] = (a, b)
    return diff


def effective_settings(cfg) -> dict:
    out = {name: _dump(getattr(cfg, name)) for name in RECIPE_FIELDS}
    out.update(mini_batch_size=cfg.mini_batch_size, seed=cfg.seed, monitor_metric=cfg.monitor_metric,
               checkpoint_name=cfg.checkpoint_name, num_trn_lists=len(cfg.trn_files),
               num_val_lists=len(cfg.val_files), num_tst_lists=len(cfg.tst_files))
    return out
