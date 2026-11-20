"""Train GenD-PE with the GenD recipe; fit only, no test stage.

Differs from upstream ``run.py`` in these ways (EXPERIMENT_PLAN.md):
  - no test files are given and ``trainer.test`` is never called
    (upstream tests automatically after fit);
  - ``--init official`` loads yermandy/GenD_PE_L into the trainable model
    (upstream would load the inference-only GenDHF);
  - checkpoint monitor is val/auroc_video.
  - --batch-size explicitly permits a recorded override of upstream's 128.
Loggers, callbacks, Trainer arguments and the data module are upstream's.

Step 2 (old protocol, group g):
  python scripts/train_gend.py --init official --train-list lists/group_01/train.txt \
      --val-list lists/val.txt --seed 42 --run-dir runs/step2 --run-name group_01
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gad import manifest as MF  # noqa: E402
from gad import recipe, weights  # noqa: E402
from gad.ablations import reserve_run_directory  # noqa: E402
from gad.upstream import add_gend_to_path, upstream_commit  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init", choices=["official", "pe"], required=True,
                        help="official: start from yermandy/GenD_PE_L (step 2); pe: raw PE weights (step 3)")
    parser.add_argument("--train-list", nargs="+", required=True)
    parser.add_argument("--val-list", nargs="+", required=True)
    parser.add_argument("--seed", type=int, required=True, help="Old protocol: 41 + group number")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--batch-size", type=int, default=recipe.DEFAULT_BATCH_SIZE,
                        help="Images per optimizer update on one GPU; default 128 matches upstream")
    parser.add_argument("--mini-batch-size", type=int, default=None,
                        help="Images per forward pass; must equal --batch-size (no gradient accumulation)")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--trainable", choices=("layernorm", "full"), default="layernorm")
    parser.add_argument("--lr", type=float, default=3e-4, help="Head/LayerNorm learning rate")
    parser.add_argument("--backbone-lr", type=float, default=1e-5, help="Other encoder parameters in full mode")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--probe-steps", type=int, help="Training-only resource probe; no checkpoint selection or testing")
    parser.add_argument("--gpu-log-interval", type=int, default=5)
    parser.add_argument("--max-epochs-debug", type=int, default=None,
                        help="Debug only: shorter run; results must not be reported")
    args = parser.parse_args()
    try:
        mini_batch_size, accumulation = recipe.batch_parameters(args.batch_size, args.mini_batch_size)
    except ValueError as error:
        parser.error(str(error))
    if args.backbone_lr <= 0 or args.lr < 1e-5 or args.gpu_log_interval < 1 or args.num_workers < 0:
        parser.error("Invalid learning rate, monitor interval or worker count")
    if args.probe_steps is not None and args.probe_steps < 1:
        parser.error("--probe-steps must be positive")

    for path in args.train_list + args.val_list:
        if "/test/" in Path(path).as_posix() or Path(path).name.startswith("test"):
            raise SystemExit("Refusing a test list in training: {}".format(path))

    add_gend_to_path()
    import torch
    from lightning import Trainer

    import run as upstream_run
    from src import dataset as datasets
    from gad.finetuning import build_model, grouped_parameters, group_report
    from gad.training_monitor import TrainingMonitor

    save_dir = Path(args.run_dir) / args.run_name
    try:
        reserve_run_directory(save_dir)
    except FileExistsError:
        raise SystemExit("Run directory exists, refusing to overwrite: {}".format(save_dir))

    cfg = recipe.gend_pe_config(trn_files=[str(Path(p).resolve().as_posix()) for p in args.train_list],
                                val_files=[str(Path(p).resolve().as_posix()) for p in args.val_list],
                                run_dir=args.run_dir, run_name=args.run_name, seed=args.seed,
                                batch_size=args.batch_size, mini_batch_size=mini_batch_size,
                                num_workers=args.num_workers, trainable=args.trainable, lr=args.lr)
    if args.max_epochs_debug:
        cfg.max_epochs = args.max_epochs_debug  # use a separate --run-dir for debug runs

    try:
        diff = recipe.compare_with_upstream(cfg)
    except Exception as error:
        diff = "could not rebuild upstream config: {!r}".format(error)
    allowed = {"max_epochs"} if args.max_epochs_debug else set()
    if args.batch_size != recipe.DEFAULT_BATCH_SIZE:
        allowed.add("batch_size")
    if args.trainable == "full":
        allowed.add("freeze_feature_extractor")
    if args.lr != 3e-4:
        allowed.add("lr")
    if isinstance(diff, dict) and set(diff) - allowed:
        raise SystemExit("Recipe differs from upstream: {}".format(diff))
    print(f"Batch: {mini_batch_size} individual images per forward, {accumulation} accumulation steps, "
          f"{cfg.batch_size} images per optimizer update. Metric losses use {mini_batch_size} images.", flush=True)

    torch.set_float32_matmul_precision("high")  # as upstream run.main
    model = build_model(cfg, args.trainable, args.backbone_lr, args.gradient_checkpointing)
    init_report = weights.load_official_into_trainable(model) if args.init == "official" else {"init": "raw PE (timm)"}

    data_module = datasets.DeepfakeDataModule(cfg, model.get_preprocessing())
    monitor = TrainingMonitor(save_dir, args.gpu_log_interval)
    callbacks = ([monitor] if args.probe_steps else upstream_run.init_callbacks(cfg) + [monitor])
    trainer = Trainer(
        devices=cfg.devices,
        max_epochs=cfg.max_epochs,
        precision=cfg.precision,
        accumulate_grad_batches=accumulation,
        log_every_n_steps=100,
        deterministic=cfg.deterministic,
        logger=upstream_run.init_loggers(cfg),
        callbacks=callbacks,
        default_root_dir=cfg.run_dir,
        max_steps=args.probe_steps if args.probe_steps else -1,
        limit_val_batches=0 if args.probe_steps else 1.0,
        num_sanity_val_steps=0 if args.probe_steps else 2,
        enable_checkpointing=not bool(args.probe_steps),
    )

    project_root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root,
                            capture_output=True, text=True).stdout.strip()
    meta = {
        "args": vars(args),
        "init": init_report,
        "gend_commit": upstream_commit(),
        "recipe_diff_vs_upstream": diff if isinstance(diff, str) else {k: [str(a), str(b)] for k, (a, b) in diff.items()},
        "effective_settings": {k: str(v) for k, v in recipe.effective_settings(cfg).items()},
        "list_sha256": {p: MF.sha256(Path(p)) for p in cfg.trn_files + cfg.val_files},
        "debug": bool(args.max_epochs_debug or args.probe_steps),
        "trainable": args.trainable, "project_git_commit": commit,
        "gradient_checkpointing": args.gradient_checkpointing,
        "ablation_plan_sha256": os.environ.get("GEND_ABLATION_PLAN_SHA256") or None,
        "optimizer_parameter_groups": group_report(grouped_parameters(
            model, args.lr, args.backbone_lr if args.trainable == "full" else args.lr, cfg.weight_decay
        ), cfg.min_lr / cfg.lr),
        "batching": {
            "sample_unit": "individual_frame", "optimizer_batch_size": cfg.batch_size,
            "forward_batch_size": cfg.mini_batch_size, "gradient_accumulation": accumulation,
            "metric_loss_batch_size": cfg.mini_batch_size,
            "upstream_optimizer_batch_size": recipe.DEFAULT_BATCH_SIZE,
        },
    }
    (save_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    try:
        trainer.fit(model, data_module)
    finally:
        monitor.stop()
        meta["performance"] = monitor.report()
        (save_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if args.probe_steps:
        meta["probe_completed"] = True
        meta["probe_steps_completed"] = trainer.global_step
        (save_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("Resource probe complete:", json.dumps(meta["performance"], indent=2))
        return
    for callback in callbacks:
        if hasattr(callback, "wait"):
            callback.wait()
    ckpt = save_dir / "checkpoints" / "{}.ckpt".format(cfg.checkpoint_name)
    if not ckpt.exists():
        raise SystemExit("Best checkpoint missing: {}".format(ckpt))
    meta["best_checkpoint"] = str(ckpt)
    meta["best_checkpoint_sha256"] = MF.sha256(ckpt)
    meta["finished"] = True
    (save_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("done; best checkpoint:", ckpt)


if __name__ == "__main__":
    main()
