"""Train GenD-PE with the GenD recipe; fit only, no test stage.

Differs from upstream ``run.py`` in exactly these ways (EXPERIMENT_PLAN.md):
  - no test files are given and ``trainer.test`` is never called
    (upstream tests automatically after fit);
  - ``--init official`` loads yermandy/GenD_PE_L into the trainable model
    (upstream would load the inference-only GenDHF);
  - checkpoint monitor is val/auroc_video.
Loggers, callbacks, Trainer arguments and the data module are upstream's.

Step 2 (old protocol, group g):
  python scripts/train_gend.py --init official --train-list lists/group_01/train.txt \
      --val-list lists/val.txt --seed 42 --run-dir runs/step2 --run-name group_01
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gad import manifest as MF  # noqa: E402
from gad import recipe, weights  # noqa: E402
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
    parser.add_argument("--mini-batch-size", type=int, default=128, help="Gradient accumulation keeps batch 128")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-epochs-debug", type=int, default=None,
                        help="Debug only: shorter run; results must not be reported")
    args = parser.parse_args()

    for path in args.train_list + args.val_list:
        if "/test/" in Path(path).as_posix() or Path(path).name.startswith("test"):
            raise SystemExit("Refusing a test list in training: {}".format(path))

    add_gend_to_path()
    import torch
    from lightning import Trainer

    import run as upstream_run
    from src import dataset as datasets
    from src.model.GenD import GenD

    save_dir = Path(args.run_dir) / args.run_name
    if save_dir.exists():
        raise SystemExit("Run directory exists, refusing to overwrite: {}".format(save_dir))

    cfg = recipe.gend_pe_config(trn_files=[str(Path(p).resolve().as_posix()) for p in args.train_list],
                                val_files=[str(Path(p).resolve().as_posix()) for p in args.val_list],
                                run_dir=args.run_dir, run_name=args.run_name, seed=args.seed,
                                mini_batch_size=args.mini_batch_size, num_workers=args.num_workers)
    if args.max_epochs_debug:
        cfg.max_epochs = args.max_epochs_debug  # use a separate --run-dir for debug runs

    try:
        diff = recipe.compare_with_upstream(cfg)
    except Exception as error:
        diff = "could not rebuild upstream config: {!r}".format(error)
    allowed = {"max_epochs"} if args.max_epochs_debug else set()
    if isinstance(diff, dict) and set(diff) - allowed:
        raise SystemExit("Recipe differs from upstream: {}".format(diff))

    torch.set_float32_matmul_precision("high")  # as upstream run.main
    model = GenD(cfg, verbose=True)
    init_report = weights.load_official_into_trainable(model) if args.init == "official" else {"init": "raw PE (timm)"}

    data_module = datasets.DeepfakeDataModule(cfg, model.get_preprocessing())
    callbacks = upstream_run.init_callbacks(cfg)
    trainer = Trainer(
        devices=cfg.devices,
        max_epochs=cfg.max_epochs,
        precision=cfg.precision,
        accumulate_grad_batches=cfg.batch_size // cfg.mini_batch_size,
        log_every_n_steps=100,
        deterministic=cfg.deterministic,
        logger=upstream_run.init_loggers(cfg),
        callbacks=callbacks,
        default_root_dir=cfg.run_dir,
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "args": vars(args),
        "init": init_report,
        "gend_commit": upstream_commit(),
        "recipe_diff_vs_upstream": diff if isinstance(diff, str) else {k: [str(a), str(b)] for k, (a, b) in diff.items()},
        "effective_settings": {k: str(v) for k, v in recipe.effective_settings(cfg).items()},
        "list_sha256": {p: MF.sha256(Path(p)) for p in cfg.trn_files + cfg.val_files},
        "debug": bool(args.max_epochs_debug),
    }
    (save_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    trainer.fit(model, data_module)
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
