"""Pick the model to test, from validation only (old protocol selection rule).

Step 2: within each group the checkpoint is the epoch with the best
val/auroc_video (saved by the training callback); across the 30 groups the
group with the highest of those values is selected (ties: lower group id).
All 30 formal groups must be finished and non-debug.

  python scripts/select_run.py --runs-dir runs/step2 --out runs/step2/selection.json
  python scripts/select_run.py --official --out runs/step1/selection.json
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gad import manifest as MF  # noqa: E402
from gad import recipe, weights  # noqa: E402

METRIC = recipe.MONITOR_METRIC


def best_of_run(run_dir: Path):
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    if not meta.get("finished") or meta.get("debug"):
        raise SystemExit("{} is unfinished or a debug run".format(run_dir))
    history = pd.read_csv(run_dir / "metrics.csv")
    history = history[history[METRIC].notna()][["epoch", METRIC]].reset_index(drop=True)
    best = history.loc[history[METRIC].idxmax()]  # first maximum = earliest epoch
    return {"run_dir": str(run_dir), "best_epoch": int(best["epoch"]), "best_val_auroc_video": float(best[METRIC]),
            "epochs_logged": int(len(history)), "checkpoint": meta["best_checkpoint"],
            "checkpoint_sha256": meta["best_checkpoint_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--runs-dir")
    mode.add_argument("--identity-run", type=Path, help="One completed identity-disjoint run directory")
    mode.add_argument("--official", action="store_true")
    parser.add_argument("--groups", type=int, nargs="*", default=list(MF.FORMAL_GROUPS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.official:
        MF.write_json(Path(args.out), {"step": 1, "checkpoint": "official", "model": weights.OFFICIAL_PE,
                                       "rule": "no training and no selection; single fixed model"})
        print("wrote", args.out)
        return

    if args.identity_run:
        record = best_of_run(args.identity_run)
        meta = json.loads((args.identity_run / "run_meta.json").read_text(encoding="utf-8"))
        val_list, = meta["args"]["val_list"]
        if record["epochs_logged"] != int(meta["effective_settings"]["max_epochs"]):
            raise SystemExit("Incomplete validation history.")
        if MF.sha256(Path(record["checkpoint"])) != record["checkpoint_sha256"]:
            raise SystemExit("Checkpoint hash differs from the training record.")
        record.update(step=2, protocol="celeb_identity", metric=METRIC,
                      rule="best video validation AUROC within this run",
                      val_frame_list_sha256=meta["list_sha256"][Path(val_list).resolve().as_posix()])
        MF.write_json(Path(args.out), record)
        print("wrote", args.out)
        return

    if sorted(args.groups) != list(MF.FORMAL_GROUPS):
        raise SystemExit("Formal selection uses exactly groups 01-30.")
    runs = {g: best_of_run(Path(args.runs_dir) / "group_{:02d}".format(g)) for g in args.groups}
    values = [r["best_val_auroc_video"] for r in runs.values()]
    chosen = max(runs, key=lambda g: (runs[g]["best_val_auroc_video"], -g))
    selection = dict(runs[chosen])
    selection.update({
        "step": 2, "selected_group": chosen, "metric": METRIC,
        "rule": "best epoch per group by val AUROC (video), then best group by the same value; test not used",
        "groups": {"group_{:02d}".format(g): r for g, r in runs.items()},
        "distribution": {"mean": statistics.mean(values), "stdev": statistics.stdev(values),
                         "median": statistics.median(values), "min": min(values), "max": max(values)},
    })
    MF.write_json(Path(args.out), selection)
    print("selected group {:02d}: val AUROC {:.4f} at epoch {}".format(
        chosen, selection["best_val_auroc_video"], selection["best_epoch"]))
    print("30-group best val AUROC: mean {mean:.4f} sd {stdev:.4f} median {median:.4f} range [{min:.4f}, {max:.4f}]".format(
        **selection["distribution"]))


if __name__ == "__main__":
    main()
