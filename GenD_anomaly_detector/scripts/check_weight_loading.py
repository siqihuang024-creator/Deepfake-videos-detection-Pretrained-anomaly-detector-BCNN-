"""Pre-training checks for step 2 (run once on a GPU node before any training).

1. Our recipe vs upstream ``run_exp`` config for wacv-PE_L-LN+L2+UnAl: any
   differing method field is printed and makes the check fail.
2. Official HF weights load into the trainable GenD with every key matched.
3. Trainable GenD and the official inference model give the same fake
   probability on the same face crops (fp32, eval mode).
4. Trainable parameter names/count are listed (only LayerNorms + head).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gad import recipe, weights  # noqa: E402
from gad.upstream import add_gend_to_path, upstream_commit  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame-list", required=True, help="e.g. lists/val.txt (validation crops only)")
    parser.add_argument("--num-images", type=int, default=64)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--out", required=True, help="JSON report path")
    args = parser.parse_args()

    add_gend_to_path()
    from src.model.GenD import GenD

    paths = [p for p in Path(args.frame_list).read_text().splitlines() if p.strip()]
    step = max(1, len(paths) // args.num_images)
    sample = paths[::step][: args.num_images]

    cfg = recipe.gend_pe_config(trn_files=[], val_files=[], run_dir="runs/check", run_name="check", seed=0)
    report = {"gend_commit": upstream_commit()}
    try:
        diff = recipe.compare_with_upstream(cfg)
        report["recipe_diff_vs_upstream"] = {k: [str(a), str(b)] for k, (a, b) in diff.items()}
    except Exception as error:  # upstream import may need its dataset lists
        report["recipe_diff_vs_upstream"] = "could not rebuild upstream config: {!r}".format(error)
        diff = None
    report["effective_settings"] = {k: str(v) for k, v in recipe.effective_settings(cfg).items()}

    module = GenD(cfg, verbose=False)
    report["load"] = weights.load_official_into_trainable(module)
    trainable = [(n, int(p.numel())) for n, p in module.named_parameters() if p.requires_grad]
    report["trainable_parameters"] = {"count": sum(n for _, n in trainable), "tensors": len(trainable),
                                      "names": [n for n, _ in trainable]}
    bad = [n for n, _ in trainable if not (n.startswith("model.linear.") or
                                           any(k in n for k in cfg.unfreeze_layers))]
    worst, rows = weights.consistency_check(module, sample)
    report["consistency"] = {"num_images": len(rows), "max_abs_diff_fake_prob": worst,
                             "tolerance": args.tolerance, "examples": rows[:5]}

    ok = worst <= args.tolerance and not bad and diff == {}
    report["passed"] = bool(ok)
    report["unexpected_trainable"] = bad
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("recipe_diff_vs_upstream", "passed", "unexpected_trainable")}, indent=2))
    print("trainable params: {} in {} tensors; max |dp| = {:.2e}".format(
        report["trainable_parameters"]["count"], len(trainable), worst))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
