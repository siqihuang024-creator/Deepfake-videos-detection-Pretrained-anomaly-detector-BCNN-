"""Run step 2 for several groups sequentially (old protocol: seed = 41 + group).

Finished groups are skipped; a group whose directory exists without a
finished run_meta.json stops the queue for inspection (never overwritten).
Test data is never touched here.

  python scripts/run_groups.py --lists lists --runs-dir runs/step2 --groups 1 2 3
  python scripts/run_groups.py ... --dry-run
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "train_gend.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lists", required=True, help="Output dir of write_frame_lists.py")
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--groups", type=int, nargs="+", default=list(range(1, 31)))
    parser.add_argument("--init", default="official", choices=["official"])
    parser.add_argument("--mini-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    lists = Path(args.lists)
    for group in args.groups:
        name = "group_{:02d}".format(group)
        run_dir = Path(args.runs_dir) / name
        meta = run_dir / "run_meta.json"
        if run_dir.exists():
            if meta.exists() and json.loads(meta.read_text()).get("finished"):
                print("skip", name, "(finished)")
                continue
            raise SystemExit("{} exists but is not finished; inspect it before continuing".format(run_dir))
        command = [sys.executable, str(SCRIPT), "--init", args.init,
                   "--train-list", str(lists / name / "train.txt"), "--val-list", str(lists / "val.txt"),
                   "--seed", str(41 + group), "--run-dir", args.runs_dir, "--run-name", name,
                   "--mini-batch-size", str(args.mini_batch_size), "--num-workers", str(args.num_workers)]
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
