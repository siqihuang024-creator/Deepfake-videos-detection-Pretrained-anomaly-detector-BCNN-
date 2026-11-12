"""Per-frame fake probabilities for a frame list (GenD softmax column 1).

  --model official            yermandy/GenD_PE_L via the upstream HF class
  --model <run>/checkpoints/best_auroc_video.ckpt   a model trained by train_gend.py

Output CSV columns: frame_path, video_key, fake_prob. Video scores are made
later by evaluate.py (mean over frames, GenD rule).

Test lists are refused unless --selection names exactly this model; the
selection file is written by select_run.py from validation results only.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gad import manifest as MF  # noqa: E402
from gad import recipe, weights  # noqa: E402
from gad.upstream import add_gend_to_path  # noqa: E402


def video_key(frame_path: str) -> str:
    parts = Path(frame_path).as_posix().split("/")
    return "/".join(parts[-4:-1])  # <dataset>/<source>/<video>


def is_test_list(path: str) -> bool:
    p = Path(path)
    return p.parent.name == "test" or p.name.startswith("test")


def check_selection(selection_path, model_arg):
    if not selection_path:
        raise SystemExit("Test list given without --selection; select on validation first (select_run.py).")
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    if model_arg == "official":
        if selection.get("checkpoint") != "official":
            raise SystemExit("Selection file does not name the official model.")
        return
    if Path(selection["checkpoint"]).resolve() != Path(model_arg).resolve():
        raise SystemExit("Checkpoint {} is not the selected one ({})".format(model_arg, selection["checkpoint"]))
    if MF.sha256(Path(model_arg)) != selection["checkpoint_sha256"]:
        raise SystemExit("Checkpoint content changed since selection.")


def load_model(model_arg, device):
    add_gend_to_path()
    import torch

    if model_arg == "official":
        hf = weights.load_hf_model().to(device).eval()
        return (lambda x: hf(x)), hf.feature_extractor.preprocess
    from src.model.GenD import GenD

    cfg = recipe.gend_pe_config(trn_files=[], val_files=[], run_dir="runs/predict", run_name="predict", seed=0)
    module = GenD(cfg, verbose=False)
    state = torch.load(model_arg, map_location="cpu", weights_only=True)["state_dict"]
    result = module.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys if k.startswith(("feature_extractor.", "model."))]
    if missing or result.unexpected_keys:
        raise SystemExit("Checkpoint keys mismatch: missing={} unexpected={}".format(missing[:5], result.unexpected_keys[:5]))
    module = module.to(device).eval()
    return (lambda x: module(x).logits_labels), module.get_preprocessing()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--frame-list", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--selection", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16",
                        help="bf16 matches GenD's bf16-mixed validation")
    args = parser.parse_args()

    if is_test_list(args.frame_list):
        check_selection(args.selection, args.model)

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    forward, preprocess = load_model(args.model, device)
    paths = [p for p in Path(args.frame_list).read_text().splitlines() if p.strip()]

    class Frames(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            return preprocess(Image.open(paths[i]).convert("RGB")), i

    loader = DataLoader(Frames(), batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    probs = np.zeros(len(paths), np.float64)
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=args.precision == "bf16"):
        for images, idx in loader:
            logits = forward(images.to(device, non_blocking=True)).float()
            probs[idx.numpy()] = logits.softmax(1)[:, weights.FAKE_COLUMN].cpu().numpy()

    out = pd.DataFrame({"frame_path": paths, "video_key": [video_key(p) for p in paths], "fake_prob": probs})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    meta = {"model": args.model, "frame_list": args.frame_list, "frame_list_sha256": MF.sha256(Path(args.frame_list)),
            "precision": args.precision, "num_frames": len(paths), "num_videos": int(out["video_key"].nunique())}
    if args.model != "official":
        meta["checkpoint_sha256"] = MF.sha256(Path(args.model))
    Path(args.out + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote {} frames / {} videos -> {}".format(len(paths), meta["num_videos"], args.out))


if __name__ == "__main__":
    main()
