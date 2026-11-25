"""Extract reusable frame features and probabilities in one frozen encoder pass."""
import argparse
import csv
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gad.feature_cache import (FrameCacheWriter, crop_stat_hash, frame_timing,
                               load_video_cache, model_state_hash, sha256_file, signature_hash)
from gad import weights
from gad.upstream import upstream_commit
from predict_frames import check_selection, is_test_list, load_frozen_module


def export_probabilities(cache, out):
    cache, out = Path(cache), Path(out)
    values = load_video_cache(cache)
    meta = values["metadata"]
    for name in ("frame_paths.txt", "frame_fake_prob.npy"):
        if sha256_file(cache / name) != meta["files"][name]["sha256"]:
            raise ValueError(f"Changed cached predictions: {name}")
    paths = (cache / "frame_paths.txt").read_text(encoding="utf-8").splitlines()
    probs = np.load(cache / "frame_fake_prob.npy", mmap_mode="r", allow_pickle=False)
    from gad.feature_cache import video_key
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() or Path(str(out) + ".meta.json").exists():
        raise FileExistsError(out)
    with out.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frame_path", "video_key", "fake_prob"))
        writer.writerows((p, video_key(p), float(v)) for p, v in zip(paths, probs))
    signature = meta["signature"]
    provenance = {
        "model": signature["model"], "checkpoint_sha256": signature.get("checkpoint_sha256"),
        "official_weights_sha256": signature["weights_sha256"] if signature["model"] == "official" else None,
        "precision": signature["precision"], "frame_list_sha256": signature["frame_list_sha256"],
        "frame_list": signature["frame_list"], "feature_cache": str(cache.resolve()),
        "num_frames": len(paths), "num_videos": len(values["video_keys"]),
        **meta["measurements"],
    }
    Path(str(out) + ".meta.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--selection")
    parser.add_argument("--official-revision", help="Optionally pin the official Hugging Face revision")
    parser.add_argument("--frame-list", type=Path, required=True)
    parser.add_argument("--records", type=Path, help="Preprocessing records, for decoded frame timestamps")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, help="Write the resolved cache location for subsequent CPU diagnosis")
    parser.add_argument("--list-registration", type=Path, help="frame_lists.json written for the diagnostic protocol")
    parser.add_argument("--probabilities-out", type=Path, help="Optional classifier CSV from this same pass")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("Invalid batch size or worker count")
    if args.model != "official" or is_test_list(str(args.frame_list)):
        check_selection(args.selection, args.model)
    paths = [p for p in args.frame_list.read_text(encoding="utf-8").splitlines() if p.strip()]
    if not paths:
        parser.error("Empty frame list")
    registration_hash = None
    if args.list_registration:
        registration = json.loads(args.list_registration.read_text(encoding="utf-8"))
        relative = args.frame_list.resolve().relative_to(args.list_registration.parent.resolve()).as_posix()
        if registration["files"].get(relative) != sha256_file(args.frame_list):
            parser.error("Frame list differs from its registration")
        registration_hash = registration["plan_sha256"]

    import torch
    from torch.nn.functional import normalize
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use an allocated GPU or explicitly choose --device cpu --precision fp32")
    if args.device == "cpu" and args.precision != "fp32":
        parser.error("Use --precision fp32 for the CPU smoke path")
    module, preprocess = load_frozen_module(args.model, "cpu", args.official_revision)
    extractor_file = Path(inspect.getfile(module.feature_extractor.__class__))
    indices, times = frame_timing(paths, args.records)
    timing_hash = __import__("hashlib").sha256(times.tobytes()).hexdigest()
    signature = {
        "model": args.model, "model_repo": weights.OFFICIAL_PE,
        "weights_sha256": model_state_hash(module),
        "checkpoint_sha256": sha256_file(args.model) if args.model != "official" else None,
        "official_revision": getattr(module.config, "_commit_hash", None) if args.model == "official" else None,
        "frame_list": str(args.frame_list.resolve()),
        "frame_list_sha256": sha256_file(args.frame_list),
        "crop_stat_sha256": crop_stat_hash(paths), "frame_times_sha256": timing_hash,
        "precision": args.precision, "gend_commit": upstream_commit(),
        "preprocess": repr(module.feature_extractor._preprocess),
        "extractor_source_sha256": sha256_file(extractor_file),
        "custom_preprocessing": str(getattr(module.config, "custom_preprocessing", None)),
        "head_configuration": str(getattr(module.config, "head", None)),
        "cache_algorithm": "frozen_frame_l2_with_norms_v1",
        "diagnostic_plan_sha256": registration_hash,
    }
    destination = args.cache_root / signature_hash(signature)
    if destination.exists():
        load_video_cache(destination, signature)
        print(f"Reused verified cache: {destination}", flush=True)
        if args.probabilities_out:
            export_probabilities(destination, args.probabilities_out)
        write_receipt(args.receipt, destination)
        return

    class Frames(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, index):
            with Image.open(paths[index]) as picture:
                return preprocess(picture.convert("RGB")), index

    loader = DataLoader(Frames(), batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=args.device == "cuda")
    module = module.to(args.device).eval().requires_grad_(False)
    writer = FrameCacheWriter(destination, paths, signature, indices, times)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16,
                enabled=args.device == "cuda" and args.precision == "bf16"):
            for step, (images, selected) in enumerate(loader):
                raw = module.feature_extractor(images.to(args.device, non_blocking=True))
                output = module.model(raw)
                logits = output if torch.is_tensor(output) else output.logits_labels
                raw32 = raw.float()
                writer.write(selected.numpy(), normalize(raw32, dim=1).cpu().numpy(),
                             raw32.norm(dim=1).cpu().numpy(),
                             logits.float().softmax(1)[:, weights.FAKE_COLUMN].cpu().numpy())
                if step % 100 == 0:
                    print(f"Cached frames: {int(writer.seen.sum())}/{len(paths)}", flush=True)
        if args.device == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        if crop_stat_hash(paths) != signature["crop_stat_sha256"]:
            raise ValueError("Crop files changed during extraction")
        measurements = {
            "inference_seconds": seconds, "images_per_second": len(paths) / seconds,
            "batch_size": args.batch_size, "num_workers": args.num_workers,
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved() if args.device == "cuda" else None,
            "device": args.device, "torch": torch.__version__, "cuda": torch.version.cuda,
        }
        writer.finish(measurements)
    finally:
        if writer.temporary is not None:
            writer.abort()
    print(f"Cache complete: {destination}", flush=True)
    if args.probabilities_out:
        export_probabilities(destination, args.probabilities_out)
    write_receipt(args.receipt, destination)


def write_receipt(path, destination):
    if path is None:
        return
    value = {"cache": str(destination.resolve()),
             "metadata_sha256": sha256_file(destination / "metadata.json")}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("Existing receipt points to a different cache; choose a new receipt")
        return
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)


if __name__ == "__main__":
    main()
