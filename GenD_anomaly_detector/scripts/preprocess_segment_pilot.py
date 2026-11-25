"""Time a registered segment preprocessing pilot, without training or test inference."""
import argparse
import json
from pathlib import Path
import socket
import sys
import time

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gad.segment_pilot import VERSION, window_spec
from gad.upstream import upstream_commit
from audit_video_durations import atomic_text, digest, write_csv
from preprocess_videos import build_detector, process


def load_pilot(plan_path):
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("version") != VERSION or plan.get("purpose") != "preprocessing timing and face-coverage pilot only":
        raise ValueError("Not a supported segment pilot")
    table = plan_path.parent / "pilot_videos.csv"
    if digest(table) != plan["pilot_table_sha256"]:
        raise ValueError("Pilot table changed after registration")
    rows = pd.read_csv(table, dtype=str, keep_default_na=False)
    if rows.empty or rows["video_key"].duplicated().any() or rows["clip_id"].duplicated().any():
        raise ValueError("Empty pilot or duplicate windows")
    if not rows["split"].isin(("train", "calibration", "dev")).all():
        raise ValueError("Test preprocessing is forbidden in this timing pilot")
    if (rows.groupby("parent_video_key")["split"].nunique() > 1).any():
        raise ValueError("An original crosses pilot roles")
    if (rows.groupby("source_clip")["split"].nunique() > 1).any():
        raise ValueError("A source family crosses pilot roles")
    for row in rows.itertuples(index=False):
        window_spec(row)
    rows["label"] = rows["label"].astype(int)
    return plan, rows


def summarize(rows, measurements, startup_seconds, invocation_seconds):
    groups = []
    for (role, dataset, label), subset in rows.groupby(["split", "dataset", "label"]):
        selected = {r.video_key for r in subset.itertuples(index=False)}
        records = [r for r in measurements if r["video_key"] in selected]
        completed = [r for r in records if r.get("status") in ("ok", "no_face")]
        fresh = [r for r in completed if not r["reused"]]
        groups.append({
            "split": role, "dataset": dataset, "label": int(label), "clips_planned": len(subset),
            "original_videos": subset.parent_video_key.nunique(), "source_groups": subset.source_clip.nunique(),
            "ok_clips": sum(r["status"] == "ok" for r in records),
            "no_face_clips": sum(r["status"] == "no_face" for r in records),
            "failed_clips": len(records) - len(completed),
            "selected_frames": sum(int(r.get("selected_frames", 0)) for r in completed),
            "fresh_clips": len(fresh),
            "fresh_processing_seconds": sum(float(r.get("seconds", 0)) for r in fresh),
            "fresh_video_seconds": sum(float(r["window_seconds"]) for r in fresh),
            "fresh_decoded_frames": sum(int(r.get("total_frames", 0)) for r in fresh),
        })
    return {"purpose": "preprocessing timing and face coverage, not detection accuracy",
            "clips_planned": len(rows), "clips_attempted": len(measurements),
            "all_clips_attempted": len(measurements) == len(rows),
            "detector_startup_seconds": startup_seconds, "invocation_seconds": invocation_seconds,
            "groups": groups,
            "limitation": "Small source-stratified pilot; long-video clips are correlated. No formal "
                          "training/early-stop/calibration or final scoring scheme is frozen. "
                          "Timing includes decoding, face detection, crop writing and I/O; reused clips excluded."}


def run(args):
    import av
    import cv2
    import onnxruntime

    started = time.perf_counter()
    plan, rows = load_pilot(args.pilot_plan)
    root, work = Path(args.video_root).resolve(), Path(args.work)
    if not root.is_dir():
        raise ValueError("Video root is not mounted")
    here = Path(__file__).resolve().parent
    contract = {"pilot_plan_sha256": digest(args.pilot_plan), "video_root": str(root),
                "detector_sha256": digest(args.retinaface), "cpu": args.cpu,
                "runtime": {"python": sys.version, "pyav": av.__version__,
                            "opencv": cv2.__version__, "onnxruntime": onnxruntime.__version__},
                "gend_commit": upstream_commit(),
                "code_sha256": {name: digest(here / name) for name in
                                ("preprocess_videos.py", "preprocess_segment_pilot.py")},
                "window_code_sha256": digest(here.parent / "gad/segment_pilot.py")}
    for row in rows.itertuples(index=False):
        path = (root / row.rel_path).resolve()
        path.relative_to(root)
        stat, spec = path.stat(), window_spec(row)
        if (stat.st_size, stat.st_mtime_ns) != (spec["source_bytes"], spec["source_mtime_ns"]):
            raise ValueError("Video changed after duration audit: " + row.rel_path)
    if work.exists():
        if not args.resume or not (work / "pilot_run_contract.json").is_file():
            raise FileExistsError("Use a new work directory, or --resume with the same pilot contract")
        if json.loads((work / "pilot_run_contract.json").read_text(encoding="utf-8")) != contract:
            raise ValueError("Pilot run inputs/code changed; use a new work directory")
    else:
        work.mkdir(parents=True)
        atomic_text(work / "pilot_run_contract.json", json.dumps(contract, indent=2))
    args.pilot_contract_sha256 = digest(work / "pilot_run_contract.json")
    args.video_root = str(root)
    for name in ("crops", "records", "detections"):
        setattr(args, name, str(work / name))
    loading = time.perf_counter()
    model = build_detector(str(args.retinaface), args.cpu)
    startup = time.perf_counter() - loading
    providers = model.session.get_providers()
    if not args.cpu and "CUDAExecutionProvider" not in providers:
        raise RuntimeError("RetinaFace CUDA provider is unavailable; refusing a silently CPU-only GPU timing run")
    print(json.dumps({"host": socket.gethostname(), "detector_providers": providers,
                      "runtime": contract["runtime"]}), flush=True)
    measurements = []
    for index, row in enumerate(rows.itertuples(index=False), 1):
        tick = time.perf_counter()
        measurement = {"video_key": row.video_key, "parent_video_key": row.parent_video_key,
                       "dataset": row.dataset, "rel_path": row.rel_path, "source_clip": row.source_clip,
                       "window_seconds": float(row.segment_end_seconds) - float(row.segment_start_seconds),
                       "reused": False}
        try:
            status = process(row, args, model)
            measurement["reused"] = status == "skip"
            record_path = Path(args.records) / (row.video_key + ".json")
            record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
            measurement.update({name: record[name] for name in
                                ("status", "total_frames", "detected_frames", "seconds", "error") if name in record})
            measurement.setdefault("status", status)
            measurement["selected_frames"] = len(record.get("selected", []))
            measurement["selected_timestamps_seconds"] = record.get("selected_timestamps_seconds", [])
        except Exception as error:
            measurement.update(status="runtime_error", error=repr(error))
        measurement["invocation_seconds"] = time.perf_counter() - tick
        measurements.append(measurement)
        write_csv(work / "pilot_measurements.csv", measurements)
        summary = summarize(rows, measurements, startup, time.perf_counter() - started)
        summary.update(host=socket.gethostname(), detector_providers=providers, runtime=contract["runtime"])
        atomic_text(work / "pilot_summary.json", json.dumps(summary, indent=2))
        filled = int(30 * index / len(rows))
        elapsed = time.perf_counter() - loading - startup
        print(f"[{'#' * filled}{'-' * (30 - filled)}] {index}/{len(rows)} "
              f"{row.dataset}: {measurement['status']}; "
              f"ETA {elapsed / index * (len(rows) - index) / 60:.1f} min", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    if any(r["status"] not in ("ok", "no_face") for r in measurements):
        raise SystemExit("Pilot contains failures; inspect pilot_measurements.csv and resume the same run")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-plan", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--retinaface", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
