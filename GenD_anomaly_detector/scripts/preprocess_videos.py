"""Face detection on every frame, then GenD-style crops of a frame subset.

Follows GenD (EXPERIMENT_PLAN.md section 0):
  RetinaFace det_10g.onnx, det threshold 0.5, NMS 0.4, input 640x640,
  largest face, 5-point alignment (upstream ``detector.align_face``),
  scale 1.3, 224x224, PNG. Subset drawn from the frames with a detected face:
  random-32 for training videos, uniform-32 for validation/test videos.

Outputs per video:
  <crops>/<video_key>/frame_XXXXX.png
  <records>/<video_key>.json   counts and selected frame indices
  <detections>/<video_key>.npz per-frame detection cache (box, score, landmarks)

Resumable: a video whose record says ok / no_face is skipped; decode_error
records are retried, and detector/runtime failures write no record (so they
are retried too). The run exits non-zero if any video failed. Shardable for Slurm
arrays with --num-shards / --shard-index.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gad import frames as FR  # noqa: E402
from gad.upstream import add_gend_to_path, upstream_commit  # noqa: E402
from gad.segment_pilot import (iter_window_frames, window_spec, temporal_face_coverage,
                               sustained_candidate)  # noqa: E402
from gad.face_detection import POLICIES, detect_with_policy, policy_spec  # noqa: E402

DET_THRESH = 0.5
NMS_THRESH = 0.4
DET_INPUT = (640, 640)
SCALE = 1.3
TARGET_SIZE = (224, 224)


def build_detector(model_file: str, cpu: bool):
    add_gend_to_path()
    from src.retinaface import RetinaFace

    providers = ["CPUExecutionProvider"] if cpu else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    model = RetinaFace(model_file, providers=providers)
    model.prepare(ctx_id=0, nms_thresh=NMS_THRESH, input_size=DET_INPUT, det_thresh=DET_THRESH)
    return model


def iter_timed_frames(path: str, window=None):
    if window is not None:
        yield from iter_window_frames(path, window["segment_start_seconds"],
                                      window["segment_end_seconds"], window["stream_start_seconds"])
        return
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            yield index, frame.to_ndarray(format="bgr24"), (float(frame.time) if frame.time is not None else np.nan)


def iter_frames(path: str):
    for index, image, _ in iter_timed_frames(path):
        yield index, image


def detect_all(model, video_path: str, window=None, detection_policy="none"):
    policy_spec(detection_policy)
    boxes, scores, landmarks, has_face, timestamps = [], [], [], [], []
    fallback = {name: [] for name in ("fallback_attempted", "fallback_used", "fallback_rejected")}
    for _, frame, timestamp in iter_timed_frames(video_path, window):
        timestamps.append(timestamp)
        det, kps, audit = detect_with_policy(model, frame, detection_policy)
        for name in fallback:
            fallback[name].append(audit[name])
        if len(det) == 0:
            has_face.append(False)
            boxes.append(np.full(4, np.nan, np.float32))
            scores.append(np.nan)
            landmarks.append(np.full((5, 2), np.nan, np.float32))
            continue
        areas = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])  # upstream: largest face
        best = int(np.argmax(areas))
        has_face.append(True)
        boxes.append(det[best, :4].astype(np.float32))
        scores.append(float(det[best, 4]))
        landmarks.append(kps[best].astype(np.float32))
    if window is not None and not has_face:
        raise ValueError("No decoded frames in the audited window; check duration/PTS alignment")
    result = {
        "has_face": np.asarray(has_face, bool),
        "boxes": np.asarray(boxes, np.float32).reshape(-1, 4),
        "scores": np.asarray(scores, np.float32),
        "landmarks": np.asarray(landmarks, np.float32).reshape(-1, 5, 2),
        "timestamps_seconds": np.asarray(timestamps, np.float64),
    }
    if detection_policy != "none":
        result.update({name: np.asarray(values, np.int32 if name == "fallback_rejected" else bool)
                       for name, values in fallback.items()})
    return result


def write_crops(video_path: str, selected, landmarks, out_dir: Path, window=None, expected_timestamps=None):
    import cv2

    add_gend_to_path()
    from detector import align_face

    wanted = set(selected)
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    written = []
    for index, frame, timestamp in iter_timed_frames(video_path, window):
        if index in wanted:
            if expected_timestamps is not None and abs(timestamp - expected_timestamps[index]) > 1e-7:
                raise ValueError("Window seek produced different PTS on the crop pass")
            face, _ = align_face(frame, landmarks[index], target_size=TARGET_SIZE, scale=SCALE)
            if not cv2.imwrite(str(tmp / FR.frame_file_name(index)), face):
                raise IOError("Failed to write crop {} of {}".format(index, video_path))
            written.append(index)
        if index >= max(wanted):
            break
    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.replace(tmp, out_dir)
    return written


def process(row, args, model):
    import av

    detection_policy = getattr(args, "detection_policy", "none")
    policy = policy_spec(detection_policy)
    window = window_spec(row) if getattr(row, "sample_unit", "") == "segment_pilot" else None
    if window is not None:
        window["pilot_contract_sha256"] = args.pilot_contract_sha256
    record_path = Path(args.records) / (row.video_key + ".json")
    video_path = os.path.join(args.video_root, row.rel_path)
    if window is not None:
        stat = Path(video_path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (window["source_bytes"], window["source_mtime_ns"]):
            raise ValueError("Video changed since duration audit; create a new audit and pilot")
    if record_path.exists():
        existing = json.loads(record_path.read_text(encoding="utf-8"))
        if existing.get("detection_policy", "none") != detection_policy:
            raise ValueError("Detection policy changed; use a new preprocessing directory")
        if window is not None and any(existing.get(k) != value for k, value in window.items()):
            raise ValueError("Pilot preprocessing record belongs to a different window or run contract")
        if existing["status"] != "decode_error":
            return "skip"  # ok / no_face are final; decode_error is retried on the next run
    start = time.time()
    record = {"rel_path": row.rel_path, "video_key": row.video_key, "split": row.split, "label": int(row.label),
              "frame_subset": row.frame_subset, "det_thresh": DET_THRESH, "nms_thresh": NMS_THRESH,
              "scale": SCALE, "target_size": list(TARGET_SIZE), "num_frames_target": FR.NUM_FRAMES,
              "train_frame_seed": FR.TRAIN_FRAME_SEED, "gend_commit": upstream_commit()}
    record.update(detection_policy=detection_policy, detection_policy_details=policy)
    if window is not None:
        record.update(window)
    try:
        det = detect_all(model, video_path, window, detection_policy)
    except av.error.FFmpegError as error:  # unreadable video: record it, never silently drop
        record.update(status="decode_error", error=repr(error), total_frames=0, detected_frames=0, selected=[])
    except Exception as error:  # detector/runtime failure: no record, so a rerun retries this video
        print("ERROR (not recorded, will be retried): {}: {!r}".format(row.rel_path, error), flush=True)
        return "runtime_error"
    else:
        det_path = Path(args.detections) / (row.video_key + ".npz")
        det_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(det_path, **det)
        detected = np.flatnonzero(det["has_face"]).tolist()
        selected = FR.select_frames(row.frame_subset, detected, row.rel_path)
        written = write_crops(video_path, selected, det["landmarks"], Path(args.crops) / row.video_key,
                              window, det["timestamps_seconds"] if window is not None else None) if selected else []
        if written != selected:
            raise AssertionError("Wrote {} of {} selected frames for {}".format(len(written), len(selected), row.rel_path))
        record.update(status="ok" if selected else "no_face", total_frames=int(len(det["has_face"])),
                      detected_frames=len(detected), selected=selected,
                      selected_timestamps_seconds=[float(det["timestamps_seconds"][i])
                          if np.isfinite(det["timestamps_seconds"][i]) else None for i in selected])
        record.update(
            detected_fraction=len(detected) / len(det["has_face"]) if len(det["has_face"]) else 0.,
            selected_frame_count=len(selected),
            fallback_attempted_frames=int(det["fallback_attempted"].sum()) if "fallback_attempted" in det else 0,
            fallback_recovered_frames=int(det["fallback_used"].sum()) if "fallback_used" in det else 0,
            fallback_rejected_candidates=int(det["fallback_rejected"].sum()) if "fallback_rejected" in det else 0,
            selected_fallback_frames=int(det["fallback_used"][selected].sum()) if "fallback_used" in det else 0,
        )
        if window is not None:
            coverage = temporal_face_coverage(
                det["timestamps_seconds"], det["has_face"],
                window["stream_start_seconds"] + window["segment_start_seconds"],
                window["stream_start_seconds"] + window["segment_end_seconds"],
            )
            record.update(temporal_face_coverage=coverage,
                          sustained_face_candidate=sustained_candidate(len(selected), coverage))
    if window is not None:
        stat = Path(video_path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (window["source_bytes"], window["source_mtime_ns"]):
            raise ValueError("Video changed during pilot preprocessing; no successful record published")
    record["seconds"] = round(time.time() - start, 2)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = record_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record), encoding="utf-8")
    os.replace(tmp, record_path)
    return record["status"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", required=True, help="videos.csv from build_video_table.py")
    parser.add_argument("--membership", default=None, help="membership.csv, needed with --groups")
    parser.add_argument("--groups", type=int, nargs="*", default=None, help="Only videos used by these groups")
    parser.add_argument("--splits", nargs="*", default=["train", "val"],
                        choices=["train", "val", "calibration", "dev", "test"],
                        help="Default excludes test; add 'test' explicitly when it is time")
    parser.add_argument("--limit-per-label", type=int, default=None, help="Small smoke test: N real + N fake")
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--work", required=True, help="Writes crops/, records/, detections/ here")
    parser.add_argument("--retinaface", required=True, help="Path to det_10g.onnx")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--detection-policy", choices=POLICIES, default="none",
                        help="Opt-in pad_on_empty_v1 fallback; use a new work directory")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    args = parser.parse_args()
    args.crops = os.path.join(args.work, "crops")
    args.records = os.path.join(args.work, "records")
    args.detections = os.path.join(args.work, "detections")

    videos = pd.read_csv(args.videos)
    if "sample_unit" in videos and videos["sample_unit"].eq("segment_pilot").any():
        parser.error("Use preprocess_segment_pilot.py --pilot-plan for window verification and timing")
    videos = videos[videos["split"].isin(args.splits)]
    if args.groups:
        member = pd.read_csv(args.membership)
        keep = set(member[member["group"].isin(args.groups)]["rel_path"])
        videos = videos[videos["rel_path"].isin(keep)]
    if args.limit_per_label:
        videos = videos.sort_values("rel_path").groupby(["split", "label"]).head(args.limit_per_label)
    videos = videos.sort_values("rel_path").reset_index(drop=True)
    videos = videos.iloc[args.shard_index::args.num_shards]
    print("shard {}/{}: {} videos, splits={}".format(args.shard_index, args.num_shards, len(videos), args.splits))

    model = build_detector(args.retinaface, args.cpu)
    counts = {}
    for i, row in enumerate(videos.itertuples(index=False), 1):
        status = process(row, args, model)
        counts[status] = counts.get(status, 0) + 1
        if i % 20 == 0 or i == len(videos):
            print("{}/{} {}".format(i, len(videos), counts), flush=True)
    if counts.get("runtime_error") or counts.get("decode_error"):
        raise SystemExit("{} runtime errors and {} decode errors; rerun the same command to retry them".format(
            counts.get("runtime_error", 0), counts.get("decode_error", 0)))


if __name__ == "__main__":
    main()
