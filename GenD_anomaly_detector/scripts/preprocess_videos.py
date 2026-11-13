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


def iter_frames(path: str):
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            yield index, frame.to_ndarray(format="bgr24")


def detect_all(model, video_path: str):
    boxes, scores, landmarks, has_face = [], [], [], []
    for _, frame in iter_frames(video_path):
        det, kps = model.detect(frame)
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
    return {
        "has_face": np.asarray(has_face, bool),
        "boxes": np.asarray(boxes, np.float32).reshape(-1, 4),
        "scores": np.asarray(scores, np.float32),
        "landmarks": np.asarray(landmarks, np.float32).reshape(-1, 5, 2),
    }


def write_crops(video_path: str, selected, landmarks, out_dir: Path):
    import cv2

    add_gend_to_path()
    from detector import align_face

    wanted = set(selected)
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    written = []
    for index, frame in iter_frames(video_path):
        if index in wanted:
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

    record_path = Path(args.records) / (row.video_key + ".json")
    if record_path.exists() and json.loads(record_path.read_text(encoding="utf-8"))["status"] != "decode_error":
        return "skip"  # ok / no_face are final; decode_error is retried on the next run
    video_path = os.path.join(args.video_root, row.rel_path)
    start = time.time()
    record = {"rel_path": row.rel_path, "video_key": row.video_key, "split": row.split, "label": int(row.label),
              "frame_subset": row.frame_subset, "det_thresh": DET_THRESH, "nms_thresh": NMS_THRESH,
              "scale": SCALE, "target_size": list(TARGET_SIZE), "num_frames_target": FR.NUM_FRAMES,
              "train_frame_seed": FR.TRAIN_FRAME_SEED, "gend_commit": upstream_commit()}
    try:
        det = detect_all(model, video_path)
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
        written = write_crops(video_path, selected, det["landmarks"], Path(args.crops) / row.video_key) if selected else []
        if written != selected:
            raise AssertionError("Wrote {} of {} selected frames for {}".format(len(written), len(selected), row.rel_path))
        record.update(status="ok" if selected else "no_face", total_frames=int(len(det["has_face"])),
                      detected_frames=len(detected), selected=selected)
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
    parser.add_argument("--splits", nargs="*", default=["train", "val"], choices=["train", "val", "test"],
                        help="Default excludes test; add 'test' explicitly when it is time")
    parser.add_argument("--limit-per-label", type=int, default=None, help="Small smoke test: N real + N fake")
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--work", required=True, help="Writes crops/, records/, detections/ here")
    parser.add_argument("--retinaface", required=True, help="Path to det_10g.onnx")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    args = parser.parse_args()
    args.crops = os.path.join(args.work, "crops")
    args.records = os.path.join(args.work, "records")
    args.detections = os.path.join(args.work, "detections")

    videos = pd.read_csv(args.videos)
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
