"""Export at most six calibration windows: five paired original/crop images each."""
import argparse
import html
from pathlib import Path
import shutil
import sys
import tarfile

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gad.segment_training import read_json, write_json
from gad.diagnostic_protocol import sha256_file
from gad.segment_pilot import iter_window_frames, window_spec


def inside(root, relative):
    root = Path(root).resolve()
    if "\\" in str(relative) or ":" in str(relative):
        raise ValueError("Expected a relative POSIX path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError("Path escapes the registered root")
    return path


def validate_request(diagnostics, data, cache_run):
    diagnostics, data, cache_run = Path(diagnostics), Path(data), Path(cache_run)
    completion = read_json(diagnostics / "completion.json")
    request = read_json(diagnostics / "inspection_request.json")
    reg_path = diagnostics / "covariance_registration.json"
    if (not completion.get("completed") or completion.get("phase") != "development_clip_covariance"
            or completion["registration_sha256"] != sha256_file(reg_path)
            or request["covariance_registration_sha256"] != sha256_file(reg_path)):
        raise ValueError("Use a completed, locked covariance-control run")
    for name in ("inspection_targets.csv", "inspection_request.json", "training_units.csv"):
        if sha256_file(diagnostics / name) != completion["artifacts_sha256"][name]:
            raise ValueError("Inspection input changed: " + name)
    if (request["target_sha256"] != sha256_file(diagnostics / "inspection_targets.csv")
            or request["frames_per_clip"] != 5
            or request["data_plan_sha256"] != sha256_file(data / "segment_data_plan.json")
            or request["candidate_list_registration_sha256"] != sha256_file(cache_run / "frame_lists.json")):
        raise ValueError("Inspection request/data provenance changed")
    targets = pd.read_csv(diagnostics / "inspection_targets.csv", dtype=str, keep_default_na=False)
    units = pd.read_csv(diagnostics / "training_units.csv", dtype=str, keep_default_na=False).set_index("video_key")
    if (not 1 <= len(targets) <= 6 or targets.video_key.duplicated().any()
            or not targets.head_role.eq("calibration").all() or not targets.label.eq("0").all()):
        raise ValueError("Inspection is limited to at most six genuine calibration units")
    for row in targets.to_dict("records"):
        if row["video_key"] not in units.index:
            raise ValueError("Inspection unit not in original training-unit table")
        for name in units.columns:
            if row.get(name) != units.loc[row["video_key"], name]:
                raise ValueError("Inspection unit metadata changed: " + name)
    lists = read_json(cache_run / "frame_lists.json")
    if sha256_file(data / "faces/pilot_run_contract.json") != lists["preprocessing_contract_sha256"]:
        raise ValueError("Preprocessing contract changed")
    return targets, lists


def run(diagnostics, data, cache_run, video_root, out, decoder=None):
    diagnostics, data, out = Path(diagnostics), Path(data), Path(out)
    archive = Path(str(out) + ".tar.gz")
    if out.exists() or archive.exists() or Path(str(archive) + ".sha256").exists():
        raise FileExistsError("Use a new inspection output and archive path")
    targets, lists = validate_request(diagnostics, data, cache_run)
    prepared = []
    for row in targets.to_dict("records"):
        key = row["video_key"]
        record_path = inside(data / "faces/records", key + ".json")
        if sha256_file(record_path) != lists["records_sha256"][key]:
            raise ValueError("Selected preprocessing record changed")
        record = read_json(record_path)
        window = window_spec(row)
        if (record["status"] != "ok" or record["video_key"] != key or record["rel_path"] != row["rel_path"]
                or any(record.get(k) != v for k, v in window.items())
                or len(record["selected"]) != 32 or len(record["selected_timestamps_seconds"]) != 32):
            raise ValueError("Inspection record differs from selected window")
        video = inside(video_root, row["rel_path"])
        stat = video.stat()
        if (stat.st_size, stat.st_mtime_ns) != (window["source_bytes"], window["source_mtime_ns"]):
            raise ValueError("Original video changed since preprocessing")
        positions = np.linspace(0, 31, 5).round().astype(int)
        samples = [(record["selected"][p], float(record["selected_timestamps_seconds"][p])) for p in positions]
        for index, _ in samples:
            crop = inside(data / "faces/crops", key + "/frame_%05d.png" % index)
            with Image.open(crop) as im:
                if im.size != (224, 224):
                    raise ValueError("Unexpected crop dimensions")
        prepared.append((row, record_path, window, video, samples))
    out.mkdir(parents=True)
    shutil.copy2(diagnostics / "inspection_targets.csv", out / "inspection_targets.csv")
    panels, manifest = [], []
    decoder = iter_window_frames if decoder is None else decoder
    for clip_number, (row, record_path, window, video, samples) in enumerate(prepared, 1):
        folder = out / ("clip_%02d" % clip_number)
        folder.mkdir()
        shutil.copy2(record_path, folder / "preprocessing_record.json")
        wanted = dict(samples)
        sheet = Image.new("RGB", (5*320, 500), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 5), "Original frames (top); stored GenD crops (bottom). Quality review only.", fill="black")
        seen = set()
        rows = []
        frames = decoder(str(video), window["segment_start_seconds"], window["segment_end_seconds"], window["stream_start_seconds"])
        try:
            for index, bgr, timestamp in frames:
                if index not in wanted:
                    continue
                if index in seen or not np.isclose(timestamp, wanted[index], rtol=0, atol=1e-7):
                    raise ValueError("Raw-frame ordinal/PTS no longer matches the crop record")
                seen.add(index)
                column = list(wanted).index(index)
                raw = Image.fromarray(np.asarray(bgr)[:, :, ::-1].copy()).convert("RGB")
                raw.save(folder / ("raw_%05d.jpg" % index), quality=90)
                crop_path = inside(data / "faces/crops", row["video_key"] + "/frame_%05d.png" % index)
                shutil.copy2(crop_path, folder / crop_path.name)
                with Image.open(crop_path) as im:
                    crop = im.convert("RGB")
                sheet.paste(ImageOps.contain(raw, (310, 210)), (column*320+5, 30))
                sheet.paste(crop, (column*320+48, 255))
                draw.text((column*320+5, 480), "Frame %d; video %.3fs" % (index, timestamp-window["stream_start_seconds"]), fill="black")
                rows.append({"frame_index": index, "pts_seconds": float(timestamp),
                             "stored_crop_sha256": sha256_file(crop_path)})
                if len(seen) == len(wanted):
                    break
        finally:
            if hasattr(frames, "close"):
                frames.close()
        if seen != set(wanted):
            raise ValueError("Some selected raw frames could not be recovered")
        stat = video.stat()
        if (stat.st_size, stat.st_mtime_ns) != (window["source_bytes"], window["source_mtime_ns"]):
            raise ValueError("Original video changed during inspection")
        sheet.save(folder / "paired_contact_sheet.jpg", quality=92)
        manifest.append({"video_key": row["video_key"], "rel_path": row["rel_path"],
            "window": window, "record_sha256": sha256_file(record_path), "frames": rows})
        panels.append('<h2>%s</h2><p>%s | %.3f–%.3fs</p><img src="clip_%02d/paired_contact_sheet.jpg">' % (
            html.escape(row["dataset"]), html.escape(row["rel_path"]), window["segment_start_seconds"], window["segment_end_seconds"], clip_number))
        print("Exported %d/%d calibration windows" % (clip_number, len(prepared)), flush=True)
    (out / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Calibration quality review</title>'
        '<style>body{font:16px system-ui;margin:24px}img{max-width:100%;border:1px solid #ccc}</style>'
        '<h1>Calibration quality review</h1><p>Five matching original/crop frames per window. '
        'This sample cannot establish full-window quality. Do not remove samples based on high scores.</p>' + ''.join(panels), encoding="utf-8")
    write_json(out / "manifest.json", {"diagnostics_registration_sha256": sha256_file(diagnostics / "covariance_registration.json"),
        "exporter_sha256": sha256_file(__file__), "clips": manifest, "final_test_opened": False})
    write_json(out / "completion.json", {"completed": True, "clips": len(manifest),
        "files_sha256": {p.relative_to(out).as_posix(): sha256_file(p) for p in sorted(out.rglob("*")) if p.is_file()}})
    with tarfile.open(archive, "x:gz") as tar:
        tar.add(out, arcname=out.name)
    Path(str(archive) + ".sha256").write_text(sha256_file(archive) + "  " + archive.name + "\n", encoding="ascii")
    print("Saved limited inspection archive: " + str(archive), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("diagnostics", "data", "cache-run", "video-root", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.diagnostics, args.data, args.cache_run, args.video_root, args.out)
