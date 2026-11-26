"""Register all existing candidate crops for reusable inference, never test/training."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def prepare(plan_path, faces, out):
    plan_path, faces, out = Path(plan_path).resolve(), Path(faces).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError("Use a new candidate-cache registration directory")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (plan.get("version") != 1 or plan.get("purpose") !=
            "registered bounded candidate preprocessing; no training or final testing"
            or plan.get("table") != "candidate_videos.csv"):
        raise ValueError("Expected the registered bounded candidate plan")
    names = ("parents.csv", "selected_originals.csv", "reserved_test.csv",
             "unused_development.csv", "candidate_videos.csv")
    for name in names:
        if plan.get("registered_files", {}).get(name) != digest(plan_path.parent / name):
            raise ValueError("Registered table changed: " + name)
    if plan["table_sha256"] != digest(plan_path.parent / plan["table"]):
        raise ValueError("Candidate table changed")
    contract_path = faces / "pilot_run_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract["pilot_plan_sha256"] != digest(plan_path):
        raise ValueError("Preprocessing belongs to a different plan")
    contract_hash = digest(contract_path)
    rows = read_csv(plan_path.parent / plan["table"])
    if not rows or len(rows) != plan["candidate_clips"]:
        raise ValueError("Candidate count disagrees with registration")
    summary = json.loads((faces / "pilot_summary.json").read_text(encoding="utf-8"))
    if (not summary["all_clips_attempted"] or summary["clips_attempted"] != len(rows)
            or summary["clips_planned"] != len(rows)):
        raise ValueError("Preprocessing is incomplete")
    if sum(g["failed_clips"] for g in summary["groups"]):
        raise ValueError("Resolve failed preprocessing records before extraction")
    sources, keys, lines, records, unusable, low_coverage = {}, set(), [], {}, [], []
    coverage = {role: {"candidates": 0, "frames": 0, "usable_keys": [], "no_face_keys": []}
                for role in ("train", "calibration", "dev")}
    for row in sorted(rows, key=lambda r: (r["split"], r["video_key"])):
        role, key, label = row["split"], row["video_key"], int(row["label"])
        parts = PurePosixPath(key)
        if (role not in coverage or len(parts.parts) != 3 or parts.as_posix() != key
                or parts.is_absolute() or any(p in (".", "..") for p in parts.parts)
                or "\\" in key or ":" in key or key in keys):
            raise ValueError("Invalid, duplicate or test candidate key")
        if label not in (0, 1) or (role != "dev" and label != 0):
            raise ValueError("Only genuine A and registered labeled development candidates are allowed")
        if ("real" in parts.parts[-2]) != (label == 0):
            raise ValueError("Candidate path disagrees with GenD label convention")
        source = row["source_clip"]
        if not source or sources.setdefault(source, role) != role:
            raise ValueError("Missing source group or source crosses roles")
        keys.add(key)
        path = faces / "records" / (key + ".json")
        record = json.loads(path.read_text(encoding="utf-8"))
        records[key] = digest(path)
        fields = ("rel_path", "video_key", "split", "parent_video_key", "clip_id", "frame_subset")
        if any(record.get(f) != row[f] for f in fields) or row["frame_subset"] != "uniform32":
            raise ValueError("Record does not match the candidate table: " + key)
        integers = ("source_bytes", "source_mtime_ns")
        seconds = ("segment_start_seconds", "segment_end_seconds", "stream_start_seconds")
        if (any(int(record[f]) != int(row[f]) for f in integers)
                or any(float(record[f]) != float(row[f]) for f in seconds)):
            raise ValueError("Record source/window fingerprint changed: " + key)
        if (record["pilot_contract_sha256"] != contract_hash or record["label"] != label
                or record["detection_policy_details"] != contract["detection_policy"]
                or record["target_size"] != [224, 224]):
            raise ValueError("Record preprocessing contract changed: " + key)
        selected, times = record["selected"], record["selected_timestamps_seconds"]
        if (len(selected) != len(times) or len(selected) > 32
                or any(type(i) is not int or not 0 <= i < record["total_frames"] for i in selected)
                or any(a >= b for a, b in zip(selected, selected[1:]))):
            raise ValueError("Invalid selected frame indices/timestamps: " + key)
        start = float(row["segment_start_seconds"]) + float(row["stream_start_seconds"])
        end = float(row["segment_end_seconds"]) + float(row["stream_start_seconds"])
        if (any(not math.isfinite(t) or not start - 1e-7 <= t < end + 1e-7 for t in times)
                or any(a > b for a, b in zip(times, times[1:]))):
            raise ValueError("Invalid frame presentation times: " + key)
        coverage[role]["candidates"] += 1
        if record["status"] == "no_face" and not selected:
            coverage[role]["no_face_keys"].append(key)
            unusable.append(key)
            continue
        if record["status"] != "ok" or not selected:
            raise ValueError("Unfinished or failed crop record: " + key)
        for index in selected:
            crop = faces / "crops" / key / ("frame_%05d.png" % index)
            if not crop.is_file():
                raise FileNotFoundError(crop)
            lines.append(crop.as_posix())
        coverage[role]["usable_keys"].append(key)
        coverage[role]["frames"] += len(selected)
        if not record["sustained_face_candidate"]:
            low_coverage.append(key)
    if not all(v["usable_keys"] for v in coverage.values()):
        raise ValueError("No scorable candidates in one of the required roles")
    if len(lines) != sum(g["selected_frames"] for g in summary["groups"]):
        raise ValueError("Selected crop counts disagree with preprocessing summary")
    out.mkdir(parents=True)
    listing = out / "candidates.txt"
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    registration = {
        "purpose": "Reusable candidate features only; not a head-training or final-test registration",
        "plan": str(plan_path), "plan_sha256": digest(plan_path),
        "preprocessing_contract_sha256": contract_hash, "records_sha256": records,
        "coverage": coverage, "files": {listing.name: digest(listing)},
        "low_coverage_included_keys": low_coverage, "unscorable_no_face_keys": unusable,
        "frame_count": len(lines), "scorable_candidates": len(keys) - len(unusable),
        "policy": "Include every selected crop, including low coverage. No test images or model-score selection."
    }
    (out / "frame_lists.json").write_text(json.dumps(registration, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: registration[k] for k in ("purpose", "frame_count", "scorable_candidates")}, indent=2))
    return registration


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--faces", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.plan, args.faces, args.out)
