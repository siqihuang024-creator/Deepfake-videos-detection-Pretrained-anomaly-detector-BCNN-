"""Audit actual local video headers on CPU, with bounded probes and safe resume.

Input is an inventory CSV, or the metadata tar containing video_inventory.csv.
This does not extract archives or decode all video frames. Header durations and
frame rates are estimates, not a full video integrity check or proof of CFR.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gad.video_probe import ffprobe_metadata, pyav_metadata


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def inventory_rows(path):
    path = Path(path)
    if path.name.lower().endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(path, "r:*") as bundle:
            matches = [m for m in bundle.getmembers()
                       if m.isfile() and PurePosixPath(m.name).name == "video_inventory.csv"]
            if len(matches) != 1 or matches[0].size > 64 * 1024 * 1024:
                raise ValueError("Expected one bounded video_inventory.csv in the metadata bundle")
            content = bundle.extractfile(matches[0]).read().decode("utf-8-sig")
    else:
        content = path.read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(content)))
    if not rows or not {"dataset", "rel_path"}.issubset(rows[0]):
        raise ValueError("Inventory needs dataset and rel_path columns")
    seen = set()
    for row in rows:
        p = PurePosixPath(row["rel_path"])
        if (not row["dataset"] or not p.parts or p.is_absolute() or ".." in p.parts
                or "\\" in row["rel_path"] or ":" in row["rel_path"]
                or p.as_posix() != row["rel_path"] or row["rel_path"] in seen):
            raise ValueError("Empty, unsafe or duplicate inventory path: " + row["rel_path"])
        seen.add(row["rel_path"])
    return sorted(rows, key=lambda row: (row["dataset"], row["rel_path"]))


def fingerprint(root, row):
    path = (Path(root) / row["rel_path"]).resolve()
    try:
        path.relative_to(Path(root).resolve())
    except ValueError:
        raise ValueError("Video resolves outside the video root: " + row["rel_path"])
    stat = path.stat()
    if not path.is_file() or stat.st_size <= 0:
        raise ValueError("Not a nonempty video file")
    if row.get("bytes") and stat.st_size != int(row["bytes"]):
        raise ValueError("File size differs from the audited inventory")
    return path, {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def probe(path, backend, timeout):
    if backend == "ffprobe":
        command = ["ffprobe", "-v", "error", "-threads", "1", "-select_streams", "v:0",
                   "-show_streams", "-show_format", "-of", "json", str(path)]
    else:
        command = [sys.executable, str(Path(__file__).resolve()), "--probe-one", str(path)]
    result = subprocess.run(command, capture_output=True, timeout=timeout, check=True)
    payload = json.loads(result.stdout.decode("utf-8"))
    return ffprobe_metadata(payload) if backend == "ffprobe" else payload


def probe_row(root, row, backend, timeout):
    result = {"dataset": row["dataset"], "rel_path": row["rel_path"], "backend": backend}
    started = time.perf_counter()
    try:
        path, before = fingerprint(root, row)
        result.update(before)
        metadata = probe(path, backend, timeout)
        if fingerprint(root, row)[1] != before:
            raise ValueError("Video changed during probing")
        result.update(metadata, status="ok", error="")
    except Exception as error:
        result.update(status="error", error=str(error)[:1000])
    result["probe_seconds"] = round(time.perf_counter() - started, 4)
    return result


def quantiles(values):
    values = sorted(values)
    def at(q):
        position = (len(values) - 1) * q
        lower = int(position)
        return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)
    return {str(q): at(q) for q in (0, .1, .5, .9, .95, .99, 1)} if values else {}


def summaries(rows):
    result = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        subset = [r for r in rows if r["dataset"] == dataset]
        ok = [r for r in subset if r["status"] == "ok"]
        durations = [float(r["duration_seconds"]) for r in ok]
        result[dataset] = {
            "files": len(subset), "ok": len(ok), "errors": len(subset) - len(ok),
            "header_hours": sum(durations) / 3600, "duration_seconds_quantiles": quantiles(durations),
            "fps_average_quantiles": quantiles([float(r["fps_average"]) for r in ok if r.get("fps_average")]),
            "unknown_stream_starts": sum(r.get("stream_start_seconds") is None for r in ok),
            "container_duration_fallbacks": sum(r["duration_source"] != "video_stream" for r in ok),
        }
    return result


def write_csv(path, rows):
    fields = ["dataset", "rel_path", "status", "error"]
    fields += sorted(set().union(*(set(row) for row in rows)) - set(fields))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def audit(inventory, video_root, out, workers=4, timeout=30, backend="auto", resume=False):
    if workers < 1 or workers > 16 or timeout <= 0:
        raise ValueError("Use 1..16 workers and a positive per-file timeout")
    inventory, video_root, out = Path(inventory), Path(video_root).resolve(), Path(out)
    rows = inventory_rows(inventory)
    if not video_root.is_dir():
        raise ValueError("Video root is not mounted: " + str(video_root))
    backend = ("ffprobe" if shutil.which("ffprobe") else "pyav") if backend == "auto" else backend
    # Fail once, before launching thousands of children, if the backend is unavailable.
    if backend == "ffprobe":
        backend_version = subprocess.run(["ffprobe", "-version"], capture_output=True, check=True,
                                         timeout=timeout).stdout.decode("utf-8", errors="replace").splitlines()[0]
    else:
        backend_version = subprocess.run([sys.executable, "-c", "import av; print(av.__version__)"],
                                        capture_output=True, check=True, timeout=timeout).stdout.decode().strip()
    contract = {"version": 1, "inventory_sha256": digest(inventory), "video_root": str(video_root),
                "backend": backend, "backend_version": backend_version,
                "python": sys.version, "probe_code_sha256": digest(Path(__file__)),
                "helper_code_sha256": digest(Path(__file__).parents[1] / "gad/video_probe.py")}
    old = {}
    if out.exists():
        if not resume or not (out / "audit_contract.json").is_file():
            raise FileExistsError("Use a new output directory, or --resume for the same audit")
        if json.loads((out / "audit_contract.json").read_text(encoding="utf-8")) != contract:
            raise ValueError("Audit inputs/backend/code changed; use a new output directory")
        lines = (out / "progress.jsonl").read_text(encoding="utf-8").splitlines() if (out / "progress.jsonl").exists() else []
        valid = []
        for index, line in enumerate(lines):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise ValueError("Corrupt audit journal before its last line")
                break  # An interrupted last write is retried.
            old[item["rel_path"]] = item
            valid.append(line)
        atomic_text(out / "progress.jsonl", "".join(line + "\n" for line in valid))
    else:
        out.mkdir(parents=True)
        atomic_text(out / "audit_contract.json", json.dumps(contract, indent=2))
    results, pending = {}, []
    for row in rows:
        cached = old.get(row["rel_path"], {})
        try:
            reusable = cached.get("status") == "ok" and all(
                cached.get(k) == value for k, value in fingerprint(video_root, row)[1].items())
        except (OSError, ValueError):
            reusable = False
        if reusable:
            results[row["rel_path"]] = cached
        else:
            pending.append(row)
    reused, started = len(results), time.perf_counter()
    print(f"CPU header audit: {len(rows)} files; {reused} reused; backend={backend}; workers={workers}", flush=True)
    with (out / "progress.jsonl").open("a", encoding="utf-8") as journal, ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(probe_row, video_root, row, backend, timeout) for row in pending]
        for future in as_completed(futures):
            row = future.result()
            results[row["rel_path"]] = row
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            journal.flush()
            done = len(results)
            if done % 50 == 0 or done == len(rows):
                elapsed = time.perf_counter() - started
                rate = (done - reused) / max(elapsed, .001)
                filled = int(30 * done / len(rows))
                print(f"[{'#' * filled}{'-' * (30 - filled)}] {done}/{len(rows)} "
                      f"{rate:.1f} files/s; ETA {(len(rows) - done) / max(rate, .001) / 60:.1f} min", flush=True)
    ordered = [results[row["rel_path"]] for row in rows]
    write_csv(out / "durations.csv", ordered)
    write_csv(out / "errors.csv", [r for r in ordered if r["status"] != "ok"])
    summary = {"datasets": summaries(ordered), "files": len(rows), "reused": reused,
               "errors": sum(r["status"] != "ok" for r in ordered),
               "wall_seconds_this_invocation": time.perf_counter() - started,
               "limitation": "Header estimates, not full decode validation. Average/nominal fps does not establish CFR. "
                             "PyAV may decode the first frame to recover a missing stream origin. "
                             "No model scores or visual-content analysis."}
    atomic_text(out / "summary.json", json.dumps(summary, indent=2))
    atomic_text(out / "SHA256SUMS.txt", "".join(f"{digest(out / name)}  {name}\n"
                for name in ("audit_contract.json", "durations.csv", "errors.csv", "summary.json")))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--probe-one":
        print(json.dumps(pyav_metadata(sys.argv[2]), allow_nan=False))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--backend", choices=("auto", "ffprobe", "pyav"), default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    summary = audit(args.inventory, args.video_root, args.out, args.workers, args.timeout, args.backend, args.resume)
    if summary["errors"]:
        raise SystemExit("Audit recorded errors; inspect errors.csv, then resume. No rows were silently dropped.")


if __name__ == "__main__":
    main()
