"""Versioned, disk-backed frame caches for frozen encoders.

One row is one selected frame, without padding or treating frames as
independent training videos. Video means preserve the old pre-quantization
mean-of-L2-embeddings rule.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

VERSION = 1
FEATURE_TYPE = "frame_l2_embeddings_with_norms"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature_hash(signature):
    return hashlib.sha256(json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def video_key(path):
    parts = str(path).replace("\\", "/").split("/")
    if len(parts) < 4:
        raise ValueError(f"Not a dataset/source/video/frame path: {path}")
    return "/".join(parts[-4:-1])


def crop_stat_hash(paths):
    """Invalidate ordinary crop changes without reading every PNG twice."""
    digest = hashlib.sha256()
    for path in paths:
        stat = Path(path).stat()
        digest.update(json.dumps([str(path), stat.st_size, stat.st_mtime_ns]).encode())
    return digest.hexdigest()


def model_state_hash(model):
    """Hash actual official weights, not just a mutable Hugging Face repo name."""
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode())
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def frame_timing(paths, records_dir=None):
    """Use recorded decoded PTS only. Older records remain explicitly unknown."""
    indices, times, records = [], [], {}
    for path in paths:
        match = re.fullmatch(r"frame_(\d+)\.png", Path(path).name)
        if not match:
            raise ValueError(f"Cannot identify the original frame index: {path}")
        index = int(match.group(1))
        indices.append(index)
        key = video_key(path)
        if key not in records:
            record_path = Path(records_dir) / (key + ".json") if records_dir else None
            mapping = {}
            if record_path is not None and record_path.is_file():
                record = json.loads(record_path.read_text(encoding="utf-8"))
                mapping = dict(zip(record.get("selected", []),
                                   record.get("selected_timestamps_seconds", [])))
            records[key] = mapping
        value = records[key].get(index)
        times.append(float(value) if value is not None else np.nan)
    return np.asarray(indices, np.int64), np.asarray(times, np.float64)


class FrameCacheWriter:
    def __init__(self, directory, paths, signature, frame_indices=None, frame_times=None):
        self.directory = Path(directory)
        self.paths = [str(p) for p in paths]
        if not self.paths or len(set(self.paths)) != len(self.paths):
            raise ValueError("Frame paths must be nonempty and unique")
        self.signature = dict(signature)
        keys = [video_key(p) for p in self.paths]
        self.keys, self.inverse = np.unique(keys, return_inverse=True)
        self.counts = np.bincount(self.inverse)
        if np.any(self.counts > 32):
            raise ValueError("Expected at most 32 selected frames per video")
        self.seen = np.zeros(len(paths), bool)
        n = len(paths)
        self.frame_indices = (np.asarray(frame_indices, np.int64) if frame_indices is not None
                              else frame_timing(paths)[0])
        self.frame_times = (np.asarray(frame_times, np.float64) if frame_times is not None
                            else np.full(n, np.nan))
        if (self.frame_indices.shape != (n,) or self.frame_times.shape != (n,)
                or np.any(self.frame_indices < 0) or np.isinf(self.frame_times).any()):
            raise ValueError("Invalid frame timing")
        self.directory.parent.mkdir(parents=True, exist_ok=True)
        if self.directory.exists():
            raise FileExistsError(self.directory)
        self.lock = self.directory.with_name(self.directory.name + ".lock")
        fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        try:
            self.temporary = Path(tempfile.mkdtemp(prefix=self.directory.name + ".partial-",
                                                   dir=self.directory.parent))
        except BaseException:
            self.lock.unlink()
            raise
        self.maps = {}
        self.totals = None
        self.prob_totals = np.zeros(len(self.keys), np.float64)

    def write(self, indices, l2_features, norms, fake_probabilities):
        indices = np.asarray(indices, np.int64)
        x, norms, probs = (np.asarray(v, np.float32) for v in (l2_features, norms, fake_probabilities))
        if (indices.ndim != 1 or len(set(indices)) != len(indices)
                or np.any(indices < 0) or np.any(indices >= len(self.paths)) or self.seen[indices].any()
                or x.ndim != 2 or x.shape[0] != len(indices)
                or norms.shape != indices.shape or probs.shape != indices.shape
                or not all(np.isfinite(v).all() for v in (x, norms, probs))
                or np.any(norms < 0) or np.any(probs < 0) or np.any(probs > 1)):
            raise ValueError("Invalid, duplicate or non-finite frame cache batch")
        stored = x.astype(np.float16)
        if not np.isfinite(stored).all():
            raise ValueError("Frame embeddings cannot be represented in float16")
        if self.totals is None:
            self.totals = np.zeros((len(self.keys), x.shape[1]), np.float64)
            for filename, dtype, shape in (
                ("frame_l2.npy", np.float16, (len(self.paths), x.shape[1])),
                ("frame_norms.npy", np.float32, (len(self.paths),)),
                ("frame_fake_prob.npy", np.float32, (len(self.paths),)),
            ):
                self.maps[filename] = np.lib.format.open_memmap(self.temporary / filename,
                                                               mode="w+", dtype=dtype, shape=shape)
        if x.shape[1] != self.totals.shape[1]:
            raise ValueError("Feature dimensions changed during extraction")
        self.maps["frame_l2.npy"][indices] = stored
        self.maps["frame_norms.npy"][indices] = norms
        self.maps["frame_fake_prob.npy"][indices] = probs
        np.add.at(self.totals, self.inverse[indices], x)
        np.add.at(self.prob_totals, self.inverse[indices], probs)
        self.seen[indices] = True

    def _close_maps(self):
        for value in self.maps.values():
            value.flush()
            value._mmap.close()
        self.maps.clear()

    def finish(self, measurements=None):
        if self.totals is None or not self.seen.all():
            raise ValueError("Cannot publish an incomplete frame cache")
        self._close_maps()
        arrays = {
            "video_features.npy": (self.totals / self.counts[:, None]).astype(np.float32),
            "video_fake_prob.npy": self.prob_totals / self.counts,
            "video_num_frames.npy": self.counts,
            "frame_video_index.npy": self.inverse,
            "frame_indices.npy": self.frame_indices,
            "frame_times_seconds.npy": self.frame_times,
        }
        for filename, values in arrays.items():
            np.save(self.temporary / filename, values, allow_pickle=False)
        (self.temporary / "video_keys.json").write_text(json.dumps(self.keys.tolist()), encoding="utf-8")
        (self.temporary / "frame_paths.txt").write_text("\n".join(self.paths) + "\n", encoding="utf-8")
        files = {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                 for p in self.temporary.iterdir() if p.is_file()}
        meta = {
            "version": VERSION, "feature_type": FEATURE_TYPE,
            "signature": self.signature, "signature_sha256": signature_hash(self.signature),
            "num_frames": len(self.paths), "num_videos": len(self.keys),
            "dimension": self.totals.shape[1], "frame_storage_dtype": "float16",
            "norm_storage_dtype": "float32", "video_storage_dtype": "float32",
            "video_aggregation": "mean_of_frame_l2_embeddings_before_storage_quantization",
            "known_timestamp_count": int(np.isfinite(self.frame_times).sum()),
            "timing_note": "Decoded PTS when recorded; unknown timestamps are NaN, not inferred FPS.",
            "input_integrity_note": "Frame-list hash plus crop file size/mtime fingerprint, not a pixel-content audit.",
            "measurements": measurements or {}, "files": files,
        }
        (self.temporary / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if self.directory.exists():
            raise FileExistsError(self.directory)
        self.temporary.rename(self.directory)
        self.lock.unlink()
        self.temporary = None
        return meta

    def abort(self):
        self._close_maps()
        if self.temporary is not None and self.temporary.exists():
            shutil.rmtree(self.temporary)
        if self.lock.exists():
            self.lock.unlink()
        self.temporary = None


def load_video_cache(directory, expected_signature=None):
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if (meta.get("version") != VERSION or meta.get("feature_type") != FEATURE_TYPE
            or meta.get("signature_sha256") != signature_hash(meta["signature"])):
        raise ValueError("Unsupported or inconsistent frame cache metadata")
    if expected_signature is not None and meta["signature"] != expected_signature:
        raise ValueError("Cache belongs to different weights, frames or preprocessing")
    # Evaluation only consumes these small files. The frame tensor has its own
    # recorded SHA256 for audits; avoid re-reading gigabytes for every head.
    selected = ("video_features.npy", "video_fake_prob.npy", "video_num_frames.npy", "video_keys.json")
    for filename in selected:
        if sha256_file(directory / filename) != meta["files"][filename]["sha256"]:
            raise ValueError(f"Cache content changed: {filename}")
    for filename, info in meta["files"].items():
        if not (directory / filename).is_file() or (directory / filename).stat().st_size != info["bytes"]:
            raise ValueError(f"Incomplete cache: {filename}")
    keys = np.asarray(json.loads((directory / "video_keys.json").read_text(encoding="utf-8")))
    features = np.load(directory / "video_features.npy", allow_pickle=False)
    probs = np.load(directory / "video_fake_prob.npy", allow_pickle=False)
    counts = np.load(directory / "video_num_frames.npy", allow_pickle=False)
    if (features.shape != (len(keys), meta["dimension"]) or counts.shape != keys.shape
            or probs.shape != keys.shape or len(set(keys)) != len(keys)
            or len(keys) != meta["num_videos"] or counts.sum() != meta["num_frames"]
            or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts <= 0) or np.any(counts > 32)
            or np.any(probs < 0) or np.any(probs > 1)
            or not np.isfinite(features).all() or not np.isfinite(probs).all()):
        raise ValueError("Invalid cached video aggregates")
    return {"video_keys": keys, "features": features, "fake_prob": probs,
            "num_frames": counts, "metadata": meta}
