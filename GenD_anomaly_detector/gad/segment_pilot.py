"""Explicit, bounded windows for a preprocessing pilot, not a training protocol."""
from decimal import Decimal, ROUND_FLOOR
import hashlib
import math

VERSION = 1
MICROSECONDS = 1000000
PLACEMENTS = ("timeline", "interior_v1")
PILOT_COVERAGE_RULE = {
    "purpose": "pilot diagnostic flag only; no automatic exclusion or final-test filtering",
    "minimum_selected_frames": 32, "minimum_detected_fraction": .5,
    "time_bins": 5, "minimum_good_bins": 4, "minimum_fraction_per_good_bin": .5,
}


def stable_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parent_key(dataset, rel_path, label):
    source = "real_external" if int(label) == 0 else "fake_external"
    return "external/" + source + "/" + stable_hash(dataset + "|" + rel_path)


def bounded_windows(duration, seconds, cap, minimum_seconds=2):
    """Uniformly cap nonoverlapping windows over the timeline; retain eligible tails."""
    values = [float(duration), float(seconds), float(minimum_seconds)]
    if not all(math.isfinite(value) and value > 0 for value in values) or cap < 1 or int(cap) != cap:
        raise ValueError("Finite positive duration/window/minimum and integer cap required")
    duration_us, window_us, minimum_us = [int((Decimal(str(value)) * MICROSECONDS).to_integral_value(
        rounding=ROUND_FLOOR)) for value in (duration, seconds, minimum_seconds)]
    if minimum_us < 1 or window_us < minimum_us:
        raise ValueError("Minimum window exceeds nominal window, or precision below one microsecond")
    full, tail = divmod(duration_us, window_us)
    count = full + int(tail >= minimum_us)
    if not count:
        return []
    # Integer arithmetic avoids materializing millions of windows for malformed headers.
    selected = ([count // 2] if cap == 1 else
                [i * (count - 1) // (cap - 1) for i in range(cap)] if count > cap else list(range(count)))
    return [(index * window_us, min((index + 1) * window_us, duration_us)) for index in selected]


def interior_windows(duration, seconds, cap, minimum_seconds=2):
    """Centered windows spread inside the video, with margins where duration permits."""
    if cap < 1 or int(cap) != cap:
        raise ValueError("Positive integer cap required")
    cap = int(cap)
    if not bounded_windows(duration, seconds, cap=1, minimum_seconds=minimum_seconds):
        return []
    duration_us, window_us = [int((Decimal(str(value)) * MICROSECONDS).to_integral_value(
        rounding=ROUND_FLOOR)) for value in (duration, seconds)]
    length = min(duration_us, window_us)
    # Reserve margins instead of forcing two windows against both endpoints.
    count = min(cap, max(1, duration_us // window_us - 1))
    starts = [(i + 1) * duration_us // (count + 1) - length // 2 for i in range(count)]
    return [(start, start + length) for start in starts]


def temporal_face_coverage(timestamps, has_face, start, end):
    """Describe where detections occur, so 32 frames in one instant do not look sustained."""
    if len(timestamps) != len(has_face) or not all(math.isfinite(v) for v in (start, end)) or end <= start:
        raise ValueError("Invalid timing/face arrays or interval")
    bins = PILOT_COVERAGE_RULE["time_bins"]
    decoded, detected, face_times = [0] * bins, [0] * bins, []
    for timestamp, found in zip(timestamps, has_face):
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or not start <= timestamp < end:
            raise ValueError("Frame PTS outside its half-open window or unknown")
        index = min(bins - 1, int((timestamp - start) / (end - start) * bins))
        decoded[index] += 1
        if found:
            detected[index] += 1
            face_times.append(timestamp)
    fractions = [n / d if d else 0. for n, d in zip(detected, decoded)]
    return {"decoded_frames_by_bin": decoded, "detected_frames_by_bin": detected,
            "detected_fraction_by_bin": fractions,
            "good_bins": sum(value >= PILOT_COVERAGE_RULE["minimum_fraction_per_good_bin"] for value in fractions),
            "detected_fraction": len(face_times) / len(timestamps) if len(timestamps) else 0.,
            "face_span_seconds": max(face_times) - min(face_times) if face_times else 0.}


def sustained_candidate(selected_frames, coverage):
    return (selected_frames >= PILOT_COVERAGE_RULE["minimum_selected_frames"]
            and coverage["detected_fraction"] >= PILOT_COVERAGE_RULE["minimum_detected_fraction"]
            and coverage["good_bins"] >= PILOT_COVERAGE_RULE["minimum_good_bins"])


def window_spec(row):
    """The identity that must match when resuming a pilot crop record."""
    if isinstance(row, dict):
        get = row.get
    else:
        get = lambda name, default=None: getattr(row, name, default)
    if get("sample_unit") != "segment_pilot":
        raise ValueError("This is not a segment preprocessing pilot row")
    start, end, origin = [float(get(key)) for key in
                          ("segment_start_seconds", "segment_end_seconds", "stream_start_seconds")]
    if not all(math.isfinite(value) for value in (start, end, origin)) or start < 0 or end <= start:
        raise ValueError("Invalid half-open time window or unknown stream origin")
    return {"sample_unit": "segment_pilot", "parent_video_key": get("parent_video_key"),
            "clip_id": get("clip_id"), "segment_start_seconds": start,
            "segment_end_seconds": end, "stream_start_seconds": origin,
            "source_bytes": int(get("source_bytes")), "source_mtime_ns": int(get("source_mtime_ns")),
            "frame_index_semantics": "ordinal within decoded window; not a global video frame number",
            "timestamp_semantics": "original stream presentation timestamp in seconds"}


def iter_window_frames(path, start, end, origin):
    """Seek to a preceding keyframe, then select [origin+start, origin+end) by PTS.

    Never convert seconds to frame numbers using an assumed fps. This also
    handles variable-frame-rate inputs. An unknown/nonmonotone PTS is an error.
    """
    import av

    if not all(math.isfinite(value) for value in (start, end, origin)) or start < 0 or end <= start:
        raise ValueError("Invalid window")
    absolute_start, absolute_end = origin + start, origin + end
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        if stream.time_base is None or stream.time_base <= 0:
            raise ValueError("Missing stream time base")
        container.seek(math.floor(absolute_start / float(stream.time_base)), stream=stream,
                       backward=True, any_frame=False)
        previous, index = None, 0
        for frame in container.decode(stream):
            if frame.time is None or not math.isfinite(float(frame.time)):
                raise ValueError("Missing frame PTS; cannot verify the window")
            timestamp = float(frame.time)
            if previous is not None and timestamp < previous:
                raise ValueError("Nonmonotone presentation timestamps")
            previous = timestamp
            if timestamp < absolute_start:
                continue
            if timestamp >= absolute_end:
                break
            yield index, frame.to_ndarray(format="bgr24"), timestamp
            index += 1


def make_pilot(parents, durations, seconds=10, clips_per_parent=2, groups_per_stratum=2,
               minimum_seconds=2, seed=42, placement="timeline"):
    """One original per source group per stratum, and capped windows per original.

    All original role/group assignments are inherited. Final-test originals
    are reserved and never selected or decoded by this pilot.
    """
    if groups_per_stratum < 1 or int(groups_per_stratum) != groups_per_stratum:
        raise ValueError("groups_per_stratum must be a positive integer")
    if placement not in PLACEMENTS:
        raise ValueError("Unknown window placement")
    lookup = {(r["dataset"], r["rel_path"]): r for r in durations}
    if len(lookup) != len(durations):
        raise ValueError("Duplicate duration rows")
    candidates, excluded = {}, []
    seen = set()
    for parent in parents:
        key = (parent["dataset"], parent["rel_path"])
        if key in seen:
            raise ValueError("Expected a whole-video manifest with distinct original paths")
        seen.add(key)
        if parent["split"] == "test":
            continue
        if parent["split"] not in ("train", "calibration", "dev"):
            raise ValueError("Unknown pilot role")
        duration = lookup.get(key)
        if not duration or duration.get("status") != "ok":
            raise ValueError("Missing successful duration audit for development original: " + str(key))
        if duration.get("stream_start_seconds") in (None, ""):
            raise ValueError("Unknown stream origin; reprobe with --backend pyav: " + str(key))
        if parent.get("bytes") and int(parent["bytes"]) != int(duration["bytes"]):
            raise ValueError("Duration inventory size disagrees with the parent manifest: " + str(key))
        generator = bounded_windows if placement == "timeline" else interior_windows
        windows = generator(duration["duration_seconds"], seconds, clips_per_parent, minimum_seconds)
        if not windows:
            excluded.append({"dataset": key[0], "rel_path": key[1], "reason": "shorter_than_minimum_window"})
            continue
        stratum = (parent["split"], parent["dataset"], str(parent["label"]))
        candidates.setdefault(stratum, {}).setdefault(parent["source_clip"], []).append((parent, duration, windows))
    rows = []
    for stratum in sorted(candidates):
        groups = candidates[stratum]
        chosen = sorted(groups, key=lambda group: stable_hash(f"{seed}|{stratum}|{group}"))[:groups_per_stratum]
        for group in chosen:
            parent, duration, windows = min(groups[group], key=lambda item: stable_hash(f"{seed}|{item[0]['rel_path']}"))
            original_key = parent_key(parent["dataset"], parent["rel_path"], parent["label"])
            for start_us, end_us in windows:
                clip_id = stable_hash(f"segment_pilot_v{VERSION}|{original_key}|{start_us}|{end_us}")
                rows.append({**parent, "sample_unit": "segment_pilot", "parent_video_key": original_key,
                             "clip_id": clip_id, "video_key": original_key.rsplit("/", 1)[0] + "/" + clip_id,
                             "frame_subset": "uniform32", "segment_start_seconds": start_us / MICROSECONDS,
                             "segment_end_seconds": end_us / MICROSECONDS,
                             "stream_start_seconds": duration["stream_start_seconds"],
                             "source_bytes": int(duration["bytes"]), "source_mtime_ns": int(duration["mtime_ns"]),
                             "source_duration_seconds": duration["duration_seconds"],
                             "duration_source": duration["duration_source"]})
    if not rows:
        raise ValueError("No eligible development windows")
    return sorted(rows, key=lambda r: (r["split"], r["dataset"], r["rel_path"], r["segment_start_seconds"])), excluded
