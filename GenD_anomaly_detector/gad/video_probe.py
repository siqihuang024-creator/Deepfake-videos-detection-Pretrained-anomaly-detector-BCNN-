"""Small, read-only metadata probes. No detector, encoder or GPU is loaded."""
import math
from fractions import Fraction


def finite_number(value, positive=False):
    try:
        result = float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError, TypeError):
        return None
    return result if math.isfinite(result) and (not positive or result > 0) else None


def ffprobe_metadata(payload):
    streams = payload.get("streams", [])
    if not streams:
        raise ValueError("No video stream")
    stream = streams[0]
    duration = finite_number(stream.get("duration"), positive=True)
    origin = finite_number(stream.get("start_time"))
    source = "video_stream"
    if duration is None:
        duration = finite_number(payload.get("format", {}).get("duration"), positive=True)
        source = "container_fallback"
    if duration is None:
        raise ValueError("Missing or invalid duration; no frame-count/fps guess is made")
    return {
        "duration_seconds": duration, "duration_source": source,
        "stream_start_seconds": origin,
        "start_source": "video_stream" if origin is not None else "unknown",
        "fps_average": finite_number(stream.get("avg_frame_rate"), positive=True),
        "fps_nominal": finite_number(stream.get("r_frame_rate"), positive=True),
        "time_base": stream.get("time_base", ""),
        "metadata_frames": finite_number(stream.get("nb_frames"), positive=True),
        "width": stream.get("width"), "height": stream.get("height"),
        "codec": stream.get("codec_name", ""),
    }


def pyav_metadata(path):
    import av

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("No video stream")
        stream = container.streams.video[0]
        stream.thread_count = 1
        duration = (finite_number(stream.duration * stream.time_base, positive=True)
                    if stream.duration is not None and stream.time_base is not None else None)
        source = "video_stream"
        if duration is None:
            duration = finite_number(container.duration / av.time_base, positive=True) if container.duration else None
            source = "container_fallback"
        if duration is None:
            raise ValueError("Missing or invalid duration; no frame-count/fps guess is made")
        origin = (float(stream.start_time * stream.time_base)
                  if stream.start_time is not None and stream.time_base is not None else None)
        start_source = "video_stream" if origin is not None else "unknown"
        if origin is None:
            # Only the first presented frame, not a scan through the entire video.
            first = next(container.decode(stream), None)
            if first is not None and first.time is not None:
                origin, start_source = float(first.time), "first_decoded_frame"
        return {
            "duration_seconds": duration, "duration_source": source,
            "stream_start_seconds": origin, "start_source": start_source,
            "fps_average": finite_number(stream.average_rate, positive=True),
            "fps_nominal": finite_number(stream.base_rate, positive=True),
            "time_base": str(stream.time_base), "metadata_frames": stream.frames or None,
            "width": stream.codec_context.width, "height": stream.codec_context.height,
            "codec": stream.codec_context.name,
        }
