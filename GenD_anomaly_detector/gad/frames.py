"""Project frame subsets, informed by GenD's experiment directory names.

Every frame is face-detected first; subsets are drawn from the frames where a
face was found:
  random32  -> training videos, seeded per video (``FF-...-all/subset/random-32-frames``)
  uniform32 -> validation/test videos            (``CDFv3-...-all/subset/uniform-32-frames``)
Videos with fewer than 32 detected frames use all of them; nothing is repeated.

This is not an exact reimplementation of upstream detector.py's
fixed_num_frames: that mode uses integer linspace over original video frame
indices before checking for faces. Here uniform32 uses rounded linspace over
the detected-frame list. Even if every frame has a face, rounding can select
different indices. Each video supplies at most 32 individual images, not clips.
"""

from __future__ import annotations

import hashlib

import numpy as np

NUM_FRAMES = 32
TRAIN_FRAME_SEED = 0  # one fixed training subset shared by all 30 groups


def per_video_seed(rel_path: str, global_seed: int = TRAIN_FRAME_SEED) -> int:
    digest = hashlib.sha256("{}|{}".format(global_seed, rel_path).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def uniform_subset(detected, num_frames: int = NUM_FRAMES):
    detected = sorted(int(i) for i in detected)
    if len(detected) <= num_frames:
        return detected
    positions = np.rint(np.linspace(0, len(detected) - 1, num_frames)).astype(int)
    return sorted({detected[p] for p in positions})


def random_subset(detected, rel_path: str, num_frames: int = NUM_FRAMES, global_seed: int = TRAIN_FRAME_SEED):
    detected = sorted(int(i) for i in detected)
    if len(detected) <= num_frames:
        return detected
    rng = np.random.default_rng(per_video_seed(rel_path, global_seed))
    return sorted(int(i) for i in rng.choice(detected, size=num_frames, replace=False))


def select_frames(subset: str, detected, rel_path: str, num_frames: int = NUM_FRAMES):
    if subset == "uniform32":
        return uniform_subset(detected, num_frames)
    if subset == "random32":
        return random_subset(detected, rel_path, num_frames)
    raise ValueError("Unknown frame subset {!r}".format(subset))


def frame_file_name(frame_index: int) -> str:
    return "frame_{:05d}.png".format(int(frame_index))
