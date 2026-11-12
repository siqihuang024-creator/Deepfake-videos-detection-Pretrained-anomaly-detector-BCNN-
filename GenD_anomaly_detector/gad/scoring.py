"""Video scores (GenD rule) and old-protocol metrics.

Video score = mean of per-frame softmax fake probability (GenD
``reduce_video_predictions="mean"``). Metrics, thresholds and bootstrap
clusters follow the old protocol via the verbatim ``old_metrics`` module,
which expects real_labels with 1 = real and scores with higher = more fake.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import old_metrics as M
from .manifest import GEND_REAL

TARGET_FPR = 0.05


def video_scores(frame_predictions: pd.DataFrame) -> pd.DataFrame:
    """frame_predictions: columns video_key, fake_prob (one row per frame)."""
    grouped = frame_predictions.groupby("video_key")["fake_prob"]
    return pd.DataFrame({"score": grouped.mean(), "num_frames_scored": grouped.size()}).reset_index()


def join_videos(videos: pd.DataFrame, scores: pd.DataFrame):
    merged = videos.merge(scores, on="video_key", how="left")
    missing = merged[merged["score"].isna()]["rel_path"].tolist()
    unknown = sorted(set(scores["video_key"]) - set(videos["video_key"]))
    if unknown:
        raise AssertionError("Scores for videos outside the list: {}".format(unknown[:5]))
    return merged[merged["score"].notna()].reset_index(drop=True), missing


def calibrate_on_validation(val_videos: pd.DataFrame) -> float:
    real = val_videos[val_videos["label"] == GEND_REAL]["score"].to_numpy()
    return M.calibrate_threshold(real, TARGET_FPR)


def evaluate_split(videos: pd.DataFrame, threshold: float, draws: int = 2000, seed: int = 42) -> dict:
    real_labels = (videos["label"] == GEND_REAL).astype(int).to_numpy()  # old convention
    scores = videos["score"].to_numpy(dtype=np.float64)
    result = M.detection_metrics(real_labels, scores, threshold, TARGET_FPR)
    result["at_fake_probability_0_5"] = M.operating_point_metrics(real_labels, scores, 0.5)
    fake_labels = 1 - real_labels
    result["identity_bootstrap"] = M.clustered_auroc_interval(
        fake_labels, scores, videos["target_id"].to_numpy(), draws=draws, seed=seed)
    result["source_family_bootstrap"] = M.clustered_auroc_interval(
        fake_labels, scores, videos["source_clip"].to_numpy(), draws=draws, seed=seed)
    per_method = {}
    real_mask = real_labels == 1
    for method in sorted(videos.loc[~real_mask, "method"].unique()):
        mask = real_mask | (videos["method"].to_numpy() == method)
        per_method[method] = M.detection_metrics(real_labels[mask], scores[mask], threshold, TARGET_FPR)
    result["per_method"] = per_method
    result["num_videos_scored"] = int(len(videos))
    result["num_real"] = int(real_mask.sum())
    result["num_fake"] = int((~real_mask).sum())
    return result


def short_summary(result: dict) -> str:
    ci = result["identity_bootstrap"]
    return ("AUROC={:.4f} identity-CI [{:.4f}, {:.4f}] | AP-fake={:.4f} AP-real={:.4f} | "
            "TPR@5%FPR={:.4f} | bal.acc@val-5%FPR={:.4f} | bal.acc@0.5={:.4f}").format(
        result["auroc"], ci["low"], ci["high"], result["fake_average_precision"],
        result["real_average_precision"], result["tpr_at_target_fpr"], result["balanced_accuracy"],
        result["at_fake_probability_0_5"]["balanced_accuracy"])


def nan_to_none(value):
    if isinstance(value, dict):
        return {k: nan_to_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [nan_to_none(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (np.floating,)):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value
