"""Fixed factorial source-composition sensitivity, not score-based cleaning."""
import numpy as np

from . import feature_geometry as geometry, covariance_controls as covariance


FIT_PROFILES = ("mixed", "hdtf_only")
CALIBRATION_PROFILES = ("mixed", "hdtf_only")


def selection_masks(rows):
    if (rows.empty or rows.video_key.duplicated().any()
            or not rows.head_role.isin(("gradient_fit", "early_stop", "calibration", "dev")).all()
            or not rows.label.isin((0, 1)).all()
            or not rows.loc[rows.head_role.ne("dev"), "label"].eq(0).all()
            or (rows.groupby("source_clip").head_role.nunique() > 1).any()):
        raise ValueError("Expected original source-isolated genuine A and development units only")
    masks = {}
    for role in ("gradient_fit", "calibration"):
        eligible = rows.head_role.eq(role)
        if not rows.loc[eligible, "dataset"].isin(("HDTF", "TalkingHead-1KH")).all():
            raise ValueError("Unexpected A source dataset")
        for profile in FIT_PROFILES:
            mask = eligible & (rows.dataset.eq("HDTF") if profile == "hdtf_only" else True)
            if not mask.any():
                raise ValueError("No eligible units for " + role + "/" + profile)
            masks[role + "/" + profile] = mask.to_numpy()
    masks["dev"] = rows.head_role.eq("dev").to_numpy()
    if set(rows.loc[masks["dev"], "label"]) != {0, 1}:
        raise ValueError("Keep the original two-class development set")
    return masks


def fit_family(features, groups):
    mean, empirical, counts = covariance.clip_covariance(features, groups)
    source_head = geometry.fit_distances(features, groups)
    np.testing.assert_allclose(mean, source_head["mean"], rtol=1e-8, atol=1e-12)
    heads = {"centroid_l2": {"mean": mean}, "mahalanobis_source_mean": source_head}
    for alpha in (.3, .5):
        floor = alpha * np.trace(empirical) / len(mean)
        head = {"mean": mean, "cholesky": np.linalg.cholesky(
            (1-alpha)*empirical + floor*np.eye(len(mean))), "shrinkage": alpha}
        heads["mahalanobis_clip_original_a" + str(int(100*alpha))] = head
    return heads, counts


def score_family(features, heads, probabilities):
    x = geometry.matrix(features)
    p = np.asarray(probabilities, float)
    if p.shape != (len(x),) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected frozen classifier probabilities")
    scores = {"classifier_reference": p}
    for name, head in heads.items():
        scores[name] = ((x-head["mean"])**2).sum(1) if name == "centroid_l2" else geometry.score_distances(x, head)[1]
    return scores
