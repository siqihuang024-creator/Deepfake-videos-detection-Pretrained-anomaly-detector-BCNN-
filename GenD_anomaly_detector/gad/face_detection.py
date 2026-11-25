"""Opt-in detection fallback; all returned geometry refers to the original image."""
import numpy as np

POLICIES = ("none", "pad_on_empty_v1")


def policy_spec(name):
    if name not in POLICIES:
        raise ValueError("Unknown detection policy: " + str(name))
    return {"name": name, "trigger": "original detection has zero candidates" if name != "none" else "disabled",
            "padding": "black; top/bottom=floor(height/2), left/right=floor(width/2)" if name != "none" else "none",
            "thresholds": "unchanged from the configured detector",
            "fallback_candidate_filter": "finite positive boxes, center and all five landmarks inside original image",
            "crop_coordinates": "original image; subtract left/top padding from boxes and landmarks"}


def detect_with_policy(model, frame, policy="none"):
    policy_spec(policy)
    detections, landmarks = model.detect(frame)
    audit = {"fallback_attempted": False, "fallback_used": False, "fallback_rejected": 0}
    if policy == "none" or len(detections):
        return detections, landmarks, audit

    audit["fallback_attempted"] = True
    height, width = frame.shape[:2]
    top, left = height // 2, width // 2
    padded = np.pad(frame, ((top, top), (left, left), (0, 0)), mode="constant")
    detections, landmarks = model.detect(padded)
    if not len(detections):
        return detections, landmarks, audit
    detections = np.asarray(detections, dtype=np.float32).copy()
    landmarks = np.asarray(landmarks, dtype=np.float32).copy()
    if detections.ndim != 2 or detections.shape[1] != 5 or landmarks.shape != (len(detections), 5, 2):
        raise ValueError("Fallback requires matching xyxy-score boxes and five-point landmarks")

    detections[:, [0, 2]] -= left
    detections[:, [1, 3]] -= top
    landmarks -= np.asarray([left, top], dtype=np.float32)
    centers = (detections[:, :2] + detections[:, 2:4]) / 2
    valid = (np.isfinite(detections).all(axis=1) & np.isfinite(landmarks).all(axis=(1, 2))
             & (detections[:, 2] > detections[:, 0]) & (detections[:, 3] > detections[:, 1])
             & (centers[:, 0] >= 0) & (centers[:, 0] < width)
             & (centers[:, 1] >= 0) & (centers[:, 1] < height)
             & (landmarks[:, :, 0] >= 0).all(axis=1) & (landmarks[:, :, 0] < width).all(axis=1)
             & (landmarks[:, :, 1] >= 0).all(axis=1) & (landmarks[:, :, 1] < height).all(axis=1))
    audit["fallback_rejected"] = int((~valid).sum())
    audit["fallback_used"] = bool(valid.any())
    return detections[valid], landmarks[valid], audit
