"""Read frozen classifier weights on CPU and bind them to a feature-cache hash."""
import gc

import numpy as np

from .feature_cache import model_state_hash
from .feature_geometry import classifier_direction


def read_direction(signature, loader=None):
    if loader is None:
        from predict_frames import load_frozen_module
        loader = load_frozen_module
    revision = signature.get("official_revision") if signature["model"] == "official" else None
    if signature["model"] == "official" and not revision:
        raise ValueError("Pin the official cached revision before reading classifier weights")
    module, _ = loader(signature["model"], "cpu", revision)
    try:
        module = module.to("cpu").eval().requires_grad_(False)
        actual_hash = model_state_hash(module)
        if actual_hash != signature["weights_sha256"]:
            raise ValueError("Frozen encoder/classifier state differs from the cache")
        head = module.model
        if not getattr(head, "normalize_inputs", False) or signature.get("head_configuration") != "LinearNorm":
            raise ValueError("Direction diagnostic requires the verified L2-input linear classifier")
        w = head.linear.weight.detach().float().cpu().numpy().copy()
        b = head.linear.bias.detach().float().cpu().numpy().copy()
        if w.shape != (2, 1024) or b.shape != (2,) or not np.isfinite(b).all():
            raise ValueError("Expected a finite 1024-to-2 linear classifier")
        return {"direction": classifier_direction(w), "linear_weights": w, "linear_bias": b}, {
            "weights_sha256": actual_hash, "official_revision": revision,
            "direction_rule": "normalize(weight_fake - weight_real); real=0, fake=1",
            "input_space": "mean of frame L2 embeddings; no re-normalization of video means",
            "aggregation_note": "Averaged linear logits and averaged frame softmax probabilities are not equivalent."}
    finally:
        del module
        gc.collect()
