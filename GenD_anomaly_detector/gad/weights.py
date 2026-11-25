"""Put the official Hugging Face GenD weights into the *trainable* GenD.

Upstream ``run.py`` loads a HF repo id through ``GenDHF``, which is
inference-only (no training step). The trainable ``src.model.GenD.GenD`` and
the HF model share parameter names (``feature_extractor.backbone.*``,
``model.linear.*``), so the HF state dict is loaded into it and every key is
checked. ``consistency_check`` then compares both models on the same images.
"""

from __future__ import annotations

from .upstream import add_gend_to_path

OFFICIAL_PE = "yermandy/GenD_PE_L"
FAKE_COLUMN = 1  # GenD: softmax column 1 = fake


def load_hf_model(repo: str = OFFICIAL_PE, revision=None):
    add_gend_to_path()
    from src.hf.modeling_gend import GenD as HFGenD

    model = HFGenD.from_pretrained(repo, **({"revision": revision} if revision else {}))
    model.eval()
    return model


def load_official_into_trainable(module, repo: str = OFFICIAL_PE) -> dict:
    hf = load_hf_model(repo)
    if hf.config.head != "LinearNorm":
        raise AssertionError("Expected LinearNorm head, HF config has {!r}".format(hf.config.head))
    if hf.config.backbone != module.config.backbone:
        raise AssertionError("Backbone mismatch: {} vs {}".format(hf.config.backbone, module.config.backbone))
    state = hf.state_dict()
    result = module.load_state_dict(state, strict=False)
    unexpected = list(result.unexpected_keys)
    missing = [k for k in result.missing_keys if k.startswith(("feature_extractor.", "model."))]
    if unexpected or missing:
        raise AssertionError("Weight loading incomplete. missing={} unexpected={}".format(missing[:10], unexpected[:10]))
    own = module.state_dict()
    for key, value in state.items():
        if not own[key].float().cpu().equal(value.float().cpu()):
            raise AssertionError("Parameter {} differs after loading".format(key))
    return {"repo": repo, "loaded_keys": len(state), "missing_other": sorted(result.missing_keys),
            "head": hf.config.head, "backbone": hf.config.backbone}


def consistency_check(module, image_paths, repo: str = OFFICIAL_PE, device: str = "cuda"):
    """Max |p_trainable - p_official| over fake probabilities, fp32, eval mode."""
    import torch
    from PIL import Image

    hf = load_hf_model(repo).to(device)
    module = module.to(device).eval()
    pre_hf = hf.feature_extractor.preprocess
    pre_mod = module.get_preprocessing()
    worst = 0.0
    rows = []
    with torch.no_grad():
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            p_hf = hf(pre_hf(image).unsqueeze(0).to(device)).float().softmax(1)[0, FAKE_COLUMN].item()
            p_mod = module(pre_mod(image).unsqueeze(0).to(device)).logits_labels.float().softmax(1)[0, FAKE_COLUMN].item()
            worst = max(worst, abs(p_hf - p_mod))
            rows.append((str(path), p_hf, p_mod))
    return worst, rows
