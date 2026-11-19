"""Video-level evaluation for the legacy or face-identity-disjoint protocol.

Video score = mean frame fake probability (GenD). Thresholds and metrics
follow the old protocol: 5% FPR threshold from *validation* real videos,
plus the fixed fake-probability 0.5 operating point; AUROC with identity and
source-family clustered bootstrap; AP both directions; EER; per method.

  validation only : --val-pred val.csv
  final test      : --val-pred val.csv --test-pred test.csv --selection selection.json
The test threshold is always the validation one; it is never re-fitted on test.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gad import manifest as MF  # noqa: E402
from gad import scoring as S  # noqa: E402
from build_identity_split import audit_splits  # noqa: E402


def load_split(pred_csv, videos, split):
    rows = videos[videos["split"] == split]
    scores = S.video_scores(pd.read_csv(pred_csv))
    merged, missing = S.join_videos(rows, scores)
    return merged, missing, len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", required=True, help="videos.csv from build_video_table.py")
    parser.add_argument("--val-pred", required=True)
    parser.add_argument("--test-pred", default=None)
    parser.add_argument("--selection", default=None)
    parser.add_argument("--protocol", choices=("old", "celeb_identity"), default="old")
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--out", required=True, help="Report JSON")
    args = parser.parse_args()

    identity = args.protocol == "celeb_identity"
    videos = pd.read_csv(args.videos, keep_default_na=not identity)
    if identity:
        overlap = audit_splits(videos)
    val, val_missing, val_total = load_split(args.val_pred, videos, "val")
    threshold = S.calibrate_on_validation(val)
    report = {
        "protocol": ("old 30-group CelebDFv3 protocol" if args.protocol == "old"
                     else "CelebDFv3 face-identity-disjoint protocol")
                    + "; video score = GenD mean frame fake probability",
        "threshold_source": "validation real videos, 5% FPR",
        "threshold": threshold,
        "val": S.evaluate_split(val, threshold, args.bootstrap_draws),
        "val_coverage": {"listed": val_total, "scored": len(val), "unscorable": val_missing},
        "val_pred_meta": json.loads(Path(args.val_pred + ".meta.json").read_text()) if Path(args.val_pred + ".meta.json").exists() else None,
    }
    if identity:
        report.update(video_table_sha256=MF.sha256(Path(args.videos)), split_overlap=overlap)
    print("val :", S.short_summary(report["val"]), "| unscorable:", len(val_missing))

    if args.test_pred:
        if not args.selection:
            raise SystemExit("--test-pred requires --selection (chosen on validation).")
        selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
        meta = json.loads(Path(args.test_pred + ".meta.json").read_text(encoding="utf-8"))
        val_meta = report["val_pred_meta"] or {}
        if meta["model"] != val_meta.get("model") or meta.get("checkpoint_sha256") != val_meta.get("checkpoint_sha256"):
            raise SystemExit("Validation and test predictions come from different models.")
        if identity and meta["precision"] != val_meta.get("precision"):
            raise SystemExit("Validation and test predictions use different precision.")
        if selection["checkpoint"] != "official" and meta.get("checkpoint_sha256") != selection["checkpoint_sha256"]:
            raise SystemExit("Test predictions are not from the selected checkpoint.")
        if selection["checkpoint"] == "official" and meta["model"] != "official":
            raise SystemExit("Expected predictions from the official model.")
        if args.protocol == "celeb_identity" and selection["checkpoint"] != "official":
            if selection.get("protocol") != args.protocol:
                raise SystemExit("Expected an identity-disjoint checkpoint selection.")
            if val_meta.get("frame_list_sha256") != selection["val_frame_list_sha256"]:
                raise SystemExit("Validation predictions do not match the validation list used to select the checkpoint.")
        test, test_missing, test_total = load_split(args.test_pred, videos, "test")
        report["test"] = S.evaluate_split(test, threshold, args.bootstrap_draws)
        report["test_coverage"] = {"listed": test_total, "scored": len(test), "unscorable": test_missing}
        if identity:
            report["test_pred_meta"] = meta
        report["selection"] = selection
        if args.protocol == "old":
            report["test_note"] = "Fixed comparison test set of the old protocol (already used in the previous round; not blind)."
        print("test:", S.short_summary(report["test"]), "| unscorable:", len(test_missing))

    MF.write_json(Path(args.out), S.nan_to_none(report))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
