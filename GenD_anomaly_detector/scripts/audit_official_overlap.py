"""Compare manually reconciled canonical video IDs, not dataset names or basenames.

Both CSVs require dataset,canonical_video_id. The candidate also has split.
Populate IDs using the same audited corpus-relative convention on both sides.
Absence of a complete official selection list is reported as unresolved.
"""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gad.diagnostic_protocol import canonical_dataset, sha256_file


def audit(official_path, candidate_path, out, complete=False):
    if Path(out).exists():
        raise FileExistsError(out)
    official = pd.read_csv(official_path, dtype=str, keep_default_na=False)
    candidate = pd.read_csv(candidate_path, dtype=str, keep_default_na=False)
    for rows in (official, candidate):
        if not {"dataset", "canonical_video_id"}.issubset(rows) or rows.empty:
            raise ValueError("Need nonempty dataset/canonical_video_id columns")
        if rows[["dataset", "canonical_video_id"]].eq("").any().any():
            raise ValueError("Canonical IDs must not be blank")
        rows["dataset"] = rows["dataset"].map(canonical_dataset).replace({"celebdfpp": "celebdfv3"})
    if "split" not in candidate or not candidate["split"].eq("test").any():
        raise ValueError("Candidate must identify the actual test videos")
    test = candidate.loc[candidate["split"].eq("test")]
    overlap = test.merge(official[["dataset", "canonical_video_id"]].drop_duplicates(),
                         on=["dataset", "canonical_video_id"], how="inner")
    result = {"official_sha256": sha256_file(official_path), "candidate_sha256": sha256_file(candidate_path),
              "official_list_declared_complete": complete,
              "canonical_id_note": "User-reconciled IDs; this tool cannot verify that different names refer to different videos",
              "test_videos": len(test), "overlapping_videos": overlap.to_dict("records"),
              "status": "overlap_found" if len(overlap) else ("no_exact_ID_overlap" if complete else "unresolved_incomplete_official_list"),
              "limitation": "Dataset development exposure remains; exact video audit does not establish person independence"}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with Path(out).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--official-list-complete", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.official, args.candidate, args.out, args.official_list_complete), indent=2))
