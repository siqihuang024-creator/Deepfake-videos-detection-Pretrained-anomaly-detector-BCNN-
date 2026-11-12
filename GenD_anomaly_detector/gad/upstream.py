"""Make the unmodified upstream GenD checkout importable as ``src.*``."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GEND_ROOT = PROJECT_ROOT / "third_party" / "GenD"
GEND_COMMIT = "387a42266dd385fbe8f7626c5d3aa03eea3bfaab"


def add_gend_to_path() -> Path:
    if not (GEND_ROOT / "src" / "model" / "GenD.py").exists():
        raise FileNotFoundError("Upstream GenD not found at {} (git submodule update --init?)".format(GEND_ROOT))
    root = str(GEND_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return GEND_ROOT


def upstream_commit() -> str:
    """Commit of the checked-out submodule, read without invoking git."""
    head = GEND_ROOT / ".git"
    try:
        if head.is_file():  # submodule: .git is a "gitdir: ..." pointer
            gitdir = (GEND_ROOT / head.read_text().split(":", 1)[1].strip()).resolve()
        else:
            gitdir = head
        ref = (gitdir / "HEAD").read_text().strip()
        if ref.startswith("ref:"):
            return (gitdir / ref.split(" ", 1)[1]).read_text().strip()
        return ref
    except OSError:
        return "unknown"
