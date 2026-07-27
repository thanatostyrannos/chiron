"""Regression: the default report path must work when the input is a single file.

Found 2026-07-26 by a curriculum module author who ran

    scripts/verify_citations.py curriculum/distributed-training-strategies.md --known ...

without --out. The script derived its output as `Path(args.path) / "citation_check.json"`,
which appends a child path to a *file* and raises FileNotFoundError on write. The whole
verification run -- several minutes of API calls -- was lost at the final step.

Worth a test rather than a one-line fix, because the failure mode is the expensive kind:
the work completes, then the artifact that records it cannot be written.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_citations", REPO_ROOT / "scripts" / "verify_citations.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_citations"] = module
    spec.loader.exec_module(module)
    return module


def test_default_output_path_for_a_directory_input_is_inside_it() -> None:
    verifier = _load_verifier()
    out = verifier.default_output_path(REPO_ROOT / "curriculum")
    assert out.parent == REPO_ROOT / "curriculum"
    assert out.name.endswith(".json")


def test_default_output_path_for_a_single_file_is_a_sibling_not_a_child() -> None:
    verifier = _load_verifier()
    target = REPO_ROOT / "curriculum" / "tokenization.md"
    out = verifier.default_output_path(target)

    assert out.parent == target.parent, (
        "A single-file input must place its report beside the file, not inside it. "
        f"Got {out}, whose parent is {out.parent}."
    )
    assert out != target, "the report must not overwrite the document being checked"
    assert out.name.endswith(".json")


def test_default_output_path_is_writable_for_a_file_input(tmp_path: Path) -> None:
    """The original bug only surfaced at write time, so assert the write actually works."""
    verifier = _load_verifier()
    doc = tmp_path / "note.md"
    doc.write_text("cites arXiv:2404.14469\n", encoding="utf-8")

    out = verifier.default_output_path(doc)
    out.write_text("{}", encoding="utf-8")  # would raise FileNotFoundError before the fix

    assert out.is_file()
