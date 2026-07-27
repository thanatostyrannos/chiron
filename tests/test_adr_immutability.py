"""An accepted ADR's body must match the SHA-256 recorded in the ADR register.

CLAUDE.md and OPERATING_INSTRUCTIONS.md both state that an accepted ADR is a permanent
record: never modified, never renamed, never deleted. Discipline alone is not a control
-- you cannot prevent someone editing a file, but you can make the edit impossible to
commit unnoticed. That is this test.

The "frozen portion" is everything below the Status block, matching the register's own
definition. The Status block itself is excluded because the one permitted write is
appending a supersession line to it.

Filesystem-based on purpose: no torch, no installed packages, runs on the CPU-only
scaffold.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ADR_DIR = REPO_ROOT / "docs" / "adr"
REGISTER = ADR_DIR / "README.md"

FROZEN_STATUSES = {"Accepted", "Deprecated", "Rejected"}

# | `slug` | Status | Date | Superseded by | Body SHA-256 |
ROW_RE = re.compile(
    r"^\|\s*`(?P<slug>[a-z0-9-]+)`\s*\|\s*(?P<status>[^|]+?)\s*\|"
    r"\s*(?P<date>[^|]*?)\s*\|\s*(?P<superseded>[^|]*?)\s*\|\s*(?P<sha>[0-9a-f]{64}|-)\s*\|",
    re.MULTILINE,
)


def frozen_body(adr_text: str) -> str:
    """Everything below the Status block -- the portion an accepted ADR may never change."""
    lines = adr_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("Status:"):
            start = i
            continue
        # The Status block ends at the first line that is neither a continuation
        # (indented, e.g. an appended supersession line) nor another header field.
        if start is not None and not line.startswith((" ", "\t", "Date:", "Deciders:")):
            return "\n".join(lines[i:]).strip()
    return ""


def adr_status(adr_text: str) -> str:
    match = re.search(r"^Status:\s*(\S+)", adr_text, re.MULTILINE)
    return match.group(1) if match else ""


def register_rows() -> dict[str, dict[str, str]]:
    return {m.group("slug"): m.groupdict() for m in ROW_RE.finditer(REGISTER.read_text("utf-8"))}


def adr_files() -> list[Path]:
    return sorted(p for p in ADR_DIR.glob("*.md") if p.name != "README.md")


def test_every_adr_has_a_register_row() -> None:
    rows = register_rows()
    missing = [p.stem for p in adr_files() if p.stem not in rows]
    assert not missing, (
        "ADRs exist with no row in docs/adr/README.md. The register is the enforcement "
        "mechanism; an unregistered ADR is unprotected:\n  " + "\n  ".join(missing)
    )


def test_no_register_row_without_an_adr() -> None:
    present = {p.stem for p in adr_files()}
    orphans = [slug for slug in register_rows() if slug not in present]
    assert not orphans, (
        "The register names ADRs that do not exist -- an ADR was renamed or deleted, "
        "which the immutability rule forbids:\n  " + "\n  ".join(orphans)
    )


def test_frozen_adr_bodies_match_their_recorded_hash() -> None:
    rows = register_rows()
    mismatches: list[str] = []
    for path in adr_files():
        text = path.read_text("utf-8")
        status = adr_status(text)
        if status not in FROZEN_STATUSES:
            continue
        recorded = rows.get(path.stem, {}).get("sha", "-")
        actual = hashlib.sha256(frozen_body(text).encode("utf-8")).hexdigest()
        if recorded != actual:
            mismatches.append(
                f"{path.name}: register {recorded[:12]}... != actual {actual[:12]}..."
            )

    assert not mismatches, (
        "The body of an ADR that is no longer editable has changed. Corrections are "
        "APPENDED via a new superseding ADR, never applied in place:\n  "
        + "\n  ".join(mismatches)
    )


def test_proposed_adrs_are_not_hashed() -> None:
    """A Proposed ADR is still editable, so pinning its hash would be a false control."""
    rows = register_rows()
    wrongly_pinned = [
        path.stem
        for path in adr_files()
        if adr_status(path.read_text("utf-8")) == "Proposed"
        and rows.get(path.stem, {}).get("sha", "-") != "-"
    ]
    assert not wrongly_pinned, (
        "Proposed ADRs must record '-' for the body hash; they are editable until "
        "accepted, and a pinned hash would fail on every legitimate edit:\n  "
        + "\n  ".join(wrongly_pinned)
    )
