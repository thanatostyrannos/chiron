"""Regenerate the ADR register table in docs/adr/README.md from the ADRs on disk.

The register carries a SHA-256 of each frozen ADR body, and `tests/test_adr_immutability.py`
fails if a frozen body no longer matches. Computing those hashes by hand would make
adding an ADR annoying enough to skip, and a control people skip is not a control.

Only ADRs in a frozen status get a hash. A `Proposed` ADR records `-`, because it is
still editable and pinning it would fail on every legitimate edit.

This script only ever rewrites the table between the header row and the end of the file.
It cannot touch an ADR body.

Usage:
    python scripts/update_adr_register.py            # rewrite the table
    python scripts/update_adr_register.py --check    # exit 1 if the table is stale
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ADR_DIR = REPO_ROOT / "docs" / "adr"
REGISTER = ADR_DIR / "README.md"

FROZEN_STATUSES = {"Accepted", "Deprecated", "Rejected"}
HEADER = "| Slug | Status | Date | Superseded by | Body SHA-256 |"
SEPARATOR = "|---|---|---|---|---|"


def frozen_body(text: str) -> str:
    """Everything below the Status block. Must match tests/test_adr_immutability.py."""
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("Status:"):
            start = i
            continue
        if start is not None and not line.startswith((" ", "\t", "Date:", "Deciders:")):
            return "\n".join(lines[i:]).strip()
    return ""


def field(text: str, name: str) -> str:
    match = re.search(rf"^{name}:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def superseded_by(text: str) -> str:
    # Plain ASCII on purpose. This table is hashed, diffed and read through tools with
    # inconsistent encoding defaults (PowerShell 5.1 reads UTF-8 as ANSI), and a stray
    # em-dash shows up as mojibake in exactly the artifact that is supposed to be
    # byte-stable.
    match = re.search(r"^\s+Superseded by (\S+)", text, re.MULTILINE)
    return f"`{match.group(1)}`" if match else "-"


def build_rows() -> list[str]:
    rows = []
    for path in sorted(p for p in ADR_DIR.glob("*.md") if p.name != "README.md"):
        text = path.read_text("utf-8")
        status = field(text, "Status").split()[0] if field(text, "Status") else "?"
        digest = (
            hashlib.sha256(frozen_body(text).encode("utf-8")).hexdigest()
            if status in FROZEN_STATUSES
            else "-"
        )
        rows.append(
            f"| `{path.stem}` | {status} | {field(text, 'Date')} | "
            f"{superseded_by(text)} | {digest} |"
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="exit 1 if the table is stale")
    args = ap.parse_args()

    text = REGISTER.read_text("utf-8")
    if HEADER not in text:
        raise SystemExit(f"register header not found in {REGISTER}")

    head = text[: text.index(HEADER)]
    rows = build_rows()
    rebuilt = head + HEADER + "\n" + SEPARATOR + "\n" + "\n".join(rows) + "\n"

    if args.check:
        if rebuilt != text:
            print("ADR register is stale -- run scripts/update_adr_register.py", file=sys.stderr)
            raise SystemExit(1)
        print(f"register current: {len(rows)} ADRs")
        return

    REGISTER.write_text(rebuilt, encoding="utf-8")
    # Count the hash column directly rather than pattern-matching the rendered row --
    # the previous version tested for " - |" and silently reported 0 hashed when 3 were.
    hashed = sum(1 for r in rows if len(r.rsplit("|", 2)[1].strip()) == 64)
    print(f"wrote {REGISTER}: {len(rows)} ADRs, {hashed} frozen and hashed")


if __name__ == "__main__":
    main()
