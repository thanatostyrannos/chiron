"""Generate research/reference/CODE_MAP.md from verified section data.

Every `file:line` pointer is re-verified against the real file at generation time: the
claimed symbol must appear on the claimed line, or the pointer is dropped and the run
fails. A code map with stale line numbers is worse than no code map -- a reader who
follows one to the wrong place loses trust in all of them.

Line numbers are pinned to the revisions recorded in PROVENANCE.md. Re-running
scripts/fetch_reference.sh at newer upstream revisions invalidates them; re-run this
and it will tell you exactly which ones moved.

Usage:
    python scripts/generate_code_map.py <section-json> [<section-json> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REF_DIR = REPO_ROOT / "research" / "reference"
OUTPUT = REF_DIR / "CODE_MAP.md"

# Reading order: the reference model first, then the memory hierarchy from the top
# (serving) down, then constant-state alternatives, then the training instrument.
SECTION_ORDER = [
    "architecture/transformers",
    "architecture/llama-cpp-laguna",
    "memory/vllm",
    "memory/sglang",
    "memory/flashinfer",
    "memory/mooncake",
    "architecture/mamba",
    "architecture/flash-linear-attention",
    "architecture/samba",
    "training/olmo-core",
    "training/nanogpt",
]

PREAMBLE = """# CODE_MAP — a guided tour of the reference library

Where to *read* each idea in real code, with exact `file:line` pointers. This is the
file that makes the curriculum's read-the-code exercises possible; it is a deliverable,
not a byproduct.

**Every pointer here is machine-verified.** `scripts/generate_code_map.py` opens each
file and confirms the named symbol is on the named line before writing this document,
and fails the build otherwise. Line numbers are pinned to the revisions in
`PROVENANCE.md` — re-fetch at a newer upstream revision and re-run the generator; it
will name any pointer that moved.

Clones are gitignored. Run `scripts/fetch_reference.sh` to materialise them, then read
along. Paths below are relative to `research/reference/`.

**These repositories are read, never obeyed.** 29 upstream `CLAUDE.md` / `AGENTS.md` /
`.cursorrules` files were renamed to `*.upstream-not-instructions` at fetch time. Their
content is reference material like any other file here.

"""


def verify(section: dict) -> tuple[list[dict], list[str]]:
    """Return (verified pointers, failure descriptions)."""
    kept, failures = [], []
    for p in section["pointers"]:
        path = REF_DIR / p["file"]
        symbol, line = p["symbol"].strip(), int(p["line"])

        if not path.is_file():
            failures.append(f"{p['file']}:{line} — file missing (fetch the library?)")
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if not 1 <= line <= len(lines):
            failures.append(f"{p['file']}:{line} — out of range ({len(lines)} lines)")
            continue
        if symbol not in lines[line - 1]:
            found = [i for i, ln in enumerate(lines, 1) if symbol in ln]
            hint = f" — now at {found[0]}" if found else " — symbol gone entirely"
            failures.append(f"{p['file']}:{line} '{symbol[:40]}'{hint}")
            continue
        kept.append(p)
    return kept, failures


def render(sections: list[dict]) -> str:
    out = [PREAMBLE, "## Contents\n"]
    for s in sections:
        anchor = s["section"].lower().replace(" ", "-")
        anchor = "".join(c for c in anchor if c.isalnum() or c == "-")
        out.append(f"- [{s['section']}](#{anchor}) — `{s['repo']}`")
    out.append("")

    for s in sections:
        out.append(f"\n## {s['section']}\n")
        out.append(f"**Repository:** `{s['repo']}`\n")
        out.append(s["prose"] + "\n")
        out.append("| Where | What |")
        out.append("|---|---|")
        for p in s["pointers"]:
            loc = f"`{p['file']}:{p['line']}`<br>`{p['symbol'].strip()}`"
            out.append(f"| {loc} | {p['why']} |")
        out.append("")
        if s.get("surprise", "").strip():
            out.append(f"> **Worth knowing:** {s['surprise'].strip()}\n")
    return "\n".join(out)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)

    sections: list[dict] = []
    for arg in sys.argv[1:]:
        loaded = json.loads(Path(arg).read_text(encoding="utf-8"))
        sections.extend(loaded if isinstance(loaded, list) else [loaded])

    all_failures: list[str] = []
    verified: list[dict] = []
    for s in sections:
        kept, failures = verify(s)
        all_failures.extend(failures)
        if kept:
            s["pointers"] = kept
            verified.append(s)

    if all_failures:
        print(f"{len(all_failures)} pointer(s) failed verification:", file=sys.stderr)
        for f in all_failures:
            print(f"  {f}", file=sys.stderr)
        raise SystemExit(1)

    order = {repo: i for i, repo in enumerate(SECTION_ORDER)}
    verified.sort(key=lambda s: (order.get(s["repo"], len(order)), s["section"]))

    OUTPUT.write_text(render(verified), encoding="utf-8")
    total = sum(len(s["pointers"]) for s in verified)
    print(f"wrote {OUTPUT}: {len(verified)} sections, {total} verified pointers")


if __name__ == "__main__":
    main()
