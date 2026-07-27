"""Check every `path/to/file.ext:LINE` pointer in a set of markdown docs still resolves.

The curriculum's read-the-code sections and CODE_MAP.md both send a reader to an exact
line in a cloned upstream repo. A pointer to the wrong line is worse than no pointer: the
reader loses trust in every other pointer in the document, and there is no way to tell a
stale one from a fabricated one by reading.

This only checks that the file exists and the line is in range -- it cannot know what
*should* be there. `scripts/generate_code_map.py` does the stronger check (the named
symbol must appear on the named line) for documents that record a symbol alongside the
pointer.

Line numbers are pinned to the revisions in research/reference/PROVENANCE.md. Re-fetching
upstream invalidates them; run this after any re-fetch.

Usage:
    python scripts/verify_code_pointers.py curriculum --base research/reference
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# `training/nanogpt/model.py:57` or `memory/vllm/vllm/v1/core/block_pool.py:647`
POINTER_RE = re.compile(r"\b([A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+\.[A-Za-z]{1,4}):(\d{1,6})\b")

SKIP_PREFIXES = ("http://", "https://")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="directory of .md files, or one file")
    ap.add_argument(
        "--base", default="research/reference", help="root the pointers are relative to"
    )
    ap.add_argument("--out")
    args = ap.parse_args()

    root = Path(args.path)
    base = Path(args.base)
    files = [root] if root.is_file() else sorted(root.rglob("*.md"))

    cited: dict[tuple[str, int], set[str]] = defaultdict(set)
    for md in files:
        for line in md.read_text(encoding="utf-8", errors="replace").splitlines():
            if any(p in line for p in SKIP_PREFIXES):
                line = re.sub(r"https?://\S+", " ", line)
            for m in POINTER_RE.finditer(line):
                cited[(m.group(1), int(m.group(2)))].add(md.name)

    print(f"{len(cited)} distinct file:line pointers across {len(files)} documents\n")

    def resolve(rel: str) -> Path | None:
        """Find the real file for a pointer written in any of the forms docs actually use."""
        # 1. relative to the reference root, the documented convention
        if (base / rel).is_file():
            return base / rel
        # 2. already carries the reference-root prefix (writers do this constantly)
        if Path(rel).is_file():
            return Path(rel)
        # 3. a short repo-relative tail like `moe/moe_utils.py` -- accept only if the
        #    suffix match is UNIQUE, otherwise it is genuinely ambiguous for a reader too
        matches = [
            p
            for p in base.rglob("*" + Path(rel).name)
            if str(p).replace("\\", "/").endswith(rel)
        ]
        return matches[0] if len(matches) == 1 else None

    ok, missing_file, out_of_range, elided = 0, [], [], []
    for (rel, line), docs in sorted(cited.items()):
        if "..." in rel:
            # A pointer a human cannot click is not a pointer. Report, do not resolve.
            elided.append((rel, line, sorted(docs)))
            continue
        target = resolve(rel)
        if target is None:
            missing_file.append((rel, line, sorted(docs)))
            continue
        n = sum(1 for _ in target.open("r", encoding="utf-8", errors="replace"))
        if line > n or line < 1:
            out_of_range.append((rel, line, n, sorted(docs)))
        else:
            ok += 1

    print(f"resolved      {ok}")
    print(f"missing file  {len(missing_file)}")
    print(f"out of range  {len(out_of_range)}")
    print(f"elided (...)  {len(elided)}")

    if elided:
        print("\nELIDED PATH -- unverifiable and unusable by a reader; write it in full:")
        for rel, line, docs in elided[:40]:
            print(f"  {rel}:{line}   in {', '.join(docs)}")

    if missing_file:
        print("\nFILE NOT FOUND (fetch the reference library, or the path is wrong):")
        for rel, line, docs in missing_file[:40]:
            print(f"  {rel}:{line}   cited in {', '.join(docs)}")
    if out_of_range:
        print("\nLINE OUT OF RANGE -- stale or fabricated:")
        for rel, line, n, docs in out_of_range[:40]:
            print(f"  {rel}:{line} (file has {n})   cited in {', '.join(docs)}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "resolved": ok,
                    "missing_file": [[r, ln, d] for r, ln, d in missing_file],
                    "out_of_range": [[r, ln, n, d] for r, ln, n, d in out_of_range],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if missing_file or out_of_range:
        sys.exit(1)


if __name__ == "__main__":
    main()
