"""Render every Mermaid diagram in the repo and fail on any that does not.

A broken diagram does not look broken in source -- it looks like a diagram. It only
fails at render time, in front of a reader, as a grey error box. Since the whole point
of choosing Mermaid over a binary format was that diagrams are reviewable text, they
have to be verifiable text too.

Covers both places Mermaid lives here:
  - docs/diagrams/*.mmd            -- standalone sources
  - ```mermaid fenced blocks       -- embedded in any markdown under the scanned paths

Validation is by ACTUAL RENDER via mermaid-cli, not a syntax heuristic: if it produces
an SVG it will render for a reader. Requires node; resolves mmdc from PATH, then from a
local node_modules, then falls back to `npx -y @mermaid-js/mermaid-cli`.

Usage:
    python scripts/validate_diagrams.py                    # docs/ and research/ and curriculum/
    python scripts/validate_diagrams.py docs/diagrams
    python scripts/validate_diagrams.py --svg docs/diagrams/rendered
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = ["docs", "research", "curriculum"]

# research/reference/<category>/<repo>/ holds cloned upstream code. Their diagrams are
# theirs: vllm and llama.cpp both ship Mermaid that does not render, and failing our
# build on someone else's broken docs would make this check something people switch off.
# We read that code; we do not own it. Top-level research/reference/*.md is ours.
EXCLUDED_PARTS = ("node_modules", ".git", "site-packages")


def _is_upstream_clone(path: Path) -> bool:
    try:
        rel = path.relative_to(REPO_ROOT / "research" / "reference")
    except ValueError:
        return False
    return len(rel.parts) > 1
FENCE_RE = re.compile(r"^[ \t]*```mermaid[ \t]*\n(.*?)^[ \t]*```", re.MULTILINE | re.DOTALL)


def _runnable(command: list[str]) -> list[str]:
    """Windows entry points are .cmd shims; CreateProcess cannot exec them directly."""
    if os.name == "nt" and command[0].lower().endswith((".cmd", ".bat")):
        return ["cmd", "/c", *command]
    return command


def find_mmdc() -> list[str] | None:
    """mmdc from PATH, a local install anywhere under a search root, or npx last."""
    for name in ("mmdc", "mmdc.cmd"):
        found = shutil.which(name)
        if found:
            return _runnable([found])

    search_roots = [REPO_ROOT]
    scratch = os.environ.get("CHIRON_MMDC_ROOT")
    if scratch:
        search_roots.append(Path(scratch))
    # npm writes two shims side by side: an extensionless sh script and a .cmd. On
    # Windows CreateProcess rejects the sh script with WinError 193, so the extension
    # is not cosmetic -- pick the one the platform can actually execute.
    wanted = (".cmd",) if os.name == "nt" else ("",)
    for root in search_roots:
        for candidate in sorted(root.rglob("node_modules/.bin/mmdc*")):
            if candidate.suffix.lower() in wanted:
                return _runnable([str(candidate)])

    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if npx:
        return _runnable([npx, "-y", "@mermaid-js/mermaid-cli"])
    return None


def collect(paths: list[str]) -> list[tuple[str, str]]:
    """Return (label, mermaid_source) for every diagram found."""
    found: list[tuple[str, str]] = []
    for raw in paths:
        root = REPO_ROOT / raw if not Path(raw).is_absolute() else Path(raw)
        if not root.exists():
            continue
        def ours(path: Path) -> bool:
            if _is_upstream_clone(path):
                return False
            return not any(part in EXCLUDED_PARTS for part in path.parts)

        for mmd in sorted(p for p in root.rglob("*.mmd") if ours(p)):
            found.append((str(mmd.relative_to(REPO_ROOT)), mmd.read_text("utf-8")))
        for md in sorted(p for p in root.rglob("*.md") if ours(p)):
            text = md.read_text("utf-8", errors="replace")
            for i, match in enumerate(FENCE_RE.finditer(text), 1):
                line = text[: match.start()].count("\n") + 1
                label = f"{md.relative_to(REPO_ROOT)}:{line} (block {i})"
                found.append((label, match.group(1)))
    return found


def render(mmdc: list[str], source: str, out_svg: Path | None) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "d.mmd"
        src.write_text(source, encoding="utf-8")
        dest = out_svg or (Path(tmp) / "d.svg")
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            proc = subprocess.run(
                [*mmdc, "-i", str(src), "-o", str(dest)],
                capture_output=True, text=True, timeout=180,
            )
        except subprocess.TimeoutExpired:
            return False, "render timed out after 180s"
    if proc.returncode == 0:
        return True, ""
    blob = (proc.stderr or "") + (proc.stdout or "")
    detail = next(
        (ln.strip() for ln in blob.splitlines() if "rror" in ln or "xpect" in ln),
        blob.strip().splitlines()[-1] if blob.strip() else "unknown failure",
    )
    return False, detail[:160]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", default=DEFAULT_PATHS)
    ap.add_argument("--svg", help="also write rendered SVGs into this directory")
    args = ap.parse_args()

    mmdc = find_mmdc()
    if mmdc is None:
        print("mermaid-cli not found and npx unavailable; cannot validate.", file=sys.stderr)
        raise SystemExit(2)
    print(f"validator: {' '.join(mmdc)}")

    diagrams = collect(args.paths or DEFAULT_PATHS)
    print(f"{len(diagrams)} diagram(s) found\n")

    failures: list[tuple[str, str]] = []
    for label, source in diagrams:
        out = None
        if args.svg and label.endswith(".mmd"):
            out = REPO_ROOT / args.svg / (Path(label).stem + ".svg")
        ok, detail = render(mmdc, source, out)
        print(f"  {'OK  ' if ok else 'FAIL'}  {label}" + ("" if ok else f"\n          {detail}"))
        if not ok:
            failures.append((label, detail))

    print(f"\nrendered {len(diagrams) - len(failures)}/{len(diagrams)}")
    if failures:
        print("\nThese render as a grey error box for a reader:", file=sys.stderr)
        for label, detail in failures:
            print(f"  {label}: {detail}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
