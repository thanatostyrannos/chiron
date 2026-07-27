"""Check every arXiv id cited in a set of markdown notes actually resolves.

A survey is only as good as its citations, and a fabricated arXiv id is invisible to a
reader: it looks exactly like a real one. This scans notes for arXiv ids, resolves them
in batches against the arXiv API, and reports the resolved title beside each so a human
can spot a plausible-but-wrong citation.

Batched deliberately: the API accepts many ids per id_list query, and one-at-a-time
would take an hour for a survey-sized reference list.

An id that does not resolve is reported with the file that cited it. Unreachable is
distinguished from not-found -- a network failure is not evidence about a citation.

Usage:
    python scripts/verify_citations.py research/memory \
        --known research/reference/papers/anchors.bib
"""

from __future__ import annotations

import argparse
import json
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

try:
    import certifi

    SSL_CONTEXT: ssl.SSLContext | None = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # pragma: no cover
    SSL_CONTEXT = None

ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
# 15, not 40: a 40-id id_list query times out against this API more often than not
# (3 of 5 batches failed at 40). Smaller batches trade a few more round trips for
# actually completing.
BATCH = 15
SPACING_S = 3.1

# 2404.14469, arXiv:2404.14469v2, 1706.03762 -- but not version-less junk or dates.
ARXIV_RE = re.compile(r"\b(?:arXiv:)?(\d{4}\.\d{4,5})(?:v\d+)?\b")


class RateLimited(Exception):
    """arXiv is throttling this client. Not evidence about any citation -- resume later."""


def default_output_path(target: Path) -> Path:
    """Where the report goes when --out is not given.

    A directory input gets the report inside it. A SINGLE FILE input gets a sibling --
    the original version appended a child path to the file, which raises
    FileNotFoundError at write time, after the whole verification run has completed.
    See tests/test_verify_citations_output_path.py.
    """
    if target.is_dir():
        return target / "citation-verification.json"
    return target.with_name(f"{target.stem}-citation-verification.json")


def extract_ids(root: Path) -> dict[str, set[str]]:
    """Map arXiv id -> set of files citing it."""
    cited: dict[str, set[str]] = defaultdict(set)
    files = [root] if root.is_file() else sorted(root.rglob("*.md"))
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in ARXIV_RE.finditer(text):
            cited[match.group(1)].add(path.name)
    return cited


def resolve_batch(ids: list[str], attempts: int = 3) -> dict[str, str] | None:
    """Return {id: title} for ids that resolve. None if the API was unreachable."""
    params = {"id_list": ",".join(ids), "max_results": len(ids)}
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "chiron-research-lab/0.1"})

    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=120, context=SSL_CONTEXT) as resp:
                root = ET.fromstring(resp.read())
            break
        except urllib.error.HTTPError as exc:
            # 429 means we are the problem, not the network. arXiv throttles after a few
            # hundred queries; retrying on a 10-second ladder just deepens the hole. Honour
            # Retry-After when offered, otherwise back off in minutes, and give up quickly
            # so the caller can resume later rather than grinding.
            if exc.code == 429:
                wait = int(exc.headers.get("Retry-After", 0) or 0)
                if wait and attempt < attempts - 1:
                    print(f"    429 rate-limited; Retry-After={wait}s", flush=True)
                    time.sleep(min(wait, 300))
                    continue
                raise RateLimited(
                    "HTTP 429 -- arXiv is throttling this client; wait and --resume"
                ) from exc
            if attempt == attempts - 1:
                return None
            time.sleep(2**attempt * 5)
        except Exception:  # noqa: BLE001 - timeouts, resets, malformed batches
            if attempt == attempts - 1:
                return None
            time.sleep(2**attempt * 5)
    else:  # pragma: no cover
        return None

    found: dict[str, str] = {}
    for entry in root.findall(f"{ATOM}entry"):
        raw = entry.findtext(f"{ATOM}id", "")
        if "/abs/" not in raw:
            continue
        bare = raw.split("/abs/")[-1].split("v")[0]
        found[bare] = " ".join((entry.findtext(f"{ATOM}title") or "").split())
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="directory of .md notes, or a single file")
    ap.add_argument("--known", help="anchors.bib whose ids are already API-verified")
    ap.add_argument("--resume", help="a previous report; its resolved ids are not re-queried")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cited = extract_ids(Path(args.path))
    print(f"{len(cited)} distinct arXiv ids cited across the notes")

    known: set[str] = set()
    if args.known and Path(args.known).is_file():
        bib = Path(args.known).read_text(encoding="utf-8", errors="replace")
        known = {m.group(1) for m in ARXIV_RE.finditer(bib)}
        print(f"{len(known)} already verified via anchors.bib")

    resolved: dict[str, str] = {}
    if args.resume and Path(args.resume).is_file():
        prior = json.loads(Path(args.resume).read_text(encoding="utf-8-sig"))
        resolved.update(prior.get("resolved", {}))
        print(f"{len(resolved)} carried over from a previous run")

    to_check = sorted(set(cited) - known - set(resolved))
    print(f"{len(to_check)} need checking\n")
    unreachable: list[str] = []
    for i in range(0, len(to_check), BATCH):
        chunk = to_check[i : i + BATCH]
        print(f"  batch {i // BATCH + 1}: {len(chunk)} ids...", flush=True)
        try:
            got = resolve_batch(chunk)
        except RateLimited as exc:
            # Stop immediately. Grinding through the remaining batches against a throttle
            # just marks everything unreachable and wastes the quota that the resume needs.
            unreachable.extend(to_check[i:])
            print(f"    {exc} -- stopping with {len(unreachable)} unchecked", flush=True)
            break
        if got is None:
            unreachable.extend(chunk)
            print("    UNREACHABLE (recorded, not treated as fabricated)")
        else:
            resolved.update(got)
        time.sleep(SPACING_S)

    missing = [i for i in to_check if i not in resolved and i not in unreachable]

    print(f"\nresolved     {len(resolved)}")
    print(f"unresolved   {len(missing)}")
    print(f"unreachable  {len(unreachable)}")

    if missing:
        print("\nDID NOT RESOLVE -- treat as fabricated until proven otherwise:")
        for mid in missing:
            print(f"  {mid}   cited in: {', '.join(sorted(cited[mid]))}")

    report = {
        "resolved": {i: resolved[i] for i in sorted(resolved)},
        "unresolved": {i: sorted(cited[i]) for i in missing},
        "unreachable": unreachable,
        "already_known": sorted(set(cited) & known),
        "by_file": {f: sorted(i for i, fs in cited.items() if f in fs) for f in
                    sorted({f for fs in cited.values() for f in fs})},
    }
    out = Path(args.out) if args.out else default_output_path(Path(args.path))

    # A network outage once produced a report with 0 resolved and 279 unreachable, and
    # it overwrote a good report in place. A run that verified nothing must never be
    # allowed to replace a run that verified something -- that is data loss disguised as
    # a result. Divert to .partial and tell the caller how to resume.
    if unreachable and not resolved and out.exists():
        partial = out.with_suffix(".partial.json")
        partial.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(
            f"\nREFUSING TO OVERWRITE {out}: this run resolved nothing and"
            f" {len(unreachable)} ids were unreachable (the API was likely down)."
            f"\nwrote {partial} instead. Re-run with --resume {out} when the API is back.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    if missing or unreachable:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
