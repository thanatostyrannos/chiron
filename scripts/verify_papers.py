"""Verify paper citations against the arXiv API and emit BibTeX from the API's own data.

Inventing a citation is the worst failure available in a research repo: it is
undetectable by reading, it propagates into every downstream document, and it destroys
the credibility of the citations that are real. So no BibTeX entry here is written from
recall. Each candidate is resolved against the live arXiv API and the entry is built
from the returned metadata -- title, authors, date, abstract -- not from what anyone
believed the paper was called.

Two checks per candidate:
  1. The claimed arXiv id resolves to a real paper.
  2. That paper's title actually matches the claimed title (token-overlap similarity).

A candidate whose id resolves to a DIFFERENT paper is the dangerous case -- a plausible
id attached to the wrong work -- so it is reported loudly rather than silently fixed.
Candidates with no id are searched by title; a confident match is adopted.

Usage:
    python scripts/verify_papers.py <candidates.json> [--out research/reference/papers]
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
from dataclasses import dataclass, field
from pathlib import Path

# The lab venv (C:\venvs\lab) ships no CA bundle, so TLS verification fails outright
# against arxiv.org. certifi supplies one. Never fall back to an unverified context --
# an unauthenticated citation source is exactly what this script exists to prevent.
try:
    import certifi

    SSL_CONTEXT: ssl.SSLContext | None = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # pragma: no cover - depends on the venv, not on logic
    SSL_CONTEXT = None
    print("certifi not installed; TLS may fail. pip install certifi", file=sys.stderr)

ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
REQUEST_SPACING_S = 3.1  # arXiv asks for one request per 3 seconds. Be a good citizen.
TITLE_MATCH_THRESHOLD = 0.65

_STOPWORDS = {"a", "an", "the", "of", "for", "and", "with", "in", "on", "to", "via", "is"}


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: list[str]
    published: str
    updated: str
    categories: list[str] = field(default_factory=list)

    @property
    def year(self) -> str:
        return self.published[:4]

    @property
    def bibkey(self) -> str:
        if self.authors:
            surname = re.sub(r"[^a-z]", "", self.authors[0].split()[-1].lower())
        else:
            surname = "anon"
        words = re.findall(r"[A-Za-z]+", self.title)
        word = next(
            (w for w in words if w.lower() not in _STOPWORDS and len(w) > 3),
            "paper",
        )
        return f"{surname}{self.year}{word.lower()}"


def _normalise(title: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", title.lower()) if w not in _STOPWORDS}


def title_similarity(a: str, b: str) -> float:
    ta, tb = _normalise(a), _normalise(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


class TransientAPIError(Exception):
    """The API could not be reached. NOT evidence that a paper does not exist."""


def _query(params: dict, attempts: int = 3) -> list[Paper]:
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "chiron-research-lab/0.1"})

    last: Exception | None = None
    for attempt in range(attempts):
        try:
            # 90s, not 45: the arXiv API is routinely slow -- a bare id_list lookup was
            # measured at 23.9s here. A short timeout turns latency into a false
            # "not found", which is the one answer this script must never invent.
            with urllib.request.urlopen(req, timeout=90, context=SSL_CONTEXT) as resp:
                root = ET.fromstring(resp.read())
            break
        except urllib.error.HTTPError as exc:
            # 400 from arXiv means a malformed/nonexistent id -- a real answer, not a flake.
            if exc.code == 400:
                return []
            last = exc
        except Exception as exc:  # noqa: BLE001 - timeouts, DNS, resets
            last = exc
        if attempt < attempts - 1:
            time.sleep(2 ** attempt * 4)
    else:
        raise TransientAPIError(f"{type(last).__name__}: {last}")

    out = []
    for entry in root.findall(f"{ATOM}entry"):
        raw_id = entry.findtext(f"{ATOM}id", "")
        if "/abs/" not in raw_id:
            continue
        out.append(
            Paper(
                arxiv_id=raw_id.split("/abs/")[-1],
                title=" ".join((entry.findtext(f"{ATOM}title") or "").split()),
                authors=[
                    (a.findtext(f"{ATOM}name") or "").strip()
                    for a in entry.findall(f"{ATOM}author")
                ],
                published=entry.findtext(f"{ATOM}published", "")[:10],
                updated=entry.findtext(f"{ATOM}updated", "")[:10],
                categories=[c.get("term", "") for c in entry.findall(f"{ATOM}category")],
            )
        )
    return out


def fetch_by_id(arxiv_id: str) -> Paper | None:
    """Raises TransientAPIError if the API was unreachable — never returns None for that."""
    bare = arxiv_id.strip().removeprefix("arXiv:").split("v")[0]
    found = _query({"id_list": bare, "max_results": 1})
    return found[0] if found else None


def search_by_title(title: str) -> Paper | None:
    """Raises TransientAPIError if the API was unreachable — never returns None for that."""
    found = _query({"search_query": f'ti:"{title}"', "max_results": 5})
    if not found:
        time.sleep(REQUEST_SPACING_S)
        loose = " ".join(sorted(_normalise(title))[:8])
        found = _query({"search_query": f"all:{loose}", "max_results": 8})
    if not found:
        return None
    best = max(found, key=lambda p: title_similarity(title, p.title))
    return best if title_similarity(title, best.title) >= TITLE_MATCH_THRESHOLD else None


def to_bibtex(paper: Paper, why: str, tier: str) -> str:
    authors = " and ".join(paper.authors) if paper.authors else "Unknown"
    primary = paper.categories[0] if paper.categories else "cs.LG"
    return "\n".join(
        [
            f"@misc{{{paper.bibkey},",
            f"  title         = {{{paper.title}}},",
            f"  author        = {{{authors}}},",
            f"  year          = {{{paper.year}}},",
            f"  eprint        = {{{paper.arxiv_id}}},",
            "  archivePrefix = {arXiv},",
            f"  primaryClass  = {{{primary}}},",
            f"  url           = {{https://arxiv.org/abs/{paper.arxiv_id}}},",
            f"  note          = {{[{tier}] {why}}},",
            "}",
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("candidates")
    ap.add_argument("--out", default="research/reference/papers")
    args = ap.parse_args()

    # utf-8-sig, not utf-8: Windows PowerShell 5.1 writes a BOM with -Encoding utf8, and
    # json.loads rejects it. Harmless on BOM-less files.
    tracks = json.loads(Path(args.candidates).read_text(encoding="utf-8-sig"))
    if isinstance(tracks, dict):
        tracks = [tracks]

    resolved: list[dict] = []
    rejected: list[dict] = []
    unreachable: list[dict] = []
    seen_ids: set[str] = set()

    for track in tracks:
        name = track.get("track", "unknown")
        print(f"\n=== {name} ===")
        for cand in track.get("papers", []):
            claimed_title = cand["title"].strip()
            claimed_id = cand.get("arxiv_id", "UNKNOWN").strip()
            time.sleep(REQUEST_SPACING_S)

            paper: Paper | None = None
            how = ""
            try:
                if claimed_id and claimed_id.upper() != "UNKNOWN":
                    by_id = fetch_by_id(claimed_id)
                    if by_id is None:
                        how = "id-not-found"
                    elif title_similarity(claimed_title, by_id.title) >= TITLE_MATCH_THRESHOLD:
                        paper, how = by_id, "id-confirmed"
                    else:
                        # The dangerous case: a real id pointing at a different paper.
                        # Never silently repair this -- it is the signature of a
                        # fabricated citation wearing a plausible id.
                        how = "mismatch"
                        rejected.append(
                            {
                                "track": name,
                                "claimed_title": claimed_title,
                                "claimed_id": claimed_id,
                                "reason": "ID MISMATCH - id resolves to a different paper",
                                "id_actually_is": by_id.title,
                            }
                        )
                        print(f"  MISMATCH  {claimed_id}")
                        print(f"            claimed: {claimed_title[:70]}")
                        print(f"            actual : {by_id.title[:70]}")

                if paper is None and how != "mismatch":
                    time.sleep(REQUEST_SPACING_S)
                    paper = search_by_title(claimed_title)
                    if paper is not None:
                        how = "title-search"

            except TransientAPIError as exc:
                # The API was unreachable. That is NOT evidence the paper is fake, and
                # recording it as "not found" would be a wrong conclusion dressed as a
                # verified one -- the precise failure this script exists to prevent.
                unreachable.append(
                    {
                        "track": name,
                        "claimed_title": claimed_title,
                        "claimed_id": claimed_id,
                        "error": str(exc),
                    }
                )
                print(f"  UNREACHED {claimed_title[:60]}  ({exc})")
                continue

            if paper is None:
                if how != "mismatch":
                    rejected.append(
                        {
                            "track": name,
                            "claimed_title": claimed_title,
                            "claimed_id": claimed_id,
                            "reason": "not found on arXiv by id or title",
                        }
                    )
                    print(f"  NOTFOUND  {claimed_title[:70]}")
                continue

            if paper.arxiv_id in seen_ids:
                print(f"  dup       {paper.arxiv_id}  {paper.title[:55]}")
                continue
            seen_ids.add(paper.arxiv_id)

            flag = "" if how == "id-confirmed" else f"  ({how})"
            print(f"  OK        {paper.arxiv_id}  {paper.title[:55]}{flag}")
            resolved.append(
                {
                    "track": name,
                    "tier": cand.get("tier", "could"),
                    "why_read": cand.get("why_read", ""),
                    "resolution": how,
                    "claimed_id": claimed_id,
                    "paper": paper.__dict__,
                }
            )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_papers.json").write_text(
        json.dumps(
            {"resolved": resolved, "rejected": rejected, "unreachable": unreachable}, indent=2
        ),
        encoding="utf-8",
    )

    print(f"\nresolved {len(resolved)}   rejected {len(rejected)}   unreachable {len(unreachable)}")
    if rejected:
        print("\nrejected — recorded, never silently dropped:")
        for r in rejected:
            print(f"  [{r['track']}] {r['reason']}: {r['claimed_title'][:60]}")

    if unreachable:
        print(
            f"\n{len(unreachable)} candidate(s) could not be checked at all. These are"
            " UNVERIFIED, not disproved — re-run before treating the list as complete:",
            file=sys.stderr,
        )
        for u in unreachable:
            print(f"  [{u['track']}] {u['claimed_title'][:60]}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
