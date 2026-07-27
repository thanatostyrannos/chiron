#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["markdown>=3.6"]
# ///
"""Build a self-contained docs/model-architecture.html from the committed markdown.

`docs/model-architecture.md` is the source of truth. This renders each ```mermaid block
to SVG and inlines it, so the result opens from disk in any browser with **no network,
no CDN and no JavaScript**. That matters for a repo that has to cold-start on a machine
with no internet, and it means the diagrams cannot silently break when a CDN moves.

Generated from the markdown rather than written alongside it, so the two cannot drift.

The SVGs render on a permanently light figure card even in dark mode: Mermaid bakes dark
text into the SVG, and "readable in one theme, invisible in the other" is a worse failure
than an unfashionable background.

Usage:
    uv run scripts/build_architecture_html.py
    uv run scripts/build_architecture_html.py --check   # exit 1 if the HTML is stale
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MD = REPO_ROOT / "docs" / "model-architecture.md"
OUTPUT_HTML = REPO_ROOT / "docs" / "model-architecture.html"
RENDERED_DIR = REPO_ROOT / "docs" / "diagrams" / "rendered"
DIAGRAM_DIR = REPO_ROOT / "docs" / "diagrams"

FENCE_RE = re.compile(r"^[ \t]*```mermaid[ \t]*\n(.*?)^[ \t]*```", re.MULTILINE | re.DOTALL)
SVG_TAG_RE = re.compile(r"<svg\b", re.IGNORECASE)

CSS = """
:root {
  --bg: #ffffff; --fg: #1a1d21; --muted: #5b6570; --rule: #e3e7ec;
  --accent: #2f6f4f; --card: #fbfcfd; --code-bg: #f4f6f8;
}
:root[data-theme="dark"], html:not([data-theme="light"]) {}
@media (prefers-color-scheme: dark) {
  :root { --bg: #14171a; --fg: #e6e9ec; --muted: #9aa4ae; --rule: #2a3037;
          --accent: #7fc9a1; --card: #1c2126; --code-bg: #1f242a; }
}
:root[data-theme="light"] { --bg:#fff; --fg:#1a1d21; --muted:#5b6570; --rule:#e3e7ec;
  --accent:#2f6f4f; --card:#fbfcfd; --code-bg:#f4f6f8; }
:root[data-theme="dark"] { --bg:#14171a; --fg:#e6e9ec; --muted:#9aa4ae; --rule:#2a3037;
  --accent:#7fc9a1; --card:#1c2126; --code-bg:#1f242a; }

* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 16px/1.65 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  /* Long unbreakable tokens (file paths, config keys) appear in prose as well as in
     code spans, and one of them pushed the page into horizontal scroll at 375px.
     break-word only breaks when a word genuinely cannot fit, so prose stays clean. */
  overflow-wrap: break-word;
}
.wrap { max-width: 1180px; margin: 0 auto; padding: 2.5rem 1.5rem 6rem; }
h1 { font-size: 2rem; line-height: 1.2; margin: 3rem 0 1rem;
     border-bottom: 2px solid var(--accent); padding-bottom: .5rem; }
h1:first-of-type { margin-top: 0; }
h2 { font-size: 1.35rem; margin: 2.5rem 0 .75rem; }
h3 { font-size: 1.1rem; margin: 1.75rem 0 .5rem; }
p, li { max-width: 78ch; }
a { color: var(--accent); }
/* overflow-wrap is load-bearing: a long inline path like
   `research/reference/architecture/transformers/.../modeling_laguna.py:365` is one
   unbreakable token and pushed the PAGE into horizontal scroll at narrow widths. */
code { background: var(--code-bg); padding: .12em .35em; border-radius: 4px;
       font: .88em ui-monospace, "Cascadia Code", Consolas, monospace;
       overflow-wrap: anywhere; word-break: break-word; }
pre { background: var(--code-bg); padding: 1rem; border-radius: 8px;
      overflow-x: auto; border: 1px solid var(--rule); }
pre code { background: none; padding: 0; }
blockquote { margin: 1.25rem 0; padding: .75rem 1.1rem; border-left: 3px solid var(--accent);
             background: var(--card); border-radius: 0 6px 6px 0; color: var(--fg); }
hr { border: 0; border-top: 1px solid var(--rule); margin: 2.5rem 0; }
table { border-collapse: collapse; margin: 1.25rem 0; display: block;
        overflow-x: auto; max-width: 100%; }
th, td { border: 1px solid var(--rule); padding: .5rem .7rem; text-align: left;
         vertical-align: top; }
th { background: var(--card); font-weight: 600; }

/* Diagrams sit on a permanently light card: mermaid bakes dark text into the SVG. */
figure.diagram {
  margin: 1.5rem 0; padding: 1.25rem; background: #ffffff;
  border: 1px solid var(--rule); border-radius: 10px; overflow-x: auto;
}
/* Deliberately NOT max-width:100%. That shrinks a wide diagram until its labels are
   unreadable, which is a silent failure -- it still looks like a diagram. Natural size
   plus a scrolling figure keeps every label legible; the page itself never scrolls. */
figure.diagram svg { height: auto; display: block; margin: 0 auto; }
figcaption { color: var(--muted); font-size: .85rem; margin-top: .9rem; text-align: center; }
figcaption a { text-decoration: none; border-bottom: 1px dotted currentColor; }

nav.toc {
  background: var(--card); border: 1px solid var(--rule); border-radius: 10px;
  padding: 1.25rem 1.5rem; margin: 2rem 0 3rem;
}
nav.toc h2 { margin: 0 0 .75rem; font-size: 1rem; text-transform: uppercase;
             letter-spacing: .06em; color: var(--muted); }
nav.toc ol { margin: 0; padding-left: 1.2rem; }
nav.toc li { margin: .3rem 0; }
.masthead { color: var(--muted); font-size: .9rem; margin-top: .5rem; }
.badge { display: inline-block; background: var(--card); border: 1px solid var(--rule);
         border-radius: 999px; padding: .15rem .7rem; font-size: .8rem;
         color: var(--muted); margin-right: .4rem; }
@media (max-width: 720px) { .wrap { padding: 1.5rem 1rem 4rem; } h1 { font-size: 1.6rem; } }
"""


def find_mmdc() -> list[str] | None:
    import os
    import shutil

    for name in ("mmdc", "mmdc.cmd"):
        found = shutil.which(name)
        if found:
            return ["cmd", "/c", found] if os.name == "nt" and found.endswith(".cmd") else [found]
    roots = [REPO_ROOT]
    if os.environ.get("CHIRON_MMDC_ROOT"):
        roots.append(Path(os.environ["CHIRON_MMDC_ROOT"]))
    wanted = (".cmd",) if os.name == "nt" else ("",)
    for root in roots:
        for cand in sorted(root.rglob("node_modules/.bin/mmdc*")):
            if cand.suffix.lower() in wanted:
                return ["cmd", "/c", str(cand)] if os.name == "nt" else [str(cand)]
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if npx:
        base = ["cmd", "/c", npx] if os.name == "nt" and npx.endswith(".cmd") else [npx]
        return [*base, "-y", "@mermaid-js/mermaid-cli"]
    return None


def svg_for(source: str, index: int, mmdc: list[str] | None) -> tuple[str, str | None]:
    """Inline SVG for one mermaid block, plus the slug it came from if identifiable.

    Prefers a pre-rendered file so a build is fast and reproducible; falls back to
    rendering on the spot for a block that lives only in the markdown.
    """
    first = source.strip().splitlines()[0].strip() if source.strip() else ""
    for mmd in sorted(DIAGRAM_DIR.glob("*.mmd")):
        if mmd.read_text("utf-8").strip() == source.strip():
            pre = RENDERED_DIR / f"{mmd.stem}.svg"
            if pre.is_file():
                return pre.read_text("utf-8"), mmd.stem
            break

    if mmdc is None:
        raise SystemExit(
            f"block {index} ({first[:40]}) has no pre-rendered SVG and mermaid-cli is "
            "unavailable. Run scripts/validate_diagrams.py --svg docs/diagrams/rendered"
        )
    with tempfile.TemporaryDirectory() as tmp:
        src, dest = Path(tmp) / "d.mmd", Path(tmp) / "d.svg"
        src.write_text(source, encoding="utf-8")
        cfg = Path(tmp) / "c.json"
        cfg.write_text('{"theme":"neutral"}', encoding="utf-8")
        proc = subprocess.run(
            [*mmdc, "-i", str(src), "-o", str(dest), "-c", str(cfg), "-b", "transparent"],
            capture_output=True, text=True, timeout=180,
        )
        if proc.returncode != 0 or not dest.is_file():
            raise SystemExit(f"block {index} failed to render: {(proc.stderr or '')[:200]}")
        return dest.read_text("utf-8"), None


def strip_svg_xml_header(svg: str) -> str:
    """Inline SVG must not carry an XML prolog or DOCTYPE inside an HTML body."""
    match = SVG_TAG_RE.search(svg)
    return svg[match.start():] if match else svg


def prepare_inline_svg(svg: str, index: int) -> str:
    """Make one mermaid SVG safe and legible once inlined next to ten others.

    Two defects that only appear when several are embedded in one document:

    1. **Every mermaid SVG uses `id="my-svg"`.** Eleven elements sharing one id is
       invalid HTML, and each SVG carries a `<style>` block scoped to `#my-svg`, so
       every diagram's stylesheet applies to every other diagram. Rewrite the id and
       its references to something unique per figure.
    2. **`width="100%"` with an inline `max-width`** makes a wide diagram shrink to the
       container instead of scrolling, until the labels are unreadable. It still looks
       like a diagram, which is the failure mode worth avoiding. Pin the width to the
       viewBox and let the figure scroll.
    """
    svg = strip_svg_xml_header(svg)
    prefix = f"d{index}"

    # Namespace EVERY id, not just the root. Mermaid emits arrow markers, node ids and
    # edge ids with fixed names, so eleven inlined diagrams collide on all of them --
    # measured at 137 duplicate ids before this. Markers are referenced by url(#id), so
    # a collision means several diagrams silently share one definition. Longest-first,
    # with a lookahead, so one id is never rewritten inside a longer one.
    ids = sorted(set(re.findall(r'\sid="([^"]+)"', svg)), key=len, reverse=True)
    for old in ids:
        new = f"{prefix}-{old}"
        svg = re.sub(rf'(?<=id="){re.escape(old)}(?=")', new, svg)
        svg = re.sub(rf"#{re.escape(old)}(?![\w\-])", f"#{new}", svg)

    viewbox = re.search(r'viewBox="([\d.\-]+) ([\d.\-]+) ([\d.]+) ([\d.]+)"', svg)
    if viewbox:
        natural_width = float(viewbox.group(3))
        svg = svg.replace('width="100%"', f'width="{natural_width:.0f}"', 1)
        svg = re.sub(r'max-width:\s*[\d.]+px;?', "", svg, count=1)
    return svg


def build() -> str:
    import markdown

    md_text = SOURCE_MD.read_text("utf-8")
    mmdc = find_mmdc()

    figures: list[tuple[str, str | None]] = []

    def swap(match: re.Match[str]) -> str:
        svg, slug = svg_for(match.group(1), len(figures) + 1, mmdc)
        figures.append((prepare_inline_svg(svg, len(figures) + 1), slug))
        return f"\n@@DIAGRAM{len(figures) - 1}@@\n"

    placeholder_md = FENCE_RE.sub(swap, md_text)
    body = markdown.markdown(
        placeholder_md,
        extensions=["tables", "fenced_code", "sane_lists", "toc"],
        output_format="html5",
    )

    for i, (svg, slug) in enumerate(figures):
        caption = ""
        if slug:
            caption = (
                f'<figcaption>Wide? scroll inside the frame &mdash; '
                f'<a href="diagrams/rendered/{slug}.svg">open {slug}.svg full size</a>'
                f'</figcaption>'
            )
        fig = f'<figure class="diagram">{svg}{caption}</figure>'
        body = body.replace(f"<p>@@DIAGRAM{i}@@</p>", fig).replace(f"@@DIAGRAM{i}@@", fig)

    heads = re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', body, re.DOTALL)
    toc = "".join(
        f'<li><a href="#{hid}">{re.sub("<[^>]+>", "", text).strip()}</a></li>'
        for hid, text in heads
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chiron — model architecture and system layouts</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<p class="masthead">
  <span class="badge">{len(figures)} diagrams</span>
  <span class="badge">self-contained &mdash; no network, no JS</span>
  <span class="badge">generated from docs/model-architecture.md</span>
</p>
<nav class="toc"><h2>Contents</h2><ol>{toc}</ol></nav>
{body}
<hr>
<p class="masthead">
  Generated by <code>scripts/build_architecture_html.py</code> from
  <code>docs/model-architecture.md</code>. Do not edit this file &mdash; edit the markdown
  and rebuild. Diagram sources are <code>docs/diagrams/*.mmd</code>, render-checked by
  <code>scripts/validate_diagrams.py</code>.
</p>
</div>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="exit 1 if the HTML is stale")
    args = ap.parse_args()

    generated = build()
    if args.check:
        if not OUTPUT_HTML.is_file() or OUTPUT_HTML.read_text("utf-8") != generated:
            print("model-architecture.html is stale — rebuild it", file=sys.stderr)
            raise SystemExit(1)
        print("model-architecture.html is current")
        return

    OUTPUT_HTML.write_text(generated, encoding="utf-8")
    kb = len(generated.encode("utf-8")) / 1024
    print(f"wrote {OUTPUT_HTML}  ({kb:,.0f} KiB, fully self-contained)")


if __name__ == "__main__":
    main()
