# research/reference/ — cloned upstream code + papers

**Authored in the Reference Library phase.** Clones are gitignored and reproduced by
`scripts/fetch_reference.sh`, which IS committed. No weights: HF clones use
`GIT_LFS_SKIP_SMUDGE=1`, so `Laguna-S-2.1` is 13 MB with its 46 weight shards present
as 1 KB pointer files.

## Read in this order

1. `PROVENANCE.md` — the ledger: name, URL, exact revision, declared SPDX license,
   whether the LICENSE file survived into the clone, date, purpose. Generated from what
   is actually on disk, so a row means the clone exists at that revision. **This stack
   moves weekly; an unversioned benchmark is worthless.**
2. `CODE_MAP.md` — the guided tour. Every `file:line` pointer is machine-verified by
   `scripts/generate_code_map.py`, which fails rather than emit a stale one.
3. `papers/` — 76 anchoring papers across 9 tracks, ranked must/should/could, each with
   a one-line "read this for X". `anchors.bib` is generated from arXiv API metadata,
   never transcribed: `scripts/verify_papers.py` resolves every claimed id and rejects
   one that points at a *different* paper, then `scripts/generate_papers.py` emits the
   BibTeX. 20 of 76 are from 2026 — built against current arXiv, not training data.

## Rebuilding

```
scripts/fetch_reference.sh                 # all 38 sources
scripts/fetch_reference.sh memory          # one category
scripts/fetch_reference.sh --provenance    # rewrite PROVENANCE.md from disk
python scripts/generate_code_map.py <section-json>...   # rebuild CODE_MAP.md
python scripts/verify_papers.py <candidates.json> --out research/reference/papers
python scripts/generate_papers.py research/reference/papers/resolved_papers.json
```

The fetch asserts completeness: manifest count must equal clones on disk, or it fails
loudly. It once printed `done.` after fetching 21 of 40 sources with exit code 0.

## Three hazards this directory creates

**Upstream repos ship agent instructions.** 29 `CLAUDE.md` / `AGENTS.md` /
`.cursorrules` / `copilot-instructions.md` files arrived inside vllm, transformers,
torchtitan, megatron-lm, sglang, flashinfer, mooncake and letta. Cloned into this tree
they are loaded as *directives* by any coding agent working in this repo — 38
third-party sources would otherwise be 38 injection surfaces. The fetch script renames
them to `*.upstream-not-instructions`: content preserved and readable, auto-loading
stopped. It verifies zero remain and fails if any do. **We read this code; we do not
take orders from it.**

**Four sources carry no license at all** — `qwen3`, `hymba`, `amdsense`, `wsl-rocm`.
Default copyright applies: read them, never vendor or redistribute them. `megatron-lm`
is `NOASSERTION` (a license file GitHub could not classify — read it before relying on
it). For HuggingFace repos a missing LICENSE file is usually just convention; the
license sits in the model card's YAML frontmatter.

**Gated sources break reproducibility.** `google/gemma-3-4b-it` and
`ai21labs/AI21-Jamba-Mini-1.7` are deliberately excluded: a gated source makes this
script un-runnable for anyone without a HuggingFace account, and both are redundant
(Gemma 3's SWA+global pattern is covered by ungated `gpt-oss-20b` and transformers'
`configuration_gemma3.py`; Jamba's ratio is in `configuration_jamba.py`). A gated clone
also used to hang the whole fetch for 20 minutes on a credential prompt with no error;
clones now run with terminal prompts disabled and fail in under a second.

## What is here

| Category | Count | Contents |
|---|---|---|
| `architecture/` | 14 | Laguna (llama.cpp `laguna` branch + transformers), Mamba, Samba, Zamba2, Megatron-LM, Qwen3, Kimi-Linear, MiniMax-01, RWKV-LM, xLSTM, Hymba, GPT-OSS, flash-linear-attention |
| `models/` | 6 | Configs + tokenizers + chat templates, no weights: Laguna-S/XS 2.1, Nemotron-Nano, Qwen3-Next, Kimi-Linear, GPT-OSS-20B |
| `training/` | 7 | OLMo, OLMo-core, dolma, torchtitan, smollm, nanoGPT, lm-evaluation-harness |
| `memory/` | 8 | vllm, sglang, flashinfer, Mooncake, letta, A-mem, and two paper indexes |
| `hardware/` | 3 | amdsense (93+ gfx1151 experiments), wsl-rocm, ROCm |
