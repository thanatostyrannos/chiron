# curriculum/ — the learning output

**31 modules and 4 support documents, ~430,000 words.** Written for an engineer with 30
years in distributed systems, storage hierarchies, caching, DR and observability, who is
new to ML internals. The method throughout is: bridge to something you already own, then
show precisely where the analogy breaks. **The break is the teaching.**

Modules are named for their subject and never numbered. Ordering lives in this file.

Every module has the same shape: what it settles → theory in plain language → the math
that matters, every symbol translated → why it matters for Proteus and Mnemosyne → read
the code, with `file:line` into `research/reference/` → 2–3 exercises runnable on the Z13
with a CPU fallback → self-check with answers at the end → what is still unsolved.

## Start here

- **`schedule.md`** — 12 weeks at ~8 hrs/week, sequenced by the prerequisite graph, with
  an honest reckoning of whether that is enough for this much material.
- **`glossary.md`** — 180 entries, plus a repo-wide notation index and the terms this lab
  has **banned** because two communities use them for different bytes.
- **`reading-list.md`** — the 76 API-verified papers ranked must/should/could, organised
  to read alongside the modules.
- **`capstones.md`** — three projects, each with a G2 hypothesis card, designed so a null
  result is still informative.

## Before any exercise runs

```powershell
scripts\fetch_reference.sh      # the clones are gitignored; file:line pointers need them
. .\scripts\activate-lab.ps1    # hipBLASLt is a NUMERICS control here, not a speed tweak
```

**`bf16-numerics-unproven` is untested and the Hardware Validation Gate has not run.**
Any exercise output that is a measured number is provisional until it passes.

## The modules, in reading order

| Module | Track | Prereqs |
|---|---|---|
| `tensors-and-autograd` | A — Foundations | — |
| `tokenization` | A | — |
| `transformer-forward-pass-by-hand` | A | tensors-and-autograd |
| `the-training-loop` | A | tensors-and-autograd, transformer-forward-pass-by-hand |
| `loss-and-optimization` | A | the-training-loop |
| `scaling-laws-and-flops-budget` | A | loss-and-optimization |
| `attention-variants-and-kv-cost` | B — Modern architecture | transformer-forward-pass-by-hand |
| `normalization-and-activations` | B | transformer-forward-pass-by-hand |
| `positional-encoding` | B | attention-variants-and-kv-cost |
| `moe-and-routing` | B | transformer-forward-pass-by-hand |
| `depth-width-and-initialization` | B | scaling-laws-and-flops-budget |
| `memory-taxonomy-for-engineers` | **C — Memory** | attention-variants-and-kv-cost, tensors-and-autograd |
| `kv-cache-mechanics` | **C** | memory-taxonomy-for-engineers |
| `kv-eviction-policies` | **C** | kv-cache-mechanics |
| `paged-attention-and-prefix-reuse` | **C** | kv-cache-mechanics |
| `constant-state-memory` | **C** | attention-variants-and-kv-cost |
| `hybrid-attention-and-ratios` | **C** | constant-state-memory, kv-cache-mechanics |
| `long-context-and-effective-context` | **C** | positional-encoding, kv-cache-mechanics |
| `agent-memory-in-practice` | **C** | memory-taxonomy-for-engineers |
| `memory-failure-modes` | **C** | kv-eviction-policies, long-context-and-effective-context |
| `measuring-memory` | **C** | memory-failure-modes |
| `distributed-training-strategies` | D — Training systems | the-training-loop, moe-and-routing |
| `checkpointing-and-resumption` | D | the-training-loop |
| `determinism-and-reproducibility` | D | the-training-loop |
| `training-telemetry-as-observability` | D | measuring-memory |
| `supervised-and-preference-finetuning` | E — Post-training & eval | the-training-loop |
| `building-an-eval-you-can-trust` | E | loss-and-optimization |
| `measuring-recall-and-memory` | E | building-an-eval-you-can-trust, memory-failure-modes |
| `quantization` | F — Inference | kv-cache-mechanics |
| `speculative-decoding-and-serving` | F | attention-variants-and-kv-cost, paged-attention-and-prefix-reuse |
| `running-laguna-locally` | F | quantization, kv-cache-mechanics |

**Track C is the deep track** and mirrors `research/memory/` 1:1 — it is where the lab's
research contribution lives. **`training-telemetry-as-observability` is the one to teach
back**: observability applied to attention, and the closest thing here to existing
expertise being directly load-bearing.

## Verification status — read before trusting a number

| Check | Coverage | Result |
|---|---|---|
| `file:line` pointers | **all 36 documents** | **723 / 723 resolve** |
| arXiv citations | 21 documents (Tracks A/B/C) | 147 resolved, **0 unresolved**, 132 unchecked |
| arXiv citations | 15 documents (Tracks D/E/F + support docs) | **never checked** |

That gap is real and was found by the glossary's cross-cutting pass, not by me. arXiv
began returning HTTP 429 after roughly a thousand queries on 2026-07-26. **Nothing has
failed verification** — 0 unresolved across everything checked repo-wide — but
"unchecked" is not "verified". Finish it with:

```
python scripts/verify_citations.py curriculum \
    --known research/reference/papers/anchors.bib \
    --resume curriculum/citation-verification.json \
    --out curriculum/citation-verification.json
```

## Known defects, recorded rather than quietly fixed

The glossary was the first document written across all 31 modules, and that vantage found
what no single author could. Notation collides between modules that are prerequisites of
each other:

- **`c` differs by 48×** — whole-stack KV bytes/token in `memory-taxonomy-for-engineers`,
  per-layer in `kv-cache-mechanics`, `hybrid-attention-and-ratios`, `quantization` and
  `speculative-decoding-and-serving`. `paged-attention-and-prefix-reuse` calls the
  whole-stack version `k`.
- **`B`** means batch, block size in tokens, eviction budget in entries, and prompt token
  budget, across four modules in one track.
- **`H`** is entropy in *nats* in `long-context-and-effective-context` and in *bits* in
  `building-an-eval-you-can-trust`.
- **`p`** is bytes-per-element, attention weight, output distribution, and survival
  probability in different modules; **`w`** is a sliding window everywhere except
  `quantization`.

No module is wrong on its own terms — each declares its symbols locally. `glossary.md`
carries the repo-wide notation index that makes the clashes visible. Reconciling them
means editing 31 documents and is a scheduled job, not an incidental one.

## What the exercises already found

Requiring exercises that produce a checkable number was not a pedagogical flourish. It
produced four hardware findings on first run, one of which changed how every experiment
here must be configured:

- **SDPA retains the score matrix by default** on gfx1151 (147.2 vs 6.6 bytes/T²).
  Deliberately not enabled by default — see `docs/adr/aotriton-attention-stays-off-by-default.md`.
- **hipBLASLt is a numerics control** worth ~2.8× in long bf16 reduction accuracy, not the
  +12% throughput setting it was first recorded as.
- **fp32 gradients match CPU to 3.9e-8** — the first gradient-correctness evidence here.
- **A reported hipBLASLt segfault did not survive retest** across eight controlled runs.
  Kept in `tokenization.md` as a worked example of a crash report tagged `[M]` without a
  repeatable basis.
