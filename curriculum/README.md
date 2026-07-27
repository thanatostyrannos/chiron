# curriculum/ — the learning output (the centerpiece)

**Authored in the Curriculum phase** (owned by curriculum-author, reviewed by every
specialist for accuracy). Audience: an expert systems engineer who is an ML beginner
and learns by building and by reading source. This file becomes the full map — what
you learn, in what order, why, and the honest prerequisite list.

Modules are named for their subject, never numbered (`attention-variants.md`,
`kv-cache-mechanics.md`). Every module has the same shape:

> theory in plain language → the math that actually matters (every symbol
> translated) → why it matters for Proteus → read-the-code (`file:line` into
> `research/reference/`) → 2–3 hands-on exercises runnable on the Z13 (gfx1151,
> native Windows) with a CPU fallback → self-check questions with answers → what's
> still unsolved here.

## Modules written so far

Read in the order listed — this table is the collection's ordering, since filenames
carry no sequence (naming rule). **Tracks A, B and C are written; D, E and F are not yet.**
Track C is the deep track and mirrors `research/memory/` 1:1.

**Verification status.** Every `file:line` pointer is machine-verified —
**416/416 resolving** (`scripts/verify_code_pointers.py`). Citations are **partially
verified: 147 of 279 arXiv ids resolved, 0 unresolved, 132 still unchecked** because
arXiv began returning HTTP 429 after roughly a thousand queries today. Unchecked is not
the same as unverified-bad: no fabrication has been found in the curriculum, and none of
the 279 has failed. Finish with:

```
python scripts/verify_citations.py curriculum --known research/reference/papers/anchors.bib     --resume curriculum/citation-verification.json --out curriculum/citation-verification.json
```

See `citation-verification.json` for exactly which ids are still outstanding.

| Module | Track | Prereqs |
|---|---|---|
| `tensors-and-autograd.md` | A | — |
| `transformer-forward-pass-by-hand.md` | A | tensors-and-autograd |
| `tokenization.md` | A | — |
| `the-training-loop.md` | A | tensors-and-autograd, transformer-forward-pass-by-hand |
| `loss-and-optimization.md` | A | the-training-loop |
| `scaling-laws-and-flops-budget.md` | A | loss-and-optimization |
| `attention-variants-and-kv-cost.md` | B | transformer-forward-pass-by-hand |
| `normalization-and-activations.md` | B | transformer-forward-pass-by-hand |
| `positional-encoding.md` | B | attention-variants-and-kv-cost |
| `moe-and-routing.md` | B | transformer-forward-pass-by-hand |
| `depth-width-and-initialization.md` | B | scaling-laws-and-flops-budget |
| `memory-taxonomy-for-engineers.md` | **C** | attention-variants-and-kv-cost, tensors-and-autograd |
| `kv-cache-mechanics.md` | **C** | memory-taxonomy-for-engineers |
| `kv-eviction-policies.md` | **C** | kv-cache-mechanics |
| `paged-attention-and-prefix-reuse.md` | **C** | kv-cache-mechanics |
| `constant-state-memory.md` | **C** | attention-variants-and-kv-cost |
| `hybrid-attention-and-ratios.md` | **C** | constant-state-memory, kv-cache-mechanics |
| `long-context-and-effective-context.md` | **C** | positional-encoding, kv-cache-mechanics |
| `agent-memory-in-practice.md` | **C** | memory-taxonomy-for-engineers |
| `memory-failure-modes.md` | **C** | kv-eviction-policies, long-context-and-effective-context |
| `measuring-memory.md` | **C** | memory-failure-modes |

### Findings the exercises produced

Module authors ran their own exercises on the Z13 before shipping them, which is the
point of requiring exercises that produce a checkable number. Three hardware claims came
out of it; **all three were then independently retested, and one did not survive.**

- **`[M]` SDPA is not memory-efficient by default on gfx1151.** 147.2 bytes/T² retained
  vs 6.6 with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`; saved activations at T=512
  fall 38.0 MiB → 2.1 MiB (~18×). `flash_sdp_enabled()` returns True either way.
  Confirmed independently — see `ASSUMPTIONS.md → sdpa-is-memory-efficient` and
  `scripts/measure_attention_memory_path.py`. The flag stays **off** by default because
  it is experimental and therefore a numerics change; the Hardware Validation Gate must
  run numerics both ways.
- **`[M]` fp32 gradients on gfx1151 match CPU to 3.9e-8 absolute** — the first
  gradient-correctness evidence on this machine, a free by-product of an exercise.
- **Refuted: the reported hipBLASLt segfault on skinny-K GEMMs.** Retested in isolated
  subprocesses across four shapes including the two originally reported, with and
  without the variables: all eight runs exited 0. Kept in `tokenization.md` as a worked
  example of a crash report that was tagged `[M]` without a repeatable basis.

## Six tracks

- **A — Foundations:** tensors/autograd; the transformer forward pass by hand;
  tokenization; the training loop; loss & optimization; scaling laws and the FLOPs
  budget (6·N·D).
- **B — Modern architecture:** attention variants & KV-cost implications; norms &
  activations; positional encoding; MoE & routing; depth/width & initialization.
- **C — MEMORY (the deep track, largest, mirrors `research/memory/` 1:1):** bridge
  each concept to systems knowledge the founder owns, then show where the analogy
  breaks (KV cache ↔ working set + eviction; paged attention ↔ virtual memory & page
  tables; prefix caching ↔ shared read-only tier with invalidation hazards; offload
  tiering ↔ hot/warm/cold storage; SSM state ↔ fixed-size rolling aggregate vs
  unbounded log; agent memory ↔ write-ahead store with schema drift, no compaction).
- **D — Training systems:** FSDP/TP/PP/EP ↔ sharding & replication; checkpointing ↔
  DR; determinism & resumption; training telemetry ↔ observability (the module he
  teaches back).
- **E — Post-training & evaluation:** SFT/DPO/RLVR; building an eval you can trust at
  small scale; measuring memory and recall specifically.
- **F — Inference:** quantization, speculative decoding, serving; running Laguna XS
  2.1 locally and reading its behavior against the architecture notes.

Plus `glossary.md`, `reading-list.md` (must/should/could), `schedule.md` (12 weeks
@ ~8 hrs/wk, memory weighted heaviest), and `capstones.md` (three: a KV-eviction
policy and where it breaks recall; a hybrid block and its ratio cliff; an
end-to-end-instrumented training run with an injected fault to detect).
