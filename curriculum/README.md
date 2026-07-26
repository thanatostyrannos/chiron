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
