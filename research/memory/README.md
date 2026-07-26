# research/memory/ — the memory-systems research track

**The lab's priority research track.** Ten notes, ~52,000 words, written 2026-07-26.
Every arXiv id cited across them was machine-verified against the live arXiv API —
**265 distinct ids, 0 unresolved** — see `citation-verification.json` and
`scripts/verify_citations.py`. Resolving an id proves the paper exists, not that it
supports the claim beside it: treat the report as a fabrication check, not a
correctness proof.

Curriculum Track C mirrors these ten 1:1.

## Reading order

Read these three first; they carry the argument.

| Note | What it settles |
|---|---|
| `memory-taxonomy.md` | Fixes the vocabulary. Five mechanically unrelated things get called "memory" — weights, recurrent state, KV cache, retrieval index, session store. The axis that actually partitions them is **reconstructibility**, not capacity or speed. |
| `memory-failure-register.md` | **The G0 Discovery Brief.** Every known failure with symptom / mechanism / evidence / open-or-solved. Claims resting only on a Future Work mention are marked UNPROVEN DEMAND. |
| `open-problems-ranked.md` | The ranking that becomes `BACKLOG.md`. Scores pain × testability-here × our-edge, and disqualifies anything untestable on one GPU regardless of how interesting it is. |

Then the mechanism notes, in dependency order:

| Note | What it settles |
|---|---|
| `kv-cache-mechanics.md` | Per-token KV cost as a closed form over five config fields, and why decode is bandwidth-bound. Derives decode arithmetic intensity = **2G/dtype_bytes** — for bf16, exactly the GQA group size, independent of context length and depth. |
| `kv-compression-and-eviction.md` | H2O, SnapKV, PyramidKV, ChunkKV, KeyDiff, StreamingLLM and the rest — what each *assumes* about token importance, and precisely where the assumption breaks. |
| `kv-serving-hierarchy.md` | The serving layer as a memory hierarchy: paged block tables, prefix reuse, prefill/decode disaggregation, offload tiering. Bridged to storage-hierarchy experience, then broken where the analogy fails. |
| `constant-state-memory.md` | SSMs and linear attention. The trade — constant state and linear time against degraded precise recall — shown concretely on associative recall rather than asserted. |
| `hybrid-architectures.md` | Inter-layer vs head-wise hybrids, and the live question of whether published ratios were *ablated* or *inherited*. |
| `long-context-behavior.md` | RoPE/YaRN/NoPE, length generalization, and effective vs advertised context. |
| `agent-memory-systems.md` | Working/episodic/semantic/procedural memory, the MemGPT-Letta lineage, and the category error of treating a context-*budget* problem as a retrieval problem. |

## What the track concluded

**Reconstructibility, not speed, is the load-bearing distinction.** The KV cache and
recurrent state are exactly recoverable by recompute — they are memo tables, not storage
tiers. The session store is the only authoritative tier, which is why the poisoning
literature exists there and nowhere else, and it tells Mnemosyne which guarantees it
actually owes.

**The field's own diagnosis is a measurement problem.** Serving papers report latency and
throughput without isolating which mechanism produced the gain; several groups argue most
of PyramidKV's reported benefit comes from SnapKV's observation window rather than the
per-layer budget allocation it claims credit for. That is an attribution failure, and
attribution is the top-ranked item in the backlog (P5 · T5 · E5).

**Ratios may be folklore.** `hybrid-architectures.md` separates papers that ablated their
interleaving ratio from papers that inherited one. Laguna's 3:1 is `[M]` from the shipped
config; whether 3:1 is *right* is untested by anyone.

## Contested, and left contested

Seven points are recorded as live disputes rather than resolved. The largest:
**whether the KV cache is "memory" at all** — serving papers treat it as a tensor buffer
with a lifetime and an owner, agent-memory papers as a first-class tier with a scheduler,
and the two camps rarely cross-cite. Also open: whether agentic memory is memory or
externalized note-taking; whether human-memory analogies earn their keep; and parametric
capacity at ~2 bits/param vs ~3.6 bits/param, which are two different quantities that
must not be averaged.

## Standing caveat

These notes are `[C]`-grade: literature and code reading, which the G0 translation calls
*cheap and directional*. Reproducing a failure in our own rig is the expensive, decisive
evidence, and none of that has happened. The Hardware Validation Gate has not run, so
**no number measured on this machine counts as evidence yet.**
