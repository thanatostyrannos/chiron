# research/memory/ — the memory-systems research track

The priority deliverable. Ten notes, each dense, sourced, with a `## Sources`
section. Read in this order (ordering lives here, not in filenames):

1. `memory-taxonomy.md` — fix the vocabulary: the five distinct things called
   "memory" (parametric, activation/recurrent state, KV cache, external retrieval,
   agent memory). Capacity, cost, read/write semantics, and hard limits of each.
2. `kv-cache-mechanics.md` — the shape math, why the KV cache dominates long-context
   memory and why decode is bandwidth-bound; GQA/MQA/MLA; FP8 KV.
3. `kv-compression-and-eviction.md` — H2O, SnapKV, PyramidKV, ChunkKV, KeyDiff,
   FastKV, RocketKV, L2 strategies; what each assumes about token importance and
   where it breaks.
4. `kv-serving-hierarchy.md` — PagedAttention block tables, prefix reuse,
   prefill/decode disaggregation, offload tiering, Mooncake, CXL-pooled KV. **Framed
   explicitly as a memory-hierarchy problem** — the founder's home turf.
5. `constant-state-memory.md` — Mamba-2/3, Gated DeltaNet, RWKV-7, mLSTM/xLSTM,
   Lightning Attention. The constant-state vs precise-recall trade, shown concretely.
6. `hybrid-architectures.md` — inter-layer (Jamba, Samba, Zamba2, Nemotron-H,
   Qwen3-Next, Kimi Linear) vs head-wise (Hymba); SWA+global interleaving. Are layer
   ratios chosen on evidence or folklore?
7. `long-context-behavior.md` — RoPE/YaRN/NoPE, length generalization, **effective
   vs advertised context**.
8. `agent-memory-systems.md` — working/episodic/semantic/procedural; MemGPT-Letta,
   A-MEM, memory-OS; the working-memory-is-a-context-budget-problem category error.
9. `memory-failure-register.md` — **the single most important file.** The G0
   Discovery Brief for this lab: every known failure with symptom, mechanism,
   evidence (citation), and open status. 5+ independent sources rule applies.
10. `open-problems-ranked.md` — synthesis: rank open problems by (a) real pain,
    (b) testability at 20M–300M params on our hardware, (c) whether systems
    expertise is a genuine edge. **This ranking becomes `BACKLOG.md`.**

Curriculum Track C mirrors these ten 1:1.
