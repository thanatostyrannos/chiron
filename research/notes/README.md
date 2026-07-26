# research/notes/ — the frontier survey

How a 2026 frontier model actually works, end to end. Seven notes, ~43,000 words,
written 2026-07-26 by the specialist owners below. Every arXiv id cited across them was
machine-verified against the live arXiv API — **384 distinct ids, 0 unresolved**
(`scripts/verify_citations.py`; report in `citation-verification.json`). Resolving an id
proves the paper exists, not that it supports the claim beside it.

The memory track in `research/memory/` is the lab's priority and is deeper; these notes
connect to it rather than duplicating it. `research/synthesis.md` integrates both.

| File | Owner | What it settles |
|---|---|---|
| `transformer-state-of-the-art.md` | ml-architect | The 2023 recipe is still the skeleton; every widely-adopted change since is a stability or bandwidth fix, and only QK-norm and attention output gating have controlled multi-scale ablations behind them. The real structural delta is that **a layer is no longer uniform** — attention type, query-head count, RoPE schedule and MLP type are per-layer lookups, and Laguna varies all four. |
| `moe-routing-and-failure-modes.md` | ml-architect | Routing, load balancing and the failure modes: expert collapse, hot experts, router saturation. Reads Laguna's shipped combination — sigmoid gating, aux-loss-free bias correction, router-logit softcapping present-but-zero — and says what it implies. |
| `pretraining-recipes.md` | training-infra-engineer | Optimizers (AdamW vs Muon and the live dispute over how its advantage scales), WSD schedules, μP transfer, scaling laws and the current critiques of Chinchilla. Constrained throughout to what actually runs on one gfx1151 GPU with no collectives. |
| `posttraining-pipelines.md` | ml-architect | SFT → preference optimization → RLVR → agentic RL, and — more usefully for us — an explicit account of which parts are **not** reproducible at our scale. |
| `inference-and-quantization.md` | training-infra-engineer | Quantization for weights *and* KV, speculative decoding, and the decode bandwidth ceiling wired to our own measured numbers. On this box FP8 buys memory economics but not tensor-core economics: `torch._scaled_mm` is unsupported on gfx1151. |
| `evaluation-landscape.md` | research-lead | The 2026 eval landscape, what is cheaply reproducible at ablation scale, and the question this lab lives on: **which evals can actually detect a memory regression** — distinguish recall from pattern-matching. |
| `open-weights-landscape.md` | research-lead | Who ships what, the openness tiers (weights-only vs weights+data vs fully reproducible), and the licence reality that determines what we may build on and publish. |

## Two findings that changed our own register

**Laguna's per-layer head structure, read from the artifact.** Query heads are 48 on the
12 full-attention layers and 72 on the 36 sliding ones, while `num_key_value_heads` is
uniform at 8. So KV bytes/token is exact at 192 KiB — query heads do not enter that
product — but the **GQA group size varies: 6 on full layers, 9 on sliding**. Decode
arithmetic intensity is therefore per-layer-type, and a cost model keyed on the top-level
`num_attention_heads: 48` is wrong for 36 of 48 layers. This corrected an error in
`ASSUMPTIONS.md` and in four memory notes.

**Inherited convention, caught by arithmetic.** Laguna's `attention_factor`
`1.4852030263919618` matches YaRN's published default temperature `0.1·ln(s)+1` at
`s=128` to the last digit, and Laguna-XS's matches it at `s=32`. And `8192 × 128 =
1,048,576` exactly — the advertised 1M context is an arithmetic consequence of the
extension factor, not an independently demonstrated capability. Invisible unless you
compute it.
