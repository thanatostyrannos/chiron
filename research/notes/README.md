# research/notes/ — the frontier survey

How a 2026 frontier model actually works, end to end. Parallel subagent dispatch;
each note is dense, sourced, with a `## Sources` section, ~3–5 pages. Recency rule:
search the last 6 months before writing; present contested topics as contested.

| File | Owner | Must answer |
|---|---|---|
| `transformer-state-of-the-art.md` | ml-architect | What a 2026 decoder looks like end to end (norms, activations, MHA→GQA→MQA→MLA, positional, depth/width, tokenizer) and why each change since the 2023 recipe won. |
| `moe-routing-and-failure-modes.md` | ml-architect | Routing (top-k, sigmoid vs softmax, shared experts), load balancing, granularity, capacity/dropped tokens, upcycling, sparsity tradeoff; expert collapse, hot experts, instability. |
| `pretraining-recipes.md` | training-infra-engineer | Optimizers, LR/batch schedules, μP/HP transfer, scaling laws, data mixing/curriculum, long-context stages, stability tricks. |
| `posttraining-pipelines.md` | ml-architect | SFT → preference optimization → RLVR → agentic RL; reasoning-mode training; how "check your work" behavior is induced. |
| `inference-and-quantization.md` | training-infra-engineer | FP8/INT4/NVFP4/GGUF, speculative decoding (incl. DFlash), batching, the decode bandwidth ceiling. |
| `evaluation-landscape.md` | research-lead | 2026 agentic + long-context evals; what's cheaply reproducible at ablation scale; why needle-in-a-haystack is insufficient. |
| `open-weights-landscape.md` | research-lead | Who ships what, openness tiers, where the genuinely open questions sit. |

The memory track lives one level up in `research/memory/` and gets the most depth.
`research/synthesis.md` is written last, to the G1 standard.
