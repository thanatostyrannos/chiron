# research/reference/ — cloned upstream code + papers

**Authored in the Reference Library phase.** Clones are gitignored (reproduced by
`scripts/fetch_reference.sh`, which IS committed); only `*.md` and `*.bib` files here
are tracked. No weights — `GIT_LFS_SKIP_SMUDGE=1` on the HF clones.

Deliverables that live here:

- `CODE_MAP.md` — the guided tour with `file:line` pointers (Laguna attention layout
  & MoE gating; PagedAttention block table; a prefix-cache hit path; Mamba-2 selective
  scan; Gated DeltaNet update rule; a hybrid's layer-interleaving config; OLMo's
  training loop / FSDP / dataloader / checkpointer). This map is what makes the
  curriculum's "read the code" exercises possible — a deliverable, not a byproduct.
- `PROVENANCE.md` — the ledger: URL, SHA/revision, license, date, purpose for
  everything fetched. This stack moves weekly; an unversioned benchmark is worthless.
- `papers/` — PDFs + a BibTeX file: the anchoring surveys (long-context, KV-cache
  management, efficient architectures, agent memory, agent-memory security) plus the
  primary paper for every technique named in the Frontier Survey.

Reference sets to clone (config + modeling code is enough where weights are
irrelevant): Laguna S/XS 2.1 and the `laguna` llama.cpp/transformers sources; the
hybrid/memory comparison set (Mamba-2/3, Jamba, Samba, Zamba2, Nemotron-H,
Qwen3-Next, Kimi Linear, MiniMax-01, RWKV-7, xLSTM, Hymba, GPT-OSS, Gemma 3);
runnable training infra (OLMo/OLMo-core, torchtitan, smollm, nanoGPT, lm-eval-harness);
serving/memory (vllm, sglang, flashinfer, Mooncake, letta); and the hardware prior
art (ROCm #6034 / amdsense reproduction scripts — reusable as our validation harness).
