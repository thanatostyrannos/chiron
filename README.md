# Model Architecture Research Lab — Prompt Pack

Rebuilt around learning and evidence rather than shipping a product. Takes an empty directory to: a reference library of current architectures, a sourced survey of how frontier models work in 2026, a **deep memory-systems research track**, a complete learning curriculum, and an ablation rig that can answer "does change X help?" with real evidence at 20M–300M params.

## Contents

```
KICKOFF_PROMPT.md         Phases 0–6: scaffold → reference library → frontier
                          survey + memory track → curriculum → rig design →
                          ablation backlog. One commit + tag per phase.
ABLATION_LOOP_PROMPT.md   Two modes. BUILD (m6–m13) constructs the rig;
                          EXPERIMENT runs pre-registered ablations forever.
repo-seed/
  OPERATING_INSTRUCTIONS.md  Your G0–G5 operating loop, verbatim, marked as the
                          governing document — plus a RESEARCH-LAB INTERPRETATION
                          appendix translating G0 (Discovery) and G3 (Economics)
                          for a research context, and the G4 TDD carve-out.
  CLAUDE.md               Rules, experimental standards, hardware, founder context.
  .claude/agents/         research-lead, ml-architect, memory-systems-researcher,
                          data-engineer, training-infra-engineer,
                          ablation-engineer, tech-writer, curriculum-author
```

Changed from the venture version: business-strategist is gone, `memory-systems-researcher` and `ablation-engineer` are new, and `study-guide-educator` became `curriculum-author` with a much larger remit. The build loop became a hypothesis loop.

## Run it

Drop `seed-lab.ps1` and `research-lab-promptpack.zip` together in `c:\projects\school`, then:

```powershell
.\seed-lab.ps1
```

No arguments. It finds the zip beside itself, extracts it, creates `c:\projects\school\chiron`,
places every file including the hidden `.claude\` directory, promotes `CLAUDE.md` /
`OPERATING_INSTRUCTIONS.md` / `.claude\` / `scripts\` to the repo root where Claude Code reads
them at session start, unblocks everything, verifies all 8 subagents landed, and runs the
environment preflight.

If Windows blocks the script: `Unblock-File .\seed-lab.ps1` — or
`powershell -ExecutionPolicy Bypass -File .\seed-lab.ps1`.

It refuses to run against a path containing spaces (breaks pip/conda on the gfx1151 stack) or
over an existing git repo (pass `-Force` to override).

Then:

1. Read `ENVIRONMENT.md` and clear anything preflight flagged. Claude Code is pre-authorized to
   install the toolchain itself — you handle BIOS UMA sizing, the AMD driver, and `gh auth login`.
2. `claude`, paste `KICKOFF_PROMPT.md` (set the project names and `CLOUD_BUDGET` first).
3. Later sessions: paste `ABLATION_LOOP_PROMPT.md`. State lives in the repo, so every session
   cold-starts cleanly.

Reading order when the kickoff finishes: `ASSUMPTIONS.md` → `research/memory/memory-taxonomy.md`
→ `research/memory/memory-failure-register.md` → `research/synthesis.md` → `curriculum/README.md`.

## The four systems

| Name | System | Owns |
|---|---|---|
| **Chiron** | the lab | repo, docs, curriculum — tutor of Achilles and Asclepius |
| **Proteus** | model architecture | decoder, attention variants, MoE, the config surface |
| **Mnemosyne** | memory subsystem | KV cache, eviction, tiering, attribution instrumentation |
| **Themis** | ablation rig | pre-registration, matched budgets, seeds, aggregation |

A uv workspace with three packages under `packages/`. `mnemosyne` depends on torch only and never
imports `proteus` — that dependency direction is what makes the separability claim mechanically
true rather than aspirational.

## The memory track

Ten research notes, mirrored 1:1 by curriculum modules, covering the five distinct things people call "memory" and refusing to conflate them:

| | |
|---|---|
| **Parametric** | knowledge in weights |
| **Activation state** | SSM/linear-attention hidden state — constant size, lossy |
| **KV cache** | attention's working set — the dominant long-context cost |
| **External retrieval** | RAG over static documents |
| **Agent memory** | cross-session stores — your Honcho/NemoAI territory |

Then: KV mechanics and the bandwidth ceiling → compression and eviction (H2O, SnapKV, PyramidKV, ChunkKV, KeyDiff) → the serving layer as a memory hierarchy (PagedAttention, prefix reuse, prefill/decode disaggregation, offload tiering, CXL pooling) → constant-state memory (Mamba-2/3, Gated DeltaNet, RWKV-7, xLSTM) → hybrids (Jamba, Samba, Nemotron-H, Qwen3-Next, Kimi Linear, Hymba; SWA+global as Laguna, Gemma 3, and GPT-OSS do it) → long-context and effective-vs-advertised context → agent memory → **the failure-mode register** → open problems ranked by whether *you specifically* can make progress on them.

That register is the file that matters most. It's also where the ablation backlog comes from.

## Why this shape

Two things in the recent literature point at the same gap. Serving papers report latency and throughput improvements without isolating which mechanism produced them. And hybrid layer ratios get inherited across papers without being retested. Both are measurement and attribution problems in a memory hierarchy — which is a thirty-year systems career pointed directly at an open question. The rig is built for attribution from milestone mnemosyne-core onward for exactly that reason.

## Hardware: Z13 primary

Decided: the ROG Flow Z13 (Ryzen AI Max+ 395, gfx1151, 128GB unified) is the
primary research machine. Three invariants baked into the pack:

- **Native Windows preferred, WSL2 as fallback** — AMD ships gfx1151 ROCm nightlies
  with PyTorch for Windows, and PyTorch runs CI on this chip. WSL2 is now officially
  supported for Strix Halo too, but a documented `librocdxg` bug clamps the ROCm
  memory pool to the `.wslconfig` limit and can't reach dedicated VRAM — which would
  negate the whole reason for choosing this machine. The Hardware Validation capacity test settles
  which platform you're on before any research runs.
- **Single-device only** — distributed collectives are incomplete on this arch, so
  FSDP/DDP is design-only until rented hardware validates it.
- **The Hardware Validation gate blocks everything** — bf16 bugs are documented on this silicon, so
  no result counts as evidence until numerics, determinism, checkpoint integrity,
  and a known-good recipe all pass. Every ROCm/PyTorch upgrade re-opens the gate.

The upside is real, not consolation: 128GB of addressable memory runs long-context
and KV-capacity experiments that won't fit on a 20GB discrete card, and a published
93-experiment campaign on this same silicon measured higher MFU than an RTX 4090 on
a memory-bound recipe. Slower wall-clock, better instrument for this specific
research.

## Cost

Ablation scale is genuinely cheap: a 100M-param model on 2B tokens is ~1.2×10¹⁸ FLOPs — under an hour on a rented H100. Spot capacity runs well under $1/hr with interruption risk, mid-tier on-demand around $2–3/hr. A few hundred dollars a month buys dozens of controlled experiments. Develop on the 7900 XT, but rent the hour when the result matters rather than fighting ROCm-under-WSL2 — and run Laguna XS 2.1 on the Z13 as your working reference the whole time.
