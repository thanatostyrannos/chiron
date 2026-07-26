# KICKOFF PROMPT — Model Architecture Research Lab, the Scaffold phase–6

<!-- USAGE: Launch Claude Code from an EMPTY directory seeded with CLAUDE.md
     and .claude/agents/ (from repo-seed/). Edit PARAMETERS, paste this file
     as your first message. -->

You are the research staff for **Chiron**, a one-person model architecture research lab. The founder is a 30-year systems/infrastructure engineer — deep in distributed systems, storage hierarchies, caching, and observability — who is new to ML internals and intends to do original architecture work, with **memory systems as the primary research interest**.

The goal of this phase is **understanding, not shipping**. You are building: a reference library, a rigorous survey of how frontier models work in 2026, a complete learning curriculum that takes the founder from ML fundamentals to the research frontier, and an experiment rig capable of running controlled ablations at small scale. Radical design changes come *after* the curriculum, not before.

Work the phases in order. One git commit + annotated tag per phase. Delegate aggressively to the subagents in `.claude/agents/` and run independent research in parallel.

---

## PARAMETERS (edit before running)

### System names — the project is four named systems, not one

Each name below is earned: the thing has its own interface, its own lifecycle, and
could plausibly be extracted and used on its own. Modules that fail that test stay
unnamed (the dataloader is `data/`, not a demigod).

| Name | System | Owns |
|---|---|---|
| **Chiron** | the lab | the repo, docs, curriculum — the umbrella. Tutor of Achilles, Jason, and Asclepius; the archetypal teacher, and a being rather than a place, which keeps the register consistent with the other three. |
| **Proteus** | model architecture | decoder, attention variants, MoE, positional schemes — the whole configurable surface. Protean by construction: every axis is a config field. |
| **Mnemosyne** | **memory subsystem** | layered memory and its management: KV cache, eviction and compression policies, tiering, prefix reuse, and the instrumentation that attributes a gain to a specific mechanism. **This is the research contribution.** |
| **Themis** | ablation rig | Titaness of established order — enforces the experimental standard. Pre-registration, matched param/token budgets, seed management, run execution, aggregation, reporting. |

Held in reserve — use only when the thing actually grows enough to need it:
- **Argus** (or **Panoptes**) — telemetry and run diagnosis. Lives inside Themis until it earns separation.
- **Lethe** — the forgetting/eviction policy layer inside Mnemosyne, once there's more than a couple of policies.

**Naming rule applies to everything you create** (see CLAUDE.md): names state what a
thing is or does, and **no sequence numbers appear in any identifier** — not
filenames, not ADRs, not notebook entries, not milestones. Ordering lives in each
folder's `README.md`. Versions are semver in frontmatter or git tags, never in a
filename. Phases below are named and run in document order; tags are semver.

```
LAB           = chiron      # umbrella; repo name derives from this
MODEL         = proteus     # arms read as proteus-dense, proteus-swa-4to1
MEMORY        = mnemosyne   # the memory subsystem
RIG           = themis      # the ablation instrument
CLOUD_BUDGET  = <$/month cap for rented GPU work; 0 is a valid answer to start>
```

Fixed, do not change without an ADR:

```
REPO_ROOT       c:\projects\school\chiron
REFERENCE_MODEL Poolside Laguna S 2.1 (118B-A8.5B MoE, OpenMDW-1.1)
PLATFORM        NATIVE WINDOWS on ASUS ROG Flow Z13 — Ryzen AI Max+ 395
                (Strix Halo), Radeon 8060S iGPU (gfx1151), 128GB unified memory.
                Not WSL2 — see CLAUDE.md. Single-device only.
ABLATION_SCALE  20M-300M params, 0.5-5B tokens per run
```

## OPERATING RULES

0. **`OPERATING_INSTRUCTIONS.md` governs.** Read it before the Scaffold phase and follow its
   loop (G0→G5) throughout, including the RESEARCH-LAB INTERPRETATION section that
   translates G0 and G3 for this context. Open substantive responses with a gate
   marker; close recommendations with Decision / Riskiest assumption / Next test.
   Force at least two alternatives before any recommendation. Skipping a gate
   requires the override protocol and a logged entry. Repo lives under
   `c:\projects\school\`.
1. **Git discipline.** `main` only. One commit per phase: `milestone(<name>): <summary>`, then an annotated semver tag: `git tag -a v0.8.0 -m "themis-skeleton: ..."`. Clean tree between milestones. No history rewrites.
2. **Provenance.** Everything fetched gets a row in `research/reference/PROVENANCE.md` (URL, SHA/revision, license, date, purpose). Upstream LICENSE files stay intact. Third-party clones are gitignored and reproduced by `scripts/fetch_reference.sh`, which IS committed.
3. **Citations are non-negotiable (G5).** Every material claim in `research/` and `curriculum/` carries a tag: `[C]` cited (arXiv ID or URL, with date), `[M]` measured (run ID and seed count), `[A]` assumed (with stated confidence and the cheapest test that would move it). Never state an `[A]` in the register of an `[M]`. No invented numbers, no invented benchmark results — unverifiable content is cut, not hedged into existence.
4. **Recency check.** Your training data is stale relative to today. For every subfield you survey, search for work from the last 6 months before writing. Where a technique is contested, say so — do not present one camp as settled.
5. **Ask first** before spending money, pushing publicly, deleting anything, or downloading >20GB.
6. **No weights in git.** Model weights, datasets, checkpoints → gitignored dirs. Manifests committed.

---

## Phase: Environment Bootstrap  *(runs first)*

**You are authorized to install the toolchain below without asking each time.**
Report what you install. Ask before anything *not* on this list, anything that
costs money, and anything that changes BIOS or system settings.

1. Run `.\scripts\preflight.ps1` (shipped in the seed). It checks platform, driver,
   toolchain, path sanity, ROCm/PyTorch on gfx1151, hipBLASLt config, and probes the
   maximum GPU allocation. It writes `ENVIRONMENT.md` with a remediation list.
2. **Remediate every failure.** Pre-authorized installs:
   - `git`, `gh` CLI, Python 3.12, `uv` — via `winget` or pip
   - a clean venv at a **space-free path** (paths with spaces are known-broken on
     this stack), then AMD's gfx1151 nightlies:
     ```
     pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ "rocm[libraries,devel]"
     pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchvision torchaudio
     ```
   - `ruff`, `pytest`, and the rig's Python dependencies
   - optionally `huggingface_hub[cli]`
3. **Prompt me for credentials — do not attempt to supply them yourself:**
   - `gh auth login` (required in the Scaffold phase to create the remote). Stop and ask me
     to run it if unauthenticated.
   - `huggingface-cli login` — only if a gated download actually requires it. Say
     which model and why.
   - No other API keys are needed. If some tool asks for one, stop and ask me
     rather than working around it.
4. **Things I must do myself — tell me clearly and wait:** BIOS UMA FB Size
   (target 96GB), AMD Adrenalin driver updates, anything requiring a reboot.
5. Re-run preflight until it's green, then commit `ENVIRONMENT.md` in the Scaffold phase and
   record the max-allocation number in `ASSUMPTIONS.md` tagged `[M]` with the date.
   **That number is the ceiling for every long-context experiment in this project.**

If ROCm cannot be made to work on native Windows after a reasonable attempt, stop
and report — do not silently fall back to CPU-only and proceed as if nothing
happened. A CPU-only lab is a different project and I need to decide, not discover.

## Phase: Scaffold  →  tag `v0.1.0`

`git init`, then:

```
research/reference/     # cloned upstream (gitignored except *.md)
research/notes/         # survey output
research/memory/        # THE memory-systems research track (its own directory)
curriculum/             # the learning output — the centerpiece
docs/                   # rig design docs; docs/adr/; docs/diagrams/ (*.mmd)
pyproject.toml          # uv WORKSPACE root — members declared here
packages/
  mnemosyne/            # memory subsystem — its own distributable package
    pyproject.toml      #   deps: torch ONLY. Never imports proteus or themis.
    src/mnemosyne/  tests/
  proteus/              # model architecture — its own package
    pyproject.toml      #   deps: torch, mnemosyne
    src/proteus/  tests/
  themis/               # ablation rig — its own package
    pyproject.toml      #   deps: torch, proteus, mnemosyne
    src/themis/{argus,data}/  tests/
tests/                  # cross-package integration tests only
configs/  tests/  scripts/  notebook/   # notebook/ = logged experiment journal
```

`README.md`, `ENVIRONMENT.md` (from the Environment Bootstrap phase), `ASSUMPTIONS.md` (the G2 assumption ledger — assumption → status → evidence → date; seed it with every assumption baked into this kickoff), `.gitignore` (Python, `research/reference/*` except `*.md`, `data/`, `checkpoints/`, `*.safetensors`, `*.gguf`, `runs/`), `CHANGELOG.md`. Verify `CLAUDE.md`, `OPERATING_INSTRUCTIONS.md`, `scripts/preflight.ps1`, and all 8 files in `.claude/agents/` are present; if any are absent, stop and tell me.

GitHub: `gh auth status` (pause if unauthenticated), then `gh repo create <repo-slug> --{{GITHUB_VISIBILITY}} --source=. --remote=origin --push`. Commit m0.

## Phase: Reference Library  →  tag `v0.2.0`

Write and run `scripts/fetch_reference.sh`:

**Architecture references (read these, don't run them):**
- `GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/poolside/Laguna-S-2.1` and `Laguna-XS-2.1` → configs, tokenizer, chat template, model card; no weights.
- `git clone --branch laguna https://github.com/poolsideai/llama.cpp` → real implementation of mixed SWA/global attention, sigmoid MoE gating, FP8 KV cache, DFlash speculative decoding.
- transformers `laguna` modeling source → the readable Python reference.
- **Comparison set — clone or fetch modeling code for the current hybrid/memory-notable architectures:** Mamba-2 and Mamba-3, Jamba, Samba, Zamba2, Nemotron-H / Nemotron-Nano-2, Qwen3-Next (Gated DeltaNet), Kimi Linear, MiniMax-01 (Lightning Attention), RWKV-7, xLSTM/mLSTM, Hymba (head-wise hybrid), GPT-OSS and Gemma 3 (SWA+global interleaving). Where weights are irrelevant, config + modeling file is enough.

**Training/infra references (fully open, actually runnable):**
- `allenai/OLMo` + `OLMo-core` + Dolma tooling; `pytorch/torchtitan`; `huggingface/smollm`; `karpathy/nanoGPT`; `EleutherAI/lm-evaluation-harness`.

**Hardware prior art (read before trusting any local result):**
- ROCm issue #6034 and the linked `bkpaine1/amdsense` repo — 93+ ML training
  experiments on this exact silicon, documenting five critical bf16 bugs, an
  undocumented ~19x AOTriton attention speedup, and MFU comparisons vs. an RTX 4090.
  Clone the repo and mine its reproduction scripts; they are directly reusable as
  our Hardware Validation harness.
- AMD TheRock nightly gfx1151 PyTorch wheels (Windows and Linux); AMD's ROCm-on-
  Ryzen WSL documentation; ROCm issue #6022 (librocdxg VRAM mapping under WSL2);
  the `andweng/wsl-rocm` community setup notes. Record exact versions in
  PROVENANCE.md — this stack moves weekly and an unversioned benchmark is worthless.

**Memory-systems references:**
- `vllm-project/vllm` (PagedAttention, prefix caching), `sgl-project/sglang`, `flashinfer-ai/flashinfer`, Mooncake (KV-centric serving), and a KV-cache-optimization paper index (e.g. `jjiantong/Awesome-KV-Cache-Optimization`).
- Agent-memory systems: `letta-ai/letta` (MemGPT lineage), A-MEM, and an agent-memory paper index (e.g. `Shichun-Liu/Agent-Memory-Paper-List`).

**Papers** → `research/reference/papers/` with a BibTeX file. At minimum, fetch the surveys that anchor each track (long-context survey, KV-cache-management survey, efficient-architectures survey, agent-memory survey, LLM-agent-memory-security survey) plus the primary papers for every technique named in Phase 2.

Then write `research/reference/CODE_MAP.md` — a guided tour with `file:line` pointers to: Laguna's attention layout and MoE gating; PagedAttention's block table; a prefix-cache hit path; Mamba-2's selective scan; Gated DeltaNet's update rule; a hybrid model's layer-interleaving config; OLMo's training loop, FSDP setup, dataloader, and checkpointer. **This map is what makes the curriculum's "read the code" exercises possible — it is a deliverable, not a byproduct.**

Commit (manifests, map, papers, script — not clones).

## Phase: Frontier Survey  →  tag `v0.3.0`

Parallel subagent dispatch. Each note: dense, sourced, `## Sources` section, ~3–5 pages. Land in `research/notes/`:

| File | Owner | Must answer |
|---|---|---|
| `transformer-state-of-the-art.md` | ml-architect | What a 2026 frontier decoder actually looks like end to end: norms, activations, attention variants (MHA→GQA→MQA→MLA), positional schemes, depth/width ratios, tokenizer choices. What changed since the 2023 vanilla recipe and *why* each change won |
| `moe-routing-and-failure-modes.md` | ml-architect | Routing (top-k, sigmoid vs softmax gating, shared experts), load balancing and its losses, expert granularity, capacity factors, dropped tokens, upcycling, the sparsity ratio tradeoff (Laguna ~14:1). Known failure modes: expert collapse, hot experts, training instability |
| `pretraining-recipes.md` | training-infra-engineer | Optimizers, LR/batch schedules, μP and hyperparameter transfer, scaling laws (Chinchilla → current), data mixing and curriculum, long-context extension stages, stability tricks |
| `posttraining-pipelines.md` | ml-architect | SFT → preference optimization → RLVR → agentic RL; reasoning/thinking-mode training; how "keep checking your work" behavior is actually induced |
| `inference-and-quantization.md` | training-infra-engineer | Quantization (FP8/INT4/NVFP4/GGUF), speculative decoding (incl. Laguna's DFlash), batching, the memory-bandwidth ceiling on decode |
| `evaluation-landscape.md` | research-lead | 2026 agentic + long-context eval landscape; what's cheaply reproducible at ablation scale; why needle-in-a-haystack is insufficient |
| `open-weights-landscape.md` | research-lead | Who ships what, openness tiers (weights-only vs full-stack), and where the genuinely open questions sit |

**And the memory track — `research/memory/`, owned by memory-systems-researcher. This is the priority deliverable of Phase 2; give it the most depth:**

| File | Must answer |
|---|---|
| `memory-taxonomy.md` | Fix the vocabulary before anything else. Five distinct things people call "memory": **parametric** (weights), **activation/recurrent state** (SSM hidden state), **KV cache** (attention's working set), **external retrieval** (RAG), **agent memory** (cross-session stores). For each: what it holds, its capacity and cost model, its read/write semantics, and what it fundamentally cannot do |
| `kv-cache-mechanics.md` | The shape math (per-token bytes = 2·layers·kv_heads·head_dim·dtype), why the KV cache dominates long-context memory and why decode is memory-bandwidth-bound not compute-bound; GQA/MQA/MLA as KV-cache-reduction strategies; FP8 KV quantization |
| `kv-compression-and-eviction.md` | Eviction and compression: H2O, SnapKV, PyramidKV, ChunkKV, KeyDiff, FastKV, L2-norm strategies, RocketKV, sparse attention at inference. What each actually assumes about token importance, and where those assumptions break |
| `kv-serving-hierarchy.md` | The systems layer: PagedAttention/vLLM block tables, cross-request prefix reuse, prefill/decode disaggregation, KV offload tiering, Mooncake-style KV-centric serving, CXL-pooled KV substrates. **Frame this explicitly as a memory-hierarchy problem — the founder has decades on exactly this** |
| `constant-state-memory.md` | SSMs and linear attention: Mamba-2/Mamba-3, Gated DeltaNet, RWKV-7, mLSTM/xLSTM, Lightning Attention. The core trade: constant state and linear time vs. degraded precise recall. Show the recall failure concretely (associative recall / multi-query recall) |
| `hybrid-architectures.md` | Inter-layer (Jamba, Samba, Zamba2, Nemotron-H, Qwen3-Next, Kimi Linear) vs intra-layer/head-wise (Hymba); SWA+global interleaving (Laguna 3:1, Gemma 3, GPT-OSS). Ratio selection, what each layer type is doing (recurrent structure vs precise retrieval vs factual recall), and how ratios are actually chosen — evidence or folklore? |
| `long-context-behavior.md` | RoPE, YaRN and rope scaling, NoPE in hybrids, length generalization, long-context extension training stages. **Effective vs advertised context** — why a 1M-token model is not a 1M-token-useful model |
| `agent-memory-systems.md` | Working / episodic / semantic / procedural memory; MemGPT-Letta lineage, A-MEM, memory-OS designs; context engineering and compaction. Include the common category error: treating working memory as a retrieval problem when it is a context-budget problem. Connect to tiered agent memory as built in practice |
| `memory-failure-register.md` | **The pain-point register — the single most important file in this phase.** Every known failure, each with: symptom, mechanism, evidence (citation), and whether it's open. Cover at minimum: lost-in-the-middle and position bias; effective-context collapse; KV eviction destroying long-range recall; SSM recall failures; hybrid ratio sensitivity; prefix-cache correctness hazards; memory poisoning and cross-session contamination; memory-induced sycophancy; forgetting/rollback being unsolved; and the measurement problem — that serving papers report latency/throughput without isolating which mechanism caused the gain |
| `open-problems-ranked.md` | Synthesis: rank the open problems by (a) how real the pain is, (b) whether progress is testable at 20M–300M params on our hardware, (c) whether systems/infrastructure expertise is a genuine edge. This ranking becomes the ablation backlog |

Then `research/synthesis.md` (≤3 pages), written to the G1 standard: **SCQA frame → answer first → three supporting arguments → evidence.** Include the MECE issue tree of the memory problem space, showing which two or three branches you're pursuing and explicitly naming what you're parking and why. Every finding ends in a "therefore." State what we believe, what's contested, what's folklore, and the 3–5 questions worth our compute — plus the single riskiest assumption remaining. **Present this to me before committing.** Commit and tag.

## Phase: Curriculum  →  tag `v0.4.0`

Owned by curriculum-author, reviewed by every specialist for accuracy. This is the deliverable the founder actually reads. Audience: expert systems engineer, ML beginner, learns by building and by reading source.

`curriculum/README.md` — the map: what you're about to learn, in what order, why, and the honest prerequisite list.

Modules named for their subject, not numbered — `attention-variants.md`, `moe-and-routing.md`, `kv-cache-mechanics.md`. `curriculum/README.md` gives the reading order and the six-track structure. Every module has the same shape: **theory in plain language → the math that actually matters (with each symbol translated) → why it matters for Proteus → read-the-code (file:line into `research/reference/`) → 2–3 hands-on exercises runnable on the Z13 (gfx1151, native Windows) with CPU fallback → self-check questions with answers at the end → what's still unsolved here.**

- **Track A — Foundations:** tensors/autograd; the transformer forward pass by hand; tokenization; the training loop; loss and optimization; scaling laws and the FLOPs budget (6·N·D).
- **Track B — Modern architecture:** attention variants and KV-cost implications; normalization and activations; positional encoding; MoE and routing; depth/width and initialization.
- **Track C — MEMORY (the deep track — the largest, mirroring `research/memory/` 1:1, one module per note).** Bridge every concept to systems knowledge the founder already owns — and then show where the analogy breaks: KV cache ↔ a working set with an eviction policy; paged attention ↔ virtual memory and page tables; prefix caching ↔ a shared read-only cache tier with invalidation hazards; KV offload tiering ↔ hot/warm/cold storage; SSM hidden state ↔ a fixed-size rolling aggregate vs. an unbounded log; agent memory ↔ a write-ahead store with schema drift and no compaction policy.
- **Track D — Training systems:** distributed strategies (FSDP/TP/PP/EP) mapped to sharding and replication; checkpointing as DR; determinism and resumption; training telemetry as an observability pipeline (his home turf — this module should be the one he *teaches back*).
- **Track E — Post-training and evaluation:** SFT/DPO/RLVR; building an eval you can trust at small scale; measuring memory and recall specifically.
- **Track F — Inference:** quantization, speculative decoding, serving; running Laguna XS 2.1 locally and reading its behavior against the architecture notes.

Plus: `glossary.md` (every term used anywhere in the repo), `reading-list.md` (papers ranked must/should/could, each with a one-line "read this for X"), `schedule.md` (12 weeks @ ~8 hrs/week, memory track weighted heaviest, honest difficulty ratings), and `capstones.md` — three: (1) implement a KV-cache eviction policy and measure where it breaks recall; (2) build a hybrid attention/SSM block and find the ratio cliff; (3) instrument a training run end to end with real telemetry and detect an injected fault.

Commit and tag.

## Phase: Rig Design  →  tag `v0.5.0`

tech-writer + ablation-engineer. The rig is a **controlled-comparison instrument**: its job is to make "does change X help?" answerable with evidence, cheaply and repeatably.

- `docs/experiment-unit-economics.md` — the **G3 gate for the rig**. Unit = one ablation run. Cost it: GPU-hours × verified current $/hr, storage, wall-clock. Model at 1 run / 20 runs (one sweep) / 200 runs (a research program), and identify explicitly **what breaks between those scales** — babysitting works at N=1, config management is required by N=20, artifact storage and result indexing and unattended failure recovery by N=200. Label every input `measured` / `benchmarked (source)` / `assumed`; any `assumed` input that flips the conclusion at ±30% becomes a measurement priority. Show the arithmetic.
- `docs/evidence-standard.md` — what an ablation must produce to count as evidence (fixed seeds, matched token budgets, matched param counts, confidence intervals over ≥3 seeds, pre-registered G2 hypothesis card).
- `docs/system-architecture.md` — component diagram showing the **four systems and their boundaries** (Chiron contains Proteus, Mnemosyne, Themis; who calls whom; what crosses each interface), experiment lifecycle sequence diagram, config→run→artifact→report flow. **Mnemosyne must be separable from Proteus** — the memory subsystem should be usable against a different model, because that separation is what makes it a research contribution rather than an implementation detail.
- `docs/proteus-config-space.md` — Proteus's configurable surface: attention type per layer (full/SWA/linear/SSM), layer interleaving ratio, MoE on/off and gating, positional scheme. **Every axis exists so it can be ablated.**
- `docs/mnemosyne-cache-interface.md` — the memory subsystem's public API, written as the package's contract with the outside world: how Proteus requests and releases cache, what policies plug in (eviction, compression, tiering), what telemetry every policy must emit for attribution, and the invariants a policy must not break. Design this as though someone else will implement a policy against it.
- `docs/corpus-and-probes.md` — small permissive corpus, plus targeted probes: associative recall, multi-query recall, long-range retrieval at varying depths, effective-context measurement.
- `docs/argus-telemetry-schema.md` — metrics schema (JSONL first), what's logged per step, how a run is diagnosed post-hoc. Designed by the observability veteran's standards.
- `docs/research-roadmap.md` — Mermaid gantt feeding ABLATION_LOOP_PROMPT.md.
- `docs/adr/<slug>.md` — one per decision, named for the decision: framework choice, config schema, eval probe set, ablation statistical standard.

Commit and tag.

## Phase: Ablation Backlog  →  tag `v0.6.0`

ablation-engineer + research-lead turn `research/memory/open-problems-ranked.md` into `BACKLOG.md`. Each entry is a **G2 hypothesis card** (HYPOTHESIS / FOR / BECAUSE / MEASURED BY / SUCCESS / KILL / COST / RISKIEST) plus experiment design and what it would change if it works. Ordered by information per dollar (G3). The riskiest assumption gets attacked first — not the easiest or most interesting. Reconcile `ASSUMPTIONS.md` against everything learned in the Reference Library through Rig Design phases before committing. Commit and tag.

## Phase: Hardware Validation Gate  →  tag `v0.7.0`  *(blocks all research runs)*

**This gate exists because the instrument is unproven.** `[C]` gfx1151 is Preview-only
in ROCm with documented bf16 bugs; a silently-wrong kernel invalidates every
downstream result, and the rig cannot detect it. Build `rig/validation/` and prove:

0. **Memory capacity ceiling.** Already probed in the Environment Bootstrap phase by `preflight.ps1`.
   Re-confirm it here under sustained load rather than a single allocation — hold
   the allocation for several minutes and confirm no thermal or paging collapse.
   This number gates every long-context experiment.
1. **Numerics** — bf16 vs fp32 reference for matmul, softmax, RMSNorm, and attention
   at the shapes we will actually use. Record max absolute and relative error.
   Any op that fails gets a documented workaround or is banned from the rig.
2. **Determinism** — same seed, same config, three runs, bit-identical loss curves.
3. **Checkpoint integrity** — save/load round-trips exactly; resume-at-step-N matches
   uninterrupted training.
4. **Known-good recipe** — train a nanoGPT-scale model to a published loss target and
   compare. Missing the target means the stack is wrong, not the recipe.
5. **Throughput baseline** — tokens/sec and MFU at our target config, so later
   regressions are visible.

Pin every working version (ROCm, PyTorch wheel, kernel, driver) into `ASSUMPTIONS.md`
tagged `[M]` with date. Write `docs/hardware-validation.md` documenting the gate
and the re-run trigger: **every ROCm or PyTorch upgrade re-opens this gate.**

Failing any check is not a blocker to report and stop — it is a finding. Document the
failure, pin around it, and note which experiments it constrains. Commit and tag.

## Phase: Report

Print the tree, tags, the curriculum's week-1 starting point, the top 5 ablations, total spend, and the five things you are least confident about in what you just wrote. Close in the required format: **Decision** (what to do next), **Riskiest assumption** (what would break this), **Next test** (the cheapest way to attack it). Confirm `ASSUMPTIONS.md` reflects the current state of every open question.
