# CLAUDE.md — Chiron

One-person model architecture research lab. **Primary research interest: memory
systems in LLMs.** Reference model under study: Poolside Laguna S 2.1
(118B-A8.5B MoE, mixed SWA/global attention, OpenMDW-1.1). Our experimental
architecture: Proteus.

Phase goal is understanding and evidence, not shipping a product. Radical design
changes come after the curriculum, not before.

## System names
Four named systems. A name is earned by having its own interface, its own lifecycle,
and being plausibly extractable. Modules failing that test stay unnamed.

| Name | System | Owns | Path |
|---|---|---|---|
| **Chiron** | the lab | repo, docs, curriculum — the umbrella. The centaur who tutored Achilles, Jason, and Asclepius: the archetypal teacher, and a *being* rather than a place, which keeps the naming register consistent with the other three. | `/` |
| **Proteus** | model architecture | decoder, attention variants, MoE, positional schemes; the full config surface | `proteus/` |
| **Mnemosyne** | memory subsystem | layered memory and its management: KV cache, eviction/compression policies, tiering, prefix reuse, attribution instrumentation. **The research contribution.** | `mnemosyne/` |
| **Themis** | ablation rig | pre-registration, matched budgets, seeds, execution, aggregation, reporting | `themis/` |

Reserved, unused until the thing grows: **Argus** (telemetry — inside Themis for now),
**Lethe** (eviction policy layer inside Mnemosyne).

**Boundary rule — enforced by the dependency graph, not by discipline:**

```
mnemosyne  →  torch                    (never imports proteus or themis)
proteus    →  torch, mnemosyne
themis     →  torch, proteus, mnemosyne
```

Mnemosyne must stay separable from Proteus. If memory management only works against
our specific model, it's an implementation detail; if it can be pointed at a
different model, it's a contribution. Because `packages/mnemosyne/pyproject.toml`
does not declare proteus, an accidental import fails at resolution rather than
surviving review. A lint contract bans the import path outright so the error is
legible. Do not "temporarily" add the dependency to unblock something — that is the
one shortcut that quietly destroys the result.

**Acceptance test for separability** (run at the mnemosyne-core milestone, not every commit):
build the mnemosyne wheel, install it into a clean venv containing only torch, and
run its test suite green. If that fails, the boundary has leaked.

Experiment arms are named from the model: `proteus-dense`, `proteus-swa-4to1`,
`proteus-moe-sigmoid`. Memory policies are named from the subsystem:
`mnemosyne-h2o`, `mnemosyne-snapkv`, `mnemosyne-window`.

## NAMING RULE (hard)

Every name must carry information. The only names permitted to be arbitrary are the
four **system names** — Chiron, Proteus, Mnemosyne, Themis — which are proper nouns
whose meaning is established once, in the table above.

Everything else — files, directories, modules, functions, tests, configs, experiment
arms, git tags, variables — must state what the thing **is** or **does**.

| Don't | Do |
|---|---|
| `docs/03b.md` | `docs/mnemosyne-cache-interface.md` |
| `utils.py`, `helpers.py`, `common.py` | `kv_eviction.py`, `rope_scaling.py` |
| `process()`, `handle()`, `do_work()` | `evict_lowest_attention_mass()` |
| `test_2()`, `test_cache()` | `test_eviction_preserves_recent_tokens()` |
| `config_v2.yaml`, `exp3.yaml` | `swa-4to1-ctx32k.yaml` |
| `m11.5`, `phase-2`, `0001-adr.md` | `mnemosyne-core`, `v0.13.0`, `adr/package-boundaries.md` |
| `arm_a` vs `arm_b` | `proteus-swa-4to1` vs `proteus-dense` |

**Do not encode ordering in identifiers. Anywhere. No exceptions.** A sequence
number is not information. It breaks on insertion — that is how you get an `03b`,
and how a milestone list acquires an `m11.5` — and it rots the moment anything is
reorganized. This applies to filenames, ADRs, notebook entries, milestones, phases,
and anything else that might tempt you to count.

Ordering lives in exactly one place per collection: the folder's `README.md`. One
file to edit instead of fifteen to renumber, and inserting something in the middle
costs one line.

**Versioning uses semver, in metadata or in git — never in a filename.**

- Documents that need versions carry `version: 1.2.0` in frontmatter. The filename
  stays stable so links never rot.
- Milestones are **named**, not numbered: `themis-skeleton`, `mnemosyne-core`,
  `attention-plugpoints`. The git tag carries the version and the annotation
  carries the name:
  ```
  git commit -m "milestone(mnemosyne-core): pluggable eviction with attribution telemetry"
  git tag -a v0.13.0 -m "mnemosyne-core: pluggable eviction with attribution telemetry"
  ```
  Semver sorts correctly forever and never needs a `.5`. Insert a milestone
  wherever you like — it just gets the next version when it lands.
- ADRs are `docs/adr/<slug>.md` and are cited by slug: "see ADR: package-boundaries".
  Chronology is what git log is for.
- Experiments are `notebook/<slug>.md`, named for the hypothesis
  (`eviction-vs-recall-depth.md`), indexed in `notebook/README.md`. If you re-run
  one with a changed design, bump `version:` in its frontmatter or write a new
  slug that says what changed — never `-v2` tacked onto a number.

A folder with more than three files gets a `README.md` listing what each one is and
the order to read them. If you cannot write a one-line description of a file that
distinguishes it from its siblings, the file is misnamed or shouldn't exist.

## DOCUMENT CLASSES (read this before the ADR rule)

Three classes. Which class a file belongs to decides whether you may edit it. Most
files are in the third class and are freely editable — the immutability rule has a
small, clearly bounded scope, and knowing the boundary is what keeps it cheap.

**1. Records — immutable. Never edited, never deleted.**
Capture what was true or believed *at a moment*. Their value is that they are
untouched.
- `docs/adr/<slug>.md` once `Accepted`
- `notebook/<slug>.md` — the hypothesis card and design freeze *before* the run;
  the results section is written once *after* and then freezes too

Corrections to a record are **appended, never applied**:
```
Correction appended 2026-09-14: Consequences states a 4:1 ratio; the arm actually
run was 8:1. The conclusion is unaffected.
```
The error and the correction both survive. That is the point.

**2. Registers — append and update status. Rows are never deleted.**
Track the current state of a set of things.
- `ASSUMPTIONS.md` — assumption → status → evidence → date; status transitions,
  rows never disappear
- `docs/adr/README.md` — slug, status, date, superseding slug, body hash
- `notebook/README.md` — every experiment and its outcome, including the failures
- `LOG.md` — append-only

**3. Documentation — mutable, and you are obligated to keep it accurate.**
Describes what is *currently* true. Everything not named above: `README.md`,
`curriculum/`, `docs/` (except `adr/`), `research/` notes, `ENVIRONMENT.md`,
interface specs, code comments.

**So: spelling and grammar.** Fix them freely and immediately in class 3 — a stale
or sloppy document is a defect, and "leave every file better than you found it"
applies in full. Do not fix them in class 1. Not because a typo matters, but because
"is this just a typo?" is a question you would have to answer every single time, at
the exact moment you least want friction. Immutable-with-no-exceptions costs nothing
to enforce; immutable-except-for-trivia costs an adjudication forever and erodes one
reasonable case at a time. The upside of fixing a misspelling in a one-page document
that is read rarely and whose meaning is unaffected is approximately zero. Leave it.

If an error in a record is bad enough to mislead a reader, that is not a typo — it
is grounds for an appended correction, or for a superseding ADR.

## ADR IMMUTABILITY (hard rule)

**An accepted ADR is a permanent record. It is never modified, never renamed, never
deleted.** If the decision changes, you write a new ADR that supersedes it. The old
one stays exactly as written, wrong reasoning and all — the wrongness is the
valuable part, because it records what you believed and why at the time you had to
choose.

`docs/adr/<slug>.md`, one page:

```
# <Decision stated as a sentence>

Status:   Accepted            (see status ladder below)
Date:     2026-07-24
Deciders: <who>

## Context
What forced a decision. Constraints, pressures, what was true at the time.

## Decision
What we chose, stated actively: "We will ..."

## Consequences
What this makes easy, what it makes hard, what it forecloses.
```

**Status ladder.** An ADR is editable *only* while `Proposed`. The moment it reads
`Accepted`, the body is frozen:

| Status | Meaning | Body editable? |
|---|---|---|
| `Proposed` | draft, under discussion | yes |
| `Accepted` | in force | **no — frozen** |
| `Superseded by <slug>` | replaced; read the successor | **no** |
| `Deprecated` | no longer applies, nothing replaced it | **no** |
| `Rejected` | considered and declined | **no** |

Keep `Rejected` ADRs. Knowing why you *didn't* do something is usually worth more
later than knowing why you did, and it's the first thing people delete.

**The one permitted write, and it is an append, not an edit.** When a decision is
superseded, append a single line to the Status block of the old ADR:

```
Status:   Accepted
          Superseded by mnemosyne-cache-ownership on 2026-09-14
```

Nothing above it is altered. No existing character is changed or removed. Context,
Decision, and Consequences are untouched forever.

**Not permitted, no matter how tempting:** fixing a typo, updating a stale link,
"clarifying" a sentence, correcting a number you later learned was wrong, tidying
formatting, or renaming the file because a better slug occurred to you. See
DOCUMENT CLASSES above for why spelling and grammar get no carveout here while
being freely fixable everywhere else. If an accepted ADR contains something
materially misleading, append a correction line or write a superseding ADR — never
an edit.

**Enforcement, because discipline alone is not a control.** `docs/adr/README.md` is
the register: every ADR's slug, status, date, superseding slug where applicable, and
a SHA-256 of the frozen portion (everything below the Status block). A test in
`tests/` recomputes those hashes and fails on mismatch. You cannot prevent someone
editing a file; you can make the edit impossible to commit unnoticed. That is the
bar.

## ENVIRONMENT
Native Windows on gfx1151. `scripts/preflight.ps1` checks the whole stack and writes
`ENVIRONMENT.md`. Re-run it after any ROCm, PyTorch, or driver change — an upgrade is
a change of instrument. The max-GPU-allocation number it reports is the hard ceiling
for long-context work and lives in `ASSUMPTIONS.md` tagged `[M]`. Installing the
known toolchain is pre-authorized; credentials are never yours to supply — prompt the
user for `gh auth login` and any HF login, and stop rather than working around a
missing key.

## GOVERNING DOCUMENT
`OPERATING_INSTRUCTIONS.md` governs this repo and **wins over anything in this
file** on conflict. Read it before your first substantive action, and re-read its
RESEARCH-LAB INTERPRETATION section — G0 (Discovery) and G3 (Unit Economics) are
translated for a research context, not suspended. The operating loop is
G0 → G1 → G2 → G3 → G4 → G5, in order, each with a written checkable artifact.
Begin substantive responses with a gate marker (e.g. `[G2 — Hypothesis]`), answer
first, and close recommendations with Decision / Riskiest assumption / Next test.
Gates may be skipped deliberately via the override protocol, never accidentally —
log every skip.

Repo lives under `c:\projects\school\`.

## Repo map
- `research/reference/` — upstream clones (gitignored except `*.md`). Rebuild via
  `scripts/fetch_reference.sh`. `CODE_MAP.md` is the guided tour; `PROVENANCE.md`
  the ledger.
- `research/notes/` — frontier survey. `research/memory/` — the memory track
  (taxonomy → KV mechanics → compression → serving → constant-state → hybrids →
  long-context → agent memory → failure modes → open problems).
- `curriculum/` — the learning output. Stays live: new findings get folded back in.
- `docs/` + `docs/adr/` + `docs/diagrams/` — rig design.
- `packages/` — a **uv workspace** with three separately-distributable packages:
  `mnemosyne/` (memory subsystem), `proteus/` (model), `themis/` (rig, containing
  `argus/` telemetry and `data/` loaders as modules). Each has its own
  `pyproject.toml`, dependency list, version, and `tests/`. Root `tests/` holds
  cross-package integration tests only. One shared venv via `uv sync`.
- `notebook/` — pre-registered experiments and results. `INDEX.md` lists all.
- `BACKLOG.md`, `LOG.md`, `BLOCKERS.md`, `CHANGELOG.md` — loop state.

## Hard rules
1. Milestone commits only: `milestone(<name>): <summary>` or `experiment(<slug>): <finding>`,
   annotated semver tag for milestones. Clean tree between iterations. No history rewrites.
2. Every material claim carries a G5 tag: `[M]` measured (run ID + seed count),
   `[C]` cited (arXiv ID or URL, with date), `[A]` assumed (state confidence and
   the cheapest test that would move it). Never state an `[A]` in the register of
   an `[M]`. No invented numbers or benchmark results — unverifiable content is cut.
3. Training data is stale relative to today. Search for last-6-months work before
   writing any survey section. Contested topics get presented as contested.
4. Provenance row for everything fetched. Upstream LICENSEs stay intact.
5. No weights/datasets/checkpoints in git; nothing >20MB.
6. Ask before: spending money, cloud launches, public pushes, deletions.
7. ADR before diverging from a decided design: `docs/adr/<slug>.md` —
   Status / Context / Decision / Consequences.

## Experimental standards
- Pre-register before running, using the **G2 hypothesis card verbatim**
  (HYPOTHESIS / FOR / BECAUSE / MEASURED BY / SUCCESS / KILL / COST / RISKIEST).
  Committed *before* the run. No post-hoc hypothesis fitting. Moving a SUCCESS or
  KILL threshold after seeing results is a change of standard and must be called
  out as one.
- Attack the riskiest assumption first, not the easiest. A hypothesis that cannot
  fail is not a hypothesis — rewrite it.
- Maintain `ASSUMPTIONS.md` as a running ledger: assumption → status (untested /
  supported / refuted) → evidence → date. Update it at every Definition of Done.
- WIP limit: one hypothesis in flight unless there's an explicit reason otherwise.
- Timebox before starting; when the box expires, report and re-decide rather than
  silently continuing.
- Matched param counts and token budgets across arms. ≥3 seeds. Report confidence
  intervals. Single-seed numbers are anecdotes and must be labeled as such.
- A falsified hypothesis is a successful experiment. Write it up with equal care.
- Instrument for *attribution*, not just outcome — the literature's weak spot is
  reporting that something helped without isolating which mechanism did it.
- If a result looks too good, suspect the harness first.

## Engineering conventions
- **G4 TDD is non-negotiable for rig code** — red, green, refactor, one failing
  test first, never behavior and structure in the same step. Test names read as
  specifications. Bug fixes start with a failing regression test. One-off analysis
  scripts under `notebook/` are exempt *only* if reproducible from committed config
  and data hashes; on reuse they migrate into the rig and acquire tests.
- Python 3.11+, `uv`, `ruff`, `pytest`, PyTorch. Type hints on public functions.
- Config via YAML in `configs/`, one typed config object. Every ablation axis is a
  config field — the config surface IS the experimental surface.
- Determinism first: seeded runs, resumable dataloaders, checkpoint round-trip tests.
- Metrics as JSONL first, dashboards later. Loss/LR/throughput/cache-stats scrapeable.
- Readable beats clever: this code is read by someone learning from it.
- YAGNI over speculative generality. No abstraction for a requirement that does not
  yet exist. Flag technical debt when taken, with cost and repayment trigger.

## Hardware — Z13 is PRIMARY (decided)

**Platform:** ASUS ROG Flow Z13, Ryzen AI Max+ 395 (Strix Halo), Radeon 8060S iGPU
(**gfx1151**), 128GB unified memory. This is the primary machine for research,
training, and inference. The RX 7900 XT box is secondary/optional.

**Why this is defensible and not a compromise:** memory-systems research is
capacity-bound and bandwidth-bound, not FLOPS-bound. 128GB unified memory holds KV
caches and long-context experiments that will not fit on a 20GB discrete card, and
the platform's low bandwidth relative to compute *magnifies* the exact decode
bottleneck under study. `[C]` A published 93-experiment campaign on this same
silicon measured 25% MFU vs. 7.7% on an RTX 4090 for the same recipe, attributing
the gap to unified memory advantages on memory-bound workloads (ROCm issue #6034,
Mar 2026). Accept slower wall-clock; you are buying capacity and a clearer signal.

**Platform decision: NATIVE WINDOWS is the primary target. WSL2 is the fallback.**
This inverts normal ML advice and is deliberate.

1. **Native Windows works and is officially exercised.** `[C]` AMD ships ROCm
   nightlies for gfx1151 with PyTorch on Windows
   (`pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch`),
   and PyTorch runs downstream CI on Strix Halo specifically. Native Windows sees
   the full UMA pool set in BIOS. **Use this path first.**
2. **WSL2 is now officially supported for this SKU — but has a capacity-killing
   bug.** `[C]` AMD's ROCm-on-Ryzen docs mark the first official Strix/Strix Halo
   WSL support via ROCDXG over Microsoft's DXCore (`/dev/dxg`). However, ROCm issue
   #6022 documents that `librocdxg` fails to map Dedicated VRAM under WSL2: the
   ROCm pool size is clamped to the `.wslconfig` memory setting, and allocations
   near that limit trigger host paging and hangs **while the 96GB dedicated pool
   sits idle**. `rocminfo` also reports `IOMMU Support: None` under WSL2 despite
   BIOS settings. **This directly negates the reason we chose this machine.**
3. **Single-device only.** `[C]` Distributed collectives (`torch._C._distributed_c10d`)
   are incomplete on gfx1151. FSDP/DDP paths are DESIGN-ONLY until validated on
   rented hardware. Any milestone requiring them is blocked, not attempted.
4. **Numerics are unproven until you prove them.** `[C]` Five critical bf16 bugs
   are documented on gfx1151. Correctness is an open assumption. See the gate below.

**The decisive early test (do this before committing to a path):** set BIOS UMA FB
Size to 96GB, then attempt a GPU allocation larger than the WSL2 `.wslconfig` memory
limit. If it reaches the dedicated pool, WSL2 is viable. If it pages, hangs, or
fails while dedicated memory sits idle, WSL2 is capped at the `.wslconfig` value and
**native Windows is mandatory for any long-context or KV-capacity experiment.**
Record the measured ceiling in `ASSUMPTIONS.md` as `[M]` with date. Community
reference for the WSL path: the `andweng/wsl-rocm` setup notes — including that
Ubuntu 24.04's stock ROCm 5.7 packages shadow ROCm 7.2 and must be purged, or
everything fails with "ROCk module is NOT loaded."

**Measured platform reference points** `[C]` (from ROCm #6022, verify on our unit):
~33 TFLOPS GEMM at 8192³, ~172 GB/s actual memory bandwidth. Note also that
misconfigured hipBLASLt can drop GEMM to ~6 TFLOPS — a 5x penalty from a library
path, so validate `HIPBLASLT_TENSILE_LIBPATH` and `TORCH_BLAS_PREFER_HIPBLASLT`
before benchmarking anything.

**Known-broken / needs workaround:** `torchao` and `bitsandbytes` crash on import;
Flash Attention 2 unavailable (xFormers fallback); Unsloth and Swift need patching.
Prefer AMD's **TheRock nightly** wheels with native gfx1151 binaries — pip-packaged
ROCm works without a system `/opt/rocm` install. Env: `HSA_OVERRIDE_GFX_VERSION=11.5.1`,
`HCC_AMDGPU_TARGET=gfx1151`. Allocate 96–110GB to the iGPU in BIOS.
An undocumented AOTriton attention speedup (~19x) was reported in the same issue —
verify whether it applies before optimizing anything else.

**HARDWARE VALIDATION GATE (the Hardware Validation Gate — blocks all research runs).**
No experimental result from this machine is trustworthy until the stack is proven.
Before any rig work: establish the **memory capacity ceiling** (largest allocation that
actually reaches the UMA pool, on whichever platform you chose); verify bf16
numerics against fp32 reference on the ops we depend on (matmul, softmax,
RMSNorm, attention); confirm determinism across repeated runs with a fixed seed;
confirm checkpoint save/load round-trips bit-exactly; validate hipBLASLt is
configured (GEMM throughput sane, not 5x low); and run a known-good tiny training
recipe (nanoGPT-scale) to a published loss target.
If any check fails, pin the working configuration and record it in `ASSUMPTIONS.md`
as `[M]` with the exact wheel/driver versions. Re-run this gate after every ROCm or
PyTorch upgrade — an upgrade is a change of instrument, not a maintenance task.

**Scale:** ablations at 20M–300M params, 0.5–5B tokens. CPU fallback config required
for every training config. Anything needing multi-GPU or exceeding the wall-clock
budget gets costed and rented, gated on approval.

## Founder context
30 years systems/infrastructure — distributed systems, storage hierarchies, caching,
Kubernetes, DR, enterprise observability. Fluent with agentic tooling and tiered
agent-memory design. New to ML internals. Teach by bridging from systems concepts,
then show where the analogy breaks. Do not condescend and do not hand-wave math.

## Subagents
`.claude/agents/`: research-lead, ml-architect, **memory-systems-researcher**,
data-engineer, training-infra-engineer, ablation-engineer, tech-writer,
curriculum-author. Delegate research and writing; run independent tasks in parallel.
