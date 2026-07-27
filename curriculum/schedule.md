---
title: Schedule — twelve weeks at eight hours, what that actually buys, and what it does not
version: 1.0.0
date: 2026-07-26
owner: curriculum-author
status: proposed — the hour budget is [A] and should be replaced with [M] after Week 3
covers: all 31 curriculum modules, sequenced by the union of the prereq graph in
  curriculum/README.md and each module's own `prereqs:` frontmatter
---

# Schedule

**A note on the week labels.** The house rule bans ordering in *identifiers*. These are
not identifiers — there is no `week-1.md`, and nothing links to "Week 7". A schedule is a
document whose subject *is* ordering, and the ordering lives here, in tables, which is
exactly where the rule says to put it. Each week additionally carries a descriptive title
so that moving one costs a line, not a renumber.

---

## 1. Answer first

**Twelve weeks at eight hours a week is not enough to finish this curriculum, and it is
not close.** The 31 modules add up to `[A]` **~188 hours** of reading and exercises;
the survey behind them adds `[A]` **~24 hours** more. Ninety-six hours covers **51%** of
the modules, or **45%** of modules-plus-survey.

So the twelve weeks are a **selection**, not a pass. This schedule spends them on:

- the minimum Track A/B spine required to read Track C honestly — trimmed hard, with every
  cut named in §5;
- **all ten Track C memory modules**, which is the point of the lab;
- the six curriculum exercises that are Hardware Validation Gate items wearing homework
  clothing — front-loaded, because until they run, every measured number downstream is
  provisional;

and it **ends** with `measuring-memory.md` Exercise B — the full-cache oracle-diff
instrument with a measured noise floor. That is the deliverable `research/synthesis.md`
names, so the twelve weeks terminate on the thing the lab exists to build rather than on
a module boundary.

Tracks D, E and F — 13 modules, `[A]` ~95 hours — are weeks 13–24. §10 schedules them.

**Realistic completion.** `[A]` low-to-medium confidence: plans of this shape slip 20–30%
for someone with a demanding day job, so the twelve-week block lands in **14–16 calendar
weeks**, and the full 31 modules land at **~30 calendar weeks (~7 months)** at 8 h/wk. The
cheapest test is to log actual hours for Weeks 1–3 against the numbers in §8 and re-plan
at the Week 3 checkpoint. Do that; do not treat the slip factor as decoration.

---

## 2. The arithmetic, so you can check it

Every number below is reproducible with `wc -w`. No number here is a measurement of you.

### 2.1 Volume

| Corpus | Command | Words |
|---|---|---|
| Curriculum modules (31) | `ls curriculum/*.md \| grep -vE 'README\|schedule\|capstones\|glossary\|reading-list' \| xargs wc -w` | **373,999** |
| Survey notes (17, excluding the two `README.md`) | `wc -w research/memory/*.md research/notes/*.md` minus 669 + 532 | **95,089** |
| `research/synthesis.md` | — | 1,590 |
| **Total, the part you must read in order** | | **470,678** |
| Companions — `capstones.md`, `glossary.md`, `reading-list.md` | landed 2026-07-26, mid-authoring | ~37,700 |

The companions are **reference, not sequence**. `glossary.md` and `reading-list.md` are
consulted, never read front-to-back, so they carry no hours in this plan. `capstones.md` is
scheduled in §10 and carries its own calendar estimates, which are in weeks, not hours.
Counts are as of **2026-07-26** and the module set is still being revised; re-run the command
before quoting a figure.

The task brief said ~330,000 and ~95,000. The survey figure is right to three digits. The
curriculum figure is **13% higher than briefed** — 374k, not 330k. Track C alone is
125,704 words across ten modules, averaging 12,570; the longest module in the repo is
`quantization.md` at 16,822 and it is in Track F. `[A]` The 330k figure predates the
D/E/F modules landing.

`[M]` 6.1% of module words sit inside fenced code blocks (measured by toggling on
``` fences and word-counting each side). So this is prose, not a code dump, and it must be
read at prose speed.

### 2.2 The reading rate, derived rather than asserted

I am not going to assert a words-per-minute figure. **Twenty-six of the 31 modules carry
their author's own reading estimate in frontmatter.** Those 26 modules total **325,291
words** and **81.5 stated hours** (range midpoints; e.g. "3–4 h" counts as 3.5).

```
325,291 words / 81.5 hours = 3,991 words/hour = 66.5 words/minute
```

**66.5 words per minute.** That is roughly a third of ordinary technical-prose speed, and
it is the correct rate for material you work with a pen: `kv-cache-mechanics.md` asks you
to hold three separate byte budgets apart while dividing, and `measuring-memory.md` asks
you to hold a null distribution in your head while reading a policy comparison. If you
find yourself reading these at 200 wpm you are skimming, and the self-check questions will
tell you so.

The five modules with no stated reading time — `long-context-and-effective-context`,
`loss-and-optimization`, `moe-and-routing`, `positional-encoding`, `tokenization`,
totalling 48,708 words — are budgeted at the same 3,991 w/h. That is the *only* place I
have extrapolated the rate rather than used an author's own number.

### 2.3 The total

| Line | Hours | How |
|---|---|---|
| Curriculum reading | **93.7** | 373,999 / 3,991 |
| Curriculum exercises | **94.4** | 74.1 h stated across 26 modules + 20.3 h estimated for the five without |
| **Curriculum subtotal** | **188.1** | |
| Survey reading | 23.8 | 95,089 / 3,991 |
| **Total** | **211.9** | |

| Cadence | Curriculum only | With survey |
|---|---|---|
| 8 h/wk | **23.5 weeks** | 26.5 weeks |
| 12 h/wk | 15.7 weeks | 17.7 weeks |
| 18 h/wk | 10.5 weeks | 11.8 weeks |

**12 weeks at 8 h/wk = 96 h.** The only way "12 weeks" and "all 31 modules" are both true
is at ~18 h/wk, which is a second job. This schedule does not pretend otherwise.

Note the split the exercise column implies: **exercises are 50% of the total effort.** For
someone who learns by building, that is the right ratio, but it is not what a page count
suggests, and it is why "I'll just read ahead" does not work here.

---

## 3. Standing hazards — read once, then keep them in view

These are properties of the instrument, not of any module. Every week inherits them.

**A single tensor ≥32 GiB hard-hangs this machine.** `[M]` 2026-07-26
(`ASSUMPTIONS.md → large-tensor-fault-32gib`): 31 GiB copies cleanly at 199.9 GB/s; 32 GiB
hangs for 11 minutes at 0% CPU with host free RAM down to 5 GB and requires a force-kill;
36 GiB raises `hipErrorLaunchFailure`. **Cap any single allocation at 24 GiB.** The failure
presents as *silence*, which is the worst possible mode for an unattended sweep — a run
that hangs at 0 CPU looks exactly like a run that is working.

**Default SDPA retains the score matrix, and that is how you reach the hang.** `[M]`
`ASSUMPTIONS.md → sdpa-is-memory-efficient`: **147.2 bytes/T² retained** by default versus
**6.6** with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, measured at B=4, nh=8;
`flash_sdp_enabled()` returns `True` either way, so the API cannot tell you which one ran.
Arithmetic on that measured coefficient — arithmetic, not a new measurement — puts the
32 GiB fault at **T ≈ 15,300 tokens at that batch/head shape**:

```
32 GiB = 34,359,738,368 bytes;  34,359,738,368 / 147.2 = 2.334e8;  sqrt = 15,278
```

So: any attention exercise past ~15k tokens on defaults walks into a silent hang. The flag
fixes it and is **experimental, therefore a numerics change** — see ADR:
`aotriton-attention-stays-off-by-default` (Accepted), which decides that the 18× win does
not justify turning it on before the Gate runs numerics both ways. **Set it per-exercise,
never in your profile**, and record which arm produced each number.

**32 GB of system RAM after the carve-out.** `[M]` `ENVIRONMENT.md` 2026-07-26. The BIOS
UMA FB Size is 96 GB, which leaves 32 GB to the host. **Every CPU fallback must stay under
~8 GiB.** The CPU fallback is not "the same thing, slower" on this machine; it is a
different, smaller machine.

**Budget long-context work against 62 GiB, not against the pool.** `[M]`
`ASSUMPTIONS.md → gpu-fast-tier-size`: flat ~200 GB/s out to **≥62 GiB** with the 96 GB
carve-out (single run per arm — an anecdote by the house standard, and the upper edge is
unmeasured because the sweep hit the 32 GiB fault, not a bandwidth knee).

**Record three environment variables with every number you write down.**
`HIPBLASLT_TENSILE_LIBPATH`, `TORCH_BLAS_PREFER_HIPBLASLT`,
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL`. `[M]`
`ASSUMPTIONS.md → hipblaslt-config`: configuring hipBLASLt is worth only +12% GEMM
throughput but **2.8× in long-reduction accuracy** (relative error of a 1M-element bf16
weighted sum: 2.01e-3 configured vs 5.60e-3 unset, 3 seeds). A long-context result taken
without it is confounded by arithmetic, not merely slow. `. .\scripts\activate-lab.ps1`
sets the first two.

**Distributed is absent, not incomplete.** `[M]`
`ASSUMPTIONS.md → single-device-only`: `torch.distributed.is_available()` is `False` and
FSDP fails at *import*. This is good news for a schedule — `distributed-training-strategies.md`
cannot half-run and produce a plausible number.

---

## 4. The Hardware Validation Gate lane

`ASSUMPTIONS.md → bf16-numerics-unproven` is **untested**. Per CLAUDE.md, no experimental
result from this machine is trustworthy until the Gate closes. That would make every
exercise in this curriculum worthless, which is obviously wrong — so the useful move is to
classify, not to block.

### 4.1 Three tiers, applied to every exercise in §8

| Tier | Definition | What the Gate does to it |
|---|---|---|
| **Tier 0 — Gate-independent** | Pure Python, numpy, exact arithmetic, or fp32/fp64 on CPU. The conclusion is a property of the algebra. | Nothing. These are true today. `[M]` fp32 gradients on gfx1151 match CPU to 3.9e-8 absolute (`curriculum/README.md`, from an exercise), so fp32 CPU-vs-GPU agreement is itself evidenced. |
| **Tier 1 — Gate-provisional (timing)** | GPU wall-clock, bandwidth, allocator behaviour, residency in bytes. bf16 *correctness* is irrelevant; library configuration is not. | Numbers stand, conditional on the three env vars being recorded. Re-run after any ROCm/PyTorch upgrade — an upgrade is a change of instrument. |
| **Tier 2 — Gate-blocked (numerics)** | Any number whose meaning depends on bf16 arithmetic being right: losses, KL divergences, attention-mass rankings, eviction-damage magnitudes, recall scores. | **Provisional.** Record them, label them provisional, and re-run after `bf16-numerics-unproven` moves to `supported`. Do not put a Tier 2 number in `ASSUMPTIONS.md` as `[M]` without that label. |

### 4.2 Which curriculum exercises are Gate items in disguise

This is the finding that shaped the sequence. CLAUDE.md lists six Gate items;
`research/synthesis.md` argues three more should be added. **Nine of those maps onto an
exercise someone already wrote.** Running the schedule in order closes most of the Gate as
a by-product.

| Gate item | Source | Closed by | Week |
|---|---|---|:--:|
| bf16 vs fp32 on matmul / softmax / RMSNorm / attention | CLAUDE.md | `transformer-forward-pass-by-hand.md` — "the bf16 error budget, stage by stage, on gfx1151" | **2** |
| Known-good tiny recipe to a published loss target | CLAUDE.md | nanoGPT `shakespeare_char` to val loss 1.4697 (`training/nanogpt/README.md:51`) | **2–3** |
| Determinism across repeated runs, fixed seed | CLAUDE.md | `the-training-loop.md` Exercise C — "Precision and repeatability on this specific machine" (its own frontmatter says it "feeds two open rows in `ASSUMPTIONS.md`") | **3** |
| hipBLASLt configured, GEMM sane | CLAUDE.md | already `[M]` — `scripts/benchmark_gemm.py`, 20.9 TFLOPS bf16 at 8192³ | done |
| Memory capacity ceiling | CLAUDE.md | already `[M]` at ≥74.40 GiB written/read/released; fast tier ≥62 GiB | done |
| **Attention-kernel roofline, not a GEMM one** | synthesis §"widen the gate" | `attention-variants-and-kv-cost.md` Exercise B — decode arithmetic intensity against `AI = 2G/b` | **4** |
| **RoPE at long position, in bf16** | synthesis §"widen the gate" | `positional-encoding.md` — "measure where bf16 breaks RoPE's relative-position identity" | **10** |
| Checkpoint save/load round-trips bit-exactly | CLAUDE.md | `checkpointing-and-resumption.md` Exercise A | 13 (deferred) |
| **fp32 discriminator for the 32 GiB fault** | synthesis §"widen the gate" | *no module owns this* — see §11 | — |

Two consequences worth naming. First, **the Gate is mostly closed by the end of Week 4**
if you run the schedule as written — which is why Weeks 1–4 are not "get the boring
foundations out of the way," they are the instrument-calibration weeks. Second, the
32 GiB fp32 discriminator (allocate 32 GiB as fp32 — same bytes, half the elements; failure
means bytes, success means element count) is a two-minute test that **no module owns**.
It should be a `notebook/` entry under the G0-LIGHT exception, not a curriculum exercise.

### 4.3 One more thing the Gate cannot fix

`ASSUMPTIONS.md → gemm-throughput-below-reference` is **refuted**: 20.9 TFLOPS is 63% of
the cited figure for this silicon, unexplained. And `building-an-eval-you-can-trust.md`
§3.6 builds every wall-clock estimate in this lab on an `[A]` assumption that an
end-to-end step reaches 35% of that — **7.3 TFLOP/s**, never measured. Week 4 runs
`scaling-laws-and-flops-budget.md`'s throughput exercise specifically to replace it. Until
then, treat §3.6's table (4.1 h/seed at 30M/0.6B; 68.5 h/seed at 300M/1B; **51 days** for a
6-arm × 3-seed design at 300M) as directionally right and numerically unsettled.

---

## 5. What is in the twelve weeks, and every cut, named

**In, complete:** `tensors-and-autograd`, `transformer-forward-pass-by-hand`,
`the-training-loop`, `attention-variants-and-kv-cost`, and all of Track C except as noted —
`memory-taxonomy-for-engineers`, `agent-memory-in-practice`, `kv-cache-mechanics`,
`kv-eviction-policies`, `paged-attention-and-prefix-reuse`, `constant-state-memory`,
`hybrid-attention-and-ratios`, `memory-failure-modes`.

**In, partial — and here is exactly what is cut:**

| Module | What you do | What is cut, and why |
|---|---|---|
| `tokenization` | Read the vocabulary-economics, special-tokens and failure sections; run "the exchange rate". | "Write the encoder" (4/5, 90 min) and "what the vocabulary costs". You need token-denomination to read the recall modules; you do not need to have written BPE. Restore in Week 15 before Track E. |
| `loss-and-optimization` | Read cross-entropy, perplexity/bits-per-byte, and the logits-allocation section. | All three exercises and the AdamW/schedule sections. You need to know what a perplexity number means before `long-context`; you do not need to have weighed the optimizer yet. |
| `scaling-laws-and-flops-budget` | Read the 6·N·D derivation and the MFU section; run "measure sustained throughput and replace the `[A]`". | "Reconcile three FLOP counts" and the IsoFLOP-fitting-bias exercise. The throughput number is load-bearing today (§4.3); the other two are not. |
| `positional-encoding` | Read all of it; run the bf16-RoPE exercise and the Laguna YaRN-ladder reconstruction. | The length-generalization sweep (`[A]` 2–5 h GPU, 4 arms × ≥3 seeds). That is a pre-registered experiment, not homework. |
| `long-context-and-effective-context` | Read all of it; run the dilution ledger and "does the needle survive the accumulator". | The scaled-RULER grid (`[A]` 3–6 h GPU). Same reason — it wants a G2 hypothesis card and a `notebook/` slug. |
| `measuring-memory` | Read all of it; **build the oracle-diff instrument and measure its noise floor** (Exercise B). | Exercise A (probe cost / hidden syncs) and Exercise C (fault-inject a retrieval metric). Exercise C is the six-fault calibration battery from `synthesis.md` question 3 — again, `notebook/`, not homework. |

**Deliberately deferred whole modules:** `normalization-and-activations`, `moe-and-routing`,
`depth-width-and-initialization` (Track B); all of D, E, F. The Track B three are genuinely
useful and genuinely not on the critical path to Track C — none of the ten memory modules
lists any of them as a prereq.

**Two exercise duplications found while sequencing, and resolved by cutting one of each:**

- `tensors-and-autograd` — "prove the gradient-accumulation identity, then break it" (45 min)
  and `the-training-loop` Exercise B — "Break gradient accumulation, then fix it" (30–45 min)
  are the same exercise. Keep the training-loop one; it is *in situ*, against a real step.
- `tensors-and-autograd` — "build an activation ledger and find the T² term" and
  `transformer-forward-pass-by-hand` — "find the T² term, then watch it disappear" overlap
  substantially. Keep the tensors one (it builds the ledger you reuse); the forward-pass
  version's payoff — watching the term vanish under the AOTriton flag — is re-derived in
  Week 6 against the real cache anyway.

---

## 6. Coverage this plan achieves

| | Hours | Share |
|---|---|---|
| Scheduled content, Weeks 1–12 | **90.7** | |
| Nominal budget (12 × 8) | 96.0 | |
| Slack | **5.3** | 5.5% |
| — of which reading | 52.1 | 57% |
| — of which hands-on | 38.6 | 43% |

52.1 reading hours × 3,991 w/h = **~208,000 words**, i.e. **56% of the curriculum's words**
in 12 weeks. Cross-checks against 52.1 / 93.7 = 55.6%. Good.

Five and a half percent slack is **thin**, and that is the honest statement rather than a
padded one. It is why §1 says 14–16 calendar weeks.

**The 57/43 reading-to-hands-on split will feel wrong to you**, because you learn by
building. It is a direct consequence of Track C's module length — these ten modules average
12,570 words each, and `kv-cache-mechanics` and `kv-eviction-policies` each ask for 4–5
hours of reading before their first exercise. If you want to invert the ratio, the lever is
to read a module's §1–§3, run its exercises, and *then* read §4–§7. Several modules
(`kv-cache-mechanics`, `paged-attention-and-prefix-reuse`) survive that reordering; the two
measurement modules do not, because their §4 is the reason their exercises are designed the
way they are.

---

## 7. Difficulty scale used below

| Rating | Means |
|---|---|
| **2/5** | Your existing systems knowledge carries you. Failure mode: moving too fast and missing where the analogy breaks. |
| **3/5** | New machinery, but one idea at a time. Failure mode: accepting an equation without translating every symbol. |
| **4/5** | Several ideas held simultaneously, or an evidence-appraisal problem rather than a math problem. Failure mode: believing a published comparison. |
| **5/5** | The material is not the hard part; the discipline is. Failure mode: reporting a number you cannot resolve. |

The modules rate themselves, and I have kept their ratings. Where a week combines modules I
report the harder one and say which part earns it.

---

## 8. The twelve weeks

### Week 1 — Orientation, tensors, and where memory first bites

| | |
|---|---|
| **Modules** | `tensors-and-autograd` (full); `tokenization` (partial, §5); begin `transformer-forward-pass-by-hand` reading |
| **Reading** | 5.1 h — orientation 0.6 (`curriculum/README.md`, `ASSUMPTIONS.md`, `research/synthesis.md`); tensors 1.5; tokenization 1.5; forward-pass 1.5 |
| **Also open, never read cover to cover** | `curriculum/glossary.md` and `curriculum/reading-list.md`. Keep both in a second window from here to Week 12; they carry zero hours in this plan by design. |
| **Hands-on** | 2.75 h |
| **Total** | 7.85 h |
| **Difficulty** | **2/5** conceptually, 3/5 in the details |
| **Gate** | Tier 0 and Tier 1. Nothing here depends on bf16. |

**Run:**
- `tensors-and-autograd` — "stride forensics and the price of `.contiguous()`" (2/5, ~30 min write, <1 min run). Tier 1: a wall-clock number; record the env vars.
- `tensors-and-autograd` — "build an activation ledger and find the T² term" (3/5, 60–90 min, ~2 min GPU). **This ledger is reused in Weeks 3, 6 and 8.** Tier 1.
- `tokenization` — "the exchange rate" (2/5, ~45 min, CPU only). Tier 0.

**Skip:** `tensors-and-autograd` — "prove the gradient-accumulation identity" (duplicated in
Week 3, §5).

**Done when:** you can state, without looking, why a `(B, T, nh, hd)` → `(B, nh, T, hd)`
transpose is free and why the `.contiguous()` after it is not; and you have a
bytes-per-activation ledger that agrees with `torch.cuda.max_memory_allocated()` to within
a few percent.

---

### Week 2 — The forward pass by hand, and the bf16 error budget

| | |
|---|---|
| **Modules** | `transformer-forward-pass-by-hand` (exercises); begin `the-training-loop` reading |
| **Reading** | 2.5 h — training-loop 2.0; forward-pass read-the-code 0.5 |
| **Hands-on** | 4.75 h |
| **Total** | 7.25 h |
| **Difficulty** | **3/5** |
| **Gate** | **This week produces Gate evidence rather than consuming it.** |

**Run:**
- "compute the toy block by hand, then break PyTorch's tie" (~1.75 h). Tier 0 — fp64 arithmetic against a reference. Do this on paper first; it is the single highest-value hour in Track A.
- **"the bf16 error budget, stage by stage, on gfx1151"** (3/5, ~1.5 h). **This is Hardware Validation Gate item 2.** Record max absolute and relative error per stage — projection, score, softmax, value-weighted sum, norm — against an fp32 reference. Write the result into `ASSUMPTIONS.md → bf16-numerics-unproven` with the wheel version (`torch 2.12.0a0+rocm7.13.0a20260313`).
- **Launch the known-good recipe** (~1.5 h attended). nanoGPT `shakespeare_char`, published config, target val loss **1.4697** (`training/nanogpt/README.md:51`, which also quotes ~3 min on one A100). **Time the first run before planning anything around it** — 20.9 TFLOPS bf16 is a bad estimator for a launch-bound 10.6M model, and several modules say so in their own words.

**Skip:** "find the T² term, then watch it disappear" (duplicated, §5).

**Done when:** you have a per-stage bf16 error table, and one nanoGPT run that reached the
published loss. If it did not reach 1.4697, stop and debug — that is the Gate telling you
something, and every measured number in Weeks 3–12 depends on it.

---

### Week 3 — The training loop, determinism, and the checkpoint you reuse all quarter

| | |
|---|---|
| **Modules** | `the-training-loop` (exercises); `loss-and-optimization` (partial); begin `attention-variants-and-kv-cost` |
| **Reading** | 2.95 h — loss/perplexity sections 1.2; attention-variants §1–3 1.75 |
| **Hands-on** | 3.85 h |
| **Total** | 6.8 h |
| **Difficulty** | **3/5** |
| **Gate** | Exercise C **is** Gate item 3. The 3-seed checkpoint set is Gate item 6. |

**Run:**
- Exercise A — "Account for every byte in one step" (2/5, 45–75 min). Tier 1.
- Exercise B — "Break gradient accumulation, then fix it" (3/5, 30–45 min). Tier 0 — it reproduces a bug that shipped in most LLM trainers for years, and it is exact arithmetic.
- **Exercise C — "Precision and repeatability on this specific machine"** (3/5, 60–90 min). **Gate item 3.** The module's own frontmatter says it feeds two open `ASSUMPTIONS.md` rows. Tier 2 for the bf16 arm, Tier 0 for the fp32 arm.
- **Produce the reusable checkpoint set**: nanoGPT `shakespeare_char`, **3 seeds**, matched token budget, checkpoints kept (~1 h attended). Weeks 8, 11 and 12 all consume these.

**A trap to know about before you produce them:** `training/nanogpt/train.py:123` samples
eval batches with `torch.randint` from the **global** RNG seeded once at
`training/nanogpt/train.py:106`. Training draws from the same stream, so two runs that
differ in step count before an eval see different eval items — **your eval set is coupled
to your training schedule.** Fix it with a dedicated generator before you produce
checkpoints you intend to compare, or you will be measuring the coupling.

**Done when:** three checkpoints exist, each at the published loss, each reproducible from
committed config; and you can say which parts of a step are bitwise reproducible on this
machine and which are not.

---

### Week 4 — KV cost, the attention roofline, and the number the whole lab budget rests on

| | |
|---|---|
| **Modules** | `attention-variants-and-kv-cost` (complete); `scaling-laws-and-flops-budget` (partial); begin `memory-taxonomy-for-engineers` |
| **Reading** | 4.75 h — attention-variants §4–7 1.75; scaling-laws 2.0; taxonomy §1–2 1.0 |
| **Hands-on** | 3.25 h |
| **Total** | 8.0 h |
| **Difficulty** | **3/5** — "the arithmetic is easy, the two places it lies to you are not" |
| **Gate** | Exercise B is the attention-kernel roofline `synthesis.md` says the Gate is missing. |

**Run:**
- Exercise A — "the KV budget calculator, checked against three known answers" (~30 min, pure Python). Tier 0. Check it against `[M]` `ASSUMPTIONS.md → kv-per-token-laguna`: 2·48·8·128·2 B = **192.0 KiB/token exactly**, 24.0 GiB at 128k. If your calculator does not close to the byte, it is wrong.
- **Exercise B — decode arithmetic intensity against `AI = 2G/b`** (~45 min, 5–10 min GPU). **Gate widening item.** Tier 1. This attacks `ASSUMPTIONS.md → decode-intensity-varies-by-layer`, which is currently `[A]` from a derivation that has never been run.
- Exercise C — "prove that `repeat_kv` materialises, and price it" (~30 min). Tier 1.
- **`scaling-laws` — "measure sustained throughput and replace the `[A]`"** (medium, ~90 min incl. runs). **This replaces the 7.3 TFLOP/s in `building-an-eval-you-can-trust.md` §3.6.** Until it runs, every wall-clock estimate in this repo — including this schedule's estimate of *its own* deferred exercises — is `[A]`.

**Where the analogy breaks, and it is worth the pen:** GQA group size `G` is **6 on
Laguna's 12 full-attention layers and 9 on its 36 sliding layers** `[M]`
(`ASSUMPTIONS.md → laguna-heads-uniform`, read from `config.json`, and see
`models/laguna-s/modeling_laguna.py:473` for the per-layer lookup). A cost model keyed on
the top-level `num_attention_heads: 48` is wrong for 75% of layers. This is the first place
"read the config" fails you.

**Done when:** you can compute KV bytes/token for any config from memory, and you have a
measured end-to-end TFLOP/s to substitute into every cost table you meet from here on.

---

### Week 5 — Five things called memory, and the one you already built

| | |
|---|---|
| **Modules** | `memory-taxonomy-for-engineers` (complete); `agent-memory-in-practice` (mostly) |
| **Reading** | 4.0 h — taxonomy §3–7 1.5; agent-memory 2.5 |
| **Hands-on** | 3.25 h |
| **Total** | 7.25 h |
| **Difficulty** | **2/5** conceptually, **4/5** in the places your instincts fire wrong |
| **Gate** | Tier 0 throughout except one arm. |

**Run:**
- Taxonomy Exercise A — "the five-leg budget, and three crossovers you can check" (~40 min write, <1 s run). Tier 0.
- Taxonomy Exercise B — "is recompute exact? Three prefill schedules, one cache" (~40 s per arm GPU; CPU fallback gives a *different* answer, which is the point). Tier 2 for the bf16 arm — the whole exercise is a numerics question.
- Taxonomy Exercise C — "map the dependency cone, and derive the repair unit" (~15 min write, <20 s; **use fp32 so the answer is structural**). Tier 0.
- Agent-memory Exercise A — "the compaction ledger: price a policy in prefill, not in summarizer tokens" (`[M]` 0.4 s runtime, ~30 min thinking). Tier 0.
- Agent-memory Exercise B — "the K/N law: prove retrieval is not the failing component" (`[M]` 166 s, CPU/numpy). Tier 0.

**Why this pairing.** This is the week where your existing expertise is worth the most and
is most likely to mislead you. You have built tiered agent memory. The taxonomy module's
job is to show you that four of the five things called "memory" have no backing store, no
fault path and no miss signal — **discarding is always legal** — and that the storage
hierarchy stops being a useful lie at exactly the point where you would reach for a
write-back policy. Read §3 of `memory-taxonomy-for-engineers` slowly.

**Done when:** you can name, for each of the five memory kinds, its reconstructibility cost
and whether a miss is even observable.

---

### Week 6 — KV cache mechanics: residency, read traffic, and the maintenance budget nobody prices

| | |
|---|---|
| **Modules** | `agent-memory-in-practice` (finish); `kv-cache-mechanics` (complete) |
| **Reading** | 4.5 h |
| **Hands-on** | 3.25 h |
| **Total** | 7.75 h |
| **Difficulty** | **4/5** — the module calls itself "hard", and it is right: three budgets held apart at once, and the machine disagrees with two of them |
| **Gate** | Tier 1 throughout. Record the three env vars on every number. |

**Run:**
- Agent-memory Exercise C — "what a compaction event costs in seconds on this machine" (`[M]` 62 s default GPU arm, 8 s AOTriton arm, ~7 s CPU). Tier 1. **Note the 62 → 8 s gap: that is the AOTriton flag, and it is the same effect as the 18× activation-memory result in `ASSUMPTIONS.md`.**
- Exercise A — "`torch.cat` versus preallocation" (~2 min GPU). Tier 1. Read `architecture/transformers/src/transformers/cache_utils.py:143` first — `self.keys = torch.cat([self.keys, key_states], dim=-2)` — then `:454`, `index_copy_`. That pair *is* the module.
- Exercise B — "the two-tier stack: measure how wrong a single-number bandwidth model is" (~3 min GPU). Tier 1. Budget against the `[M]` ≥62 GiB fast tier; cap allocations at 24 GiB.
- Exercise C — "does halving `b` halve the time? (It does not.)" (~2 min). Tier 1.

**Read the code, minimum set:** `architecture/transformers/src/transformers/cache_utils.py:112`
(`DynamicLayer` — four lines of state and one `update`) against `:372` (`StaticLayer`). Then
`memory/vllm/vllm/v1/kv_cache_interface.py:227` and its docstring, which says that with the
hybrid allocator off, sliding-window layers get blocks allocated **for every token anyway** —
a 4× residency difference from a feature flag, with the model and the config unchanged.

**The analogy break to carry out of this week:** residency is a property of the *allocator*,
read traffic of the *kernel*, maintenance traffic of the *cache class*. **None of the three
is a property of the architecture, and the config file describes none of them.**

---

### Week 7 — Eviction: deciding what to forget before you know what you will be asked

| | |
|---|---|
| **Modules** | `kv-eviction-policies` (reading + two exercises); begin `paged-attention-and-prefix-reuse` |
| **Reading** | 6.0 h — eviction 4.5; paged §1–2 1.5 |
| **Hands-on** | 1.5 h |
| **Total** | 7.5 h |
| **Difficulty** | **4/5** — "the algebra is four lines; the discipline of not believing a policy comparison is the hard part" |
| **Gate** | Tier 0 — both exercises are CPU fp32. |

**Run:**
- Exercise A — "the error identity, and what actually predicts damage" (~30 min write, CPU fp32). Tier 0.
- Exercise B — "the policy null: calibrate before you conclude" (~20 min write, seconds to run). Tier 0. **This is the most important twenty minutes in Track C.** It gives you a seed-to-seed null distribution, and without one, every policy comparison you will read for the rest of your career is uninterpretable.

**Warning: this is the heaviest reading week in the plan** — 6.0 hours of reading against
1.5 hands-on, an 80/20 split that inverts how you like to work. It is deliberate: the
eviction module's §4 is an evidence-appraisal argument, and its exercises are only
meaningful after it. If you need to break it, read §1–§4, run Exercise B, then read §5–§7.

**The thing that makes this track worth doing:** `[C]` under StreamingLLM / SnapKV / TOVA /
H2O, some instructions are **dropped entirely** while LongBench scores look fine
([2510.00231](https://arxiv.org/abs/2510.00231), ACL 2026, rev. May 2026, via
`research/reference/papers/anchors.bib`). And `[C]` refusals fall 15.2% across 11 models and
1,894 prompts at 1.03× perplexity under KV quantization
([2606.09864](https://arxiv.org/abs/2606.09864)). The outcome metric held; the mechanism
broke. That dissociation is the lab's entire research thesis, and Week 7 is where you first
see the mechanism that produces it.

---

### Week 8 — Paged attention: the page table with no fault handler

| | |
|---|---|
| **Modules** | `kv-eviction-policies` (finish); `paged-attention-and-prefix-reuse` (complete) |
| **Reading** | 1.5 h — paged §3–7 |
| **Hands-on** | 5.5 h |
| **Total** | 7.0 h |
| **Difficulty** | **4/5** — "the code is easy, the four analogy breaks are not" |
| **Gate** | Exercise C Tier 1; eviction Exercise C **Tier 2** (bf16 on a real model). |

**Run:**
- Eviction Exercise C — "the prematurity penalty on a real model" (~45 min write). **Uses the Week 3 checkpoints.** Tier 2 — label the number provisional.
- Paged Exercise A — "the allocator, and the block size that byte accounting recommends" (45–60 min write, <1 s run). Tier 0.
- Paged Exercise B — "the prefix chain, and hit rate measured three ways" (1.5–2 h write). Tier 0. Three definitions of "hit rate" that disagree — this is the module.
- Paged Exercise C — "what does non-contiguity actually cost on gfx1151?" (1.5–2 h write). Tier 1.

**Note on pacing, honestly.** `paged-attention-and-prefix-reuse.md` says in its own
frontmatter: *"4–5 h for the three exercises. Two weeks at 8 h/wk."* I am giving it about
1.5 weeks by moving its reading into Week 7. If it overruns, it overruns here, and Week 9
has 0.5 h of give. Do not compress Exercise B to make the calendar work — the three hit-rate
definitions are the payload.

**Read the code:** `memory/vllm/vllm/v1/core/block_pool.py:647` (`get_new_blocks` — on
failure there is **no fault and no demotion**; the request is preempted) and `:719`
(`free_blocks` — why eviction can be a pointer update). Virtual memory has a fault handler.
This does not. That is analogy break number one of four.

---

### Week 9 — Constant state: the fixed-size rolling aggregate

| | |
|---|---|
| **Modules** | `constant-state-memory` (complete); begin `positional-encoding` |
| **Reading** | 5.0 h — constant-state 3.5; positional-encoding §1–3 1.5 |
| **Hands-on** | 2.5 h |
| **Total** | 7.5 h |
| **Difficulty** | **4/5** — "the recurrence is four lines of code; what is hard is that its failure mode is invisible from the outside" |
| **Gate** | Exercise A Tier 0 (CPU fp64); B and C Tier 1. |

**Run:**
- Exercise A — "the capacity of a state, with no training at all" (`[M]` ~90 s runtime; 2/5 to run, **4/5 to interpret**). Tier 0 — CPU, fp64. The module says this is "90 seconds of compute and an evening of staring at the output". Believe it, and budget the evening.
- Exercise B — "the two crossovers, and why they are 8× apart" (2–3 min GPU, 5–10 min CPU). Tier 1.
- Exercise C — "price the prefill, and find the chunk length nobody optimizes for" (<1 min; the derivation is the work). Tier 1.

**The bridge, and where it breaks.** An SSM hidden state is a fixed-size rolling aggregate
against an unbounded log — you have built this exact thing, and you know its properties:
constant memory, O(1) update, lossy in a way that depends entirely on what the aggregate
function forgot. The break is that **you cannot inspect what it forgot.** A rolling
percentile sketch has known error bounds; this does not, and Exercise A is how you find its
capacity empirically because no one can tell you it analytically.

---

### Week 10 — Hybrid ratios, and where RoPE breaks in bf16

| | |
|---|---|
| **Modules** | `hybrid-attention-and-ratios` (complete); `positional-encoding` (partial, finish) |
| **Reading** | 4.6 h — hybrid 3.5; positional-encoding §4–6 1.1 |
| **Hands-on** | 3.2 h |
| **Total** | 7.8 h |
| **Difficulty** | **3/5** on the arithmetic, **4/5 on the evidence appraisal** |
| **Gate** | RoPE Exercise A is a Gate-widening item. Hybrid Exercise B Tier 1. |

**Run:**
- Hybrid Exercise A — "the crossover calculator" (~30 min write, pure Python). Tier 0.
- Hybrid Exercise B — "does decode time track residency across schedules?" (~25 min GPU, six subprocess launches at 60–90 s each). Tier 1.
- Hybrid Exercise C — "receptive field under placement, at matched global count" (~45 min write). Tier 0.
- **`positional-encoding` — "measure where bf16 breaks RoPE's relative-position identity"** (2/5, 45–75 min write, <1 min run). **This is the Gate item `research/synthesis.md` says CLAUDE.md is missing.** Tier 2 by construction — it *is* the numerics test.
- `positional-encoding` — "reconstruct Laguna's YaRN ladder from `config.json` and check all four published constants" (2/5, 60–90 min, no GPU). Tier 0.

**The folklore to attack this week.** "3:1 is the right hybrid ratio." Four labs shipped it;
`research/synthesis.md` records that **none ablated it against the others**, and every
ablation reporting a quality surface reports a *flat* one. The best available evidence is
`[C]` 72 trained models sweeping six linear-attention variants against five ratios
([2507.06457](https://arxiv.org/abs/2507.06457)), landing on 3:1–6:1 and showing recall
collapse as full-attention layers thin — and whether that ratio sets a **capability
ceiling** or only governs how fast long-context ability **emerges** is left contested in the
survey. `[M]` Laguna itself is 12 full + 36 sliding in a strict GSSS pattern — a 3:1 ratio
read from the artifact, not quoted. The 4/5 rating on this week is entirely about holding
that distinction while reading confident papers.

---

### Week 11 — Long context: advertised capacity, usable capacity, and the gap with no alarm

| | |
|---|---|
| **Modules** | `long-context-and-effective-context` (partial); begin `memory-failure-modes` |
| **Reading** | 5.2 h — long-context 3.2; failure-modes §1–4 2.0 |
| **Hands-on** | 2.25 h |
| **Total** | 7.45 h |
| **Difficulty** | **4/5** |
| **Gate** | Exercise A Tier 0; Exercise B **Tier 2**. |

**Run:**
- Exercise A — "the dilution ledger, and the eviction paradox" (2/5, 45–60 min write, seconds to run, no GPU). Tier 0.
- Exercise B — "does the needle survive the accumulator?" (3/5, 60–90 min write, 2–5 min run). Tier 2. **Uses the Week 3 checkpoints.**

**Deferred with reason:** Exercise C, the scaled RULER at nanoGPT scale (`[A]` 3–6 h GPU,
one evening of writing). The module itself says to pre-register it with a G2 hypothesis card
before running. That makes it a `notebook/` experiment, not a homework problem, and it goes
in the deferred register (§9).

**Why the needle exercise matters more than it looks.** `research/synthesis.md` makes the
argument you should be testing against: a needle is a **high-salience span that attracts
attention mass**, which is exactly what heavy-hitter eviction retains — so
needle-in-a-haystack **structurally cannot fail** for H2O-style policies. If your Exercise B
result shows the needle surviving everything, that is not good news about the policy; it is
a statement about the eval. Write it down that way.

**Also this week, folklore worth naming:** "the model has a 1M context." `[M]` Laguna's
1,048,576 is 8192 × 128 exactly — pretraining length times the YaRN extension factor — and
its `attention_factor` matches YaRN's default temperature formula to the last digit.
Inherited convention, not demonstrated capability.

---

### Week 12 — Failure modes, and the instrument the lab actually owes itself

| | |
|---|---|
| **Modules** | `memory-failure-modes` (complete); `measuring-memory` (read + the one exercise that matters) |
| **Reading** | 6.0 h — failure-modes §5–7 1.5; measuring-memory 4.5 |
| **Hands-on** | 2.5 h |
| **Total** | 8.5 h — **the one week over budget, and it is over on purpose** |
| **Difficulty** | **4/5** methodologically; the math is one page, the epistemology is not |
| **Gate** | Exercises A and C Tier 0; measuring-memory Exercise B **Tier 2**, and it is the deliverable, so the Gate label matters most here. |

**Run:**
- Failure-modes Exercise A — "Split eviction damage into information loss and renormalisation" (~4 min for part 1 at three seeds, CPU/numpy). Tier 0.
- Failure-modes Exercise B — "Measure the needle's salience rank, and locate the eval's blind spot" (~15 min including a ~500 MB download on first run, CPU). Tier 0. **Download this before the week starts.**
- **`measuring-memory` Exercise B — "build the oracle-diff instrument and measure its noise floor"** (45–90 min to write, <2 min on GPU, ~5 on CPU). Tier 2. **This is the twelve weeks' terminal deliverable.**

**Skip:** failure-modes Exercise C ("Three definitions of 'the victim', one trace") — good,
but it is a definitional exercise and the two above carry the week.

**Why the plan ends here rather than on a module boundary.** `research/synthesis.md` and
ADR: `attribution-instrument-over-eviction-policy` (**Proposed** as of 2026-07-26 — if it is
rejected at review, this week's target changes and so does §10) say the lab's contribution is
*an attribution instrument, not another eviction policy* — a field
with ~30 policies and no dominance result does not need a 31st. Attribution requires a
**full-cache oracle on every probe**: you run the expensive thing you were trying to avoid.
At 300M that is ~600 MB of weights against a `[M]` ≥62 GiB fast tier; at 70B it is
unaffordable. **Small scale is the enabling condition for this question, not a compromise** —
and Week 12 is where you build the thing that exploits it.

**Riskiest thing you will have built.** `research/synthesis.md` names it: that
distributional divergence from a full-cache oracle measures anything decision-relevant.
Divergence and task accuracy can dissociate in *both* directions — a policy can shift the
output distribution without flipping any argmax, or flip one critical token at negligible
average KL. The next test is the first thing in Week 13: drop a **known** cache entry and
check whether per-token KL localises to that entry and moves only when the recoverable token
moves. If it does not, the whole plan changes, which is what makes it worth running first.

---

## 9. Deferred exercise register

These were cut from Weeks 1–12 for time, not for value. Each states where it goes.

| Exercise | Module | Cost | Where it goes |
|---|---|---|---|
| Length-generalization sweep, 4 positional schemes × ≥3 seeds | `positional-encoding` | `[A]` 2–5 h GPU + an evening | `notebook/`, pre-registered. Time one arm first — the module's estimate scales an A100 figure by our GEMM number and says so. |
| Scaled RULER at nanoGPT scale with control arms | `long-context-and-effective-context` | `[A]` 3–6 h GPU + an evening | `notebook/`, pre-registered. Reuses the `rope_full` checkpoint from the row above; do them adjacently. |
| Fault-inject a retrieval metric until it fails | `measuring-memory` | 20 min write, ~15 s run | This is `synthesis.md` question 3 — the six-fault calibration battery. Zero training runs, publishable methodology. **Highest information-per-hour item in the whole deferred set.** |
| Price the probe, find the hidden syncs | `measuring-memory` | ~3 min GPU | Week 13, alongside the instrument's first real use. |
| Write the BPE encoder | `tokenization` | 4/5, ~90 min | Week 15, before Track E. |
| What the vocabulary costs | `tokenization` | 3/5, ~60 min | Week 15. |
| All three `loss-and-optimization` exercises | `loss-and-optimization` | ~2.5 h | Week 13 — the bytes-per-logit guard threshold is genuinely useful before any training sweep. |
| Reconcile three FLOP counts; IsoFLOP fitting bias | `scaling-laws-and-flops-budget` | ~2 h | Week 14. |
| **32 GiB fp32 discriminator** | *no module owns it* | ~2 min | `notebook/` under G0-LIGHT, **immediately**. Same bytes, half the elements: failure means bytes, success means element count. |

---

## 10. Weeks 13–24 — the continuation, at the same cadence

Sequenced by prereq, not by track letter. Hours are reading + exercises at the same rate.

| Weeks | Modules | Hours | Why here |
|---|---|---|---|
| 13–14 | `measuring-memory` (finish); the oracle-localisation test from §8 Week 12; `checkpointing-and-resumption` | ~14 | Checkpointing Exercise A closes the last CLAUDE.md Gate item (bit-exact round-trip) and *becomes rig code*. |
| 15–16 | `determinism-and-reproducibility`; `tokenization` (finish); `normalization-and-activations` | ~16 | Determinism Exercise C explicitly "closes a Hardware Validation Gate item" and depends on Week 11's measurement. |
| 17–18 | `moe-and-routing`; `depth-width-and-initialization`; `scaling-laws` (finish) | ~15 | Track B closure. MoE is a prereq for `quantization`. |
| 19–20 | `training-telemetry-as-observability`; `distributed-training-strategies` | ~16 | Telemetry is **the module you teach back** — you have built this pipeline; its value is the three places your instincts are wrong. Distributed is design-only `[M]`: nothing runs, and the module says so in its first paragraph. |
| 21–22 | `building-an-eval-you-can-trust`; `measuring-recall-and-memory` | ~18 | Both need `measuring-memory` complete. `measuring-recall` also needs `constant-state` and `long-context`, which land in Weeks 9 and 11. |
| 23–24 | `supervised-and-preference-finetuning`; `quantization`; `speculative-decoding-and-serving`; `running-laguna-locally` | ~24 | Track F last because it is the least load-bearing for a memory research agenda — and `running-laguna-locally` is the reward: all three of its exercises run **today, without weights**. |

**Weeks 23–24 are 12 h/wk of content, not 8.** Say it now rather than discovering it. Either
extend to Week 26 or cut `supervised-and-preference-finetuning`'s Exercise B — the module
itself argues against running most of what it teaches, and `[C]` RLVR at 135M on one GPU
made GSM8K exact match *fall* (via `research/synthesis.md`).

**The three capstones** are specified in `curriculum/capstones.md`, and they are *projects*,
not weeks — they carry calendar estimates rather than hour estimates because each is a real
pre-registered experiment with training runs in it:

| Capstone | Build / evidence difficulty | Calendar | Earliest start |
|---|---|---|---|
| `telemetry-detects-injected-fault` | 3/5 build, 4/5 evidence | 5–7 weeks | after Week 20 — it needs `training-telemetry-as-observability`, `determinism-and-reproducibility` and `checkpointing-and-resumption` |
| `eviction-recall-breakpoint` | 4/5, 5/5 | 7–10 weeks | after the first — it consumes that one's run manifest, silent-hang watchdog and measured seed-to-seed null |
| `hybrid-ratio-recall-cliff` | 4/5, 5/5 | 10–14 weeks | after the second — it needs its oracle-diff attribution harness |

The ordering is a **dependency, not a difficulty ramp**, and `capstones.md` draws the graph.
Serially that is 22–31 weeks on top of Week 24. **Do one.** `telemetry-detects-injected-fault`
is the right one: it is the only capstone whose primary skill you already have, and it is the
prerequisite for both others — so finishing it leaves the option open rather than closing it.

---

## 11. Re-planning triggers

Timebox, then re-decide. Do not silently continue.

| Trigger | Action |
|---|---|
| **End of Week 3.** Log actual hours for Weeks 1–3 against the 21.9 h scheduled. | If actual > 26 h, the reading rate in §2.2 is wrong for you. Re-derive it from your own three weeks and re-plan the remaining nine. This is the single most valuable measurement in the schedule and it costs nothing. |
| **Week 2, nanoGPT does not reach val loss 1.4697.** | Stop. Gate item 6 has failed. Every Tier 1 and Tier 2 number in Weeks 3–12 is void until it passes. Pin the working configuration in `ASSUMPTIONS.md` with exact wheel and driver versions. |
| **Week 2, bf16 per-stage error is worse than `[A]` ~1e-2 relative at any stage.** | Do not proceed to Tier 2 exercises on bf16. Run Weeks 3–12 in fp32 at reduced shapes; the conclusions in Track C are structural and mostly survive it. Record which arms you switched. |
| **Week 4, measured end-to-end throughput differs from the `[A]` 7.3 TFLOP/s by >30%.** | Per the operating instructions' ±30% rule, that input was already top-priority to measure. Rewrite `building-an-eval-you-can-trust.md` §3.6's table with your number and re-cost every deferred exercise in §9. |
| **Any week runs >10 h.** | Cut an exercise, not the reading — the reading is what the next module assumes. Log the cut here. |
| **Any ROCm, PyTorch or driver upgrade.** | Re-run `scripts/preflight.ps1` and re-run the Gate. An upgrade is a change of instrument, not maintenance. Every Tier 1 and Tier 2 number taken before it is now provisional again. |

---

## 12. Close

**Decision.** Run the twelve weeks as written: Track A/B trimmed to the bone in Weeks 1–4
(which also closes most of the Hardware Validation Gate), all ten Track C modules in Weeks
5–12, terminating on the oracle-diff instrument. Accept that this is 51% of the curriculum
and schedule Weeks 13–24 for the rest.

**Riskiest assumption.** The 3,991 words/hour reading rate. It is derived from 26 module
authors' own estimates of *their own* material, which is a systematically optimistic
estimator — an author knows where the argument is going. If the real rate is 3,000 w/h, the
188-hour total becomes 219 and the twelve weeks cover 43% rather than 51%.

**Next test.** Time yourself on `kv-cache-mechanics.md` §3 alone — 2,100 words of the
densest arithmetic in Track C. At the assumed rate that is 32 minutes. Do it in Week 1 as a
calibration, before it is on the critical path, and scale the whole schedule by the ratio.
Cost: 30 minutes. It moves every number in §2.
