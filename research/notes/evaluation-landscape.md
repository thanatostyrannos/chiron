---
title: The 2026 evaluation landscape — what can fail, what can be run here, and what detects a memory regression
version: 1.0.0
date: 2026-07-26
---

# Evaluation landscape: agentic, long-context, and the metrics that can actually fail

This note settles three things. **First**, needle-in-a-haystack is not merely weak, it is
*adversely selected* for this lab's research question — a unique high-salience needle is
exactly the token an attention-mass eviction policy keeps, so NIAH is a metric that cannot
fail for the thing we most want to break; RULER replaced it as a methodology, and the
2026 stack (HELMET, NoLiMa, LongBench Pro, ATLAS, PredicateLongBench) replaced RULER as a
*result*. **Second**, at 20M–300M parameters essentially the entire agentic leaderboard is
out of reach and contamination inverts — we own the corpus, so leakage is computable and
the real risk is that our models are too weak for any real-task benchmark to move at all;
what remains cheap is the generator-based synthetic family, and it is cheap enough that
evaluation is a rounding error against training cost. **Third**, the only class of
instrument that can register a memory regression at our scale is *differential*, not
absolute: a full-cache oracle diffed against a policy, with a seed-to-seed null
distribution, calibrated by deliberate fault injection — because an accuracy metric needs
a competent model and a divergence metric needs only a reference run.

This note is the evaluation half of `research/memory/`. It does not re-derive the memory
taxonomy, the KV arithmetic, or the agent-memory benchmark critique in
`agent-memory-systems.md §6`; it supplies the measurement discipline those notes assume.

---

## 1. The one distinction that organises everything

Every eval answers one of two questions, and the field routinely reports the first while
claiming the second:

- **Outcome:** did the model produce the right answer?
- **Attribution:** did it produce the right answer *by the mechanism under test*?

For a memory-systems lab, only the second is evidence. A model can answer a retrieval
question by recalling a specific cached token, by exploiting lexical overlap between query
and needle, by a prior over plausible answers, or by an artifact of how the haystack was
constructed. All four produce the same number.

**Systems bridge.** This is the SLO-versus-trace distinction. "p99 improved" is an outcome;
"p99 improved because the read path started hitting the page cache" is attribution, and
thirty years of experience says the first without the second is how you ship a regression
that surfaces two quarters later. **Where the analogy breaks — three ways, and they
compound.** (1) There is no request id: causality runs through a continuous attention
distribution, not a call graph. (2) The counterfactual is only obtainable by running the
expensive thing you were trying to avoid — full-cache recomputation on every probe.
(3) The failure is silent by construction. A KV cache miss is not a latency event, it is a
correctness event; FlashInfer's page table cannot even *represent* a miss
(`CODE_MAP.md`, `decode.py:1239`), and the model emits a fluent wrong answer rather than an
error. Your instinct that a broken cache announces itself is wrong here. See
`research/memory/open-problems-ranked.md §2` for the same break stated from the policy side.

The lab's own rule follows: **an eval you have never seen fail is not an eval, it is a
decoration.** Section 6 turns that into a protocol.

---

## 2. Why needle-in-a-haystack is insufficient — precisely

NIAH inserts a sentence ("the magic number is 7492") into a long distractor document and
asks for it back. Five independent defects, in increasing order of how much they matter to
us.

**(a) It saturates.** A metric pinned at 99–100% has no dynamic range. `[C]` RULER
(arXiv 2404.06654, Apr 2024) evaluated 17 models all claiming ≥32K: nearly all score
near-perfectly on vanilla NIAH, and only about half hold up on RULER's harder families at
32K. A benchmark that everyone passes cannot rank anything.

**(b) It is a lexical-match test, not a recall test.** This is the sharpest published
refutation. `[C]` NoLiMa (arXiv 2502.05167, Feb 2025; ICML 2025) rebuilds NIAH so the
needle shares *minimal literal overlap* with the question, forcing at least one associative
hop. Across 13 models claiming ≥128K: **11 drop below 50% of their own short-context
baseline at 32K**, and GPT-4o falls from **99.3% to 69.7%**. Same nominal task, same
lengths; delete the string match and two-thirds of the capability evaporates. The
mechanism the authors propose is that attention has a much harder time locating relevant
material at length when there is no literal anchor.

**(c) Single needle, single hop, one position.** Multi-key and multi-hop variants are
categorically harder. Our own reading of the constant-state literature makes this concrete:
in the Gated DeltaNet family at 1.3B/100B tokens, **every architecture including the two
published in 2026 scores under 32% at 4K on multi-key retrieval** — inside the 4K training
length, so this is capacity failure, not extrapolation (`research/memory/constant-state-memory.md`,
tables from `[C]` 2412.06464 and `[C]` 2605.22791).

**(d) It measures average case.** `[C]` PredicateLongBench (arXiv 2607.08284, Jul 2026,
NVIDIA) makes exactly this complaint — existing long-context evals "predominantly measure
average-case performance, and many are either saturated or lack robustness" — and replaces
the needle with a predicate-satisfaction search over the whole input (longest contiguous
subsequence satisfying a constraint), reporting that frontier models degrade as difficulty
is scaled along its axes. The KV-compression literature reached the same place from the
other direction: `[C]` DefensiveKV (arXiv 2510.13334, Oct 2025) argues worst-case rather
than mean aggregation changes the ranking of eviction policies.

**(e) The one that matters most here: NIAH is biased *in favour* of the policies we want
to stress.** The needle is a low-frequency, high-salience, semantically anomalous span. It
attracts attention mass. Heavy-hitter eviction (`[C]` H2O, 2306.14048) and observation-window
selection (`[C]` SnapKV, 2404.14469) are precisely the policies that retain high-attention
tokens. So a policy can shed 90% of the cache, destroy ordinary long-range dependence, and
still pass NIAH. `[A]` **High confidence, and this is the note's central claim about
existing practice**: NIAH's pass criterion is positively correlated with what
attention-based eviction preserves, which makes it a metric that structurally cannot fail
for the mechanism under test. Cheapest test that would move it: run H2O and SnapKV at a
severe budget against (i) standard S-NIAH and (ii) a NoLiMa-style associative needle at
matched length; if the gap between (i) and (ii) is large and grows with compression, the
claim holds. That is a one-day experiment on the inference rig.

Corroborating evidence that benchmark choice, not policy quality, is deciding published
rankings: `[C]` The Pitfalls of KV Cache Compression (arXiv 2510.00231, rev. 2026) shows
StreamingLLM/SnapKV/TOVA/H2O/K-Norm silently dropping *specific instructions* while
aggregate LongBench scores look fine; `[C]` SCBench (arXiv 2412.10319, Dec 2024) shows
rankings not surviving multi-turn cache reuse.

---

## 3. What replaced it — the 2026 long-context stack

RULER's durable contribution is **methodology, not data**: it is a *generator* over 13
synthetic task families (retrieval, multi-hop tracing, aggregation, QA) at controllable
length, plus a threshold definition of *effective context* (the longest length at which a
model still beats a fixed short-context reference score). Cite it for the harness design;
cite something newer for numbers.

| Benchmark | What it actually measures | What it misses | Tier (§5) | Runnable at 20M–300M? |
|---|---|---|---|---|
| NIAH | lexical retrieval of a salient span | everything in §2 | generator | yes — and near-useless |
| **RULER** `[C]` 2404.06654 | 13 synthetic families, controlled length, effective-context threshold | realism; single-number framing | **generator** | **yes, scaled** |
| **NoLiMa** `[C]` 2502.05167 | retrieval *without* literal overlap | multi-hop composition; generation | generator + curated | yes, needs a needle set built at our vocabulary |
| **HELMET** `[C]` 2410.02694 | 7 application categories to 128K, model-based metrics | needs a competent model and judges | static data + open harness | **no** |
| BABILong `[C]` 2406.10149 | reasoning (not just retrieval) in a haystack | still synthetic scaffolding | generator | yes, partially |
| Michelangelo `[C]` 2409.12640 | *latent structure* queries — answer is a function of the whole context (Latent List, MRCR) | narrow task family | generator | **yes — see §6** |
| LongProc `[C]` 2501.05414 | long *output* / procedural generation | input-side retrieval | static | no (generation cost, §5) |
| LongBench v2 / Pro `[C]` 2412.15204 / 2601.02872 | realistic bilingual tasks, 8k–256k, 46 models | contamination over time; needs real comprehension | static | no |
| ATLAS `[C]` 2605.28079 | 8K–1M grid, 26 models, **rank stability** | cost | static | no |
| PredicateLongBench `[C]` 2607.08284 | worst-case difficulty axes, predicate search | recency; too new to have a comparator base | generator | yes, in principle |
| SCBench `[C]` 2412.10319 | the **KV cache** across multi-turn reuse | model quality | harness over models | inference rig only |
| CL-bench `[C]` 2602.03587 | learning *from* context rather than retrieving from it | — | static | no |

Three findings are now well replicated and should be treated as settled:

1. **Effective context is far below advertised context, and the gap has not closed.**
   RULER (2024) → LongBench Pro `[C]` 2601.02872 (Jan 2026, 46 models, naturally occurring
   tasks) → ATLAS `[C]` 2605.28079 (May 2026, to 1M).
2. **Synthetic retrieval does not predict downstream use.** `[C]` HELMET
   (arXiv 2410.02694, Oct 2024) states it directly: NIAH does not reliably predict
   downstream performance, its seven categories have *low mutual correlation*, and the
   open/closed gap widens with length on full-context reasoning. `[C]` ATLAS reports the
   same disease at 1M scale: strong retrieval does not transfer, and **7 of 26 models shift
   ≥2 rank positions** between the 8K–128K and 8K–1M regimes, with gaps up to 12 positions.
3. **Therefore a single long-context number induces an unstable ordering.** This is not a
   precision complaint. It says the *ranking* — the only thing an ablation produces — is a
   function of the length grid you chose.

**Contested, and left contested.** Whether the future is *more synthetic* (RULER, NoLiMa,
Michelangelo, PredicateLongBench: controllable, generator-based, contamination-immune, but
arguably measuring an artificial skill) or *more realistic* (HELMET, LongBench Pro, ATLAS:
externally valid but static, contaminating, and expensive). `[C]` 100-LongBench
(arXiv 2505.19293, May 2025) argues de-facto benchmarks are not literally evaluating
long-context ability at all and rebuilds them length-controllably; `[C]` Dolce
(arXiv 2409.06338, Sep 2024) argues the axis that matters is retrieval-versus-holistic and
that most suites do not separate the two. There is no consensus, and a lab that picks one
side silently is picking its own conclusion. **Our position, stated as a position:** for
*mechanism* work at ablation scale, generator-based synthetic wins on every axis we care
about (contamination-proof, scales down, known ground truth, computable chance level), and
the realism deficit is a threat to external validity that must be *declared* rather than
papered over — which is what `ASSUMPTIONS.md: ablation-scale-sufficient` already is.

---

## 4. Agentic evaluation in 2026 — and why almost none of it is ours

**The landscape.** Coding: SWE-bench `[C]` 2310.06770 and its long-horizon successor
SWE-bench Pro `[C]` 2509.16941 (Sep 2025); Terminal-Bench 2.0 (tbench.ai — no arXiv id
found; shell-agent tasks with verified rubrics). Assistants and tools: GAIA `[C]` 2311.12983;
τ-bench `[C]` 2406.12045 and τ²-bench `[C]` 2506.07982 (dual-control conversational
agents); BrowseComp `[C]` 2504.12516; OSWorld `[C]` 2404.07972. Economic realism: GDPval
(openai.com/index/gdpval — vendor-run, no arXiv id found). Capability trend: METR's
time-horizon methodology `[C]` 2503.14499 ("Measuring AI Ability to Complete Long Software
Tasks"). Infrastructure: the Holistic Agent Leaderboard `[C]` 2510.11977 exists precisely
because an agentic score without a pinned harness is not a number.

**The validity crisis is the actual 2026 story, and it is well documented.**

- `[C]` **Agentic Benchmark Checklist** (arXiv 2507.02825, Jul 2025). Task validity ("the
  task is solvable iff the agent has the target capability") and outcome validity ("the
  reward measures task completion"). Reported: existing benchmarks under- or over-estimate
  performance **by up to 100% in relative terms** — SWE-bench Verified with insufficient
  tests, τ-bench counting empty responses as success — and applying ABC to CVE-Bench cuts
  overestimation by 33%.
- `[C]` **The SWE-Bench Illusion** (arXiv 2506.12286, Jun 2025). Models identify buggy file
  paths from the issue text alone at up to **76%** on SWE-bench, versus up to **53%** on
  repositories outside it; verbatim 5-gram reproduction of the target function reaches
  **35%** on SWE-bench Verified against **18%** elsewhere. This is memorisation wearing an
  agentic costume, and it is the single best illustration in the literature of the
  recall-versus-pattern-match confusion this note is about.
- `[C]` **Search-Time Contamination** (arXiv 2606.05241, Jun 2026). Deep-research agents
  with live web access retrieve the benchmark's own published answers at run time. The
  contamination is not in the weights; it is in the environment. Nothing about
  training-data hygiene protects you.
- `[C]` **Do Coding Agents Deceive Us?** (arXiv 2606.07379, Jun 2026) and `[C]` **Automated
  Benchmark Auditing** (arXiv 2605.26079, May 2026) — cheating detection via randomized
  held-out tests, and automated auditing of benchmark defects.
- `[C]` **Efficient Benchmarking of AI Agents** (arXiv 2603.23749, Mar 2026) — the cost
  problem stated as a research problem.

**Systems bridge.** An agentic benchmark is a distributed system: containers, tool servers,
network, retries, flaky tests, environment drift, and a stochastic oracle (the LLM judge).
Reproducing a score means pinning the model, the scaffold, the container image, the tool
versions, *and* the judge. **Where it breaks:** your integration tests have a deterministic
pass criterion; here the grader is itself a model whose biases correlate with the system
under test, and whose version moves under you.

**Verdict for Chiron, unhedged: none of §4 is runnable at 20M–300M.** Every task in that
list requires instruction following, tool-schema adherence and multi-step planning that a
300M model does not have; the scores would be zero, and a metric floored at zero has the
same problem as one pinned at 100. Reading these papers buys us *methodology* (ABC's
validity checklist, HAL's harness pinning, randomized held-out grading) and nothing else.
The one exception is the second rig already identified in
`research/memory/open-problems-ranked.md` — inference-only studies on an off-the-shelf
7–14B model inside the 62 GiB fast tier `[M]` — where instruction-following failures under
cache compression `[C]` (2510.00231), alignment collapse under KV quantization `[C]`
(2606.09864, Jun 2026) and governance decay under compaction `[C]` (2606.22528, Jun 2026)
become measurable. Numbers seen only on leaderboard blogs are excluded from this note by
the house rule; blog posts are leads, not evidence.

---

## 5. Contamination — and why it inverts at our scale

**Two different problems now share one word.**

*Training-time contamination*: benchmark items appear in pretraining data. *Runtime
contamination*: the agent fetches the answer during evaluation `[C]` (2606.05241).

**Detection families, and their honest state.**

| Family | Method | Status |
|---|---|---|
| Substring / n-gram | 13-gram overlap (the GPT-3 convention) | Necessary, wholly insufficient — `[C]` 2311.04850 shows paraphrase and translation walk straight through it, that a 13B model overfit on rephrased test items reaches GPT-4-level scores, and that **8–18% of HumanEval appears in RedPajama-1T and StarCoder-Data** |
| Likelihood / membership inference | Min-K% `[C]` 2310.16789, Min-K%++ `[C]` 2404.02936 | **Weak.** `[C]` 2402.07841 evaluates MIAs on Pile-trained models from 160M to 12B and finds they "barely outperform random guessing" in most settings; apparent successes are attributable to temporal distribution shift between members and non-members |
| Performance-based | ConStat `[C]` 2405.16281, PaCoST `[C]` 2406.18326 | Statistically principled — compares against a reference set rather than looking inside the model; needs a trusted uncontaminated comparator |
| Behavioural | ICL-based detection `[C]` 2510.27055; canary strings | Promising, unsettled |
| Code-specific | TRACER `[C]` 2605.24079 (May 2026), semantic-aware, fine-grained | New; the n-gram-evasion answer for code |
| Post-hoc repair | Inference-time decontamination `[C]` 2601.19334 (Jan 2026) | Treats leakage as recoverable at eval time rather than requiring retraining |

And the load-bearing negative result: `[C]` **Evading Data Contamination Detection for
Language Models is (too) Easy** (arXiv 2402.02823, Feb 2024) — a malicious or merely
careless trainer defeats the standard detectors. Surveys: `[C]` 2404.00699, `[C]` 2406.04244,
`[C]` 2502.14425, `[C]` 2502.17521 (static→dynamic).

**Mitigation has converged on three moves**, none of which is detection: rotate the data
(`[C]` LiveBench 2406.19314, `[C]` LiveCodeBench 2403.07974 — release windowed by date);
construct contamination-free by design (`[C]` MMLU-CF 2412.15194); or *generate* items so
the surface form is fresh every run. `[C]` **GSM-Symbolic** (arXiv 2410.05229, Oct 2024) is
the cleanest demonstration of why generation is the right primitive: re-instantiating the
same maths problems with different names and numbers moves scores, which means the original
score was partly measuring surface familiarity. That is the pattern-match-versus-reason
separation performed by perturbation, and it transfers directly to memory evals.

**Systems bridge.** Contamination is benchmarking with a warm cache you forgot to drop, and
membership inference is trying to prove after the fact, from timing alone, that the cache
was cold. `[C]` 2402.07841 is the measurement that says you cannot. **Where it breaks:**
`echo 3 > /proc/sys/vm/drop_caches` exists. There is no drop_caches for a 15T-token
pretraining run — which is exactly why the field moved from detection to rotation.

**Now the inversion, which is the part that matters for us.** At 20M–300M on 0.5–5B
self-selected tokens, we *own the corpus*. Contamination becomes a computable property:
exact-substring and n-gram overlap between eval items and the training shards, computed
once, reported as a number. Our synthetic evals are generated at eval time from seeds we
control, so training-time contamination is structurally impossible for them. **Our actual
risks are three different ones, and they are under-discussed:**

1. **Under-capability, not contamination.** A 300M model is at chance on most knowledge
   benchmarks. `[C]` Quantifying Variance in Evaluation Benchmarks (arXiv 2406.10229,
   Jun 2024) finds seed variance and non-monotonicity large enough that differences at
   small scale are frequently not meaningful, and that reframing choice tasks as completion
   tasks reduces variance at ~7B. Below that, most multiple-choice suites are noise
   generators. `[C]` Are Emergent Abilities a Mirage? (arXiv 2304.15004, Apr 2023) supplies
   the mechanism: discontinuous metrics (exact match, accuracy) manufacture cliffs that
   continuous metrics (per-token log-likelihood, edit distance) do not. **Prefer continuous
   metrics at our scale — not for elegance, for statistical power.**
2. **Harness leakage.** If the needle is drawn from the same generator that produced the
   training data, the model can reconstruct it without recalling it. Our synthetic evals
   need a *held-out symbol inventory*, and that must be checked, not assumed.
3. **Statistical power.** `[C]` Adding Error Bars to Evals (arXiv 2411.00640, Nov 2024)
   gives the framework — question-level clustering, paired analysis across arms, power
   analysis before the run. The house rule already demands ≥3 seeds and CIs; the missing
   piece is stating the *minimum detectable effect* in the pre-registration card, because a
   suite of 100 items cannot resolve a 3-point difference and pretending otherwise is how
   a null gets reported as a win.

**Openness tiers, applied to evals** (the same discipline this lab applies to model
releases, and it is orthogonal to model openness — an open-weights model scored on a closed
benchmark yields a number nobody can reproduce):

- **Generator-open** (RULER, MQAR/Zoology, flip-flop, GSM-Symbolic, PredicateLongBench):
  code emits items. Infinitely re-instantiable, contamination-immune, scales down. **Build
  only on this tier.**
- **Data-open + harness-open** (LongBench, HELMET, LongMemEval, SWE-bench): reproducible
  today, contaminating monotonically.
- **Harness-open, data-rotating or held-out** (LiveBench, LiveCodeBench, private splits):
  reproducible only on the maintainer's cadence; you cannot audit the items.
- **Vendor-run** (GDPval, most model-card long-context claims): a lead, never evidence.

---

## 6. The question that matters: which evals detect a *memory* regression?

Target definition, stated so it can be falsified: an eval detects a memory regression if
its score degrades when, and only when, the system's ability to **recover information about
a specific earlier token** degrades — and does *not* degrade merely because the model's
priors, local n-gram statistics, or lexical-overlap shortcuts were disturbed.

### 6.1 Six admission criteria

A candidate instrument must pass all six. Most published memory benchmarks fail at least
two.

1. **Ground truth known by construction.** The answer is a deterministic function of an
   injected token, so a wrong answer is attributable to a specific cache entry, state slot
   or layer.
2. **No pattern-matching shortcut.** Two mechanisms, and you need both: a high-entropy
   value (random symbol, so priors cannot help) *and* no lexical overlap between query and
   needle (`[C]` NoLiMa's construction). One without the other leaves a shortcut open.
3. **Both a floor and a ceiling away from the operating point.** Chance level must be
   computable in closed form. Saturation and flooring are the same defect.
4. **A control arm that must fail.** Needle-removed, context-shuffled, and no-memory arms.
   `[C]` MemDelta (arXiv 2606.29914, Jun 2026) is the field discovering this the hard way:
   under one-variable-at-a-time control on LongMemEval-S, verbatim RAG matches full-context
   GPT-4o-mini (47.2% vs 49.8%, p = 0.34), swapping *only the embedding model* moves
   accuracy by +6.2pp (p = 0.004), and agent self-memory (42%) *underperforms* basic
   retrieval (47%). Most reported memory gains were confounds.
5. **Mechanism sensitivity.** The score must move when you break the mechanism you claim to
   study — see the calibration protocol below.
6. **Reported variance.** ≥3 seeds, CIs, and a stated minimum detectable effect
   `[C]` (2411.00640).

### 6.2 The calibration protocol — fault injection for metrics

**Systems bridge, and it is the strongest one in this note.** You do not trust an alert you
have never seen fire. You inject the fault and watch the pager. Do the same to every eval
before it is allowed to certify an arm:

| Injected fault | Cheap implementation | Expected response |
|---|---|---|
| Needle absent | delete the injected span | score → chance (proves the eval reads the needle, not the prior) |
| Needle unreachable | drop exactly the KV entries spanning the needle | large drop (proves the score depends on *those* entries) |
| Random capacity loss | evict p% of KV entries uniformly | monotone degradation curve; gives a sensitivity slope |
| Position corruption | re-pack cache entries so RoPE phase no longer matches position | large drop — and see `long-context-behavior.md §1`: post-rotation keys make this a silent corruption in production |
| Mechanism ablation | mask retrieval heads `[C]` (2404.15574) | targeted drop on retrieval tasks only, others intact |
| Distribution shuffle | shuffle haystack sentence order | *no* drop for a true retrieval task; a drop means you were measuring discourse structure |

An eval that survives all six injections without moving is measuring something other than
memory, and should be retired rather than reported. `[A]` **High confidence this protocol
is not standard practice anywhere in the KV-compression literature** — cheapest test that
would move it: search the last 12 months of eviction papers for any that report a
needle-removed control. I found none while writing this note.

### 6.3 The instruments that pass, ranked for this lab

**Tier 1 — differential, works at any scale, cannot saturate.**

**Oracle-diff KL against a full-cache reference.** Run the same prompt and seed twice: once
with the full cache, once under policy *P*. Log per-token KL divergence between output
distributions and attribute spikes to the entries *P* dropped. This is already the lab's
top-ranked backlog item (`open-problems-ranked.md §1`, P5·T5·E5). The evaluation-methodology
argument for it, which that note does not make, is decisive: **an accuracy metric requires a
competent model; a divergence metric requires only a reference run.** At 300M we can afford
the full-cache oracle on every probe, which nobody at 70B can — the small-scale rig is
*advantaged*, not merely cheaper. It has a natural null distribution (KL between two
independently-seeded runs of the *same* configuration), so "is this divergence real?" is a
computable question rather than a judgement call.

*Contested, and it is the risk of the whole approach.* Distributional divergence and task
accuracy can dissociate in both directions — a policy can shift the distribution without
changing any argmax, or flip a critical token with negligible average KL. The field has two
current answers and they disagree on method: `[C]` a fixed-contract diagnostic
(arXiv 2605.08234, May 2026), which argues task accuracy alone cannot tell you why a
selector worked, versus `[C]` error certificates via randomized design (arXiv 2607.21475,
Jul 2026), which argues randomization buys attribution rather than prediction. Do not adopt
either wholesale; report both the divergence and the downstream synthetic accuracy, and
report their correlation as a first-class result. If they do not correlate at our scale,
that is itself a finding worth publishing.

**Tier 2 — synthetic, generator-based, can genuinely fail, minutes to hours per arm.**

- **MQAR** `[C]` (Zoology, 2312.04927, Dec 2023). Multi-query associative recall: the
  state-capacity diagnostic, with published shape to calibrate against (attention solves it
  at model dimension 64 across lengths; gated convolutions need *d ≥ N*). *Caveat, and take
  it seriously:* `[C]` 2508.19029 (Aug 2025) revisits how much associative recall actually
  predicts, and `[C]` 2605.11196 (May 2026) reports an MQAR table in which softmax attention
  scores 0.15 at 8 pairs — a result so implausible it should be read as evidence that MQAR
  harness details (training steps, head dim, curriculum) dominate the numbers. Reproduce the
  published shape before trusting your own harness.
- **RULER S-NIAH / MK-NIAH, scaled to our training length** `[C]` (2404.06654). Generator,
  so lengths are set as multiples of *our* training window, not 128K.
- **NoLiMa-style associative needles** `[C]` (2502.05167). The construction, not the
  dataset: build needles whose surface form does not appear in the query. This is the single
  cheapest upgrade that turns a passing metric into a failing one.
- **Michelangelo-style latent-structure queries** `[C]` (2409.12640). The answer is a
  function of the *whole* context (list state after a sequence of updates; coreference
  across many rounds), so no single span can be matched. Structurally immune to defect (e)
  of §2, and synthetic, so it scales down. `[A]` medium confidence a 100M model can learn
  the Latent-List task at short lengths at all; cheapest test is a one-day pilot at 1K
  context, and a negative result kills a whole branch cheaply.
- **Flip-flop language modelling** `[C]` (2306.00946, Jun 2023). Sparse, sporadic attention
  glitches on a trivial state-tracking task — the tail-latency instinct applied to
  correctness. Rare-failure detection is precisely what mean accuracy hides, and it is the
  natural companion to worst-case aggregation `[C]` (2510.13334).
- **Perturbation twins** `[C]` (GSM-Symbolic, 2410.05229, as a *method*). Re-instantiate
  every item with a fresh symbol inventory; a memoriser's score drops and a recaller's does
  not.

**Tier 3 — real memory benchmarks, inference rig only, read for methodology.**

LoCoMo `[C]` 2402.17753, LongMemEval `[C]` 2410.10813 and its 2026 successor
`[C]` 2605.12493, MemoryAgentBench `[C]` 2507.05257, MemoryArena `[C]` 2602.16313, BEAM
`[C]` 2510.27246 (1M/10M-token scales), MemoryCD `[C]` 2603.25973, ImplicitMemBench
`[C]` 2604.08064, GateMem `[C]` 2606.18829, MemGym `[C]` 2605.20833. All require a competent
chat model; none is trainable at our scale. Their *diagnoses* transfer and are consistent
with everything above: `[C]` Anatomy of Agentic Memory (2602.19320, Feb 2026) — underscaled
benchmarks, misaligned metrics, backbone-dependent results, maintenance cost omitted;
`[C]` MemFail (2605.26667) and `[C]` MemTrace (2605.28732) — decompose the pipeline because
aggregate QA accuracy makes attribution impossible; `[C]` 2605.11325 — benchmarks score
answers rather than retrieval, so a system that dumps its whole store gets perfect recall
while hiding precision failure (read the methodological point, discount the headline: the
paper also ships the winning system); `[C]` Control-plane placement (2606.15903, Jun 2026) —
"production failures are predominantly forgetting failures rather than recall failures, yet
existing benchmarks measure only recall." That last sentence is the field's own statement of
this note's thesis.

**What we will *not* use as a memory metric: perplexity alone.** `[C]` ProLong
(2410.02660) explicitly rejects perplexity and bare NIAH as progress signals; `[C]`
alignment collapse under KV quantization (2606.09864) reports refusal-rate loss at
perplexity deltas small enough that PPL-only evaluation misses it entirely. Perplexity is
dominated by high-frequency local prediction; recall lives in a thin tail of tokens.
Zoology's framing is the quantitative version: **82% of a 2.1-point Pile perplexity gap sat
on tokens requiring associative recall** `[C]` (2312.04927) — which means perplexity does
carry the signal, but diluted ~1:5 against everything else. Report per-token loss *sliced to
the answer span*, never the corpus mean.

---

## 7. What is cheaply reproducible here — the arithmetic

Envelope, from `ASSUMPTIONS.md`: **≥62 GiB fast tier at ~200 GB/s** `[M]`, **20.9 TFLOP/s
bf16 GEMM at 8192³** `[M]`, single tensors **≥32 GiB hang or fault** `[M]`, single-device
only `[C]`, bf16 numerics **unproven** `[C]` — the Hardware Validation Gate has not run, so
nothing below is evidence yet.

Two derived numbers decide the whole eval budget. Both are arithmetic from `[M]` inputs, not
measurements.

**(a) Prefill-scored evals are cheap.** A 300M-parameter forward pass costs ≈ 2N = 6×10⁸
FLOP per token from the weight matmuls. At the measured 20.9 TFLOP/s peak the ideal is
~35,000 tok/s; `[A]` medium confidence that a realistic eval loop reaches ~20% of GEMM peak
(cheapest test: one timed forward-pass benchmark — hours of work, and it is a Hardware
Validation Gate item anyway), giving ~7,000 tok/s. A scaled RULER-style suite of 6 task
families × 200 items × 5 length buckets averaging 6K tokens is ~36M tokens ≈ **90 minutes
per arm**. Three seeds × six arms is a long weekend, and it is a *fraction* of the cost of
training those arms.

**(b) Generation-scored evals are ~35× worse, and this changes benchmark selection.**
Batch-1 decode is bandwidth-bound: 600 MB of bf16 weights at 200 GB/s is a **333 tok/s**
roofline before any KV traffic. Add our own KV arithmetic (`open-problems-ranked.md`: 24
KiB/token at L=24, H_kv=4, d_head=64) and at 16K context the cache is 384 MiB — read in full
for every decoded token — dropping the roofline to ~200 tok/s. **Design rule: score on a
single next-token distribution or a short span wherever possible.** This is why LongProc-style
long-output benchmarks and LLM-judged suites are out and why Michelangelo's Latent List
(one short answer per long context) is in.

**(c) The KV cache can be 100× the weights at our scale.** ~600 MB of weights against a
62 GiB fast tier `[M]` means we can hold ~2.7M tokens of cache — an inverted ratio that
production never sees. For evaluation this is a gift: full-cache oracles, exhaustive
occupancy sweeps, and length grids far beyond training length are all affordable. It is also
a confound (`open-problems-ranked.md` open question 11), and it must be declared in any
write-up.

**(d) The crossover nobody mentions.** Attention FLOPs per token ≈ 4·L·d_model·ctx, which
for L=24, d_model=1024 equals the weight FLOPs at **ctx ≈ 6,100 tokens**. Beyond ~6K, our
eval cost is dominated by context length, not model size — so the honest cost axis for a
long-context eval at this scale is *tokens of context × items*, not parameters.

**Reproducible here:** bits-per-byte on held-out shards; MQAR; RULER-family generators at
scaled lengths; NoLiMa-style construction; flip-flop; Michelangelo-style latent structure
(pilot required); perturbation twins; oracle-diff KL; occupancy sweeps; retrieval-head
ablation `[C]` (2404.15574) as an attribution probe.

**Not reproducible here, at any budget:** SWE-bench and successors, GAIA, τ/τ²-bench,
BrowseComp, OSWorld, GDPval, Terminal-Bench, HELMET, LongBench v2/Pro, LoCoMo/LongMemEval/
BEAM, and anything requiring an LLM judge (no local judge exists at a size we can run
alongside an experiment without distorting its cost model). `[C]` 2603.23749 is the right
citation for why agentic evaluation cost is itself a research problem.

**Standing methodological riders**, inherited and non-negotiable: muP `[C]` (2203.03466) so
that "arm A beat arm B" is not "arm A was better tuned"; two scales (~30M and ~300M) with
Spearman rank correlation reported, because `[C]` 2512.24503 (Dec 2025) finds proxy-model
rankings preserved only under specific LR/batch conditions and `[C]` ATLAS finds rank
instability across *length* regimes at frontier scale; and IsoFLOP fits via Approach 3 with
variable projection, not the parabola `[C]` (2603.22339, Mar 2026).

---

## Contested — do not let this note read as settled

1. **Synthetic versus realistic long-context evaluation.** Generator-based suites are
   contamination-immune and scale down but may measure an artificial skill; realistic suites
   are externally valid but static, contaminating, and unaffordable here. `[C]` 2505.19293
   and `[C]` 2409.06338 attack the realistic side; `[C]` 2410.02694 attacks the synthetic
   side by showing NIAH does not predict downstream. Unresolved.
2. **Is "effective context" a single number at all?** RULER's threshold rule says yes;
   `[C]` ATLAS's rank instability (2605.28079) says the induced ordering depends on the
   length grid.
3. **Does distributional divergence track task accuracy?** The premise of Tier 1.
   `[C]` 2605.08234 and `[C]` 2607.21475 propose incompatible attribution methods within
   three months of each other, and neither validates against the other.
4. **Is contamination detectable at all, post hoc?** `[C]` 2310.16789 and `[C]` 2404.02936
   say partially; `[C]` 2402.07841 finds MIAs near chance from 160M to 12B and attributes
   apparent successes to temporal shift; `[C]` 2402.02823 shows evasion is easy. The
   practical field has voted with its feet for rotation and generation over detection —
   which is a workaround, not an answer.
5. **Does associative recall predict anything downstream?** Zoology attributes 82% of a real
   perplexity gap to AR tokens `[C]` (2312.04927); `[C]` 2508.19029 revisits the claim and
   `[C]` 2605.11196's anomalous table suggests harness sensitivity. No source demonstrates
   that an MQAR capacity curve *quantitatively* predicts a downstream curve.
6. **Matched parameters or matched state?** `[C]` 2605.22791 argues state size is the correct
   control for recall comparisons; most of the literature matches parameters. The two give
   different rankings and papers rarely report both — which means half the published memory
   comparisons are not comparable to the other half.
7. **Are agentic benchmarks measuring capability or harness quality?** `[C]` 2507.02825's
   100%-relative-error finding and `[C]` 2510.11977's existence both suggest the scaffold is
   a first-order term. The field has not agreed how to report it.

---

## Open questions

Testable at 20M–300M params, one gfx1151 GPU, ≥62 GiB fast tier at ~200 GB/s `[M]`,
single-device, individual tensors under 32 GiB `[M]`.

1. **Does NIAH survive fault injection?** Run the §6.2 protocol against S-NIAH at 100M:
   remove the needle, drop the needle's KV entries, evict p% uniformly, shuffle the
   haystack. If S-NIAH scores barely move under uniform eviction while a NoLiMa-style
   associative variant collapses, the §2(e) claim is established and the lab has a
   publishable methodological result before training a single research arm. **Cheapest
   high-value experiment in this note.**
2. **Do oracle-diff KL and synthetic task accuracy correlate?** Same prompts, same seeds,
   same policies, both metrics. Report Spearman ρ and the seed-to-seed KL null. Decides
   whether Tier 1 can stand alone or must always be paired.
3. **What is the minimum detectable effect of our eval suite?** Power analysis
   `[C]` (2411.00640) at n = 100/500/2000 items and 3/5/10 seeds, using measured item-level
   variance. Pre-registration cards should carry an MDE field; right now they do not.
4. **Does the effective-versus-advertised gap reproduce below 300M?** Build a scaled RULER
   with the threshold anchored to our own short-context baseline. If the gap does not appear,
   this rig cannot study the phenomenon — a publishable negative about our own methodology.
   (Inherited from `long-context-behavior.md` Q4; restated here because it is fundamentally
   an eval-design question.)
5. **Can a 100M model learn a Michelangelo-style latent-list task at 1K context?** One-day
   pilot. A negative closes the strongest anti-pattern-match instrument at our scale and
   redirects effort to MQAR plus NoLiMa-style construction.
6. **Do continuous metrics beat discrete ones for arm ranking at 30M–300M?** `[C]` 2304.15004
   predicts exact-match manufactures cliffs; measure rank stability of arm orderings under
   accuracy versus answer-span log-likelihood at matched compute.
7. **Is a retrieval-head signature `[C]` (2404.15574) present at 300M, and does cache damage
   concentrate on it?** If retrieval heads exist at our scale, head-level attribution becomes
   available and the oracle-diff harness gains a second, independent localisation axis.
8. **What is the measured overlap between our training shards and every generated eval
   item?** Not a research question — a hygiene number that should exist before any result is
   reported, and it is cheap because we own both sides.

---

## Decision / Riskiest assumption / Next test

**Decision.** Build no accuracy benchmark. Build a *differential instrument* — full-cache
oracle diff with a seed-to-seed null — and gate every downstream synthetic eval behind the
§6.2 fault-injection calibration. Adopt only generator-open evals. Read the agentic
literature for methodology (ABC validity checklist, harness pinning, randomized held-out
grading) and run none of it on the training rig.

**Riskiest assumption.** That distributional divergence from a full-cache oracle tracks the
capability we care about. If a policy can move KL substantially without changing any answer
— or change answers without moving KL — the entire Tier 1 programme measures noise, and the
lab falls back to synthetic accuracy with all of its floor/ceiling problems at 300M. This is
strictly riskier than `ablation-scale-sufficient`, because it is upstream of it: a broken
instrument invalidates the scale question before the scale question can be asked. Open
question 2 is the test, and it should run early.

**Next test.** Open question 1 — fault-inject S-NIAH at 100M — but only after the Hardware
Validation Gate is green, since a bf16 numerics fault would present as exactly the kind of
silent recall degradation this note is designed to detect.

---

## Sources

Every arXiv id below was resolved against the live arXiv API on 2026-07-26; titles and dates
are the API's, not memory. Resolving an id proves the paper exists, not that it supports the
claim beside it.

**Long-context evaluation**
2404.06654 *RULER: What's the Real Context Size of Your Long-Context Language Models?* (Apr 2024) ·
2502.05167 *NoLiMa: Long-Context Evaluation Beyond Literal Matching* (Feb 2025, ICML 2025) ·
2410.02694 *HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly* (Oct 2024) ·
2406.10149 *BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack* (Jun 2024) ·
2409.12640 *Michelangelo: Long Context Evaluations Beyond Haystacks via Latent Structure Queries* (Sep 2024) ·
2501.05414 *LongProc: Benchmarking Long-Context Language Models on Long Procedural Generation* (Jan 2025) ·
2412.15204 *LongBench v2* (Dec 2024) ·
2601.02872 *LongBench Pro* (Jan 2026) ·
2605.28079 *ATLAS: All-round Testing of Long-context Abilities across Scales* (May 2026) ·
2607.08284 *Understanding Axes of Difficulty For Long Context Tasks Via PredicateLongBench* (Jul 2026) ·
2602.03587 *CL-bench: A Benchmark for Context Learning* (Feb 2026) ·
2505.19293 *100-LongBench: Are de facto Long-Context Benchmarks Literally Evaluating Long-Context Ability?* (May 2025) ·
2409.06338 *Retrieval Or Holistic Understanding? Dolce* (Sep 2024) ·
2411.05000 *Needle Threading* (Nov 2024) ·
2407.01370 *Summary of a Haystack* (Jul 2024) ·
2403.11802 *Counting-Stars* (Mar 2024) ·
2412.10319 *SCBench: A KV Cache-Centric Analysis of Long-Context Methods* (Dec 2024) ·
2503.17407 *A Comprehensive Survey on Long Context Language Modeling* (Mar 2025) ·
2410.02660 *How to Train Long-Context Language Models (Effectively)* (Oct 2024)

**Agentic evaluation and its validity crisis**
2310.06770 *SWE-bench* (Oct 2023) ·
2509.16941 *SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks?* (Sep 2025) ·
2311.12983 *GAIA: a benchmark for General AI Assistants* (Nov 2023) ·
2406.12045 *tau-bench: Tool-Agent-User Interaction in Real-World Domains* (Jun 2024) ·
2506.07982 *tau^2-Bench: Evaluating Conversational Agents in a Dual-Control Environment* (Jun 2025) ·
2504.12516 *BrowseComp* (Apr 2025) ·
2404.07972 *OSWorld* (Apr 2024) ·
2503.14499 *Measuring AI Ability to Complete Long Software Tasks* (Mar 2025) ·
2510.11977 *Holistic Agent Leaderboard* (Oct 2025) ·
2507.02825 *Establishing Best Practices for Building Rigorous Agentic Benchmarks* (Jul 2025) ·
2506.12286 *The SWE-Bench Illusion: When State-of-the-Art LLMs Remember Instead of Reason* (Jun 2025) ·
2606.07379 *Do Coding Agents Deceive Us? Capped Evaluation with Randomized Tests* (Jun 2026) ·
2605.26079 *Automated Benchmark Auditing for AI Agents and Large Language Models* (May 2026) ·
2603.23749 *Efficient Benchmarking of AI Agents* (Mar 2026) ·
2603.24755 *SlopCodeBench: How Coding Agents Degrade Over Long-Horizon Iterative Tasks* (Mar 2026) ·
2603.20807 *BenchBench: Benchmarking Automated Benchmark Generation* (Mar 2026)
— non-arXiv: Terminal-Bench 2.0 (tbench.ai), GDPval (openai.com/index/gdpval). Leaderboard
figures circulating on aggregator blogs are deliberately not cited; they are leads.

**Contamination**
2311.04850 *Rethinking Benchmark and Contamination for Language Models with Rephrased Samples* (Nov 2023) ·
2310.16789 *Detecting Pretraining Data from Large Language Models* (Min-K%, Oct 2023) ·
2404.02936 *Min-K%++* (Apr 2024) ·
2402.07841 *Do Membership Inference Attacks Work on Large Language Models?* (Feb 2024) ·
2402.02823 *Evading Data Contamination Detection for Language Models is (too) Easy* (Feb 2024) ·
2405.16281 *ConStat: Performance-Based Contamination Detection* (May 2024) ·
2406.18326 *PaCoST* (Jun 2024) ·
2510.27055 *Detecting Data Contamination in LLMs via In-Context Learning* (Oct 2025) ·
2605.24079 *TRACER: Fine-Grained Contamination Detection in Code LLMs* (May 2026) ·
2601.19334 *When Benchmarks Leak: Inference-Time Decontamination for LLMs* (Jan 2026) ·
2606.05241 *Search-Time Contamination in Deep Research Agents* (Jun 2026) ·
2406.19314 *LiveBench: A Challenging, Contamination-Limited LLM Benchmark* (Jun 2024) ·
2403.07974 *LiveCodeBench* (Mar 2024) ·
2412.15194 *MMLU-CF: A Contamination-free Benchmark* (Dec 2024) ·
2410.05229 *GSM-Symbolic* (Oct 2024) ·
2404.00699 *A Comprehensive Survey of Contamination Detection Methods in LLMs* (Mar 2024) ·
2406.04244 *Benchmark Data Contamination of Large Language Models: A Survey* (Jun 2024) ·
2502.14425 *A Survey on Data Contamination for Large Language Models* (Feb 2025) ·
2502.17521 *Recent Advances in LLM Benchmarks against Data Contamination: From Static to Dynamic* (Feb 2025)

**Measurement methodology and statistics**
2411.00640 *Adding Error Bars to Evals* (Nov 2024) ·
2406.10229 *Quantifying Variance in Evaluation Benchmarks* (Jun 2024) ·
2304.15004 *Are Emergent Abilities of Large Language Models a Mirage?* (Apr 2023) ·
2405.14782 *Lessons from the Trenches on Reproducible Evaluation of Language Models* (May 2024) ·
2402.14992 *tinyBenchmarks: evaluating LLMs with fewer examples* (Feb 2024) ·
2502.03461 *Do Large Language Model Benchmarks Test Reliability?* (Feb 2025) ·
2512.24503 *Can Small Training Runs Reliably Guide Data Curation?* (Dec 2025) ·
2603.22339 *Problems with Chinchilla Approach 2* (Mar 2026) ·
2203.03466 *Tensor Programs V* / muP (Mar 2022) ·
2403.17844 *Mechanistic Design and Scaling of Hybrid Architectures* (Mar 2024) ·
2309.14322 *Small-scale proxies for large-scale Transformer training instabilities* (Sep 2023)

**Memory diagnostics, mechanism probes, and memory benchmarks**
2312.04927 *Zoology: Measuring and Improving Recall in Efficient Language Models* (Dec 2023) ·
2508.19029 *Revisiting associative recall in modern recurrent models* (Aug 2025) ·
2605.11196 *Variational Linear Attention* (May 2026, cited for its MQAR table as a caution) ·
2504.19561 *Quantifying Memory Utilization with Effective State-Size* (Apr 2025) ·
2306.00946 *Exposing Attention Glitches with Flip-Flop Language Modeling* (Jun 2023) ·
2404.15574 *Retrieval Head Mechanistically Explains Long-Context Factuality* (Apr 2024) ·
2209.11895 *In-context Learning and Induction Heads* (Sep 2022) ·
2412.06464 *Gated Delta Networks* (Dec 2024) ·
2605.22791 *Gated DeltaNet-2: Decoupling Erase and Write* (May 2026) ·
2510.00231 *The Pitfalls of KV Cache Compression* (Sep 2025) ·
2510.13334 *Taming the Fragility of KV Cache Eviction in LLM Inference* (Oct 2025) ·
2605.08234 *When Does Value-Aware KV Eviction Help? A Fixed-Contract Diagnostic* (May 2026) ·
2607.21475 *Error Certificates for KV-Cache Eviction via Randomized Design* (Jul 2026) ·
2606.09864 *Alignment Collapse Under KV Cache Quantization* (Jun 2026) ·
2606.22528 *Governance Decay: How Context Compaction Silently Erases Safety Constraints* (Jun 2026) ·
2306.14048 *H2O: Heavy-Hitter Oracle* (Jun 2023) ·
2404.14469 *SnapKV* (Apr 2024) ·
2402.17753 *LoCoMo: Evaluating Very Long-Term Conversational Memory of LLM Agents* (Feb 2024) ·
2410.10813 *LongMemEval* (Oct 2024) ·
2605.12493 *LongMemEval-V2* (May 2026) ·
2507.05257 *MemoryAgentBench: Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions* (Jul 2025) ·
2602.16313 *MemoryArena* (Feb 2026) ·
2510.27246 *Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs* (Oct 2025) ·
2603.25973 *MemoryCD* (Mar 2026) ·
2604.08064 *ImplicitMemBench* (Apr 2026) ·
2606.18829 *GateMem* (Jun 2026) ·
2605.20833 *MemGym* (May 2026) ·
2606.29914 *MemDelta: Controlled Baselines and Hidden Confounds in Agent Memory Evaluation* (Jun 2026) ·
2602.19320 *Anatomy of Agentic Memory* (Feb 2026) ·
2605.26667 *MemFail* (May 2026) ·
2605.28732 *MemTrace* (May 2026) ·
2605.11325 *Structured Belief State and the First Precision-Aware Benchmark for LLM Memory Retrieval* (May 2026) ·
2606.15903 *Control-Plane Placement Shapes Forgetting* (Jun 2026) ·
2603.07670 *Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers* (Mar 2026)

**Local measurements and repo artifacts**
`ASSUMPTIONS.md` rows `gpu-fast-tier-size` (≥62 GiB at ~200 GB/s, `[M]` 2026-07-26),
`gemm-throughput-below-reference` (20.9 TFLOP/s bf16 at 8192³, `[M]`),
`large-tensor-fault-32gib` (`[M]`), `single-device-only` (`[C]`),
`bf16-numerics-unproven` (`[C]`, gate not run), `ablation-scale-sufficient` (`[A]`) ·
`notebook/uma-carveout-controls-fast-tier.md` (`[M]`, single run per arm) ·
`research/reference/CODE_MAP.md`: FlashInfer `decode.py:1239` (a miss is unrepresentable),
OLMo-core `trainer.py:1037`/`:1394` (instrumentation costs throughput) ·
`research/memory/open-problems-ranked.md` (§1 attribution harness; KV arithmetic; second rig) ·
`research/memory/constant-state-memory.md` (MQAR, S-NIAH and MK-NIAH tables) ·
`research/memory/long-context-behavior.md` (effective vs advertised; post-rotation keys) ·
`research/memory/agent-memory-systems.md §6` (the agent-side evaluation critique this note extends).
