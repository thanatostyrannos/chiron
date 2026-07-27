---
title: Building an eval you can trust â€” decision rules, noise floors, contamination, and calibration by fault injection
version: 1.0.0
date: 2026-07-26
track: E â€” Post-training and evaluation
prereqs: loss-and-optimization, scaling-laws-and-flops-budget, measuring-memory
recommended: memory-failure-modes
difficulty: medium-hard â€” the statistics are undergraduate; the discipline is not, and one section of arithmetic will tell you something about this lab's schedule you will not enjoy
time: 3â€“4 h reading and working the arithmetic; 2â€“3 h for the three exercises (Exercise C needs the GPU for ~5 min and the rest is CPU)
mirrors: research/notes/evaluation-landscape.md, research/memory/memory-failure-register.md
---

# Building an eval you can trust

**What this module is not.** It does not re-derive the oracle-diff instrument, the
per-token KL arithmetic, the `m_E Â· â€–vÌ„_E âˆ’ vÌ„_Kâ€–` decomposition, or the unpaired/paired
sample-size formulas. Those are `curriculum/measuring-memory.md` Â§3, they are a hard
prerequisite, and repeating them here would be padding. This module is the layer *around*
a metric: how you decide the item set is legitimate, how you decide the number moved, and
how you find out â€” before you trust it â€” whether the thing can move at all.

---

## 1. What this module settles

**One.** An eval is not a number; it is a decision procedure with five parts â€” statistic,
item set, noise model, threshold, stopping rule â€” and in ML practice four of the five are
usually unwritten, which is why "we improved 2 points" is not a claim anyone can check.
**Two.** At 20Mâ€“300M parameters on a corpus we own, contamination inverts from a detection
problem into a computable hygiene number, and the binding constraints become *statistical
power* (seed variance does not shrink when you add items, and on this machine three
training seeds at 300M costs weeks) and *environment* (a single environment variable on
gfx1151 changes the arithmetic that produces the metric `[M]`). **Three.** An eval you have
never seen fail is a decoration, the six-fault battery is the acceptance suite that turns
one into an instrument â€” and the battery as published has a hole: five of its six faults
test sensitivity and only one tests specificity, so a metric that moves for *everything*
passes it.

**Three findings this module's own exercises produced, folded in â€” one of which refutes what
I expected to find.**

- `[M]` **Fixed-`n` contamination detection has no good operating point on a self-similar
  corpus.** At `n = 13`, **35%** of genuinely held-out 40-word spans are flagged as
  contaminated â€” that is the detector's measured false-positive rate â€” while corrupting one
  word in eight drops detection of *actually copied* text from 1.000 to **0.000**. The
  evasion threshold is arithmetic, not luck: a corruption every `k` words defeats every
  `n â‰¥ k`. Â§7, Exercise A.
- `[M]` **I predicted this machine's environment noise would swamp a real effect. It does
  not â€” for a mean.** The largest nuisance-induced shift in the *mean* metric is
  `4.3 Ã— 10â»â¶` bits/token against a standard error of `9.1 Ã— 10â»Â³` bits/token: a ratio of
  1:2,100. The prediction was wrong and the reason it was wrong is the useful part: the
  perturbations are zero-mean, so they cancel in a mean and **do not cancel in a maximum**.
  The same bf16 change moves individual items by up to `3.5 Ã— 10â»Â³` bits â€” 280Ã— the mean
  shift. Â§7, Exercise C.
- `[M]` **fp32 forward passes on this stack are bit-reproducible within a process and not
  across processes.** Batch sizes 1, 8 and 64 gave bit-identical per-item values inside every
  process; two processes with *identical* environments gave means differing by
  `4.1 Ã— 10â»â¸` bits. Four processes, three distinct values. Â§7, Exercise C.

---

## 2. Theory in plain language

### 2.1 A metric is not a decision rule

Here is a metric: `p99 read latency = 214 ms`.

Here is a decision rule:

```
alert: ReadLatencyHigh
  expr:  histogram_quantile(0.99, read_latency_bucket) > 0.250
  for:   5m
  every: 30s
  labels: {severity: page}
  inhibit_if: RegionEvacuation
```

You would never confuse the two, and you would never ship the second without knowing its
false-page rate. The `for: 5m` clause exists precisely because you learned, painfully, that
evaluating a threshold every 30 s and paging on the first crossing produces a pager that
nobody reads. That clause is a *multiple-comparisons correction* invented by operators who
had never heard the term.

Now here is how the same object appears in ML:

> "best validation loss is 1.4697" â€” `training/nanogpt/README.md:51`

That sentence is doing the job of the whole alerting rule. It is a pass/fail threshold for
this lab's Hardware Validation Gate. And it is a Monte Carlo mean over 200 randomly drawn
batches (`training/nanogpt/train.py:216`), reported to four decimal places, with no
standard error, over an item set that is *redrawn on every call*, from a global RNG stream
that training itself also consumes. Every one of those is a defect if you treat 1.4697 as a
threshold. None of them is a defect if you treat it as a sanity number â€” which is what its
author intended and says in the surrounding text.

So write the five parts down, every time:

| Part | What it is | The production analogue | The usual ML omission |
|---|---|---|---|
| **Statistic** `S` | the number computed from one run | `histogram_quantile(0.99, â€¦)` | usually present |
| **Item set** `I` | the fixed, enumerable, hash-identified set of things scored | the target selector | resampled, truncated, or undocumented |
| **Noise model** `N` | the distribution of `S` when nothing has changed | your baseline dashboards | absent, or assumed to be zero |
| **Threshold** `Ï„` | the value of `S âˆ’ S_baseline` that changes the decision | `> 0.250` | chosen after seeing the result |
| **Stopping rule** | when you are allowed to look and when you must stop | `for: 5m`, evaluation interval | unstated; you look whenever a checkpoint lands |

> **Systems bridge, and it is the spine of this module.** You already own all five. An SLO
> is a statistic plus a threshold; an SLI definition is an item set; a burn-rate alert is a
> stopping rule; and your capacity dashboards are an empirical noise model.
>
> **Where it breaks â€” three places, in increasing order of how much they cost.**
>
> **(a) Your noise model is refreshed continuously and for free; ours costs a training
> run.** You have millions of samples per hour of "nothing changed." We get one number per
> multi-day run, and the honest number of independent draws is the number of *seeds*, not
> the number of items. Â§3.2 makes this exact, and Â§3.6 prices it.
>
> **(b) Your null is stationary across deploys; ours moves when the toolchain moves.** A
> latency histogram means the same thing after a kernel upgrade. On gfx1151, whether
> `hipBLASLt` is configured changes the relative error of a long bf16 reduction by ~2.8Ã—
> `[M]` (`ASSUMPTIONS.md â†’ hipblaslt-config`) â€” the *arithmetic that computes the metric*
> is different. Two arms measured in shells with different environment variables are not
> comparable, and nothing in the output says so.
>
> **(c) You can add hosts; we cannot add seeds cheaply.** In production the cheap axis
> (more requests, more hosts) is also the informative one. Here the cheap axis (more eval
> items) hits a wall that the expensive axis (more seeds) sets. That inversion is the single
> most consequential thing in this module.

### 2.2 Three ways an eval becomes a decoration, and a fourth nobody names

An eval is a decoration when its output cannot change in response to the thing you claim to
be studying. Four mechanisms:

1. **Saturation.** Everyone scores 99%. `[C]` RULER (2404.06654) evaluated 17 models all
   claiming â‰¥32K context; nearly all are near-perfect on vanilla needle-in-a-haystack. A
   benchmark everyone passes ranks nothing.
2. **Flooring.** Everyone scores chance. This is *our* risk, not the field's: at 20Mâ€“300M a
   model is at chance on essentially every knowledge or agentic benchmark, and a metric
   pinned at zero has exactly the same dynamic-range problem as one pinned at one hundred
   `[C]` (evaluation-landscape Â§4 states this unhedged; Â§5 gives the mechanism via
   `[C]` 2406.10229).
3. **Adverse selection.** The pass criterion is positively correlated with what the
   mechanism under test preserves. The canonical case is NIAH versus attention-mass
   eviction, and it is treated in full in `curriculum/memory-failure-modes.md` Â§2.5 and
   `curriculum/measuring-memory.md` Â§2.6 with the closed form `s* = âˆ’ln(b)`. Do not
   re-derive it; know that it exists and that it is the reason this lab does not build
   accuracy benchmarks on salient needles.
4. **A denominator you do not control.** This one is under-discussed and it is the easiest
   to commit. If the number of items scored varies between runs, `S` is a mean over a
   different population each time, its standard error changes, and the difference between
   two runs contains a term you never measured. Two production examples, both in the
   reference library:
   - `training/nanogpt/train.py:222` â€” `get_batch` is called *inside* the eval loop, so the
     200 eval batches are freshly sampled with replacement on every call, from the global
     torch RNG seeded once at `train.py:106`. Two evaluations of the same checkpoint score
     different items. Worse: because training also draws from that stream, a run with a
     different number of steps before the eval sees a different eval set.
   - `training/olmo-core/src/olmo_core/train/callbacks/evaluator_callback.py:155` â€” the eval
     loop `break`s when a `Duration` budget expires. If the budget binds, `n` is set by
     wall-clock. OLMo-core is aware of this and ships the fix as a documented flag
     (`training/olmo-core/src/olmo_core/eval/evaluator.py:23`, `:41`) â€” read that docstring; it is a better statement of the
     hazard than most papers manage.

### 2.3 Contamination, and why it inverts at our scale

**Two problems share one word.** *Training-time* contamination: benchmark items were in the
pretraining data. *Runtime* contamination: the system fetches the answer during evaluation
`[C]` (2606.05241, Jun 2026 â€” deep-research agents retrieving the benchmark's own published
answers at run time; nothing about training-data hygiene protects you).

**The detection literature's honest state is: weak.** Membership-inference attacks "barely
outperform random guessing" on Pile-trained models from 160M to 12B, and the apparent
successes are attributable to temporal distribution shift between the member and non-member
sets `[C]` (2402.07841). Evasion of the standard detectors is easy `[C]` (2402.02823).
Paraphrase and translation walk straight through 13-gram overlap, and 8â€“18% of HumanEval
already appears in RedPajama-1T and StarCoder-Data `[C]` (2311.04850). The field has
responded not by detecting better but by *rotating* (`[C]` LiveBench 2406.19314,
LiveCodeBench 2403.07974) and by *generating* (`[C]` GSM-Symbolic 2410.05229 â€” re-instantiate
the same problems with fresh names and numbers, and scores move, which means the original
score was partly measuring surface familiarity).

> **Systems bridge.** Contamination is benchmarking against a warm cache you forgot to drop,
> and membership inference is trying to prove after the fact, from timing alone, that the
> cache was cold. `[C]` 2402.07841 is the measurement that says you cannot.
> **Where it breaks:** `echo 3 > /proc/sys/vm/drop_caches` exists. There is no
> `drop_caches` for a 15T-token pretraining run.
>
> **And where the break un-breaks, for us specifically.** At 20Mâ€“300M on 0.5â€“5B
> self-selected tokens, *we own the corpus*. We do have `drop_caches`. So the question stops
> being "can we detect leakage" and becomes "what is our certification policy" â€” an
> intersection of two sets we both possess, plus a written statement of what counts as a
> match. That is a hygiene number, computed once per corpus, not a research problem.

**Which leaves our three actual risks, and they are different from the field's.**

1. **Under-capability masquerading as a clean result.** A null at 300M is usually "the model
   cannot do the task," not "the intervention did nothing."
2. **Harness leakage.** If your synthetic eval generator draws its symbols from the same
   inventory that produced the training shards, the model can *reconstruct* the needle
   rather than recall it. The fix is a held-out symbol inventory, and it must be checked
   rather than assumed.
3. **A false-positive floor on the detector you do use.** This is the part Â§3.5 and
   Exercise A make quantitative, and it is the interesting one: 13-gram overlap is a
   *decision rule with an unstated false-positive rate*, and its FPR is a function of your
   corpus size and the entropy of the text you are matching â€” not a universal constant.

### 2.4 Where the noise actually comes from

The standard small-scale eval mistake, stated as a sentence you have said in another
context: *"I ran 10,000 requests against one host and the confidence interval was tiny."*

You know why that is wrong. You measured the request distribution, not the fleet. Host-level
variance â€” this host's NUMA layout, this host's noisy neighbour, this host's firmware â€” does
not shrink when you send more requests to the same host.

Same structure here, with the roles renamed:

| Production | Eval | Shrinks with more items? |
|---|---|---|
| request-to-request variance | **item variance** `ÏƒÂ²_item` | yes, as `1/n` |
| host-to-host variance | **seed variance** `ÏƒÂ²_seed` | **no** |
| kernel/firmware/config skew | **environment variance** `ÏƒÂ²_env` | **no** â€” and it is not random, it is a bias |

`ÏƒÂ²_seed` is generated by initialization, data order, and any nondeterminism in the training
run. It is the variance of the *arm*, not of the measurement, and it is the one the â‰¥3-seed
house rule exists to estimate.

`ÏƒÂ²_env` is the one this machine forces us to take seriously, because three of its terms are
already measured here and all three are invisible in the output:

- `[M]` `hipBLASLt` configured versus not changes the relative error of a length-1M bf16
  weighted sum by **2.8Ã—** (2.01e-3 â†’ 5.60e-3 against an fp64 reference, three seeds, fresh
  processes; `ASSUMPTIONS.md â†’ hipblaslt-config`). Throughput moved only 12%; the *numerics*
  moved 2.8Ã—. It is a correctness control wearing a performance control's clothes.
- `[M]` `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` changes which attention kernel runs
  (147.2 â†’ 6.6 retained bytes per `TÂ²`), and `torch.backends.cuda.flash_sdp_enabled()`
  returns `True` either way â€” the API reports what is *permitted*, not what *ran*
  (`ASSUMPTIONS.md â†’ sdpa-is-memory-efficient`). The only honest signal is a `UserWarning`
  on stderr.
- `[M]` `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction` is **inert** on
  this stack â€” toggling it changes the result by exactly zero bits
  (`ASSUMPTIONS.md â†’ bf16-reduced-precision-knob-works`, refuted). An axis you might
  reasonably have controlled for does nothing, which is its own hazard: you will believe you
  controlled something.

Add the standing one: `[C]` bf16 numerics on gfx1151 are **unproven** â€” the Hardware
Validation Gate has not run. Until it does, every bf16 number from this machine carries an
unbounded environment term.

**The operational consequence, stated as a rule.** *The environment is part of the arm's
identity.* Two runs in shells with different variables are two different arms, not two
seeds of one. Â§4.2 turns that into a required field on the pre-registration card.

**And the caveat I earned by measuring it, which you should read before over-applying the
rule.** `[M]` Exercise C (Â§7) finds these terms are *tiny in a mean* â€” the largest shift is
`4.3 Ã— 10â»â¶` bits/token against a standard error of `9.1 Ã— 10â»Â³`. They are not tiny in a
*maximum*: the same change moves the worst individual item by `3.5 Ã— 10â»Â³` bits, 280Ã— the
mean shift. So `ÏƒÂ²_env` is real, it is zero-mean, and whether it matters is decided entirely
by your **aggregation function** â€” which is a decision-rule parameter (Â§3.3), not a
reporting preference. Mean-aggregated metrics are robust to it; worst-case, threshold-count,
and rank-based metrics are not. That is an independent, mechanical reason why the
mean-versus-worst-case aggregation question `[C]` (2510.13334) changes conclusions, and it
is not the reason that paper gives.

### 2.5 Effect size, the noise floor, and the winner's curse

`curriculum/measuring-memory.md` Â§3.4 derives the sample size and the minimum detectable
effect; assume it. Two things it does not cover, and both change how you *read* a result.

**Exaggeration (Type M error).** When power is low, the estimates that clear the
significance threshold are, on average, much larger than the truth â€” because only the
lucky-large draws clear it. At 20% power the average significant estimate overstates the
true effect by about **2.3Ã—** (Â§3.4 computes this in closed form). This is not a subtlety.
It means a underpowered study that reports "a significant 6-point gain" is consistent with a
true effect of 2.6 points, and the *published* number is biased upward by construction, not
by misconduct.

**Sign error (Type S error).** At very low power a nontrivial fraction of significant
results point the *wrong way*. At 10% power it is ~4.5%.

> **Systems bridge.** This is survivorship in your incident record. The outages you remember
> are the ones that paged, the ones that paged are the large excursions, and your intuition
> about "how bad a bad day is" is therefore biased upward by exactly this mechanism.
> **Where it breaks:** you eventually see the whole latency distribution, so the bias is
> correctable by looking at the histogram instead of the incident log. Here there is no
> histogram. You get one draw per multi-week run, and the file drawer is the only place the
> non-significant draws exist.

**And the stopping rule, which is the part everyone commits.** Evaluate a checkpoint every
1,000 steps, stop when it looks good, report the p-value from that look. Under a true null,
testing at *k* equally spaced looks with a nominal two-sided Î± = 0.05 gives an actual
false-positive rate of roughly 0.083 at 2 looks, 0.142 at 5, 0.193 at 10, and 0.246 at 20
`[C]` (the classical repeated-significance-test result, Armitage, McPherson & Rowe 1969,
*JRSS-A* 132(2):235â€“244 â€” not an arXiv paper; Exercise B reproduces the table). At twenty
looks, one comparison in four fires under the null.

You already defend against this. `for: 5m` is an alpha-spending function. The break is that
`for: 5m` works because the signal is stationary and you get a fresh independent sample
every 30 s; a training run gives you one draw per run and there is no duration clause that
can help.

### 2.6 Fault injection as calibration â€” and the hole in the battery

The rule, from `research/notes/evaluation-landscape.md` Â§6.2 and taught in
`curriculum/measuring-memory.md` Â§2.6: **you do not trust an alert you have never seen
fire.** Inject the fault, watch the pager. Do it to the *metric*, not the system.

The battery, drawn from `research/notes/evaluation-landscape.md` Â§6.2, with **fault zero**
added by `curriculum/memory-failure-modes.md` Â§4.2(3), plus two columns this module adds:

| # | Injected fault | Class | Expected response | What a *flat* response means |
|---|---|---|---|---|
| 0 | **Measure the target's salience rank** `R` in the attention distribution | precondition | `R > Ï*Â·T` for the budget `Ï*` you want power against | your target is un-evictable; the eval cannot fail by construction |
| 1 | **Needle absent** â€” delete the injected span | sensitivity | score â†’ chance, computed in closed form | the eval reads the model's prior, not the needle |
| 2 | **Needle unreachable** â€” drop exactly the KV entries spanning it | sensitivity | large drop (but see below) | the score does not depend on *those* entries |
| 3 | **Random capacity loss** â€” evict `p%` uniformly | sensitivity | monotone degradation; gives a slope | the eval is insensitive to capacity at all |
| 4 | **Position corruption** â€” re-pack entries so RoPE phase mismatches position | sensitivity | large drop | the eval ignores order; check that is intended |
| 5 | **Mechanism ablation** â€” mask retrieval heads `[C]` (2404.15574) | sensitivity | targeted drop on retrieval tasks only | either no retrieval heads at this scale, or the eval is not retrieval |
| 6 | **Distribution shuffle** â€” permute haystack sentence order | **specificity** | ***no*** drop | (here a *drop* is the failure: you were measuring discourse structure) |

**Now the hole.** Count the classes: one precondition, **five sensitivity tests** â€” "does the
number move when I break something?" â€” and **exactly one specificity test**. A metric that
moves for *everything* passes five of the six faults and fails only the last. That metric is
useless: it cannot attribute, because every intervention lights it up. The battery is a
smoke test for a dead sensor, not a calibration.

> **Systems bridge, and the break is the teaching.** This is chaos engineering, and the
> discipline transfers. **First break** (from `measuring-memory.md` Â§2.6, restated because
> it governs what follows): in production the *system* is under test and a pass means
> nothing broke; here the *metric* is under test and a pass means the number **moved**. A
> flat line is the failure. **Second break, which is this module's addition:** a chaos
> experiment needs no negative control, because "we injected nothing and nothing happened"
> is not informative about a system. A *calibration* experiment needs negative controls
> desperately, because an instrument that responds to everything has no resolving power.
> You know this from a different room: a smoke detector that alarms on toast is not a
> sensitive smoke detector, it is a broken one, and no amount of setting fires will reveal
> that. You have to burn the toast on purpose.

**Three negative controls to add**, none of which appears in the published battery:

- **N1 â€” irrelevant-span deletion.** Delete a *different* span of the same length and
  entropy, far from the needle. Expected: no drop. A drop means the metric is length- or
  perplexity-sensitive rather than content-sensitive.
- **N2 â€” semantically-null perturbation.** Re-instantiate the item with a fresh symbol
  inventory (`[C]` GSM-Symbolic 2410.05229 as a *method*, not a dataset). Expected: no drop
  for a recaller, a drop for a memoriser. This is simultaneously a specificity control and a
  contamination probe, which is why it is the best value in the list.
- **N3 â€” a null intervention on the *system*.** Run the "policy" arm with a policy that
  evicts nothing (`budget = âˆž`). Expected: bit-identical to the oracle. If it is not, your
  harness has a side effect and every number in the study is suspect. This is the cheapest
  test in the whole module and it catches the class of bug that produces spectacular
  results.

**And one calibration the battery omits entirely: a known mass.** You calibrate a scale by
weighing something whose weight you already know. The eval equivalent is a **known-ordering
check**: score two models whose relative quality you are certain of for reasons *external to
the eval* â€” the same architecture at 2Ã— the tokens, or a checkpoint at step 1,000 versus step
10,000. If the eval does not order them correctly with a margin larger than its noise floor,
it cannot order anything you do not already know. Nothing in the fault battery tests this,
because every fault in the battery is a *degradation*; a monotone-degradation response is
compatible with an eval that only measures "is this model broken."

### 2.7 Where the honest limit is, on this machine

Stated plainly so Â§6 does not have to hedge:

- **Multi-seed at 300M is not affordable here.** Â§3.6 does the arithmetic. It is the single
  largest constraint this module surfaces, and it is a schedule problem, not a statistics
  problem.
- **Anything needing an LLM judge is out.** No local judge exists at a size we can run
  alongside an experiment without distorting its cost model
  (`research/notes/evaluation-landscape.md` Â§7).
- **The whole agentic evaluation stack is out at our scale**, for capability reasons, not
  cost reasons â€” scores would be floored at zero `[C]` (evaluation-landscape Â§4).
- **Distributed evaluation is out.** Collectives are incomplete on gfx1151 `[C]`
  (`ASSUMPTIONS.md â†’ single-device-only`), so `MeanMetric.compute`'s `all_reduce_value`
  path (`training/olmo-core/src/olmo_core/eval/metrics.py:81`) is code we read and cannot
  run. Read it for the design, not for the throughput.
- **No number from this machine is evidence yet**, because the Hardware Validation Gate has
  not run and bf16 numerics are unproven `[C]`. Every measurement below, including this
  module's own, is an instrument shakedown.

---

## 3. The math that actually matters

### 3.1 Symbols

| Symbol | Reads as |
|---|---|
| `S` | the statistic â€” the number an eval run produces |
| `I` | the item set: a fixed, enumerable, hash-identified collection of scored things |
| `n` | number of items in `I` |
| `k` | number of seeds (independent training runs per arm) |
| `x_{s,i}` | the per-item score for seed `s`, item `i` |
| `ÏƒÂ²_item` | variance of `x` across items within a seed |
| `ÏƒÂ²_seed` | variance of the seed-level mean across seeds |
| `ICC` | intraclass correlation, `ÏƒÂ²_seed / (ÏƒÂ²_seed + ÏƒÂ²_item)` |
| `d` | true effect size (difference between arms, in the metric's units) |
| `dÌ‚` | the estimate of `d` from one experiment |
| `Î¼` | true effect expressed in standard-error units, `d / SE` |
| `z_{Î±/2}` | 1.95996 at Î± = 0.05 two-sided |
| `z_Î²` | 0.84162 at 80% power |
| `Ï†`, `Î¦` | standard normal density and cumulative distribution |
| `D` | corpus size in tokens |
| `H` | entropy per token of the text being matched, in **bits** |
| `n*` | the n-gram length at which one accidental match is expected |
| `Ï*` | the KV budget fraction an eval needs power against |
| `T` | context length in tokens |

### 3.2 Variance components: the seed wall

Model each per-item score as a grand mean plus a seed effect plus an item effect:

```
x_{s,i} = Î¼_arm + u_s + e_{s,i}
```

- `Î¼_arm` â€” the arm's true mean.
- `u_s` â€” the seed's deviation. Same for every item in that seed. `Var(u) = ÏƒÂ²_seed`.
- `e_{s,i}` â€” the item's deviation within that seed. `Var(e) = ÏƒÂ²_item`.

The grand mean over `k` seeds and `n` items is `xÌ„ = (1/kn) Î£_s Î£_i x_{s,i}`. Because `u_s`
is shared by all `n` items of seed `s`, it does not average away across items:

```
Var(xÌ„) = ÏƒÂ²_seed / k  +  ÏƒÂ²_item / (kÂ·n)
```

Take `n â†’ âˆž`:

```
SE_floor = Ïƒ_seed / âˆšk
```

**That is the seed wall.** No number of eval items reduces it. The only lever is `k`, and
`k` is training runs.

**How many items are worth buying?** Solve for the `n` at which the standard error is within
10% of the wall:

```
ÏƒÂ²_seed + ÏƒÂ²_item/n  â‰¤  1.21 Â· ÏƒÂ²_seed
n  â‰¥  ÏƒÂ²_item / (0.21 Â· ÏƒÂ²_seed)  =  4.76 Â· (ÏƒÂ²_item / ÏƒÂ²_seed)
```

**Items are worth buying up to about five times the variance ratio, and worthless after
that.** Worked, with the ratio on the left:

| `ÏƒÂ²_item / ÏƒÂ²_seed` | items worth buying | SE there | SE at 10Ã— the items |
|---|---|---|---|
| 10 | 48 | 1.100 Â· Ïƒ_seed/âˆšk | 1.010 Â· Ïƒ_seed/âˆšk |
| 100 | 476 | 1.100 Â· Ïƒ_seed/âˆšk | 1.010 Â· Ïƒ_seed/âˆšk |
| 1,000 | 4,762 | 1.100 Â· Ïƒ_seed/âˆšk | 1.010 Â· Ïƒ_seed/âˆšk |
| 10,000 | 47,620 | 1.100 Â· Ïƒ_seed/âˆšk | 1.010 Â· Ïƒ_seed/âˆšk |

Read the last two columns: buying **ten times** the items past that point moves the standard
error from 1.100 to 1.010 times the wall â€” an 8% improvement for a 10Ã— cost, and it can never
exceed 10% no matter what you spend. The right-hand column is the same number in every row
because the wall does not care how many items you bought; that invariance is the point.

You cannot compute the left-hand column without measuring both variances, which requires
`k â‰¥ 2` â€” so **the first thing an eval plan must produce is a variance decomposition, not a
score.** Exercise C measures the item half on this machine; the seed half requires trained
arms and is the lab's next cheap experiment.

**The design move that actually buys something: pair on the seed.** Run both arms at the
*same* seeds â€” same initialization, same data order, same everything but the intervention.
Then in the difference `Î”_s = xÌ„_{A,s} âˆ’ xÌ„_{B,s}`, the seed main effect `u_s` cancels:

```
Var(Î”Ì„) = 2Â·ÏƒÂ²_item/(kÂ·n)  +  2Â·ÏƒÂ²_{seedÃ—arm}/k
```

- `ÏƒÂ²_{seedÃ—arm}` â€” the *interaction*: how much the arm's effect itself varies by seed.

Pairing removes the main effect, not the interaction. That is the honest statement, and the
interaction is not a nuisance â€” "arm A is more seed-sensitive than arm B" is a finding, and
it is exactly the kind of finding a hybrid-ratio or eviction-policy study should be looking
for. Report it rather than averaging it away.

**Clustering, one line, because it interacts with the above.** If your `n` items are really
`n/m` prompts of `m` tokens each and the tokens within a prompt are correlated at `Ï_intra`,
the effective sample size is

```
n_eff = n / (1 + (m âˆ’ 1)Â·Ï_intra)
```

The denominator is the *design effect*. At `m = 1,000` tokens per prompt and
`Ï_intra = 0.05`, the design effect is 50.95 â€” a 100,000-token eval is worth about 1,963
independent observations. `curriculum/measuring-memory.md` Â§3.4 makes the same point for
oracle-diff KL and says "state the effective n"; this is the formula that lets you.

### 3.3 The stopping rule, and alpha under repeated looks

Two different multiplicity problems, and only one has a closed form.

**Forking paths â€” independent metrics.** If you compute `m` metrics that are mutually
independent and report the best one, the probability that at least one clears a nominal Î±
under the null is

```
P(any false positive) = 1 âˆ’ (1 âˆ’ Î±)^m
```

At Î± = 0.05: `m = 5` gives 0.226; `m = 10` gives 0.401; `m = 20` gives 0.642. Real eval
suites are not independent, so the true rate is lower â€” but it is bounded below by Î± and
above by `min(1, mÂ·Î±)`, and "we looked at six metrics and one moved" is not evidence.

**Peeking â€” repeated looks at accumulating data.** No such closed form exists, because
partial sums of the same data are strongly correlated: the look at `n = 500` and the look at
`n = 1,000` share 500 observations. The classical numerical answer, for repeated two-sided
tests at nominal Î± = 0.05 on accumulating normal data:

| looks `k` | actual false-positive rate |
|---|---|
| 1 | 0.050 |
| 2 | 0.083 |
| 3 | 0.107 |
| 5 | 0.142 |
| 10 | 0.193 |
| 20 | 0.246 |
| âˆž | 1.000 |

`[C]` Armitage, McPherson & Rowe, "Repeated significance tests on accumulating data," *JRSS
Series A* 132(2):235â€“244, 1969. (Not an arXiv paper â€” cited by title and journal. Exercise B
reproduces these numbers by simulation, which is how you check that I have quoted them
correctly.)

The `k = âˆž` row is the one to internalize: **a rule that says "stop when significant" has a
false-positive rate of 1** against a null, given unlimited looks. This is the formal version
of "if you stare at the dashboard long enough, every deploy looks like a regression."

**The three legitimate fixes**, in increasing order of cost to you:

1. **Pre-register a single look.** One evaluation, at a pre-committed step count. Free, and
   it is what the house rule already implies.
2. **Spend alpha.** Group-sequential boundaries â€” Pocock (constant, easy, spends early),
   O'Brienâ€“Fleming (boundary `âˆ 1/âˆšt`, conservative early, near-nominal at the end). If you
   want early stopping for cost reasons, this is the correct machinery and it is thirty
   lines.
3. **Report the trajectory, not a test.** Plot the metric against step for every seed and
   argue from the shape. This is what you actually do with a latency graph, it is honest, and
   it forfeits any p-value â€” which is fine, because at `k = 3` a p-value was never going to
   be worth much.

### 3.4 Type M and Type S: what a significant result at low power means

Let `dÌ‚ ~ Normal(d, SEÂ²)` and let `Î¼ = d / SE` be the true effect in standard-error units.
Declare significance when `|dÌ‚| > zÂ·SE` with `z = 1.95996`.

**Power** â€” the probability of declaring significance:

```
power(Î¼) = Î¦(Î¼ âˆ’ z) + Î¦(âˆ’Î¼ âˆ’ z)
```

**Type S rate** â€” given significance, the probability the estimate has the wrong sign:

```
P(sign wrong | significant) = Î¦(âˆ’Î¼ âˆ’ z) / power(Î¼)
```

**Type M, the exaggeration ratio** â€” given significance, the expected magnitude of the
estimate divided by the truth. For `Z ~ Normal(Î¼, 1)`:

```
E[|Z| | |Z| > z]  =  [ Ï†(z âˆ’ Î¼) + Ï†(z + Î¼) + Î¼Â·(Î¦(Î¼ âˆ’ z) âˆ’ Î¦(âˆ’Î¼ âˆ’ z)) ] / power(Î¼)

exaggeration(Î¼) = E[|Z| | |Z| > z] / Î¼
```

Every symbol: `Ï†` is the standard normal density, `Î¦` its CDF, `z` the critical value, `Î¼`
the true effect in SE units. The numerator's first two terms are the density mass piled just
outside each tail boundary; the third is the truth times the probability of landing in a
tail.

Worked, by hand, at 20% power (`Î¼ = 1.116`):

```
Ï†(1.960 âˆ’ 1.116) = Ï†(0.844)  = 0.27940
Ï†(1.960 + 1.116) = Ï†(3.076)  = 0.00353
Î¦(1.116 âˆ’ 1.960) = Î¦(âˆ’0.844) = 0.19930
Î¦(âˆ’1.116 âˆ’ 1.960) = Î¦(âˆ’3.076) = 0.00105
power = 0.19930 + 0.00105 = 0.20035
numerator = 0.27940 + 0.00353 + 1.116Â·(0.19930 âˆ’ 0.00105) = 0.50418
E[|Z| | sig] = 0.50418 / 0.20035 = 2.5165
exaggeration = 2.5165 / 1.116 = 2.25
Type S = 0.00105 / 0.20035 = 0.52%
```

The whole table:

| power | `Î¼` | exaggeration | Type S |
|---|---|---|---|
| 0.10 | 0.652 | **3.71Ã—** | 4.5% |
| 0.20 | 1.116 | **2.25Ã—** | 0.52% |
| 0.35 | 1.523 | 1.68Ã— | 0.06% |
| 0.50 | 1.960 | 1.40Ã— | 0.005% |
| 0.80 | 2.802 | **1.13Ã—** | ~0% |
| 0.95 | 3.605 | 1.03Ã— | ~0% |

**Read the first and second rows together with Â§3.6.** If our seed budget puts us at 10â€“20%
power â€” and it will, at 300M â€” then a significant result is expected to overstate the truth
by 2â€“4Ã—. That is not a reason to skip the experiment. It is a reason to (a) report the
effect size with its interval rather than the point estimate, (b) treat any *large* effect
found at low power as a hypothesis rather than a measurement, and (c) prefer the
continuous, per-token metrics that raise `n` by three orders of magnitude over the
exact-match metrics that do not `[C]` (2304.15004 supplies the mechanism: discontinuous
metrics manufacture cliffs that continuous ones do not).

### 3.5 When is an n-gram match an accident?

The 13-gram convention is inherited from GPT-3 and is used as though it were a constant. It
is not; it is the solution to an equation with two free parameters, and both of them move at
our scale.

A specific n-gram `g` occurs in a corpus of `D` tokens an expected

```
E[count(g)] = (D âˆ’ n + 1) Â· p(g)  â‰ˆ  D Â· p(g)
```

number of times. For a *typical* n-gram â€” one in the typical set of the source â€” the
asymptotic equipartition property gives `p(g) â‰ˆ 2^{âˆ’nÂ·H}`, where `H` is the entropy per
token of the source, in bits. So

```
E[count(g)] â‰ˆ D Â· 2^{âˆ’nÂ·H}
```

Set that to 1 and solve for the length at which one accidental match is expected:

```
n* = logâ‚‚(D) / H
```

**Every symbol:** `D` is the corpus size in tokens, `H` is bits of entropy per token of the
text you are matching, `n*` is the n-gram length at which chance alone produces one match.
Above `n*`, matches are exponentially unlikely by chance; below it, they are expected.

Now put numbers in, and notice that `n*` is not a constant:

| Setting | `D` | `H` (bits/token) | `n*` |
|---|---|---|---|
| GPT-3-scale corpus, ordinary prose | 3Â·10Â¹Â¹ | 3.0 | **12.7** |
| Our corpus, ordinary prose | 10â¹ | 2.0 | **15.0** |
| Our corpus, boilerplate ("as shown in the table below") | 10â¹ | 0.7 | **42.7** |
| Our corpus, high-entropy synthetic eval items | 10â¹ | 8.0 | **3.7** |
| Our corpus, uniform random symbols from a 4,096 inventory | 10â¹ | 12.0 | **2.5** |

Three conclusions, and the third is the actionable one.

1. **The 13-gram convention is `logâ‚‚ D / H` evaluated at GPT-3's corpus size and ordinary
   English.** It is not wrong; it is *specific*, and citing it at a different `D` or on
   different text is a category error. `[A]` high confidence in the derivation (it is the
   AEP plus one division); the empirical `H` values are order-of-magnitude, not measured.
2. **A single global `n` has a heterogeneous, unstated false-positive rate.** Boilerplate
   collides at 40-grams; a random-symbol needle collides at 3-grams essentially never. A
   detector with one threshold is simultaneously far too strict for content and far too lax
   for template text.
3. **Therefore: match on entropy, not on length.** Instead of "flag any 13-gram overlap,"
   flag any overlapping span whose **self-information under a unigram model exceeds
   `logâ‚‚ D` bits**. Same derivation, no free parameter, and it automatically demands longer
   matches from low-entropy text. This is one pass over the corpus to build the unigram
   table and one cumulative sum per candidate span. `[A]` medium-high confidence this is
   strictly better than fixed-`n`; I have not found it in the contamination literature,
   which uses fixed-`n` throughout `[C]` (2311.04850, 2404.00699, 2502.14425, 2605.24079).
   Cheapest test that would move it: Exercise A, extended to score spans by information
   content instead of length, on the same injected-contamination set.

**A refinement worth one line.** The quantity that governs "does *any* n-gram of B appear in
A" is not `H` but the RÃ©nyi-2 (collision) entropy rate `Hâ‚‚ â‰¤ H`, and the expected number of
colliding pairs across corpora of sizes `D_A`, `D_B` is `D_A Â· D_B Â· 2^{âˆ’nÂ·Hâ‚‚}`. Because
`Hâ‚‚ â‰¤ H`, the fixed-`n` threshold derived from `H` is *optimistic*. Exercise A computes the
exact i.i.d. prediction numerically and compares it to real prose, so you do not have to
take either form on faith.

### 3.6 The arithmetic that constrains the schedule

This is the section that will annoy you, and it should be done before any eval plan is
written.

Training FLOPs for one arm, one seed, from the Track A budget rule:

```
FLOPs = 6 Â· N Â· D_train
```

- `N` â€” parameters. `D_train` â€” training tokens. The 6 is forward + backward.

Throughput: `[M]` our measured bf16 GEMM peak is **20.9 TFLOP/s at 8192Â³**
(`ASSUMPTIONS.md â†’ gemm-throughput-below-reference`, which also records that this is 63% of
the figure cited for this silicon and the gap is unexplained). A training step is not pure
GEMM. `[A]` medium confidence that an end-to-end step reaches **35%** of measured GEMM
throughput â‰ˆ **7.3 TFLOP/s** â€” the cheapest test is one timed nanoGPT-scale step, which the
Hardware Validation Gate must run anyway. Substitute your own number; the structure is what
matters.

```
wall-clock per run = 6 Â· N Â· D_train / 7.3e12  seconds
```

| Arm | `N` | `D_train` | FLOPs | hours per seed |
|---|---|---|---|---|
| small | 3Â·10â· | 6Â·10â¸ | 1.08Â·10Â¹â· | **4.1** |
| small | 3Â·10â· | 2Â·10â¹ | 3.6Â·10Â¹â· | **13.7** |
| large | 3Â·10â¸ | 1Â·10â¹ | 1.8Â·10Â¹â¸ | **68.5** |
| large | 3Â·10â¸ | 5Â·10â¹ | 9.0Â·10Â¹â¸ | **342** |

Multiply by arms and seeds:

| Design | Runs | Wall-clock |
|---|---|---|
| 2 arms Ã— 3 seeds at 30M / 0.6B | 6 | **~1 day** |
| 6 arms Ã— 3 seeds at 30M / 0.6B | 18 | **~3 days** |
| 2 arms Ã— 3 seeds at 300M / 1B | 6 | **~17 days** |
| 6 arms Ã— 3 seeds at 300M / 1B | 18 | **~51 days** |
| 6 arms Ã— 3 seeds at 300M / 5B | 18 | **~257 days** |

**The house rule of â‰¥3 seeds and the two-scale rider (~30M and ~300M) are in direct tension
with this machine at the upper scale, and the tension is a schedule fact, not a statistics
opinion.** Three defensible resolutions, all of which must be *declared*:

1. **Seeds at the small scale, one run at the large scale, labelled an anecdote.** This is
   what the two-scale rider is really for: `k = 3` at 30M gives the variance estimate and the
   Spearman rank check; the 300M run tests whether the *ordering* survives, and a single
   ordering is a weaker but non-vacuous claim.
2. **Pair on seed and spend the budget on the difference.** Â§3.2: pairing cancels the seed
   main effect, so the same `k` buys a much smaller `Var(Î”Ì„)`. This is free and should be the
   default regardless.
3. **Cost it and rent it.** `ASSUMPTIONS.md â†’ cloud-budget-zero` sets the budget at $0 and
   the operating instructions rank spend by information per dollar. A 6Ã—3 design at 300M is
   the first thing in this lab whose information-per-dollar might justify a rental, and the
   number above is what the request should quote.

**And the eval side is, by comparison, free.** `research/notes/evaluation-landscape.md` Â§7
derives ~90 minutes per arm for a scaled RULER-style prefill-scored suite at 300M, and ~35Ã—
worse for generation-scored suites. Against 68.5 hours of training per seed, a 1.5-hour eval
is **2.2%** of the run. That ratio is the argument for spending eval effort lavishly:
calibration runs, fault batteries, negative controls, and a full variance decomposition
together still cost less than the training they certify.

---

## 4. Why it matters for Proteus and Mnemosyne

### 4.1 The eval is a gate on the lab's only deliverable

`research/synthesis.md`'s decision is to ship the *instrument*, not policy number 31. An
instrument is certified by its calibration record. So the eval discipline in this module is
not overhead around the research â€” for this lab it substantially *is* the research, and the
methodological result ("nobody in the KV-compression literature reports a needle-removed
control") is available before a single research arm trains
(`curriculum/measuring-memory.md` Â§8 item 4, still unrefuted as of 2026-07-26).

### 4.2 Three fields the pre-registration card is missing

The G2 hypothesis card is HYPOTHESIS / FOR / BECAUSE / MEASURED BY / SUCCESS / KILL / COST /
RISKIEST. Everything in Â§2 and Â§3 says three more fields are load-bearing. Proposed, with an
example filled in â€” this is a recommendation to the card's owner, not an edit to it:

```
MEASURED BY   answer-span log-likelihood, bits/token, on mnemosyne-eval-mqar-v1
              (item set hash 9f2câ€¦, 512 items, generator seed 20260726)
DECISION RULE arm ordering by paired difference across k=3 shared seeds;
              one look, at step 20,000; no interim analysis;
              SUCCESS if the paired 95% CI excludes 0 AND |Î”| > MDE;
              KILL if |Î”| < MDE with the CI containing 0.
MDE           0.004 bits/token  (2.8016Â·Ïƒ_d/âˆšn_eff, Ïƒ_d = 0.031 measured in the
              calibration run, n_eff = 471 after the design-effect correction)
ENVIRONMENT   torch 2.12.0a0+rocm7.13.0a20260313 / HIP 7.2.0 / gfx1151 / Windows 11 26200
              HIPBLASLT_TENSILE_LIBPATH=<set>  TORCH_BLAS_PREFER_HIPBLASLT=1
              TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=<unset>
              dtype=fp32  attention_backend=math (confirmed from stderr, not from the API)
```

Three notes on that block, each earned above.

- **The item set carries a hash.** Not a name, a hash. Â§2.2's fourth failure mode is that
  `I` drifted; a hash makes drift a build failure rather than a mystery.
- **`ENVIRONMENT` is not metadata, it is arm identity.** Â§2.4: three environment terms on
  this machine change the arithmetic, and two of them are invisible to the API that claims
  to report them `[M]`.
- **`MDE` is quoted from a *calibration run*, before the experiment.** If the MDE exceeds
  the smallest effect that would change a decision, the experiment is not worth running and
  you have learned that for the price of an afternoon rather than a fortnight.

### 4.3 Faults are a config axis, not a debug branch

The house rule says the config surface IS the experimental surface. Then fault injection
must be config, because the fault battery is part of every experiment's execution, not a
thing you hack in when suspicious. Concretely, for Mnemosyne:

```
eval:
  item_set: mnemosyne-eval-mqar-v1
  item_set_sha256: 9f2c...
  faults:                       # each is an arm, run alongside the real one
    - none                      # N3: the null intervention; must be bit-identical to oracle
    - needle_absent             # fault 1
    - needle_kv_dropped         # fault 2
    - uniform_eviction: [0.1, 0.25, 0.5, 0.9]   # fault 3
    - position_repack           # fault 4
    - haystack_shuffle          # fault 6  (negative control)
    - irrelevant_span_deleted   # N1      (negative control)
    - fresh_symbol_inventory    # N2      (negative control + contamination probe)
  known_ordering_check:
    weaker: checkpoints/step-1000
    stronger: checkpoints/step-10000
```

The plug point already exists in the reference designs and is worth copying exactly.
SGLang's entire replacement-policy surface is one abstract method
(`memory/sglang/python/sglang/srt/mem_cache/evict_policy.py:10`, `:16`) â€” `get_priority(node)`,
one line per policy. A `RandomStrategy` and a `WorstCaseStrategy` are three lines each, and
injecting them is how you find out whether your metric notices. Mnemosyne's eviction
interface should be exactly this narrow, for exactly this reason: **an interface narrow
enough to implement a real policy in one line is an interface narrow enough to implement a
deliberately broken one.** That is a testability argument for a design choice, and it is the
kind of argument that survives review.

### 4.4 What this says about the Hardware Validation Gate

The gate as written checks capacity, numerics, determinism, checkpoint round-trip, hipBLASLt
configuration, and a known-good tiny recipe. Three additions follow from this module, and
all three are nearly free once the gate is running:

1. **Run the nanoGPT gate recipe with a *fixed* eval item set.** As shipped, the gate's
   target number (`training/nanogpt/README.md:51`, 1.4697) is a Monte Carlo mean over
   resampled batches, so "did we hit it" is a question with a standard error nobody has
   computed. Fix the eval draw first, then compare; otherwise the gate's own pass criterion
   has an unmeasured false-pass rate. The CPU-fallback target is 1.88
   (`training/nanogpt/README.md:85`) and has the same issue with `eval_iters=20`, i.e. ten
   times fewer samples and therefore ~3.2Ã— the standard error.
2. **Record the metric's environment sensitivity as a gate output â€” and record it under two
   aggregations, not one.** Exercise C is the procedure. `[M]` Â§7.3 shows the answer differs
   by 280Ã— depending on whether you summarise with a mean or a maximum, so a single number
   here would be actively misleading. Report both, and put them in `ASSUMPTIONS.md` next to
   `bf16-numerics-unproven`.
   **Also resolve an ambiguity in the gate's own wording while you are there.** The gate
   requires "determinism across repeated runs with a fixed seed." `[M]` Â§7.3(b) finds
   repeated *calls* are bit-exact and repeated *processes* are not, on fp32, at 4.1e-08
   bits/token. Those are different requirements with different pass/fail outcomes, and the
   gate does not currently say which it means.
3. **Fix nanoGPT's MFU denominator before using MFU as a signal.** `estimate_mfu` divides by
   a hardcoded `flops_promised = 312e12` (`training/nanogpt/model.py:301`), which is A100
   bf16 peak. On the 8060S that ratio is meaningless and will read absurdly low even when
   hipBLASLt is configured correctly â€” and the printed value is additionally EWMA-smoothed
   at `0.9Â·old + 0.1Â·new` (`training/nanogpt/train.py:326`), so its apparent stability is
   manufactured: an EWMA with `Î± = 0.1` has an effective sample size of about
   `2/Î± âˆ’ 1 = 19`, meaning the displayed number is roughly a 19-sample average pretending to
   be an instantaneous reading. A metric that is both mis-normalized and silently filtered
   is the perfect small example of this module's thesis.

---

## 5. Read the code

All paths relative to `research/reference/`. Clones are gitignored; run
`scripts/fetch_reference.sh` first. Line numbers are pinned to the revisions in
`PROVENANCE.md`. Every pointer below was opened and the named construct confirmed on the
named line on 2026-07-26.

### 5.1 An eval that is a sanity number pretending to be a gate

Read nanoGPT's eval as a whole; it is fourteen lines and it contains four separate defects,
all of which are *fine* for its purpose and *fatal* for ours. That gap is the lesson.

| Where | What to look at, and why |
|---|---|
| `training/nanogpt/train.py:214` | The comment: *"helps estimate an arbitrarily accurate loss over either split using many batches."* It is not arbitrarily accurate. It is a Monte Carlo mean of `eval_iters` samples with a standard error of `Ïƒ/âˆševal_iters`, and nothing in the code computes `Ïƒ`. |
| `training/nanogpt/train.py:216` | `estimate_loss` â€” the whole eval. Fourteen lines. Read them before reading anything else in this module's exercises. |
| `training/nanogpt/train.py:220` | `losses = torch.zeros(eval_iters)` â€” a per-batch buffer that **could** produce a standard deviation for free. Look at line 226 to see that it does not. |
| `training/nanogpt/train.py:222` | `X, Y = get_batch(split)` **inside** the loop. The item set is resampled on every call. This is failure mode 4 of Â§2.2, in one line. |
| `training/nanogpt/train.py:123` | `ix = torch.randint(len(data) - block_size, (batch_size,))` â€” sampling with replacement from the **global** torch RNG, seeded once at `train.py:106`. Training draws from the same stream, so two runs that differ in step count before an eval see different eval items. Your eval set is coupled to your training schedule. |
| `training/nanogpt/train.py:226` | `out[split] = losses.mean()` â€” the mean is returned; the 200 individual values are discarded. One extra line (`losses.std()`) is the difference between a number and a number with an error bar. |
| `training/nanogpt/README.md:51` | `1.4697` â€” the published threshold, four decimal places, no interval. This is the Hardware Validation Gate's target. Read it against `train.py:216` and decide for yourself what "reproducing it" means. |
| `training/nanogpt/README.md:85` | The CPU fallback invocation, with `--eval_iters=20` and its own published target of `1.88`. Ten times fewer eval samples means âˆš10 â‰ˆ 3.2Ã— the standard error on a target quoted to two decimals. |
| `training/nanogpt/model.py:301` | `flops_promised = 312e12` â€” a metric with a hardcoded denominator from different hardware. |
| `training/nanogpt/train.py:326` | `running_mfu = 0.9*running_mfu + 0.1*mfu` â€” and it is EWMA-smoothed on top. Effective sample size â‰ˆ 19. |
| `training/nanogpt/train.py:323` | `lossf = loss.item()` with the comment *"note: this is a CPU-GPU sync point."* Cited here only because the author flagged the hazard in a comment and then did it anyway at `log_interval` cadence, which is the correct engineering call â€” instrumentation cost is a rate. (`curriculum/measuring-memory.md` Â§3.5 prices it.) |

### 5.2 An eval built by people who had been burned

OLMo-core's evaluator is the counter-example, and the interesting part is *which* hazards
it bothered to encode.

| Where | What to look at, and why |
|---|---|
| `training/olmo-core/src/olmo_core/eval/evaluator.py:23` | The `deterministic` parameter's docstring. Read the whole paragraph. It states, in a library docstring, the exact hazard from Â§2.2: if the eval is truncated, a reshuffle means *different instances are evaluated each time*. Somebody paid for this knowledge. |
| `training/olmo-core/src/olmo_core/eval/evaluator.py:41` | `deterministic: bool = True` â€” and it defaults to safe. Note that the safe behaviour is a *pinned epoch*, not a fixed list: `:68`â€“`:71` reshuffles with `epoch=1` so the order is a pure function of the seed. |
| `training/olmo-core/src/olmo_core/eval/evaluator.py:82` | `def total_batches(self) -> Optional[int]` â€” **Optional**. The size of the eval set is not necessarily knowable. Any statistic whose denominator may be `None` is a statistic you cannot put a confidence interval on. |
| `training/olmo-core/src/olmo_core/train/callbacks/evaluator_callback.py:155` | `if self.eval_duration.due(...): break` â€” the truncation the docstring warns about, in the loop. `n` is set by a time budget. |
| `training/olmo-core/src/olmo_core/train/callbacks/evaluator_callback.py:152` | `with cuda_sync_debug_mode(0):` â€” the eval path *disables* the host-device-sync warning that the training path enables. Correct, and worth noticing: eval is allowed to be slow, training is not. Instrumentation policy differs by phase. |
| `training/olmo-core/src/olmo_core/eval/metrics.py:53` | `class MeanMetric` â€” the production metric class. |
| `training/olmo-core/src/olmo_core/eval/metrics.py:64` | `self.weighted_sum` and `self.weight` â€” **two** accumulators. This class is structurally incapable of producing a variance. Adding `sum_of_squares` and a count is two more device scalars and one more reduction, and it would give every OLMo eval an error bar. **The most useful three minutes in this section: work out what those two extra lines would cost, and then ask why they are not there.** The answer is not laziness; it is that the metric was designed to be *logged*, not to be *tested against a threshold*, and nobody re-derived the requirement when it started being used for the second thing. |
| `training/olmo-core/src/olmo_core/eval/metrics.py:81` | `compute` â€” the mean is formed *after* an `all_reduce_value` across ranks. Read it for the design; we cannot run it (`ASSUMPTIONS.md â†’ single-device-only`). |
| `training/olmo-core/src/olmo_core/eval/lm_evaluator.py:118` | `metric.update(0.0, 0.0)` with the comment *"could be nan but that's okay."* It is okay for a dashboard. A NaN entering a decision rule is not okay, and this is the code path a truncated eval takes. |
| `training/olmo-core/src/olmo_core/eval/lm_evaluator.py:121` | `out[f"{label}/PPL"] = torch.exp(ce_loss)` â€” perplexity is `exp` of a mean loss, so a symmetric interval on CE becomes an asymmetric one on PPL. If you report PPL Â± something, you have reported an interval that is wrong on one side. Report CE, or transform the interval endpoints. |
| `training/olmo-core/src/olmo_core/data/data_loader.py:667` | `_build_global_indices`, and `:673` `rng = get_rng(self.seed + self.epoch)` â€” the pattern to copy for eval item sets: the item ordering is a **pure function of (seed, epoch)** and is therefore reproducible on any machine without persisting it. This is what "the item set carries a hash" should mean in practice. |

### 5.3 The fault-injection plug point

| Where | What to look at, and why |
|---|---|
| `memory/sglang/python/sglang/srt/mem_cache/evict_policy.py:10` | `class EvictionStrategy(ABC)` with one abstract method, `get_priority(node)`. The entire replacement-policy surface. |
| `memory/sglang/python/sglang/srt/mem_cache/evict_policy.py:16` | `class LRUStrategy` â€” one line: `return node.last_access_time`. LFU, FIFO, MRU and FILO follow in three lines each. An interface this narrow makes a deliberately-broken policy as cheap as a real one, which is exactly what fault injection needs. (This pointer is also used in `curriculum/measuring-memory.md` Â§5.2; here it is being read as a *testability* argument rather than an observability one.) |

---

## 6. Exercises

Three. A and B are CPU-only and produce numbers checkable against closed forms or published
tables â€” that is deliberate, because an exercise whose answer you cannot check independently
teaches you nothing about whether your harness works. C uses the GPU and is the one that
produces a number this lab needs.

Activate first, in PowerShell, dot-sourced so the variables survive:

```powershell
. .\scripts\activate-lab.ps1
```

**Standing caveats.** Single tensors â‰¥32 GiB hang silently at 0% CPU `[M]`
(`ASSUMPTIONS.md â†’ large-tensor-fault-32gib`); nothing here comes close, but keep the habit.
bf16 numerics are `untested` `[C]`, so fp32 is the default and bf16 appears only as an
experimental *axis*. The Hardware Validation Gate has not run, so nothing measured here is
evidence by house standard â€” these are instrument shakedowns and should be labelled that way
in the notebook.

Write scratch scripts under `notebook/`. Exercise C is the seed of a rig component and
acquires tests on reuse.

---

### Exercise A â€” find the false-positive floor of a contamination detector, then break it

**Goal.** Establish that n-gram overlap is a decision rule with an FPR set by language
statistics, and measure how easily it is defeated. Produces four checkable numbers.

**Hardware:** none. numpy plus the standard library. **Runtime:** ~2 minutes, dominated by
building n-gram sets over the repo's prose.

Two arms, because one arm cannot tell you whether your harness is broken:

- **Arm 1 â€” i.i.d. Zipf tokens.** Overlap has a closed form, so a harness bug shows up as a
  mismatch instead of as a plausible curve. For each distinct n-gram `g` in split B, the
  probability it occurs somewhere in split A of `N_A` tokens is `1 âˆ’ exp(âˆ’N_A Â· p(g))`, and
  `p(g)` is a product of unigram probabilities because the tokens are independent.
- **Arm 2 â€” real prose** from this repository's `research/`, `curriculum/` and `docs/`
  markdown, split by *document* (alternating) so A and B are never adjacent text.

Then the fault: copy 20 spans of 40 words verbatim from A into a fresh "eval set" and see
whether the detector flags them; repeat with one word in eight replaced by a token that
cannot occur in the corpus, and again with one in four. That is `[C]` 2311.04850's
rephrased-samples result, reproduced at toy scale in a form you can run in two minutes.

```python
# notebook/contamination-ngram-floor.py  (the load-bearing four lines; Â§9.1 has the
# exact configuration I ran, and the rest is corpus loading and printing)
def ngram_types(toks, n):
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}

def overlap_fraction(a_toks, b_toks, n):
    """Fraction of B's distinct n-gram types that also occur in A."""
    a, b = ngram_types(a_toks, n), ngram_types(b_toks, n)
    return len(a & b) / len(b), len(b)
```

**Predict before you run.** From Â§3.5, `n* = logâ‚‚(D)/H`. For arm 1 at `D = 2.5Â·10âµ` tokens
per split, `logâ‚‚ D = 17.9`; the effective per-token entropy of Zipf(1.1) truncated at 20,000
is around 7 bits, so `n* â‰ˆ 2.5` â€” collisions at `n = 2` and `n = 3`, essentially none by
`n = 5`. For arm 2 the corpus is larger (`logâ‚‚ D â‰ˆ 20.7`) but word-level English entropy is
higher (~10 bits), so the i.i.d. prediction is `n* â‰ˆ 2`. **Both null models say a 5-gram
collision should be rare and a 13-gram collision should never happen. Write that down before
you run, because arm 2 is going to disagree by a very large factor, and the gap is the whole
exercise.**

**Deliverables â€” four numbers.**

1. Arm 1: measured versus analytic overlap at each `n`. They must agree; if they do not,
   your harness is wrong and nothing else in the exercise means anything.
2. Arm 2: the smallest `n` at which real-prose overlap drops below 1%.
3. The ratio of arm 2's overlap to arm 1's analytic prediction at `n = 5`. This is the
   "excess over chance," and it is the detector's false-positive driver.
4. The verbatim-versus-paraphrased detection rates at `n = 13`. Report the clean-item rate
   too â€” that is the measured FPR of the detector on genuinely held-out text.

**What a falsification would mean.** If arm 2's overlap fell to arm 1's level, the 13-gram
convention would be far more conservative than necessary and Â§3.5's conclusion 2 would be
wrong. Report that if you see it. (It did not; see Â§7.)

---

### Exercise B â€” calibrate a decision rule instead of a metric

**Goal.** Measure the false-positive rate of three decision rules you will be tempted to use,
against published or closed-form values.

**Hardware:** none. numpy. **Runtime:** 5â€“15 minutes for 200,000 simulations per cell; drop
to 20,000 for a 1-minute version with looser agreement.

Three simulations, all under a known-variance z-test so the theory is exact:

1. **Peeking.** Two arms with no true difference. Look `k` times at equal increments of the
   accumulating data; "reject" if any look crosses `|z| > 1.95996`. Compare against the
   Armitage/McPherson/Rowe table in Â§3.3.
2. **Forking paths.** `m` independent metrics, no true effect anywhere, report the best.
   Compare against `1 âˆ’ (1 âˆ’ Î±)^m`.
3. **Type M / Type S.** Fix a true effect `Î¼` calibrated to a target power by bisection;
   among the significant draws, report `E[|dÌ‚|]/d` and the fraction with the wrong sign.
   Compare against Â§3.4's closed-form table.

```python
def peeking(k_looks, n_final=1000, sims=200_000, rng=RNG):
    step, fires = n_final // k_looks, 0
    for _ in range(sims):
        cs = np.cumsum(rng.normal(0.0, math.sqrt(2.0), n_final))   # paired difference
        ns = np.arange(step, n_final + 1, step)
        z = cs[ns - 1] / (math.sqrt(2.0) * np.sqrt(ns))
        fires += bool(np.any(np.abs(z) > 1.959963984540054))
    return fires / sims
```

**Deliverables â€” three tables and one sentence.**

1. Peeking FPR at `k âˆˆ {1, 2, 3, 5, 10, 20}` against the published column. Agreement to
   Â±0.003 at 200k sims.
2. Forking FPR at `m âˆˆ {1, 2, 3, 5, 10, 20}` against `1 âˆ’ 0.95^m`. This one is exact; a
   mismatch is a bug.
3. Exaggeration and Type S at target powers `{0.10, 0.20, 0.35, 0.50, 0.80, 0.95}` against
   Â§3.4's table.
4. **The sentence.** Write down, in your own words, which of your existing experiment plans
   this invalidates. Mine was "check the eval every 1,000 steps and stop when the arms
   separate," which is the `k = 20` row.

---

### Exercise C â€” measure the nuisance-axis noise floor of a task metric on this machine

**Goal.** The number every subsequent claim from this hardware has to clear: how far does a
fixed metric on a fixed item set move when you change only things you do not care about?

**Hardware:** one gfx1151 GPU, native Windows. **CPU fallback:** identical code; drop
`N_ITEMS` to 128 and `T` to 128. The bf16 arm is meaningless on CPU (no bf16 kernels worth
the name), so on CPU you get the batch-composition and cross-process arms only â€” which is
itself informative, because those are the arms that *should* be zero everywhere.
**Runtime:** ~1 minute per environment arm on GPU after torch imports, so ~5 minutes for
four arms; ~4 minutes on CPU.

**Design.** One randomly initialized 4-layer transformer, one fixed set of 512 items of
length 256, one metric: mean next-token NLL over the last 32 positions of each item, in
bits per token. Random initialization is not a cheat â€” the quantities being measured are
properties of the kernels and the environment, not of the weights, exactly as in
`curriculum/measuring-memory.md` Exercise B. What a random model *cannot* tell you is
whether a real effect would clear the floor; that requires a trained checkpoint and is the
follow-up.

Then vary, one at a time:

| Arm | Axis | Should be |
|---|---|---|
| `rerun_bs8_fp32` | nothing (same process) | exactly 0 |
| `bs1_fp32`, `bs64_fp32` | batch composition | ~0; nonzero means reduction order is batch-dependent |
| `bs8_bf16`, `bs64_bf16` | dtype | nonzero; this is the number that matters |
| second process, identical env | process boundary | exactly 0 |
| `HIPBLASLT_TENSILE_LIBPATH` unset | the numerics control `[M]` | unknown â€” that is why you run it |
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` | attention kernel `[M]` | unknown |

Run the same script four times from a shell script, with the environment differing per
invocation, and diff the JSON. Do **not** try to change these inside one process; they are
read at library init.

**Deliverables â€” three numbers and a comparison.**

1. `Ïƒ_item`, the per-item standard deviation of the metric in bits/token, and the resulting
   unpaired MDE at `n = 512` (`2.8016 Â· Ïƒ_item Â· âˆš(2/n)`).
2. The largest `|Î” mean|` across all nuisance arms. **This is the noise floor.**
3. The ratio (2)/(1). If it exceeds 1, your eval can "detect" a difference that is entirely
   an environment difference, and no result from this configuration is interpretable without
   an environment fingerprint.
4. Whether `bs1` and `bs64` agree bit-for-bit with `bs8`. A disagreement here invalidates
   every cross-batch-size comparison in the lab and is worth a `BLOCKERS.md` entry.

**Failure modes to expect.** If the bf16 arm returns `inf` or `nan`, you have found a
numerics bug â€” `BLOCKERS.md`, and tell the Hardware Validation Gate. If every arm is exactly
zero including bf16, check that the dtype cast actually applied to the *activations* and not
only the weights. If the cross-process arm is nonzero, something in your environment is not
being captured by the fingerprint, and finding out what is more valuable than the rest of the
exercise.

**Expect a stderr `UserWarning` about AOTriton on every arm that does not set the flag.**
That is `ASSUMPTIONS.md â†’ sdpa-is-memory-efficient` reproducing itself live, and it is the
only honest signal of which attention path ran. Capture stderr in your harness.

---

## 7. Findings from running these exercises

Everything here was produced on 2026-07-26 on the machine described in Â§9.1. Exercises A and
B are pure computation and deterministic given their stated seeds, so a single fresh-process
run *is* a reproduction. Exercise C ran four fresh processes, one of which is a
same-environment repeat â€” and that repeat is where the most interesting result came from.

### 7.1 Exercise A â€” fixed-`n` contamination detection has no good operating point

**The harness validates.** `[M]` Arm 1, i.i.d. Zipf(1.1) over a 20,000-symbol inventory,
250k tokens per split: measured type-overlap **0.07369** at `n = 3` against an analytic
prediction of **0.07330**, and **0.00265** vs **0.00254** at `n = 5`. Both are zero to five
decimal places from `n = 8` upward. Agreement to ~0.5% means the set arithmetic is right, so
arm 2's numbers can be believed.

**Real prose disagrees with chance by a factor you should look at twice.** `[M]` Arm 2, 3,085
markdown documents from `research/`, `curriculum/` and `docs/` (3,378,202 word tokens; the
document list hashes to `c9950ed5d8b9`), split by alternating document:

| `n` | i.i.d. analytic | real prose, measured | excess |
|---|---|---|---|
| 3 | 0.0733 | **0.2215** | 3.0Ã— |
| 5 | 0.00254 | **0.1432** | **56Ã—** |
| 8 | 0.00000 | **0.1205** | unbounded |
| 13 | 0.00000 | **0.1017** | unbounded |
| 20 | 0.00000 | **0.0867** | unbounded |

Deliverable 2 has no answer: **overlap never falls below 1% at any `n â‰¤ 20`.** Nearly nine
percent of distinct 20-grams in half this corpus also occur in the other half, at a length
where the i.i.d. null puts the expected count below 10â»Â³â°. There is no amount of "it was a
coincidence" that covers that gap.

*Read the "excess" column as directional, not exact:* the two arms have different corpus
sizes (250k vs ~1.69M tokens per split) and different per-token entropies, so the analytic
column is arm 1's null, not arm 2's. Correcting for size would raise the null by at most
`logâ‚‚(6.8) â‰ˆ 2.8` bits' worth of `n`, i.e. shift the collapse point right by well under one
gram. It does not survive contact with a 56Ã— gap at `n = 5`.

**Then the detector's operating characteristic, which is the actual deliverable.** `[M]`
Twenty 40-word items per condition; "flagged" means the item contains at least one `n`-gram
present in split A:

| `n` | clean items (**false positives**) | verbatim copies | 1-in-8 corrupted | 1-in-4 corrupted |
|---|---|---|---|---|
| 3 | **1.000** | 1.000 | 1.000 | 1.000 |
| 5 | **0.600** | 1.000 | 1.000 | 0.000 |
| 8 | **0.500** | 1.000 | 0.000 | 0.000 |
| 13 | **0.350** | 1.000 | 0.000 | 0.000 |
| 20 | **0.250** | 1.000 | 0.000 | 0.000 |

Read the table as a receiver operating characteristic and three things fall out:

1. **The false-positive rate at the conventional threshold is 35%.** One held-out item in
   three is flagged as contaminated when it is not. A "contamination check" reporting 35% on
   clean data is not a check.
2. **Evasion is arithmetic, not cleverness.** Corrupting one word in `k` leaves a longest
   clean run of `k âˆ’ 1`, so every `n â‰¥ k` detects nothing. Period 8 dies at `n = 8`; period 4
   dies at `n = 5`. This is `[C]` 2311.04850's rephrased-samples result in its most trivial
   possible form, and it means an adversary â€” or an honest data-cleaning pipeline that
   normalises whitespace and quotation marks â€” needs no sophistication at all.
3. **There is no `n` that is simultaneously acceptable on both axes.** Small `n` catches
   paraphrase and flags everything; large `n` has a tolerable false-positive rate and is
   defeated by a corruption every 20 words. The two failure modes are on opposite ends of the
   same single parameter. That is the structural argument for Â§3.5's conclusion 3 â€” threshold
   on self-information rather than on length â€” and it is stronger than the argument I made
   from the formula alone, because a formula can be wrong in a direction and a
   two-sided-failure ROC cannot.

**Two honest caveats.** First, this corpus is unusually self-similar: it includes
`research/reference/`'s upstream markdown, which carries duplicated licence blurbs, changelog
templates and boilerplate across repositories. A curated corpus would give a lower
false-positive rate. Second, that is also the *realistic* condition â€” web-scraped pretraining
data is exactly a pile of near-duplicates â€” so the direction of the bias is toward optimism
in curated settings and toward this measurement in real ones. Neither caveat touches the
evasion result, which is pure arithmetic.

### 7.2 Exercise B â€” the decision-rule numbers reproduce, including the ones I quoted

`[M]` 200,000 simulations per cell, `numpy.default_rng(1337)`, known-variance z-test.

| looks `k` | measured FPR | Â§3.3 published |
|---|---|---|
| 1 | 0.0498 | 0.050 |
| 2 | 0.0837 | 0.083 |
| 3 | 0.1067 | 0.107 |
| 5 | 0.1419 | 0.142 |
| 10 | 0.1942 | 0.193 |
| 20 | **0.2475** | 0.246 |

Every cell agrees within 0.0015. That matters for a reason beyond the exercise: I quoted the
1969 table from a non-arXiv source, and this is the check on my quotation. If you are going
to cite a fifty-year-old numerical result, reproduce it.

`[M]` Forking paths matched `1 âˆ’ 0.95^m` to three decimals at every `m` (0.0494 vs 0.0500 at
`m = 1`; 0.6413 vs 0.6415 at `m = 20`) â€” exact agreement expected, so this is purely a
harness check, and it passed.

`[M]` Type M / Type S, against Â§3.4's hand-computed closed forms:

| target power | achieved | exaggeration, measured | Â§3.4 closed form | sign error |
|---|---|---|---|---|
| 0.10 | 0.100 | **3.72Ã—** | 3.71Ã— | 0.0419 |
| 0.20 | 0.201 | **2.26Ã—** | 2.25Ã— | 0.0055 |
| 0.35 | 0.349 | 1.67Ã— | 1.68Ã— | 0.0005 |
| 0.50 | 0.500 | 1.41Ã— | 1.40Ã— | 0.0001 |
| 0.80 | 0.799 | 1.12Ã— | 1.13Ã— | ~0 |
| 0.95 | 0.950 | 1.03Ã— | 1.03Ã— | ~0 |

Nothing surprising, which is the point: Â§3.4's algebra is checkable, and now checked.

### 7.3 Exercise C â€” the prediction failed, and the failure is more useful than the prediction

Four fresh processes. Model: `torch.nn.TransformerEncoder`, 4 layers / 4 heads /
`d_model = 256` / `dim_feedforward = 1024`, `norm_first=True`, dropout 0, randomly
initialised at seed 1337, `eval()`, causal float mask. Items: 512 fixed sequences of 256
token ids, `vocab = 4096`, generator seed 20260726. Metric: mean next-token NLL over the last
32 positions, bits/token.

**(a) fp32 is bit-identical across batch size, inside a process.** `[M]` `bs = 1`, `8` and
`64` produced per-item values with `max |Î”| = 0.0` exactly, in all four processes. This is a
real relief and worth stating positively: **cross-batch-size comparisons are valid on this
stack at this shape**, which is the one nuisance axis `curriculum/measuring-memory.md` Â§2.7
flags as needing re-checking whenever padding is involved.

**(b) fp32 is *not* bit-identical across processes.** `[M]` Four processes produced three
distinct fp32 means:

| process | environment | fp32 mean (bits/token) | Î” vs process 1 |
|---|---|---|---|
| 1 | hipBLASLt configured, AOTriton off | 12.470379628241062 | â€” |
| 2 | **identical to process 1** | 12.470379669219255 | **+4.098e-08** |
| 3 | hipBLASLt unset | 12.470379645004869 | +1.676e-08 |
| 4 | AOTriton on | 12.470379628241062 | **0.0** |

Read rows 2 and 4 together. The pair with *identical* environments disagrees; a pair with
*different* environments agrees bit-for-bit. **Process-to-process variation is at least as
large as the environment variation I set out to measure**, which means an environment
fingerprint is necessary and not sufficient. `[A]` medium confidence the mechanism is
non-deterministic kernel or algorithm selection at library initialisation; I have not
isolated it and four processes is a small sample. Cheapest test that would move it: twenty
fresh processes in the same environment, reporting the distinct-value count â€” twenty minutes,
and it converts an observation into a distribution.

This does not contradict `curriculum/measuring-memory.md` Exercise B's "repeat null is
exactly zero"; that null was measured *within* a process, and this measurement confirms it
(the in-process rerun arm was exactly 0.0 in all four processes). It extends it: **the
degenerate null is a property of a process, not of the machine.** An oracle-diff harness that
runs the reference and the policy in separate processes has a non-zero null it did not
account for.

*Housekeeping:* a Track D module on determinism and reproducibility was in flight while this
was written. If it measures the same quantity, the two results must be **reconciled** â€” one
of them promoted, the other cited â€” rather than both reported as independent findings. Two
modules quoting slightly different numbers for the same thing is how a curriculum acquires an
`[M]` nobody can retest.

**(c) The dtype nuisance is invisible in the mean and loud in the tail.** `[M]` bf16 weights
and activations against the fp32 reference, same items:

| quantity | hipBLASLt configured | hipBLASLt unset |
|---|---|---|
| shift in the **mean** | 3.71e-06 (bs 8), 4.28e-06 (bs 64) | 1.26e-06 |
| **paired sd** across items | 1.054e-03 (bs 8), 1.071e-03 (bs 64) | 1.095e-03 |
| **max per-item** shift | 3.450e-03 (bs 8), 3.445e-03 (bs 64) | 3.931e-03 |

The mean shift is **280Ã—** smaller than the paired standard deviation, and the maximum
per-item shift is **3.3 standard deviations** â€” precisely what you expect for a maximum over
512 roughly-normal zero-mean perturbations, which is itself a consistency check. So:

> A zero-mean numerical perturbation is harmless to a mean-aggregated metric and directly
> harmful to a worst-case, threshold-count, or rank-based one. **The aggregation function
> decides whether your environment is a confound.** Pre-register it.

**(d) hipBLASLt's numerics effect is a function of reduction length, and it has a side
effect nobody mentions.** `[M]` Unsetting hipBLASLt raised the max per-item bf16 shift from
3.450e-03 to 3.931e-03 (**+14%**) and the paired sd by +3.9%. That is far smaller than the
**2.8Ã—** recorded in `ASSUMPTIONS.md â†’ hipblaslt-config`, and there is no contradiction: that
row measured a **length-1,048,576** reduction, and the longest reduction here is
`d_model = 1024` in the feed-forward block. The effect scales with reduction length, which is
the expected behaviour of accumulation error and is worth stating as a rule â€”
**`hipblaslt-config`'s 2.8Ã— is a long-reduction number and should not be quoted for
short-reduction workloads.**

The side effect is the interesting half. `[M]` **With hipBLASLt configured, the bf16 results
depend on batch size** (bs 8 and bs 64 differ in mean, paired sd and max). **With hipBLASLt
unset, they are bit-identical to each other.** Configuring hipBLASLt buys accuracy on long
reductions and costs batch-size invariance in bf16. Both halves of that trade belong in the
environment fingerprint, and the consequence is concrete: *if you run bf16 with hipBLASLt
configured, batch size is part of the arm's identity and must be held fixed across arms.*

**(e) The AOTriton flag changed the kernel and changed no digit.** `[M]` With
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` the stderr `UserWarning` about experimental
mem-efficient attention **disappeared** â€” so the flag took effect â€” and every arm's values
were bit-identical to process 1. At `T = 256`, 4 heads, this shape, the flag is not a
numerics change. It remains a *memory* change (`ASSUMPTIONS.md â†’ sdpa-is-memory-efficient`,
147.2 â†’ 6.6 bytes/TÂ²), and `[A]` no confidence that bit-identity holds at longer `T` â€” the
test is the same script at `T = 2048` and up. Note also what served as the detector: **the
absence of a warning on stderr.** There is still no API that reports which path ran.

**(f) The prediction, and its refutation.** I predicted the environment floor might exceed
the eval's minimum detectable effect. `[M]` It does not, by nearly four orders of magnitude:

```
Ïƒ_item                       = 0.2054 bits/token
SEM at n = 512               = 0.2054 / âˆš512      = 9.08e-03 bits/token
MDE unpaired at n = 512      = 2.8016Â·ÏƒÂ·âˆš(2/512)  = 3.60e-02 bits/token
MDE unpaired at n = 4,096                         = 1.27e-02 bits/token
largest nuisance mean shift                       = 4.28e-06 bits/token
ratio (nuisance shift / SEM)                      = 4.7e-04
```

**At this shape, item sampling dominates environment noise by ~2,100:1 for a mean-aggregated
metric.** The honest conclusion is the opposite of the one I set out to write: *for a mean,
buy items and stop worrying about the environment; for anything else, the environment is the
whole game.* Both halves are in the numbers above and neither was obvious before running it.

**What this does not establish.** The model is randomly initialised, so `Ïƒ_item = 0.2054` is
the item variance of a *random* model's answer-span NLL and there is no reason to think a
trained model's is the same. Nothing here says whether a real architectural effect clears
`3.60e-02` bits/token at `n = 512` â€” that requires trained arms and is unsolved item 5 in
Â§10. And by house standard, none of this is evidence until the Hardware Validation Gate runs.

---

## 8. Self-check

Answers at the end. Do not scroll.

1. Your ablation has 6 arms. You evaluate each at 20 checkpoints and report the arm and
   checkpoint with the largest gap over baseline, at p < 0.05. Estimate the probability that
   you report *something* under a complete null, and say which of the two multiplicity
   effects dominates.

2. You have 3 seeds per arm and you can afford either (a) 10Ã— more eval items or (b) one more
   seed. Under what condition is (a) the right choice? Give the condition as an inequality in
   quantities you can measure, and say what you must run first to evaluate it.

3. A colleague reports that eviction policy P improves answer accuracy by 6 points, p = 0.04,
   from a study you estimate had about 20% power. What is your best estimate of the true
   effect, and what single number would you ask for before believing any of it?

4. You are checking a generated eval item set against a 1B-token training corpus. The items
   are natural-language sentences; the needles are random 6-character symbols. Should you use
   the same n-gram threshold for both? Justify with the formula, and give the two thresholds.

5. Your retrieval eval scores identically whether you shuffle the haystack sentence order or
   not. Is that a pass or a fail? Now the same eval scores identically whether the needle is
   present or absent. Which of the two results is more urgent, and why?

6. On this machine, name the three environment terms that change a metric without changing
   any line of your code, say which one is measured to change *numerics* rather than speed,
   and name the API that reports a capability it does not deliver.

---

## 9. Sources and measurements

### 9.1 Local measurements produced for this module

Environment for all runs: `torch 2.12.0a0+rocm7.13.0a20260313` (HIP 7.2.0), AMD Radeon 8060S
(gfx1151), native Windows 11 build 26200, Python 3.12, `C:\venvs\lab`, 2026-07-26. The
Hardware Validation Gate has **not** run, so by house standard these are instrument
shakedowns, not evidence.

*Exercises A and B are pure computation and deterministic given the stated seed; a single
fresh-process run reproduces exactly. Exercise C's four arms are four fresh processes, one of
which is a same-environment repeat â€” and that repeat produced finding 7.3(b).*

**All result tables are in Â§7.** Configurations, exactly as run:

| Exercise | Configuration |
|---|---|
| **A** | Arm 1: i.i.d. Zipf(s = 1.1) over `V = 20,000`, 250,000 tokens per split, `numpy.default_rng(1337)`. Arm 2: 3,085 markdown files under `research/`, `curriculum/`, `docs/`; lowercased, tokenised by `[a-z0-9']+`; 3,378,202 word tokens; split by alternating document; document-list `sha256[:12] = c9950ed5d8b9`. Fault arms: 20 spans Ã— 40 words, corruption token `QQZZ` (absent from the corpus) at period 8 and 4. Runtime under a minute. |
| **B** | 200,000 simulations per cell, `numpy.default_rng(1337)`, `z_crit = 1.959963984540054`, `n_final = 1000` for the peeking arm, 400,000 draws for the Type M arm with `Î¼` solved by 80-step bisection to the target power. |
| **C** | `torch 2.12.0a0+rocm7.13.0a20260313`, HIP 7.2.0, gfx1151, native Windows 11 build 26200, `C:\venvs\lab`. `TransformerEncoder`, `L = 4`, `H = 4`, `d_model = 256`, `dim_feedforward = 1024`, `norm_first = True`, `dropout = 0`, seed 1337, `eval()`, causal float mask. 512 items Ã— 256 tokens, `V = 4096`, item generator seed 20260726. Metric: mean next-token NLL over the last 32 positions, bits/token, logits cast to fp32 before `log_softmax`. Four fresh processes: (1) hipBLASLt configured / AOTriton off, (2) identical to 1, (3) hipBLASLt unset, (4) AOTriton on. |

**No number in this module is stated as `[M]` unless it was produced in a fresh process on
the machine above.** Derived quantities â€” the MDE table, the `4.7e-04` ratio, everything in
Â§3.6 â€” are arithmetic on `[M]` inputs and are labelled as derivations in place.

**Retest instructions.** Exercises A and B should reproduce bit-for-bit; if they do not, the
corpus changed (A) or the numpy version changed (B), and the document-list hash tells you
which. Exercise C's fp32 means should **not** be expected to reproduce exactly across
processes â€” that is finding 7.3(b). Compare the in-process deltas, the paired sds, and the
max per-item shifts; those were stable across all four processes and are the numbers to hold
this module to.

### 9.2 Repo artifacts this module depends on

- `ASSUMPTIONS.md` rows: `gpu-fast-tier-size` (â‰¥62 GiB at ~200 GB/s `[M]`, single run per
  arm), `gemm-throughput-below-reference` (20.9 TFLOP/s bf16 at 8192Â³ `[M]`),
  `large-tensor-fault-32gib` (`[M]`), `sdpa-is-memory-efficient` (147.2 vs 6.6 bytes/TÂ²
  `[M]`; `flash_sdp_enabled()` returns True either way), `hipblaslt-config` (refuted as a
  throughput cliff, upgraded to a **numerics** control: 2.01e-3 vs 5.60e-3 relative error,
  ~2.8Ã—, 3 seeds `[M]`), `bf16-reduced-precision-knob-works` (**refuted** â€” the knob is inert
  `[M]`), `bf16-numerics-unproven` (`untested`), `single-device-only` (`[C]`),
  `ablation-scale-sufficient` (`[A]`), `cloud-budget-zero`.
- `notebook/uma-carveout-controls-fast-tier.md` â€” the fast-tier sweep, single run per arm.
- `research/notes/evaluation-landscape.md` â€” Â§1 (outcome vs attribution), Â§2 (why NIAH is
  insufficient), Â§5 (contamination and its inversion), Â§6.2 (the fault battery), Â§7 (the
  eval cost arithmetic). This module is the curriculum form of Â§5, Â§6.2 and the statistics
  half of Â§5.
- `research/memory/memory-failure-register.md` â€” `attribution-gap-in-serving-results`,
  `eviction-destroys-long-range-recall`, `quantization-breaks-alignment-not-perplexity`.
- `curriculum/measuring-memory.md` â€” Â§2.7 (nulls from nuisance axes), Â§3.3 (per-token KL and
  its dissociation from accuracy), Â§3.4 (sample size, paired design, MDE), Â§2.6 (the fault
  battery as taught). Hard prerequisite; deliberately not repeated.
- `curriculum/memory-failure-modes.md` â€” Â§2.5 (NIAH adverse selection in full), Â§3.5 (the
  salience-rank design rule that is fault zero), Â§4.2(3).
- `curriculum/scaling-laws-and-flops-budget.md` â€” the `6Â·NÂ·D` rule used in Â§3.6.
- `research/reference/CODE_MAP.md` and `PROVENANCE.md` â€” the revisions all `file:line`
  pointers are pinned to.

### 9.3 Citations

Non-arXiv, cited by title because it has no arXiv id:

- **Armitage, P., McPherson, C. K., & Rowe, B. C.**, *Repeated significance tests on
  accumulating data*, Journal of the Royal Statistical Society Series A, 132(2):235â€“244,
  1969. Source of the repeated-looks table in Â§3.3. Exercise B reproduces it by simulation
  precisely so the quotation can be checked.
- **Gelman, A. & Carlin, J.**, *Beyond power calculations: assessing Type S (sign) and Type M
  (magnitude) errors*, Perspectives on Psychological Science 9(6):641â€“651, 2014. The origin
  of the Type S / Type M framing in Â§3.4; the closed forms in Â§3.4 are derived here directly
  from the normal model rather than quoted.

arXiv `[C]`. Ids marked â€  were drawn from `research/notes/evaluation-landscape.md`, whose
author resolved every id against the live arXiv API on 2026-07-26; this module adds no id
that is not in that note's source list.

- `2304.15004`â€  â€” *Are Emergent Abilities of Large Language Models a Mirage?* (Apr 2023).
  Discontinuous metrics manufacture cliffs; the argument for continuous metrics at small
  scale.
- `2311.04850`â€  â€” *Rethinking Benchmark and Contamination for Language Models with Rephrased
  Samples* (Nov 2023). Paraphrase defeats n-gram detection; 8â€“18% of HumanEval in
  RedPajama-1T / StarCoder-Data. The result Exercise A reproduces at toy scale.
- `2402.02823`â€  â€” *Evading Data Contamination Detection for Language Models is (too) Easy*
  (Feb 2024).
- `2402.07841`â€  â€” *Do Membership Inference Attacks Work on Large Language Models?* (Feb
  2024). MIAs near chance from 160M to 12B; apparent successes attributed to temporal shift.
- `2403.07974`â€  â€” *LiveCodeBench* (Mar 2024). Rotation as the practical mitigation.
- `2404.00699`â€  â€” *A Comprehensive Survey of Contamination Detection Methods in LLMs* (Mar
  2024).
- `2404.06654`â€  â€” *RULER* (Apr 2024). Saturation of vanilla NIAH across 17 models; generator
  methodology.
- `2404.15574`â€  â€” *Retrieval Head Mechanistically Explains Long-Context Factuality* (Apr
  2024). Fault 5 of the battery.
- `2406.10229`â€  â€” *Quantifying Variance in Evaluation Benchmarks* (Jun 2024). Seed variance
  and non-monotonicity large enough to make small-scale differences frequently meaningless.
- `2406.19314`â€  â€” *LiveBench* (Jun 2024).
- `2410.05229`â€  â€” *GSM-Symbolic* (Oct 2024). Perturbation twins as a method; negative control
  N2.
- `2411.00640`â€  â€” *Adding Error Bars to Evals* (Nov 2024). Question-level clustering, paired
  analysis, power analysis before the run.
- `2409.06338`â€  â€” *Retrieval Or Holistic Understanding? Dolce* (Sep 2024). Cited in Â§10 item
  9 as one side of the synthetic-versus-realistic dispute.
- `2410.02694`â€  â€” *HELMET* (Oct 2024). NIAH does not predict downstream performance; the
  seven categories have low mutual correlation.
- `2502.05167`â€  â€” *NoLiMa: Long-Context Evaluation Beyond Literal Matching* (Feb 2025). The
  high-entropy-value *and* no-lexical-overlap construction; 11 of 13 models drop below 50% of
  their own short-context baseline at 32K.
- `2502.14425`â€  â€” *A Survey on Data Contamination for Large Language Models* (Feb 2025).
- `2505.19293`â€  â€” *100-LongBench: Are de facto Long-Context Benchmarks Literally Evaluating
  Long-Context Ability?* (May 2025).
- `2510.13334`â€  â€” *Taming the Fragility of KV Cache Eviction in LLM Inference* (Oct 2025).
  Worst-case aggregation inverts the ranking of eviction policies; Â§7.3(c) supplies an
  independent mechanical reason the aggregation choice matters.
- `2605.24079`â€  â€” *TRACER: Fine-Grained Contamination Detection in Code LLMs* (May 2026).
- `2605.28079`â€  â€” *ATLAS: All-round Testing of Long-context Abilities across Scales* (May
  2026). Rank instability across length regimes: 7 of 26 models shift â‰¥2 positions.
- `2606.05241`â€  â€” *Search-Time Contamination in Deep Research Agents* (Jun 2026). Runtime
  contamination; training-data hygiene does not protect you.

---

## 10. What is still unsolved here

Everything below is testable at 20Mâ€“300M on one gfx1151 GPU inside the `[M]` â‰¥62 GiB fast
tier, unless marked otherwise. Each needs a pre-registered hypothesis card before it runs.

1. **We do not know `ÏƒÂ²_seed` for anything.** The entire Â§3.2 apparatus is unusable until
   somebody trains the same arm three times and measures it. This is the cheapest
   decision-changing experiment in Track E: three runs at 30M / 0.6B is ~12 hours by Â§3.6's
   arithmetic, and the result determines whether every eval plan in this lab should be
   buying items or buying seeds. **Nothing downstream is interpretable without it.**

2. **The three-seed rule and the 300M scale are in conflict on this hardware, and the
   conflict is unresolved.** Â§3.6: 6 arms Ã— 3 seeds at 300M / 1B tokens is ~51 days of
   continuous wall-clock under an `[A]` 35%-of-GEMM-peak utilization assumption. The
   utilization figure has never been measured end to end â€” that is a Hardware Validation
   Gate item and it moves the answer by a factor of two either way. Until it is measured, the
   lab's schedule rests on an assumption, and `ASSUMPTIONS.md` should carry a row for it.

3. **Entropy-thresholded contamination detection appears to be unpublished, and Exercise A
   strengthened the case for it.** Â§3.5 conclusion 3: flag a span when its self-information
   exceeds `logâ‚‚ D` bits, rather than when its length exceeds a fixed `n`. The measurement
   (Â§7.1) shows fixed-`n` failing on *both* axes simultaneously â€” 35% false positives at
   `n = 13`, total evasion by a corruption every 8 words â€” with the two failures controlled
   by the same single parameter in opposite directions, so no tuning fixes it. `[A]`
   medium-high confidence the information threshold dominates; I could not find it in the
   contamination literature, which is uniformly fixed-`n` `[C]` (2311.04850, 2404.00699,
   2502.14425, 2605.24079). **What is unmeasured is the thing that matters:** whether it
   actually beats fixed-`n` on the same ROC. That is Exercise A plus twenty lines â€” build the
   unigram table, score each candidate span by cumulative self-information, sweep the
   threshold, and plot both detectors on the same axes. One afternoon. If it wins it is a
   small publishable methods result; if it loses, Â§3.5 conclusion 3 is wrong and should be
   struck.

4. **The fault battery has no accepted specificity half.** Â§2.6 proposes three negative
   controls and a known-ordering check. None of them is standard, and I have not found a
   published eval that reports any of them. `[A]` high confidence that the *sensitivity* half
   is also unpracticed â€” `curriculum/measuring-memory.md` Â§8 item 4 records a twelve-month
   search of the eviction literature finding no needle-removed control, and a second search
   on 2026-07-26 found nothing contradicting it. Two searches by two authors is not a
   systematic review; treat it as a strong prior, not a fact.

5. **The environment floor turned out not to bind for a mean â€” and nobody has checked the
   aggregation functions where it does.** `[M]` Â§7.3(f): the largest nuisance mean shift is
   2,100Ã— below the standard error at `n = 512`. But the same perturbation moves individual
   items by 3.3 sd, so worst-case, threshold-count and rank-based aggregations are exposed
   and *were not measured*. The open question is therefore sharper than the one I planned:
   **at what aggregation function does this machine's numerical noise start to change a
   ranking?** Sweep aggregation (mean, trimmed mean, p95, max, count-above-threshold, rank
   correlation) over the same fp32/bf16 pair and find the crossover. Inference-only, one
   afternoon, and it directly informs the pre-registration requirement in Â§4.2. The related
   unknown remains: with a *trained* model, is `Ïƒ_item` similar to the 0.2054 bits/token
   measured at random init? No reason to assume so.

6. **fp32 is not bit-reproducible across processes on this stack, and the mechanism is
   unattributed.** `[M]` Â§7.3(b): four processes, three distinct fp32 means, including a
   disagreement between two processes with *identical* environments (4.1e-08 bits/token).
   Within a process everything is exact. `[A]` medium confidence this is non-deterministic
   kernel selection at library init. It is small enough to ignore for mean-based metrics and
   large enough to break any harness that assumes a degenerate null across processes â€” which
   is exactly what an oracle-diff instrument built as two separate invocations would assume
   (`curriculum/measuring-memory.md` Â§4.3). Cheapest test: twenty fresh processes, identical
   environment, report the number of distinct values and their spread. Twenty minutes. If the
   spread is stable it becomes a documented floor; if it is not, it is a
   determinism failure and a Hardware Validation Gate blocker, since the gate explicitly
   requires "determinism across repeated runs with a fixed seed" and nobody has specified
   whether "repeated runs" means repeated calls or repeated processes. **That ambiguity in
   the gate's own wording should be resolved before the gate runs.**

7. **Contested: is a single "effective context" number meaningful at all?** RULER's threshold
   rule says yes; `[C]` ATLAS (2605.28079) finds 7 of 26 models shifting â‰¥2 rank positions
   between length regimes, i.e. the induced *ordering* depends on the length grid you chose.
   Since an ordering is the only thing an ablation produces, this is not a reporting
   preference. Left contested by `research/notes/evaluation-landscape.md` and left contested
   here.

8. **Contested: mean versus worst-case aggregation.** `[C]` 2510.13334 shows worst-case
   aggregation inverts the ranking of eviction policies. The aggregation function is
   therefore part of the decision rule and must be pre-registered â€” but which one is
   *right* is unsettled, and picking silently is picking your conclusion. `[M]` Â§7.3(c) adds
   a second, independent reason to care that has nothing to do with policy quality: on this
   machine the numerical noise floor is 280Ã— larger under a maximum than under a mean, so the
   two aggregations do not even have the same *instrument*.

9. **Contested: synthetic versus realistic evaluation.** Generator-based suites are
   contamination-immune and scale down but may measure an artificial skill; realistic suites
   are externally valid but static, contaminating, and unaffordable here. `[C]` 2505.19293
   and 2409.06338 attack the realistic side; `[C]` 2410.02694 (HELMET) attacks the synthetic
   side by showing NIAH does not predict downstream performance and that its seven categories
   have low mutual correlation. This lab has taken a side â€” generator-open only â€” and
   `research/notes/evaluation-landscape.md` Â§3 states it *as a position*, which is the
   correct way to hold a contested choice. Restated here so the curriculum does not read it
   as settled.

10. **There is no accepted way to report an eval's calibration record.** Papers report scores.
   Nothing reports "here is what my metric does under fault injection." A one-page
   calibration certificate attached to every eval â€” battery results, negative controls,
   known-ordering check, MDE, environment fingerprint â€” is obvious in retrospect, absent in
   practice, and cheap. Whether anybody outside this lab would adopt it is a different
   question, and I have no evidence either way.

---

## Answers to the self-check

**1.** Both effects are present and the *peeking* effect dominates. Six arms is the forking
term: `1 âˆ’ 0.95^6 = 0.265` if the arms were independent. Twenty checkpoints per arm is the
peeking term: 0.246 per arm from Â§3.3's table. Combining across six arms, the probability
that *no* arm ever crosses is roughly `(1 âˆ’ 0.246)^6 = 0.184`, so you report something with
probability about **0.82**. The estimate is crude â€” arms trained on the same data are
correlated and looks within an arm are strongly correlated, both of which push the true
number down â€” but the order of magnitude is the point: under a complete null this procedure
reports a finding four times out of five. Peeking dominates because 20 > 6 and because the
per-arm inflation (0.05 â†’ 0.246, a factor of 4.9) is larger than the cross-arm inflation
would be for six independent looks. The fix that costs nothing: pre-register one look.

**2.** More items is right when the item term still dominates the seed term. From Â§3.2,
`Var(xÌ„) = ÏƒÂ²_seed/k + ÏƒÂ²_item/(kÂ·n)`, so buying items only helps while
`ÏƒÂ²_item/n > ÏƒÂ²_seed`, i.e. while

```
n  <  ÏƒÂ²_item / ÏƒÂ²_seed
```

and it is worth buying up to `n â‰ˆ 4.76 Â· ÏƒÂ²_item/ÏƒÂ²_seed` before you are within 10% of the
wall. Note what is *not* in that inequality: the cost of either option. Adding a seed also
reduces the item term (both terms have `k` in the denominator), so a seed strictly dominates
an item at equal cost â€” the only reason to buy items is that they are three orders of
magnitude cheaper. **What you must run first:** at least two seeds of the same arm, to
estimate `ÏƒÂ²_seed` at all. With `k = 1` the ratio is unmeasurable and any answer to this
question is a guess. That experiment is unsolved item 1 in Â§10.

**3.** At 20% power the exaggeration ratio is **2.25Ã—** (Â§3.4), so your best point estimate of
the true effect is about `6 / 2.25 â‰ˆ 2.7` points, and the sign is probably right (Type S at
20% power is 0.5%). The one number to ask for is **the confidence interval, not the
p-value** â€” at p = 0.04 the interval barely excludes zero, so it runs from roughly 0 to
about 12 points, and quoting 6 as if it were the answer discards the fact that the data are
consistent with a negligible effect. If you may ask a second question, ask for the
minimum detectable effect the design was powered for; if the MDE is 6 points, the study was
constructed so that only an exaggerated estimate could be reported.

**4.** No â€” the thresholds differ by a factor of about four. From Â§3.5, `n* = logâ‚‚(D)/H` with
`D = 10â¹`, so `logâ‚‚ D = 29.90`. For natural-language sentences at `H â‰ˆ 2` bits/token,
`n* â‰ˆ 15`: use a 15-gram threshold (13 is in the right neighbourhood, which is why the
convention survives). For random 6-character symbols drawn from a large inventory, `H` is
essentially the log of the inventory size â€” at 4,096 symbols, `H = 12` bits â€” giving
`n* â‰ˆ 2.5`: a **3-symbol** match is already beyond chance. Using 13 on the needles means you
would miss a leak of four consecutive needle symbols, which is a total leak of that item.
The general rule is Â§3.5 conclusion 3: stop thresholding on length and threshold on
self-information, at `logâ‚‚ D â‰ˆ 30` bits. One rule, no free parameter, correct on both.

**5.** The shuffle result is a **pass**: fault 6 is the battery's only specificity test and a
flat response is the expected one for a genuine retrieval task. (Had it moved, you would have
been measuring discourse structure rather than retrieval.) The needle-absent result is a
**fail**, and it is far more urgent â€” it is fault 1, the cheapest test in the battery, and it
says the score is coming from the model's prior, the haystack's construction, or lexical
overlap between the query and something that is not the needle. Everything downstream of that
eval is void, including the shuffle result, because you cannot interpret the specificity of
an instrument that has no demonstrated sensitivity. Cheapest next check: compute the task's
chance level in closed form and compare. If needle-absent equals chance, you deleted the wrong
span; if it is well above chance, rebuild the item with a high-entropy value *and* no lexical
overlap with the query â€” both, because either alone leaves a shortcut open `[C]` (2502.05167).

**6.** The three: (a) `HIPBLASLT_TENSILE_LIBPATH` together with `TORCH_BLAS_PREFER_HIPBLASLT`;
(b) `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL`; (c) dtype selection interacting with
`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction`, which is a trap because
it is **inert** here â€” you will believe you controlled something you did not
(`ASSUMPTIONS.md â†’ bf16-reduced-precision-knob-works`, refuted `[M]`). The one measured to
change *numerics* rather than speed is (a): configuring hipBLASLt moves the relative error of
a length-1M bf16 weighted sum from 5.60e-3 to 2.01e-3, about **2.8Ã—**, while throughput moves
only 12% `[M]`. The lying API is
`torch.backends.cuda.flash_sdp_enabled()`, which returns `True` whether or not the
memory-efficient attention path actually ran; the only honest signal is a `UserWarning` on
stderr `[M]` (`ASSUMPTIONS.md â†’ sdpa-is-memory-efficient`). All three belong in the
`ENVIRONMENT` block of every pre-registration card (Â§4.2), because on this machine the
environment is part of the arm's identity, not metadata about it.
