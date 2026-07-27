---
title: Measuring recall â€” the task suite, its null, and why the standard needle eval cannot fail
version: 1.0.0
date: 2026-07-26
track: E â€” Post-training and evaluation
prereqs: measuring-memory, memory-failure-modes, constant-state-memory, long-context-and-effective-context
assumes: kv-eviction-policies (top-k selection under a salience score), tokenization (length is tokenizer-denominated)
difficulty: 3/5 for the math, 5/5 for the discipline â€” every number here is easy; refusing to report one you cannot resolve is not
time: 3â€“4 h reading and working the arithmetic; Exercise A 30 min, Exercise B 1â€“1.5 h, Exercise C 2â€“4 h including a training grid
mirrors: research/notes/evaluation-landscape.md Â§2, Â§3, Â§6; research/memory/constant-state-memory.md Â§3
---

# Measuring recall and memory

**Where this sits.** Track C's `measuring-memory.md` builds the *instrument* â€” the oracle
diff, the null distribution, the probe budget, the fault battery. This module builds the
*task suite* the instrument points at: what a recall eval is made of, what each of its axes
actually measures, what its chance level is in closed form, and why the one everybody runs is
structurally unable to fail for the mechanism this lab exists to break. Together they are the
evaluation half of the lab thesis. Neither is useful alone: an instrument with no task is a
thermometer in an empty room, and a task with no null is a decoration.

I do not re-derive anything from the prerequisites. In particular: the interference bound
`âˆš(d_k/(nâˆ’1))` is `constant-state-memory.md Â§3.4`; the eviction error identity
`m_EÂ·(vÌ„_E âˆ’ vÌ„_K)` is `kv-eviction-policies.md Â§3.3`; the six-fault calibration battery and
the power arithmetic are `measuring-memory.md Â§2.6` and `Â§3.4`; the adverse-selection argument
in its qualitative form is `memory-failure-modes.md Â§2.5`. What is new here is the
*quantitative* form of that argument, the multi-query composition law that follows from it,
and a source-level audit of the benchmark the field actually runs.

---

## 1. What this module settles

**One.** Recall evaluation is two instruments, not one, and the field conflates them: an
**associative-recall** task (MQAR) measures *capacity* â€” how many keyâ†’value bindings a memory
holds before interference destroys the read â€” while a **needle-retrieval** task (NIAH, RULER)
measures *localisation* â€” whether attention can find one binding among many at distance; the
first has a closed-form capacity prediction you can check your harness against, the second
does not, and that is the whole reason to run both.

**Two.** The adverse selection of NIAH against heavy-hitter eviction is not a rhetorical
point, it is a two-parameter law â€” under an exponential salience model a needle with mean
salience `s` survives top-`b` retention with probability `b^(1/s)`, and requiring `Q` facts at
once is *exactly* equivalent to requiring one fact at salience `s/Q` (a derivation, Â§3.2â€“3.3,
`[M]` checked by Monte Carlo in two fresh processes) â€” so **multi-query recall is the strictly
higher-power instrument**, and the standard scorer throws that power away.

**Three.** RULER as shipped in this repo's own reference harness hands you three of the
defects it was designed to fix â€” the question template repeats the needle's own words
(code-verified), the score is unordered substring containment over a 128-token generation so
*"emit every candidate in the context"* is a winning strategy that resolves no binding at all
(code-verified), and the harness's standard-error function returns `N/A` for every RULER metric
(`[A]` high confidence, a code trace read but not run; Exercise B confirms or refutes it) â€”
which means "we ran RULER" is a statement about which script executed, not a measurement.

**A finding this module's own exercise produced, folded in.** `[M]` Two fresh processes,
different seeds, 200,000 Monte Carlo trials per cell: with `b = 0.50`, `s = 2`, the
all-or-nothing pass rate over `Q` needles falls **0.708 â†’ 0.500 â†’ 0.250 â†’ 0.062** for
`Q = 1,2,4,8` while RULER's partial-credit mean stays **flat at 0.707 for every `Q`**. The
metric the harness reports is numerically insensitive to the exact axis that makes the task
harder. Full config in Â§6, Exercise A.

---

## 2. Theory in plain language

### 2.1 Two questions that produce the same wrong answer

Put a fact in, ask for it back, get it wrong. Two entirely different things can have happened.

**Capacity failure.** The memory *held* too many bindings at once and the read came back as a
blend. This is the constant-state failure mode â€” `constant-state-memory.md Â§3.4` gives the
signal-to-interference ratio `âˆš(d_k/(nâˆ’1))`, which says the read degrades as the square root
of how full the state is and hits parity at `n = d_k + 1`. It is a function of **how many**
things you stored, and it is almost independent of **where** they were.

**Localisation failure.** The memory held the binding perfectly â€” the KV entry is right there,
byte-exact â€” and attention did not put enough weight on it. This is the softmax-dilution and
position-bias family (`long-context-and-effective-context.md Â§3.2`). It is a function of
**where** the thing is and **how far** the query is from it, and it is almost independent of
how many other things you stored.

> **Systems bridge.** You have separated these before. A cache serving stale-or-missing data
> is either *too small for the working set* (capacity misses) or *badly indexed* (conflict
> misses, hash collisions, a hot set mapping to one way). You separate them with a counter:
> `capacity_miss` and `conflict_miss` are two different registers, and a set-associativity
> sweep tells you which one you have in an afternoon.
>
> **Where it breaks, and it is the whole reason this module exists.** There is no counter.
> Both failures emit the same thing â€” a fluent wrong answer at full speed
> (`memory-failure-modes.md Â§2.2`). The *only* way to tell them apart is to build two tasks
> that vary different axes and read the **shape of the curve**: capacity failure is a **cliff
> in the number of stored bindings**, at a location predicted in closed form; localisation
> failure is a **slope in distance or depth**, with no predicted location at all. You are not
> reading a counter, you are fingerprinting a curve â€” closer to reading a latency histogram's
> shape to distinguish queueing from a slow disk than to reading a hit-rate gauge.

### 2.2 The six axes a recall task actually has

Every synthetic recall eval in the literature is a point in the same six-dimensional design
space. Naming the axes is most of the work, because the standard needle eval pins five of them
at their easiest setting and then reports one number.

| Axis | What it varies | Which failure it excites | NIAH's default |
|---|---|---|---|
| `N` â€” bindings stored | how many keyâ†’value pairs are in the context | capacity / interference | **1** |
| `Q` â€” bindings queried | how many must be produced at once | capacity *and* joint survival under eviction | **1** |
| `H` â€” value entropy | 7-digit number vs word vs UUID | how much the model's prior can substitute for recall | number (~23 bits) |
| `Î›` â€” lexical overlap | whether the query's surface form appears in the needle | whether string matching can substitute for retrieval | **maximal** |
| `Î´` â€” depth / distance | where the target sits, and how far the query is from it | position bias, dilution, window truncation | swept (the one axis NIAH does vary) |
| `Îº` â€” haystack coherence | whether the filler is homogeneous, coherent prose, or made of distractors | whether the target's *distributional oddity* is the retrieval cue | homogeneous filler |

Two of these deserve immediate comment because they are the ones almost nobody varies.

**`Î›`, lexical overlap, is the NoLiMa axis.** `[C]` NoLiMa (2502.05167, Feb 2025) rebuilt NIAH
so the needle shares minimal literal surface with the question, forcing at least one
associative hop. Across 13 models claiming â‰¥128K, **11 fall below 50% of their own
short-context baseline at 32K**, and GPT-4o goes **99.3% â†’ 69.7%**. Same task, same lengths.
Two-thirds of the measured capability was string matching. Â§5.1 shows you `Î›` at maximum in
the reference harness's source, in two adjacent template strings.

**`Îº`, haystack coherence, is newer and cuts the other way.** `[C]` *A Controllable Examination
for Long-Context Language Models* (2506.02921, Jun 2025; surfaced by search 2026-07-26, claim
from the abstract) argues that inserting a needle into an irrelevant context *destroys the
coherence of the original document*, which is itself a shortcut: the model can find the needle
because it is the one sentence that does not belong. That is the same mechanism as
`memory-failure-modes.md Â§2.5`'s "distributionally odd span" argument, restated as a benchmark
defect rather than a policy interaction â€” and it means the "distribution shuffle" fault from
the standard battery has two possible readings, not one. `[C]` NeedleChain (2507.22411,
Jul 2025) takes the extreme position and removes the haystack entirely: every sentence in the
context is relevant.

### 2.3 Retrieval depth is three axes wearing one name

`long-context-and-effective-context.md Â§2.1` separates *length*, *distance* and *occupancy*.
For eval **design** the practical consequence is sharper than that note needed to make it:

- **Depth** is the target's fractional position, `Î´ âˆˆ [0,1]`.
- **Distance** is `(1 âˆ’ Î´)Â·T` tokens between target and query.
- **Occupancy** is `T / T_max`, how full the window is.

A conventional depth sweep holds `T` fixed and varies `Î´`, which moves **depth and distance
together** and holds occupancy constant. So a "depth curve" cannot distinguish "the model is
bad at the middle" from "the model is bad at 20,000 tokens away". Separating them needs a 2-D
grid: at least two lengths per depth, and the honest report is a surface, not a curve. `[C]`
*Positional Failures in Long-Context LLMs* (2605.23170, May 2026; search-surfaced) is the most
recent statement that this confound is still live in reasoning benchmarks.

There is a fourth thing called depth that is not a position at all, and it is the one this lab
cares about most: **rank**. `memory-failure-modes.md Â§2.5` ends by proposing rank-stratified
targets â€” placing the required fact at a controlled *attention-mass percentile* rather than at
a controlled position â€” and notes that nobody publishes it. Â§4.2 turns that proposal into an
algorithm, including the ordering hazard that makes it non-trivial.

### 2.4 Why NIAH cannot fail, in one paragraph, plus the part that is new

The argument in full is `memory-failure-modes.md Â§2.5`; compressed: heavy-hitter and
question-aware eviction policies retain the highest-scoring spans, a needle is by construction
the highest-scoring span, so the eval's target sits inside the policy's retention set at every
budget anyone deploys. The honest form of the claim is *"a low-power test with adversely
selected targets,"* not an impossibility theorem, and that is the form I defend.

**What is new since that module was written is that the coupling got tighter, not looser.**
The 2026 policy generation selects explicitly on retrieval-head attention: `[C]` CompressKV
(2606.24467, Jun 2026; search-surfaced, claim from the abstract) identifies *"Semantic
Retrieval Heads"* and keeps the tokens those heads attend to. Retrieval heads are the exact
mechanism `[C]` 2404.15574 identified as responsible for needle retrieval. So the retention
rule and the eval's pass criterion are now *the same function of the same heads*, no longer
merely correlated. A policy of the CompressKV family scored on NIAH is being graded by its own
selection function.

The generalisation worth carrying: **adverse selection is a property of the evalâ€“policy pair.**
NIAH remains informative against policies that do *not* score by salience â€” uniform eviction,
block-granular eviction, quantisation. It is uninformative against every policy that does.

### 2.5 The null is not zero, and for one RULER family it is not small

Admission criterion 3 from `research/notes/evaluation-landscape.md Â§6.1` says chance level must
be computable in closed form. Almost nobody computes it, and the reason it matters here is that
**RULER's task families have chance levels differing by roughly six orders of magnitude and are
then aggregated into one group score.**

Work them out. The value inventory is set in the generator (Â§5.1):

| Family | Value type | Inventory size | Chance of a blind hit |
|---|---|---|---|
| `niah_single_1`, `niah_single_2` | 7-digit number | 9,000,000 | ~1e-7 per guess |
| `niah_single_3`, `niah_multikey_3` | UUID4 | 2Â¹Â²Â² | 0 |
| `niah_multikey_2`, `niah_multikey_3` | number / UUID, **haystack made of distractor needles** | *the distractors in the context* | **1 / n_distractors** |

The third row is the interesting one and it is a structural property, not a quibble. When
`type_haystack="needle"` the filler is itself composed of needle-formatted lines with random
keys and values (Â§5.1). A model that emits an arbitrary magic number *copied from the context*
is right with probability `1/n_distractors`. `[A]` medium confidence, arithmetic not
measurement: at a 4,096-token budget with ~19 tokens per distractor line, `n_distractors â‰ˆ 200`,
so the blind-copy rate is around **0.5%** â€” three orders of magnitude above the essay-haystack
families in the same aggregate.

And it gets worse once you read the scorer, which is Â§2.6.

### 2.6 The scorer is the eval

`common_utils.py:43` scores by **case-insensitive substring containment of the reference inside
the prediction**, summed over references and divided by their count â€” unordered, partial
credit, no penalty for anything else the model emitted. `niah_single_1.yaml:36-37` gives the
model **128 generated tokens with no stop string**. Put those two facts together:

> **Emitting every candidate value present in the context scores 1.000 without resolving a
> single keyâ†’value binding, provided the candidates fit in 128 tokens.**

For `niah_multiquery` this is not even a shortcut, it is the intended answer set. Read the
generator (Â§5.1): `num_needle_k = max(num_needle_k, num_needle_q) = 4`, four needles are
inserted, `indices = random.sample(range(4), 4)` selects **all four**, and
`answers = [a for i in indices for a in values[i]]` is therefore **the complete set of needle
values in the context**. The task named "multi-query" asks for exactly the set of things that
are there, and the scorer does not check which key each belongs to. `niah_multivalue` is the
same shape with one key and four values. `niah_multikey_1` has four needles and one query, and
dumping all four values still scores 1.000 because a reference of length one is satisfied by
containment anywhere in the generation.

This is `[C]` 2605.11325's complaint â€” *benchmarks score answers rather than retrieval, so a
system that dumps its whole store gets perfect recall while hiding precision failure* â€”
instantiated in the standard harness's own source, machine-checkable at four line numbers.

The one family where the dumping strategy is *constrained* is the distractor-needle haystack,
because ~200 candidates do not fit in 128 tokens. `[A]` medium confidence: with a 7-digit
number costing ~4 tokens, a model can emit ~30 candidates, giving an effective blind-dump score
near **30/200 = 0.15**. That is a floor of fifteen points on a metric routinely reported to one
decimal place, and I can find no paper that states it.

> **Systems bridge, and it is the one you will feel.** This is a benchmark that measures
> **recall without precision** â€” the retrieval-systems sense of both words. You would never
> accept a search-quality report that gave credit for returning the whole index. Here the
> harness does exactly that and nobody notices, because the generation is capped at 128 tokens
> and models are polite.
>
> **Where the analogy breaks.** In a search system you can always compute precision after the
> fact: you have the returned set and the relevant set. Here the "returned set" is a natural
> language string, and defining precision requires deciding what counts as an assertion.
> That is why the fix is not a precision metric, it is a **constrained answer format** plus
> scoring the *binding* â€” did the model attach value `v_i` to key `k_i` â€” rather than
> containment. Two lines of the scorer, and it changes what the benchmark means.

### 2.7 What replaced what, and what is still contested

Compressed from `research/notes/evaluation-landscape.md Â§3`, which is the authority; the point
here is which of these you can *build*, not which to read.

- **NIAH (2023)** â€” a lexical retrieval test that saturates. Keep it as a smoke test for
  positional and truncation faults, never as evidence about eviction.
- **RULER (2024)** `[C]` 2404.06654 â€” the durable contribution is a **generator over 13
  families at controllable length**, plus a threshold definition of effective context. Cite it
  for methodology; get numbers elsewhere. It is in this repo (Â§5) and you should read its
  source before you trust its output.
- **NoLiMa (2025)** `[C]` 2502.05167 â€” the `Î›` axis. The single cheapest upgrade that turns a
  passing metric into a failing one.
- **Michelangelo (2024)** `[C]` 2409.12640 â€” latent-structure queries, where the answer is a
  function of the whole context so no span can be matched. Structurally immune to adverse
  selection. `[A]` medium confidence a 100M model can learn it at all; a one-day pilot at 1K
  context settles it and a negative closes a branch cheaply.
- **MQAR / Zoology (2023)** `[C]` 2312.04927 â€” the capacity instrument, and the only one in
  this list with a published shape to calibrate a harness against.
- **Flip-flop LM (2023)** `[C]` 2306.00946 â€” sparse, sporadic state-tracking glitches. The
  tail-latency instinct applied to correctness, and the natural companion to worst-case
  aggregation `[C]` 2510.13334.
- **Sequential-NIAH (2025)** `[C]` 2504.04713 (search-surfaced) â€” needles that must be returned
  in order, which is the cheapest way to make the dumping strategy of Â§2.6 fail.

**Contested, and staying contested.** Whether the future is more synthetic or more realistic is
unresolved (`evaluation-landscape.md` Contested Â§1). Whether associative recall predicts
anything downstream is unresolved: Zoology attributes **82% of a 2.1-point Pile perplexity gap**
to associative-recall tokens `[C]` 2312.04927, which is the strongest transfer evidence
available, and `[C]` 2508.19029 revisits the claim. No source demonstrates that an MQAR capacity
curve *quantitatively* predicts a downstream curve, and this module does not assume it does.

---

## 3. The math that actually matters

### 3.1 Symbols

| Symbol | Reads as |
|---|---|
| `T` | context length in cache entries (tokens) |
| `b` | retention budget as a **fraction** of entries kept, `b âˆˆ (0,1]` |
| `s` | **salience** of a target span: its mean attention mass in units of the background mean |
| `m_j` | attention mass on background entry `j`, modelled as `Exponential(1)` |
| `M` | attention mass on the target, modelled as `sÂ·E` with `E ~ Exponential(1)` |
| `p` | probability that **one** target survives the retention rule |
| `Q` | number of targets that must **all** be recovered for the item to count as correct |
| `N` | number of keyâ†’value bindings stored in the context |
| `d_k` | key dimension of a superposed (state-based) memory |
| `n` | number of eval items |
| `d` | standardized effect size: mean shift divided by per-item standard deviation |
| `V_val` | size of the value inventory the answer is drawn from |
| `Ïƒ` | per-item standard deviation of a metric |

### 3.2 The survival law, derived

`measuring-memory.md` Exercise C models the target's mass as a **deterministic** multiple `s` of
the background mean, and gets a step function: the needle survives iff `s > âˆ’ln b`. That is
correct and it is the right first cut. It is also degenerate â€” a real needle's attention mass is
a random variable like everything else's, and restoring that variability turns the step into a
curve with a closed form. This section **refines** that result; it does not contradict it.

Model. Background masses `m_j ~ Exponential(1)`, i.i.d. across `T` entries. A top-`b` policy
keeps the largest `bT` of them, so for large `T` the survival threshold is the `(1âˆ’b)` quantile
of `Exponential(1)`:

```
q(b) = âˆ’ln(b)
```

- `q(b)` â€” the mass an entry must exceed to be kept.
- `âˆ’ln(b)` â€” minus the natural log of the budget fraction. At `b = 0.10`, `q = 2.303`.

Give the target mass `M = sÂ·E` with `E ~ Exponential(1)`, so its mean is `s` times the
background mean. Then

```
p  =  P(M > q(b))
   =  P(sÂ·E > âˆ’ln b)
   =  exp( âˆ’(âˆ’ln b)/s )
   =  b^(1/s)
```

**A target of salience `s` survives a top-`b` policy with probability `b^(1/s)`.** Three sanity
readings, all of which must hold or your implementation is wrong:

- `s = 1` (the target is indistinguishable from background) gives `p = b`. Correct: it is one
  entry among equals, kept with probability equal to the budget fraction.
- `s â†’ âˆž` gives `p â†’ 1`.
- `b = 1` (keep everything) gives `p = 1` for every `s`.

| `b` (kept) | `s = 1` | `s = 2` | `s = 5` | `s = 10` |
|---|---|---|---|---|
| 0.50 | 0.500 | 0.707 | 0.871 | 0.933 |
| 0.25 | 0.250 | 0.500 | 0.758 | 0.871 |
| 0.10 | 0.100 | 0.316 | 0.631 | 0.794 |
| 0.05 | 0.050 | 0.224 | 0.549 | 0.741 |
| 0.01 | 0.010 | 0.100 | 0.398 | 0.631 |

Read the `s = 10` column â€” a needle ten times more salient than ordinary text â€” and notice that
even at a **99% eviction rate** it survives 63% of the time. That is the adverse-selection
claim as a number rather than an argument.

### 3.3 The composition law: Q facts at salience s is one fact at salience s/Q

Now require `Q` independent targets, all of which must survive:

```
p_all  =  p^Q  =  ( b^(1/s) )^Q  =  b^(Q/s)
```

Compare with the single-target law `b^(1/s')`. They are identical when `s' = s/Q`:

> **Requiring `Q` facts of salience `s` is exactly as hard, under a top-`b` retention rule, as
> requiring one fact of salience `s/Q`.**

That single line is the quantitative case for multi-query recall as an eviction probe, and it
is why the field's habit of testing eviction with single-needle NIAH is a choice to run the
weakest available instrument. Worked, at `b = 0.10`:

| `s` | `Q = 1` | `Q = 2` | `Q = 4` | `Q = 8` | equivalent single-needle salience at `Q = 8` |
|---|---|---|---|---|---|
| 2 | 0.316 | 0.100 | 0.010 | 1.0e-4 | 0.25 |
| 5 | 0.631 | 0.398 | 0.158 | 0.025 | 0.63 |
| 10 | 0.794 | 0.631 | 0.398 | 0.158 | 1.25 |

The `s = 10, Q = 8` cell is the design target: a *salient* needle set, at a *deployable* 10%
budget, failing 84% of the time. Single-needle NIAH at the same budget and salience passes 79%
of the time. Same policy, same context, same salience â€” one instrument sees the damage and the
other does not.

`[M]` The operationally sharpest cell is the low-salience one, because it is what a NoLiMa-style
associative needle looks like. At `b = 0.25` â€” a **75% retention budget**, i.e. barely
compressing at all â€” and `s = 1.2`, the all-`Q` pass rate measured over 200,000 trials in two
fresh processes is **0.317 / 0.100 / 0.009 / 0.0001** for `Q = 1,2,4,8`, against a closed form of
0.315 / 0.099 / 0.0098 / 0.0001. Four weakly-salient facts are effectively unrecoverable at a
budget nobody would call aggressive. The partial-credit scorer reads **0.315 at every `Q`**.

**Where the model is a stipulation, and it matters.** `Exponential(1)` for background attention
mass is an assumption, inherited from `measuring-memory.md` Exercise C and flagged there as
unmeasured. A heavier tail moves `q(b)` and therefore every number above. Independence across
the `Q` targets is also a stipulation and is probably optimistic â€” targets inserted into one
context share a haystack and a query, so their survival is positively correlated, which makes
`p^Q` a *lower* bound on the joint pass rate and the composition law a *conservative* statement
of multi-query's advantage. Both assumptions are listed in Â§8.

### 3.4 Partial credit throws the advantage away

RULER's scorer gives partial credit: with `Q` references, the item's score is the **fraction**
recovered, not an indicator that all were. Write down what each metric's distribution is, per
item, under the survival model:

- **All-or-nothing.** Bernoulli with mean `u = p^Q`. Variance `u(1âˆ’u)`.
- **Partial credit.** Mean of `Q` Bernoulli(`p`) draws. Mean `p`, variance `p(1âˆ’p)/Q`.

The mean of the partial-credit metric **does not depend on `Q` at all**. Increasing the number
of queried facts makes the task strictly harder and moves the reported number not one bit.

Now compare statistical power. Take the reference arm to be the full cache (`p = 1`, zero
variance), which is the paired design this lab uses anyway. The standardized effect â€” mean shift
over per-item standard deviation â€” is

```
d_exact    =  (1 âˆ’ u) / sqrt( u(1âˆ’u) )      =  sqrt( p^(âˆ’Q) âˆ’ 1 )
d_partial  =  (1 âˆ’ p) / sqrt( p(1âˆ’p)/Q )    =  sqrt( QÂ·( p^(âˆ’1) âˆ’ 1 ) )
```

- `d_exact` â€” standardized effect of the all-or-nothing score.
- `d_partial` â€” standardized effect of the partial-credit score.
- `u = p^Q` â€” probability all `Q` survive.

Which is bigger? Put `x = 1/p â‰¥ 1` and compare `x^Q âˆ’ 1` against `Q(x âˆ’ 1)`. Factor:
`x^Q âˆ’ 1 = (x âˆ’ 1)(x^(Qâˆ’1) + x^(Qâˆ’2) + â€¦ + 1)`, and that bracket has `Q` terms each `â‰¥ 1`, with
strict inequality for `x > 1`, `Q > 1`. Therefore

```
x^Q âˆ’ 1  >  Q(x âˆ’ 1)          for all  x > 1,  Q > 1
```

**All-or-nothing scoring dominates partial credit on standardized effect, for every `p < 1` and
every `Q > 1`.** Computed:

| `p` | `Q` | `p^Q` | `d_exact` | `d_partial` | ratio | `n` needed for `nÂ·p^Q â‰¥ 10` |
|---|---|---|---|---|---|---|
| 0.95 | 4 | 0.815 | 0.477 | 0.459 | 1.04 | 13 |
| 0.95 | 8 | 0.663 | 0.712 | 0.649 | 1.10 | 16 |
| 0.90 | 8 | 0.431 | 1.150 | 0.943 | 1.22 | 24 |
| 0.75 | 4 | 0.316 | 1.470 | 1.155 | 1.27 | 32 |
| 0.75 | 8 | 0.100 | 2.998 | 1.633 | 1.84 | 100 |
| 0.50 | 4 | 0.063 | 3.873 | 2.000 | 1.94 | 160 |
| 0.50 | 8 | 0.0039 | 15.969 | 2.828 | 5.65 | 2,560 |

**And now the caveat that stops this from being a blanket rule.** The last column is the
constraint. The standardized-effect formula assumes a normal approximation, which needs the
expected number of *successes* to be non-trivial â€” the usual working rule is `nÂ·u â‰³ 10`. As
`Q` grows, `u = p^Q` collapses and the all-or-nothing metric **floors**: every item in both arms
scores zero and the arms are indistinguishable regardless of how large `d_exact` looks on paper.

That resolves an apparent conflict with `research/notes/evaluation-landscape.md Â§5.1`, which
(correctly, citing `[C]` 2304.15004) says to prefer continuous metrics at small scale. Both are
right, in different regimes:

> **Near the ceiling** â€” the model is competent and the *policy* is what you are perturbing â€”
> all-or-nothing over `Q` targets is the higher-power metric. **Near the floor** â€” the model
> itself cannot do the task â€” continuous metrics are the only ones that move at all.
> **The design rule is one line: choose `Q` such that `p^Q â‰¥ 10/n`, and report both scorers
> always**, because the pair tells you which regime you are in and neither does alone.

At a plausible `n = 500` and `p = 0.9`, `10/n = 0.02` and `p^Q â‰¥ 0.02` allows `Q â‰¤ 37`. At
`p = 0.5` it allows `Q â‰¤ 5`. The rule bites exactly when the model is weak, which at 20Mâ€“300M
is most of the time.

### 3.5 What MQAR's curve certifies, and the anchor that catches a broken harness

Do not re-derive the interference bound; `constant-state-memory.md Â§3.4` has it. Take from it
the one prediction an eval harness can be checked against:

- A **superposed** memory of key dimension `d_k` reads correctly while `N â‰² d_k` and degrades
  as `âˆš(d_k/(Nâˆ’1))` beyond it. The observable is a **cliff in `N` at roughly `d_k`**.
- A **softmax attention** memory keeps `N` entries separate and suppresses non-matching terms
  exponentially. Its capacity is bounded by context length, not by `d_k`. The observable is
  **no cliff in `N` at all**.

That second line is the anchor, and it is the most valuable thing MQAR gives you:

> **Softmax attention must score â‰ˆ1.0 on MQAR at every `N` your context can hold, at a model
> dimension as small as 64.** `[C]` Zoology (2312.04927) reports exactly that across all tested
> sequence lengths. If your softmax control does not reach â‰ˆ1.0, **stop** â€” your harness is
> broken and no other column in your table means anything.

This is not hypothetical. Two published MQAR tables fail their own control:

- `[C]` Variational Linear Attention (2605.11196, May 2026) reports softmax attention at
  **0.152 with 8 pairs** at head dimension 32 â€” a task Zoology shows attention solving
  essentially perfectly. `research/memory/constant-state-memory.md Â§3` already flags this;
  I repeat it because it is the canonical example.
- `[C]` Echo (2605.06997, May 2026; search-surfaced, claim from the abstract) reports a pure
  Mamba-2 failing to exceed **chance accuracy (~3%)** on MQAR *across all gap lengths and pair
  counts*, while the same family scores 59.2 on 1K multi-key retrieval in `[C]` 2605.22791's
  table. Those two results are not obviously reconcilable, and the resolution is almost
  certainly harness configuration â€” training steps, head dimension, curriculum â€” rather than
  architecture.

Two papers, three months apart, whose MQAR baselines contradict the reference result and each
other. Reproduce the published *shape* before trusting a number your own harness produced.
That is Exercise C.

**MQAR's chance level, in closed form.** If the value is drawn uniformly from an inventory of
`V_val` symbols and the model must emit `Q` of them:

```
chance(all-or-nothing)  =  V_val^(âˆ’Q)
chance(partial credit)  =  1 / V_val
```

Two different nulls for the same task, differing by a factor of `V_val^(Qâˆ’1)`. Report which one
you used or the number is not interpretable.

### 3.6 Effective context is a hypothesis test, and RULER does not report its variance

RULER's threshold rule â€” *effective context is the longest length at which the model still beats
a fixed short-context reference* â€” is a comparison of two numbers, which makes it a hypothesis
test whether or not anyone treats it as one. The reference is Llama-2-7B's **85.6 at 4K**
(`long-context-and-effective-context.md Â§3.4`).

The generator's own constant gives you the sample size: `num_samples = 500` per task per length
(`prepare_niah.py:222`; each family passes it explicitly at e.g. `niah_utils.py:33`). For a
binary metric at `p â‰ˆ 0.856`:

```
Ïƒ      =  sqrt( p(1âˆ’p) )        =  sqrt(0.856 Ã— 0.144)  =  0.351
stderr =  Ïƒ / sqrt(n)           =  0.351 / sqrt(500)    =  0.0157
95% CI =  Â±1.96 Ã— stderr        =  Â±0.031
```

**Â±3.1 accuracy points at RULER's own default sample size**, which means an "effective context
length" reported from a single seed is resolvable only to the length bucket at which the score
moves by more than ~3 points. Under the paired design (`measuring-memory.md Â§3.4`, `Ï = 0.9`)
the minimum detectable effect improves to

```
Ïƒ_d  =  sqrt( 2ÏƒÂ²(1âˆ’Ï) )  =  sqrt(2 Ã— 0.1233 Ã— 0.1)  =  0.157
MDE  =  2.8016 Â· Ïƒ_d / sqrt(n)  =  2.8016 Ã— 0.157 / 22.36  =  0.020
```

â€” 2.0 points, and only if everything except the arm is held identical.

**The harness does not report either number.** Â§5.3 traces it: RULER's aggregation is a custom
function, `stderr_for_metric` only knows how to produce a standard error for a whitelist of
aggregations plus `mean` and `acc_all`, and `evaluator_utils.py:214` writes the string `"N/A"`
when the lookup misses. `[A]` high confidence â€” this is a code trace, read but not run, and it
is Exercise B's first deliverable to confirm or refute by running it.

### 3.7 What a recall eval costs on this machine

Arithmetic from `[M]` inputs, not a measurement. Inputs: **20.9 TFLOP/s bf16 GEMM at 8192Â³**
(`ASSUMPTIONS.md â†’ gemm-throughput-below-reference`), **~200 GB/s fast tier out to â‰¥62 GiB**
(`gpu-fast-tier-size`), a 300M-parameter model at ~600 MB of bf16 weights.

**Prefill.** Weight FLOPs are `2N = 6e8` per token. `[A]` medium confidence that an eval loop
reaches ~20% of GEMM peak (this is the `evaluation-landscape.md Â§7` assumption, unmeasured;
cheapest test is one timed forward-pass benchmark, which the Hardware Validation Gate needs
anyway):

```
4.18e12 FLOP/s Ã· 6e8 FLOP/token  â‰ˆ  7,000 tok/s
```

RULER's default is one length bucket (`common_utils.py:15`), 13 tasks Ã— 500 items = 6,500 items
at 4,096 tokens = **26.6M tokens â‰ˆ 63 minutes of prefill per arm.**

**Decode, and the config line that costs you 40 minutes.** Batch-1 decode is bandwidth-bound:
600 MB at 200 GB/s is a **333 tok/s** roofline before any KV traffic. `niah_single_1.yaml:36-37`
sets `max_gen_toks: 128` with `until: []` â€” **no stop string, so every item generates all 128
tokens** even though the answer is one number:

```
6,500 items Ã— 128 tokens Ã· 333 tok/s  =  2,500 s  â‰ˆ  42 minutes
```

Set a stop string and the answer costs ~10 tokens: **3.3 minutes**. One line of YAML, a 13Ã—
reduction in the decode half of the eval, and it changes nothing about the score. This is the
kind of thing you only find by reading the config rather than the paper.

**The better move is not to generate at all.** Score the answer span by log-likelihood under
teacher forcing: decode cost goes to zero, the metric becomes continuous, and Â§3.4's floor
problem softens. The cost is that you no longer measure the model's ability to *produce* the
answer, only its ability to *prefer* it. Record which you did.

**Capacity is not binding here, and the â‰¥32 GiB fault is not either â€” until it is.** At
`T = 4,096` nothing is close to the limits. Inherited from `measuring-memory.md Â§3.6`: if you
ever materialise the attention score matrix for attribution at batch 1, 8 heads, bf16, the
tensor crosses 32 GiB at `T = 46,341` and the documented failure is `[M]` a **hang at 0% CPU
with no error** (`ASSUMPTIONS.md â†’ large-tensor-fault-32gib`). A scaled RULER whose length grid
reaches 48K, run with attribution logging on, stalls silently.

**And a numerics control that must be in the run record.** `[M]` `ASSUMPTIONS.md â†’
hipblaslt-config`: the relative error of a length-1M bf16 weighted sum against an fp64 reference
is **2.01e-3 with hipBLASLt configured and 5.60e-3 without** â€” a 2.8Ã— swing, reproduced across
three seeds and fresh processes. A long-context recall score taken without hipBLASLt configured
is confounded by arithmetic, and the confound grows with reduction length, which is to say with
context. Every eval run must record it.

---

## 4. Why it matters for Proteus and Mnemosyne

### 4.1 The suite, and what each member is for

The lab's deliverable is an instrument, not a policy (`research/synthesis.md`). An instrument
needs a task suite whose null is computable and whose members fail for *different* reasons.
Four members, each earning its slot by covering an axis the others do not:

| Member | Axis it owns | Why it is in the suite | Chance level |
|---|---|---|---|
| **MQAR** | `N`, `Q` | the only member with a closed-form capacity prediction, so it is the harness's calibration anchor | `V_val^(âˆ’Q)` or `1/V_val` |
| **Rank-stratified needles** | rank (Â§4.2) | the only member that measures the evalâ€“policy coupling directly instead of hoping it is absent | computable per value inventory |
| **NoLiMa-style associative needles** | `Î›` | removes the string-matching shortcut; the cheapest upgrade that turns a passing metric into a failing one | as above |
| **Scaled RULER-VT (variable tracking)** | multi-hop `Î´` | the answer is a chain, so no single span is the target and dumping does not work | `1/|var inventory|^hops` |

Everything else is read for methodology and not run (Â§4.5).

### 4.2 The rank-stratified generator, as an algorithm, including the hazard

`memory-failure-modes.md Â§2.5` proposes placing the target at controlled attention-mass
percentiles and notes nobody publishes it. Here is the procedure, and the reason it is not as
easy as it sounds.

```
1. Build the haystack WITHOUT the needle. Tokenise.
2. Run one full-cache forward pass. At the query position, accumulate per-entry
   attention mass  a_j  over the layers and heads the policy scores on.
   (Which layers and heads is a CHOICE and must be recorded: a policy that scores
   per-head does not see the same ranking as one that scores per-layer.)
3. Bin candidate insertion positions by percentile of  a_j.
4. Insert the needle at a position in the target bin. Re-tokenise.
5. Run a SECOND full-cache forward pass and MEASURE the needle's own mass percentile.
6. Report accuracy against the measured percentile from step 5, never the target
   percentile from step 3.
```

**Step 5 is the hazard and it is why this is a contribution rather than a chore.** Inserting the
needle changes the attention distribution â€” that is the entire point of a needle â€” so the
percentile you aimed for in step 3 is not the percentile you got. Worse, softmax is a
normaliser: adding a high-mass entry *reduces* every other entry's mass, so the whole ranking
shifts. Reporting the target percentile instead of the measured one would silently misreport the
independent variable. `[A]` high confidence in the mechanism (it is softmax renormalisation,
the same algebra as `kv-eviction-policies.md Â§3.2`); no confidence at all about the magnitude of
the shift at our scale. Measuring that shift is itself a publishable half-day.

Cost: two forward passes per item instead of one. At Â§3.7's ~7,000 tok/s that doubles prefill
to ~2 hours per arm for a full RULER-sized suite, which is affordable and is exactly the kind of
thing `evaluation-landscape.md Â§6.3` means by *"the small-scale rig is advantaged, not merely
cheaper."*

### 4.3 Where the suite lives, under the boundary rule

The boundary is `mnemosyne â†’ torch`, never `mnemosyne â†’ proteus` (CLAUDE.md). Applied here:

- **Themis owns the task generators and the harness.** They need a tokenizer, a model and a
  config surface. `themis/` is the right home, and the generators are ordinary Python with a
  seed.
- **Mnemosyne owns the scorers that are tensor functions.** Per-token KL against a reference,
  the answer-span log-likelihood, the per-decision `m_E` and `â€–vÌ„_E âˆ’ vÌ„_Kâ€–â‚‚`, and the survival
  counters of Â§3.2 â€” all of these take tensors and return tensors, and none of them needs to
  know what a needle is. Keeping them Mnemosyne-side is what lets the instrument be pointed at
  a different model, which is the difference between a contribution and an implementation
  detail.
- **The survival law is a Mnemosyne-side observable.** `p` is directly measurable inside the
  eviction hook: for each decision, was the target entry in the retained set? That is a boolean
  the policy already computes. Logging it turns Â§3.2 from a model into a measurement, and it
  costs one bit per decision.

### 4.4 The eval run record, as an interface obligation

Every recall number this lab reports must carry, or it is uninterpretable:

```
generator_seed, harness_seed(s)      lm-eval takes FOUR (evaluator.py:81-84) and RULER's
                                     generators read the GLOBAL random module, so the eval
                                     ITEMS are a function of the harness seed
tokenizer identity + hash            length buckets are denominated in the model's own tokens
attention_backend                    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL, and note that
                                     flash_sdp_enabled() lies about which path ran  [M]
dtype, and the bf16-vs-fp32 null     bf16-numerics-unproven is still untested
hipblaslt_configured                 [M] 2.8x on long-reduction accuracy, not a speed knob
scorer                               containment / binding-aware / all-or-nothing / span-LL
chance_level                         closed form, per family, computed not assumed
n, and effective n after clustering  question-level clustering per [C] 2411.00640
MDE                                  the pre-registration card does not have this field yet
```

The seed row is not boilerplate. RULER's item generation calls the **global** `random` module
(`prepare_niah.py:91`, `:95`, `:100`) and `np.random` (`vt_utils.py:63`), which
`simple_evaluate` seeds at `evaluator.py:200` and `:204`. **Two runs with different harness
seeds are not two samples of one benchmark; they are two different benchmarks.** That is fine â€”
it is what "generator-open" buys you â€” but it must be reported as a *nuisance axis* in the sense
of `measuring-memory.md Â§2.7`, not ignored.

### 4.5 What our hardware forbids, said plainly

- **No agentic or LLM-judged eval, at any budget.** No local judge exists at a size we can run
  alongside an experiment without distorting its cost model, and single-device means no
  judge-and-subject in parallel. `evaluation-landscape.md Â§4` is unhedged about this and I agree.
- **No multi-node or multi-GPU eval sharding.** `[C]` `single-device-only`: collectives are
  incomplete on gfx1151. A 6,500-item suite runs sequentially or not at all.
- **No claim about bf16 recall numbers until the gate runs.** `bf16-numerics-unproven` is
  `untested`. Every recall result must be taken in fp32, or in bf16 *with* the fp32 arm as its
  null.
- **No frontier comparator.** We cannot run HELMET, LongBench Pro, ATLAS or any static suite at
  a scale where the numbers mean anything, so "our effective context is X" has no external
  anchor. The threshold must be anchored to **our own short-context baseline**, and that
  substitution must be declared every time.

---

## 5. Read the code

Paths relative to `research/reference/`. Clones are gitignored; run `scripts/fetch_reference.sh`
first. Line numbers are pinned to the revisions in `PROVENANCE.md` â€”
`lm-evaluation-harness` at `f4d4b3de3ee6`, `nanogpt` at `3adf61e154c3`, `transformers` at
`b6d5084fb4a5`.

### 5.1 RULER's generator â€” the benchmark you are told to use, read as source

This is the most valuable read in the module. Everything in Â§2.5, Â§2.6 and Â§3.6 is visible here.

| Where | What to look at, and why |
|---|---|
| `training/lm-evaluation-harness/lm_eval/tasks/ruler/prepare_niah.py:45`<br>`NEEDLE = "One of the special magic {type_needle_v} for {key} is: {value}."` | The needle template. Now read the next row **immediately**. |
| `training/lm-evaluation-harness/lm_eval/tasks/ruler/niah_utils.py:11`<br>`TEMPLATE = """Some special magic {type_needle_v} are hiddenâ€¦What are all the special magic {type_needle_v} for {query}â€¦"""` | The question template. **"special magic numbers" appears verbatim in both.** The `Î›` axis of Â§2.2 is pinned at maximum by construction, in two adjacent files. This is the thing NoLiMa exists to remove, and you can see it without running anything. |
| `prepare_niah.py:58`<br>`DEPTHS = list(np.round(np.linspace(0, 100, num=40, â€¦)))` | The depth grid: 40 depths, 0â€“100%. Note what is *not* here â€” a second length per depth, which is what Â§2.3 says you need to separate depth from distance. |
| `prepare_niah.py:88`<br>`def generate_random_number(num_digits=7)` | The value inventory: 9,000,000 seven-digit numbers. This is the closed-form chance level of Â§2.5, and it is one line. |
| `prepare_niah.py:172`<br>`elif type_haystack == "needle":` | The haystack is itself made of needle-formatted lines with random keys and values. **This is a built-in distractor control that the benchmark does not advertise as one** â€” it is the only family where the target's surface form is not distinctive, and it is therefore the only family whose result speaks to salience-based retention. It is also the family with a non-trivial chance level (Â§2.5). |
| `prepare_niah.py:188`<br>`indices = random.sample(range(num_needle_k), num_needle_q)` | For `niah_multiquery`, `num_needle_k == num_needle_q == 4`, so this samples **all four** indices and the answer set becomes the complete set of needle values in the context. Â§2.6's dumping strategy is not a loophole, it is the specification. |
| `prepare_niah.py:143`<br>`random.Random(random_seed).shuffle(needles)` | A *fixed* seed (42) re-created per call, so for a given needle count the layout permutation is identical across all 500 samples. The depths vary (`:155` draws from `DEPTHS` with the global `random`); the order does not. A nuisance factor held constant by accident. |
| `prepare_niah.py:222`<br>`num_samples: int = 500` | The sample size, and therefore the denominator of every standard error Â§3.6 computes. Each family passes it explicitly â€” e.g. `niah_utils.py:33`. |
| `prepare_niah.py:262`<br>`total_tokens = len(TOKENIZER(input_text + â€¦).input_ids)` | Length buckets are denominated in **the model's own tokenizer**. "4K context" is a different amount of text for a 32K-vocab char-level model than for a 128K-vocab BPE. Cross-model length comparisons are not apples to apples, and `tokenization.md` is the prerequisite that tells you by how much. |
| `prepare_niah.py:318`<br>`if formatted_output["outputs"][0] not in formatted_output["input"]:` â†’ `assert False` on `:319` | A harness self-check that the needle **is** present. Note carefully what is absent: there is no path that constructs the needle-**absent** item. The generator validates the positive control and does not ship the negative one â€” fault #1 of the standard battery, missing from the standard benchmark. |
| `prepare_niah.py:332`<br>`datasets.load_dataset("baber/paul_graham_essays", â€¦)` | The essay haystack is a network download. Plan for it or use `type_haystack="repeat"`, which is self-contained. |

### 5.2 The scorer â€” where precision goes missing

| Where | What to look at, and why |
|---|---|
| `training/lm-evaluation-harness/lm_eval/tasks/ruler/common_utils.py:43`<br>`def string_match_all(preds, refs)` | Read all eight lines. `1.0 if r.lower() in pred.lower() else 0.0` â€” **unordered, case-insensitive substring containment**, averaged over references. There is no penalty for anything else in the prediction, and no check that value `v_i` was attached to key `k_i`. Â§2.6 is this function. |
| `training/lm-evaluation-harness/lm_eval/tasks/ruler/common_utils.py:53`<br>`def string_match_part(preds, refs)` | `max([...]) / len(preds)` â€” a maximum divided by a count, which is only meaningful because `preds` is always length 1 at every call site. Not a bug today; a landmine if anyone ever batches. Worth noticing because it tells you how much scrutiny this layer has had. |
| `training/lm-evaluation-harness/lm_eval/tasks/ruler/common_utils.py:65`<br>`metrics = {str(length): -1.0 for length in DEFAULT_SEQ_LENGTHS}  # hacky` | Length buckets with no samples report `-1.0`, filtered later at `:84`. A sentinel value inside a metric's own range is a reporting hazard; if the filter is ever bypassed, `-1` averages in as a score. |
| `training/lm-evaluation-harness/lm_eval/tasks/ruler/common_utils.py:15`<br>`DEFAULT_SEQ_LENGTHS = [4096]` | The default is **one** length bucket, while `niah_single_1.yaml:14-32` declares six metrics up to 131072. The declared surface and the default behaviour differ, which is how a table acquires empty columns nobody notices. |
| `training/lm-evaluation-harness/lm_eval/tasks/ruler/niah_single_1.yaml:34-35`<br>`do_sample: false` / `temperature: 0.0` | Greedy. Good â€” it removes sampling as a nuisance axis, and it is why `repeats: 1` at `:38` is defensible. |
| `training/lm-evaluation-harness/lm_eval/tasks/ruler/niah_single_1.yaml:36`<br>`max_gen_toks: 128` and `:37` `until: []` | **No stop string.** Every item generates 128 tokens for a 1-token answer. Â§3.7 prices this at ~40 minutes per arm on our machine, and Â§2.6 shows it is also what gives the dumping strategy room to work. Two separate defects from one config line. |
| `training/lm-evaluation-harness/lm_eval/tasks/ruler/ruler.yaml:16-18`<br>`aggregate_metric_list: - metric: "4096"  weight_by_size: False` | The group score is an **unweighted mean over 13 tasks whose chance levels differ by six orders of magnitude** (Â§2.5). One number, incommensurable parts. |

### 5.3 Where the error bar is supposed to come from, and does not

| Where | What to look at, and why |
|---|---|
| `training/lm-evaluation-harness/lm_eval/api/metrics.py:318`<br>`def mean_stderr(arr)` | The closed form, two lines, correct: `sample_stddev(arr) / sqrt(len(arr))`. |
| `training/lm-evaluation-harness/lm_eval/api/metrics.py:555`<br>`def stderr_for_metric(metric, bootstrap_iters)` | Read the docstring, then the body. Standard errors are produced for a **whitelist** (`:571`) or for two specific aggregation functions (`:585`), and otherwise the function returns `None`. |
| `training/lm-evaluation-harness/lm_eval/api/metrics.py:585`<br>`stderr = {mean: mean_stderr, acc_all: acc_all_stderr}` | The lookup is keyed on the **function object**, not on a name. RULER's aggregation is `common_utils.aggregate_metrics`, which is neither key. Trace it: `stderr_for_metric` returns `None`. |
| `training/lm-evaluation-harness/lm_eval/evaluator_utils.py:207-214`<br>`stderr_fn = stderr_for_metric(metric=agg_fn, â€¦)` â€¦ `else "N/A"` | Where the `None` lands. `[A]` high confidence, read not run: **every RULER metric in this harness reports `"N/A"` for its standard error.** Exercise B's first deliverable is to run it and confirm or refute. |
| `training/lm-evaluation-harness/lm_eval/api/metrics.py:619`<br>`### don't use this unless a statistician has told you it's the right thing to do ###` | Inside `combined_sample_stderr`. Read the surrounding comment: the formula was removed from use because it *"seems to give erroneously huge stderrs for groupings of tasks."* An honest comment about an unresolved problem, in the most widely used eval harness in the field. |
| `training/lm-evaluation-harness/lm_eval/api/metrics.py:640-643`<br>`def aggregate_subtask_metrics(â€¦)  # TODO: does not hold for non-mean aggregations` | The group aggregation, with a TODO stating that it is invalid for exactly the case RULER uses. |
| `training/lm-evaluation-harness/lm_eval/evaluator.py:81-84`<br>`random_seed`, `numpy_random_seed`, `torch_random_seed`, `fewshot_random_seed` | Four seeds, defaults `0` and `1234` (`defaults.py:7-8`). For a generator-based task these seeds change **the items**, not just the order. Â§4.4. |

### 5.4 Multi-hop, read as a generator

`ruler_vt` is the family worth building on, because its answer is a chain rather than a span.

| Where | What to look at, and why |
|---|---|
| `training/lm-evaluation-harness/lm_eval/tasks/ruler/vt_utils.py:63`<br>`this_chain = [f"VAR {this_vars[0]} = {np.random.randint(10000, 99999)}"]` | The first link binds a variable to a literal 5-digit value. That value is the query (`:114`), so **hop 1 has a lexical anchor** and hops 2..`num_hops` do not. The task is partially NoLiMa-resistant by construction, which is unusual and worth stealing. |
| `training/lm-evaluation-harness/lm_eval/tasks/ruler/vt_utils.py:65`<br>`this_chain.append(f"VAR {this_vars[j+1]} = VAR {this_vars[j]} ")` | The alias chain. `num_hops = 4` by default (`:135`), so the answer is 5 variable names and dumping does not help: the model must produce a *set defined by transitive closure*, and the distractor variables are drawn from the same 5-uppercase-letter inventory (`:52`). |
| `training/lm-evaluation-harness/lm_eval/tasks/ruler/vt_utils.py:92`<br>`positions = list(sorted(random.sample(range(len(sentences)), len(chain_i))))` | Chain links are scattered at random depths but kept **in order**. Reversing that order is a one-line fault injection that should collapse the score for a true multi-hop task and not for a lookup-table shortcut. Nobody runs it. |
| `training/lm-evaluation-harness/lm_eval/tasks/ruler/vt_utils.py:229-234`<br>`icl_example = sys_vartrack_w_noise_random(â€¦, max_seq_length=500, â€¦)` | One in-context example is generated at 500 tokens and reused for every item, with its answer tokens randomised per item (`randomize_icl`, `:120`). So the prompt has a fixed structural prefix â€” which is a **prefix-cache hit** in any serving path, and therefore a confound if you ever compare cache-policy arms on this task. |

### 5.5 Where a recall eval breaks against the model you will actually run

| Where | What to look at, and why |
|---|---|
| `training/nanogpt/model.py:314`<br>`idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]` | **Silent left-truncation.** Feed a 4,096-token RULER item to a `block_size = 256` nanoGPT and the front of the context â€” where the needle usually is â€” is *deleted with no warning*. The eval then measures the model's prior, which is fault #1 of the battery, arrived at by accident. This is the single most likely way a scaled-RULER exercise produces a confident zero. |
| `training/nanogpt/model.py:128`<br>`wpe = nn.Embedding(config.block_size, config.n_embd)` | Why the truncation is not optional: there is no positional embedding row for index `block_size`. There is no fault-in path; eviction-by-truncation is hardcoded and the policy is "drop the oldest." |
| `training/nanogpt/train.py:216`<br>`def estimate_loss()` | The continuous metric Â§3.4 says to report alongside the discrete one. Note `eval_iters` defaults to 200, so the published `1.4697` (`nanogpt/README.md:51`) is a Monte Carlo mean with a standard error â€” a threshold without a variance estimate is a coin flip. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:366`<br>`self.sliding_window = config.sliding_window if self.is_local_attention else None` | For a Laguna-shaped Proteus, 36 of 48 layers have a 512-token window `[M]` (`ASSUMPTIONS.md â†’ reference-model`). **A needle more than 512 tokens back is architecturally unreadable by three quarters of the model**, so a long-context recall score is a measurement of 12 layers. Not a defect â€” but if you ablate the SWA ratio and the recall curve moves, you have not necessarily learned anything about memory policy. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:337`<br>`return attn_output, attn_weights` | The only path that hands you the attention distribution â€” needed for Â§4.2's rank stratification â€” and it is the eager path that materialises `BÂ·n_headsÂ·TÂ²`. The rank-stratified generator and the efficient kernel are in direct conflict, priced in `measuring-memory.md Â§3.6`. |

---

## 6. Exercises

Activate first, in PowerShell, dot-sourced so the variables survive:

```powershell
. .\scripts\activate-lab.ps1
```

**Standing caveats.** `[M]` Single tensors â‰¥32 GiB hang silently at 0% CPU
(`large-tensor-fault-32gib`) â€” not binding at these shapes, but keep it in mind if you extend
Exercise C's lengths. `bf16-numerics-unproven` is `untested`, so exercises default to fp32.
`hipblaslt-config` is `[M]` a **numerics** control worth 2.8Ã— in long-reduction error â€” record
whether it was set. The Hardware Validation Gate has not run, so nothing measured here is
evidence by house standard; these are instrument-shakedown runs and should be labelled as such
in the notebook.

Write scratch scripts under `notebook/`. Exercise C is the seed of a Themis component and
acquires tests when it is reused.

---

### Exercise A â€” the survival law, and the metric that cannot see it

**Goal:** derive Â§3.2 and Â§3.3 yourself, confirm them by Monte Carlo, and then watch RULER's
scorer fail to register the effect you just proved exists.

**Hardware:** none. Pure numpy. **CPU fallback:** it *is* the CPU version.
**Runtime:** ~90 s for the full grid at 200,000 trials per cell; ~10 s for the three key rows.

**Do the arithmetic before you run it.** Write down `b^(1/s)` at `b = 0.10`, `s = 5` and
`b^(Q/s)` at `Q = 4`. If you get 0.631 and 0.158 you have understood Â§3.2 and Â§3.3, and the
simulation is now a check on your code rather than a source of the result.

```python
"""Survival of Q targets under top-b retention, and what two scorers see."""
import numpy as np

rng = np.random.default_rng(20260726)

def survival(T, b, s, Q, trials=200_000):
    """Returns (all-Q pass rate, mean per-target survival = partial-credit score)."""
    keep = int(round(b * T))
    nm = s * rng.exponential(1.0, size=(trials, Q))          # target masses
    # how many of the T-Q background entries beat each target
    exceed = rng.binomial(T - Q, np.exp(-nm))
    other = np.zeros_like(exceed)                            # ...and how many other targets
    for i in range(Q):
        other[:, i] = (nm > nm[:, i][:, None]).sum(axis=1)
    survived = (exceed + other) < keep
    return survived.all(axis=1).mean(), survived.mean()

print(f"{'b':>6} {'s':>5} {'Q':>3} {'b^(Q/s)':>10} {'all-Q':>10} {'b^(1/s)':>10} {'mean':>10}")
for b, s in ((0.50, 2.0), (0.10, 5.0)):
    for Q in (1, 2, 4, 8):
        allq, mean = survival(4096, b, s, Q)
        print(f"{b:>6.2f} {s:>5.1f} {Q:>3} {b**(Q/s):>10.4f} {allq:>10.4f} "
              f"{b**(1/s):>10.4f} {mean:>10.4f}")

# the two anchors that make this an exercise rather than a demo
_, a = survival(4096, 0.10, 1.0, 1)
print(f"anchor s=1  : closed form 0.1000, MC {a:.4f}")   # a target like any other entry
_, c = survival(4096, 0.10, 100.0, 1)
print(f"anchor s=100: closed form {0.1**0.01:.4f}, MC {c:.4f}")
```

**Deliverables â€” one table, two anchors, one conclusion.** My run is in the right-hand columns.

| # | Deliverable | Mine `[M]` |
|---|---|---|
| 1 | `all-Q` matches `b^(Q/s)` across the grid | max abs error **â‰¤0.0020** across two fresh processes, seeds 20260726 and 424242, `T âˆˆ {4096, 16384}` |
| 2 | `mean` matches `b^(1/s)` and is **flat in `Q`** | at `b=0.50, s=2`: **0.7079 / 0.7072 / 0.7066 / 0.7070** for `Q=1,2,4,8` |
| 3 | `all-Q` at the same cells | **0.7079 / 0.4996 / 0.2496 / 0.0617** |
| 4 | anchor `s = 1` | closed form 0.1000, MC **0.1008** and **0.0994** |
| 5 | anchor `s = 100` | closed form **0.9772**, MC **0.9771** and **0.9775** |
| 6 | invariance to `T` | 4,096 and 16,384 agree within MC error â€” the asymptotic `q(b) = âˆ’ln b` is already good at our lengths |

**Deliverable 3 against deliverable 2 is the whole exercise.** The all-or-nothing pass rate falls
by an order of magnitude as `Q` goes 1 â†’ 8. The partial-credit score â€” the one RULER reports
(`common_utils.py:43`) â€” does not move at all. If you had run this eval to compare two eviction
budgets, the scorer would have shown you a flat line while the underlying task got 11Ã— harder.

**A mistake I made, kept because it is the lesson.** My first script printed *"s=100 â†’ expect
~0.955"* and the Monte Carlo returned 0.9771, which looked like a 2-point discrepancy. It was
not: `0.1^(1/100) = 0.9772`, and my expectation line was the thing that was wrong. The house
rule says *if a result looks wrong, suspect the harness first* â€” and here "the harness" was my
own hand-arithmetic in a print statement. Check your expectation before you distrust your
simulation.

**Then the extension that matters.** Add a third scorer: **ordered** exact match, where the
model must produce the `Q` values in the order queried (`[C]` 2504.04713's construction). Its
pass rate is `p^Q` as well â€” order costs nothing extra *under this model*, because survival is
the only failure mode being modelled. That negative result is worth having: **ordering defends
against the dumping strategy of Â§2.6, not against eviction.** They are different defects and one
control does not cover both.

---

### Exercise B â€” audit RULER, and compute the chance level of every family

**Goal:** stop treating a benchmark as a black box. Read the generator, compute each family's
null in closed form, and confirm or refute the module's `[A]` claim that the harness reports no
standard error.

**Hardware:** none for the audit. **CPU fallback:** the whole exercise is CPU.
**Runtime:** 45â€“60 min reading, 15 min arithmetic, plus one harness run if you choose to do
part 3 (which needs `pip install wonderwords nltk` and a dataset download).

**Part 1 â€” the reading, with specific questions.** Open the seven files in Â§5.1 and Â§5.2 and
answer, in writing:

1. Quote the two strings that share the phrase "special magic". Which axis of Â§2.2 do they pin,
   and what is the minimum edit that unpins it?
2. For `niah_multiquery`, trace `num_needle_k`, `indices`, and `answers`. Is the answer set a
   strict subset of the needle values in the context, or is it all of them?
3. `string_match_all` is given a prediction and a list of references. Construct, on paper, a
   prediction that scores 1.000 and demonstrates zero keyâ†’value binding.
4. Which single family has a chance level above 0.001, and why?

**Part 2 â€” the arithmetic, which is the checkable number.** Write a twenty-line script with a
stub tokenizer (`len(text) // 4` is close enough for the estimate; use your real tokenizer for
the number you report) that computes, for each RULER-NIAH family at a 4,096-token budget:

- `V_val`, the value inventory size, read out of `prepare_niah.py:88-100`;
- `n_candidates`, the number of values present in the context;
- `chance_blind` = 1 / `V_val` â€” guessing from the prior;
- `chance_copy` = 1 / `n_candidates` â€” copying an arbitrary candidate from the context;
- `chance_dump` = min(1, `tokens_available` / (`tokens_per_candidate` Â· `n_candidates`)) â€” the
  Â§2.6 strategy under the 128-token generation cap.

**Predictions to write down before you run it**, so a harness bug announces itself:

| Family | `chance_blind` | `chance_copy` | `chance_dump` |
|---|---|---|---|
| `niah_single_1/2` (essay or repeat haystack, 1 needle) | ~1.1e-7 | 1.000 | **1.000** |
| `niah_single_3` (UUID value) | ~0 | 1.000 | **1.000** |
| `niah_multikey_1` (4 needles, 1 query) | ~1.1e-7 | 0.250 | **1.000** |
| `niah_multiquery` (4 needles, all queried) | ~1.1e-7 | â€” | **1.000** |
| `niah_multikey_2/3` (distractor-needle haystack) | ~1.1e-7 | ~0.005 `[A]` | **~0.15 `[A]`** |

The `chance_copy = 1.000` cells are not a typo and they are the point: when the context contains
exactly one magic number, "emit the only magic number you can see" is a perfect strategy that
involves no retrieval at all â€” the *haystack* is doing the discrimination, not the model. Only
the distractor-needle families make copying a real decision. `[A]` medium confidence on the
last row's magnitudes, which depend on your tokenizer; the *ordering* of the rows is structural
and should hold for any tokenizer.

**Part 3, optional â€” run it and check the `N/A` claim.** If you install the RULER extras:

```
lm_eval --model hf --model_args pretrained=<tiny-model> \
        --tasks niah_single_1 --limit 20 --batch_size 1
```

Look at the `4096_stderr,none` column. Â§3.6 predicts the literal string `"N/A"`, traced through
`metrics.py:585` â†’ `evaluator_utils.py:214`. **Report what you actually see.** If a standard
error appears, the trace is wrong and this module needs an appended correction â€” which is a
better outcome than being right, because it costs one line and buys certainty.

**Deliverables.** The five-column chance table with your tokenizer's numbers; the prediction that
`niah_multikey_2` is the only family whose null is worth reporting; and either a confirmation or
a refutation of the `N/A` claim, with the command you ran.

---

### Exercise C â€” MQAR at nanoGPT scale, and the anchor that catches a broken harness

**Goal:** reproduce the one published shape in this field that a small rig can check itself
against, and establish whether *your* harness can produce it â€” because two 2026 papers' MQAR
tables (Â§3.5) could not.

**Hardware:** gfx1151 or CPU. `[A]` medium confidence the **CPU is competitive or faster** at
these shapes: a `d_model = 64`, 2-layer model at sequence length 128 is kernel-launch-bound, and
the Z13 is a capacity machine, not a latency machine. Run one config both ways and find out â€”
that measurement is worth having and takes five minutes.
**Runtime:** ~15â€“25 min for the 8-cell grid on either device at the settings below; budget 2 h
including the writing and the failure you will probably have on the first pass.

**The task.** A sequence of `N` keyâ†’value pairs followed by `Q` queries, from disjoint key and
value inventories. `A 4 B 3 C 6 â†’ C ? A ?` must produce `6, 4`. Loss and accuracy are computed
**only on the answer positions**; everything else is masked out. That masking is the most common
place this exercise goes wrong: if you train on the whole sequence, the model spends its capacity
learning the filler distribution and the accuracy curve is uninterpretable.

**The generator, which is the part you own:**

```python
import torch

def mqar_batch(batch, n_pairs, n_queries, n_keys, n_values, gen):
    """Returns (tokens, targets) with targets = -100 everywhere except answer slots.
    Vocab layout: [0, n_keys) = keys, [n_keys, n_keys+n_values) = values."""
    L = 2 * n_pairs + 2 * n_queries
    x = torch.zeros(batch, L, dtype=torch.long)
    y = torch.full((batch, L), -100, dtype=torch.long)
    for b in range(batch):
        keys = torch.randperm(n_keys, generator=gen)[:n_pairs]
        vals = torch.randint(n_keys, n_keys + n_values, (n_pairs,), generator=gen)
        x[b, 0:2 * n_pairs:2] = keys
        x[b, 1:2 * n_pairs:2] = vals
        qi = torch.randperm(n_pairs, generator=gen)[:n_queries]
        for j, i in enumerate(qi):
            pos = 2 * n_pairs + 2 * j
            x[b, pos] = keys[i]
            x[b, pos + 1] = vals[i]        # teacher forcing; the model predicts this slot
            y[b, pos] = vals[i]            # loss at the QUERY position predicts the value
    return x, y
```

**Three ways to get that generator wrong, so you do not rediscover them.** (1) **The off-by-one.**
`y[b, pos] = vals[i]` is the target for the logits *at* position `pos`, i.e. it assumes your loss
is `cross_entropy(logits[:, t], y[:, t])` with no further shift. If your training loop shifts
targets itself â€” nanoGPT's does â€” you must not shift twice, and a double shift trains the model
to predict the *key* of the next query, which looks like a capacity failure and is not.
(2) **`n_keys` must be â‰¥ `n_pairs`**, or `randperm(n_keys)[:n_pairs]` silently gives you fewer
pairs than you asked for; and if you replace `randperm` with `randint` you introduce duplicate
keys, which makes the task ill-posed rather than merely harder. (3) **Values may repeat across
pairs** under `randint`, which slightly lowers the effective chance level; if you want the clean
closed form of Â§3.5, draw values with `randperm` too and say which you did.

**Settings.** 2 layers, 2 heads, `d_model âˆˆ {16, 64}`, learned positional embeddings, AdamW at
`lr = 1e-3` with cosine decay, batch 64, 3,000 steps, fresh batches every step (so there is no
train/test split to get wrong â€” the generator is the test set), `n_keys = 64`, `n_values = 64`,
`n_pairs âˆˆ {8, 16, 32, 64}`, `n_queries = 4`, fp32. `[A]` These hyperparameters are from the
Zoology-family setups and are **untested on this machine** â€” if the anchor below fails at 3,000
steps, raise the step count before concluding anything about architecture.

**The three deliverables, in the order they must be checked.**

1. **The anchor, and nothing else counts until it passes.** Softmax attention at `d_model = 64`
   must reach **â‰¥0.95 accuracy at every `n_pairs`**, including `n_pairs = 64`, because attention's
   capacity is context length and not `d_model` (Â§3.5). `[C]` 2312.04927 reports it solving MQAR
   at model dimension 64 across all tested lengths. If your run does not, **your harness is
   broken and the rest of the table is noise** â€” which is exactly the failure visible in `[C]`
   2605.11196's published table (softmax at 0.152 with 8 pairs) and plausibly in `[C]`
   2605.06997's (Mamba-2 at ~3% across every configuration).
2. **The cliff, from the control.** Replace the attention with a single linear-attention layer
   (no softmax; `constant-state-memory.md Â§5.1` has the four-line recurrence) at the same
   `d_model`. Prediction: accuracy â‰ˆ1 for `n_pairs < d_model` and collapsing beyond it, with the
   knee near `n_pairs â‰ˆ d_model`. Report the knee location, not just the shape.
3. **The null you cannot skip.** Chance level is `1/n_values = 1/64 = 0.0156` per query under
   partial credit and `(1/64)^4 = 6.0e-8` all-or-nothing (Â§3.5). Report accuracy against the
   *right* one and say which. Then run the two nuisance nulls from `measuring-memory.md Â§2.7`
   that apply here: **re-run identical** (expect exactly zero difference; if not, you have
   non-determinism to find before anything else) and **bf16 vs fp32 on the same weights** (expect
   a small but non-zero gap â€” and if that gap is comparable to the differences between your
   `n_pairs` cells, every bf16 number in this exercise is uninterpretable, which is the same
   conclusion `measuring-memory.md` Exercise B reached at an SNR of 0.1).

**What a falsification would mean.** If softmax attention *does* cliff at `n_pairs â‰ˆ d_model`,
either your loss masking is wrong (most likely), your key inventory is too small so keys repeat
within a sequence (second most likely), or Zoology's result does not reproduce at this
configuration â€” in which case you have found something worth a notebook entry and a careful
re-read. Rank those three explanations in that order before you write anything down.

**Then the extension that connects this module to the lab's actual question.** Once the anchor
holds, apply a top-`b` eviction policy to the trained attention model's KV cache at inference and
sweep `b`. Â§3.3 predicts the all-or-nothing accuracy falls as `b^(Q/s)` for whatever effective
salience `s` the key tokens have. **Fit `s` from the curve.** That number â€” the measured salience
of a key token in a trained model, in units of the background mean â€” is the missing input to
every closed form in this module and, as far as I can establish, is unpublished.

---

## 7. Self-check

Answers at the end. Do not scroll.

1. A colleague reports that their eviction policy at a 10% budget "passes needle-in-a-haystack at
   98%, so recall is preserved." Give the two numbers you would ask for, and state what each one
   would settle.

2. You have a recall task with `Q = 4` targets. Your suite has `n = 200` items and your model
   recovers a single target with probability `p â‰ˆ 0.75`. Should you score all-or-nothing or
   partial credit? Show the arithmetic, and state what changes your answer.

3. `niah_multiquery` inserts 4 needles and queries all 4. Explain, from the scorer's source, why
   a model that emits all four magic numbers in any order scores 1.000, and name the property of
   memory that this eval therefore does not measure.

4. Your MQAR harness reports: softmax attention 0.34, DeltaNet 0.91, Mamba-2 0.03, all at 8 pairs
   and `d_model = 64`. What do you conclude about DeltaNet?

5. You run a scaled RULER at 4,096 tokens against a nanoGPT trained with `block_size = 256`, and
   every item scores at chance. Name the most likely cause, give the one-line code reference, and
   say which fault in the standard battery you have accidentally run.

6. Someone proposes reporting a single "effective context length" for each of your six ablation
   arms, defined by RULER's threshold rule against your own short-context baseline. Give two
   reasons this number is weaker than it looks, one statistical and one structural.

---

## 8. What is still unsolved here

Everything below is testable at 20Mâ€“300M params on one gfx1151 GPU with a `[M]` â‰¥62 GiB fast
tier, or on the inference rig. Each needs a pre-registered hypothesis card before it runs.

1. **The background distribution of per-token attention mass is unpublished, and every closed
   form in Â§3 depends on its tail.** `Exponential(1)` is a stipulation inherited from
   `measuring-memory.md` Exercise C. If real attention mass over text is heavier-tailed, `q(b)`
   moves and so does `b^(1/s)`. Measuring it on a real model at a few context lengths is a
   one-day job on the inference rig and converts three of this module's laws from simulations
   into predictions. This is the highest-value item on the list and it requires no training.

2. **The effective salience `s` of an eval target in a trained model has never been measured.**
   Â§3.2 and Â§3.3 are parameterised by it and nobody reports it. Exercise C's extension fits it
   from an eviction sweep; `memory-failure-modes.md` Exercise B measures the rank directly. Two
   independent routes to the same number, which is the right situation to be in.

3. **Are the `Q` targets' survivals independent?** Â§3.3 assumes so and notes the assumption is
   probably conservative â€” targets in one context share a haystack and a query, so positive
   correlation is likely and `p^Q` is a lower bound on the joint pass rate. The correlation is
   directly measurable in the same eviction sweep and I can find no estimate of it.

4. **Does the rank-stratified generator's step-5 shift matter?** Â§4.2: inserting the needle
   changes the very distribution you used to choose where to insert it. `[A]` high confidence in
   the mechanism, zero confidence in the magnitude. If the shift is small the whole construction
   is easy; if it is large, rank stratification needs an iterative placement loop and becomes
   expensive.

5. **Is a retrieval-head signature `[C]` (2404.15574) present at 300M?** If it is, head-level
   attribution becomes available and the adverse-selection coupling of Â§2.4 becomes directly
   measurable rather than argued. If it is not, the CompressKV-family policies `[C]` (2606.24467)
   cannot even be implemented at our scale, which changes what we can ablate.

6. **Contested: does associative recall predict anything downstream?** Zoology attributes 82% of
   a real perplexity gap to AR tokens `[C]` (2312.04927), the strongest transfer evidence
   available; `[C]` 2508.19029 revisits it; and Â§3.5's two anomalous tables suggest MQAR results
   are dominated by harness details. No source demonstrates a *quantitative* prediction from an
   MQAR curve to a downstream curve, at any scale. Do not let a capacity curve stand in for a
   capability claim.

7. **Contested: matched parameters or matched state?** `[C]` 2605.22791 argues state size is the
   correct control for recall comparisons; most of the literature matches parameters. They give
   different rankings and papers rarely report both, which means half the published recall
   comparisons are not comparable to the other half. Our arms must report both or pick one and
   say so in the card.

8. **Contested: is "effective context" a single number?** RULER's threshold rule says yes; `[C]`
   ATLAS (2605.28079) finds **7 of 26 models shifting â‰¥2 rank positions** between length regimes,
   with gaps up to 12, which says the induced *ordering* depends on the length grid you chose.
   Since a ranking is the only thing an ablation produces, this is not a precision complaint.

9. **Nobody reports a needle-absent control.** `research/notes/evaluation-landscape.md Â§6.2`
   marks this `[A]` high confidence after a twelve-month search; Â§5.1 shows the standard
   generator validating the positive control and shipping no negative one. I searched again on
   2026-07-26 and found nothing contradicting it. **This is a publishable methodological result
   available before any research arm runs**, which is an unusual position to be in and worth
   exploiting.

10. **The dumping-strategy floor of Â§2.6 is unmeasured.** `[A]` medium confidence that
    `niah_multikey_2/3` have an effective floor near 0.15 under a 128-token generation cap. It is
    a half-day to measure with any instruction-following model on the inference rig, and if it
    holds, a large number of published RULER tables have an unreported floor of fifteen points
    on their hardest family.

11. **Is a 100M model capable of a Michelangelo-style latent-structure task at 1K context?**
    `[A]` medium confidence. A one-day pilot settles it, and a negative closes the strongest
    anti-pattern-match instrument available at our scale, redirecting effort to MQAR plus
    NoLiMa-style construction.

12. **Our pre-registration cards still have no MDE field.** Â§3.6 computes it for RULER's own
    defaults (Â±3.1pp unpaired, 2.0pp paired at `n = 500`). The framework is `[C]` 2411.00640 and
    the arithmetic takes ninety seconds. This is not research, it is a missing column, and it is
    the cheapest item here.

---

## Answers to the self-check

**1.** Ask for (a) **the target's rank under the policy's own scoring function**, and (b) **the
needle-absent score.** (a) settles adverse selection: if the needle ranks in the top-`k` at a 10%
budget then the eval could not have failed and the 98% carries no information about the policy
(`memory-failure-modes.md Â§2.5`; Â§2.4 here). (b) settles whether the eval reads the needle at
all: if deleting the needle leaves the score near 98%, the model was answering from the prior,
the haystack's structure, or lexical overlap, and both numbers are void. Neither costs a training
run. If you may ask for a third, ask for the **multi-query** version at the same budget â€” Â§3.3
says `Q = 4` targets is equivalent to one target at a quarter the salience, and it is the same
prompt with three more needles.

**2.** `p^Q = 0.75^4 = 0.316`. The floor rule is `nÂ·p^Q â‰¥ 10`: `200 Ã— 0.316 = 63`, comfortably
above 10, so all-or-nothing is usable â€” and Â§3.4 says it is the more powerful metric whenever it
is usable, with `d_exact = 1.470` against `d_partial = 1.155`, a ratio of 1.27. **Score
all-or-nothing, and report partial credit alongside it.** What changes the answer is `p`
dropping: at `p = 0.5`, `p^4 = 0.0625` and `nÂ·p^Q = 12.5` â€” right at the boundary â€” and at
`p = 0.4` it is 5.1 and the metric floors, at which point partial credit is the only one that
moves. Since `p` falls as you compress harder, **the correct scorer changes along the sweep**,
which is the argument for always computing both from the same predictions rather than choosing
one in advance.

**3.** From `common_utils.py:43`: the score is
`sum(1.0 if r.lower() in pred.lower() else 0.0 for r in ref) / len(ref)` â€” containment of each
reference *anywhere* in the prediction, unordered, with no penalty for extra content. From
`prepare_niah.py:188`, `niah_multiquery` sets `num_needle_k = num_needle_q = 4` and samples all
four indices, so `answers` is the complete set of needle values in the context. Emitting all four
therefore satisfies all four references and scores 1.000. **What it does not measure is binding**
â€” the association between key `k_i` and value `v_i`, which is the entire content of the word
"associative" in associative recall. The eval measures whether the values are *present and
extractable as a set*, which is a strictly weaker property, and it is `[C]` 2605.11325's
recall-without-precision complaint made concrete in the standard harness.

**4.** **Nothing.** The softmax control is broken. Attention's MQAR capacity is bounded by context
length, not by `d_model`, so at 8 pairs and `d_model = 64` it must score â‰ˆ1.0 `[C]` (2312.04927).
A control at 0.34 means the harness is misconfigured â€” the most likely causes, in order, are loss
computed over the whole sequence instead of the answer positions, a key inventory small enough
that keys repeat within a sequence, or too few training steps. Until that is fixed, the DeltaNet
0.91 is not evidence of anything: whatever is suppressing the control may be inflating or
deflating the other columns too. This is exactly the situation in `[C]` 2605.11196's published
table, and the house rule applies â€” **if a result looks wrong, suspect the harness first.**

**5.** `training/nanogpt/model.py:314` silently left-truncates any input longer than
`block_size`, and `model.py:128` shows why it must: `wpe` has exactly `block_size` rows and there
is no positional embedding for index 256. A 4,096-token item is cropped to its last 256 tokens,
which almost certainly deletes the needle. **You have accidentally run fault #1 of the standard
battery â€” needle absent â€” on every item**, and the eval is correctly reporting that a model with
no access to the needle scores at chance. The fix is to scale the eval's length grid to multiples
of *your* training length, which is the whole reason to prefer a generator-based benchmark over a
static one.

**6.** *Statistical:* the threshold is a comparison of two numbers and needs a variance estimate.
At RULER's own default of 500 items per bucket, a score near 85.6 carries a Â±3.1pp 95% CI
unpaired, 2.0pp paired at `Ï = 0.9` (Â§3.6) â€” so the "effective length" is resolvable only to a
bucket boundary where the score moves by more than that, and the harness reports the standard
error as `"N/A"` (`evaluator_utils.py:214`). Report six such numbers without CIs and you are
ranking arms on noise. *Structural:* the induced **ordering** depends on the length grid. `[C]`
ATLAS (2605.28079) finds 7 of 26 models shifting â‰¥2 rank positions between the 8Kâ€“128K and
8Kâ€“1M regimes, with gaps up to 12 positions. Since the only thing an ablation produces is an
ordering, a single-number summary whose ordering is grid-dependent is not a weaker measurement of
the right thing â€” it is a measurement of the grid. Report the curve, and report the rank
correlation between grids.

---

## Sources

### Local measurements produced for this module

Environment: Python 3.12.10, numpy 2.4.4, native Windows 11 build 26200, 2026-07-26. **CPU only
â€” no GPU, no torch, no model.** Two independent runs in fresh processes.

- **Run 1** â€” `numpy.default_rng(20260726)`, `T = 4096`, 200,000 Monte Carlo trials per cell,
  grid `b âˆˆ {0.50, 0.25, 0.10, 0.05, 0.01} Ã— s âˆˆ {1, 2, 5, 10}` for Claim 1 and
  `b âˆˆ {0.50, 0.25, 0.10} Ã— s âˆˆ {2, 5, 10} Ã— Q âˆˆ {1, 2, 4, 8}` for Claim 2.
- **Run 2** â€” separate fresh process, `numpy.default_rng(424242)`, `T âˆˆ {4096, 16384}`, same
  trial count, three `(b, s)` cells including `s = 1.2` (a NoLiMa-style needle with almost no
  salience advantage).

| Quantity | Closed form | Run 1 (`T`=4096) | Run 2 (`T`=4096) | Run 2 (`T`=16384) |
|---|---|---|---|---|
| `P(survive)`, `b=0.10`, `s=5`, `Q=1` | 0.6310 | 0.6297 | 0.6295 | 0.6308 |
| `P(all Q)`, `b=0.50`, `s=2`, `Q=1/2/4/8` | 0.7071 / 0.5000 / 0.2500 / 0.0625 | 0.7079 / 0.4996 / 0.2496 / 0.0617 | 0.7066 / 0.4997 / 0.2500 / 0.0625 | 0.7081 / 0.5021 / 0.2495 / 0.0627 |
| **partial-credit mean, same cells** | 0.7071 for every `Q` | **0.7079 / 0.7072 / 0.7066 / 0.7070** | **0.7066 / 0.7065 / 0.7073 / 0.7073** | **0.7081 / 0.7078 / 0.7067 / 0.7073** |
| `P(all Q)`, `b=0.25`, `s=1.2`, `Q=1/2/4/8` | 0.3150 / 0.0992 / 0.0098 / 0.0001 | â€” | 0.3168 / 0.1000 / 0.0093 / 0.0001 | 0.3153 / 0.0993 / 0.0100 / 0.0001 |
| anchor `s = 1`, `b = 0.10` | 0.1000 | 0.1008 | 0.0994 | â€” |
| anchor `s = 100`, `b = 0.10` | 0.9772 | 0.9771 | 0.9775 | â€” |
| max abs error, closed form vs MC | â€” | 0.0020 over 36 cells | 0.0018 over 12 cells | 0.0016 over 12 cells |

Two things the retest adds beyond reproduction. **The result is invariant to `T`** â€” 4,096 and
16,384 agree to within Monte Carlo error, which is the evidence that the large-`T` asymptotic
`q(b) = âˆ’ln(b)` is already accurate at the context lengths we work at. And the `s = 1.2` row is
the case that matters operationally: a needle only 20% more salient than background dies at
`Q = 4` even at a **75% retention budget** (0.0093), while the partial-credit scorer at the same
cells reads a flat 0.315.

**What this establishes and what it does not.** It establishes that `p = b^(1/s)` and
`p_all = b^(Q/s)` are correct for the stated model, and that a partial-credit scorer's expected
value is invariant to `Q` while an all-or-nothing scorer's is not. **This is a simulation of a
selection rule under a stipulated background distribution, not a measurement of a model or of
this hardware** â€” the same caveat `measuring-memory.md` Exercise C carries, and for the same
reason: `Exponential(1)` is an assumption (Â§8 item 1). The closed forms are derivations; the
Monte Carlo is a check on the derivation and on the implementation, and it is reported as `[M]`
only in that narrow sense.

The Claim-3 table in Â§3.4 is **pure arithmetic**, not a measurement â€” `d_exact`, `d_partial` and
the `nÂ·p^Q â‰¥ 10` column are evaluated formulas and are labelled as such.

### Local artifacts and prior measurements relied on

- `ASSUMPTIONS.md` rows: `gpu-fast-tier-size` (â‰¥62 GiB at ~200 GB/s, single run per arm),
  `gemm-throughput-below-reference` (20.9 TFLOP/s bf16 at 8192Â³), `large-tensor-fault-32gib`
  (â‰¥32 GiB single tensors hang at 0% CPU), `sdpa-is-memory-efficient` (147.2 vs 6.6 bytes/TÂ²;
  `flash_sdp_enabled()` returns True either way), `hipblaslt-config` (2.01e-3 vs 5.60e-3
  relative error on a length-1M bf16 reduction â€” a numerics control, not a speed knob),
  `bf16-reduced-precision-knob-works` (**refuted** â€” inert, do not use as an axis),
  `bf16-numerics-unproven` (`untested`), `single-device-only`, `reference-model` (3:1 GSSS,
  `sliding_window` 512), `ablation-scale-sufficient` (`untested`).
- `notebook/uma-carveout-controls-fast-tier.md` â€” the fast-tier sweep.
- `curriculum/measuring-memory.md` â€” the differential instrument, the fault battery, the power
  arithmetic, and Exercise C's deterministic-needle survival model, which Â§3.2 refines.
- `curriculum/memory-failure-modes.md Â§2.5` â€” the adverse-selection argument in full, and the
  rank-stratified-target proposal that Â§4.2 turns into an algorithm.
- `curriculum/constant-state-memory.md Â§3.4â€“3.5` â€” the interference bound and the Welch floor,
  assumed and not re-derived.
- `curriculum/long-context-and-effective-context.md Â§2.1, Â§3.4` â€” length vs distance vs
  occupancy, and the threshold definition.
- `research/notes/evaluation-landscape.md` â€” the survey this module teaches; Â§2, Â§3 and Â§6 are
  the direct source for Â§2.4, Â§2.7 and Â§4.1 here.
- `research/memory/constant-state-memory.md Â§3` â€” the MQAR and S-NIAH tables, including the
  anomalous VLA control.
- Code pointers: every `file:line` in Â§5 was opened and the named construct confirmed on the
  named line on 2026-07-26, against the revisions in `research/reference/PROVENANCE.md`
  (`lm-evaluation-harness` `f4d4b3de3ee6`, `nanogpt` `3adf61e154c3`, `transformers`
  `b6d5084fb4a5`).

### arXiv `[C]`

Ids drawn from `research/notes/evaluation-landscape.md` and `research/memory/` were resolved
against the live arXiv API on 2026-07-26 by those notes' authors. Ids marked **(search-surfaced)**
were found by web search on 2026-07-26; the id and title come from the arXiv listing URL returned
by that search and the claim beside them comes from the abstract snippet, which is weaker
evidence than reading the paper â€” treat them as leads.

- `2203.03466` â€” *Tensor Programs V* / muP (2022). The standing rider for two-scale arms.
- `2304.15004` â€” *Are Emergent Abilities of Large Language Models a Mirage?* (2023). Discrete
  metrics manufacture cliffs; the source of the "prefer continuous" advice Â§3.4 qualifies.
- `2306.00946` â€” *Exposing Attention Glitches with Flip-Flop Language Modeling* (2023).
- `2306.14048` â€” *H2O: Heavy-Hitter Oracle* (2023).
- `2307.03172` â€” *Lost in the Middle* (2023).
- `2309.17453` â€” *Efficient Streaming Language Models with Attention Sinks* (2023).
- `2312.04927` â€” *Zoology: Measuring and Improving Recall in Efficient Language Models* (2023).
  MQAR; the `d â‰¥ N` requirement; attention solving it at model dimension 64; 82% of a 2.1-point
  Pile perplexity gap on associative-recall tokens. The anchor in Â§3.5 and Exercise C.
- `2402.18668` â€” *Based: simple linear attention language models balance the recall-throughput
  tradeoff* (2024). Recall as a dial rather than a property.
- `2404.06654` â€” *RULER* (2024). The generator methodology and the effective-context threshold.
- `2404.14469` â€” *SnapKV* (2024). Observation-window selection.
- `2404.15574` â€” *Retrieval Head Mechanistically Explains Long-Context Factuality* (2024).
- `2406.10149` â€” *BABILong* (2024).
- `2406.10229` â€” *Quantifying Variance in Evaluation Benchmarks* (2024).
- `2409.06338` â€” *Retrieval Or Holistic Understanding? Dolce* (2024).
- `2409.12640` â€” *Michelangelo: Long Context Evaluations Beyond Haystacks via Latent Structure
  Queries* (2024).
- `2410.02694` â€” *HELMET* (2024). NIAH does not predict downstream performance.
- `2410.05229` â€” *GSM-Symbolic* (2024). Perturbation twins as a method.
- `2411.00640` â€” *Adding Error Bars to Evals* (2024). Clustering, paired analysis, power, MDE.
- `2412.06464` â€” *Gated Delta Networks* (2024). The S-NIAH tables Â§3.5 compares against.
- `2412.10319` â€” *SCBench* (2024). Rankings do not survive multi-turn cache reuse.
- `2502.05167` â€” *NoLiMa: Long-Context Evaluation Beyond Literal Matching* (2025). The `Î›` axis;
  11 of 13 models below 50% of their short-context baseline at 32K.
- `2503.00353` â€” *U-NIAH: Unified RAG and LLM Evaluation for Long Context Needle-In-A-Haystack*
  (Mar 2025). **(search-surfaced)** Cited only as evidence that NIAH-variant construction is an
  active line.
- `2504.04713` â€” *Sequential-NIAH: A Needle-In-A-Haystack Benchmark for Extracting Sequential
  Needles from Long Contexts* (Apr 2025). **(search-surfaced)** The ordered-answer construction
  Exercise A's extension uses.
- `2506.02921` â€” *A Controllable Examination for Long-Context Language Models* (Jun 2025).
  **(search-surfaced)** The haystack-coherence shortcut, i.e. the `Îº` axis of Â§2.2.
- `2507.22411` â€” *NeedleChain: Measuring Intact Long-Context Reasoning Capability of Large
  Language Models* (Jul 2025). **(search-surfaced)** Removes the haystack entirely.
- `2508.19029` â€” *Revisiting associative recall in modern recurrent models* (Aug 2025). The
  standing caution against reading MQAR as a downstream predictor.
- `2510.00231` â€” *The Pitfalls of KV Cache Compression* (2025). Specific instructions dropped
  while LongBench held.
- `2510.13334` â€” *Taming the Fragility of KV Cache Eviction in LLM Inference* (2025). Worst-case
  aggregation changes the ranking of eviction policies.
- `2601.02872` â€” *LongBench Pro* (Jan 2026).
- `2602.16837` â€” *A Structural Theory of Position Bias in Transformers* (Feb 2026).
- `2603.20397` â€” *KV Cache Optimization Strategies for Scalable and Efficient LLM Inference*
  (Mar 2026). No single method dominates.
- `2605.06997` â€” *Echo: KV-Cache-Free Associative Recall with Spectral Koopman Operators*
  (May 2026). **(search-surfaced)** Cited in Â§3.5 for its MQAR baseline â€” pure Mamba-2 reported
  at chance (~3%) across all configurations â€” as a second instance of a control that does not
  match the reference result. Treated as a lead, not as evidence about Mamba-2.
- `2605.08234` â€” *When Does Value-Aware KV Eviction Help? A Fixed-Contract Diagnostic* (May 2026).
- `2605.11196` â€” *Variational Linear Attention* (May 2026). Cited **only** for its MQAR table as
  a caution: softmax at 0.152 with 8 pairs is a broken control.
- `2605.11325` â€” *Structured Belief State and the First Precision-Aware Benchmark for LLM Memory
  Retrieval* (May 2026). Benchmarks score answers rather than retrieval, so dumping the store
  scores perfectly â€” the complaint Â§2.6 instantiates in source.
- `2605.22791` â€” *Gated DeltaNet-2: Decoupling Erase and Write* (May 2026). The MK-NIAH tables,
  and the matched-state-vs-matched-params argument.
- `2605.23170` â€” *Positional Failures in Long-Context LLMs: A Blind Spot in Reasoning Benchmarks*
  (May 2026). **(search-surfaced)** The depth/distance confound, still live.
- `2605.28079` â€” *ATLAS: All-round Testing of Long-context Abilities across Scales* (May 2026).
  7 of 26 models shift â‰¥2 rank positions between length regimes.
- `2606.24467` â€” *CompressKV: Semantic-Retrieval-Guided KV-Cache Compression for
  Resource-Efficient Long-Context LLM Inference* (Jun 2026). **(search-surfaced)** Selects tokens
  by Semantic Retrieval Head attention â€” the tightening of the adverse-selection coupling in Â§2.4.
- `2606.29914` â€” *MemDelta: Controlled Baselines and Hidden Confounds in Agent Memory Evaluation*
  (Jun 2026). One-variable-at-a-time control finding most reported memory gains were confounds.
- `2607.08284` â€” *Understanding Axes of Difficulty For Long Context Tasks Via PredicateLongBench*
  (Jul 2026). Worst-case difficulty axes.
- `2607.21475` â€” *Error Certificates for KV-Cache Eviction via Randomized Design* (Jul 2026).
