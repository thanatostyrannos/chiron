---
title: Agent memory in practice — a write-ahead store with schema drift and no compaction policy
version: 1.0.0
date: 2026-07-26
track: C — Memory (the deep track)
mirrors: research/memory/agent-memory-systems.md
prereqs: attention-variants-and-kv-cost (required), tensors-and-autograd (light)
difficulty: 3/5 conceptually, 2/5 mathematically — the hard part is unlearning, not learning
time: 2–3 h reading; 30 min for Exercise A, 45 min for Exercise B, 45 min for Exercise C
---

# Agent memory in practice

## 1. What this module settles

**One:** the working/episodic/semantic/procedural quartet is a persistence-semantics taxonomy
wearing cognitive-science clothes, and the only test that matters is whether the four boxes have
four *different retention policies* — if they share one, you have one memory type and four table
names. **Two:** the dominant failure in the field is a category error you can state in one line —
teams treat a **context-budget allocation problem under an unknown future query** as a
**retrieval** problem, and this module proves with a measurement that no amount of retrieval
quality fixes it, because the top-*k* budget you can afford is a constant while the number of
distractors grows without bound. **Three:** the agent layer keeps re-deriving in text the
mechanisms the KV serving layer already implements one level down, and it re-derives them without
the two properties that make the serving-layer versions work — a resurrection path for
freed-but-not-yet-evicted state, and an accounting of what a mid-prefix edit costs.

This module mirrors `research/memory/agent-memory-systems.md`. That note surveys; this teaches.
Everything here is consistent with it except one place where a June 2026 paper disputes the note's
sharpest claim — flagged explicitly in §8.1, with my working shown.

**Where this sits in Track C.** It is the *top* of the stack and can be read early, because its
argument is about budgets rather than kernels. It borrows serving-layer mechanisms without
re-teaching them: `paged-attention-and-prefix-reuse.md` and `kv-eviction-policies.md` are the
sibling modules that own those in depth, and `memory-taxonomy-for-engineers.md` owns the
reconstructibility axis this module assumes. The one hard prerequisite is
`attention-variants-and-kv-cost.md`, for the per-token KV product used in §4.3 and for the reason
decode reads the whole cache every step.

---

## 2. Theory in plain language

### 2.1 The bridge, stated at your level, before it breaks

You have built this system. Here is its shape in your vocabulary, and the mapping is not
decorative — it is close enough that the differences are the entire lesson.

| Your thing | Its agent-memory counterpart |
|---|---|
| Write-ahead log | The transcript. Every turn, tool call and tool result appended in order. |
| Log records with no schema version | Notes/memories written by an LLM whose output format drifts as the prompt, the model, and the extraction instructions change. |
| A secondary index over the log | The vector store. Rebuilt or incrementally updated on write. |
| Read amplification | Every query re-reads the top-*k* notes into the prompt, and pays for them in tokens, every turn. |
| No compaction, no vacuum, no TTL | Notes accumulate forever. Nothing supersedes anything. There is no tombstone. |
| The mmap'd hot page set | The context window — the only thing the model can actually address. |

Four properties transfer exactly, and you should lean on them:

1. **Append is cheap, mid-log rewrite is expensive.** True here, and worse than you expect (§3.2).
2. **The index is not the data.** An embedding is a lossy hash for similarity; the note is the
   record. Losing the index is recoverable, losing the note is not.
3. **Unbounded growth with no reclaim is a latent outage,** not a background concern.
4. **The write path is an ingest path, and ingest paths get attacked.** `[C]` The memory-security
   survey (arXiv 2604.16548) traces the write → persist → propagate → resist-cleanup chain, and
   `[C]` arXiv 2606.04329 names compaction-driven writes as a first-class attack surface. Your
   summarizer is an untrusted ingest path with write privileges to the store it summarizes.

Now the breaks. There are three, and they are the module.

### 2.2 Break one — retrieval quality degrades with store size, and no index rebuild fixes it

The reflex from twenty years of storage work is: *quality degradation with size is an index
problem*. Fragmented index, stale statistics, wrong shard key, ANN recall traded for latency.
Rebuild it, re-tune it, add a tier.

None of that applies. Here is why, and Exercise B measures it.

Take the **best possible index**: exhaustive brute-force cosine similarity over every note. No
approximation, no recall/latency trade, nothing to rebuild. Now grow the store. The target note's
similarity to the query does not change — it is a property of the embedding model and the query,
not of *N*. But every additional note is an additional chance to *outscore* the target. The
target's rank therefore grows linearly in *N*, and the fraction of the store that outranks it is
**constant** (§3.4 derives this; Exercise B measures `rank/N` constant to three significant figures
across three orders of magnitude of *N*).

Meanwhile the number of notes you can put in the prompt — the top-*k* you can afford — is fixed by
your token budget. So:

> **Recall at a fixed context budget falls as O(K/N).** The index is exact. The embedder is
> unchanged. The only thing that moved was the size of the store.

And the obvious fix does not work asymptotically. A better embedding model lowers the constant
fraction — Exercise B measures it falling from 0.1006 to 0.0002 as query noise drops — which buys
you a *constant multiple* more store at the same recall. It does not change the exponent. You are
buying time, not solving the problem.

> **What this rules out.** "Retrieval quality is bad, so improve retrieval" is a dead end past a
> certain store size. It is the storage engineer's instinct and it is wrong here, because the
> failing component is not the lookup. It is the **budget**.

### 2.3 Break two — the category error: this is a budget problem, not a retrieval problem

State the real problem precisely and the misdiagnosis becomes visible. At turn *t* the harness holds
a set of candidate context objects — system prompt, tool schemas, prior user turns, tool results,
retrieved notes, plan state — and must decide which go in the prompt. That is a **0/1 knapsack**
(§3.1), and retrieval is nothing more than a **scoring function for one term in the objective**.

Three things break the knapsack, and all three are where real systems fail:

1. **You do not know the utilities.** You discard before the query arrives. `[C]` The rate–distortion
   view (arXiv 2607.08032, Jul 2026) names this as the failure mode shared by KV eviction, prompt
   pruning, recurrent-state bounding and agent consolidation alike: attention and recency signals
   force an irreversible discard *before the query is known*.
2. **The cost is delayed and the problem is sequential.** `[C]` OSL-MR (arXiv 2606.10616) formulates
   retention as constrained stochastic optimization with delayed miss, reacquisition and stale
   penalties, and proves the multi-step version NP-hard. Its ablation shows single-step optimization
   cannot anticipate future demand shifts — which is the formal statement of "greedy eviction is
   wrong."
3. **Utility is not additive.** Attention is not a sum over independent slots, and total occupancy
   degrades quality by itself. `[C]` arXiv 2508.07479 shows the familiar lost-in-the-middle U-shape
   holds only up to roughly 50% context occupancy; past that, primacy decays and the bias becomes
   distance-based. **The position prior your eviction policy exploits changes as the window fills.**
   A fixed prior is wrong in one of the two regimes.

> **Bridge: this is admission control plus Denning working-set theory.** `[C]` *The Missing Memory
> Hierarchy* (arXiv 2603.09023, Mar 2026) says so in as many words — "the context window of a large
> language model is not memory, it is L1 cache" — and reports the most systems-legible numbers in
> the literature: 857 production sessions, 4.45M effective input tokens, 21.8% structural waste;
> 1.4M simulated evictions at a 0.0254% fault rate; a live deployment over 681 turns cutting context
> consumption up to 93% (5,038 KB → 339 KB). It also reports the expected thrashing pathology under
> sustained pressure, which is the detail that makes it credible. `[A]` high confidence: this is a
> single-system production report, not a controlled comparison, and "up to 93%" is a best case.
>
> **Where it breaks: at the fault handler.** In demand paging the fault is *transparent* to the
> faulting process and is serviced by a different, deterministic entity. Here the "process" and the
> "pager" are **the same stochastic model**, the fault must be *chosen* via a tool call, and each one
> costs a full turn of latency and tokens. There is no MMU, no present bit, no dirty bit, and no
> reference bit. `[C]` VISTA (arXiv 2606.30005, Jun 2026) makes the sharpest version of the point:
> frontier models are "proprioceptively blind to their own context" — from the prompt alone they
> cannot see how large, how old, or how used each block is. Its training-free interface exposes typed
> addressable blocks plus a dashboard of per-block token usage, recency and access history, and lifts
> Gemini-3-Flash from 22.7% to 50.7% on LOCA-Bench, with ablations showing the dashboard matters
> beyond the archive and recovery tools it ships with.
>
> **If exposing counters recovers that much, the problem was never retrieval. It was accounting with
> no counters.** You have run systems where the fix was a Prometheus metric that did not exist. Same
> shape.

### 2.4 Break three — compaction is a mid-log rewrite, and it is lossy with no checksum

Two independent problems, usually conflated.

**Problem A — it costs the tail.** vLLM keys each KV block by a *chain* hash, not a content hash:
a block's key folds in its parent's key. Consequently an edit at token position *p* invalidates
every block from ⌊p/P⌋ onward, where *P* is the block size in tokens. §3.2 does the arithmetic and
Exercise A runs it. The headline: almost every compaction paper prices summarization in
*summarizer* tokens and ignores the invalidated prefill, which is usually the larger number.
`[C]` TokenPilot (arXiv 2606.17016) is built entirely around this trade-off and reports 56–87% cost
reduction by making prefix stability a first-class constraint; `[C]` Self-GC (arXiv 2607.00692)
lists "cache-aware commit" as an explicit mechanism.

**Problem B — it is a lossy rewrite with no way to detect the loss.** This is the one that should
worry you most, and there is now a clean measurement of it. `[C]` *Governance Decay*
(arXiv 2606.22528, Jun 2026, 1,323 episodes, seven model families) holds a safety policy in context
and then compacts: **violation rises from 0% with the policy in full context to 30% after
compaction, reaching 59% for some models.** The decomposition is the useful part — when the
constraint survives summarization, violation stays at 0%; when it is dropped, violation reaches 38%.
Their mitigation, "constraint pinning," restores 0%.

Read that as a systems engineer: **a compaction that silently drops a record produces a 38%
failure rate on the workload that depended on it, and the pin is a `mlock`.** An LSM merge that
dropped a key would be a data-loss bug with a CRC to catch it. A summarizer that drops a
constraint is behaving normally and there is no checksum that could detect it, because the output
is not supposed to equal the input.

### 2.5 What the serving layer already has that the agent layer keeps re-deriving

This is the transplant list, and it is where original work is available.

| Serving layer has it | Agent layer re-derives it, worse | The missing property |
|---|---|---|
| `free_blocks` decrements a refcount and pushes the block back on the LRU **with contents and hash intact**; `touch` can resurrect it on a later prefix hit; eviction happens lazily at reallocation | Compaction rewrites the transcript and the tokens are **gone** | **A resurrection path.** Self-GC's recoverable sidecars and VISTA's full-fidelity archive are both attempts to reintroduce, in text, what the KV layer gets for free. |
| SGLang's `evict` only ever considers **leaves** from an incrementally maintained set, peeling the frontier inward; `inc_lock_ref` walks to the root, so one in-flight request pins an arbitrarily deep prefix chain | Recency-ordered eviction over a flat note list | **Topological constraint.** An active sub-task pins its whole plan chain above it. `[C]` CWL (arXiv 2606.11213) re-derives this as typed, dependency-linked episodes with a deterministic LLM-free policy over the dependency graph. |
| Mooncake's `BatchEvict` only considers objects whose **lease has expired**; `TryPushPromotionQueue` gates disk→DRAM promotion behind a count-min-sketch TinyLFU threshold so one cold hit cannot pollute the fast tier | Recency plus embedding similarity, with **no admission control at all** | **A lease and a frequency gate on writes.** Straight transplant. Nobody has run it. |
| Evicting KV is **lossless with recompute** | Evicting a note is lossy and unrecomputable | **The clean-page distinction** — *unless* the effect is already persisted outside the context. `[C]` CWL's criterion is exactly this: shed action episodes "whose effects are already persisted in the environment." It is the only principled zero-cost eviction in the space, and CWL reports one session completing 89 sequential tasks across 80M tokens with no measurable accuracy degradation versus per-task isolated sessions. |

### 2.6 The four boxes, and the only test that makes them real

| Box | What it holds | Retention policy | Read path | Write path |
|---|---|---|---|---|
| **Working** | tokens currently in the context window | hard byte-bounded, destroyed at session end | attention; O(1) addressable, zero lookup latency | append (cheap) or rewrite (see §3.2) |
| **Episodic** | time-indexed records of what happened | decay / archival | similarity or temporal index | append-only log |
| **Semantic** | facts abstracted from episodes | indefinite until superseded | index lookup | read-modify-write with conflict resolution |
| **Procedural** | reusable how-to, i.e. skills | evidence-gated revision | matched on situation | distilled from trajectories |

**The test: if your store has one retention policy, you have one memory type no matter how many
tables you have.** Most shipped systems fail it — a single vector index with a timestamp column,
called four things in the docs. `[C]` *The Missing Knowledge Layer* (arXiv 2604.11364, Apr 2026)
makes the argument cleanly: CoALA `[C]` (arXiv 2309.02427) and JEPA both lack an explicit Knowledge
layer with its own persistence semantics, producing "a category error: systems apply cognitive decay
to factual claims, or treat facts and experiences with identical update mechanics."

Procedural memory is the box with the best 2026 evidence *and* the sharpest caveat. `[C]` The AFTER
benchmark (arXiv 2606.23127, Jun 2026; 382 enterprise tasks, six roles, 22 skills) reports a single
refinement round improving aggregate performance by 3.7–6.7 points, with skills evolved from
*multi-model* traces reaching 73.1% cross-model test accuracy — beating any single-model source.
`[C]` But the skill-lifecycle study (arXiv 2605.23899) finds model-generated skills "beneficial on
average but exhibit non-trivial negative transfer," and — the load-bearing bit — a model can be a
strong skill *extractor* and a weak skill *consumer*, with utility uncorrelated with model scale.
**Procedural memory is not monotone: adding skills can make an agent worse.** `[C]` Skill-Pro
(arXiv 2602.01869) adds score-based maintenance to keep the store compact without parameter updates.

**Contested, and do not resolve it.** `[C]` *Memory in the Age of AI Agents* (arXiv 2512.13564)
states outright that "traditional taxonomies such as long/short-term memory have proven
insufficient" and replaces them with forms × functions × dynamics. `[C]` The rate–distortion view
(arXiv 2607.08032) and `[C]` the Mar 2026 survey (arXiv 2603.07670) treat the cognitive mapping as
decorative. `[C]` Meanwhile arXiv 2504.15965 and `[C]` arXiv 2605.06716 lean on the cognitive terms
structurally. **Use the cognitive words as labels for retention policies, never as an argument.**

---

## 3. The math that actually matters

### 3.1 The knapsack, with every symbol translated

Let the prompt hold **B** tokens. At turn *t* the harness holds candidate context objects indexed
*i*, each of token length **s_i** with future utility **u_i**, and a keep/drop decision
**x_i ∈ {0,1}**:

> maximize  Σ_i u_i · x_i   subject to   Σ_i s_i · x_i ≤ B

| Symbol | Reads as |
|---|---|
| *i* | one context object — a tool result, a retrieved note, a prior turn, a plan step |
| *s_i* | its length **in tokens**, which is the only currency the prompt accepts |
| *u_i* | how much having it resident improves the **next** decision — unknown at decision time |
| *x_i* | 1 = keep it in the prompt this turn, 0 = drop it |
| *B* | the token budget you have actually chosen to spend — **not** the advertised context length |

That is a 0/1 knapsack. Retrieval computes an *estimate* of *u_i*. It is one term in the objective.

`[C]` ContextBudget/BACM (arXiv 2604.01664, Apr 2026) is the cleanest direct statement of the
correct framing: context management as a sequential decision problem with an explicit budget
constraint, where the agent assesses remaining budget *before* ingesting an observation. Its RL
variant reports over 1.6× gains over strong baselines at high task complexity, **with the advantage
growing as the budget shrinks** — which is the signature of a genuine allocation effect rather than
a retrieval one. That growth-as-budget-shrinks pattern is the diagnostic you should carry: if a
technique's advantage *grows* under pressure, it is fixing allocation; if it is flat, it is fixing
scoring.

### 3.2 The chain hash, and what a mid-prefix edit actually costs

vLLM computes a block's cache key as

> **h_j = H( h_{j−1} , tokens[ j·P : (j+1)·P ] , extra_keys )**

| Symbol | Reads as |
|---|---|
| *h_j* | the cache key of logical block *j* |
| *h_{j−1}* | the key of the **parent** block — this is what makes it a chain |
| *P* | block size in tokens (16 by default) |
| *n* | total context length in tokens |
| *p* | the token position at which you edit |

Because the parent key is folded in, a key is position-dependent and strictly prefix-ordered. The
match loop therefore **breaks at the first miss**, because a later hit is impossible by
construction. Two consequences, and they are different numbers:

**Case 1 — length-preserving in-place edit at position *p*:**

```
recompute_tokens = n − ⌊p/P⌋ · P
```

**Case 2 — prefix replacement, i.e. actual compaction.** You replace tokens [0, *p*) with a summary
of *s* tokens. The new sequence is `summary(s) ++ tail(n − p)`. Block 0's *content* changed, so
h_0 changed, so every downstream key changed:

```
new_context      = s + (n − p)
recompute_tokens = s + (n − p)          ← the entire new context
```

**Worked, at n = 200,000, P = 16, summary = 10% of the edited span** `[M]` (deterministic output of
the Exercise A script, `recompute_tokens_inplace` / `recompute_tokens_replace`):

| Edit at | In-place recompute | Prefix-replacement new context = recompute |
|---|---|---|
| p = 50,000 (25%) | 150,000 (75% of context) | 155,000 |
| p = 100,000 (50%) | 100,000 (50%) | 110,000 |
| p = 120,000 (60%) | 80,000 (40%) | 92,000 |
| p = 180,000 (90%) | 20,000 (10%) | 38,000 |

Read the shape: **the cost of an edit is proportional to everything after it, not to what you
changed.** Rewriting the oldest 60% of a 200k transcript does not cost "the summary" — it costs a
full prefill of the ~92k-token result.

> **Bridge, and where it breaks.** This is a log-structured store with a hash chain: append cheap,
> mid-stream overwrite costs the tail. You have priced exactly this in LSM compaction and in
> content-addressed backup.
>
> **Where it breaks:** in an LSM you compact to *reclaim space you already own* and the merged output
> is byte-verifiable against its inputs. Here you compact to *fit a budget you do not own* and the
> output is required to differ from its inputs, so there is nothing to verify against. The write
> amplification is real; the integrity check is not available.

### 3.3 The compaction ledger — why gentle compaction is dominated

Now put the per-event cost into a session. Let *A* be the total tokens the agent appends over the
session, *B* the context budget, *f* the fraction of context compacted when the budget is hit,
and *r* the summary ratio (summary length ÷ length of the span it replaces).

Each compaction event:

```
new context after event   = B·(1 − f) + B·f·r  =  B·(1 − f(1−r))
headroom bought per event = B − B·(1 − f(1−r)) =  B·f·(1−r)
prefill charged per event = B·(1 − f(1−r))          ← the whole surviving context
```

Number of events over the session and the prefill overhead relative to a never-compacting baseline
that pays only *A* (append-only, chain intact):

```
events        ≈ ⌈ (A − B) / (B · f · (1−r)) ⌉

prefill_ratio = 1 + events · B · (1 − f(1−r)) / A

              → 1 / ( f · (1−r) )        as A ≫ B
```

**Every symbol:** *A* = total tokens the session appends; *B* = budget; *f* = compacted fraction;
*r* = summary ratio; *events* = how many times compaction fires; *prefill_ratio* = total prefill
tokens paid divided by the prefill an uncompacted session would pay.

**Check the closed form against the measurement** `[M]` (Exercise A, *A* = 240,000, *B* = 120,000,
*r* = 0.10, deterministic):

| *f* | events (formula) | events (run) | prefill_ratio (formula) | prefill_ratio (run) | attention_ratio (run) |
|---|---|---|---|---|---|
| 0.30 | ⌈120000/32400⌉ = 4 | 4 | 1 + 4·87,600/240,000 = **2.46** | **2.47** | 0.68 |
| 0.50 | ⌈120000/54000⌉ = 3 | 3 | 1 + 3·66,000/240,000 = **1.83** | **1.83** | 0.63 |
| 0.60 | ⌈120000/64800⌉ = 2 | 2 | 1 + 2·55,200/240,000 = **1.46** | **1.46** | 0.60 |
| 0.90 | ⌈120000/97200⌉ = 2 | 2 | 1 + 2·22,800/240,000 = **1.19** | **1.19** | 0.52 |

The closed form and the simulation agree to within 0.01 at every point (the one 0.01 gap, at
*f* = 0.30, is integer truncation of `int(n·f)` inside the simulation), which is the only reason to
trust either.

**The non-obvious conclusion, and it inverts the usual instinct:** compacting *gently* — a small
*f* — fires more often, and **each firing costs the entire surviving context**, which is large
precisely because you compacted gently. So gentle compaction costs **more** prefill (2.47× vs 1.19×)
*and* saves **less** attention volume (0.68 vs 0.52). Under a chain-hash prefix cache, gentle
compaction is dominated on both axes.

> The only reason left to compact gently is **information preservation** — and that is exactly the
> axis nobody puts on the same plot as the token cost. `[C]` Governance Decay (arXiv 2606.22528)
> measures the quality side; the compaction papers in §5 of the mirrored note measure the token
> side; no paper I can find measures both against the same *f*. That is a gap you can close at
> 300M scale on this machine.

### 3.4 Retrieval under a fixed budget — the K/N law

This is the math behind break one. Model: *N* notes, each a unit vector in **R^D**, isotropically
distributed. A query is the target note plus Gaussian noise, renormalised:

```
q = unit( v_target + σ · g ),   g ~ N(0, I_D)
```

| Symbol | Reads as |
|---|---|
| *N* | number of notes in the store |
| *D* | embedding dimension |
| σ | query noise — how far the phrasing of the query sits from the note that answers it |
| *K* | how many notes the context budget can afford to admit (top-*k*) |
| *s\** | cosine similarity between the query and its true target |
| *R* | the target's rank — how many notes score above it |

For a distractor *v_j* independent of *q*, the similarity *q·v_j* has a distribution that depends
on *D* only — mean 0, standard deviation ≈ 1/√D — and **does not depend on N at all**. The target's
similarity *s\** concentrates near 1/√(1+σ²), also independent of *N*. So conditional on *s\**, the
rank is binomial:

```
R ~ Binomial( N − 1 , p )        where  p = P( q·v_distractor > s* )

E[R] = (N − 1) · p        ⟹        E[R] / N  →  p ,  a constant in N
```

**and therefore**

```
hit@K = P(R < K)   ⟹   to hold hit@K at a fixed level you need   K ∝ N
```

**Measured** `[M]` (Exercise B: exact brute-force cosine, *D* = 256, σ = 0.55, 5 seeds × 2,000
queries, `numpy` 2.4.4, CPU):

| *N* | hit@1 | hit@10 | mean rank | **rank/N** |
|---|---|---|---|---|
| 100 | 0.260 ± 0.007 | 0.698 ± 0.014 | 9.7 ± 0.4 | **0.0969** |
| 1,000 | 0.085 ± 0.005 | 0.294 ± 0.012 | 97.3 ± 2.8 | **0.0973** |
| 10,000 | 0.024 ± 0.003 | 0.097 ± 0.004 | 1,004.7 ± 32.3 | **0.1005** |
| 100,000 | 0.007 ± 0.001 | 0.027 ± 0.003 | 10,063.2 ± 153.9 | **0.1006** |

`rank/N` is constant to three significant figures across a 1,000× range of store size. And the
budget consequence, measured directly at σ = 0.45:

| *N* | *K* needed for hit@K ≥ 0.90 | **K/N** |
|---|---|---|
| 1,000 | 200 | **0.2000** |
| 10,000 | 2,000 | **0.2000** |
| 100,000 | 20,000 | **0.2000** |

**To keep 90% recall you must admit 20% of the entire store into the prompt, at every scale.** That
is not a retrieval result; it is a budget result, and it is why §2.3 calls the usual framing a
category error.

**Does a better embedder save you?** It moves *p*, not the exponent `[M]` (Exercise B, *N* = 10,000,
*K* = 10):

| σ (query noise) | hit@1 | hit@10 | rank/N |
|---|---|---|---|
| 0.20 | 0.843 ± 0.008 | 0.967 ± 0.003 | **0.0002** |
| 0.35 | 0.159 ± 0.009 | 0.391 ± 0.005 | 0.0223 |
| 0.45 | 0.054 ± 0.002 | 0.183 ± 0.007 | 0.0590 |
| 0.55 | 0.024 ± 0.003 | 0.097 ± 0.004 | 0.1005 |
| 0.70 | 0.010 ± 0.002 | 0.043 ± 0.004 | 0.1576 |

A 500× improvement in the fractional rank buys a 500× larger store at the same recall. It is a
constant. *N* is not.

**Two honest caveats.** First, this model is *optimistic*: real note stores are clustered and
anisotropic, and they accumulate near-duplicates of the very notes you want — which are exactly the
distractors that outrank the target. Second, it assumes an exact index; every real ANN index is
strictly worse. So the measured curve is a **ceiling**, not a forecast. `[C]` This is consistent
with PrecisionMemBench (arXiv 2605.11325, May 2026), which points out that benchmarks score
*answers*, not retrieval, so a system that dumps its whole store gets perfect recall while hiding a
precision failure — reported baseline precision clusters at 0.22 and below. `[A]` high confidence:
that paper also ships a competing system that scores perfectly, so read the headline as
vendor-adjacent and the methodological point as sound.

### 3.5 Memory system versus just keeping the tokens — the crossover

Before building anything, price the null. A long-context baseline pays *n* tokens of prefill per
turn if the prefix cache misses, or *a* tokens if it hits. A fact-store pays a one-time write cost
plus a fixed per-turn read of *K·s* tokens. `[C]` arXiv 2603.04814 (Mar 2026) finds long-context
GPT-5-mini achieves *higher* factual recall than a Mem0-based fact store on LongMemEval and LoCoMo,
with the memory system competitive only on PersonaMemv2 — while giving the memory system a
structurally better cost curve, **crossing over at roughly ten turns at 100k context**. `[C]` Mem0
itself (arXiv 2504.19413) reports the opposite quality ordering on LoCoMo. `[C]` And arXiv 2604.11628
argues the bottleneck is not architecture at all but "signal sparsity," with a minimalist
retrieve-and-generate baseline beating hierarchical-summarization systems.

Three answers, all 2025–26, unresolved. **The actionable part is that the cost curve and the quality
curve point in opposite directions, so any deployment decision is a crossover calculation and not a
ranking.** And the crossover moves: `[C]` *Can I Buy Your KV Cache?* (arXiv 2606.13361, Jun 2026)
measures KV reuse at **9–50× cheaper in compute than prefill on Qwen3-4B**, which pushes the
crossover further toward "just keep the tokens and reuse the cache" every time prefix-cache hit rate
improves.

---

## 4. Why it matters for Proteus and Mnemosyne

### 4.1 The boundary rule says agent memory is not Mnemosyne — and that is the point

`CLAUDE.md` fixes the dependency graph: `mnemosyne → torch`, never upward. An agent-memory system
sits **above** the model; Mnemosyne sits **below** it, owning the KV tier. So nothing in this module
becomes a Mnemosyne module.

What *does* become one is the observation from §2.5: every mechanism the agent layer is missing is a
**read on state Mnemosyne already owns**. Occupancy per block. Recency per block. Access count per
block. Whether a block is pinned. What a mutation would cost in recomputed tokens. Those are
model-agnostic quantities — which means exposing them passes the separability acceptance test
(build the wheel, install into a clean torch-only venv, run the suite green) while being exactly
what an agent-memory policy needs.

### 4.2 Three consequences, stated as consequences and not as decisions

**One — occupancy telemetry is an interface, not instrumentation.** VISTA's result says the
per-block token/recency/access counters are what the *policy consumer* needs, whether that consumer
is a model or a heuristic. Mnemosyne should expose them as a first-class read on the cache, not as a
logging side-channel. `Argus` is reserved for exactly this.

**Two — price the prefix invalidation, always.** Any Mnemosyne policy that mutates history must
report `recompute_tokens` alongside tokens saved. §3.2 is the formula; the vLLM pointers in §5 are
the reference implementation. A policy that reports only tokens saved is reporting half a ledger.

**Three — measure forgetting, not only recall.** `[C]` arXiv 2606.15903 and `[C]` arXiv 2606.30306
independently observe that the field does not. A supersede/release/purge surface with its own tests
is cheap and, until very recently, uncontested. It is contested now — see §8.5 — which raises the
value of doing it, not lowers it.

### 4.3 The counterfactual our hardware buys, computed at a real config

`research/memory/agent-memory-systems.md` closes with an `[A]` arithmetic claim it explicitly marks
"unverified for any specific config." Here it is, verified at two.

Per-token KV bytes = `2 · L · n_kv · d_head · b` (derived in `attention-variants-and-kv-cost.md` §3.3).

| Config | *L* | *n_kv* | *d_head* | KV / token | Tokens that fit in `[M]` 62 GiB |
|---|---|---|---|---|---|
| the Exercise C model (37.8M params) | 12 | 8 | 64 | 24 KiB | **2,708,821** |
| a 300M-class Proteus arm `[A]` | 24 | 8 | 64 | 48 KiB | **1,354,410** |

(62 GiB = 65,011,712 KiB; divide by the KiB/token column.)

`[M]` fast tier ≥62 GiB at ~200 GB/s (`notebook/uma-carveout-controls-fast-tier.md`, single run per
arm). So at our scale **the entire uncompacted KV of a million-token agent trajectory fits in fast
memory**. That makes the expensive counterfactual cheap here: you can run a compaction policy *and*
its never-compacted control in the same experiment and measure exactly what the policy cost. That is
the attribution measurement the whole evaluation literature says is missing, and it is
capacity-bound rather than FLOPS-bound — precisely what this machine buys.

Two hard constraints on how you build it, both `[M]`: single tensors ≥32 GiB **hang silently at 0%
CPU** (`ASSUMPTIONS.md → large-tensor-fault-32gib`), so any KV pool must be **paged** rather than
allocated as one buffer — which is the vLLM block-table design in §5 for a reason that is not about
fragmentation. And `[M]` default SDPA on gfx1151 retains the score matrix
(`ASSUMPTIONS.md → sdpa-is-memory-efficient`), which combines with the 32 GiB fault to cap
single-shot prefill hard — see §6, Exercise C.

### 4.4 What this module says *not* to build

`research/synthesis.md` decided: **ship an attribution instrument, add no new eviction policy to a
field that has ~30 of them and no dominance result.** This module is consistent with that. The three
exercises are proto-instruments — a cost ledger, a budget-law probe, a prefill timer — not policies.
Resist the pull of §5 of the mirrored note; it lists eight families of compaction and the honest
summary is that none of them has been compared to the others.

---

## 5. Read the code

All paths relative to `research/reference/`. Clones are gitignored; run `scripts/fetch_reference.sh`
first. Line numbers are pinned to the revisions in `PROVENANCE.md`. Every pointer below was opened
and the named symbol confirmed on the named line on 2026-07-26.

**Read these in the order given.** The point is not to learn vLLM; it is to see four mechanisms
implemented correctly one layer below where the agent literature is re-inventing them badly.

### 5.1 The chain hash — why compaction costs the tail

| Where | What to look for |
|---|---|
| `memory/vllm/vllm/v1/core/kv_cache_utils.py:596` | `hash_block_tokens`. Read the hashed tuple at `:622` — `hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))`. **The parent hash is an input.** That single design choice is the whole of §3.2. Note also `:617` — a missing parent defaults to `NONE_HASH`, so position 0 is a real anchor, not a special case. |
| `memory/vllm/vllm/v1/core/kv_cache_utils.py:691` | `get_request_block_hasher`. Where a prompt is chopped into `hash_block_size` chunks and hashed incrementally, resuming from `len(request.block_hashes)`. Look for the fact that **only full blocks are hashed** — the tail partial block is never keyed, which is a silent alignment tax on every edit. |
| `memory/vllm/vllm/v1/core/single_type_kv_cache_manager.py:658` | `find_longest_cache_hit`. Read it to confirm the loop **breaks at the first miss**. Ask yourself why that is not a performance shortcut: with a chain hash a later hit is impossible by construction, so there is nothing to gain by continuing. This is the difference between a chain hash and a content hash, in five lines. |
| `memory/vllm/vllm/v1/core/kv_cache_manager.py:225` | `get_computed_blocks`. Note `max_cache_hit_length = request.num_tokens - 1`: **a 100% prefix match never skips 100% of the work**, because you need one forward pass to get logits, and the result is then floored to block alignment. An exact-duplicate prompt still recomputes a whole trailing block. |

### 5.2 Free is not evict — the resurrection path the agent layer lacks

| Where | What to look for |
|---|---|
| `memory/vllm/vllm/v1/core/block_pool.py:719` | `free_blocks`. It decrements `ref_cnt` and pushes the block back on the free queue **with contents and hash intact**. Look at `:741`–`:742`: hash-less blocks are *prepended* (die first), hashed blocks *appended* (linger as reuse candidates). The free list and the LRU victim cache are the same list. |
| `memory/vllm/vllm/v1/core/block_pool.py:702` | `touch`. The resurrection: on a prefix hit, a zero-refcount block is unlinked from the free queue in O(1) and its refcount bumped. **A freed block is still matchable.** |
| `memory/vllm/vllm/v1/core/block_pool.py:679` | `_maybe_evict_cached_block`. The only place a block leaves the hash table — and it is called lazily from the *allocation* path, not from the free path. So "blocks in use" and "entries available for hits" are two different, non-obvious numbers. |
| `memory/vllm/vllm/v1/core/block_pool.py:198` | `get_cached_block`. The probe. Returns `None` if **any** KV cache group misses, so a hybrid SWA/global model matches at the *intersection* of its two block tables. Worth holding next to Laguna's 12-full/36-sliding layout. |

**The transplant question to hold while reading:** every agent-memory compaction implementation you
have seen destroys the compacted text at commit time. What would `touch` look like at the note tier,
and what is the equivalent of `ref_cnt`?

### 5.3 Eviction as a topology problem, not a recency problem

| Where | What to look for |
|---|---|
| `memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:565` | `evict`. Read `:571` — the candidate set is `self.evictable_leaves`, **leaves only**. Then `:585`: a parent becomes a candidate only after it loses its last child *and* holds no lock. Eviction peels the frontier inward; a hot child keeps a cold parent resident indefinitely. |
| `memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:594` | `inc_lock_ref`. It walks all the way to the root. **One in-flight request pins an arbitrarily deep prefix chain** — closer to a pinned dentry chain than to a page refcount. The agent analogue is exact: an active sub-task pins its whole plan chain. |
| `memory/sglang/python/sglang/srt/mem_cache/evict_policy.py:16` | `LRUStrategy`. The entire replacement-policy surface is one `get_priority(node)` function returning a float or tuple; LRU is `node.last_access_time`, LFU is `(hit_count, last_access_time)`. **This is the shape a pluggable Mnemosyne policy should have** — one comparator, everything else structural. |
| `memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:217` | `TreeNode`. Note the four fields every policy reads: `lock_ref`, `last_access_time`, `hit_count`, `priority`. That is the counter set VISTA says the agent layer is missing. It already exists here. |

### 5.4 Admission control and leases — the untried transplant

| Where | What to look for |
|---|---|
| `memory/mooncake/mooncake-store/src/master_service.cpp:6382` | `BatchEvict`. Read the predicate at `:6393`: a replica is evictable only when completed and at refcount zero — and the surrounding sweep only considers objects whose **lease has expired**. The hot signal is a TTL renewal, not a touch bit. |
| `memory/mooncake/mooncake-store/src/master_service.cpp:5211` | `TryPushPromotionQueue`. The frequency gate at `:5224`–`:5228`: a count-min sketch increments and the object is rejected unless it clears `promotion_admission_threshold_`. **A single cold hit cannot pollute the fast tier.** Then a second, independent watermark gate at `:5233`. Two gates, both cheap, both absent from every agent-memory system in the literature. |
| `memory/mooncake/mooncake-store/include/replica_selection.h:122` | `SelectBestReplica`. Read the comment at `:118`–`:121`: the tier ladder is a **fixed preference order**, not a latency/cost model. Useful calibration — even the most storage-hierarchy-shaped system in the reference library does not run a cost model. |

**The open experiment, stated concretely:** Mooncake gates *promotion* by frequency. No agent-memory
system gates *writes* by frequency. Does a TinyLFU-style admission gate on note writes beat
unconditional write-on-observation at matched budget? Nobody has run it.

### 5.5 One pointer back into Track A/B

| Where | What to look for |
|---|---|
| `training/nanogpt/model.py:306` | `generate`. It re-runs the full prefix per sampled token — quadratic and cacheless by design. Read it once to internalise what the prefix cache is buying and why every number in §3.3 is about prefill rather than decode. |

---

## 6. Exercises

Activate the lab environment first, in PowerShell, dot-sourced so variables survive:

```powershell
. .\scripts\activate-lab.ps1
```

**Standing caveats.** The Hardware Validation Gate has not run, so nothing measured on this machine
is evidence by house standard yet — these are instrument-shakedown runs and should be labelled as
such in your notebook entry. `[M]` bf16 numerics on gfx1151 are unproven
(`ASSUMPTIONS.md → bf16-numerics-unproven`); timing claims are unaffected, accuracy claims are
provisional. `[M]` Single tensors ≥32 GiB hang silently at 0% CPU
(`ASSUMPTIONS.md → large-tensor-fault-32gib`) — Exercise C explains why this binds here specifically.

Write scratch scripts under `notebook/`. All three are exempt from TDD only until reused; on reuse
they migrate into `themis/` and acquire tests.

---

### Exercise A — the compaction ledger: price a policy in prefill, not in summarizer tokens

**Goal.** Turn "compaction is cheap" from an assumption into a number, and reproduce a closed form
that most of the 2026 compaction literature does not compute.

**Hardware:** none. Pure Python, no numpy, no torch. **Runtime:** `[M]` 0.4 s to run; 30 minutes to
write and reason about.

```python
"""Price a compaction policy in prefill tokens, not summarizer tokens."""
P = 16  # prefix-cache block size in tokens (vLLM default)

def recompute_tokens_inplace(n: int, p: int, block: int = P) -> int:
    """Length-preserving edit at position p in an n-token context."""
    return n - (p // block) * block

def recompute_tokens_replace(n: int, p: int, summary_len: int) -> int:
    """Replace tokens [0,p) with summary_len tokens. The chain restarts at 0."""
    return summary_len + (n - p)

def run_session(turns=200, tokens_per_turn=1200, budget=120_000,
                compact_fraction=0.6, summary_ratio=0.10):
    n = n_compact = 0
    prefill_plain = prefill_compact = 0
    attn_plain = attn_compact = 0
    compactions = 0
    for _ in range(turns):
        n += tokens_per_turn
        prefill_plain += tokens_per_turn      # prefix chain intact: only new tokens
        attn_plain += n
        n_compact += tokens_per_turn
        prefill_compact += tokens_per_turn
        if n_compact > budget:
            p = int(n_compact * compact_fraction)
            s = int(p * summary_ratio)
            prefill_compact += recompute_tokens_replace(n_compact, p, s)
            n_compact = s + (n_compact - p)
            compactions += 1
        attn_compact += n_compact
    return dict(compactions=compactions,
                prefill_ratio=prefill_compact / prefill_plain,
                attn_ratio=attn_compact / attn_plain,
                final_plain=n, final_compact=n_compact)

for f in (0.30, 0.50, 0.60, 0.75, 0.90):
    r = run_session(compact_fraction=f)
    print(f"f={f:.2f}  events={r['compactions']}  "
          f"prefill_x={r['prefill_ratio']:.2f}  attn_x={r['attn_ratio']:.2f}")
```

**Deliverable — a table and one derived claim.**

1. Reproduce this exactly `[M]` (deterministic integer arithmetic — if you do not get these digits,
   your accounting differs from mine and one of us is wrong):

   | *f* | events | prefill_x | attn_x |
   |---|---|---|---|
   | 0.30 | 4 | 2.47 | 0.68 |
   | 0.50 | 3 | 1.83 | 0.63 |
   | 0.60 | 2 | 1.46 | 0.60 |
   | 0.75 | 2 | 1.33 | 0.55 |
   | 0.90 | 2 | 1.19 | 0.52 |

2. Check each `prefill_x` against the closed form in §3.3,
   `1 + events·B·(1 − f(1−r))/A`. It should match to two decimals.
3. Sweep `budget ∈ {60_000, 120_000, 240_000}` and confirm the third produces **zero** compactions
   and `prefill_x == 1.00` exactly. If it does not, your budget check fires on the wrong side of the
   append.
4. **Write the one-line finding in your notebook entry:** state whether gentle or aggressive
   compaction is cheaper on *both* axes, and name the single term in the closed form that causes it.

**How this can fail, which is why it is worth running.** If you model prefill as "only the summary
is recomputed" — the way most papers price it — every `prefill_x` collapses to roughly 1.00 and the
whole ranking disappears. Run it that way once, deliberately, and see the conclusion invert. That
inversion is the module.

---

### Exercise B — the K/N law: prove retrieval is not the failing component

**Goal.** Show that recall at a fixed context budget degrades with store size **under an exact
index**, measure the exponent, and measure how much a better embedder actually buys.

**Hardware:** CPU only, `numpy`. No GPU, no torch. **Runtime:** `[M]` 166 s for all three parts
below (`numpy` 2.4.4, Python 3.12.10); the snippet as printed is the first part and takes ~20 s.
Budget ~45 min including reading the output properly. Peak memory ~1 GB at the *N* = 100,000 rows —
if that is tight, drop the last row.

```python
"""Does retrieval quality degrade with store size when the index is exact?"""
import numpy as np

D, K, NOISE, SEEDS = 256, 10, 0.55, (0, 1, 2, 3, 4)

def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

def trial(n_notes, n_queries, seed, k=K, noise=NOISE):
    rng = np.random.default_rng(seed)
    store = unit(rng.standard_normal((n_notes, D)))
    targets = rng.integers(0, n_notes, size=n_queries)
    q = unit(store[targets] + noise * rng.standard_normal((n_queries, D)))
    sims = q @ store.T                                   # exact cosine, no index
    kk = min(k, n_notes - 1)
    topk = np.argpartition(-sims, kk, axis=1)[:, :kk]
    hit_at_k = (topk == targets[:, None]).any(axis=1).mean()
    hit_at_1 = (sims.argmax(axis=1) == targets).mean()
    rank = (sims > sims[np.arange(n_queries), targets][:, None]).sum(axis=1)
    return hit_at_1, hit_at_k, rank.mean()

for n in (100, 1_000, 10_000, 100_000):
    r = np.array([trial(n, 2000, s) for s in SEEDS])
    m, sd = r.mean(0), r.std(0, ddof=1)
    print(f"N={n:>7}  hit@1={m[0]:.3f}+-{sd[0]:.3f}  hit@K={m[1]:.3f}+-{sd[1]:.3f}  "
          f"rank/N={m[2]/n:.4f}")
```

**Deliverable — one invariant, one curve, one falsification test.**

1. **The invariant.** `rank/N` should be constant across all four *N*. Mine `[M]`: 0.0969, 0.0973,
   0.1005, 0.1006 (5 seeds, 2,000 queries each). Report yours with its spread. **This is the number
   that proves the index is not the problem** — the search was exhaustive.
2. **The curve.** Extend the script to find, for σ = 0.45, the smallest *K* with hit@K ≥ 0.90 at
   *N* ∈ {1,000, 10,000, 100,000}. Mine `[M]`: 200, 2,000, 20,000 — i.e. `K/N = 0.2000` at every
   scale. Plot `K` against `N` on log-log axes and report the slope. **Prediction: 1.0.**
3. **The falsification you should try.** Sweep σ ∈ {0.20, 0.35, 0.45, 0.55, 0.70} at *N* = 10,000
   and see whether a better embedder changes the *shape* or only the *offset*. Mine `[M]`: rank/N
   goes 0.0002 → 0.0223 → 0.0590 → 0.1005 → 0.1576, a ~500× range in the constant with no change to
   the *N*-scaling. If you find a σ where the log-log slope departs from 1.0, that is a real result
   and it contradicts §3.4 — write it up.
4. **The honest caveat to record.** This is i.i.d. isotropic embeddings, which is the *best* case.
   Real stores are clustered and accumulate near-duplicates of the target, which are precisely the
   distractors that outrank it. State in your entry that the measured curve is a **ceiling**.

---

### Exercise C — what a compaction event costs in seconds on this machine

**Goal.** Convert `recompute_tokens` into wall clock at Proteus ablation scale, and in the process
hit the two hardware constraints that decide how any Mnemosyne KV pool must be built. This closes
open question 2 in `research/memory/agent-memory-systems.md` ("what does compaction actually cost in
prefill? No training required").

**Hardware:** gfx1151, native Windows, lab venv. **CPU fallback given.**
**Runtime:** `[M]` 62 s for the default GPU arm, 8 s for the AOTriton arm, ~7 s on CPU with a stock
torch build (both arms are the same six shapes; the difference *is* the finding).

Build a random-init decoder at ablation scale — no weights downloaded, no training — and time
forward-only prefill against context length. Full script shape:

```python
"""Turn recompute_tokens into seconds. Random init; we only want prefill wall clock."""
import time, torch, torch.nn as nn, torch.nn.functional as F

L, D, NH, NKV, DH = 12, 512, 8, 8, 64          # 37.8M params
LENGTHS = (512, 1024, 2048, 4096, 8192, 16384)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(D), nn.RMSNorm(D)
        self.q = nn.Linear(D, NH * DH, bias=False)
        self.k = nn.Linear(D, NKV * DH, bias=False)
        self.v = nn.Linear(D, NKV * DH, bias=False)
        self.o = nn.Linear(NH * DH, D, bias=False)
        self.up = nn.Linear(D, 4 * D, bias=False)
        self.dn = nn.Linear(4 * D, D, bias=False)
    def forward(self, x):
        B, T, _ = x.shape
        h = self.n1(x)
        q = self.q(h).view(B, T, NH, DH).transpose(1, 2)
        k = self.k(h).view(B, T, NKV, DH).transpose(1, 2)
        v = self.v(h).view(B, T, NKV, DH).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, NH * DH))
        return x + self.dn(F.silu(self.up(self.n2(x))))
# stack L Blocks, .eval(), bf16 on cuda / fp32 on cpu, best-of-3 under torch.no_grad()
# print T, milliseconds, microseconds-per-token
```

Run it **twice**: once as-is, once with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` set in the
environment.

**Deliverable — three numbers and one refusal.**

1. **The µs/token curve.** Mine `[M]`, 37.8M params, batch 1, bf16, best-of-3, two independent
   fresh-process runs agreeing to <0.2%, torch `2.12.0a0+rocm7.13.0a20260313`:

   | *T* | default µs/token | with AOTriton flag µs/token |
   |---|---|---|
   | 512 | 27.0 | 14.6 |
   | 1,024 | 71.7 | 13.7 |
   | 2,048 | 115.9 | 12.5 |
   | 4,096 | 161.6 | 17.8 |
   | 8,192 | 317.3 | 24.3 |
   | 16,384 | 631.8 | 38.3 |

   **Prediction to check: the default column roughly doubles with each doubling of *T* — prefill is
   quadratic — while the flagged column is nearly flat to 2k and then grows slowly.** Over a 32×
   range of context the per-token cost blows up ~23× by default and ~2.6× with the flag.

2. **The flag's effect at the largest shape.** Mine `[M]`: at *T* = 16,384, 10,350.62 ms and
   10,366.04 ms default versus 627.06 ms and 627.55 ms flagged — **16.5×**. This is the first local
   evidence for the "~19× AOTriton attention speedup" that `CLAUDE.md` flags as reported but
   unverified. Two fresh-process runs per arm is **below** the house ≥3-seed standard; report it as
   an instrument observation, and pre-register it properly before it enters `ASSUMPTIONS.md`.
   The flag is experimental and therefore a **numerics** change — do not make it a default until the
   Hardware Validation Gate runs numerics both ways.

3. **Seconds per compaction event.** Take your measured ms at the largest *T* you ran and multiply
   through §3.2. At *T* = 16,384 that is a re-prefill costing **10.35 s** by default and **0.63 s**
   flagged, *per compaction event*, on a 37.8M model. Now put that next to Exercise A's event
   counts and state, in one line, what a 200-turn session actually pays.

4. **The refusal — and this is the most important part of the exercise.** Do **not** raise *T* to
   131,072 to match the context lengths the compaction papers assume. Compute the score-matrix size
   first: `n_heads · T² · 2 bytes`. At *T* = 131,072 with 8 heads that is **274.9 GB** as a single
   tensor. `[M]` Default SDPA on this machine retains the score matrix
   (`ASSUMPTIONS.md → sdpa-is-memory-efficient`, 147.2 bytes/T²) and `[M]` single tensors ≥32 GiB
   hang silently at 0% CPU (`large-tensor-fault-32gib`). Solving `8 · T² · 2 ≤ 32 GiB` gives
   **T_max ≈ 46,340** — derived from two measured rows, not itself measured, and deliberately not
   tested because the failure mode is a silent hang. **Record the arithmetic; do not run the
   experiment.** Knowing which experiment not to run is the deliverable.

**CPU fallback, and a trap worth knowing.** Set `HIP_VISIBLE_DEVICES=-1`, use fp32, and cut
`LENGTHS` to `(256, 512, 1024, 2048)`. But **use a stock CPU or CUDA torch build, not the ROCm
wheel**: `[M]` on this machine, at *T* = 2,048 fp32, the same script takes 435 / 456 ms under
`torch 2.11.0+cu128` on CPU and **7,466 / 8,072 ms** under `torch 2.12.0a0+rocm7.13.0a20260313` on
CPU — a **~17× penalty** from the wheel's CPU backend alone, two runs per arm, stable across four
sequence lengths. That is a curriculum-wide caveat, not a local one: every CPU fallback in every
module inherits it.

For a fair device comparison use a matched *T*. At *T* = 2,048 `[M]`: CPU (stock wheel, fp32)
435–456 ms; GPU default (bf16) 228–237 ms; GPU flagged (bf16) 26–28 ms.

---

## 7. Self-check

Answers under their own heading at the end. Do not scroll.

1. Your agent-memory store has grown from 10,000 to 100,000 notes and answer quality has fallen.
   You migrate from HNSW to exact brute-force search, which is strictly more accurate. State what
   happens to recall@10 and why, in terms of a single quantity that did not change.

2. A compaction policy replaces the oldest 50% of a 200,000-token transcript with a 10% summary,
   under a prefix cache with block size 16. The summarizer emitted 10,000 tokens. How many tokens
   must be prefilled, and why is "10,000" not merely an underestimate but the wrong *kind* of
   number?

3. Two teams argue about compaction aggressiveness. Team Gentle compacts 30% of context each time
   the budget is hit; Team Aggressive compacts 90%. Both use the same summary ratio and the same
   budget. Which pays more total prefill over a long session, which retains more attention volume,
   and what is the one axis on which Team Gentle's position could still be correct?

4. vLLM's `free_blocks` does not evict. Name the two distinct states a KV block can be in that a
   naive "free list" model collapses into one, and say what the agent-memory equivalent of the
   *second* state would have to provide.

5. You add a per-note recency counter, a per-note token count, and a per-note access count to your
   agent's prompt as a small table. You change nothing else — same retrieval, same store, same
   policy. `[C]` VISTA reports this kind of change lifting a frontier model from 22.7% to 50.7% on
   its benchmark. Explain, in the vocabulary of §2.3, why an accounting change can produce a
   capability change, and state the one thing this result does *not* license you to conclude.

6. Mooncake gates disk→DRAM promotion behind a count-min-sketch frequency threshold. Sketch the
   agent-memory transplant precisely: what is the object, what is the "hit," what is the fast tier,
   and what is the specific failure mode the gate would prevent that recency-plus-similarity does
   not?

---

## 8. What is still unsolved here

### 8.1 Contested — and here I depart from the mirrored note

`research/memory/agent-memory-systems.md` §3 calls the mid-prefix edit cost "the break nobody in the
agent literature models" and treats `recompute_tokens = n − ⌊p/P⌋·P` as a structural fact. Read at
face value, that is a claim about **vLLM's implementation**, and it is accurate — I verified the
chain hash and the break-on-first-miss loop myself (§5.1).

**It may not be a claim about transformers.** `[C]` *Models Take Notes at Prefill: KV Cache Can Be
Editable and Composable* (arXiv 2606.17107, Jun 2026) argues the opposite: that the KV entries a
model writes during prefill are **position-portable**, so precompiled segments can be
RoPE-repositioned and spliced into an arbitrary context "indistinguishable from full recompute"
(reported logit cosine similarity 0.90–0.999 across twelve models), turning time-to-first-token from
O(L²) into O(L), with a unified edit-and-compose agent reported "decision-identical to recompute at
up to 14.9× lower latency" and p90 TTFT cut 53–398× in a production benchmark.

If that holds, then the invalidation cost this module spends §3.2 and §3.3 computing is a property
of **chain hashing as a cache-keying strategy**, not of attention — and the correct Mnemosyne
response is not to price the invalidation but to **avoid it**, by keying on content and repositioning
rather than on the chain.

**Both positions are live and I am not resolving them here.** The note is not wrong about vLLM;
2606.17107 is a single-author June 2026 preprint with strong claims and, as far as I can establish,
no independent replication. **The cheapest test we can run:** take the Exercise C model, prefill a
segment at offset *a*, re-apply RoPE to move it to offset *b*, splice it into a different context,
and compare the resulting logits against a full recompute. That is one afternoon, it needs no
training, and it discriminates directly. Until it runs, price the invalidation.

### 8.2 Contested — where the leverage in context management actually is

Three 2026 papers, published within eight weeks of each other, give incompatible answers, and **none
of them compares against the others**:

- `[C]` BACM (2604.01664) and CompactionRL (2607.05378): you need a **learned policy**. CompactionRL
  reports GLM-4.5-Air to 66.8% SWE-bench Verified (+7.0) and 24.5% Terminal-Bench 2.0 (+3.1).
- `[C]` VISTA (2606.30005): you need **no policy at all**, only an interface that exposes occupancy.
  Training-free, 22.7 → 50.7 on LOCA-Bench.
- `[C]` CWL (2606.11213): you need **neither** — typed dependency-linked structure plus a
  deterministic LLM-free rule beats both. 89 tasks, 80M tokens, one session, no measurable
  degradation.

Do not let this curriculum pick one. The honest state is that the field has three mutually exclusive
diagnoses and no head-to-head.

### 8.3 Nobody plots token cost and information loss on the same axis

§3.3 shows the token-cost curve as a function of compaction aggressiveness *f*, and it monotonically
favours aggressive compaction. `[C]` Governance Decay (2606.22528) shows the quality curve — 0% →
30% mean, 59% worst-case violation — but at a single compaction setting. **The two curves cross
somewhere and nobody has drawn the crossing.** At 300M scale with a 62 GiB fast tier you can hold
the never-compacted control resident and sweep *f* against both axes in one experiment. That is a
tractable, publishable, and currently unoccupied measurement.

### 8.4 Is context proprioception emergent or trainable in?

VISTA's lift is measured on frontier backbones. Train two matched ~100M models on identical token
budgets, one with a synthetic occupancy-dashboard block in the prompt, and score on a multi-hop
synthetic retrieval task across occupancy levels. **A null at 100M says proprioception is a
scale-gated capability; a positive says it is an interface the whole size range can use** — which is
a strictly stronger claim than the paper makes.

### 8.5 Evaluation: the gap is closing faster than the mirrored note implies

The note's §6 concludes "a field that benchmarks reads and neglects deletes is not a mature field.
That is the opening." That was true when written and it is narrowing as of this month:

- `[C]` **MemOps** (arXiv 2607.12893, 2026-07-14 — twelve days old) benchmarks lifecycle memory
  *operations*: remembering, forgetting, updating, reflecting, and their compositions, via six
  categories of operation-level probes across long-context, retrieval-based, parametric and
  managed-memory systems. Its finding: systems are "far from uniformly reliable," session-level
  retrieval beats turn-level, and long-context models are notably weak at reconstructing ordered
  memory-state trajectories.
- `[C]` **Memora / FAMA** (arXiv 2604.20006, Apr 2026) introduces a Forgetting-Aware Memory Accuracy
  metric that explicitly penalises reliance on obsolete or invalidated memory, over conversations
  spanning weeks to months, across four LLMs and six memory agents — and reports frequent reuse of
  invalid memories with only marginal improvement over baselines.
- `[C]` **Control-plane placement** (arXiv 2606.15903, Jun 2026) remains the sharpest structural
  result: across 13 configurations on a 385-case adversarial surface, deterministic primitives get
  5% on identifier obfuscation and 0% cross-lingual; an inscribe-time LLM recovers canonicalization
  to 100% but gets 0% on intent-aware deletion; a mutation-time hook recovers intent-aware deletion
  (78–85%) and lifts overall to 91.7–93.2%. **Where the model sits in the pipeline determines which
  failures are even addressable.**

**What this changes for us:** "measure forgetting" is no longer uncontested territory, and any
Mnemosyne work on supersede/release/purge now has two external benchmarks it should be scored
against rather than a vacuum to fill. That is better, not worse — but it is a change since the note
was written and it should be treated as one.

### 8.6 The untried transplants

Both are small, both are ours to run, and neither exists in the literature as far as I can establish:

- **Admission control on writes.** A TinyLFU frequency gate on note *writes* versus unconditional
  write-on-observation, at matched budget. Mooncake does this at the KV tier
  (`master_service.cpp:5211`); no agent-memory system does it at the note tier.
- **Clean-versus-dirty eviction, isolated.** Build a synthetic agent task where some episode effects
  are externally persisted and some are not. Does a policy that reads that one bit beat recency at
  matched token budget? This isolates CWL's central claim from its LLM-annotation machinery, which
  is the part that makes CWL hard to attribute.

### 8.7 Is any of this memory at all?

`[C]` arXiv 2604.27707 (Apr 2026) argues current agentic memory implements **lookup**, not memory:
retrieval generalizes by similarity to stored cases while weight-based memory generalizes by applying
abstract rules; conflating them yields a generalization ceiling that no context size or retrieval
quality can overcome. It invokes Complementary Learning Systems theory — biology pairs fast exemplar
storage with slow weight consolidation, and agents implement only the first half. The MemOS/MemCube
line `[C]` (arXiv 2507.03724, arXiv 2505.22101) takes the opposite position, with a parametric /
activation / plaintext trichotomy and a scheduler migrating content between "tiers."

**A warning about that trichotomy, because it is the place the storage-hierarchy analogy is most
dangerous.** A storage hierarchy's defining property is that a datum can live at *any* level and the
level is a **performance** decision. In MemOS the level is a **type** decision: you cannot promote
plaintext to parametric without a training step, you cannot demote parametric to plaintext at all,
and the conversions are lossy and mostly one-way. Treat MemOS as a **catalog with a migration
policy**, not as a cache hierarchy. Both papers admit the lifecycle mismatch; neither resolves it.

### 8.8 The measurement this module could not make

Everything in §3.4 is synthetic isotropic embeddings. The K/N law is derived and measured on a model
whose distributional assumptions real note stores violate in the *unfavourable* direction. The
cheapest useful extension: run Exercise B against embeddings from a real sentence encoder over a real
corpus and check whether the log-log slope is still 1.0. `[A]` medium confidence it is *worse* than
1.0 because near-duplicate accumulation adds correlated distractors; low confidence in the magnitude.
That measurement would upgrade §3.4 from a clean derivation to a claim about the systems people
actually run.

---

## Answers to the self-check

**1.** Recall@10 falls, and the migration does not help — it may not change the number measurably at
all, because HNSW at reasonable settings is already near-exact on the top-10. The quantity that did
not change is the **query's similarity to its target**, which is a property of the embedder and the
phrasing, not of *N*. What changed is the number of distractors that get a draw from the same
similarity distribution. The target's *fractional* rank is invariant (§3.4, measured constant to
three significant figures), so its *absolute* rank grew 10× while your top-10 budget stayed at 10.
The failing component is the budget, not the index — and "exact search" is the ceiling, so there is
nothing further to buy on that axis.

**2.** `recompute_tokens = s + (n − p) = 10,000 + 100,000 = 110,000`. It is the wrong *kind* of
number because 10,000 is a **summarizer** cost — tokens generated by a model call — while 110,000 is
a **prefill** cost against an invalidated cache. They are different resources with different price
points and different latency profiles, and adding them is a category error. The block size of 16 is
a red herring here: prefix *replacement* changes block 0's contents, so the chain restarts at 0 and
the alignment term never enters. Block alignment only matters for a length-preserving in-place edit.

**3.** Team Aggressive pays **less** total prefill (`[M]` 1.19× vs 2.47× at the parameters in
Exercise A) and retains **less** attention volume (0.52 vs 0.68) — it wins on both cost axes. The
mechanism is in the closed form: each event costs the *entire surviving context*, and gentle
compaction both fires more often and leaves a larger context to re-prefill each time. The one axis
where Team Gentle could still be right is **information preservation**: aggressive compaction
discards more, and `[C]` Governance Decay (2606.22528) shows a dropped constraint takes violation
from 0% to 38%. Nobody has plotted both axes against the same *f* — §8.3.

**4.** The two states are (a) **allocated and referenced** — `ref_cnt > 0`, un-evictable — and
(b) **freed but still cached** — `ref_cnt == 0`, sitting on the free queue with its contents *and*
its hash entry intact, so `touch` can resurrect it on a later prefix hit
(`block_pool.py:719` → `:702`). A naive free list destroys state (b). Actual eviction happens later
and elsewhere, at allocation time (`block_pool.py:679`). The agent-memory equivalent of state (b)
would have to provide a **resurrection path**: compacted text must remain byte-addressable and
re-insertable after it leaves the prompt, with a stable identity that a later turn can hit. Self-GC's
recoverable sidecars and VISTA's full-fidelity archive are both attempts at exactly this, in text,
and both are reconstructions of something the KV layer gets for free.

**5.** In §2.3's vocabulary the problem is a knapsack whose *utilities* **u_i** must be estimated at
decision time. The model is the entity making the keep/drop decision, and from the prompt alone it
cannot observe *s_i* (how many tokens a block costs), how stale it is, or whether it has been used —
so it is solving a knapsack with no visibility into the constraint or the item weights. Adding
counters does not add capability; it **makes an already-latent policy executable**, which is why
VISTA is training-free. What the result does *not* license: concluding that the counters would help
a small model. The lift is measured on frontier backbones, and whether context proprioception is
scale-gated or interface-gated is exactly the open question in §8.4. Assuming it transfers to a 100M
model is an untested extrapolation — and testing it is cheap here.

**6.** Object = a candidate note (or episode) at write time. "Hit" = an observation matching an
existing note's key/topic, i.e. the same fact or situation recurring. Fast tier = the *retrievable
note store* — or, tighter, the subset of it eligible to be admitted into the prompt. The gate: on
each candidate write, increment a count-min sketch on a canonicalised key and refuse to create a
durable note until frequency clears a threshold; below threshold the observation stays in the
transcript only. The failure mode it prevents that recency-plus-similarity does not: **single-shot
pollution.** One incidental observation — a transient error string, a one-off tool output, an
injected instruction — becomes a permanent, retrievable, high-recency note that competes for the
top-*k* budget forever, and worse, participates in later similarity matches. Recency actively favours
it. A frequency gate requires corroboration before durability, which is both a quality control and,
per `[C]` arXiv 2606.04329 and `[C]` arXiv 2604.16548, a security control — a single injected write
cannot reach the store.

---

## Sources

### Local artifacts and measurements

`[M]` measurements produced by this module, all on the Z13 (gfx1151, native Windows) on 2026-07-26.
The Hardware Validation Gate has **not** run, so these are instrument-shakedown numbers, not
evidence by house standard.

- **Exercise A ledger** — deterministic integer arithmetic, no randomness, no hardware dependence.
  Parameters: 200 turns × 1,200 tokens/turn, budget 120,000, summary ratio 0.10, block size 16.
  Outputs reproduced exactly in §3.3 and §6.
- **Exercise B retrieval law** — `numpy` 2.4.4, Python 3.12.10, CPU. *D* = 256, 5 seeds × 2,000
  queries per cell, exact brute-force cosine. `rank/N` constant at 0.0969–0.1006 across
  *N* ∈ {100, 1e3, 1e4, 1e5}; `K/N` = 0.2000 for hit@K ≥ 0.90 at σ = 0.45 across
  *N* ∈ {1e3, 1e4, 1e5}. 166 s end-to-end.
- **Exercise C prefill timing** — torch `2.12.0a0+rocm7.13.0a20260313`, HIP 7.2.0, 37.8M-param
  12-layer decoder (*d* = 512, n_q = n_kv = 8, head_dim = 64), batch 1, bf16, forward-only,
  best-of-3 within process, **two independent fresh-process runs per arm**. At *T* = 16,384:
  10,350.62 / 10,366.04 ms default; 627.06 / 627.55 ms with
  `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` → **16.5×**. Below the house ≥3-seed standard; the
  two runs agree to <0.2%.
- **CPU-backend penalty** — same script, CPU, fp32, *T* = 2,048: 435 / 456 ms under
  `torch 2.11.0+cu128`; 7,466 / 8,072 ms under `torch 2.12.0a0+rocm7.13.0a20260313`. **~17×.**
- `ASSUMPTIONS.md` rows relied on: `gpu-fast-tier-size` (≥62 GiB at ~200 GB/s),
  `large-tensor-fault-32gib`, `sdpa-is-memory-efficient`, `bf16-numerics-unproven`, `torch-build`,
  `kv-per-token-laguna`, `single-device-only`.
- `notebook/uma-carveout-controls-fast-tier.md` — the fast-tier measurement, single run per arm.
- `research/memory/agent-memory-systems.md` — the note this module mirrors.
- `research/synthesis.md` — the decision to build an instrument rather than a policy.
- `curriculum/attention-variants-and-kv-cost.md` — the KV product used in §4.3.

### Code pointers

All verified on 2026-07-26 against the revisions in `research/reference/PROVENANCE.md`.

- `memory/vllm/vllm/v1/core/kv_cache_utils.py:596` — `hash_block_tokens`, the chain hash
- `memory/vllm/vllm/v1/core/kv_cache_utils.py:691` — `get_request_block_hasher`
- `memory/vllm/vllm/v1/core/single_type_kv_cache_manager.py:658` — `find_longest_cache_hit`
- `memory/vllm/vllm/v1/core/kv_cache_manager.py:225` — `get_computed_blocks`
- `memory/vllm/vllm/v1/core/block_pool.py:198` — `get_cached_block`
- `memory/vllm/vllm/v1/core/block_pool.py:679` — `_maybe_evict_cached_block`
- `memory/vllm/vllm/v1/core/block_pool.py:702` — `touch`
- `memory/vllm/vllm/v1/core/block_pool.py:719` — `free_blocks`
- `memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:217` — `TreeNode`
- `memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:565` — leaf-constrained `evict`
- `memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:594` — `inc_lock_ref`
- `memory/sglang/python/sglang/srt/mem_cache/evict_policy.py:16` — `LRUStrategy`
- `memory/mooncake/mooncake-store/src/master_service.cpp:5211` — TinyLFU-gated `TryPushPromotionQueue`
- `memory/mooncake/mooncake-store/src/master_service.cpp:6382` — lease-expiry `BatchEvict`
- `memory/mooncake/mooncake-store/include/replica_selection.h:122` — `SelectBestReplica`
- `training/nanogpt/model.py:306` — `generate`, the cacheless baseline

### arXiv

Ids inherited from `research/memory/agent-memory-systems.md` were machine-verified there on
2026-07-26. Five ids new to this module were verified by fetching their arXiv abstract pages on
2026-07-26 and are marked **new**.

**Taxonomy and framing**
- `2604.11364` — The Missing Knowledge Layer in Cognitive Architectures for AI Agents (Apr 2026)
- `2309.02427` — Cognitive Architectures for Language Agents (CoALA) (2023)
- `2512.13564` — Memory in the Age of AI Agents (Dec 2025, rev. Jan 2026)
- `2603.07670` — Memory for Autonomous LLM Agents (Mar 2026)
- `2504.15965` — From Human Memory to AI Memory (2025)
- `2605.06716` — From Storage to Experience (May 2026)
- `2607.08032` — What to Keep, What to Forget: A Rate–Distortion View of Memory Compaction (Jul 2026)

**Budget, compaction and context management**
- `2604.01664` — ContextBudget / BACM (Apr 2026)
- `2606.10616` — Learning What to Remember (OSL-MR) (Jun 2026)
- `2606.30005` — LLM Agents Are Latent Context Managers (VISTA) (Jun 2026)
- `2606.11213` — Beyond Compaction: Structured Context Eviction (CWL) (2026)
- `2607.00692` — Self-GC: Self-Governing Context for Long-Horizon LLM Agents (Jul 2026)
- `2606.17016` — TokenPilot: Cache-Efficient Context Management for LLM Agents (Jun 2026)
- `2607.05378` — CompactionRL (Jul 2026)
- `2606.22528` — **new** — Governance Decay: How Context Compaction Silently Erases Safety
  Constraints in Long-Horizon LLM Agents (2026-06-21)
- `2606.17107` — **new** — Models Take Notes at Prefill: KV Cache Can Be Editable and Composable
  (2026-06-14)
- `2606.13361` — **new** — Can I Buy Your KV Cache? (2026-06-11)

**Systems lineage**
- `2310.08560` — MemGPT: Towards LLMs as Operating Systems (2023)
- `2504.13171` — Sleep-time Compute (Apr 2025)
- `2502.12110` — A-MEM: Agentic Memory for LLM Agents (2025)
- `2605.28773` — FluxMem (May 2026)
- `2507.03724` / `2505.22101` — MemOS (Jul 2025 / May 2025)
- `2504.19413` — Mem0 (Apr 2025)
- `2603.09023` — The Missing Memory Hierarchy: Demand Paging for LLM Context Windows (Mar 2026)

**Procedural memory**
- `2606.23127` — AFTER: Managing Procedural Memory in LLM Agents (Jun 2026)
- `2605.23899` — From Raw Experience to Skill Consumption (May 2026)
- `2602.01869` — Skill-Pro (Feb 2026)

**Evaluation, failure and security**
- `2605.11325` — PrecisionMemBench / Structured Belief State (May 2026)
- `2606.15903` — Control-Plane Placement Shapes Forgetting (Jun 2026)
- `2606.30306` — Always-On Agents: Persistent Memory, State, and Governance (Jun 2026)
- `2607.12893` — **new** — MemOps: Benchmarking Lifecycle Memory Operations in Long-Horizon
  Conversations (2026-07-14)
- `2604.20006` — **new** — From Recall to Forgetting: Benchmarking Long-Term Memory for
  Personalized Agents (Memora / FAMA) (2026-04-21)
- `2604.16548` — A Survey on Long-Term Memory Security in LLM Agents (Apr 2026)
- `2606.04329` — From Untrusted Input to Trusted Memory (Jun 2026)
- `2603.04814` — Beyond the Context Window: Fact-Based Memory vs. Long-Context LLMs (Mar 2026)
- `2604.11628` — Back to Basics: Let Conversational Agents Remember with Just Retrieval and
  Generation (Apr 2026)
- `2604.27707` — Contextual Agentic Memory is a Memo, Not True Memory (Apr 2026)

**Long-context behaviour**
- `2508.07479` — Positional Biases Shift as Inputs Approach Context Window Limits (Aug 2025)
