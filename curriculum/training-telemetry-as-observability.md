---
title: Training telemetry as observability — what to emit, what it costs, and what it cannot tell you
version: 1.0.0
date: 2026-07-26
track: D — Training systems
prereqs: the-training-loop, loss-and-optimization, measuring-memory
difficulty: 2/5 conceptually — you have built this pipeline before; 4/5 in the three places where your instincts are actively wrong
time: 3–4 h reading and working the arithmetic; 1.5–2.5 h for the three exercises — but very little of that is compute (measured: Exercise A ~4 min, B ~2 min, C ~1 min of GPU / a few minutes of CPU for all three arms). The hours are reading the output and running the two follow-ups Exercise C asks for.
mirrors: research/notes/pretraining-recipes.md §9 (the JSONL schema), research/synthesis.md (the attribution deliverable)
---

# Training telemetry as observability

**This is the module you should be able to teach back.** Not because it is the hardest — Track C
is harder — but because it is the one where thirty years of your own experience is the *majority*
of the content, and the residue that is genuinely new is small, sharp, and enumerable. If you can
state the three breaks from memory and defend the arithmetic behind each, you own this.

**Difficulty and time, honestly.** Sections 2 and 5 will feel like a tour of a system you designed.
Section 3 is where the work is: five short derivations, none harder than an exponential decay, two
of which will change how you configure a run — and one of which my own exercise falsified, which is
the most useful thing in the module. Budget an evening for §2–§4, a second for §5 and Exercise C.
Exercises A and B are ~10 minutes of compute together and produce numbers you will reuse all year.

This module builds directly on `curriculum/measuring-memory.md` (which priced a single probe) and
`curriculum/the-training-loop.md` (which owns the loop, determinism, and the 6·N·D arithmetic). It
does not re-teach either. It mirrors `research/notes/pretraining-recipes.md` §9 and does not
contradict it; where I extend that schema I say so and show the working.

---

## 1. What this module settles

**One:** training telemetry is a real observability pipeline and almost all of your instincts
transfer — JSONL first, dashboards later; a run manifest separate from the per-step record; a
durable sink and a curated human sink — but the cost model is inverted, because *reading* a metric
is what costs, not writing it, and the cost is paid in throughput rather than in bytes. **Two:**
the three breaks are (a) evaluating a device tensor drains the pipeline, so instrumentation cost
scales with the number of series you read and not with the volume you store; (b) there is no
request id — causality runs through a scalar mean over half a million tokens and through an
optimizer state with a known exponential memory, so the two questions "which example did this" and
"when did this start" both require statistics rather than a trace; and (c) the counterfactual is a
second run you must pay for, not the other half of the fleet. **Three:** the storage bill is
irrelevant at our scale — `[M]` a full-fidelity, every-step, every-tensor record of a 353M-parameter
100k-step run is **15.80 GiB, about four optimizer checkpoints** — which means the only defensible
reason to log less is the sync, and the sync is a solved engineering problem you should solve once
and never think about again.

**Findings this module's own exercises produced, folded back in.**

- `[M]` **Draining metrics one `.item()` at a time costs ~38 µs per series, flat, regardless of
  how many you have** — so cost is linear in cardinality. Collating them into one `torch.stack(...)
  .to("cpu")` costs ~1.29 µs per series marginal. At the 2,796 series a 353M model produces under
  per-tensor monitoring that is **106 ms versus 3.6 ms per drain, a 29× gap**, reproduced in three
  fresh processes. The break-even is at about **two** metrics.
- `[M]` **The telemetry surface of a model is a static, computable property of its module graph**:
  `7 + 13L` modules and `3 + 6L` parameter tensors for a nanoGPT-shaped decoder, giving 744 / 516 /
  2,796 series per step for the three configs in §3.6. You can size your pipeline before you write
  a line of it.
- `[M]` **The dilution bound is real and large.** In a 32-document batch with exactly one document
  replaced by uniform noise, the batch-mean loss moves **0.91σ** — the 6σ rule does not fire — while
  the per-document maximum moves **24.3σ** and fires immediately. **A 26.7× difference in
  detectability, from one extra `.max()`.** Exercise C, §3.4. **One run of this arm**, so an anecdote
  by the house standard; the effect is two orders of magnitude and matches a closed-form prediction,
  which is why it is worth stating, but three seeds would make it evidence.
- `[M]` **A prediction in this module's own §3.3 was falsified by its own exercise.** I predicted the
  `update_to_param_ratio` perturbation would decay with Adam's `v` half-life of 13.5 steps at
  β₂ = 0.95. Measured: the deviation is only 3.3–4.0% of baseline and halves in **3** steps. The
  reason is that the observable is `|m̂|/√v̂`, and `m` (half-life 6.6) and `v` (half-life 13.5) are
  kicked in *opposite* directions by the same event, so the signature is a difference of two
  exponentials, not one. §3.3 is rewritten around the measurement.
- `[A]` **On a single device, OLMo-core's own batching optimisation does not engage.** Read
  `trainer.py:548` with `trainer.py:366`: `async_bookkeeping` is only ever set inside an
  `is_distributed()` guard, so it stays `None`, `bookkeeping_device` falls through to the GPU,
  `move_metrics` becomes a no-op, and `reduce_metrics` takes the `not is_distributed()` branch at
  `utils.py:216` — one `.item()` per metric. Combined with the measurement above, that is the 106 ms
  number, on our machine, out of the box. High confidence from the code; **not yet measured
  end-to-end**, and §8 names the test.

---

## 2. Theory in plain language

### 2.1 The pipeline you already own

Strip the ML vocabulary and a training run is a single long-lived process emitting a fixed-schema
event per unit of work. You have built this. The mapping is close to exact:

| Your world | Training run | Notes |
|---|---|---|
| Service instance | one training process | exactly one, for us — see §4.5 |
| Request | one optimizer step | the unit of work and the only time axis |
| Structured log line | one JSONL record per step | same file format, same reason |
| Metric | a named scalar per step | but see §2.4 — no label dimension |
| Resource attributes (OTel) | the run manifest: git sha, config hash, seed, wheel versions | emitted once, not per step |
| Sink fan-out | callbacks: console, W&B, TensorBoard, disk | `console_logger.py:33` is a glob allow-list |
| Alerting rule | spike detector: 6σ over a 128-step window | `stability_monitor.py:29`,`:35` |
| Autoscaler acting on a metric | skip-step optimizer, same 6σ/128 rule | `skip_step_optimizer.py:37`,`:38` |
| Retention / downsampling | drain interval, sampling interval | two *different* knobs — §2.5 |
| Post-mortem from logs | diagnosing a dead run three days later | the actual design requirement |

The requirement that generates the schema is the one you already use: **someone must diagnose a
failed run from this file alone, three days later, with no console and no dashboard.** That is
`research/notes/pretraining-recipes.md` §9's framing and it is correct. Everything in §2.3 falls
out of it.

### 2.2 What this replaced

The lineage is short and unflattering. `print(loss)` in the training loop (still the reference
implementation — `nanogpt/train.py:327`), then TensorBoard event files (binary, protobuf,
append-only, awkward to grep, tied to a viewer), then hosted experiment trackers (W&B, Comet —
`olmo-core` ships callbacks for both), and now, in every serious trainer, a plain structured record
on local disk *plus* whichever tracker the org pays for.

The reason JSONL won the durable tier is the reason it won for you: a tracker is a network
dependency inside a process that must not die, its query language is not `jq`, and its retention is
someone else's policy. `olmo-core`'s `MetricSaverCallback` (`metric_saver.py:85`) writes JSON to
the run's save folder for exactly this reason, and torchtitan's `LoggerContainer`
(`metrics.py:177`) exists so the tracker is one sink among several rather than the sink.

**This bridge does not break.** That is worth saying plainly, because a curriculum that finds a
clever inversion in every section is being cute rather than accurate. JSONL-first, one record per
unit of work, provenance in the record, sinks fanned out behind an interface, human-facing view is
a *filtered view* — all of it transfers unchanged. Spend your scepticism on §2.6–§2.8.

### 2.3 What to emit per step

The canonical schema for this lab is `research/notes/pretraining-recipes.md` §9. Read it there;
I will not duplicate it. What that note states as a list, I want you to be able to *derive*, so
here is the generating rule. Every field is in the record because it answers one of five questions:

1. **Which run is this?** `run_id`, `git_sha`, `config_sha`, `seed`, `stage`, and the version
   quartet `torch_version`/`rocm_version`/`driver_version`/`gfx_arch`. These are *manifest* fields —
   see §4.2 for why they must also appear per-record despite being constant.
2. **Where in the run am I?** `step`, `tokens_seen`, `epoch`, `data_cursor_tokens`, `wallclock_s`.
   Note `tokens_seen` rather than `batches`: batch size can change mid-run (`batch_size_scheduler.py`
   exists), so batches are not a stable clock and tokens are.
3. **What was the input?** `lr`, `wd`, `batch_tokens`, `microbatches`, `seq_len`, `mixture_weights`.
   Everything the step was *told* to do. If you cannot reconstruct the step's inputs from the
   record, you cannot re-run it.
4. **What did the model do?** `loss_train`, `z_loss`, `aux_loss`, `grad_norm_preclip`,
   `grad_norm_postclip`, `clip_fraction`, `skipped_step`, `update_to_param_ratio_*`,
   `attn_max_logit`, `param_norm`.
5. **What did the machine do?** `tokens_per_s`, `tflops_est`, `step_time_s`, `step_time_p99`,
   `mem_alloc_gib`, `mem_reserved_gib`, `mem_max_alloc_gib`, `largest_single_tensor_gib`.

**Two fields I would add to that schema, and one warning.**

*Add `loss_doc_max` and `loss_doc_std`* — the maximum and spread of per-example loss within the
batch, not just the mean. §3.4 proves that the mean **cannot** detect a single bad document at
realistic batch sizes and the max detects it at ~46σ. Both are one reduction over a tensor you
already have, and under the buffered-drain discipline of §2.6 they cost nothing.

*Add `hipblaslt_configured` and `attention_backend` to the manifest.* `[M]`
`ASSUMPTIONS.md → hipblaslt-config`: configuring hipBLASLt changes the relative error of a
length-1M bf16 reduction from 5.60e-3 to 2.01e-3, a factor of 2.8. `[M]`
`ASSUMPTIONS.md → sdpa-is-memory-efficient`: `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` switches
which attention kernel actually runs, and `torch.backends.cuda.flash_sdp_enabled()` returns `True`
either way. Neither is a metric. Both change the arithmetic. A run record without them is not
interpretable on this machine.

*The warning:* `largest_single_tensor_gib` is in that schema for a specific local reason — `[M]`
`ASSUMPTIONS.md → large-tensor-fault-32gib`, single tensors ≥32 GiB hang at 0% CPU with no error.
An allocation-size watermark is the only telemetry that will ever warn you before that fault, and
the fault does not raise, does not log, and does not increment anything. It is the one field in the
schema whose job is to catch a failure mode that is invisible to every other field.

### 2.4 Cardinality — the analogy mostly holds, and the axis that replaces it

In Prometheus, series count is `|metric names| × Π |label values|`, and the operational hazard is a
label whose domain you do not control: put `user_id` in a label and the TSDB falls over. Your whole
discipline here is about *bounding an unbounded runtime dimension*.

Training telemetry has **no label dimension at all**. There is one time axis (`step`) and a flat
namespace of metric names. So cardinality is `|metric names|`, full stop — and metric names come
from the model's module graph, which is fixed at config time.

`[M]` **You can therefore compute your series count exactly, before the run.** For a nanoGPT-shaped
decoder with `bias=False` and weight tying, counting `named_modules()` and `named_parameters()`:

```
modules          = 7 + 13·L
parameter tensors = 3 + 6·L
```

(The 7 is `transformer`, `wte`, `wpe`, `drop`, `h`, `ln_f`, `lm_head`; the 13 per block is the Block
itself plus two LayerNorms, the attention module and its two Linears and two Dropouts, and the MLP
with its two Linears, GELU and Dropout. The 3 is `wte`, `wpe`, `ln_f`; weight tying at
`nanogpt/model.py:138` means `lm_head.weight` is the *same tensor* as `wte.weight`, and
`named_parameters()` de-duplicates by default, so it is not counted twice.)

Under OLMo-core's per-tensor monitoring — `gap_monitor.py:183` `record_tensor_stats` emits max,
mean and var for each of four kinds (`param`, `grad`, `activation`, `activation_grad`) — the upper
bound is

```
series = 3 · 2 · (3 + 6L)   +   3 · 2 · (7 + 13L)
         └── params, grads ┘     └── activations, their grads ┘
```

`[M]` measured by instantiating the models (Exercise B, deterministic, cross-checked against the
closed form above):

| Config | L | params | param tensors | modules | series/step |
|---|---|---|---|---|---|
| nanoGPT `shakespeare_char` | 6 | 10,646,784 | 39 | 85 | **744** |
| nanoGPT CPU fallback | 4 | 795,904 | 27 | 59 | **516** |
| a 353.55M ablation target | 24 | 353,551,360 | 147 | 319 | **2,796** |

**Where the analogy breaks, and it breaks in your favour for once.** Cardinality here is static,
knowable, and small. Three thousand series would be a conversation in your Prometheus fleet; here
it is an afternoon. The unbounded axis is not labels — it is **resolution**: series × steps, and
steps run to 10⁵–10⁶.

**And where it breaks against you.** In production you downsample old data because the raw points
have served their purpose and the aggregate is still queryable. Here, a step you did not record is
gone, and there is no aggregate to fall back on, because the interesting object is often a single
step. §2.5 is the escape hatch — and it is a strange one.

### 2.5 Sampling — you cannot tail-sample, but you can recompute

Trace sampling has two idioms. **Head sampling**: decide at ingress, keep 1%, cheap, and you lose
the rare interesting request. **Tail sampling**: buffer the whole trace, decide at egress once you
know it was slow or errored, expensive, and you keep exactly the interesting ones.
vLLM's KV metrics collector head-samples at 1% (`kv_cache_metrics.py:49`), which
`curriculum/measuring-memory.md` §5.1 already dissects.

**Tail sampling is structurally unavailable in training.** To decide that step 11,940 was
interesting you need to observe a consequence, and the consequence — a loss spike, a diverged run —
arrives tens to hundreds of steps later (§3.3 makes that lag exact). By then the step's activations
are freed, its gradients are zeroed, and its batch has scrolled past. There is no buffer to
retroactively promote.

What you have instead is something no production system has: **the run is deterministic, so the
step is recomputable.** Given `(checkpoint, seed, data cursor, config, code sha)` you can
re-materialise step 11,940 exactly and instrument it as heavily as you like. That inverts the
economics completely — telemetry becomes a *materialized view* over a recomputable source rather
than a primary record. §4.4 develops this, including the large caveat that on our machine
bit-reproducibility is **unproven**.

Meanwhile there are two sampling knobs, and conflating them is the single most common mistake:

- **Sampling interval** — how often you *compute* a statistic. `gap_monitor.py:43`,
  `interval: int = 1`, gated inside the recorder at `gap_monitor.py:189`. This reduces both compute
  and storage, and it loses information.
- **Drain interval** — how often you *read* buffered statistics back to the host.
  `trainer.py:200`, `metrics_collect_interval: int = 5`, gated at `trainer.py:1514`. This reduces
  *only* the sync cost. It loses nothing: OLMo-core's own docstring (`trainer.py:205`–`212`) is
  explicit that callbacks still receive every step's metrics, merely delayed.

**The drain interval controls throughput. The sampling interval controls storage and compute. They
are orthogonal, and the drain interval is nearly free to make small.** Almost everybody sets them
as if they were the same knob.

### 2.6 Break one — the read stalls the pipeline, and the cost is per series

`curriculum/measuring-memory.md` §2.4 establishes the mechanism and prices one probe: `[M]` reading
a scalar off the device costs a median **236×** what reading the same scalar off a host tensor
costs, ~55 µs against 0.25 µs on a quiet machine. I am not going to re-derive that. The extension
this module owns is what happens when you have more than one metric.

The naive drain is a loop:

```python
out = {k: v.item() for k, v in buffer.items()}      # olmo-core utils.py:216
```

Every iteration is an independent device-to-host round trip. The first one waits for the queue to
retire — work you owed anyway. Every subsequent one waits for nothing at all and still pays full
latency. `[M]` Exercise A, three fresh processes, `torch 2.12.0a0+rocm7.13.0a20260313`, gfx1151,
native Windows, fp32 scalars:

| Series `M` | per-`.item()` drain | collated drain | ratio | µs per series (per-item) |
|---|---|---|---|---|
| 1 | 37.9 µs | 41.0 µs | **0.9×** | 37.9 |
| 8 | 307 µs | 60 µs | 5.1× | 38.4 |
| 64 | 2,283 µs | 135 µs | 16.9× | 35.7 |
| 512 | 19,183 µs | 772 µs | 24.9× | 37.5 |
| 2,796 | 106,041 µs | 3,645 µs | **29.1×** | 37.9 |

(Medians of 7 interleaved trials; the two paths run back-to-back inside one timed block so
machine-level drift hits both. Numbers shown are from process 1; processes 2 and 3 gave 30.4× and
29.9× at M = 2,796. Taking the median across all three processes at each `M`, the per-series figure
sits in **35.7–39.9 µs across a 2,796× range in M** — essentially flat, with a mild upward drift
(35.7 µs at M = 64, 39.9 µs at M = 2,796) that is plausibly Python dict-iteration overhead rather
than device cost. That flatness is the whole finding.)

The collated drain is what `move_metrics` does when the target device differs
(`training/olmo-core/src/olmo_core/train/utils.py:126`–`138`): stack every buffered scalar into one
tensor and issue **one**
transfer.

```python
stacked = torch.stack(list(buffer.values())).to("cpu")   # one sync, one copy
out = dict(zip(buffer.keys(), stacked.tolist()))         # host reads, ~0.25 µs each
```

Three things to take from the table.

**It is a marginal-cost story, not a fixed-cost story.** Fit the collated path: `c₀ ≈ 40 µs`
fixed, `c₁ ≈ 1.29 µs` per series. The per-item path is `c₁ ≈ 38 µs` per series with no meaningful
fixed cost. Collating buys ~29× on the *marginal* cost; it does not make telemetry free. At extreme
cardinality the `torch.stack` itself becomes visible (it is a concatenation over M inputs), which
is why the collated column is slightly superlinear — 772 µs measured at M = 512 against 700 µs
predicted by the two-parameter fit.

**The break-even is two metrics.** Solve `c₀ + M·c₁_col = M·c₁_item` → `40 = M(38 − 1.29)` →
`M ≈ 1.1`. Measured: at M = 1 the collated path is *slower* (0.8–0.9× across three processes); at
M = 8 it is already 4.4–6.0× faster. An optimisation with a break-even at M = 2 is one you apply
unconditionally.

**And the local sting.** `[A]` high confidence, from reading the code: **on a single device,
OLMo-core takes the slow path.** `trainer.py:366` only sets `async_bookkeeping` inside
`if ... is_distributed():`; it therefore remains `None`; `trainer.py:548` tests
`if self.async_bookkeeping and backend_supports_cpu()` and `None` is falsy, so `bookkeeping_device`
returns the *GPU*; `move_metrics` then filters on `m.device.type != device.type`
(`utils.py:132`) and finds nothing to move; and `reduce_metrics` falls into the
`not is_distributed()` branch and calls `.item()` per metric (`utils.py:216`). The collation exists
and is correct — it simply never engages for us. `run_bookkeeping_op` likewise falls through to the
synchronous `else: wrapped_op(...)` at `trainer.py:1352`, so the background bookkeeping thread does
not run either. This is not a bug in OLMo-core; it is a design tuned for the multi-rank case, met
by a machine that has one rank. **It is exactly the class of thing that makes a borrowed trainer's
performance numbers non-transferable, and you would only ever find it by reading.** The cheapest
test is in §8.

### 2.7 Break two — there is no request id

`curriculum/measuring-memory.md` §2.2 makes this argument for *inference*: causality runs through a
continuous attention distribution, attention weights are not effects, and they do not compose
across layers. The training-side version is different in mechanism and identical in consequence,
and it has two halves.

**Half one: the loss is a mean, and a mean is a lossy aggregation with a computable dilution
factor.** The scalar you log is `(1/N) Σ ℓ_i` over `N = batch_tokens`, which at our intended scale
is 10⁴–10⁶ tokens. One pathological document contributes its share and nothing more. §3.4 turns
this into a hard bound: at 512 documents per batch, **no single document can move the mean loss by
even 1σ, because per-token surprise is capped at `ln V`.** There is no `request_id` to group by,
because the grouping was destroyed by the reduction before you ever saw the number.

**Half two: the optimizer is a low-pass filter, so the *time* of a cause is smeared too.** Adam's
second moment is an exponential moving average with decay `β₂`. A single bad gradient at step `s`
perturbs it at step `t` by a factor that decays as `β₂^(t−s)`. §3.3 gives the half-life exactly:
**13.5 steps at β₂ = 0.95, 693 steps at β₂ = 0.999.** So even when you correctly identify that
something went wrong, "when" is an interval, not a timestamp — and the width of that interval is a
hyperparameter you chose. (§3.3 also shows, from measurement, that the *observable* proxy for this
decays much faster than the state does, which makes the problem worse rather than better.)

> **Systems bridge, and this is the one to teach back.** You have a span tree. Spans are disjoint,
> they nest, their durations sum, and every span carries the request id, so "62% of p99 is in the
> auth call" is a *fact about that request*. Here you have neither property. The batch mean destroys
> the grouping (no id), and the optimizer EMA destroys the timestamp (no clean start). What you get
> instead is a signal-processing problem: a known impulse response, a known dilution factor, and a
> noise floor.
>
> **Where the bridge breaks is also where it helps.** Because the smearing is an *EMA with a known
> constant*, it is invertible in principle in a way a lost trace never is. You know the kernel. The
> reason nobody does this is that the input is not observable, only the output — but it does mean
> the right instrument is a matched filter, not a threshold, and §8 lists that as open. `[M]` §3.3
> also shows the kernel is not as clean as this paragraph implies: two EMAs with different constants
> and opposite signs, and which one dominates depends on the fault. Keep the ambition; discount the
> claim.

### 2.8 Break three — the counterfactual is a run, not the other half of the fleet

`curriculum/measuring-memory.md` §2.3 states this for memory policies: the control arm is the same
model on the same prompt with the **full cache**, which is exactly the configuration the policy
exists to avoid, and you must pay for it on every probe. The training analogue is blunter. "Did
this change help?" requires a second training run with everything else held fixed, and the seed
alone moves the answer: `[C]` (2406.10229) measures how large seed variance already is at small
scale.

Three consequences you should carry into every Track D decision.

**The oracle and the cost measurement cannot come from the same run.** Restating
`measuring-memory` §2.3 in Track D terms: a run instrumented at full GAP cardinality has different
memory behaviour, therefore possibly a different achievable microbatch size, therefore a different
arithmetic intensity, therefore a different throughput. So the throughput number from your
heavily-instrumented run is not the throughput of the uninstrumented one. Hold microbatch size
fixed across arms, and say in the write-up that you did.

**Your minimum detectable effect must be computed before the run, not after.**
`measuring-memory.md` §3.4 gives the arithmetic; Track D's version is that the relevant `σ` is
seed-to-seed variance in the final loss, which you must measure once and then reuse. Exercise C's
harness is the smallest thing that can produce it.

**And a control you actually can afford.** A 20M–300M run at our scale is hours, not weeks. The
matched-control discipline that is unaffordable at frontier scale is routine here. That is the same
inversion `research/synthesis.md` argues for the memory track — small scale is the enabling
condition for the expensive counterfactual — applied to training systems.

### 2.9 Post-hoc diagnosis: what "diagnose it from the file" actually means

Concretely, three days later, with a dead run and a JSONL, you should be able to answer:

1. **Did it diverge, stall, or finish?** `loss_train` tail plus `wallclock_s` gaps. A stall shows as
   a `wallclock_s` gap with no missing steps — which on this machine is the signature of
   `large-tensor-fault-32gib`, a hang at 0% CPU with no exception.
2. **Was it the data or the optimizer?** `loss_doc_max` moves with a bad shard; `grad_norm_preclip`
   and `update_to_param_ratio` move with an optimizer or LR problem. §3.4 is why you need the first
   one at all — and `[M]` Exercise C found that on a single-bad-document fault the gradient norm
   moved *in the wrong direction* (z = −1.08), so "watch the grad norm for bad data" is folklore
   that fails on the first test.
3. **When did it start?** Not when the loss moved — when the *earliest* indicator moved. §3.3 says
   the optimizer-state indicator has a much longer memory than the loss, so it is the one that
   preserves the event.
4. **Was it real or was it noise?** Requires a null: either a clean control run or the seed-variance
   number from §2.8. Without one you are eyeballing a line chart, which you would not accept from an
   SRE and should not accept from yourself.
5. **Was the machine configured the same way?** Manifest fields. On this box that specifically means
   hipBLASLt and the attention backend (§2.3), because both change the arithmetic.

An observation worth internalising: **item 4 is the one that is always missing.** Every trainer in
`research/reference/training/` emits enough to answer 1, 2, 3 and 5. None of them emits anything
that answers 4, because the null is not a property of the run.

---

## 3. The math that actually matters

### 3.1 Symbols

| Symbol | Reads as |
|---|---|
| `M` | number of distinct metric series emitted per step (cardinality) |
| `L` | number of transformer layers |
| `c₀` | fixed cost of one drain — one host-device synchronisation and transfer setup |
| `c₁` | marginal cost of one additional series in a drain |
| `k` | drain interval, in steps |
| `t` | wall-clock duration of one optimizer step |
| `β₂` | Adam's second-moment decay coefficient (typically 0.95 or 0.999) |
| `g_t` | gradient at step `t` |
| `v_t` | Adam's second-moment estimate at step `t` — an EMA of `g_t²` |
| `D` | number of independent examples (documents) in a batch |
| `ℓ_d` | mean loss of document `d`, in nats |
| `σ_doc` | standard deviation of per-document loss across a batch |
| `Δℓ` | excess loss of one anomalous document over a typical one |
| `V` | vocabulary size; `ln V` is the maximum possible surprise of one token |
| `n` | window length of the spike detector (128 in both reference implementations) |
| `δ` | per-step drift of a metric that is trending |
| `z` | detection threshold in standard deviations |

### 3.2 The drain equation, with cardinality

`measuring-memory.md` §3.5 gives the single-probe version, `overhead = c/(k·t)`. Generalise to `M`
series:

```
overhead = (c₀ + M·c₁) / (k · t)
```

- Numerator: the cost of one drain of `M` series.
- Denominator: the useful work done between drains.

`[M]` Exercise A, on our machine, fp32 scalars:

| Path | `c₀` | `c₁` |
|---|---|---|
| per-`.item()` (`utils.py:216`) | ~0 | **37.9 µs** |
| collated (`utils.py:138`) | **~40 µs** | **1.29 µs** |

Solve for the drain interval that keeps observability under 1% of throughput,
`k ≥ (c₀ + M·c₁)/(0.01·t)`. Take two step times bracketing our scale — `[A]` derived from
`6·N·D` at an assumed 20% MFU against the `[M]` 20.9 TFLOP/s bf16 GEMM ceiling, so treat them as
order-of-magnitude:

| `M` | drain cost, per-item | drain cost, collated | `k` for 1% at `t` = 250 ms | `k` for 1% at `t` = 2 s |
|---|---|---|---|---|
| 744 | 28.2 ms | 1.0 ms | 12 / 1 | 2 / 1 |
| 2,796 | 106.0 ms | 3.6 ms | **43 / 2** | **6 / 1** |

(Each cell is *per-item / collated*.)

**Now connect that to §3.3, because this is the point of the whole section.** At M = 2,796 on the
per-item path with a 250 ms step, holding observability to 1% forces a drain interval of 43 steps.
The half-life of an Adam-`v` perturbation at β₂ = 0.95 is 13.5 steps, and `[M]` the half-life of the
thing you can actually *log* is **3** steps (§3.3). **You would be sampling at roughly one
fourteenth of the rate needed to see the phenomenon you are trying to attribute** — the transient is
long gone before your second sample. The collated path puts you at `k = 2`, which resolves the
3-step recovery marginally and the 13.5-step one comfortably.

Note that the drain interval does not throw away the *record* — OLMo-core still passes every step's
metrics to callbacks (`trainer.py:205`–`212`). What a coarse `k` costs you is the *option* to react,
and the freshness of anything a callback computes online. But if you are drain-limited to `k = 43`
you will almost certainly also have reduced the sampling interval to match, and then the record is
gone too. The two knobs stay separate only if you keep `k` small, which is the argument for
collating.

The design rule that follows, and it is the whole module in one line:

```
k ≤ (half-life of the fastest state you intend to attribute) / 4
```

and then choose the drain implementation that makes that `k` affordable, rather than choosing `k` to
make a bad drain implementation affordable.

**One honest caveat on the measurement.** The 37.9 µs was measured with an *empty* queue. In a live
step the first `.item()` additionally waits for the step's kernels to retire — but that is work you
owed regardless, so it belongs to the step, not to the probe. The marginal figure is the right one
for this arithmetic, and it is also why the drain must sit *after* the optimizer step rather than in
the middle of it.

### 3.3 The optimizer as a low-pass filter, and the half-life of a mistake

AdamW's second moment `[C]` (1412.6980; decoupled decay `[C]` 1711.05101):

```
v_t = β₂ · v_{t−1} + (1 − β₂) · g_t²
```

- `v_t` — running estimate of the squared gradient magnitude, per parameter.
- `β₂` — how much of the old estimate survives each step. Typically 0.95 (nanoGPT, and
  `research/notes/pretraining-recipes.md` §9's recipe) or 0.999 (PyTorch's default).
- `g_t²` — this step's squared gradient, elementwise.

Unroll it. The contribution of a single step `s` to `v_t` for `t > s` is

```
(1 − β₂) · β₂^(t−s) · g_s²
```

which is an exponential decay in the number of steps since `s`. Two time constants fall out.

**e-folding time** (contribution falls to 1/e):

```
τ = 1 / ln(1/β₂)
```

**Half-life** (contribution falls to one half):

```
h = ln 2 / ln(1/β₂)
```

| `β₂` | τ (steps) | h (steps) | ~4 half-lives |
|---|---|---|---|
| 0.9 (β₁ default) | 9.49 | **6.58** | 26 |
| **0.95** | 19.50 | **13.51** | 54 |
| 0.99 | 99.50 | 68.97 | 276 |
| **0.999** | 999.50 | **692.80** | 2,771 |

Read the table twice. Once as a *forensics* statement: at β₂ = 0.999, a bad batch is still visibly
distorting your updates **two thousand seven hundred steps later**, so a loss anomaly you notice at
step 15,000 may have originated anywhere in a two-thousand-step window. And once as a *sampling*
statement: your drain interval must satisfy `k ≤ h/4` if you want to resolve the decay at all,
giving `k ≤ 3` at β₂ = 0.95 and `k ≤ 173` at β₂ = 0.999.

**OLMo-core's default of `metrics_collect_interval = 5` (`trainer.py:200`) is therefore slightly too
coarse for a β₂ = 0.95 recipe and enormously conservative for a β₂ = 0.999 one.** That is not a
criticism of the default; it is a demonstration that the correct value depends on a hyperparameter
in a completely different config section, which is precisely the kind of coupling that a "sensible
default" cannot express. Set it from `β₂`.

**What you actually observe — and where I got this wrong.** `v` is not directly logged. Its effect
reaches you through the Adam update, `−lr · m̂ / (√v̂ + ε)`, so the natural observable is the
update-to-parameter ratio,

```
r_t = ‖θ_t − θ_{t−1}‖₂ / ‖θ_t‖₂
```

- Numerator: how far the weights moved this step.
- Denominator: how big the weights are, which makes `r` dimensionless and comparable across widths.
  (This is also the muP sanity check `[C]` 2203.03466 — under muP `r` should be roughly
  scale-invariant, which is why `research/notes/pretraining-recipes.md` §9 already logs its median
  and max.)

I predicted that after a gradient spike `r` would dip and recover with `h = 13.51` steps, giving a
much longer-lived signature than the one-step loss excursion. **That prediction is wrong, and
Exercise C falsified it.** `[M]` with the whole batch replaced by uniform noise at step 200, the
deviation in `r` at step 201 is **3.3–4.0% of baseline and is halved by step 203** — three steps,
not thirteen.

The mechanism is that `r` is not a function of `v` alone. It is `|m̂| / √v̂`, and the same bad
gradient kicks both moments:

```
m gets + (1 − β₁)·g_bad          numerator ↑    half-life ln2/ln(1/0.9)  = 6.58 steps
v gets + (1 − β₂)·g_bad²         denominator ↑  half-life ln2/ln(1/0.95) = 13.51 steps
```

The two effects have **opposite signs on `r` and different time constants**. The observable is
therefore a difference of two exponentials, dominated early by the faster one, which both shrinks
the peak deviation and shortens the apparent recovery. The correct statement is:

> `v` has a half-life of `ln2/ln(1/β₂)`. **The thing you can log does not.**

Two things survive from the original argument, and they are the ones that matter. First, the
*existence* of a multi-step memory is real: `r` still takes several steps to return, whereas the
loss returns in one. Second, and more importantly, **the forensic window is still governed by β₂**
in the sense that the perturbation to `v` is genuinely there for ~4·13.5 ≈ 54 steps — it is simply
partly cancelled in the ratio. If you want to *see* it, log `√v̂` directly (one norm over the
optimizer state, one more device scalar) rather than inferring it from `r`. Nobody does; §8 item 3.

`[A]` medium confidence in the two-exponential explanation. It is consistent with the sign, the
magnitude and the timescale, but I did not isolate it. **Cheapest test:** re-run Exercise C with
β₁ = 0.0 (no momentum) — the cancellation disappears and the recovery should stretch toward 13.5
steps. Twenty minutes, and it would settle it.

### 3.4 The dilution bound — why the mean loss cannot attribute a bad document

Let the batch hold `D` documents of equal length, with per-document mean losses `ℓ_d`. The logged
loss is `L = (1/D) Σ_d ℓ_d`.

**Signal.** Replace one document's loss with `ℓ_typ + Δℓ`:

```
ΔL = Δℓ / D
```

**Noise.** Under the null (all documents drawn from the same distribution, independent — the usual
i.i.d.-shuffle assumption, and see the caveat below):

```
σ_L = σ_doc / √D
```

**Detectability at `z` standard deviations** requires `ΔL ≥ z · σ_L`:

```
Δℓ / D  ≥  z · σ_doc / √D
⟺  Δℓ   ≥  z · σ_doc · √D
⟺  D    ≤  ( Δℓ / (z · σ_doc) )²
```

Now bound `Δℓ` from above. Per-token cross-entropy cannot exceed the surprise of a uniform
distribution over the vocabulary, `ln V`. With `V = 50,257`, `ln V = 10.825` nats; with a typical
converged per-token loss around 2.5 nats, the largest possible excess for a maximally adversarial
document is `Δℓ_max ≈ 8.3` nats. Take `z = 3` and `σ_doc = 0.5` nats:

```
D ≤ (8.3 / (3 × 0.5))² = (5.53)² = 30.6
```

**A single document is 3σ-detectable in the batch mean only if the batch contains fewer than about
31 documents.** At `D = 512` — a modest global batch — the same maximally adversarial document
moves the mean by `8.3/512 = 0.0162` nats against a noise floor of `0.5/√512 = 0.0221` nats:
**0.73σ. Invisible, and not because it was subtle, but because it was averaged.**

**Now the max.** For `D` roughly-Gaussian documents, the expected maximum sits at

```
E[max_d ℓ_d] ≈ μ + σ_doc · √(2 ln D)
```

with standard deviation approximately `σ_doc / √(2 ln D)`. At `D = 512`, `√(2 ln 512) = 3.532`, so
the null maximum is `μ + 1.77` nats with a spread of `0.142` nats. The adversarial document sits at
`μ + 8.3`, i.e.

```
(8.3 − 1.77) / 0.142 ≈ 46σ
```

**0.73σ in the mean, 46σ in the max, from the same event, at the same cost.** `loss_doc_max` is one
`.max()` over a tensor you already computed, buffered as one more device scalar. Under §2.6's drain
discipline its marginal cost is 1.29 µs every `k` steps.

**`[M]` And it holds when you run it.** Exercise C, arm `one-doc`: a 32-document batch with exactly
one document replaced by uniform noise at step 200, measured as a z-score against the trailing
128-step window:

| Series | z at the fault | 6σ/128 fires? |
|---|---|---|
| `loss_mean` | **0.91** | **never** |
| `loss_doc_max` | **24.30** | **yes, step 200, zero lag** |
| `grad_norm_preclip` | −1.08 | never |
| `update_to_param_ratio` | 0.26 | never |

**26.7× more detectable in the max than in the mean, from the same event, on the same step.** The
mean is not merely weaker; at 0.91σ it is indistinguishable from an ordinary batch. And note the
third row: the *gradient norm went slightly down*. Folklore says watch the gradient norm for bad
data. On this fault it carried no signal at all.

`[A]` high confidence in the structure, moderate in the constants: the derivation assumes equal
document lengths, independence across documents within a batch, and approximate normality of `ℓ_d`.
Real corpora violate all three mildly (length varies, documents from the same shard correlate,
`ℓ_d` is right-skewed) — skew in particular makes the null maximum heavier-tailed and shrinks the
46σ. The measured 24.3σ at D = 32 is the right order but the constants are not tight; treat the
closed form as a design guide, not a prediction. **Cheapest test for your own corpus: log
`loss_doc_max` and `loss_doc_std` for 200 clean steps and plot the empirical distribution of the
max.** Exercise C's harness emits both fields, so the test is a by-product.

**And the ordering inverts.** `[M]` same harness, arm `all-docs`, where the *whole* batch is
corrupted: `loss_mean` z = 73.1, `loss_doc_max` z = 32.3, `grad_norm_preclip` z = 13.1,
`update_to_param_ratio` z = 5.1. The mean is now the most sensitive series by a factor of two, and
the max has lost ground because a uniformly-bad batch raises the maximum along with everything else.

> **No single series dominates. The max wins for localised faults, the mean wins for global ones,
> and which fault you have is exactly what you are trying to find out.** That is why you log both,
> and it is the training-side instance of the outcome-versus-attribution distinction that
> `curriculum/measuring-memory.md` §2.1 builds the whole memory track on.

Note also what this says about the reference implementations. torchtitan logs
`global_max_loss` (`metrics.py:499`) — but read the docstring at `metrics.py:466`–`467`: that is
`max` over **ranks**, not over documents. It is a data-parallel imbalance detector, and it inherits
the same dilution problem within each rank. It is the right statistic applied to the wrong axis.

### 3.5 The spike detector's operating characteristic

Both reference implementations use the identical rule: flag when a value exceeds the mean of the
trailing `n = 128` values by `z = 6` sample standard deviations. As telemetry it is
`stability_monitor.py:117` `_is_spike`; as a control action it is `skip_step_optimizer.py:86`
`get_step_factor`, with defaults `rolling_interval_length = 128, sigma_factor = 6`
(`skip_step_optimizer.py:37`–`38`).

**False positives, if the series were stationary and Gaussian.** `P(x > μ + 6σ) = 9.87 × 10⁻¹⁰`.
Over 100,000 steps and two monitored series, `2 × 10⁵ × 9.87 × 10⁻¹⁰ ≈ 2 × 10⁻⁴` expected false
alarms. Effectively never. Good.

**But the mean and standard deviation are estimated from `n = 128` samples**, so the threshold
itself is noisy. The standard error of a sample standard deviation is `σ/√(2(n−1))`, which at
`n = 128` is `σ/15.94 = 0.0627σ`. The 6σ threshold therefore wobbles by about `±0.38σ` (one
standard error), i.e. an effective range of roughly 5.6σ–6.4σ. Immaterial for impulses; worth
knowing before you tune the constant.

**The real problem is the trend.** A training loss is not stationary — it falls. Over a window of
`n` samples of a linear ramp with slope `δ` per step, the sample standard deviation is

```
σ̂_trend = δ · √( (n² − 1) / 12 )
```

which at `n = 128` is `36.95 · δ`. And the trailing mean lags the current value by `(n/2)·δ`, which
for a falling series means the reference point sits *above* where the series actually is, by
`64·δ`.

Work an example. nanoGPT's shakespeare_char run falls from ~4.2 to ~2.0 nats over roughly 500 steps,
so `δ ≈ 0.0044` nats/step:

```
σ̂_trend      = 36.95 × 0.0044 = 0.163 nats     (pure artefact of the trend)
threshold     = mean + 6 × 0.163 = mean + 0.98 nats
mean lag      = 64 × 0.0044     = 0.28 nats above the current value
excursion needed to fire ≈ 0.98 + 0.28 = 1.26 nats
```

**During the fast-descent phase the detector requires a 1.26-nat excursion — more than half the
total loss drop of the entire run — before it fires.** It is not broken; it is a *transient*
detector, and a trend is a legitimate reason to be insensitive. But state the consequence plainly:

> A 6σ/128 spike rule detects impulses. It is structurally blind to regressions.

And a memory-policy degradation is a regression, not an impulse. `curriculum/measuring-memory.md`
§2.6 makes the general form of this argument — an eval you have never seen fail is a decoration.
Here is the training-side instance: **the only anomaly detector shipped by our reference trainers
cannot detect the class of fault this lab exists to study.** Exercise C fires it deliberately so you
have seen it work, and then shows you the fault it misses.

**One more thing, and it is the kind of duplication you have written incident reports about.** The
same 128/6σ rule is implemented twice: once on the device as a *control* action that multiplies the
update by 0 or 1, and once on the host as a *metric*. They run on different copies of the data —
`get_step_factor` uses the local pre-reduction loss, `StabilityMonitorCallback.pre_log_metrics` sees
whatever `reduce_metrics` produced. In a distributed run they can disagree. The controller and the
monitor implementing the same predicate in two places is the setup for a very familiar postmortem.

### 3.6 The storage bill, measured, and why it is never the constraint

`[M]` Exercise B serialises a realistic record — the provenance fields plus one float per series —
and measures the encoded length:

| Config | series/step | bytes/record | 10k steps | 100k steps |
|---|---|---|---|---|
| nanoGPT `shakespeare_char` | 744 | 44,774 | 0.42 GiB | 4.17 GiB |
| nanoGPT CPU fallback | 516 | 31,094 | 0.29 GiB | 2.90 GiB |
| 353.55M ablation target | 2,796 | 169,690 | 1.58 GiB | **15.80 GiB** |

Compare against one checkpoint of the same 353.55M model: fp32 master weights plus Adam's `m` and
`v` is 12 bytes per parameter,

```
353,551,360 × 12 = 4,242,616,320 B = 3.95 GiB
```

**A complete, every-step, every-tensor telemetry record of a 100,000-step run costs about four
checkpoints.** Against our `[M]` ≥62 GiB fast tier and a machine with 128 GB of unified memory,
that is not a number anybody should be optimising.

Two conclusions, and the second is the load-bearing one.

**Emit generously.** The instinct to trim the schema comes from a cost model that does not apply
here. The `[M]` cost of one more field is 1.29 µs per drain and ~60 bytes per step.

**Therefore the *only* defensible reason to log less is the sync**, which is a fixed engineering
problem with a known 29× solution. Solve it once, in the transport, and then never litigate the
schema again. That is the opposite of the discipline your Prometheus bill taught you, and it is
correct here.

---

## 4. Why it matters for Proteus and Mnemosyne

### 4.1 One transport, two contracts

`curriculum/measuring-memory.md` §4.2 specifies Mnemosyne's telemetry contract: per eviction
decision, per layer, per head, emit `m_E` and `‖v̄_E − v̄_K‖₂` as **unevaluated device tensors**, plus
small host-side integers, drained once every `k` steps, with `attention_backend` and `dtype`
recorded once per run. This module supplies the other half — the training-side schema — and, more
importantly, the observation that **they are the same transport**.

That matters concretely for the package boundary. The transport is a buffer of named device
scalars with a drain policy. It depends on `torch` and nothing else. It therefore belongs on the
`mnemosyne → torch` side of the boundary rule (CLAUDE.md), can be used by Proteus and Themis, and
survives the clean-venv separability acceptance test. If the drain lands in Themis and Mnemosyne
calls into it, the boundary has leaked. Build it once, in the memory subsystem, with no knowledge
of what a "step" is — it takes `(name, tensor)` and a drain trigger.

`[A]` medium-high confidence that one transport suffices for both. The riskiest part is that
Mnemosyne's contract is *per layer, per head, per decision* — cardinality that scales with the
model and with the eviction rate, potentially far above the 2,796 measured here. The cheapest test
is to run Exercise A at M = 50,000 and see whether the collated path stays linear. If `torch.stack`
degrades, the fix is a preallocated device buffer written by index, which is §8's item 2.

### 4.2 The manifest and the record — and the three fields that make our results interpretable

OpenTelemetry separates *Resource* attributes (properties of the emitting entity, constant for its
lifetime) from span/metric attributes (per event). Training telemetry needs the same split, and for
the same reason: constant fields belong in one place so they cannot drift.

The twist is that you should **also denormalise the manifest into every record**. Two reasons, both
operational: JSONL files get concatenated, `jq`'d, and moved between machines, and a record that
cannot identify its own run is worthless out of context; and `[M]` at 60 bytes per field per step
(§3.6) the denormalisation is free. This is the one place where the classic normalise-your-schema
instinct is wrong, and it is wrong for a boring reason — storage is not the constraint.

Three manifest fields are load-bearing on *this* machine and appear in no upstream trainer's schema:

| Field | Why | Evidence |
|---|---|---|
| `hipblaslt_configured` | changes the relative error of a long bf16 reduction by 2.8× (2.01e-3 configured vs 5.60e-3 not) | `[M]` `ASSUMPTIONS.md → hipblaslt-config` |
| `attention_backend` | `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` selects a different kernel with different numerics, and `flash_sdp_enabled()` reports `True` either way | `[M]` `ASSUMPTIONS.md → sdpa-is-memory-efficient` |
| `mfu_denominator_flops` | see §4.3 — MFU is otherwise not comparable across machines or across trainers | `[M]` `ASSUMPTIONS.md → gemm-throughput-below-reference` |

Do **not** record `allow_bf16_reduced_precision_reduction` as an experimental axis: `[M]`
`ASSUMPTIONS.md → bf16-reduced-precision-knob-works` shows it changes the result by exactly zero
bits on this stack. It is an interface that reports a capability it does not deliver, and logging
it would imply a control we do not have. Record it in the manifest as an *environment fact* if you
like; never treat it as a knob.

### 4.3 MFU is not a metric on this machine until you patch the denominator

`curriculum/the-training-loop.md` §3.8 already establishes the arithmetic and the fix. What belongs
here is the *observability* framing and one new finding.

MFU is `achieved FLOP/s ÷ device peak FLOP/s`. The numerator is computed from your config. The
denominator is a **hardcoded table keyed on a device-name string**, and all three reference trainers
default it to the same value when the string does not match:

| Trainer | Where | Behaviour on an unknown device |
|---|---|---|
| nanoGPT | `model.py:301` | `flops_promised = 312e12`, unconditional — no device check at all |
| OLMo-core | `speed_monitor.py:111`–`113` | `else: # for other GPU types, assume A100` → `624e12 × 0.5` |
| torchtitan | `tools/utils.py:210`–`211` | logs `"Peak flops undefined for: …, fallback to A100"`, returns `312e12` |

`[M]` our measured bf16 GEMM ceiling is **20.9 TFLOP/s at 8192³**
(`ASSUMPTIONS.md → gemm-throughput-below-reference`), so all three understate MFU on this machine
by `312/20.9 = 14.93×`. torchtitan is the only one that warns, and it warns via `logger.warning`,
not into the metric record — so the *artifact* is silently wrong even though the *console* was
honest for one line at startup. That is a schema defect, not a code defect: **a derived metric whose
definition depends on an unlogged constant is not a metric, it is a coincidence.** Log
`mfu_denominator_flops` alongside `mfu` or do not log `mfu`.

**The new finding, and it strengthens an open assumption.** `ASSUMPTIONS.md → z13-is-right-instrument`
rests on `[C]` a reported 25% MFU on this silicon versus 7.7% on an RTX 4090 (ROCm #6034). Run the
consistency check that the denominator discipline demands: if that 25% used an A100 denominator, the
implied achieved throughput is `0.25 × 312 = 78` TFLOP/s — **3.7× above our measured 20.9 TFLOP/s
GEMM ceiling, which is impossible.** So the cited figure was computed against some other
denominator, and until we know which one, the 25%-vs-7.7% comparison cannot be interpreted at all —
not even as a ratio, unless both sides used the same table entry. This does not refute the
assumption; it identifies a cheap, specific question that must be answered before the assumption can
be promoted: **which peak-FLOPs figure did that campaign divide by?** That is a note-read, not an
experiment.

### 4.4 Telemetry as a materialized view — the DR bridge, and its unpaid precondition

Your durability instinct says: the log is the source of truth, so protect it. Here that is backwards
in an interesting way.

If a run is bit-reproducible, then `(code sha, config, seed, data order, checkpoint)` is the
authoritative state, and every metric is a *pure function* of it. The checkpoint plus the seed is
the write-ahead log; the JSONL is a materialized view. Losing the JSONL costs you recompute, not
information. This is genuinely unlike production, where a dropped log line is gone forever, and it
is what makes the "recompute step 11,940 with heavy instrumentation" move in §2.5 possible.

It also reframes checkpoint cadence. `curriculum/the-training-loop.md` §3.7 covers determinism and
OLMo-core's replay machinery — data order re-derived from `(seed, epoch)` alone
(`data_loader.py:667`), resume by dropping the first `batches_processed` rows
(`data_loader.py:720`), rename-granularity commit (`checkpoint.py:498`). Read those again with this
in mind: **your checkpoint interval is also your telemetry RPO.** It is the maximum number of steps
you would have to recompute to re-derive a metric you wish you had logged.

**The precondition is unpaid.** All of it depends on bit-reproducibility, and `[M]`/`[C]`
`ASSUMPTIONS.md → bf16-numerics-unproven` is `untested`, the Hardware Validation Gate has not run,
and `hipblaslt-config` shows the arithmetic itself moves by 2.8× with an environment variable. Until
the gate closes, treat the JSONL as a primary record with no backup, exactly as you would in
production. Say so in the run's notebook entry rather than assuming the nicer model.

### 4.5 What Track D can and cannot deliver on this hardware

`[C]` `ASSUMPTIONS.md → single-device-only`: distributed collectives are incomplete on gfx1151.
For this module specifically, that means the following code paths can be **read and reasoned about
but never exercised here**, and no number about them may ever be tagged `[M]` from this machine:

- Cross-rank metric reduction (`utils.py:203` `reduce_metrics`, the `is_distributed()` branch and
  everything below it), including `ReduceType` semantics and the divide-factor trick at
  `utils.py:220`.
- Metric-consistency checking across ranks (`utils.py:190` `check_metrics_consistent`) and the
  pipeline-parallel warning it emits at `trainer.py:1411`–`1418`.
- The **entire async bookkeeping design**: the separate process group (`trainer.py:375`), the
  background thread pools (`trainer.py:564`, `:576`), and the overlap of metric reduction with the
  next step. §2.6 explains why it never engages for us.
- `global_max_loss` as a rank-imbalance detector (`torchtitan/components/metrics.py:499`).
- Rank-0-only sink semantics (`metric_saver.py:49`, `console_logger`'s implicit assumption).

We can still *design* against them, and we should — an interface that assumes one rank will need
rewriting. But the honest statement in any write-up is: **the transport was validated at world
size 1.**

---

## 5. Read the code

All paths relative to `research/reference/`. Clones are gitignored; run
`scripts/fetch_reference.sh` first. Line numbers are pinned to the revisions in `PROVENANCE.md`.

Read §5.1 in order — those five files are one mechanism and the sequence is the argument.

### 5.1 The drain, end to end

| Where | What to look at, and why |
|---|---|
| `training/olmo-core/src/olmo_core/train/trainer.py:1037` | `record_metric` — the ingress. Note `value.detach().float()` at `:1064` and that the result goes into an `OrderedDict` keyed by step. Nothing is evaluated. Note also `merge_strategy` (`:1071`–`:1089`): duplicate names within a step are combined *on device* by sum/mean/max, so aggregation happens without a read. That is a pre-aggregation tier, in five branches. |
| `training/olmo-core/src/olmo_core/train/trainer.py:1514` | `if first_batch or self.global_step % self.metrics_collect_interval == 0:` — the modulo gate. Four lines of control flow are the entire difference between §3.2's two columns. |
| `training/olmo-core/src/olmo_core/train/trainer.py:1394` | `_log_metrics` — read the comment at `:1398`–`:1402` in full. It states the design in the authors' own words: the sync is unavoidable, so do it *early* and *once*, then finish in a separate thread. |
| `training/olmo-core/src/olmo_core/train/utils.py:121` | `move_metrics` — **the trick, in thirteen lines.** `:126`–`:133` builds one flat list of every buffered tensor whose device differs from the target; `:138` does `torch.stack(...)` and one `move_to_device`. N transfers become one. `:141`–`:151` then re-scatters the moved values back into the per-step dict by index. |
| `training/olmo-core/src/olmo_core/train/utils.py:137` | `with cuda_sync_debug_mode(0):` — they *suppress* the sync warning here, because this is the one sync they intend. An explicitly-declared exception to your own lint is a sign the author understood the rule. |
| `training/olmo-core/src/olmo_core/train/utils.py:203` | `reduce_metrics` — and at `:216`, `out[step][name] = value.item()`, one per metric. Read `:213` first: `if not is_distributed():` short-circuits straight to this loop. **This is the single-device path, and it is §2.6's slow one.** |
| `training/olmo-core/src/olmo_core/train/trainer.py:548` | `bookkeeping_device` — `if self.async_bookkeeping and backend_supports_cpu()`. Then read `trainer.py:366`, where `async_bookkeeping` is only ever assigned inside `if ... is_distributed():`. Trace the two together and you have the finding in §2.6. |
| `training/olmo-core/src/olmo_core/train/trainer.py:1352` | `else: wrapped_op(*args, **kwargs)` — the fallthrough. On one device the "background" bookkeeping runs inline on the training thread. |

### 5.2 The syncs designed away

| Where | What to look at, and why |
|---|---|
| `training/olmo-core/src/olmo_core/optim/skip_step_optimizer.py:86` | `get_step_factor` — **the deepest expression of this module's thesis.** Read the docstring at `:91`–`:92`: the returned tensor "can be used within the optimizer's step computation to essentially skip a step without a host-device sync." A control decision that depends on a measurement, executed as *arithmetic* (multiply the update by 0.0 or 1.0) instead of as a Python `if`. This is predication — `cmov` instead of a branch — except the misprediction penalty is a full pipeline drain across a PCIe-class boundary. |
| `training/olmo-core/src/olmo_core/optim/skip_step_optimizer.py:97` | `torch.std_mean(torch.stack(self._losses[:-1]))` — the rolling window is a list of *device* tensors, and the statistic is computed on device. Compare `stability_monitor.py:127`–`129`, which computes the identical statistic in pure Python on floats. Same rule, two implementations, two data copies (§3.5). |
| `training/olmo-core/src/olmo_core/optim/skip_step_optimizer.py:37` | `rolling_interval_length: int = 128, sigma_factor: int = 6` — the constants, matching `stability_monitor.py:29`/`:35` exactly. |
| `training/olmo-core/src/olmo_core/train/trainer.py:1517` | `torch.cuda.set_sync_debug_mode("warn")` — enabled immediately *after* the intended drain, so every other sync in the step gets caught. The tool you did not know you had; use it in Exercise A. |
| `training/olmo-core/src/olmo_core/utils.py:749` | `cuda_sync_debug_mode` — the same as a context manager, to wrap a suspect block rather than the whole run. |
| `training/nanogpt/train.py:321` | `# get loss as float. note: this is a CPU-GPU sync point` — Karpathy annotated it himself. Then read `train.py:37`: `log_interval = 1`. **A deliberate, documented, every-step sync in the reference implementation.** At `[M]` ~38 µs and nanoGPT's ~100 ms steps it is 0.04% and completely correct; at 2,796 series it would be 106 ms. Instrumentation cost is a rate, not a constant. |
| `training/nanogpt/train.py:225` | `losses[k] = loss.item()` inside `estimate_loss`, with `eval_iters = 200`. 200 syncs per eval call — ~7.6 ms of pure stall, negligible against a 200-batch eval and catastrophic in a training step. Same construct, opposite verdict, because the denominator changed. |

### 5.3 Cardinality and sampling, as shipped

| Where | What to look at, and why |
|---|---|
| `training/olmo-core/src/olmo_core/train/callbacks/gap_monitor.py:119`–`129` | `pre_train` registers a forward hook **and** a full backward pre-hook on *every* named module. This is where the cardinality in §2.4 comes from — it is a property of the module graph, and it is why `named_modules()` is the right thing to count. |
| `training/olmo-core/src/olmo_core/train/callbacks/gap_monitor.py:183` | `record_tensor_stats` — three statistics (`max`, `mean`, `var`) for four kinds. Note `:189`, `if self.step % self.interval != 0: return` — the *sampling* gate, distinct from the drain gate, and `interval` defaults to 1 (`:43`). Note also `:191` and `:197`, which skip 0-d and 1-d tensors, so the real count is below the closed-form upper bound of §2.4. |
| `training/olmo-core/src/olmo_core/train/callbacks/gap_monitor.py:211`–`231` | The `merge_strategy=sum` / divide-by-local-batch-size pattern for handling gradient accumulation. Worth reading closely: the metric is correct only because the merge happens on device and the normalisation is folded into the value before recording. Get this wrong and your activation statistics silently depend on the microbatch count. |
| `training/olmo-core/src/olmo_core/train/callbacks/console_logger.py:33` | The console sink's allow-list of glob patterns. **The two-tier sink pattern, shipped:** everything goes to the durable sink, a curated dozen patterns reach the human. Compare `metric_saver.py:27` `metrics_to_capture`, the same idea for the JSON file. |
| `training/olmo-core/src/olmo_core/train/callbacks/gpu_memory_monitor.py:48` | `torch.cuda.memory_stats(self.device)` — a **host-side** read of the caching allocator's own bookkeeping. No device sync. Not every metric costs a stall, and knowing which ones are free is half the battle. Note `:71` `reset_peak_memory_stats()`, which makes `active_bytes.all.peak` a *per-step* peak — a windowed max, and a decision about semantics hiding in a one-line call. |
| `training/olmo-core/src/olmo_core/train/callbacks/metric_saver.py:85` | `self.trainer.write_file(fname, json.dumps(metrics))` — the durable sink, in one line. Note it writes whole-file JSON snapshots rather than appended JSONL, and note `:49`, `get_rank() != 0`. Our schema (`pretraining-recipes.md` §9) chooses append-only JSONL instead; this is the closest upstream thing and the difference is deliberate. |

### 5.4 The three MFU denominators

Read all three in one sitting; the convergence is the lesson (§4.3).

| Where | What to look at, and why |
|---|---|
| `training/nanogpt/model.py:301` | `flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS` — no device check whatsoever. |
| `training/olmo-core/src/olmo_core/train/callbacks/speed_monitor.py:111`–`113` | `else: # for other GPU types, assume A100` — reached for gfx1151. Note `:99`, `dense_correction = 0.5`, which corrects for NVIDIA quoting sparsity-enabled numbers; a second layer of denominator convention that also has to be logged. |
| `training/torchtitan/torchtitan/tools/utils.py:93` | `get_peak_flops` — and marvel at `:96`, which shells out to `lspci`. On native Windows that raises `FileNotFoundError`, gets caught at `:105`, and falls back to the device name. Then `:210`–`:211`: warn and return `312e12`. |
| `training/torchtitan/torchtitan/components/metrics.py:483`–`488` | `has_quantization` → `mfu = None`. They *refuse to emit MFU* when the denominator's assumption (bf16 peak) is violated by quantization. Exactly the right instinct, applied to one violation and not to the other. |

### 5.5 The counter-examples worth reading

| Where | What to look at, and why |
|---|---|
| `training/torchtitan/torchtitan/components/metrics.py:451` | `def log(self, step, global_avg_loss: float, global_max_loss: float, grad_norm: float, ...)` — the signature takes **floats**. The sync has already happened upstream, outside the metrics component. A metrics subsystem that cannot see the sync it causes is a metrics subsystem that cannot optimise it. Contrast with `record_metric`, which takes `Union[float, torch.Tensor]`. |
| `training/torchtitan/torchtitan/components/metrics.py:309` | `ntokens_since_last_log: int` — a plain host-side integer accumulated per step and zeroed at `:534`. The cheapest possible telemetry: a counter that never touches the device. Most throughput metrics can be built this way and should be. |
| `training/torchtitan/torchtitan/components/metrics.py:177` | `class LoggerContainer(BaseLogger)` — sink fan-out behind one interface, so the tracker is one sink rather than the sink. Twenty lines. This is the piece to copy. |
| `training/olmo-core/src/olmo_core/train/callbacks/stability_monitor.py:51`–`67` | `state_dict` / `load_state_dict` — **the detector's window is checkpointed.** Resume mid-run and the spike history survives, so the detector does not re-enter its 128-step warm-up. A small thing that took someone an incident to learn, and the analogue of persisting an alerting rule's state across a restart. |
| `training/olmo-core/src/olmo_core/train/callbacks/stability_monitor.py:108`–`115` | `SpikeScore` is only emitted once the rolling window is full, and the cumulative variant only after `window_size` steps. A metric that is *absent* rather than wrong during warm-up — the right choice, and one that will break a naive dashboard. |

---

## 6. Exercises

Activate first, in PowerShell, dot-sourced so the variables survive:

```powershell
. .\scripts\activate-lab.ps1
```

**Standing caveats, from `ASSUMPTIONS.md`.** Single tensors ≥32 GiB hang the GPU silently at 0% CPU
(`large-tensor-fault-32gib`) — nothing here goes near that, but the watermark field in §2.3 exists
because of it. bf16 numerics on gfx1151 are `untested` (`bf16-numerics-unproven`), so all three
exercises use fp32. The Hardware Validation Gate has not run, so these are instrument-shakedown
runs and must be labelled as such in the notebook.

Write scratch scripts under `notebook/`. Exercise C's harness is the seed of a rig component and
acquires tests on reuse (house rule: one-off analysis scripts are exempt from TDD only until reuse).

---

### Exercise A — the drain cost is linear in cardinality; find your break-even

**Goal:** reproduce §3.2's `c₀` and `c₁` on your own machine, and locate the crossover where
collation starts to pay.

**Hardware:** one gfx1151 GPU, native Windows. **CPU fallback:** runs unchanged and is *informative
by contrast* — with no asynchronous queue there is nothing to drain, both paths cost roughly the
same, and the ratio should sit near 1.0 at every `M`. That flat line is the point: the entire break
in §2.6 is an artefact of the accelerator, and seeing it vanish on CPU is the cleanest possible
demonstration.

**Runtime:** ~4 minutes on GPU after torch imports (the import itself is slow on this stack — budget
a minute); ~2 minutes on CPU.

```python
"""How does the cost of draining N buffered metrics scale with N?

Two drain implementations, both from olmo-core:
  per_item : reduce_metrics() when not is_distributed()   -> utils.py:216
  collated : move_metrics() when the target device differs -> utils.py:138
Interleaved within each timed block so machine-level drift cancels.
"""
import json, statistics, time
import torch

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
print(json.dumps(dict(torch=torch.__version__, device=dev,
                      gpu=(torch.cuda.get_device_name(0) if dev == "cuda" else None))))

CARDS, TRIALS = [1, 8, 64, 512, 2796], 7

def make_buffer(n):                       # mirrors trainer.py:1037
    return {f"m{i}": torch.tensor(float(i), device=dev) for i in range(n)}

def drain_per_item(buf):                  # utils.py:216
    return {k: v.item() for k, v in buf.items()}

def drain_collated(buf):                  # utils.py:138
    keys = list(buf.keys())
    stacked = torch.stack([buf[k] for k in keys]).to("cpu")   # one transfer
    return dict(zip(keys, stacked.tolist()))                  # host reads, free

results = {}
for n in CARDS:
    buf = make_buffer(n)
    for _ in range(3):                                        # warm up both paths
        drain_per_item(buf); drain_collated(buf)
    if dev == "cuda":
        torch.cuda.synchronize()
    per, col = [], []
    for _ in range(TRIALS):
        t0 = time.perf_counter(); drain_per_item(buf); t1 = time.perf_counter()
        drain_collated(buf);                                  t2 = time.perf_counter()
        per.append((t1 - t0) * 1e6); col.append((t2 - t1) * 1e6)
    results[n] = (statistics.median(per), statistics.median(col))

print(f"{'N series':>9} {'per_item us':>13} {'collated us':>13} {'ratio':>8} {'us/series':>10}")
for n, (p, c) in results.items():
    print(f"{n:>9} {p:>13.1f} {c:>13.1f} {p/c:>8.1f}x {p/n:>10.2f}")
```

**Run it three times in fresh processes.** A single run of a timing benchmark is an anecdote; the
curriculum has already retracted one `[M]` that did not survive retest
(`curriculum/README.md`, the hipBLASLt segfault).

**Part 2 — the sync detector.** Add `torch.cuda.set_sync_debug_mode("warn")` (as OLMo-core does at
`trainer.py:1517`) and re-run. Then go find three constructs in code you have already written that
sync without looking like they do: `if loss > 1.0:` on a device tensor; `print(t)`; `t.tolist()`;
`torch.nonzero(mask)`; `tensor[mask]`; `len(t[t > 0])`; any `assert` on a device scalar.

**Deliverables — three numbers and one shape.**
1. `c₁` for the per-item path (µs per series). Mine: `[M]` **37.9 µs**, flat within 35.7–39.9 across
   `M` = 1…2,796, three fresh processes.
2. `c₀` and `c₁` for the collated path. Mine: `[M]` **~40 µs** and **1.29 µs**.
3. The break-even `M`. Mine: `[M]` between 1 and 8; at `M` = 1 collation is 0.8–0.9× (i.e. slower),
   at `M` = 8 it is 4.4–6.0×. The two-parameter fit predicts `M ≈ 1.1`.
4. The shape: per-series cost **flat** for per-item, **falling** for collated. If your per-item
   µs/series is not flat, you have a contended machine — re-run.

---

### Exercise B — census your telemetry surface before you build it

**Goal:** compute, exactly and in advance, how many series your instrumentation will produce and
what they will cost to store. Then verify the closed form in §2.4 against the real module graph.

**Hardware:** CPU only. No GPU required, and no GPU used even if present.
**Runtime:** ~2 minutes, essentially all of it torch's import.

```python
"""Cardinality census: how many time series does per-module telemetry create?"""
import json, sys
sys.path.insert(0, r"C:\projects\School\chiron\research\reference\training\nanogpt")
import torch
from model import GPT, GPTConfig

CONFIGS = {
    "shakespeare_char": dict(n_layer=6,  n_head=6,  n_embd=384,  block_size=256,
                             vocab_size=65,    dropout=0.0, bias=False),
    "cpu_fallback":     dict(n_layer=4,  n_head=4,  n_embd=128,  block_size=64,
                             vocab_size=65,    dropout=0.0, bias=False),
    "ablation_300m":    dict(n_layer=24, n_head=16, n_embd=1024, block_size=1024,
                             vocab_size=50304, dropout=0.0, bias=False),
}
STATS = 3     # gap_monitor.py:183 records max, mean, var

rows = []
for name, kw in CONFIGS.items():
    m = GPT(GPTConfig(**kw))
    n_p = sum(1 for _ in m.named_parameters())
    n_m = sum(1 for n, _ in m.named_modules() if n != "")
    rows.append((name, m.get_num_params(), n_p, n_m, STATS * 2 * n_p + STATS * 2 * n_m))
    del m

for name, w, n_p, n_m, series in rows:
    L = CONFIGS[name]["n_layer"]
    print(f"{name:<18} L={L:<3} params={w:>12,} p-tensors={n_p:>4} (closed form {3+6*L:>4}) "
          f"modules={n_m:>4} (closed form {7+13*L:>4}) series/step={series:>6,}")

for name, w, n_p, n_m, series in rows:                       # the storage bill
    rec = {"run_id": "abcd1234", "git_sha": "0"*40, "config_sha": "0"*64, "seed": 1337,
           "step": 123456, "tokens_seen": 12345678901, "wallclock_s": 12345.678}
    rec.update({f"gap/activations/transformer.h.{i}.attn/max": 1.2345678e-3
                for i in range(series)})
    nbytes = len(json.dumps(rec).encode()) + 1
    for label, steps in (("10k steps", 10_000), ("100k steps", 100_000)):
        print(f"{name:<18} {series:>6,} series -> {nbytes:>8,} B/rec x {steps:>7,} = "
              f"{nbytes*steps/2**30:>7.2f} GiB  [{label}]")
```

**Then do the arithmetic that makes it a decision.** For the 300M row, divide the 100k-step figure
by one optimizer checkpoint: fp32 master weights + Adam `m` + `v` = 12 bytes/param.

**Deliverables — four numbers.**
1. Series/step for the three configs. Mine: `[M]` **744 / 516 / 2,796**.
2. Whether `3 + 6L` and `7 + 13L` match the measured counts exactly. Mine: `[M]` they do —
   39/85, 27/59, 147/319. If yours differ, you changed `bias` or broke weight tying
   (`nanogpt/model.py:138`).
3. Bytes per record and total GiB. Mine: `[M]` **169,690 B** and **15.80 GiB** for 300M at 100k steps.
4. Telemetry ÷ one checkpoint. Mine: `[M]` 15.80 / 3.95 = **4.0 checkpoints**.

**The point of the exercise is the conclusion, not the numbers:** storage is not the constraint, so
stop trimming the schema and go fix the drain.

---

### Exercise C — diagnose a run you did not watch

**Goal:** the module's teach-back. Train on a source whose loss floor you know *exactly*, inject a
fault of controllable blast radius, and diagnose it *from the JSONL alone* using the reference
6σ/128 rule. Three arms give you §3.4's dilution bound, §3.5's detector characteristic, and §3.3's
optimizer memory, in one 5-minute run.

**Why a synthetic source.** The corpus is a seeded order-1 Markov chain, so its entropy rate `H` is
computable in closed form and *is* the asymptote of the training loss. That gives you a curve with a
**known** floor — one you can call healthy or unhealthy without a baseline run, which is the only
place in this whole module where the counterfactual problem of §2.8 is dodged rather than paid for.
No downloads; reproducible from the seed alone.

**The three arms.**

| Arm | Fault at step 200 | Tests |
|---|---|---|
| `one-doc` | 1 of 32 documents replaced by uniform noise | §3.4 — can the mean see one bad document? |
| `all-docs` | all 32 replaced | a maximal impulse; the detector's easy case |
| `no-clip` | all 32, gradient clipping raised to 1e9 | whether clipping was suppressing the impulse |

**Hardware:** runs on either. `[M]` on the Z13 GPU, three arms of 400 steps took **5.5 s each**;
CPU is a few minutes, most of it the Python loop that generates the corpus. **Note the incidental
`[M]`:** running nanoGPT's `model.py` on this box prints
`UserWarning: Mem Efficient attention on Current AMD GPU is still experimental` — the *only* honest
signal about which attention kernel ran, exactly as `ASSUMPTIONS.md → sdpa-is-memory-efficient`
records. Do not suppress it.

```python
"""Train on a source whose entropy you know, inject a fault, diagnose from JSONL only."""
import json, math, sys, time
from pathlib import Path
sys.path.insert(0, r"C:\projects\School\chiron\research\reference\training\nanogpt")
import numpy as np, torch
from model import GPT, GPTConfig

SEED, V, N_TOK = 1337, 32, 1_500_000
STEPS, BATCH, BLOCK, FAULT_STEP, BETA2 = 400, 32, 64, 200, 0.95
dev = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(SEED)

# ---- source with a known entropy rate --------------------------------------
P = rng.dirichlet(np.full(V, 0.3), size=V)                 # row i = P(next | cur = i)
evals, evecs = np.linalg.eig(P.T)
pi = np.real(evecs[:, np.argmin(np.abs(evals - 1.0))]); pi = pi / pi.sum()
H = float(-(pi[:, None] * P * np.log(P + 1e-300)).sum())   # nats/token -- the loss floor
toks = np.empty(N_TOK, dtype=np.uint16); cum = P.cumsum(1); u = rng.random(N_TOK); s = 0
for t in range(N_TOK):
    toks[t] = s; s = int(np.searchsorted(cum[s], u[t]))
data = torch.from_numpy(toks.astype(np.int64))
print(json.dumps(dict(device=dev, entropy_rate_nats=round(H, 5),
                      uniform_nats=round(math.log(V), 5))))

def run(arm, n_bad_docs, clip):
    torch.manual_seed(SEED)
    model = GPT(GPTConfig(n_layer=2, n_head=4, n_embd=128, block_size=BLOCK,
                          vocab_size=V, dropout=0.0, bias=False)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, BETA2), weight_decay=0.1)
    gen = torch.Generator().manual_seed(SEED)
    out = Path(f"run-{arm}.jsonl"); t0 = time.perf_counter()
    with out.open("w", encoding="utf-8") as f:
        for step in range(STEPS):
            ix = torch.randint(len(data) - BLOCK - 1, (BATCH,), generator=gen)
            x = torch.stack([data[i:i + BLOCK] for i in ix])
            y = torch.stack([data[i + 1:i + 1 + BLOCK] for i in ix])
            if step == FAULT_STEP:                          # THE FAULT
                x[:n_bad_docs] = torch.randint(V, (n_bad_docs, BLOCK), generator=gen)
                y[:n_bad_docs] = torch.randint(V, (n_bad_docs, BLOCK), generator=gen)
            x, y = x.to(dev), y.to(dev)
            logits, _ = model(x, y)
            per_tok = torch.nn.functional.cross_entropy(
                logits.view(-1, V), y.view(-1), reduction="none").view(BATCH, BLOCK)
            loss = per_tok.mean(); loss_doc = per_tok.detach().mean(dim=1)
            opt.zero_grad(set_to_none=True); loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            before = [p.detach().clone() for p in model.parameters()]
            opt.step()
            d2 = sum(((p.detach() - b) ** 2).sum() for p, b in zip(model.parameters(), before))
            p2 = sum((p.detach() ** 2).sum() for p in model.parameters())
            f.write(json.dumps(dict(
                step=step, arm=arm, wallclock_s=round(time.perf_counter() - t0, 4),
                loss_mean=float(loss.detach()), loss_doc_max=float(loss_doc.max()),
                loss_doc_std=float(loss_doc.std()), grad_norm_preclip=float(gnorm),
                update_to_param_ratio=float(d2.sqrt() / p2.sqrt()),
                clip=clip, beta2=BETA2, entropy_rate_nats=H, seed=SEED,
                fault_step=FAULT_STEP, n_bad_docs=n_bad_docs)) + "\n")
    return out

# ---- post-hoc diagnosis, from the files only --------------------------------
def spike_steps(recs, key, window=128, k=6.0):   # stability_monitor.py:117, verbatim rule
    hist, fired = [], []
    for r in recs:
        v = r[key]
        if len(hist) >= window:
            m = sum(hist) / len(hist)
            sd = math.sqrt(sum((x - m) ** 2 for x in hist) / len(hist))
            if sd > 1e-10 and v > m + k * sd:
                fired.append(r["step"])
        hist.append(v)
        if len(hist) > window: hist.pop(0)
    return fired

def zscore_at_fault(recs, key, window=128):
    hist = [r[key] for r in recs if FAULT_STEP - window <= r["step"] < FAULT_STEP]
    m = sum(hist) / len(hist)
    sd = math.sqrt(sum((x - m) ** 2 for x in hist) / len(hist))
    v = next(r[key] for r in recs if r["step"] == FAULT_STEP)
    return (v - m) / sd if sd > 1e-12 else float("nan")

for arm, nbad, clip in (("one-doc", 1, 1.0), ("all-docs", BATCH, 1.0), ("no-clip", BATCH, 1e9)):
    recs = [json.loads(l) for l in run(arm, nbad, clip).read_text().splitlines()]
    print(f"\n=== {arm}  ({nbad}/{BATCH} bad docs, clip {clip})  "
          f"final loss {recs[-1]['loss_mean']:.4f} vs H {H:.4f} ===")
    for key in ("loss_mean", "loss_doc_max", "grad_norm_preclip", "update_to_param_ratio"):
        print(f"  {key:<22} z@fault {zscore_at_fault(recs, key):>9.2f}   "
              f"6-sigma fires {spike_steps(recs, key) or 'never'}")
    base = [r["update_to_param_ratio"] for r in recs
            if FAULT_STEP - 60 <= r["step"] < FAULT_STEP]
    b = sum(base) / len(base)
    post = [(r["step"] - FAULT_STEP, abs(r["update_to_param_ratio"] - b) / b)
            for r in recs if FAULT_STEP < r["step"] <= FAULT_STEP + 80]
    half = next((d for d, x in post if x < 0.5 * post[0][1]), None)
    print(f"  update ratio: deviation at fault+1 = {post[0][1]*100:.1f}%, half by fault+{half} "
          f"(predicted Adam-v half-life {math.log(2)/math.log(1/BETA2):.2f})")
    gn = [r["grad_norm_preclip"] for r in recs if FAULT_STEP - 3 <= r["step"] <= FAULT_STEP + 1]
    print(f"  grad_norm_preclip around fault: {[round(g, 3) for g in gn]}  (clip {clip})")
```

**What I measured** `[M]`, one run per arm — single-run, so an anecdote by the house standard, but
the *ordering* is the finding and it is large. (The `all-docs` arm was run twice, in two separate
processes, and reproduced to four decimal places on final loss and exactly on the update-ratio
recovery; `one-doc` and `no-clip` were run once.)

| Series | `one-doc` z | `all-docs` z | `no-clip` z |
|---|---|---|---|
| `loss_mean` | **0.91** (never fires) | **73.08** (fires) | 72.82 (fires) |
| `loss_doc_max` | **24.30** (fires) | 32.32 (fires) | 32.35 (fires) |
| `grad_norm_preclip` | −1.08 (never) | 13.07 (fires) | 12.98 (fires) |
| `update_to_param_ratio` | 0.26 (never) | 5.07 (never) | 5.46 (never) |

Final loss 2.5304–2.5419 nats against an entropy rate of **2.49254** — converged to within 0.04
nats of the analytic floor in 400 steps, with no baseline run required to know it.

**Deliverables — five numbers and two things that should bother you.**

1. **The dilution ratio.** `loss_doc_max` z ÷ `loss_mean` z in the `one-doc` arm. Mine: `[M]`
   24.30 / 0.91 = **26.7×**. This is §3.4, measured.
2. **Detection lag.** Steps from fault to first firing, per series. Mine: `[M]` **zero** on every
   series that fires at all. The 6σ/128 rule is not slow on impulses; it is armed by step 128 and
   fires on the step itself.
3. **The observed half-life of the `update_to_param_ratio` perturbation.** Mine: `[M]`
   **3 steps at a 3.3–4.0% deviation**, against a predicted Adam-`v` half-life of **13.51**. My
   prediction was wrong; §3.3 explains why and names the test that would settle it (`β₁ = 0`).
4. **The gradient norm on the bad batch.** Mine: `[M]` **1.035 against a typical 0.46** — only
   2.3× — while the loss was 73σ out. Folklore says watch the grad norm; on this fault it was the
   *third*-best series in the global arm and *actively negative* in the localised one.
5. **Final loss minus `H`.** Mine: `[M]` **+0.038 nats**. A run materially above `H` after 400 steps
   has a capacity or LR problem, and you know it without a control.

**Thing one that should bother you: clipping made no difference.** I added the `no-clip` arm
expecting it to explain the small optimizer perturbation. It did not — `all-docs` and `no-clip`
agree to two significant figures on every series. The reason is visible in the last line of output:
the corrupted batch produced a gradient norm of 1.035 against a clip threshold of 1.0, so clipping
was barely active. **A uniform-noise batch produces a bounded gradient, not an explosive one.** If
you want to study clipping as a limiter you need a fault that actually explodes the gradient — a
learning-rate spike, or an adversarially-constructed batch. Write that down as a design note about
fault injection: *your injected fault must be in the failure mode you claim to be studying*, and
mine was not.

**Thing two: run the regression, not the impulse.** Change the LR from `1e-3` to `6e-4` at step 200
with no data fault at all. The 6σ/128 rule fires on nothing, because a permanent 40% change in
dynamics is absorbed into the trailing window within `n` steps (§3.5). Now you have personally
watched the only anomaly detector our reference trainers ship fail to notice the class of fault this
lab exists to study. That is `curriculum/measuring-memory.md` §2.6's discipline — an eval you have
never seen fail is a decoration — applied to Track D.

**Extension, if you have another twenty minutes:** run the clean configuration at three seeds and
report the standard deviation of the final loss. That number is the null for every Track D ablation
you will ever run (§2.8), and nothing downstream is interpretable without it. It is also the number
that would tell you whether the +0.038 nats above `H` is convergence or noise.

---

## 7. Self-check

Answers at the end.

1. You add 2,000 per-tensor statistics to a training run whose step time is 250 ms, and throughput
   drops by 30%. Your colleague proposes reducing the sampling interval from 1 to 10. Why is that
   the wrong knob, what is the right one, and roughly what value should it take?

2. A run diverges. The loss looks clean until step 15,000, then climbs. `β₂ = 0.999`. What is the
   width of the window in which the cause plausibly lies, and which logged field would have
   preserved the event better than the loss?

3. Your batch contains 512 documents and one of them is a corrupted 4 KB blob of base64. Estimate,
   in σ, how visible that is in `loss_train`. Then say what one extra field, costing one reduction,
   would have made it obvious — and by roughly how much.

4. Two trainers report MFU for the same recipe on two different machines: 25% and 7.7%. Under what
   condition is the ratio meaningful, and what single field in the record would let you check that
   condition?

5. You are told "OLMo-core batches its metric reads into one host-device sync every 5 steps, so
   observability is under 1%." On our machine that statement is wrong in two independent ways. Name
   both.

6. Why is a 6σ-over-128-steps spike rule nearly useless for detecting a memory-policy regression,
   and what property of the fault — not of the detector — is responsible?

---

## 8. What is still unsolved here

**1. Whether OLMo-core's drain actually degenerates on a single device, end to end.** §2.6's finding
is `[A]` from reading five files. It has not been observed in a running trainer. **Cheapest test:**
run OLMo-core's smallest config on this box with `torch.cuda.set_sync_debug_mode("warn")` enabled
for a whole step and count the warnings, or wrap `reduce_metrics` and count `.item()` calls. Half a
day, and it would either confirm a real performance trap for every single-GPU user or find that
something else engages the fast path. Either outcome is worth a notebook entry.

**2. Nobody ships the obviously-better drain.** Both reference implementations either read one
scalar at a time or `torch.stack` a list of scalars — and `torch.stack` over M inputs is itself
O(M) device work, which is why §3.2's collated column is slightly superlinear. The evident design is
a **preallocated device buffer written by index**: `record_metric` does
`buf[slot] = value` (a scatter into an existing tensor, no allocation, no stack), and the drain is
one contiguous `buf[:n].to("cpu")`. This is a ring buffer with a batched flush, which you have built
before. I can find no trainer that does it. `[A]` medium confidence it is worth 2–5× over collation
at high cardinality; unmeasured. It is also the design Mnemosyne needs if its per-head, per-decision
contract (§4.1) pushes cardinality past 10⁴.

**3. Nobody logs the optimizer state itself, and §3.3 says that is why the forensics do not work.**
The perturbation to Adam's `v` genuinely persists for ~54 steps at β₂ = 0.95, but `[M]` the
observable everyone logs — `update_to_param_ratio` — recovers in 3, because it is `|m̂|/√v̂` and the
two moments cancel. The obvious fix is to log `‖√v̂‖` (or its median and max) directly: one norm over
the optimizer state, one more device scalar, free under the collated drain. **No trainer in
`research/reference/training/` logs any statistic of the optimizer's second moment at all** —
`gap_monitor.py` covers params, grads, activations and activation-grads, and stops there. `[A]`
medium confidence this is a real gap rather than a thing tried and discarded; the cheapest test is
Exercise C with `‖√v̂‖` added and β₁ = 0, which should show the clean 13.5-step decay.

**3b. And there is no matched filter.** The impulse response is a known exponential with a known
constant, and standard signal processing says the optimal detector for a known waveform in noise is
a matched filter, not a threshold. Every trainer uses a threshold. `[A]` medium confidence a matched
filter detects smaller perturbations at the same false-alarm rate; the honest obstacle is that the
"known waveform" assumes a clean impulse, and `[M]` Exercise C already shows the real waveform is a
difference of two exponentials whose relative weight depends on how the fault projects onto the
gradient — which is not known in advance.

**4. The null distribution for a training run is not a solved object.** `measuring-memory.md` §2.7
works out the inference-side version — re-running the same config is a *degenerate* null under
determinism, so the null must come from a nuisance axis. Track D has the same problem with a worse
menu of nuisance axes: seed is the obvious one, but seed changes both initialisation and data order,
which are two different mechanisms. `[A]` we do not know how to decompose them cheaply, and
`[C]` (2406.10229) documents the magnitude of the combined effect without separating them.

**5. Attribution across the batch remains genuinely open, and it is the training-side twin of the
lab's deliverable.** §3.4 says the max recovers a bad *document*. It says nothing about a bad
*mixture weight*, a bad *tokenizer edge case*, or a slowly-degrading shard — all of which are
distributional, not impulsive, and all of which the max is as blind to as the mean.
`research/synthesis.md` argues the lab's contribution is an attribution instrument for memory; the
same gap exists one layer up, in data, and nobody has a good answer there either. It is explicitly
**not** our project — noting it so the map has an edge rather than a fade.

**6. The recomputability property in §4.4 is unproven on this machine and load-bearing if true.**
It would let us treat telemetry as a materialized view, re-derive any metric we forgot, and make
checkpoint cadence the telemetry RPO. It depends entirely on bit-reproducibility, which is
`ASSUMPTIONS.md → bf16-numerics-unproven`, `untested`. The Hardware Validation Gate's determinism
item is the test, and this module adds a reason it matters that the gate as written does not state.

**7. Contested, and left contested: whether per-tensor monitoring is worth its cost at all.**
OLMo-core ships GAP monitoring with `interval = 1` and a master switch defaulting to off; torchtitan
ships nothing equivalent. There is no published ablation showing that per-tensor statistics ever
changed a decision at small scale. `[A]` low confidence either way. Our own §3.6 arithmetic says the
storage is free and §3.2 says the sync is affordable *if collated*, so the cost objection is weaker
than folklore holds — but "cheap" is not "useful", and the useful case is unevidenced.

---

## Answers to the self-check

**1.** Sampling interval and drain interval are different knobs (§2.5). Reducing the sampling
interval to 10 throws away 90% of the *information* to fix a cost that is not caused by computing
the statistics — the statistics are device-side reductions over tensors already resident. The cost
is the read. Fix the drain: either raise `metrics_collect_interval` or, far better, collate the
drain (§2.6, 29× on marginal cost) and leave the interval small. With `M = 2,000`, per-item
`c₁ = 37.9 µs` and `t = 250 ms`, the 1% drain interval is
`k ≥ 2000 × 37.9 / (0.01 × 250,000) = 30`; collated it is
`k ≥ (40 + 2000 × 1.29)/2500 ≈ 1.1`, so `k = 2`. Collate, keep `k` small, keep every sample.

**2.** At `β₂ = 0.999` the half-life of an Adam-`v` perturbation is
`ln 2 / ln(1/0.999) = 693` steps, and four half-lives is ~2,771 steps (§3.3). So the cause
plausibly lies anywhere in roughly steps 12,000–15,000, and the loss alone cannot narrow it.
`update_to_param_ratio` is the field that preserves the event: a gradient excursion inflates `v`,
which shrinks subsequent updates, and that dip decays on the 693-step timescale rather than
vanishing in one step the way the loss excursion does. `grad_norm_preclip` is the second-best
choice — it captures the impulse itself, but only at the exact step.

**3.** Per-token cross-entropy is capped at `ln V`, so the excess of even a maximally adversarial
document over a typical one is about 8.3 nats at `V = 50,257`. Diluted over 512 documents that is
`8.3/512 = 0.0162` nats, against a noise floor of `σ_doc/√D = 0.5/22.6 = 0.0221` nats — **about
0.73σ, invisible** (§3.4). `loss_doc_max` — one `.max()` over the per-document losses you already
computed — puts the same event at roughly `(8.3 − 1.77)/0.142 ≈ 46σ` above the expected maximum of
512 clean documents. **Roughly 63× more detectable, at the cost of one reduction and 1.29 µs per
drain.** `[M]` Exercise C ran the same experiment at `D = 32` with a real model and measured
0.91σ in the mean against 24.30σ in the max — **26.7×**, right order, and the mean did not fire.

**4.** The ratio is meaningful only if both numbers used the **same denominator**, and the levels
are meaningful only if each used the *correct* one. All three reference trainers silently fall back
to A100's 312 TFLOP/s for unrecognised devices (§4.3), so two different machines can easily share a
denominator by accident and differ by design. The field that settles it is
`mfu_denominator_flops` — log the constant you divided by, or do not log MFU. And run the sanity
check: `0.25 × 312 = 78` TFLOP/s implied achieved, against our `[M]` 20.9 TFLOP/s GEMM ceiling, is
impossible — so that particular 25% did not use an A100 denominator, and the comparison is
uninterpretable until we know which one it did use.

**5.** First, the batching does not engage: `async_bookkeeping` is only assigned inside an
`is_distributed()` guard (`trainer.py:366`), so it stays `None`, `bookkeeping_device` returns the
GPU (`trainer.py:548`), `move_metrics` finds nothing to move, and `reduce_metrics` takes the
`not is_distributed()` branch and calls `.item()` per metric (`utils.py:216`). One sync per series,
not one per drain. Second, "under 1%" is a statement about a numerator, a cardinality and a step
time, none of which were given: at `M = 2,796` and `t = 250 ms`, the per-item drain is 106 ms and
`k = 5` gives **8.5%**, not 1%. Both errors point the same way, so they compound.

**6.** Because a 6σ threshold over a trailing 128-step window is a *transient* detector. A trend
inflates the window's sample standard deviation by `δ·√((n²−1)/12) = 36.95·δ` and simultaneously
drags the reference mean `64·δ` away from the current value, so a slow change never crosses the
threshold — it just moves the threshold with it (§3.5). The responsible property belongs to the
**fault, not the detector**: a memory-policy regression is a persistent shift in level, and a
persistent shift is absorbed into the baseline within one window length. Detecting it requires a
comparison against something *outside* the run — a control arm, or a source with a known floor as in
Exercise C — which is break three (§2.8) restated.

---

## Sources

**Code, at the revisions pinned in `research/reference/PROVENANCE.md`.**
OLMo-core: `train/trainer.py` (`:200`, `:366`, `:548`, `:564`, `:576`, `:1037`, `:1352`, `:1394`,
`:1514`, `:1517`), `train/utils.py` (`:121`, `:137`, `:190`, `:203`, `:216`),
`train/callbacks/{gap_monitor,console_logger,metric_saver,speed_monitor,gpu_memory_monitor,stability_monitor}.py`,
`optim/skip_step_optimizer.py`, `utils.py:749`.
nanoGPT: `model.py:138`, `model.py:301`, `train.py:37`, `:225`, `:321`.
torchtitan: `components/metrics.py` (`:177`, `:309`, `:451`, `:483`, `:499`, `:534`),
`tools/utils.py` (`:93`, `:210`).

**Papers.** `[C]` Adam, arXiv 1412.6980. `[C]` Decoupled weight decay (AdamW), arXiv 1711.05101.
`[C]` PaLM (the MFU definition every trainer cites), arXiv 2204.02311. `[C]` Tensor Programs V /
muP, arXiv 2203.03466. `[C]` "Small-scale proxies for large-scale Transformer training
instabilities" (Wortsman et al.), arXiv 2309.14322 — the result that makes stability studiable in
our box. `[C]` seed variance at small scale, arXiv 2406.10229. `[C]` evaluation power and
question-level clustering, arXiv 2411.00640.

**Our own measurements.** `ASSUMPTIONS.md` rows `gemm-throughput-below-reference`,
`hipblaslt-config`, `bf16-reduced-precision-knob-works`, `sdpa-is-memory-efficient`,
`large-tensor-fault-32gib`, `gpu-fast-tier-size`, `single-device-only`, `bf16-numerics-unproven`.
New `[M]` from this module's exercises, `torch 2.12.0a0+rocm7.13.0a20260313`, gfx1151, native
Windows, 2026-07-26.
*Exercise A*, three fresh processes, fp32 device scalars, 7 interleaved trials per point: drain
cost linear in cardinality at 37.9 µs/series (by-`M` medians 35.7–39.9 across `M` = 1…2,796);
collated drain `c₀ ≈ 40 µs`, `c₁ ≈ 1.29 µs`; ratio 29.1 / 30.4 / 29.9× at `M` = 2,796; break-even
between `M` = 1 and `M` = 8.
*Exercise B*, one deterministic process, cross-checked against the closed forms `3 + 6L` and
`7 + 13L`: param tensors / modules / series = 39 / 85 / 744, 27 / 59 / 516, 147 / 319 / 2,796;
169,690 B/record and 15.80 GiB for 353.55M at 100k steps, versus 3.95 GiB per optimizer checkpoint.
*Exercise C*, one run per arm (single-run — an anecdote by the house standard, reported for the
ordering not the level), 2-layer/4-head/d=128 model, `V` = 32, `D` = 32, `T` = 64, 400 steps,
AdamW(1e-3, β = 0.9/0.95, wd 0.1), seed 1337, Markov-source entropy rate 2.49254 nats: z-at-fault
0.91 / 24.30 / −1.08 / 0.26 (`one-doc`) and 73.08 / 32.32 / 13.07 / 5.07 (`all-docs`) for
mean/doc-max/grad-norm/update-ratio; update-ratio deviation 3.3–4.0% halving in 3 steps against a
predicted 13.51; grad norm 1.035 at the fault versus ~0.46 typical; final loss 2.5304 (+0.038 over
`H`); clipping arm showed no effect.

**Sibling modules.** `curriculum/measuring-memory.md` (§2.4 prices one probe; §2.7 the null
distribution; §3.5 the single-probe drain equation; §4.2 Mnemosyne's telemetry contract).
`curriculum/the-training-loop.md` (§3.7 determinism; §3.8 the 6·N·D arithmetic and the MFU trap).
`research/notes/pretraining-recipes.md` §9 (the JSONL schema this module derives rather than
repeats). `research/synthesis.md` (the attribution deliverable).

---

## Decision / Riskiest assumption / Next test

**Decision.** Build one telemetry transport, inside Mnemosyne, that (a) accepts named unevaluated
device tensors, (b) drains them **collated**, never per-item, (c) exposes the drain interval as a
config field defaulted from `β₂` rather than to a constant, and (d) writes append-only JSONL with
the manifest denormalised into every record, including `hipblaslt_configured`,
`attention_backend` and `mfu_denominator_flops`. Emit generously — `[M]` the storage is four
checkpoints for a full run — and add `loss_doc_max` and `loss_doc_std` to the schema in
`research/notes/pretraining-recipes.md` §9.

**Riskiest assumption.** That one transport serves both the training-step contract (M ≈ 10³, once
per step) and Mnemosyne's per-layer, per-head, per-decision contract (M possibly ≫ 10⁴, once per
decision). If `torch.stack` degrades superlinearly at high cardinality — and §3.2 already shows it
is not perfectly linear at M = 512 — the collated design does not carry across, and the two
contracts need different transports, which puts a seam in exactly the wrong place.

**Next test.** Run Exercise A at `M ∈ {10³, 10⁴, 5×10⁴}` and fit `c₁` at each. Thirty minutes. If
`c₁` stays near 1.3 µs the transport is settled; if it climbs, build the preallocated indexed buffer
from §8 item 2 *before* Mnemosyne's contract is frozen rather than after.
