---
title: Capstones — three projects specified tightly enough to build, and to fail honestly
version: 1.0.0
date: 2026-07-26
audience: the founder — 30 years distributed systems, storage, caching, DR, observability; new to ML internals
prereqs: all of Track A; Track B `attention-variants-and-kv-cost`; Track C in full for capstones one and two; Track D `training-telemetry-as-observability`, `determinism-and-reproducibility`, `checkpointing-and-resumption` for capstone three
mirrors: research/memory/open-problems-ranked.md (the ablation backlog), research/synthesis.md, docs/adr/attribution-instrument-over-eviction-policy.md
---

# Capstones

Three projects. Each is a real experiment with a pre-registered hypothesis card, matched
arms, a kill threshold, and a definition of done. Each is sized for one gfx1151 GPU, no
collectives, 20M–300M parameters, and a person with a demanding day job. Each is designed
so that a good result is publishable and a null result is still informative — that second
property is the harder one to design for and it is where most of the specification effort
below went.

**Capstone slugs** (house naming rule — the name states the claim, not an ordinal):

| Slug | The one-sentence claim it attacks | Build difficulty | Evidence difficulty | Calendar |
|---|---|---|---|---|
| `telemetry-detects-injected-fault` | A training run's telemetry can detect a fault you injected on purpose, before the loss curve shows it. | 3/5 | 4/5 | 5–7 weeks |
| `eviction-recall-breakpoint` | There is a KV budget at which recall breaks, the break is sharp, and where it sits depends on the policy — not just on the budget. | 4/5 | 5/5 | 7–10 weeks |
| `hybrid-ratio-recall-cliff` | The published hybrid ratio is a recall cliff, not a perplexity preference — and the cliff does not close with more training. | 4/5 | 5/5 | 10–14 weeks |

**Build them in that order.** The reason is a dependency, not a difficulty ramp:
`telemetry-detects-injected-fault` produces the run record, the manifest, the watchdog and
the null-distribution discipline that the other two consume, and it is the only one of the
three whose primary skill you already have. `eviction-recall-breakpoint` produces the
attribution harness that `hybrid-ratio-recall-cliff` needs in order to say anything about
mechanism rather than outcome. The dependency graph is:

```
telemetry-detects-injected-fault
   ├─ run manifest + JSONL schema + collated drain  ──┐
   ├─ silent-hang watchdog (wallclock-gap detector)  ──┤
   └─ seed-to-seed null distribution, measured       ──┤
                                                       ├──> eviction-recall-breakpoint
                                                       │       └─ oracle-diff harness ──┐
                                                       │                                 │
                                                       └────────────────────────────────┴──> hybrid-ratio-recall-cliff
```

---

## Before any of them: the envelope

Everything below is bounded by these. They are instrument characterisation, not results,
and every one of them changed at least one design decision in this document.

| Constraint | Value | Tag and source |
|---|---|---|
| Fast memory tier, flat bandwidth | **≥62 GiB at ~200 GB/s**, upper edge unmeasured | `[M]` 2026-07-26, `ASSUMPTIONS.md: gpu-fast-tier-size`, `notebook/uma-carveout-controls-fast-tier.md` — **single run per arm, an anecdote by the house standard** |
| Single tensors ≥32 GiB | 31 GiB copies clean at 199.9 GB/s; **32 GiB hard-hangs at 0% CPU with no error**; 36 GiB raises `hipErrorLaunchFailure` | `[M]` `ASSUMPTIONS.md: large-tensor-fault-32gib` |
| Distributed | `torch.distributed.is_available()` is **False**; `torch._C._distributed_c10d` **is not in the build**; FSDP fails at *import* | `[M]` `ASSUMPTIONS.md: single-device-only` |
| GEMM, bf16, 8192³ | 20.9 TFLOP/s with hipBLASLt configured, 18.6 without | `[M]` `ASSUMPTIONS.md: hipblaslt-config`, `scripts/benchmark_gemm.py` |
| hipBLASLt as a numerics control | relative error of a length-1,048,576 bf16 weighted sum against fp64: **2.01e-3 configured, 5.60e-3 unset**, 3 seeds | `[M]` `ASSUMPTIONS.md: hipblaslt-config` |
| bf16 numerics generally | **unproven** — the Hardware Validation Gate has not run | `[C]` five documented gfx1151 bf16 bugs (ROCm #6034) |
| SDPA memory path | default retains the score matrix at **147.2 bytes/T²**; `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` gives 6.6. `flash_sdp_enabled()` returns True either way | `[M]` `ASSUMPTIONS.md: sdpa-is-memory-efficient`; flag stays off by default per ADR `aotriton-attention-stays-off-by-default` |
| fp32 gradients | match a CPU reference to **3.9e-8 absolute** | `[M]` `curriculum/README.md` findings |
| Host RAM after the carve-out | **32 GB** | `[M]` `ENVIRONMENT.md` 2026-07-26 — this is what bounds every CPU-fallback arm |

### Four constraints measured while writing this file, not yet in the register

These were run in the lab venv (`C:\venvs\lab`, `torch 2.12.0a0+rocm7.13.0a20260313`,
native Windows, gfx1151) on 2026-07-26. **They are not in `ASSUMPTIONS.md` yet and should
be appended as rows** — three of them are load-bearing for capstone two and one for
capstone three.

1. `[M]` **There is no Triton on this stack.** `torch.utils._triton.has_triton()` returns
   `False`. `importlib.util.find_spec` returns `None` for `triton`, `mamba_ssm`,
   `causal_conv1d`, `fla` and `einops`.
2. `[M]` **There is no Triton wheel to install.** `pip index versions triton` and
   `pip index versions pytorch-triton-rocm`, both against
   `https://rocm.nightlies.amd.com/v2/gfx1151/ --pre`, return
   `ERROR: No matching distribution found`.
3. `[M]` **The distributed hole is wider than the register records.**
   `import torch.distributed.checkpoint` and `import torch.distributed.tensor` both raise
   `ModuleNotFoundError: No module named 'torch._C._distributed_c10d'`. So DCP and DTensor
   are gone too, not just FSDP. `from torch.nn.parallel import DistributedDataParallel`
   imports (it lives in `torch.nn`), which is a trap: the symbol exists and construction is
   where it would die. `[A]` high confidence, untested — I did not construct one.
4. `[A]` **Consequence, high confidence, from (1):** `torch.compile`'s inductor GPU backend
   generates Triton. With `has_triton()` False there is no fused-kernel codegen path on
   this machine at all. Cheapest test that would move it: `torch.compile` a two-layer block
   and check whether it silently falls back to eager or raises.

**Read (3) twice if you are about to borrow a trainer.** OLMo-core's checkpointer is built
on `torch.distributed.checkpoint`. On this machine it fails at import, so every capstone
below rolls its own checkpoint I/O with plain `torch.save`/`torch.load` plus a manifest.
That is not a shortcut; it is the only available option, and it is why
`checkpointing-and-resumption.md` exists as its own module.

### One number the curriculum does not agree with itself about

Every wall-clock estimate below is a function of **sustained end-to-end training
throughput**, and that number has never been measured on this machine. The curriculum
currently carries three different `[A]` values for it: 7.3 TFLOP/s
(`building-an-eval-you-can-trust.md` §3.6 and `schedule.md`, taken as 35% of the `[M]` GEMM
figure), 6 TFLOP/s (`scaling-laws-and-flops-budget.md` and `the-training-loop.md`), and this
document's band. They cannot all be right and the spread is 20%+ before you even ask whether
any of them is close.

This file therefore does **not** pick one. It parameterises every cost by `η`, the achieved
fraction of the `[M]` 20.9 TFLOP/s GEMM figure, and quotes the tables at
`η ∈ {0.10, 0.20, 0.30}` — a band that brackets 2.1–6.3 TFLOP/s and sits *below* both
existing assumptions, which is the conservative direction for planning. `[A]` medium
confidence in the band.

**The cheapest test is one timed 200-step nanoGPT-scale run**, and it is the first thing to
do in each capstone's week one. It is also, per `OPERATING_INSTRUCTIONS.md` G3, already the
top-priority measurement: the conclusion (how many arms fit in the calendar) flips when this
input moves ±30%, and it has moved by more than that between two documents in this same
directory.

**Platform.** Native Windows only. WSL2 clamps the ROCm pool to the `.wslconfig` memory
value and cannot reach dedicated VRAM `[C]` (ROCm issue #6022), which negates the ≥62 GiB
fast tier — the single measured property that makes all three of these affordable. Do not
run any capstone arm under WSL2, including "just to check something."

**CPU fallback is real but small.** 32 GB of host RAM `[M]` after the 96 GB carve-out means
the CPU path runs at reduced shapes, and every capstone below states its CPU shape
explicitly. A CPU fallback that OOMs the host is worse than no fallback, because it takes
the machine with it.

---

## An honesty note about two of these three

The kickoff fixed the three capstones. Two of them sit on nodes that
`research/memory/open-problems-ranked.md` and `research/synthesis.md` deliberately **parked**,
and pretending otherwise would be exactly the kind of hand-waving this curriculum is
supposed to be free of.

- **Eviction policy design** is issue-tree node 3.1, parked, because roughly thirty policies
  exist and the March 2026 survey scores five families against seven deployment scenarios
  and finds **no method dominates** `[C]` (2603.20397). ADR
  `attribution-instrument-over-eviction-policy` (Proposed, 2026-07-26) says in as many
  words: add no new eviction policy until we can measure one honestly.
- **Hybrid ratio selection** is parked with P 4 · T 3 · E 2, on the grounds that the contest
  is about token budget and token budget is the axis we can least afford to sweep — the
  entry fee in the largest published sweep was 72 trained models `[C]` (2507.06457).

Both capstones are therefore specified as **instrument work with a research by-product**,
not as policy or architecture proposals:

- `eviction-recall-breakpoint` re-implements **published** policies as *calibration targets*
  for the attribution harness. The deliverable is the harness and its null distribution.
  The thirty-first policy is not written. If it turns out the harness cannot separate
  policies at all, that is the ADR's own kill condition firing, which is a result.
- `hybrid-ratio-recall-cliff` is scoped to the one question the parked item does not
  answer and the two published readings actively disagree about: whether the ratio sets a
  **capability ceiling** `[C]` (2507.06457) or governs **how fast** the capability emerges
  `[C]` (2606.15378). Running two token budgets is what distinguishes them, and it is the
  cheapest available discriminator between two live positions.

If you decide to run them as stated anyway, that is a legitimate override — log it per
`OPERATING_INSTRUCTIONS.md` → OVERRIDE PROTOCOL, naming the gate and the risk. What is not
legitimate is running them and reporting them as if the backlog had ranked them first.

---
---

# Capstone: `telemetry-detects-injected-fault`

**Difficulty:** 3/5 to build (this is your day job with different nouns), 4/5 on the
statistics (the detection threshold is where it gets hard, and it is the part nobody in ML
does properly).
**Calendar:** 5–7 weeks at ~8 h/week. Roughly 45 h of your hands plus ~40 h of unattended
GPU.
**Prerequisites:** `the-training-loop`, `loss-and-optimization`,
`training-telemetry-as-observability`, `determinism-and-reproducibility`,
`checkpointing-and-resumption`, `measuring-memory` §2.6–§2.7.

## The G2 hypothesis card

```
HYPOTHESIS   A JSONL-first training telemetry pipeline that emits per-document loss
             statistics and a four-field environment manifest detects each of five
             deliberately injected training faults within 200 optimizer steps of
             injection, at a false-alarm rate of at most 1 alarm per 10,000 clean
             steps — while the batch-mean loss, which is the only signal the reference
             trainers log by default, detects at most two of the five at the same
             false-alarm budget.

FOR          Themis/Argus, the training telemetry subsystem, on a ~26M-parameter
             decoder, single gfx1151 device, native Windows.

BECAUSE      (a) The batch mean is a lossy aggregation with a computable dilution
             factor: [M] in a 32-document batch with one document replaced by uniform
             noise, the batch-mean loss moves 0.91 sigma while the per-document maximum
             moves 24.3 sigma — a 26.7x difference in detectability from one extra
             .max() (curriculum/training-telemetry-as-observability.md, one run, an
             anecdote by the house standard).
             (b) Two of the five faults are specific to this machine and are silent by
             construction: a >=32 GiB allocation hangs at 0% CPU with no exception [M]
             (ASSUMPTIONS.md: large-tensor-fault-32gib), and unsetting
             HIPBLASLT_TENSILE_LIBPATH changes the relative error of a long bf16
             reduction from 2.01e-3 to 5.60e-3 [M] (ASSUMPTIONS.md: hipblaslt-config)
             without changing anything a normal trainer records.
             (c) No result from this machine is admissible until the instrument that
             produces it can be shown to fire. research/synthesis.md makes eval
             calibration by fault injection gate everything downstream.

MEASURED BY  Detection latency in optimizer steps, per fault, at a threshold calibrated
             on clean runs to the stated false-alarm budget. Instrumented by
             themis.argus: one JSONL record per step, one manifest per run, detectors
             run offline over the JSONL so the detector is not in the training loop.
             Secondary: false-alarm rate on the three clean seeds; throughput cost of
             the instrumentation in tokens/s.

SUCCESS      >=4 of 5 faults detected within 200 steps at <=1 alarm / 10,000 clean
             steps, AND the mean-loss-only baseline detects <=2 at the same budget,
             AND the specificity control (a cosmetic change that is not a fault) does
             not fire, AND instrumented throughput is within 5% of uninstrumented.

KILL         The mean-loss baseline detects >=4 of 5 at the same budget — the extra
             fields buy nothing and the schema should be cut back to what OLMo-core
             already emits. OR: the clean-run false-alarm rate cannot be brought under
             budget for any threshold that also catches >=2 faults, which would mean
             run-to-run variance on this chassis dominates the signal and the whole
             detection framing is wrong at this scale.

COST         $0 cloud. ~45 h of hands over 5-7 weeks; ~40 h unattended GPU (18 runs at
             26M x 0.1B tokens, plus 3 short hang-fault runs that are force-killed).
             Timebox: if the clean-run null is not characterised by end of week 3,
             report and re-decide rather than continuing.

RISKIEST     That a stable clean-run null exists on this hardware at all. [M] The
             hybrid module's Exercise B found two identical decode-stream runs twenty
             minutes apart differed by 24% in absolute bandwidth, consistent with
             thermal throttling on a tablet chassis. If the same drift appears in
             training-step statistics, thermal noise is larger than several of the
             faults and the detector thresholds on the wrong thing. Cheapest test that
             would move it: three clean runs at three times of day, and report the
             between-run variance of every logged series before building any detector.
```

## What it builds

A telemetry subsystem inside Themis, and nothing else. No new model, no new memory policy.

1. **A per-step JSONL record** with the fields generated by the five questions in
   `training-telemetry-as-observability.md` §2.3 — which run is this, where in the run am I,
   what was the input, what did the model do, what did the machine do. Plus the two fields
   that module adds and the reference trainers do not have: `loss_doc_max` and
   `loss_doc_std`.

2. **A run manifest** carrying, at minimum, `git_sha`, `config_sha`, `seed`,
   `torch_version`, `rocm_version`, `driver_version`, `gfx_arch`, and the four environment
   fields that change this machine's arithmetic: `hipblaslt_configured`,
   `hipblaslt_tensile_libpath`, `aotriton_experimental`, `attention_backend`. A run record
   without those four is not interpretable here `[M]`, and both of the numerics-relevant
   ones have a measured effect size in `ASSUMPTIONS.md`.

3. **A collated metric drain.** One `torch.stack(list(buffer.values())).to("cpu")` per drain
   instead of one `.item()` per series. `[M]` The measured marginal cost is ~38 µs per
   series for the per-item path against ~1.29 µs for the collated path, a 29× gap at 2,796
   series, with a break-even at about two metrics
   (`training-telemetry-as-observability.md` §2.6). You are implementing this yourself
   because `[A]` OLMo-core's own collation never engages on a single device — the code path
   is guarded by `is_distributed()`.

4. **A silent-hang watchdog.** A `wallclock_s` gap with no missing step numbers is the
   signature of `large-tensor-fault-32gib` `[M]`. The watchdog is a separate process reading
   the JSONL tail; it must be a separate process, because the hang is at 0% CPU inside the
   training process and nothing in that process will ever run again.

5. **Fault injection as a config axis, not a debug branch.** `building-an-eval-you-can-trust.md`
   §4.3 is right and this is the place to obey it: every fault is a field in the run config,
   with a step at which it activates, and it appears in the manifest of every run including
   the clean ones (as `fault: none`). If injecting a fault requires editing code, the fault
   battery will rot within a month.

6. **Checkpoint save/load with a round-trip test.** Rolled by hand — `[M]`
   `torch.distributed.checkpoint` fails at import on this wheel. The round-trip test asserts
   bit-exact recovery of model, optimizer, LR-scheduler and data-cursor state, because two of
   the five faults are resume faults and you cannot detect a resume fault with a resume path
   you have not proven.

## What it measures

**Primary:** detection latency in optimizer steps, per fault, at a fixed false-alarm budget.

The framing is deliberately the one you already use for alerting, because it is the right
one and ML does not use it: a detector is a point on an ROC curve, a threshold is chosen by
the false-alarm budget you can live with, and reporting sensitivity without the
corresponding false-alarm rate is meaningless. Concretely:

```
For each series s and each detector D:
  1. Run three clean seeds. Compute the distribution of D(s) over all clean steps.
  2. Choose the threshold tau_s such that P(D(s) > tau_s | clean) <= 1e-4 per step.
     (1 alarm per 10,000 steps. Pick this number before you look at any faulted run.)
  3. Run the faulted seeds. Detection latency = first step after injection at which
     D(s) > tau_s, or "not detected" if it never fires before the run ends.
```

`training-telemetry-as-observability.md` uses OLMo-core's 6σ-over-a-128-step-window rule as
the reference detector (`stability_monitor.py:29`, `:35`). Use it as one arm. It is a
z-score against a rolling window, which is a perfectly respectable detector and also
exactly the detector that the dilution arithmetic says cannot see a single bad document in
a large batch.

**Secondary:**
- Clean-run false-alarm rate actually achieved (it will not equal 1e-4; report the gap).
- Throughput cost: tokens/s instrumented against uninstrumented, at fixed microbatch size.
  Hold microbatch size fixed across arms and say so in the write-up — a heavier-instrumented
  run can have different memory behaviour, therefore a different achievable microbatch,
  therefore a different arithmetic intensity, therefore a throughput number that is not the
  throughput of the run you care about.
- Between-run variance of every series across the three clean seeds. This is the number the
  RISKIEST line is about, and it is worth publishing on its own.

## Arms and controls

| Arm | What is injected | At step | What should notice it |
|---|---|---|---|
| `clean` (×3 seeds) | nothing | — | nothing. This arm *defines* the thresholds. |
| `fault-poisoned-document` | one document per batch replaced by uniform-random tokens | 2,000 | `loss_doc_max`; `[M]` predicted 24.3σ against 0.91σ for the mean |
| `fault-dataloader-cursor-reset` | resume from checkpoint with the data cursor silently reset to 0 | resume at 3,000 | `data_cursor_tokens` directly; `loss_train` via a repeat-data drop |
| `fault-scheduler-restart` | LR scheduler restarts at step 0 on resume while the optimizer does not | resume at 3,000 | `lr` directly; `update_to_param_ratio` and `grad_norm_preclip` indirectly |
| `fault-hipblaslt-unset` | `HIPBLASLT_TENSILE_LIBPATH` cleared for the child process | 2,000 | `[M]` long-reduction relative error 2.01e-3 → 5.60e-3; expected to show in `grad_norm_preclip` distribution and `update_to_param_ratio` |
| `fault-duplicate-shard` | one data shard repeated in place of the next | 2,000 | `loss_train` slope; `loss_doc_std` compression |
| `fault-silent-hang` (separate short runs) | allocate a 32 GiB tensor `[M]` | 500 | the wallclock-gap watchdog, and *only* the watchdog |
| **`control-cosmetic`** | rename three metrics and reorder the JSONL keys | 2,000 | **nothing.** This is the specificity control. |

Three things about that table.

**The cosmetic control is the part the literature does not have.**
`building-an-eval-you-can-trust.md` §2.6 notes that the published six-fault battery is five
sensitivity tests and one specificity test, so a metric that moves for *everything* passes
it. A detector that fires on a key reorder is a detector that will fire on your next config
change, and you will learn to ignore it. Include it.

**`fault-hipblaslt-unset` is the interesting one and it may well fail to be detected.**
That is fine and it is informative: it would mean a 2.8× degradation in long-reduction
accuracy is invisible to every field a trainer logs, which is a sharper statement of the
`bf16-numerics-unproven` risk than the register currently makes. Pre-register the
possibility so it is not a post-hoc story.

**`fault-silent-hang` cannot share a run with anything else** because it ends the process.
Run it separately, at 500 steps, force-kill after a fixed timeout, and record the timeout in
the manifest. Note the failure mode you are exercising: 11 minutes at 0 CPU with host free
RAM falling to 5 GB `[M]`. Do not run it on a machine you need for anything else that
evening.

## Model and data shape

| | GPU arm | CPU fallback |
|---|---|---|
| Params | ~26M (`d_model` 512, `L` 8, `n_head` 8, `n_kv` 2, `d_head` 64, `d_ff` 2048, vocab 8,192, tied embeddings) | same architecture, `d_model` 256, `L` 4 |
| Context | 1,024 | 256 |
| Batch | 32 documents | 8 |
| Tokens | 0.1B | 5M (enough to see the faults, not enough to train) |
| Precision | bf16 autocast with fp32 master weights; **also run one clean seed entirely in fp32** | fp32 |
| Runtime per run | `[A]` ~2.1 h at 10% of the `[M]` 20.9 TFLOP/s GEMM figure; see the cost arithmetic below | ~4–6 h |

0.1B tokens is below `CLAUDE.md`'s 0.5–5B ablation floor. That floor is for ablations; this
is instrument calibration and the model does not need to be good, it needs to be
*instrumented*. Say so in the notebook entry rather than quietly deviating.

**The fp32 clean seed is not optional.** `bf16-numerics-unproven` is untested, and if the
bf16 arm's clean null is wider than the fp32 arm's, you have measured something about the
hardware rather than about detection — which is itself a Hardware Validation Gate result.

## Cost arithmetic, and why it is an `[A]`

Training FLOPs are `6·N·D` — six FLOP per parameter per token, forward plus backward
(`scaling-laws-and-flops-budget.md`). At `N = 26e6`, `D = 1e8`:

```
6 · 2.6e7 · 1e8  =  1.56e16 FLOP per run
```

Wall-clock is that divided by achieved throughput, and **achieved throughput on this machine
is unmeasured.** The only anchor is `[M]` 20.9 TFLOP/s on an 8192³ GEMM, which is a best
case and not what a training loop achieves. Write `η` for the achieved fraction:

| `η` (achieved fraction of the `[M]` GEMM figure) | hours per run | 18 runs |
|---|---|---|
| 0.10 | 2.07 | 37 h |
| 0.20 | 1.04 | 19 h |
| 0.30 | 0.69 | 12 h |

`[A]` medium confidence that `η` lands in 0.10–0.30; the cited 25% MFU figure for this
silicon is one un-peer-reviewed GitHub issue and `z13-is-right-instrument` was downgraded to
`untested` on exactly that basis. **Measure `η` in week one with a 200-step pilot and
re-plan.** If `η < 0.05`, cut the token budget rather than the arms — you need all three
clean seeds and all five faults, and you do not need a good model.

## Read the code before you start

All paths relative to `research/reference/`. Line numbers pin to the revisions in
`PROVENANCE.md`.

| Where | What to look at |
|---|---|
| `training/nanogpt/train.py:327` | `print(f"iter {iter_num}: loss {lossf:.4f}, ...")` — the entire telemetry surface of the reference implementation, on one line. This is the baseline your SUCCESS criterion is measured against. |
| `training/olmo-core/src/olmo_core/train/trainer.py:1037` | `record_metric` — how a serious trainer buffers a metric as an *unevaluated device tensor* rather than a float. The reason is that reading it stalls the pipeline. |
| `training/olmo-core/src/olmo_core/train/utils.py:216` | `out[step][name] = value.item()` — the per-series device-to-host round trip. This is the 38 µs, once per series, that your collated drain replaces. |
| `training/olmo-core/src/olmo_core/train/utils.py:126` | `metrics_to_move_list = [` — the collation that exists and, on a single device, never runs. |
| `training/olmo-core/src/olmo_core/train/trainer.py:366` | `if self._bookkeeping_pg is None and is_distributed():` — the guard that makes the previous line dead code for us. Read `:548` next. |
| `training/olmo-core/src/olmo_core/train/trainer.py:200` | `metrics_collect_interval: int = 5` — the **drain** interval. |
| `training/olmo-core/src/olmo_core/train/callbacks/gap_monitor.py:43` | `interval: int = 1` — the **sampling** interval. These are different knobs and conflating them is the single most common mistake in this area. |
| `training/olmo-core/src/olmo_core/train/callbacks/stability_monitor.py:29`, `:35` | `window_size: int = 128`, `threshold_std: float = 6.0` — the reference detector, and your baseline arm. |
| `training/olmo-core/src/olmo_core/train/callbacks/metric_saver.py:85` | `self.trainer.write_file(fname, json.dumps(metrics))` — why the durable tier is a plain file and not the tracker. |
| `training/olmo-core/src/olmo_core/train/callbacks/console_logger.py:33` | the glob allow-list that turns the full metric set into the human-facing view. Your dashboards are a filtered view of the record, not a second record. |

**The read to do slowly:** put `trainer.py:366` and `trainer.py:548` side by side and
convince yourself that `async_bookkeeping` stays `None` on a single device. That one
inference is the difference between borrowing OLMo-core's performance characteristics and
borrowing its code.

## Definition of done

- [ ] Every rig component arrived red-then-green. The JSONL schema has a test that fails
      when a required field is absent; the collated drain has a test asserting it returns
      the same values as the per-item path; the checkpoint round-trip test asserts bit
      exactness and was seen to fail before the checkpointer existed.
- [ ] Three clean seeds run to completion, and the between-run variance of every logged
      series is published in the notebook entry **before** any threshold is chosen.
- [ ] Thresholds chosen from the clean runs only, committed to git, and not touched again.
      Moving one after seeing a faulted run is a change of standard and must be labelled as
      one.
- [ ] All five faults plus the hang plus the cosmetic control run at three seeds each.
- [ ] Detection-latency table published with the false-alarm rate actually achieved, not the
      budget aimed at.
- [ ] Throughput cost measured at fixed microbatch size, instrumented against
      uninstrumented, 3 repeats.
- [ ] `ASSUMPTIONS.md` updated: the four unregistered rows from this file's envelope
      section, plus whatever this capstone establishes about clean-run variance.
- [ ] The notebook entry is written whether the hypothesis held or not, with equal care.
- [ ] The watchdog is running in every subsequent capstone's runs. This is the deliverable
      that outlives the experiment.

## Connection to the ablation backlog

| Backlog item (`research/memory/open-problems-ranked.md`) | How this feeds it |
|---|---|
| **#1 Attribution** (P5·T5·E5, the only 5/5/5) | The oracle-diff harness in `eviction-recall-breakpoint` emits per-probe records. This capstone builds the record format, the manifest, and the discipline of measuring the null before the effect. Without it, #1 produces numbers with no error bars. |
| **Open question 3** — the clean fast/slow bandwidth ratio under one BIOS configuration with ≥3 seeds | Blocked on a stable timing instrument. The clean-run variance table this capstone publishes is the prerequisite. |
| **The Hardware Validation Gate** | `research/synthesis.md` argues the gate as written in `CLAUDE.md` is under-specified. Three items here belong in it: the fp32-versus-bf16 clean null, the `hipblaslt-unset` detection arm, and the wallclock-gap watchdog. |
| **Issue-tree node 4.3** — eval calibration by fault injection, `PURSUE`, gates everything | This is that node, applied to training telemetry instead of to a recall eval. The methodology transfers directly. |

## If it works, and if it nulls

**If it works,** you have a fault-injection calibration protocol for training telemetry with
measured detection latencies and false-alarm rates. `research/synthesis.md` already argues
that fault-injection calibration of memory *evals* is a publishable methodology result with
zero training runs; the training-telemetry version is the same argument in a field
(observability) that has a lot of practitioners and very little published discipline. The
detection-latency-at-fixed-false-alarm-rate table is the artifact.

**If it nulls** — that is, the mean-loss baseline catches four of five — you have shown that
the schema extension is not worth its cost, which saves you from carrying `loss_doc_max`
through every subsequent run and from believing a `[M]` figure that came from one run. Cut
the schema and say why.

**If the RISKIEST fires** — no stable clean null on this chassis — that is the most
important of the three outcomes, because it invalidates every small effect size this lab
intends to measure, including the ~0.01-perplexity deltas that hybrid ratio work lives on.
Escalate it to `BLOCKERS.md` immediately and do not start capstone two.

## Self-check before you start

1. Why can a batch-mean loss not detect one bad document in a 512-document batch, *in
   principle*, independently of noise?
2. You have 2,796 series and a drain every 5 steps. What is the throughput cost, and what
   would it be if you drained every step?
3. Why must the hang watchdog be a separate process?
4. Which of the five faults would a production APM tool detect out of the box, and which
   would it structurally miss?

---
---

# Capstone: `eviction-recall-breakpoint`

**Difficulty:** 4/5 to build, 5/5 on the evidence. The code is a scoring function and a
gather; the discipline of not believing your own policy comparison is the entire project.
**Calendar:** 7–10 weeks at ~8 h/week. Roughly 60 h of hands plus ~35 h of GPU.
**Prerequisites:** all of Track C through `measuring-memory`, plus
`measuring-recall-and-memory` and `building-an-eval-you-can-trust` from Track E.
`telemetry-detects-injected-fault` must be done — you need its run record and its null
discipline.

## The gate, run before anything is built

ADR `attribution-instrument-over-eviction-policy` names a cheapest decisive test and it
takes an afternoon. Run it first. If it fails, this capstone changes shape.

> Take a model you already have. Construct a prompt with a needle at a known token range.
> Run decode twice with identical seeds: once with the full cache, once with **exactly the
> KV entries spanning the needle** dropped and nothing else. Log per-token KL divergence
> between the two output distributions. Then repeat with an equal number of entries dropped
> from **non-needle** positions.
>
> **Pass:** the KL spike localises — it appears at the tokens that recover the needle, and
> the non-needle drop produces a spike within the seed-to-seed null.
> **Fail:** KL smears uniformly, or the non-needle drop produces a comparable spike.
>
> Run it in fp32. `bf16-numerics-unproven` is untested and this test is about small
> divergences. It also depends on determinism, which is a Hardware Validation Gate item.

If it fails, the differential instrument does not localise and the capstone reduces to a
recall-breakpoint measurement with no attribution — still worth doing, but the ADR is
superseded rather than patched, and you should write that superseding ADR before continuing.

## The G2 hypothesis card

```
HYPOTHESIS   For each of five KV-retention policies there exists a budget fraction b*
             (the recall breakpoint) below which recall@1 on a salience-stratified
             retrieval suite falls below the midpoint between full-cache recall and the
             suite's closed-form chance level; b* differs across policies by at least a
             factor of 2 at matched bytes-held AND matched bytes-read; and the ordering
             of b* at ~26M agrees with the ordering at ~251M with Spearman rho >= 0.7.

FOR          Mnemosyne's cache interface — write-time admission, deferred eviction,
             read-time selection — on a single gfx1151 device at two model scales.

BECAUSE      (a) The exact eviction-error identity is
             o_t - o_S = sum_{i not in S} a_i (v_i - o_S), a product of dropped
             attention mass and dropped-value distance; every attention-score policy
             optimises the first factor and ignores the second entirely
             (curriculum/kv-eviction-policies.md §3.3). Two policies with the same
             dropped mass can differ by 22x in error [M] on a worked four-token example.
             Therefore b* should be policy-dependent, and if it is not, the identity is
             not doing the work it appears to do.
             (b) The field has ~30 policies and no dominance result [C] (2603.20397),
             five policies silently dropping specific instructions while aggregate
             scores hold [C] (2510.00231), and an explicit argument that task accuracy
             alone cannot say why a selector worked [C] (2605.08234).
             (c) [M] At 22 KiB/token and a >=62 GiB fast tier, the full-cache oracle
             costs ~176 MiB per probe against a ~500 MB model. The counterfactual that
             is unaffordable at 70B is free here. Small scale is the enabling condition,
             not the compromise.

MEASURED BY  recall@1 on the salience-stratified probe suite, primary, reported against
             its closed-form chance level and its measured seed-to-seed null, with 3
             seeds and 95% CIs. Instrumented by mnemosyne.attribution: per probe, per
             head, per decode step, the three scalars ||u||, (1-A)||o_S|| and
             cos(u, o_S) that reconstruct the eviction error exactly without
             materialising a T x T matrix. Both compression numbers reported always:
             bytes held and bytes read.

SUCCESS      b* located for all five policies at both scales with non-overlapping 95%
             CIs between at least two policies, a >=2x spread in b*, and Spearman
             rho >= 0.7 between the two scales' orderings.

KILL         (i) The full-cache arm's own recall is within the null of chance — the
             probe suite is broken and nothing downstream means anything. Fix the suite
             or stop. OR (ii) every policy's b* is within the null width of the
             matched-budget RANDOM-eviction control — there is nothing to rank, which
             refutes the premise of the eviction literature at this scale and is a
             publishable negative. OR (iii) Spearman rho <= 0 between scales —
             ablation-scale-sufficient is refuted and the entire lab backlog needs
             re-planning, which is the single most valuable outcome available here.

COST         $0 cloud. ~60 h of hands over 7-10 weeks; ~35 h GPU (see the probe
             arithmetic). No training runs beyond the checkpoints that capstone
             hybrid-ratio-recall-cliff or a nanoGPT reproduction already produced.
             Timebox: if the harness is not producing a reproducible null distribution
             by end of week 4, report and re-decide.

RISKIEST     That per-token divergence from the full-cache oracle localises to
             identifiable dropped cache entries rather than smearing across all of them.
             Both independent syntheses named this without prompting
             (research/synthesis.md). If it smears, the attribution half collapses and
             only the recall number survives — which is precisely the
             outcome-without-mechanism result this lab exists to avoid. Attacked by the
             gate above, before any code is written.
```

## What it builds

**Mnemosyne, the memory subsystem, in its first real form.** The boundary rule from
`CLAUDE.md` applies with full force: `mnemosyne` imports `torch` and never `proteus` or
`themis`. That is enforced by `packages/mnemosyne/pyproject.toml` not declaring the
dependency and by the lint contract in `tests/test_package_boundaries.py`, which has been
proven red-then-green `[M]`. Do not "temporarily" add the dependency.

1. **Three plug points, not one hook.** `kv-eviction-policies.md` §5.1 establishes that
   `score(keys, values, attention) -> subset` cannot host the literature:

   | Plug point | Runs | Inputs available | Hosts |
   |---|---|---|---|
   | write-time admission | as each token's KV is produced | `k_i`, `v_i` only, `O(d)`, no attention matrix | L2-norm `[C]` (2406.11430), KeyDiff `[C]` (2504.15364) |
   | deferred eviction | after a bounded staging delay | prompt-side or `k`-step-lagged attention | H2O `[C]` (2306.14048), SnapKV `[C]` (2404.14469), PyramidKV `[C]` (2406.02069), Ada-KV `[C]` (2407.11550), KVpop `[C]` (2607.05061) |
   | read-time selection | per decode step | the actual `q_t` | Quest `[C]` (2406.10774), RocketKV stage 2 `[C]` (2502.14051) |

   This decision belongs in `docs/adr/mnemosyne-cache-plugpoints.md` before a line of policy
   code is written, because it is not recoverable later.

2. **Five policies, all of them published, none of them new.** `mnemosyne-window`
   (StreamingLLM: sink tokens plus a recent window, `[C]` 2309.17453), `mnemosyne-h2o`
   (accumulated attention), `mnemosyne-snapkv` (observation-window attention at end of
   prefill), `mnemosyne-keydiff` (key distinctiveness, write-time), and
   `mnemosyne-random` (uniform random at matched budget — the control, and it is a policy
   arm, not a footnote).

3. **The attention oracle.** Top-`B` by *true future* attention, computed offline by running
   the full cache alongside. This is the half of Belady's MIN that transfers: the trace is
   computable after the fact. It is a **diagnostic ceiling, not an optimum** — the
   compressed output appears inside its own error term, so dropping token `i` changes the
   cost of dropping token `j` and the errors are not additive. Report two gaps at every
   budget: policy-to-oracle is *policy headroom*, oracle-to-full-cache is *budget headroom*.
   As far as this curriculum could establish, no eviction paper reports both.

4. **The oracle-diff harness.** Same prompt, same seed, one run with the full cache and one
   under policy `P`; per-token KL divergence between the two output distributions;
   attribution of each divergence spike to the specific entries the policy dropped. Plus the
   three scalars per head per step:

   ```
   u          = sum_{i not in S} a_i v_i          the lost signal
   (1-A)·o_S                                       the renormalisation kickback
   cos(u, o_S)                                     the angle between them

   ||o_t - o_S||^2 = ||u||^2 + ((1-A)||o_S||)^2 - 2(1-A)||u||·||o_S||·cos(u, o_S)
   ```

   `[M]` Log those three and you reconstruct the exact error to 1.1e-06 in fp32 at every
   budget. Do **not** log `(1−A)` alone: `[M]` its within-budget R² against the actual error
   is 0.003 on Gaussian data, while `(1−A)·‖o_S‖` reaches 0.97–0.999
   (`kv-eviction-policies.md` §3.6). That measurement is on synthetic data, and testing it
   on real attention is one of this capstone's by-products.

5. **The null distribution.** The same harness on structureless data — Gaussian keys and
   values, no trained model. A policy ranking that reproduces on Gaussian noise is not a
   finding about language models. This is the number nothing downstream is interpretable
   without, and it is cheap: CPU, fp32, seconds.

## What it measures

**Primary: the recall breakpoint `b*`.** Defined precisely so it cannot be moved later:

```
Let R_full  = recall@1 with the full cache,  R_chance = the suite's closed-form chance level.
b* = the largest budget fraction b in the swept grid at which
     recall@1(b) < (R_full + R_chance) / 2.
```

Sweep `b ∈ {1, 2, 5, 10, 25, 50, 100}%`. Report `b*` with a bootstrap CI over probes and
seeds. If `recall@1` is not monotone in `b`, say so loudly — non-monotonicity is a real
result and it is what a value-outlier effect would look like.

**The probe suite, and why the obvious one is disqualified.** Needle-in-a-haystack cannot be
the primary probe here, and the reason is structural rather than stylistic: a needle is a
low-frequency, semantically anomalous, high-salience span, so it *attracts attention mass* —
which is exactly what heavy-hitter eviction retains. A policy can shed 90% of the cache,
destroy ordinary long-range dependence, and still pass. `[C]` NoLiMa (2502.05167) is the
independent corroboration: rebuild NIAH so the needle shares minimal literal overlap with
the question and 11 of 13 models claiming ≥128K fall below half their own short-context
baseline at 32K.

So the suite is **salience-stratified**. Before scoring anything, run fault zero from
`memory-failure-modes.md` §3.5: measure the target span's salience rank under the full
cache, and stratify probes into salience quintiles. Report `b*` per quintile. `[A]` High
confidence, and it is the prediction that makes this capstone worth running: **`b*` will be
strongly quintile-dependent for the attention-score policies and much less so for the
window policy**, because the window does not look at salience at all. If that is right, a
single aggregate `b*` is a weighted average over a distribution you chose when you built the
haystack, which is a criticism that applies to most published eviction numbers.

Task axes, following RULER `[C]` (2404.06654) and the multi-position design in
`measuring-recall-and-memory.md`: single retrieval, multi-query associative recall `[C]`
(2312.04927), and variable-tracking. Positions swept across the context, because the
positional value prior is occupancy-dependent — the lost-in-the-middle U-shape `[C]`
(2307.03172) holds only to roughly 50% context occupancy, past which primacy decays and the
bias becomes distance-based `[C]` (2508.07479).

**Secondary:** the three attribution scalars; the two compression numbers (bytes held and
bytes read — RocketKV's abstract carries "400× compression" and "32.6% peak memory
reduction" simultaneously `[C]` (2502.14051), and the gap between those two numbers is
almost the whole subject); policy headroom and budget headroom at every budget; and the KL
localisation statistic from the gate.

## Arms and controls

**Policy arms** (5) × **budgets** (7) × **scales** (2) × **seeds** (3), on a fixed probe
suite. Plus:

| Control | Why it exists | What it would mean if it fails |
|---|---|---|
| `full-cache` on every probe | the counterfactual; it is the whole instrument | nothing works |
| `mnemosyne-random` at matched budget | the floor. A policy that does not beat random at matched budget is not a policy. | If nothing beats random, KILL (ii) fires and that is a real finding |
| `oracle-topB` at every budget | the ceiling, splitting policy headroom from budget headroom | if policies are at the ceiling, budget is the lever, not policy |
| **needle-absent** | delete the target span; recall must fall to chance | if it does not, the probe is answerable from the prior and measures nothing |
| **haystack-shuffle** | shuffle distractor order; a true retrieval task should **not** move | if it moves, you are measuring discourse structure |
| **structureless null** | Gaussian K/V, no model | if policy rankings reproduce here, they are an artifact of top-`k` selection |
| **seed null** | 3 seeds, same config | defines every "significant" claim in the write-up |
| **matched bytes-read** | re-run the best eviction arm against a sparse-read arm holding the same bytes-read budget | `[M]` on this machine bandwidth binds ~3.3× before capacity, so eviction cannot be justified on capacity grounds and must win on quality-per-byte-read |

That last row is the one a reader from a discrete-GPU background will find strange, so state
the arithmetic in the write-up. At 24 layers, `n_kv` 8, `d_h` 64, bf16 — 48 KiB/token — a
62 GiB fast tier holds 1,354,411 tokens, while 199.9 GB/s at a 10 tok/s target allows only
406,698 tokens of re-read per step. On a 20 GiB discrete card at ~1 TB/s `[A]` the ordering
inverts: capacity binds 4.7× before bandwidth. **A published "eviction beats retention"
result may be a statement about a bus, not about language models.**

## Model, probe and cost arithmetic

Two scales, ~9.6× apart, both inside the 20M–300M envelope, both with GQA group size
`G = 4` so the decode arithmetic intensity is matched:

| | small | large |
|---|---|---|
| `d_model` / `L` / `n_head` / `n_kv` / `d_head` | 512 / 8 / 8 / 2 / 64 | 1024 / 22 / 16 / 4 / 64 |
| Parameters (tied embeddings, vocab 8,192, `d_ff = 4d`) | **~26M** | **~251M** |
| KV bytes per token, `2·L·n_kv·d_head·b` | 4 KiB | 22 KiB |
| Full-cache KV at `T` = 8,192 | 32 MiB | 176 MiB |
| Oracle + policy caches, per probe | 64 MiB | 352 MiB |

Against `[M]` a ≥62 GiB fast tier, both oracles are free. That is the point of the whole
design, and it is worth writing it in the paper if this ever becomes one: **the reason this
instrument is affordable is that the KV cache is 100× the model here, which is a ratio no
production system sees.**

**muP is mandatory, not optional** `[C]` (2203.03466). Without it, "policy A beat policy B
at 251M" is indistinguishable from "the 251M model happened to be better tuned." The
two-scale rider is the backlog's item #3 and its whole value depends on the scales being
comparable.

**Probe cost.** One probe is a forward pass over `T = 8,192` for the oracle plus one per
policy-budget cell, plus a short generation. Forward FLOPs are roughly
`2·N·T + 4·L·T²·d_head·n_head`:

```
small : 2(2.6e7)(8192) + 4(8)(8192^2)(512)   ≈ 4.3e11 + 1.1e12  ≈ 1.5e12 FLOP
large : 2(2.51e8)(8192) + 4(22)(8192^2)(1024) ≈ 4.1e12 + 6.0e12 ≈ 1.0e13 FLOP
```

Note that the attention term dominates at both scales — at `T` = 8,192 with a small model,
`T²` is the whole cost. At `η = 0.10` of the `[M]` 20.9 TFLOP/s figure:

| | per forward | 500 probes × (1 oracle + 35 cells) | ×3 seeds |
|---|---|---|---|
| small | 0.73 s | 3.7 h | 11.0 h |
| large (descoped to 150 probes) | 4.9 s | 7.3 h | 21.9 h |

Total ~33 h of unattended GPU. `[A]` on `η`; measure it and re-plan.

## The hardware constraints that shape the design

Four of them, and each one changes code rather than a caveat paragraph.

**Eager attention is mandatory, and it is expensive in a specific way.** `[M]`
`sdpa_attention.py:90` has return type `tuple[torch.Tensor, None]` — the SDPA path
*structurally cannot* return attention weights, so H2O, SnapKV, PyramidKV and Ada-KV cannot
be implemented behind it. The only line in the forward pass where `a_i` exists as a value is
`modeling_laguna.py:332`, and `:337` hands the weights back to a caller that throws them
away. **The information every attention-score policy needs is computed and discarded on
every forward pass; the cost of "using" it is not computing it, it is keeping it.**

**The score matrix hits the 32 GiB silent hang at shapes you will actually use.** Bytes are
`b · n_h · T²` per layer per batch element:

| dtype | heads | `T` at which one score matrix reaches 32 GiB |
|---|---|---|
| bf16 | 8 | 46,341 |
| **fp32** | **8** | **32,768 — exactly** |

That fp32 row is not approximate: `4 × 8 × 32768² = 34,359,738,368` bytes `= 2^35 = 32 GiB`
on the nose. Since the gate test and the identity work both run in fp32 for numerics
reasons, a perfectly reasonable "let's try 32k context in fp32" will hang the GPU at 0% CPU
for eleven minutes with no error `[M]`. **Chunk the score computation over the key axis, or
cap `T` at 8,192.** Both. And run capstone one's watchdog.

**Allocate the cache per layer, never as one tensor.** At 22 KiB/token a whole-stack tensor
reaches 32 GiB at 1.5M tokens; per layer at 1 KiB/layer/token it reaches it at 32M. That is
not a style preference, it is a 22× headroom decision against a failure mode that does not
raise. It also happens to be the layout every real implementation uses, for an unrelated
reason: vLLM's pool is per-layer-group because page size in bytes differs per layer type.

**Record which SDPA path ran.** `[M]` By default this build retains the score matrix
(147.2 bytes/T²); with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` it does not (6.6). ADR
`aotriton-attention-stays-off-by-default` keeps the flag off because it is a numerics
change. The consequence for this capstone is sharp: **our default configuration hides the
very constraint the field is organised around.** Prototype an attention-score policy against
default SDPA here and it will look free, because the memory-hungry path was already running.

## Read the code before you start

| Where | What to look at |
|---|---|
| `memory/sglang/python/sglang/srt/mem_cache/evict_policy.py:16` | `class LRUStrategy(EvictionStrategy)` — a production engine's entire replacement-policy surface is one function returning a sort key. |
| `memory/sglang/python/sglang/srt/mem_cache/evict_policy.py:21` | `class LFUStrategy` — note the field it reads: `hit_count`. **That field can exist because hits are observable.** Nothing at token level has an equivalent. |
| `memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:565` | `def evict(...)` — eviction is topologically constrained to leaves, so a hot child keeps a cold parent resident forever. No LRU chain you have run behaves that way. |
| `memory/vllm/vllm/v1/core/block_pool.py:679` | `_maybe_evict_cached_block` — the only place a block leaves the hash table, called lazily at reallocation rather than at free. |
| `memory/vllm/vllm/v1/core/block_pool.py:702` | `def touch(...)` — the resurrection path. A zero-refcount block is still matchable. Compare with a token-level eviction, which is irreversible. |
| `memory/flashinfer/flashinfer/decode.py:1239` | `def plan(...)` — the paged decode plan. There is no present bit anywhere in it: a page is in `kv_indices` or the token does not exist. **A miss is unrepresentable, not slow.** |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:328` | the materialised `[B, n_h, T, T]` score matrix. This is the tensor the table above prices. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:332` | `nn.functional.softmax(..., dtype=torch.float32)` — the only line where `a_i` exists as a value, and note the fp32 upcast. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:337` | `return attn_output, attn_weights` — produced and discarded. |
| `architecture/transformers/src/transformers/integrations/sdpa_attention.py:90` | `-> tuple[torch.Tensor, None]` — the field's governing incompatibility, visible as a type. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:397` | `past_key_values.update(...)` — the one line where bytes enter the cache, and therefore the write-time admission plug point. Note that QK-norm and RoPE have already been applied, so a cached key is `RoPE(RMSNorm(k))` and KeyDiff scores a doubly-transformed quantity. |
| `memory/mooncake/mooncake-store/src/master_service.cpp:6382` | `MasterService::BatchEvict` — a real storage tier's eviction: lease-expiry-gated, partially sorted by timeout, pinning one replica for SSD writeback before freeing the rest. Read it to feel what a KV cache does *not* have. |
| `training/nanogpt/model.py:306` | `def generate(...)` re-runs the full prefix per sampled token — quadratic and cacheless. The degenerate control: a system with no KV cache has no eviction problem and pays in compute. |

## Definition of done

- [ ] The ADR gate above ran, in fp32, and its result is recorded — including if it failed.
- [ ] `docs/adr/mnemosyne-cache-plugpoints.md` exists and is Accepted before policy code.
- [ ] Every policy arrived red-then-green with a test named as a specification
      (`test_eviction_preserves_sink_tokens`, `test_budget_is_respected_exactly`,
      `test_random_policy_is_seed_reproducible`).
- [ ] The structureless null ran **before** any policy ranking was looked at, and its table
      is in the notebook entry.
- [ ] Three-scalar reconstruction verified against the exact error to fp32 tolerance on real
      attention, not just on Gaussian data. This is the fold-back that
      `kv-eviction-policies.md` §3.6 explicitly asks for and flags as untested.
- [ ] `b*` reported per policy, per scale, **per salience quintile**, with bootstrap CIs.
- [ ] Both compression numbers reported for every arm. A single number is unreportable.
- [ ] Policy headroom and budget headroom reported separately at every budget.
- [ ] Spearman ρ between the two scales' orderings reported with its own CI.
- [ ] `ASSUMPTIONS.md`: `ablation-scale-sufficient` moved off `untested` in whichever
      direction the ρ points.
- [ ] Mnemosyne's separability acceptance test run: build the wheel, install it into a clean
      venv containing only torch, run its test suite green.

## Connection to the ablation backlog

| Backlog item | How this capstone attacks it |
|---|---|
| **#1 Attribution** (P5·T5·E5) | This *is* item #1's arm, built as specified there: oracle diff, per-token KL, attribution to dropped entries, then decomposition of a policy into observation window / scoring function / budget allocation. |
| **#3 Does a ranking at 20M–300M survive to deployment scale?** | The two-scale rider, run exactly as the backlog specifies: two scales roughly 10× apart, muP mandatory, Spearman ρ on the arm ordering. KILL (iii) is the case where this refutes `ablation-scale-sufficient`. |
| **#4 Does the right retention prior change as the context fills?** | Partially. Sweeping target position across occupancy gives the occupancy-dependent positional bias for free. It does not build the occupancy-scheduled policy — that is the follow-on. |
| **#2 Eviction versus retention at a tier ratio of 3×** | The matched-bytes-read control is step zero of item #2. Item #2 proper needs the BIOS carve-out swept as an independent variable, which is the lab's most defensible contribution and is downstream of this harness. |
| **Open questions 1, 4, 6, 11** | 1 (does KL localise) is the gate. 4 (rank agreement across scale) is the Spearman rider. 6 (SnapKV decomposition) is the natural follow-on once the harness exists. 11 (is our 100× KV-to-weights ratio a confound or a finding) is worth a paragraph in the write-up either way. |
| **Follow-on unlocked** | Learned eviction `[C]` (2602.10238) is downstream of this: you cannot train a policy without a per-decision reward signal, and the per-decision reward signal is exactly what this harness produces. |

## If it works, and if it nulls

**If it works,** the publishable artifact is not "our policy is better" — it is *the
breakpoint methodology*: `b*` reported per salience quintile with a measured null and both
compression numbers, plus the policy-headroom/budget-headroom split that no eviction paper
reports. That is a methods contribution to a field whose own diagnosis is that it has a
measurement problem `[C]` (2510.00231; 2605.08234; 2607.21475).

**If KILL (ii) fires** — no policy separates from random at matched budget at this scale —
that is a genuine negative result about the eviction literature's transferability to small
models, and it is publishable as such. Do not soften it into a "learning opportunity"; state
the decision it forces, which is that Mnemosyne stops carrying policy implementations.

**If KILL (iii) fires** — orderings do not transfer across a 10× scale span — it is the most
consequential outcome in this entire curriculum, because `ablation-scale-sufficient` is the
load-bearing assumption under the whole lab. It would mean every ablation this lab runs is a
curiosity until proven otherwise, and the correct response is to re-plan the backlog around
the inference rig (7–14B off-the-shelf models, which fit the ≥62 GiB fast tier `[M]`) rather
than the training rig.

## Self-check before you start

1. Why does dropping 5% of the attention mass not imply a 5% error in the layer output?
2. Why is the top-`B`-by-true-future-attention oracle a diagnostic ceiling rather than an
   optimum, when Belady's MIN *is* an optimum?
3. ARC learns from its own mistakes with ghost lists. What would a KV ghost list have to
   store, and why does that make it not a ghost list?
4. You measure that policy A beats policy B at a 5% budget. Name three controls without
   which that sentence carries no information.

---
---

# Capstone: `hybrid-ratio-recall-cliff`

**Difficulty:** 4/5 to build, 5/5 on the evidence appraisal. The arithmetic is one sum split
in two; holding "this was ablated" and "the ablation showed the ratio barely mattered" in
one sentence is the skill.
**Calendar:** 10–14 weeks at ~8 h/week. Roughly 70 h of hands plus 120–290 h of unattended
GPU depending on measured throughput — this is the one that needs the machine overnight for
weeks, and the one whose descope ladder you should read before you start.
**Prerequisites:** `attention-variants-and-kv-cost`, `constant-state-memory`,
`hybrid-attention-and-ratios`, `depth-width-and-initialization`, `measuring-recall-and-memory`.
Both prior capstones done.

## The kernel check, done, and what it costs you

**There is no fused SSM kernel on this machine and there is no way to install one.** This was
checked before the capstone was specified, on 2026-07-26, in `C:\venvs\lab`
(`torch 2.12.0a0+rocm7.13.0a20260313`, native Windows, gfx1151):

- `[M]` `torch.utils._triton.has_triton()` → `False`.
- `[M]` `importlib.util.find_spec` → `None` for `triton`, `mamba_ssm`, `causal_conv1d`,
  `fla`, `einops`.
- `[M]` `pip index versions triton` and `pip index versions pytorch-triton-rocm` against
  `https://rocm.nightlies.amd.com/v2/gfx1151/ --pre` → `No matching distribution found` for
  both.

The consequences, stated plainly because each one removes an experiment:

1. **`mamba_ssm`'s Triton SSD path is unavailable**, as is
   `flash-linear-attention`'s entire kernel library. Both are Triton. `mamba_ssm`'s other
   fast path is a CUDA C++ extension, which is worse than unavailable — it is
   architecture-specific to NVIDIA.
2. **The SSM arm must run the pure-PyTorch reference implementation**,
   `architecture/mamba/mamba_ssm/modules/ssd_minimal.py:34`, `ssd_minimal_discrete` — about
   forty lines of einsums with the algorithm's steps numbered. It is faithful, it is
   correct, and it is slow. It also materialises the intra-chunk decay matrix
   `L = torch.exp(segsum(A))` at `:54`, which the fused path does not, so its activation
   memory is `O(chunk_len²)` per chunk where the fused path's is not.
3. **Therefore no throughput comparison between the SSM arm and the attention arms is
   admissible.** Not "is noisy" — *inadmissible*. An unfused reference implementation
   against a vendor-tuned attention path measures the kernels, not the architectures. The
   only comparison this capstone is allowed to make on the SSM axis is **quality at matched
   state bytes**, which is the frame `research/memory/constant-state-memory.md` and
   `[C]` (2402.18668) recommend anyway.
4. `[A]` **`torch.compile` cannot rescue it**, since inductor's GPU backend generates Triton.
   High confidence, untested; the cheapest test is to compile a two-layer block and see
   whether it falls back silently or raises.

**So the primary axis of this capstone is SWA:global, not linear:full.** A sliding window is
a mask — no kernel, no dependency, no risk — and it is the ratio Laguna itself ships `[M]`
(12 `full_attention` + 36 `sliding_attention`, strict GSSS, `sliding_window` 512, read from
the artifact at revision `b0a9fd7c850e`). The SSM arm is a **secondary, matched-state-bytes**
comparison with the throughput disclaimer above stated in its own section of the write-up.

If that reads as a downgrade, weigh it against what the SWA axis buys: it is the axis the
reference model actually uses, it is measurable at 26M in an hour per run, and the two
published readings of the ratio disagree about it.

## The G2 hypothesis card

```
HYPOTHESIS   At ~26M parameters under matched parameter count, matched token budget and
             matched runtime state bytes, multi-query associative recall is a CLIFF in
             the global-layer fraction: recall@1 at 7:1 and 15:1 sliding:global sits at
             least 20 points below 3:1, while validation perplexity moves less than 0.05
             nats across the entire 0:1-to-all-sliding sweep. Quadrupling the token
             budget does NOT close the recall gap.

FOR          Proteus's attention-schedule config surface (layer_types as an explicit
             list, sliding_window, placement at fixed L_g, efficient primitive) on a
             single gfx1151 device.

BECAUSE      (a) Every published hybrid ablation that reported a quality surface
             reported a FLAT one: Gemma 3 swept 1:1 to 7:1 and reported minimal
             perplexity impact [C] (2503.19786); Jamba found 1:3 ~= 1:7; Kimi Linear's
             spread is ~0.01-0.05 PPL across a 15x span of ratios [C] (2510.26692). So
             the shipped ratios were selected on a tie-breaker, usually memory or
             throughput.
             (b) The one metric where the surface is NOT flat is recall: the 72-model
             open sweep finds language modelling stable across ratios but recall
             degrading sharply once full-attention layers thin below 3:1 [C]
             (2507.06457). It is currently the SINGLE source for that claim -- our own
             memory-failure-register cited 2510.26912 as corroboration and that was a
             misreading, corrected in curriculum/hybrid-attention-and-ratios.md §2.5.
             (c) The contest is live and it is about token budget: 2606.15378 argues
             configurations converge given enough training, so the ratio governs how
             FAST long-context ability emerges rather than its ceiling [C]. Two token
             budgets is the cheapest available discriminator between a ceiling and a
             schedule.

MEASURED BY  MQAR recall@1 at a fixed (number of key-value pairs, sequence length),
             primary, against its closed-form chance level, 3 seeds, 95% CIs.
             Secondary and expected flat: validation perplexity. Reported alongside two
             config-derived quantities that cost nothing and that no paper reports: the
             crossover T_x = r*w and the reach-limited fraction of context per layer.

SUCCESS      A cliff located with a >=20-point drop between adjacent swept ratios, CIs
             non-overlapping, reproduced at BOTH token budgets, while perplexity moves
             <0.05 nats across the sweep.

KILL         (i) Both surfaces flat within the seed null at both budgets -- the ratio is
             not a capability knob at this scale, which contradicts 2507.06457's finding
             at 340M/1.3B and is a real (negative) datapoint on whether it is
             scale-local. OR (ii) the recall gap closes at 4x tokens -- 2606.15378 is
             right, the ratio is a training-speed knob, and the whole "3:1 ceiling"
             framing is wrong. Both KILL branches are publishable; neither is a failure.

COST         $0 cloud. ~70 h of hands over 10-14 weeks. GPU: set A (synthetic, 27 runs)
             ~28 h; set B (natural language, 12 runs at two budgets) 93-186 h depending
             on measured throughput. Runs unattended overnight; capstone one's watchdog
             is a hard prerequisite because of the silent-hang mode. Timebox: if set A
             has not located a cliff or a flat by end of week 6, do not start set B.

RISKIEST     ablation-scale-sufficient -- that a ratio conclusion at 26M transfers at
             all. 2507.06457's own result is at 340M/20B and 1.3B/100B; Ring-linear
             ships 4:1 at 16B and 7:1 at 104B from the same fitting procedure [C]
             (2510.19338), which is direct evidence that the ratio is scale-dependent.
             So a null here is ambiguous between "the ratio does not matter" and "26M is
             too small to have formed retrieval heads." Mitigation, and it must be
             pre-registered: measure retrieval-head formation over training steps in
             every arm, so a null can be attributed to absent mechanism rather than
             absent effect.
```

## What it builds

**Proteus's attention-schedule surface, and one honest SSM block.**

1. **`layer_types` as an explicit list, never a modulo.** Samba is the cautionary tale and it
   is in the reference tree: it recomputes `layer_idx % mb_per_layer` in the block
   constructor (`architecture/samba/lit_gpt/model.py:323`) and then makes a *second,
   independent* modulo decision about windowed-versus-global inside the attention module
   (`:452`) — two sources of truth for one property, in two classes, with no shared
   definition. Combined with `full_per_layer` defaulting to 1,000,000, every Samba attention
   layer is windowed and there are zero global layers `[M]`. Samba's own escape hatch is an
   explicit position list at `:321`. Start there.

2. **Four ablation axes, and arm names that carry all four.** The house naming rule makes
   `proteus-swa-3to1` under-specified; the arm is
   `proteus-swa-3to1-w512-densefirst`. The axes are `layer_types`, `sliding_window`,
   placement at fixed `L_g`, and the efficient primitive.

3. **An arm manifest computed from the config, checked before the run starts.** Three
   quantities, all free:
   - **KV residency**, `c · (L_g·T + L_s·min(T,w))` where `c = 2·n_kv·d_h·b`.
   - **The crossover** `T× = r·w`, the context at which the growing term equals the fixed
     term. Below it you are tuning the window; above it you are tuning the ratio. Trivial
     arithmetic, and `hybrid-attention-and-ratios.md` §3.4 could not find it stated in any
     hybrid paper.
   - **The reach-limited fraction.** `k` stacked windowed layers propagate information at
     most `k·(w−1)+1` positions, so at fixed `L_g` the placement determines how much of the
     context is *physically invisible* to the layers below the first global one. Under a
     back-loaded placement at `L=48`, `L_g=12`, `w=512` and a 32,768-token context, the
     bottom 36 layers can reach at most 18,397 positions — the first 14,371 tokens are
     invisible to three quarters of the stack. Not "weakly attended". Invisible. No training
     budget fixes a graph-reachability property.

   **The run fails at startup if matched state bytes differ across arms by more than the
   pre-registered tolerance.** Matching parameters is not matching memory: at Laguna's shape
   an SWA layer holds 2.0 MiB and a Mamba-2 layer at `d_model` 3072 holds 3.0 MiB `[A]`.

4. **The SSM block, from the reference implementation, with its cost stated.** Built against
   `ssd_minimal.py:34`. Its state allocation has **no sequence dimension at all**
   (`architecture/mamba/mamba_ssm/modules/mamba2.py:352`), which is the entire difference
   from the SWA block and the thing worth internalising: an SWA layer's 2 MiB is an *exact*
   record of the last `w` tokens; an SSM layer's state is a *lossy superposition* of
   everything, degrading by interference rather than by eviction. `[M]` The chunk states are
   computed in fp32 regardless of model dtype (`ssd_combined.py:375`,
   `states_in_fp32=True`) — the "constant state" is quietly the most numerically fragile
   part of the layer, which matters directly against `bf16-numerics-unproven`.

5. **A retrieval-head formation probe**, run at fixed step intervals. `[C]` (2606.15378)'s
   Large-Window Laziness result was findable *only* because someone measured **when**
   retrieval heads formed rather than whether the model was good. It is the mitigation for
   this capstone's riskiest assumption and it costs one probe pass per checkpoint.

## What it measures

**Primary: MQAR recall@1** — multi-query associative recall `[C]` (2312.04927), which
isolates the failure the ratio is supposed to cause, at a fixed number of key-value pairs
and a fixed sequence length, reported against its closed-form chance level.

**Never report a hybrid-ratio result on perplexity alone.** A flat perplexity surface is the
*expected* result and carries no information — you will have reproduced the field's
consensus that the ratio does not move perplexity, and learned nothing about whether it
moves capability.

**Secondary:** validation perplexity (expected flat, reported so the flatness is on the
record); retrieval-head formation step per arm; per-layer attention mass beyond `w` on the
global layers — the closest thing to a hit rate that a hybrid admits, and it does not exist
in any implementation read for this curriculum; and the two config-derived manifest
quantities above.

## Arms and controls

**Set A — synthetic, wide, cheap.** Train directly on MQAR-style synthetic sequences,
following the Zoology protocol `[C]` (2312.04927). ~50M tokens per run, ~1 h at `η = 0.10`.

| Arm | `L_g` : `L_s` at `L`=16 | Why |
|---|---|---|
| `proteus-attn-allglobal-ctx2048` | 16 : 0 | the ceiling. Exact everywhere. |
| `proteus-swa-1to1-w512-densefirst` | 8 : 8 | |
| `proteus-swa-3to1-w512-densefirst` | 4 : 12 | the shipped ratio; the predicted cliff edge |
| `proteus-swa-7to1-w512-densefirst` | 2 : 14 | |
| `proteus-swa-15to1-w512-densefirst` | 1 : 15 | |
| `proteus-swa-allsliding-w512` | 0 : 16 | the floor. Recall must collapse here or the probe is broken. |
| `proteus-swa-3to1-w512-middleloaded` | 4 : 12, globals at 6–9 | placement control at **fixed `L_g`** — tests §3.7's receptive-field arithmetic against `[C]` (2510.04800)'s "never front-load" recipe |
| `proteus-swa-3to1-w512-backloaded` | 4 : 12, globals at 12–15 | the reach-limited extreme |
| `proteus-ssd-3to1-statematched` | 4 attention : 12 SSD | secondary axis; matched state bytes, **no throughput claim** |

9 arms × 3 seeds = 27 runs ≈ 28 h.

**Set B — natural language, narrow, expensive.** Only the arms bracketing whatever set A
found, at two token budgets 2× apart (0.5B and 1.0B — the `CLAUDE.md` ablation floor is
0.5B). 2 arms × 3 seeds × 2 budgets = 12 runs.

**Controls that are not arms:**

| Control | Why |
|---|---|
| MQAR chance level, closed form | a recall number without its chance level is decoration |
| seed null, 3 seeds | defines "cliff" quantitatively before you see the sweep |
| matched state bytes, asserted at startup | otherwise "3:1 SWA" and "3:1 SSD" are not comparable, and they differ by ~50% at reference shapes |
| matched parameters and matched tokens | non-negotiable; arms with mismatched budgets are not comparable |
| the `allsliding` floor and the `allglobal` ceiling | if the floor does not collapse and the ceiling does not saturate, the probe is measuring something else |
| window held **constant** across ratio arms | `r` and `w` interact through `T× = r·w`; varying both makes the result uninterpretable |
| retrieval-head formation | distinguishes "no effect" from "mechanism has not emerged at 26M" |

**One experiment you must not run.** Do not widen the window on a *trained* checkpoint to
test long context. Laguna's SWA layers apply plain RoPE over all 128 head dims at θ=10000
while its global layers use YaRN-scaled RoPE over 64 dims at θ=500000
(`architecture/llama-cpp-laguna/src/models/laguna.cpp:184`) `[M]`. The sliding layers were
never trained with a positional encoding that reaches past the window, so you will get
numbers and they will be measuring the encoding failing. The window is a *training-time*
variable.

## Cost arithmetic and the descope ladder

`6·N·D` at `N = 26e6`, and hours at `η` × the `[M]` 20.9 TFLOP/s figure:

| `D` | FLOP | `η`=0.10 | `η`=0.20 | `η`=0.30 |
|---|---|---|---|---|
| 0.05B (set A) | 7.8e15 | 1.04 h | 0.52 h | 0.35 h |
| 0.5B (set B short) | 7.8e16 | 10.4 h | 5.2 h | 3.5 h |
| 1.0B (set B long) | 1.56e17 | 20.7 h | 10.4 h | 6.9 h |

Set A: 27 runs → **28 h** at `η`=0.10. Set B: 6 runs at 0.5B + 6 at 1.0B → **186 h** at
`η`=0.10, **93 h** at `η`=0.20. `[A]` on `η`; run a 200-step pilot in week one.

**Descope ladder, in the order to apply it:**

1. Drop the `proteus-ssd-3to1-statematched` arm. It is the secondary axis and the one with
   the kernel disclaimer.
2. Drop set B's long-budget arms. You keep the cliff, you lose the
   ceiling-versus-schedule discriminator — say so explicitly, because that discriminator is
   the reason the capstone is interesting.
3. Drop set B entirely and report the synthetic result alone, labelled as synthetic.
4. Drop to `L`=12 and 20M parameters.

Do **not** descope by cutting seeds. A single-seed ratio comparison against a literature
whose reported effect sizes are 0.01–0.05 perplexity is an anecdote, and this document's own
`[M]` evidence says the timing instrument on this chassis drifts 24% over twenty minutes.

## Read the code before you start

| Where | What to look at |
|---|---|
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:365` | `self.is_local_attention = config.layer_types[layer_idx] == "sliding_attention"` — the entire hybrid mechanism, as a list lookup at construction. Every question in this capstone is a question about what goes in that list. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:366` | the ternary setting `sliding_window` to 512 or `None`. One line, and it is the difference between `O(T)` and `O(w)` residency. |
| `architecture/llama-cpp-laguna/src/llama-kv-cache-iswa.cpp:69` | `const uint32_t size_base = kv_size;` — the expensive tier, filtered to non-SWA layers. |
| `architecture/llama-cpp-laguna/src/llama-kv-cache-iswa.cpp:73` | `size_swa = GGML_PAD(min(size_base, n_swa·n_seq + n_ubatch), 256)` — the fixed term, allocated. The clearest single line of capacity planning in the reference library. |
| `architecture/llama-cpp-laguna/src/llama-kv-cache-iswa.cpp:76` | `if (swa_full)` — the escape hatch that promotes the cheap tier to the expensive one. There is no inverse switch anywhere. Promotion is always correct; demotion never is. |
| `architecture/llama-cpp-laguna/src/models/laguna.cpp:41` | `hparams.set_swa_pattern(swa_period, /*dense_first=*/true)` — Laguna front-loads, which is exactly what `[C]` (2510.04800) says never to do for Mamba hybrids at 1:12. Your placement arms adjudicate. |
| `architecture/llama-cpp-laguna/src/models/laguna.cpp:184` | `n_rot_l` — where the two tiers stop being numerically interchangeable. |
| `architecture/samba/lit_gpt/model.py:323`, `:452` | the two independent modulos. Read them together; this is the bug shape that makes an ablation unreproducible six weeks later. |
| `architecture/samba/lit_gpt/model.py:321` | the explicit position list. Build from this. |
| `architecture/mamba/mamba_ssm/modules/ssd_minimal.py:34` | `ssd_minimal_discrete` — the pure-PyTorch chunked scan, your only SSM path. Read it before any kernel. |
| `architecture/mamba/mamba_ssm/modules/ssd_minimal.py:54` | `L = torch.exp(segsum(A))` — the intra-chunk decay matrix, `O(chunk_len²)`, which the fused path does not materialise. This is the memory cost of having no Triton. |
| `architecture/mamba/mamba_ssm/modules/mamba2.py:352` | `ssm_state = torch.zeros(batch, nheads, headdim, d_state)` — **no `seqlen` dimension.** Byte-identical at 1K and 1M tokens. Put it beside `llama-kv-cache-iswa.cpp:73`, which does have a token-count term. |
| `architecture/mamba/mamba_ssm/ops/triton/ssd_state_passing.py:80` | `states = scale * states + new_states` — the entire inter-chunk recurrence, and the line where the storage-tier analogy dies. The carry is destructively overwritten, so there is nothing to promote, demote, or fault in. |
| `architecture/flash-linear-attention/fla/ops/gated_delta_rule/naive.py:54` | `h = h.clone() * g[...].exp()[...]` — one scalar decay applied to the entire state matrix. The gate is *indiscriminate* decay; the delta term is the *targeted* erase. The folk story has these backwards. |
| `memory/vllm/vllm/v1/kv_cache_interface.py:539` | `class SlidingWindowSpec(AttentionSpec)` — a windowed layer is a different spec *type*, not a flag. |
| `memory/vllm/vllm/v1/core/kv_cache_coordinator.py:685` | `find_longest_cache_hit` — an explicit fixed-point iteration, because full attention prefix-matches (`single_type_kv_cache_manager.py:708`) and sliding attention suffix-matches (`:918`). A hybrid's prefix-cache hit is the **intersection** across tiers. |
| `memory/vllm/vllm/v1/core/kv_cache_utils.py:1191` | *"we don't know how to implement it cleanly yet"* — the general hybrid is unimplemented in the most-deployed serving engine. |
| `memory/vllm/vllm/v1/core/kv_cache_utils.py:1224` | `min_num_layers = min(...)` with a `FIXME` noting it only works because every open hybrid ships an n:1 pattern. **The architecture's ratio is load-bearing inside the memory allocator.** |
| `memory/vllm/vllm/v1/core/kv_cache_utils.py:1547` | `unify_hybrid_kv_cache_specs` — vLLM's `swa_full`, second implementation of the same one-way escape hatch. |

## Definition of done

- [ ] `layer_types` is an explicit list in config with a test that fails on a modulo
      reimplementation; the arm manifest asserts matched state bytes and refuses to start
      otherwise.
- [ ] The crossover `T×` and the reach-limited fraction are in every arm's manifest,
      computed from config, before the run.
- [ ] The pilot measured `η` and the plan was re-costed against it, in writing, before set A.
- [ ] Set A complete: 9 arms × 3 seeds, MQAR recall@1 with chance level and 95% CIs, and the
      perplexity surface reported alongside so its flatness is on the record.
- [ ] Retrieval-head formation step reported per arm. A null without this is uninterpretable.
- [ ] The `allsliding` floor collapsed and the `allglobal` ceiling saturated. If not, stop
      and fix the probe.
- [ ] Set B complete, or explicitly descoped with the descope logged and the lost inference
      named.
- [ ] The SSM arm's write-up carries the no-fused-kernel disclaimer in its own paragraph, and
      makes **no** throughput claim.
- [ ] `research/memory/hybrid-architectures.md` and
      `curriculum/hybrid-attention-and-ratios.md` updated with whatever this found — the
      curriculum is a living document.
- [ ] `ASSUMPTIONS.md` updated, including the four unregistered rows in this file's envelope.

## Connection to the ablation backlog

| Backlog item | How this capstone relates |
|---|---|
| **#5 An exact cache in front of a lossy state** (P4·T4·E4) | This is the closest live item, and this capstone is its rig. An SWA layer is a *bounded exact* tier; an SSD layer is a *lossy compressive* one; a hybrid is both. Backlog open question 7 — at what exact-cache size does a bounded KV tier in front of a delta-rule state recover full-attention MQAR accuracy — is the direct follow-on and reuses every arm here. |
| **Parked: hybrid ratio selection** (P4·T3·E2) | Parked because the contest is about token budget, which is the expensive axis. This capstone runs the *minimum* two-budget discriminator rather than a sweep, which is why it is affordable. The parked item's un-park trigger — post-hoc conversion `[C]` (2606.30562) proving out — is unaffected. |
| **#3 Scale transfer** | Set A runs at one scale only, so this capstone does **not** discharge the two-scale rider. Say so. Its contribution to `ablation-scale-sufficient` is negative evidence at best: a null at 26M against a positive at 340M `[C]` (2507.06457) would itself be a datapoint that the ratio effect is scale-local. |
| **Contested, and left contested** | Whether the ratio sets a capability ceiling `[C]` (2507.06457) or governs emergence speed `[C]` (2606.15378). This capstone's two-budget design is built to adjudicate it and the write-up must present both readings, not pick one and cite the other in a footnote. |

## If it works, and if it nulls

**If the cliff reproduces at 26M and does not close at 4× tokens,** you have an independent
replication of the single most load-bearing claim in the hybrid literature, at a scale an
order of magnitude below the original, from a lab with one GPU. That is publishable on its
own, and it makes `2507.06457` no longer a single source.

**If the gap closes at 4× tokens,** you have supported `2606.15378`'s reading against
`2507.06457`'s at small scale, which reframes every shipped hybrid ratio as a
training-efficiency decision rather than an architecture decision. Also publishable, and
more interesting.

**If both surfaces are flat,** the result is ambiguous *unless* you measured retrieval-head
formation — which is exactly why that probe is in the definition of done. Flat surfaces plus
formed retrieval heads is evidence the ratio does not matter at this scale. Flat surfaces
plus no retrieval heads is evidence that 26M is below the emergence threshold, which is a
statement about `ablation-scale-sufficient` and belongs in the register.

**In every branch, the placement arms are a separate finding.** The receptive-field
arithmetic in `hybrid-attention-and-ratios.md` §3.7 is not reported in any paper this
curriculum surveyed, `[C]` (2510.04800) says never front-load while Laguna front-loads `[M]`,
and three arms at fixed `L_g` adjudicate it for the cost of nine runs.

## Self-check before you start

1. A "3:1 hybrid" saves how much KV memory asymptotically, and why is the answer not 3?
2. At `w` = 512 and a 2,048-token training context, how much does residency actually differ
   between a 3:1 arm and a 7:1 arm, and what does that tell you about scoring a ratio
   ablation on memory?
3. Why can you not test long-context behaviour by widening the window of a trained
   checkpoint — and give two independent reasons?
4. You have no fused SSM kernel. Which comparisons remain valid and which are now
   inadmissible?

---
---

## Sequencing, and what all three cannot do

**Calendar, assuming ~8 h/week of hands and the machine free overnight:**

| Weeks | Work |
|---|---|
| 1–7 | `telemetry-detects-injected-fault`. Weeks 1–3 are the clean-run null; do not skip ahead. |
| 8–17 | `eviction-recall-breakpoint`. Week 8 is the ADR gate — an afternoon that can change the plan. |
| 18–31 | `hybrid-ratio-recall-cliff`. Week 18 is the throughput pilot; weeks 19–24 are set A; set B runs overnight through the end. |

That is roughly seven months, which is honest rather than encouraging. The 12-week
curriculum schedule covers the *modules*; the capstones are what comes after, and two of the
three are gated on the Hardware Validation Gate, which has not run.

**What all three cannot do, stated once so it is not rediscovered three times:**

- **Nothing multi-device.** Not FSDP, not DDP, not tensor or pipeline or expert parallel, not
  sharded checkpointing, not DTensor. `[M]` These fail at *import*, which is cleaner than
  failing at runtime: design-only work cannot accidentally half-run and produce a plausible
  number.
- **Nothing that needs a fused SSM or linear-attention kernel.** `[M]` No Triton, no wheel to
  install.
- **Nothing whose conclusion depends on bf16 arithmetic** until the Hardware Validation Gate
  runs. `bf16-numerics-unproven` is `untested`, five bf16 bugs are documented on gfx1151
  `[C]`, and `[M]` hipBLASLt configuration alone moves long-reduction relative error by 2.8×.
  Every arm records `hipblaslt_configured` and `aotriton_experimental` or its numbers are
  uninterpretable.
- **Nothing at instruction-following scale.** The most *painful* problems in the field —
  instruction dropping under compression `[C]` (2510.00231), alignment collapse under KV
  quantization `[C]` (2606.09864) — need capability a 300M model does not have. Those live
  on the second rig: inference-only studies on an off-the-shelf 7–14B model, which fits the
  `[M]` ≥62 GiB fast tier with room for a large cache. That rig is not covered here and is
  the obvious fourth capstone.
- **Nothing with a single tensor at or above 32 GiB.** `[M]` It hangs at 0% CPU with no
  error. Every capstone above allocates per layer and runs the watchdog.

**And the one honest caveat about the whole set.** Three of the numbers these designs lean on
hardest are weaker than they look: the ≥62 GiB fast tier is **one run per arm**; the ~104
FLOP/byte ridge is a ratio of two single-run numbers of different kinds, neither of them an
attention kernel; and the 32 GiB fault rests on two observations with an untested mechanism
(`[A]` medium confidence it is a 32-bit overflow — the cheapest discriminating test is a
32 GiB fp32 allocation: same bytes, half the elements). If any of those moves, re-cost the
capstone that depends on it before continuing rather than after.

---

## Answers to the self-checks

### `telemetry-detects-injected-fault`

**1.** Because the logged scalar is `(1/N) Σ ℓ_i` over `N` tokens and per-token surprise is
bounded above by `ln V` for a vocabulary of size `V`. One document's worst possible
contribution is therefore capped, and its share of the mean falls as `1/N_docs`. At 512
documents the arithmetic gives a bound below 1σ of the mean's own step-to-step variation:
there is no threshold that both fires on this and does not fire constantly. The grouping that
would let you attribute it was destroyed by the reduction before the number existed. The fix
is not a better threshold, it is a second statistic — `max` instead of `mean` — computed from
a tensor you already have.

**2.** `[M]` The collated drain costs ~40 µs fixed plus ~1.29 µs per series: 2,796 series is
~3.6 ms per drain, every 5 steps, so ~0.72 ms per step. The per-`.item()` path is ~38 µs per
series with no meaningful fixed cost: ~106 ms per drain, ~21 ms per step. Draining every step
multiplies both by five — 3.6 ms/step collated, 106 ms/step per-item. That is the point of the
two knobs: the **drain** interval controls throughput and loses nothing (callbacks still see
every step, merely delayed), while the **sampling** interval controls storage and compute and
does lose information. Almost everyone sets them as if they were the same knob.

**3.** Because the fault being detected is a hang *inside* the training process at 0% CPU.
Nothing in that process will schedule again — no timer, no signal handler you registered, no
`finally` block. `[M]` The observed behaviour is 11 minutes at 0 CPU with host free RAM
falling to 5 GB before a force-kill. A watchdog that lives in the process is a watchdog that
hangs with it. This is the same reason your liveness probe does not run inside the container's
main thread.

**4.** A production APM tool would catch `fault-scheduler-restart` (an emitted config value
changed) and `fault-dataloader-cursor-reset` (a monotonic counter went backwards) — both are
ordinary metric anomalies. It would structurally miss `fault-poisoned-document`, because the
aggregation that hides it happens *before* emission and APM cannot un-aggregate; it would miss
`fault-hipblaslt-unset`, because the effect is a change in floating-point reduction accuracy
with no counter attached anywhere in the stack; and it would miss `fault-silent-hang`, because
the process is alive, holding memory, and simply not progressing — which most liveness checks
score as healthy. Three of five. That ratio is the argument for the capstone.

### `eviction-recall-breakpoint`

**1.** Because the softmax renormalises. Dropping entries removes their terms from the
denominator, so every surviving token's weight is multiplied by `1/A` where `A` is the
retained mass. The exact identity is `o_t − o_S = Σ_{i∉S} a_i (v_i − o_S)`: the error is a
product of *how much mass you dropped* and *how far the dropped values sit from the retained
average*. On the worked example in `kv-eviction-policies.md` §3.4, dropping 5% of the mass
moves the output by 122% of its own magnitude when the dropped token carries a value outlier,
and by 2.6% when it does not — same policy, same budget, same dropped mass, 22× different
error. No cache you have run has this property: dropping a line does not change the value of
the other lines.

**2.** Belady's MIN is optimal because cache misses are **additive and independent** — the
cost of missing on `i` does not depend on whether you also missed on `j`. The eviction error is
neither. `o_S` appears inside its own error term, so dropping token `i` changes the cost of
dropping token `j`. Top-`B` by true future attention is therefore the right *diagnostic
ceiling* — it tells you how much of your gap is policy and how much is budget — and it is
demonstrably not the optimum. H2O's own formulation concedes the shape of this by casting
selection as dynamic submodular maximisation with a guarantee holding only "under mild
assumptions" `[C]` (2306.14048).

**3.** A KV ghost list would have to answer "would the current query have attended strongly to
the entry I dropped?", and answering that requires `q_t · k_i`, which requires `k_i` — the
thing you deleted. There is no cheap metadata, because the metadata that answers the question
*is* the key. Keep the keys and drop only the values and you have a ghost list; you have also
re-invented sparse retention (Quest's per-page key bounds `[C]` 2406.10774, SparQ's key-channel
subset), and the price is that you never get the capacity back. That is why RocketKV's abstract
carries "400× compression" and "32.6% peak memory reduction" in the same breath `[C]`
(2502.14051): most of the 400× is bandwidth, not bytes held.

**4.** Any three of: (a) a matched-budget **random** control, without which you do not know the
comparison is about the policies rather than about the budget; (b) a **seed null**, without
which "beats" has no width; (c) the **structureless null**, without which the ranking may be an
artifact of top-`k` selection rather than a fact about language models; (d) **both compression
numbers** — bytes held and bytes read — without which "at a 5% budget" is ambiguous between two
different experiments; (e) the **full-cache oracle**, without which you have no counterfactual;
(f) **salience stratification of the target**, without which the aggregate is a weighted average
over a distribution you chose when you built the haystack.

### `hybrid-ratio-recall-cliff`

**1.** Asymptotically the saving is the *period* of the pattern, `L / L_g = r + 1`, not `r`.
For Laguna, 48/12 = **4×**. The naming convention counts the cheap layers; the saving counts
all of them. If you quote 3× you are wrong by a third, in the same direction, every time — and
this error appears in published blog posts.

**2.** At `L`=48, `w`=512, `T`=2048: the 3:1 arm holds `12·2048 + 36·512 = 43,008` token-units
and the 7:1 arm holds `6·2048 + 42·512 = 33,792`. That is a **1.27×** byte difference across a
ratio change of 2.3×, because at `T` = 2048 you are near the crossover `T× = r·w` (1,536 for
3:1, 3,584 for 7:1) where the fixed window term still dominates. Consequence: **a ratio
ablation scored on residency or on decode wall-clock is nearly blind at our context lengths.**
If the ablation is about memory you need `T ≫ r·w`, which at `w`=512 and `r`=7 means `T ≳ 32k`.
If it is about recall — which is what the evidence says it should be — context length is set by
the probe, and the memory argument is simply not why you are running it.

**3.** First, the **positional encoding**: `[M]` Laguna's SWA layers use plain RoPE over all 128
head dims at θ=10000, its global layers YaRN-scaled RoPE over 64 dims at θ=500000. The sliding
layers were never trained with an encoding that reaches past the window, so a widened window
puts them in a positional regime they have no representation for and you will measure the
encoding failing. Second, and independently, the **window is a training-time variable**: both
`[C]` (2509.24552) and `[C]` (2606.15378) report that window size shapes the optimisation
trajectory — short windows force the model to train its long-range machinery, long windows delay
retrieval-head formation. Whatever widening does to a finished checkpoint, it cannot tell you
what training with that window would have produced. Fixing the RoPE would not fix the second
reason.

**4.** Still valid: **quality at matched parameters, matched tokens and matched state bytes** —
MQAR recall, perplexity, retrieval-head formation step, and every config-derived quantity
(residency, `T×`, reach-limited fraction). Still valid within the attention arms: wall-clock
and bandwidth comparisons between SWA ratios, since they run the same kernel. Now inadmissible:
any throughput, latency, tokens/s, MFU or energy comparison **between** the SSM arm and any
attention arm, because an unfused reference implementation against a vendor-tuned attention
path measures the kernels and not the architectures. Also inadmissible: any claim about the
SSM arm's *memory* advantage at decode that is inferred from a training-time measurement, since
the pure-PyTorch chunked scan materialises an `O(chunk_len²)` intra-chunk decay matrix
(`ssd_minimal.py:54`) that the fused path does not.

---

## Sources

**Local measurements (`[M]`)**

`ASSUMPTIONS.md` rows: `gpu-fast-tier-size`, `large-tensor-fault-32gib`,
`hardware-capacity-ceiling`, `single-device-only`, `hipblaslt-config`,
`bf16-reduced-precision-knob-works`, `gemm-throughput-below-reference`,
`sdpa-is-memory-efficient`, `bf16-numerics-unproven`, `ablation-scale-sufficient`,
`reference-model`, `kv-per-token-laguna`, `laguna-heads-uniform`, `torch-build`,
`mnemosyne-separable`, `z13-is-right-instrument` (all 2026-07-24 to 2026-07-26) ·
`ENVIRONMENT.md` 2026-07-26 (32 GB host RAM after the carve-out; hipBLASLt path configured) ·
`notebook/uma-carveout-controls-fast-tier.md` (single run per arm) ·
`curriculum/kv-eviction-policies.md` §3.4, §3.6, §3.7, §3.9, §3.10 ·
`curriculum/hybrid-attention-and-ratios.md` §2.5, §3.2–§3.7, §6 (Exercise B) ·
`curriculum/training-telemetry-as-observability.md` §2.6, §3.4 ·
`curriculum/constant-state-memory.md` §2.4, §2.5 · `curriculum/measuring-memory.md` §2.6 ·
`docs/adr/attribution-instrument-over-eviction-policy.md`,
`docs/adr/aotriton-attention-stays-off-by-default.md`,
`docs/adr/hipblaslt-is-a-numerics-control.md`, `docs/adr/bios-uma-carveout-at-96gb.md`.

Four measurements taken while writing this file, 2026-07-26, `C:\venvs\lab`,
`torch 2.12.0a0+rocm7.13.0a20260313`, native Windows, gfx1151 — **not yet rows in
`ASSUMPTIONS.md`**: `has_triton()` False; `find_spec` None for triton / mamba_ssm /
causal_conv1d / fla / einops; no `triton` or `pytorch-triton-rocm` distribution on the
gfx1151 nightly index; `torch.distributed.checkpoint` and `torch.distributed.tensor` both
raise `ModuleNotFoundError: No module named 'torch._C._distributed_c10d'`.

**Repo documents these capstones execute**

`research/memory/open-problems-ranked.md` (the ablation backlog — items #1–#6, the parked
list, and open questions 1–11) · `research/synthesis.md` (the MECE issue tree and the
five questions worth our compute) · `research/memory/memory-failure-register.md` ·
`research/reference/CODE_MAP.md` · `research/reference/PROVENANCE.md` (the revisions every
`file:line` above is pinned to) · `OPERATING_INSTRUCTIONS.md` (G2 card format, G3 research
translation, override protocol) · `CLAUDE.md` (naming, document classes, experimental
standards).

**arXiv (`[C]`)**

- `1911.02150` — *Fast Transformer Decoding: One Write-Head is All You Need* (2019).
- `2203.03466` — *Tensor Programs V* / muP (2022).
- `2306.14048` — *H2O: Heavy-Hitter Oracle for Efficient Generative Inference* (2023).
- `2307.03172` — *Lost in the Middle* (2023).
- `2309.17453` — *Efficient Streaming Language Models with Attention Sinks* (2023).
- `2312.00752` — *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (2023).
- `2312.04927` — *Zoology: Measuring and Improving Recall in Efficient Language Models* (2023).
- `2402.18668` — *Simple linear attention language models balance the recall-throughput tradeoff* (2024).
- `2404.06654` — *RULER: What's the Real Context Size of Your Long-Context Language Models?* (2024).
- `2404.14469` — *SnapKV: LLM Knows What You are Looking for Before Generation* (2024).
- `2404.15574` — retrieval-head masking, cited for the mechanism-ablation fault in `curriculum/measuring-memory.md` §2.6.
- `2405.21060` — *Transformers are SSMs* (2024).
- `2406.02069` — *PyramidKV* (2024).
- `2406.10229` — cited in `curriculum/training-telemetry-as-observability.md` §2.8 for seed variance at small scale.
- `2406.10774` — *Quest* (2024).
- `2406.11430` — L2-norm-based KV compression (2024).
- `2407.11550` — *Ada-KV* (2024).
- `2412.06464` — *Gated Delta Networks* (2024).
- `2502.05167` — *NoLiMa* (2025).
- `2502.14051` — *RocketKV* (2025).
- `2503.19786` — *Gemma 3 Technical Report* (2025).
- `2504.15364` — *KeyDiff* (2025).
- `2504.19561` — *Quantifying Memory Utilization with Effective State-Size* (2025).
- `2507.06457` — *A Systematic Analysis of Hybrid Linear Attention* (2025, rev. Jun 2026). The 72-model sweep; recall collapse below 3:1.
- `2508.07479` — *Positional Biases Shift as Inputs Approach Context Window Limits* (2025).
- `2509.24552` — *Short window attention enables long-term memorization* (2025, rev. May 2026).
- `2510.00231` — *The Pitfalls of KV Cache Compression* (2025).
- `2510.04800` — *Hybrid Architectures for Language Models: Systematic Analysis and Design Insights* (2025, rev. Apr 2026).
- `2510.13334` — *Taming the Fragility of KV Cache Eviction in LLM Inference* (2025).
- `2510.19338` — *Every Attention Matters* (Ring-linear, 2025). 4:1 at 16B, 7:1 at 104B.
- `2510.26692` — *Kimi Linear: An Expressive, Efficient Attention Architecture* (2025).
- `2510.26912` — *Understanding and Enhancing Mamba-Transformer Hybrids for Memory Recall and Language Modeling* (2025). Sequential versus parallel fusion — **not** a ratio-cliff corroboration; see the correction in `curriculum/hybrid-attention-and-ratios.md` §2.5.
- `2602.10238` — *Learning to Evict from Key-Value Cache* (Feb 2026).
- `2603.20397` — *KV Cache Optimization Strategies for Scalable and Efficient LLM Inference* (Mar 2026). No method dominates.
- `2605.08234` — *When Does Value-Aware KV Eviction Help? A Fixed-Contract Diagnostic* (May 2026).
- `2606.09864` — *Alignment Collapse Under KV Cache Quantization* (Jun 2026).
- `2606.15378` — *Rethinking the Role of Efficient Attention in Hybrid Architectures* (Jun 2026). Large-Window Laziness; the convergence framing.
- `2606.30562` — *Morphing into Hybrid Attention Models* (FlashMorph, Jun 2026).
- `2607.02574` — *From Tensor Buffer to Distributed Memory Hierarchy* (Jun 2026).
- `2607.05061` — *KVpop* (Jul 2026).
- `2607.21475` — *Error Certificates for KV-Cache Eviction via Randomized Design* (Jul 2026).
