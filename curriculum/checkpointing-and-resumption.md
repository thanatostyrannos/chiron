---
title: Checkpointing and resumption â€” DR for a system with no partial restore and no integrity check
version: 1.0.0
date: 2026-07-26
owner: curriculum-author
track: D â€” Training systems
prereqs: the-training-loop, loss-and-optimization
difficulty: 2/5 conceptually (this is your home turf), 4/5 in the two places the analogy breaks
time: ~1.5 h reading, ~2.5 h exercises (Exercise A is the one that matters; B and C are 45 min each)
---

# Checkpointing and resumption

## 1. What this module settles

You can already write a runbook for a system whose recovery objective is stated in minutes;
what this module settles is exactly which parts of that runbook survive contact with a
training run and which parts silently do not â€” specifically, that the recovery unit is a
**five-part consistency group** (parameters, optimizer moments, step count, RNG state,
dataloader position) rather than a file, that **resume-equivalence is a two-sided test
against a measured noise floor** rather than a bit-exactness assertion, and that the
optimal checkpoint interval has a closed form you can evaluate for your own machine in
twenty minutes. It also settles three hardware facts you will not find written down: on this
stack, `torch.load` **does not verify the CRC-32 that its own container already stores**, so
a single flipped bit restores as a plausible weight with a norm ratio of 0.99999994 `[M]`;
a checkpoint carrying model + optimizer but not RNG state produces a resumed run in which
**100% of parameters differ** from the uninterrupted reference `[M]`, while the loss curve
looks entirely normal; and the RNG fix that fully works on CPU â€” Python, NumPy, torch â€” is
**still wrong on gfx1151**, because dropout draws from a fourth generator, leaving a residual
of 5e-3 across 99.998% of parameters until `torch.cuda.get_rng_state()` is added, at which
point the resume is bit-exact `[M]`. Everything about *sharded* checkpoint formats is taught
as design here and cannot be run on this machine â€” `ASSUMPTIONS.md: single-device-only` â€”
and that limit is stated rather than papered over.

---

## 2. Theory in plain language

### 2.1 What a training checkpoint actually is

A checkpoint is a serialized copy of everything the training process needs in order to be
*indistinguishable* from a process that was never interrupted. That is the whole definition,
and the word doing the work is **indistinguishable**, not *functional*.

The distinction matters because there are two very different jobs people call
"checkpointing":

- **Publishing a model.** You need parameters. Nothing else. This is what a `.safetensors`
  release is, and it is what `model.state_dict()` gives you.
- **Continuing a run.** You need parameters *and* every other piece of state that the next
  step reads. Miss one and the run continues â€” it just continues onto a different
  trajectory, forever, with no error.

Almost every trainer conflates them, and the conflation is invisible because both produce a
file that loads.

### 2.2 What it replaced â€” three eras of "what gets saved"

Nothing was replaced, exactly; models used to train in minutes and nobody needed DR. What
changed is the *scope* of the saved object, in three steps:

1. **Save the weights.** Enough to use the model. Not enough to continue it: restart and
   AdamW's momentum starts from zero, which is a different optimizer than the one you were
   running.
2. **Save the weights and the optimizer.** Enough to continue. Not enough to *reproduce* â€”
   the RNG streams restart, so dropout masks and data order diverge from step one.
   **This is where most shipping code still is, including nanoGPT** (`train.py:277`).
3. **Save the run.** Weights, optimizer, step count, all RNG streams, dataloader position,
   and every other stateful component (loss scaler, EMA, schedulers, callbacks). Resume
   becomes a no-op. OLMo-core is here (`trainer.py:87`).

A fourth axis runs orthogonal to those three: **consolidated â†’ sharded**, forced once
`12 Ã— parameter_count` bytes stopped fitting in one machine's memory. That is a format
change, not a scope change, and it is covered in Â§3.5.

### 2.3 The bridges, and where each one breaks

This is the section to read slowly. The left column is yours. The right column is the whole
module.

| You already own | Counterpart here | Where the analogy breaks |
|---|---|---|
| Backup and restore of a database | Checkpoint and resume | The database has a schema, constraints, and page checksums, and a bad restore fails loudly. This artifact is a bag of float arrays whose only enforced invariant is "the shapes match." A restore that produces the *wrong model* is byte-for-byte as valid as one that produces the right one. |
| **RPO** â€” how much data you accept losing | How many steps you accept re-running | No data is lost. The tokens are still on disk. What you lose is **work**, so RPO here is pure economics with a closed-form optimum (Â§3.3) â€” not something you negotiate with a business owner. |
| **RTO** â€” time until service is restored | Time until the run is back *on the same trajectory* | The RTO clock does not stop when the process comes up. If the checkpoint was incomplete, it never stops, and nothing reports that. And the dominant term is not restore time â€” it is **detection** time, which for a one-person lab is "how long until you next look." |
| A torn write, caught by the page checksum | A torn checkpoint | Two different failures wear one name. **Truncation is caught** `[M]`: the zip central directory is written last, so a half-written `.pt` fails to load. **A bit flip inside an intact container is not caught** `[M]`: the CRC-32 is right there in the file and `torch.load` never reads it (Â§6, Exercise B). |
| Partial restore â€” recover the one damaged tablespace, replay the log forward | â€” | There is no analogue and no log. The five components are a single consistency group; there is no meaningful "restore layer 7 from an older copy." Â§6 Exercise B part 3 measures what splicing actually costs, and the answer is more interesting than the folklore. |
| Incremental / differential backup | â€” | Every save is a full save. No mainstream trainer ships a delta format, because AdamW's second moment changes in every element on every step. Research exists (Â§8); production does not. |
| GFS retention (grandfather / father / son) | permanent vs. ephemeral checkpoints | **This one holds cleanly.** `save_interval=250` steps for permanent, `max_checkpoints=3`, plus one rolling ephemeral for preemptible jobs (`callbacks/checkpointer.py:63`, `:102`, `:313`). Same rotation you have configured a hundred times. |
| A crash-consistent snapshot across a consistency group | The five state components | **Holds, and it is the right frame.** They must all be from the same instant. Async checkpointing is exactly the "quiesce vs. copy-on-write" problem you already know. |
| Schema migration on restore | `state_dict` key drift | Breaks badly. A missing column is an error. A missing key under `strict=False` is a **return value that nobody reads** â€” the layer keeps its random initialization and the run proceeds `[M]`. |
| Resilvering a replica / rebalancing shards after a node change | Resharding a checkpoint across a different world size | Holds *iff* the on-disk keys are logical. Vanilla PyTorch keys optimizer state by **integer index into param groups**, which is a function of construction order; DCP re-keys by parameter name. Â§3.5. |
| A DR drill you run quarterly against a stated pass criterion | Resume-equivalence in CI | Your DR drill's criterion is "the app serves reads." This one has no criterion until you have measured the **noise floor** â€” what two uninterrupted, identically-seeded runs do to each other. Without that number the test is either vacuous or impossible to pass. Â§3.4. |

Two of those rows are the reason this module exists. **There is no partial restore**, and
**a torn checkpoint is silently wrong rather than obviously corrupt.** Everything else is
your existing runbook with the units changed.

---

## 3. The math that actually matters

### 3.1 Checkpoint size, exactly

From `the-training-loop.md` Â§3.4 you already have the persistent-state inventory. Subtract
the one thing that is *not* saved â€” gradients, because they are recomputed on the next
backward â€” and you get the checkpoint:

```
S  =  P Â· (b_Î¸ + b_m + b_v)
```

- `S` â€” checkpoint size in bytes
- `P` â€” number of parameters
- `b_Î¸` â€” bytes per parameter for the saved master weights (4 for fp32)
- `b_m` â€” bytes per parameter for AdamW's first moment `m` (4 for fp32)
- `b_v` â€” bytes per parameter for AdamW's second moment `v` (4 for fp32)

For the standard fp32-master AdamW recipe that is **12 bytes per parameter**, and it is
exact rather than approximate. `[M]` Measured, three model sizes, `torch
2.12.0a0+rocm7.13.0a20260313`, fresh process, 2026-07-26: 12.0027, 12.0007, and 12.0002
bytes per parameter at 2.1 M, 8.4 M and 33.6 M parameters. The excess over 12.0000 is the
pickle header and the per-tensor zip entries, and it amortizes away exactly as you would
expect.

Note what is *not* in there. Under `torch.autocast` there is no persistent bf16 weight copy
to save â€” the cast is transient. If your recipe keeps an explicit bf16 shard you are at 14
B/param; with 8-bit optimizer states you are at 6.

| `P` | checkpoint bytes at 12 B/param |
|---|---|
| 20 M (smallest ablation arm) | 240 MB |
| 124 M (GPT-2 small) | 1.49 GB |
| 300 M (largest planned arm) | 3.60 GB |
| 1 B | 12.0 GB |
| 8.5 B (Laguna active params) | 102 GB |
| 118 B (Laguna total params) | **1.42 TB** |

The last row is the one that reorganizes the field: a frontier MoE checkpoint is a
terabyte-scale object written every few hundred steps, and almost all of it is expert
weights that were touched by a small fraction of the batch. That is the entire motivation
for the MoE-specific checkpoint work in Â§8.

**And a Z13-specific consequence that is easy to miss.** `ENVIRONMENT.md` reports **32 GB of
system RAM** visible to the host, because 96 GB of the 128 GB is carved out to the GPU
(`notebook/uma-carveout-controls-fast-tier.md`). Asynchronous checkpointing works by
de-staging the state dict to *host* memory and writing it out in the background
(`training/olmo-core/src/olmo_core/distributed/checkpoint/__init__.py:275`, and `_prepare_state_dict` sets `cpu_offload=True`
at `:711`). On this machine the staging buffer lives in the **small** pool. A 300 M-param
arm needs 3.6 GB of host RAM to stage â€” fine. A 2 B-param model would need 24 GB of a 32 GB
host pool, next to everything else running. The usual assumption that host memory is the
cheap tier is inverted here by our own BIOS setting.

### 3.2 RPO and RTO, in the units that actually apply

Define them precisely, because the standard definitions do not transfer cleanly.

**RPO.** With a checkpoint every `k` steps and a step time of `t_step` seconds, a crash
arriving uniformly at random inside an interval loses

```
E[lost work]  =  (k / 2) Â· t_step        seconds of compute
```

- `k` â€” checkpoint interval, in optimizer steps
- `t_step` â€” wall-clock seconds per optimizer step
- the factor `1/2` â€” the expectation of a uniform arrival inside the interval

Note the unit. It is **seconds of GPU time**, not records. Nothing is unrecoverable; you are
buying back electricity and wall-clock.

**RTO.** Decompose it, because only one term is what you think it is:

```
RTO  =  T_detect  +  T_restart  +  T_load  +  T_warmup
```

- `T_detect` â€” from failure to somebody or something noticing
- `T_restart` â€” process launch, imports, model construction, device init
- `T_load` â€” reading `S` bytes and materializing the state
- `T_warmup` â€” re-JIT, autotune cache misses, allocator warm-up before steady-state step time

`T_load` is the term the literature optimizes and it is the smallest one here. `[M]` Load of
a 403 MB checkpoint took 0.066 s median of three (CPU, page-cache warm) against 0.387 s to
write it; loads are cheaper than saves because the write must actually reach the device.
Extrapolating linearly to a 300 M-param arm gives ~0.6 s to read 3.6 GB. That is noise.

`T_detect` is the term nobody models and it dominates for a one-person lab. On this machine
specifically the worst failure mode we have measured is silent: `ASSUMPTIONS.md:
large-tensor-fault-32gib` records `[M]` a 32 GiB allocation hanging **at 0% CPU with no
error** for eleven minutes before being force-killed. A job in that state is not crashed. It
has no exit code, it emits no log line, and if it happens at 02:00 your `T_detect` is seven
hours. Every RTO number in the checkpointing literature assumes instantaneous detection
because it assumes an orchestrator with a health check. **On this rig the cheapest available
DR improvement is not faster checkpoints; it is a step-progress watchdog**, which is one
JSONL heartbeat and one timer.

### 3.3 The optimal checkpoint interval, derived

This is the one piece of genuine math in the module, it is fifty years old, and it is
directly usable.

Let:

- `Î´` (delta) â€” the wall-clock cost of writing one checkpoint, in seconds
- `M` â€” mean time between failures for this job, in seconds
- `R` â€” restart cost after a failure (the `RTO` above minus the lost work), in seconds
- `Ï„` (tau) â€” the checkpoint interval expressed in seconds of *useful* compute

Per second of useful work you pay two overheads.

**Checkpoint overhead.** You write one checkpoint per `Ï„` seconds of useful work, each
costing `Î´`, so the fraction is `Î´ / Ï„`.

**Failure overhead.** Failures arrive at rate `1/M`. Each costs the restart `R` plus, in
expectation, half an interval of redone work, `Ï„/2`. So the fraction is `(Ï„/2 + R) / M`.

```
waste(Ï„)  =  Î´/Ï„  +  (Ï„/2 + R)/M
```

Differentiate with respect to `Ï„` and set to zero. `R` and `M` are constants, so:

```
d/dÏ„ [ Î´/Ï„ + Ï„/(2M) ]  =  âˆ’Î´/Ï„Â²  +  1/(2M)  =  0
```

```
Ï„Â²  =  2 Î´ M                    â‡’        Ï„*  =  âˆš(2 Î´ M)
```

**The optimal checkpoint interval is `âˆš2` times the geometric mean of the checkpoint cost
and the mean time between failures.** That is Young's 1974 result `[C]` (J. W. Young, "A
first order approximation to the optimum checkpoint interval", *CACM* 17(9), 1974);
Daly's refinement `[C]` (J. T. Daly, "A higher order estimate of the optimum checkpoint
interval for restart dumps", *Future Generation Computer Systems* 22(3), 2006) gives
`Ï„* = âˆš(2Î´(M+R)) âˆ’ Î´` for `Î´ â‰ª M` and a different form when `Î´` is comparable to `M`. At our
scale the first-order version is more than accurate enough.

Substituting `Ï„*` back gives the minimum achievable overhead:

```
waste(Ï„*)  =  âˆš(2Î´/M)  +  R/M
```

**Now the property that makes this practically useful.** Write `Ï„ = k Â· Ï„*` for a
misestimate factor `k`. Substituting and simplifying:

```
waste(k Ï„*) / waste(Ï„*)  =  (k + 1/k) / 2          (ignoring the R/M floor)
```

This is symmetric in `k` and `1/k`, and it is *flat*. Being 2Ã— too frequent and 2Ã— too
infrequent cost exactly the same, and both cost only 25% more overhead than optimal. You
have to be 10Ã— off before you pay 5Ã—. **So do not tune this. Get within a factor of three
and move on** â€” which is precisely the opposite of the effort the literature spends on it,
and worth knowing before you spend an afternoon on it.

**Worked for this machine.** Take a 300 M-param arm: `S = 3.6 GB`, and `[M]` measured save
throughput of 992 MB/s at the largest size tested (403 MB; it fell from 1675 MB/s at 25 MB,
so treat ~1 GB/s as the honest figure and expect it to drop further at 3.6 GB). Add the
device-to-host copy, which on unified memory is a memcpy inside the same physical DRAM and
is **not measured** â€” call the total `Î´ â‰ˆ 5 s`.

We do not know `M` for the Z13. So invert the formula, which is the more useful direction:

```
M  =  Ï„*Â² / (2Î´)
```

With `Î´ = 5 s`, and reading the overhead column as `waste(Ï„*) = âˆš(2Î´/M)`, which splits exactly
in half between the two terms (`Î´/Ï„*` and `Ï„*/2M` are equal at the optimum â€” verify that, it is
a good check on the algebra):

| If you checkpoint everyâ€¦ | â€¦it is optimal when the machine fails everyâ€¦ | overhead at that point |
|---|---|---|
| 60 s | 6 min | 16.7% |
| 300 s (5 min) | 2.5 h | 3.3% |
| 900 s (15 min) | 22.5 h | 1.1% |
| 3600 s (1 h) | 15 days | 0.28% |

Read the table as a bet on `M`. If you believe this laptop runs about a day between
interruptions â€” which `ASSUMPTIONS.md` gives no basis for either way â€” checkpoint every
fifteen minutes and the overhead is about one percent. **The honest conclusion for our
hardware is that checkpoint cost is irrelevant at our scale and the whole optimization
literature is aimed at a regime we are not in** (Î´ in minutes, M in hours, thousands of
GPUs). What binds here is disk space and `T_detect`.

Note the shape of the table: the top row is the trap. Checkpointing every minute *feels*
prudent and costs 17% of your throughput unless the machine really is failing every six
minutes â€” which is `Î´/Ï„ = 5/60` of pure write overhead before a single failure occurs.

**For contrast, the regime the literature targets.** `[C]` The Llama 3 paper (arXiv
2407.21783, Jul 2024) reports 466 job interruptions in a 54-day pre-training window on
16,384 H100s â€” 419 of them unexpected â€” which is `M â‰ˆ 54Â·86400/466 â‰ˆ 10,000 s`, about one
failure every 2.8 hours. With a terabyte-scale checkpoint at `Î´ â‰ˆ 60 s`, `Ï„* = âˆš(2Â·60Â·10000)
= 1095 s`, i.e. roughly every 18 minutes, at an overhead of `âˆš(2Â·60/10000) = 11%` before
counting restart. Eleven percent of a 16,384-GPU cluster is the entire budget of a small
lab, which is why that overhead is worth a research programme and ours is not.

### 3.4 Resume-equivalence as a testable property

Here is the formal statement, because "the checkpoint works" is not one.

**The run state at step `t`:**

```
Ïƒ_t  =  ( Î¸_t , m_t , v_t , t , Ï_t , c_t , s_t )
```

- `Î¸_t` â€” the parameter vector
- `m_t`, `v_t` â€” AdamW's first and second moment estimates, one element per parameter
- `t` â€” the step counter
- `Ï_t` â€” the RNG state, which is a *vector* of independent streams: Python's `random`,
  NumPy's, torch's CPU generator, and torch's CUDA generator
- `c_t` â€” the dataloader position
- `s_t` â€” everything else that carries state: the gradient scaler's current scale and
  growth counter, EMA/model-averaging shadows, any scheduler with internal state, callback
  counters, curriculum position

**The step map** is `Ïƒ_{t+1} = Î¦(Ïƒ_t)`. It is a pure function of `Ïƒ_t` if and only if the
kernels are deterministic â€” which is exactly the caveat `the-training-loop.md` Â§3.7 spends
a section on and which this module inherits rather than restates.

**The checkpoint** is a pair of maps: `W : Ïƒ â†’ bytes` and `R : bytes â†’ Ïƒ`.

**Resume-equivalence** is the property

```
R(W(Ïƒ_t))  =  Ïƒ_t          and therefore        Î¦^k(R(W(Ïƒ_t)))  =  Î¦^k(Ïƒ_t)   for all k
```

Two entirely different things can break it, and a test that cannot tell them apart is
useless:

1. **Incompleteness.** `R âˆ˜ W â‰  identity` because `W` drops a component of `Ïƒ`. This is
   deterministic, reproducible, and a *bug*.
2. **Non-determinism of `Î¦`.** Even when `R âˆ˜ W = identity`, `Î¦^k` can differ between two
   processes because float addition is not associative, GPU reductions combine partials in
   completion order, atomics complete nondeterministically, and library autotuning can pick
   a different kernel. This is not a bug and cannot be fixed by saving more state. The
   mechanisms are `the-training-loop.md` Â§3.7 and Track D's determinism module; this module
   uses them only as the source of the noise floor.

**Therefore the test needs a null hypothesis.** Define:

```
Îµ_noise    =  max | Î¸_A âˆ’ Î¸_B |   over two UNINTERRUPTED runs, same seed, two fresh processes
Îµ_resume   =  max | Î¸_split âˆ’ Î¸_ref |   for the split-and-resume run against the reference
```

and the pass criterion is

```
Îµ_resume  â‰¤  Îµ_noise
```

**not** `Îµ_resume = 0`. If you skip the noise measurement you will either demand bit-exactness
on hardware that cannot deliver it, or accept a large `Îµ_resume` because "floats are like
that" when in fact your checkpoint is missing the RNG state.

`[M]` With this wheel the noise floor is **exactly zero on both CPU and gfx1151** â€” three
seeds, two fresh processes each, `torch.equal` true and max difference 0.0, and reproduced by
two independent scripts with different model shapes. So on this machine, at these shapes, the
criterion collapses to bit-exactness and the test is maximally sharp. Do not over-read that:
the toy is fp32, small, and does not call `scaled_dot_product_attention`, whose backend
selection on this stack is itself environment-dependent
(`ASSUMPTIONS.md: sdpa-is-memory-efficient`). A zero floor at 33 K parameters is not a zero
floor at 300 M, and `the-training-loop.md` Â§3.7 lists four mechanisms that would break it.
**Re-measure the floor whenever the model, the dtype, or the environment changes** â€” that is
the whole reason it is a fixture and not a constant.

**Why compare parameters and not the loss curve.** `[M]` A checkpoint containing model +
optimizer + step but no RNG state produced, across three seeds, a resumed run in which
**100% of parameters differed** from the reference and the maximum absolute difference was
~1.0e-2 â€” while the loss over those same steps stayed entirely within the range you would
call normal. The loss is one scalar summarizing ten million numbers; it is a smoke alarm.
The parameter comparison is a checksum. `research/reference/CODE_MAP.md` makes the same point
about the Hardware Validation Gate and it is worth restating: **compare weights bit-exactly,
never loss trajectories.**

**`Ï_t` is a vector, and on this hardware the last element is the one that bites.** `[M]`
Measured on gfx1151 with the noise floor at exactly 0.0, so the residuals below are
incompleteness and nothing else:

| What the checkpoint carried | `max \|Î”Î¸\|` vs. the uninterrupted reference | fraction of parameters differing |
|---|---|---|
| model + optimizer + step | 1.01e-2 / 0.94e-2 / 1.02e-2 | **1.000** |
| ...plus Python, NumPy and torch-CPU RNG | **5.20e-3 / 6.47e-3 / 5.49e-3** | **0.99998** |
| ...plus `torch.cuda.get_rng_state()` | **0.0, bit-exact** | 0.0 |

Three seeds, each arm in its own subprocess. The middle row is the finding: the three-stream
fix that closes the gap completely on CPU **halves the error on GPU and leaves it wrong**,
because `nn.Dropout` on a device tensor draws from the CUDA generator, which is a fourth,
independent stream. A partial RNG restore is *worse than obviously broken* â€” it looks like it
worked, the error shrank, and 99.998% of your parameters are still different. This is why
`EnvRngStates` (`train/utils.py:37`) enumerates four streams rather than three, and it is
worth going and reading that class after seeing the number.

**Why the step counter is state, and why it is stored twice.** Two independent reasons:

1. The LR schedule is a function `Î·(t)`. Resume with the wrong `t` and you resume at the
   wrong learning rate â€” and under WSD (`research/notes/pretraining-recipes.md` Â§2) that can
   mean resuming the stable phase at decay-phase LR.
2. AdamW's bias correction divides by `1 âˆ’ Î²â‚^t` and `1 âˆ’ Î²â‚‚^t`. At `t = 1` with `Î²â‚ = 0.9`
   the divisor is 0.1, a 10Ã— amplification. Resuming a step-5000 model with `t` reset to 0
   makes the first post-resume update roughly ten times too large.

And here is the trap: PyTorch stores `step` **inside the per-parameter optimizer state**, so
it rides along with `optimizer.state_dict()` whether or not you also save a global step
counter â€” `[M]` verified, a 5-step optimizer round-tripped with `step = 5.0` intact. Meanwhile
the trainer keeps its own `global_step` (`trainer.py:792`). Two counters, two code paths, and
nothing checks that they agree.

### 3.5 Sharded versus consolidated, and the one criterion that decides resharding

This section covers the checkpoint *format* question only. Why a model is sharded in the first
place â€” FSDP, tensor, pipeline and expert parallelism â€” belongs to Track D's distributed-training
module and is not repeated here.

**Consolidated** means one logical file containing `{parameter_name: tensor}`, written by one
process. **Sharded** means `N` data files plus an index, each rank writing only its own slice.

The cost model is straightforward. With `N` ranks, per-node write bandwidth `B`, and
interconnect bandwidth `B_link`:

```
t_consolidated  =  S / (N Â· B_link)   [gather to rank 0]   +   S / B      [one writer]
t_sharded       =  S / (N Â· B)                                            [N writers]
```

- `S` â€” total checkpoint bytes, `12 P`

Sharded wins by roughly `NÃ—` on the write and, more importantly, bounds peak host memory at
`S/N` per node instead of `S` on one node. Once `S` exceeds one machine's RAM, consolidated is
not slow â€” it is impossible. That is why the format exists.

**The load side is where the interesting structure is.** A sharded checkpoint is a metadata
file plus opaque data files. The metadata maps each logical tensor slice to a
`(relative_path, offset, length)` extent, and a load is a set of byte-range reads â€” literally
range GETs against local files or S3 (`filesystem.py:371`). No rank reads bytes it does not
need.

**Now the criterion, which is the takeaway of the section:**

> A checkpoint is topology-agnostic **iff** every stored extent's key is a function of the
> *logical* tensor identity and index range alone â€” never of rank id, world size, or
> construction order.

Apply it and everything follows. Model weights key naturally by parameter name
(`blocks.3.attn.q_proj.weight`) plus an index range, so they are fine. **Optimizer state is
where it breaks**, because vanilla PyTorch keys optimizer state by *integer index into the
flattened param groups*, which is a function of the order in which you handed parameters to
the constructor. Change the param-group split â€” say, from one group to the standard
decay/no-decay split â€” and the mapping changes underneath you.

`[M]` Measured: saving `AdamW` state from a single-group optimizer and loading it into the
same model's parameters re-grouped into decay/no-decay raises
`ValueError: loaded state dict has a different number of parameter groups`. Loud, which is
the good case. It is loud only because the *count* changed; a reorder within the same count
would not be.

OLMo-core avoids the whole class of problem by going through
`torch.distributed.checkpoint`'s `get_optimizer_state_dict`, which re-keys optimizer state by
**parameter fully-qualified name** (`training/olmo-core/src/olmo_core/distributed/checkpoint/__init__.py:702`, with
`full_state_dict=False, cpu_offload=True` at `:709`â€“`:711`). CODE_MAP's phrasing is exactly
right: **resharding is re-indexing, not data movement.**

**Read amplification, which nobody warns you about.** If a checkpoint written by `N` writers
is read by `N'` readers, each reader's logical slice of each tensor can span up to
`âŒˆN/N'âŒ‰ + 1` writer files. With `T` tensors that is on the order of `T Â· (N/N' + 1)` range
reads. At `T = 1000`, `N = 64`, `N' = 8`, that is ~9,000 small reads. `[C]` "Understanding LLM
Checkpoint/Restore I/O Strategies and Patterns" (arXiv 2512.24511, Dec 2025) measures exactly
this and finds uncoalesced small-buffer operations halve throughput relative to a coalesced
synthetic workload.

**On our hardware:** `N = 1`. `ASSUMPTIONS.md: single-device-only`. Everything in this
subsection is design-only for us, and the acceptance test for it lives on rented hardware.
See "not reproducible here" in Â§6.

### 3.6 The detection gap, quantified

A checkpoint has no integrity check in practice. Here is how bad that is, made arithmetic.

`[M]` **Measured on this stack, fresh process, with a control.** A `.pt` file is a zip
container, and every entry carries a valid CRC-32 â€” Python's `zipfile.testzip()` returns
`None` (no bad entries) on an unmodified `torch.save` output. Flip one byte inside the tensor
payload and `testzip()` names the corrupt entry, `crc_control/data/0`. `torch.load(...,
weights_only=True)` on that same file **returns without error**: one element changed, maximum
absolute error 0.253, every value finite, and the Frobenius norm ratio to the original
**0.99999994**.

So: the checksum exists, in the file, computed by PyTorch's own writer. The reader does not
look at it. And no downstream health check would find it either â€” a weight-norm monitor sees
a 6e-8 relative change.

**Which bit flips are even detectable in principle?** Take bf16: 1 sign bit, 8 exponent bits,
7 mantissa bits.

| Bit position | Effect of a flip | Would a finiteness / max-magnitude check catch it? |
|---|---|---|
| sign (1 bit) | `w â†’ âˆ’w` | No. Magnitude is unchanged; a norm check is blind by construction. |
| exponent MSB (1 bit) | `Ã— 2^128` or `Ã— 2^âˆ’128` | Usually yes â€” overflows to `inf` or flushes to a subnormal. |
| exponent bits 6â€¦4 (3 bits) | `Ã— 2^64`, `Ã— 2^32`, `Ã— 2^16` | Often, via a max-magnitude threshold. |
| exponent bits 3â€¦0 (4 bits) | `Ã— 2^8` down to `Ã— 2^1` | Rarely. A 2Ã— weight is an unremarkable weight. |
| mantissa (7 bits) | relative change from `2^âˆ’7` to `~0.5` | Never. The largest is the same size as bf16 rounding error. |

Roughly **4 of 16 bit positions are catchable by any check a trainer actually runs**, and the
other 12 restore as a plausible model. That is the quantitative content of "silently wrong
rather than obviously corrupt."

This is not a hypothetical concern at scale: `[C]` "Exploring Silent Data Corruption as a
Reliability Challenge in LLM Training" (arXiv 2604.00726, Apr 2026) surveys SDC in training,
and secondary reporting of Google's Gemini training describes SDC events on the order of once
every one to two weeks â€” a rate at which the question is not *whether* a corrupted checkpoint
enters a run but *whether you would ever know*.

**The fix is three lines and it is not in any trainer.** `zipfile.ZipFile(path).testzip()`
before `torch.load`. Exercise B measures what it costs.

---

## 4. Why it matters for Proteus and Mnemosyne

**One: resume-equivalence is an ablation-validity property, not an ops property.** This is
the argument that should change how the rig is built. Themis compares arms at matched
parameter counts and matched token budgets with â‰¥3 seeds. If resume is not equivalent, then
an arm that was interrupted twice and an arm that was interrupted once **differ by their
interruption history** â€” a variable that is not in the config, not in the run record, and not
controlled for. It is a confound with no name. On a laptop that sleeps, gets unplugged, and
shares a machine with a browser, interruptions are not rare events; they are the normal case.
So resume-equivalence goes in `tests/` as a rig invariant with a red-then-green regression
test, exactly like the package-boundary contract (`ASSUMPTIONS.md: mnemosyne-separable`).

**Two: the config surface gains a checkpoint section, and it is genuinely experimental.**

| Config field | Why it is on the experimental surface, not just the ops surface |
|---|---|
| `checkpoint.interval_steps` | Â§3.3 says get within 3Ã— and stop. Make it a field so a run record states it; do not sweep it. |
| `checkpoint.save_rng_state` | Default `true`, and it means **all four streams** â€” `[M]` Â§3.4 shows three is a silent half-fix on GPU. Exists as a field so `false` can *reproduce* the incompleteness bug and measure its size at our scale, the same reason `loss.div_factor_policy` exists in `the-training-loop.md` Â§4. |
| `checkpoint.verify_crc_on_load` | Default `true`. Â§3.6. Costs measured in Exercise B. |
| `checkpoint.retention.{permanent,ephemeral}` | GFS rotation; bounded by disk, not by policy taste. |
| `checkpoint.environment_fingerprint` | See below. This is the one that is ours. |
| `determinism.init_seed`, `determinism.data_order_seed` | Already in the surface from `the-training-loop.md` Â§4; the checkpoint must round-trip *both* generators, not just torch's. |

**Three: the checkpoint must carry an environment fingerprint, and no reference trainer does
this.** `ASSUMPTIONS.md: hipblaslt-config` records `[M]` that hipBLASLt configuration changes
the relative error of a length-1M bf16 reduction by **2.8Ã—** (2.01e-3 configured vs 5.60e-3
unconfigured, three seeds, fresh processes). `ASSUMPTIONS.md: sdpa-is-memory-efficient`
records `[M]` that `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` swaps the attention kernel
entirely. Both are *numerics changes* selected by environment variables that leave no trace in
any artifact. A checkpoint resumed under a different setting than it was written under is a
different experiment, and nothing anywhere would tell you.

OLMo-core's `CheckpointMetadata` carries exactly two fields, `ephemeral` and `version`
(`train/checkpoint.py:73`). For Proteus that is not enough. The fingerprint should record:
torch wheel string, HIP version, GPU arch, `TORCH_BLAS_PREFER_HIPBLASLT`,
`HIPBLASLT_TENSILE_LIBPATH` presence, `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL`, autocast
dtype, and the config hash â€” and the loader should **warn loudly on mismatch and refuse under
`--strict-provenance`**. This is cheap, it is ours, and `bf16-numerics-unproven` is the reason
it is not optional.

**Four, and this is the Mnemosyne-specific one: memory policies have state, and nobody has a
convention for checkpointing it.** A KV-eviction policy in the H2O family accumulates
per-token attention mass; a prefix cache holds a hash chain and a refcounted free list
(`kv-eviction-policies.md`, `paged-attention-and-prefix-reuse.md`); an attribution instrument
holds histograms. These are `s_t` in Â§3.4 â€” real state that the next step reads â€” and they are
not model parameters, not optimizer state, and not covered by any `state_dict` convention.

The consequence for the interface is concrete and should be settled before code is written:
**Mnemosyne's policy interface must require `state_dict()` / `load_state_dict()`, or a policy
must declare itself stateless and be tested for it.** A stateful policy with no serialization
means a long-context evaluation that was interrupted is not reproducible, and â€” because
Mnemosyne is the research contribution â€” an unreproducible policy is an unpublishable one.

Note where the agent-memory analogy from Track C inverts here. An agent memory store is a
write-ahead log with schema drift and no compaction: its problem is that it keeps everything.
A policy's counters are the opposite object â€” a lossy fixed-size aggregate over a stream, with
no log behind it. You cannot rebuild an H2O accumulator by replay, for the same reason
`constant-state-memory.md` gives for an SSM carry: the information was destroyed, not
relocated. So it must be *saved*, or the policy must be *stateless*. There is no third option.

**Five: what we cannot do, stated plainly.** Sharded checkpointing, resharding across world
sizes, async checkpointing with a separate CPU process group, and every FSDP-adjacent
consideration in Â§3.5 are **design-only on this machine**
(`ASSUMPTIONS.md: single-device-only`, `[C]` `torch._C._distributed_c10d` incomplete on
gfx1151). We write consolidated checkpoints with `torch.save` and we read the sharded code to
understand it. Any milestone that needs a real sharded round-trip is a rented-hardware
milestone and should be costed as one.

---

## 5. Read the code

Two codebases, read in this order: nanoGPT for the minimal version and everything it omits,
then OLMo-core for what the same job looks like when it has to survive a 54-day run. Paths are
relative to `research/reference/`; run `scripts/fetch_reference.sh` first if the clones are not
materialized.

### nanoGPT â€” the incomplete checkpoint, and why it is incomplete

| Where | What to look at, and why |
|---|---|
| `training/nanogpt/train.py:277` | The entire checkpoint: six keys â€” `model`, `optimizer`, `model_args`, `iter_num`, `best_val_loss`, `config`. Before reading further, write down Â§3.4's `Ïƒ_t` from memory and cross off what is here. You should be left with `Ï_t` (all four RNG streams), `c_t` (data position), and the `GradScaler` state. |
| `training/nanogpt/train.py:286` | `torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))` â€” **in place, over the previous checkpoint, with no temp-and-rename.** This single line is the whole torn-write hazard. `[M]` Exercise B shows the good checkpoint is destroyed the instant the new write begins: a crash mid-save leaves you with neither. Compare against `checkpoint.py:498` below and note that the fix is a context manager. |
| `training/nanogpt/train.py:274` | `if losses['val'] < best_val_loss or always_save_checkpoint:` â€” the save is gated on *validation improvement*, so the checkpoint interval is not what you set; it is whenever the model got better. As an RPO policy that is unbounded: a plateau means no checkpoint for an arbitrary number of steps. |
| `training/nanogpt/train.py:158`â€“`:180` | The resume path. `:162` loads with `torch.load(..., map_location=device)`; `:166` force-overrides six architecture fields from the checkpoint so a resume cannot silently change the model shape (good); `:179`â€“`:180` restore `iter_num` and `best_val_loss` and **stop**. |
| `training/nanogpt/train.py:196` | `scaler = torch.cuda.amp.GradScaler(...)` â€” constructed fresh, never saved, never restored. Under fp16 the scaler is a stateful control loop (`the-training-loop.md` Â§3.6) and a resume silently restarts its search for the scale factor. Inert under bf16, which is the only reason this has not bitten anyone. |
| `training/nanogpt/train.py:116` | `get_batch` â€” random offsets drawn from the global torch RNG, with replacement. There is no cursor to save, so restoring the **torch RNG state alone** restores the data position as a side effect. That is a legitimate design; it is just not the one OLMo-core made, and the two have different failure modes. |
| `training/nanogpt/train.py:106` | `torch.manual_seed(1337 + seed_offset)` â€” the only seeding in the file. Diff against `seed_all` and `EnvRngStates` below to enumerate the streams that are loose. |

### OLMo-core â€” the complete checkpoint

| Where | What to look at, and why |
|---|---|
| `training/olmo-core/src/olmo_core/train/trainer.py:87` | `class TrainerStateDict(TypedDict)` â€” nine fields. **Read this as the authoritative answer to "what must be saved."** `global_step`, tokens seen, petaflops, `max_steps`, the data loader's state, `epoch`, `world_size`, `rng`, and every callback's state. Hold it next to nanoGPT's six keys. |
| `training/olmo-core/src/olmo_core/train/trainer.py:787` | `state_dict()` â€” where those nine fields are populated. Note `world_size` is saved *on purpose* (the comment says so) and note that `callbacks` fans out to `cb.state_dict()` for each one, which is the extension point for anything with `s_t` state â€” including, for us, a Mnemosyne policy. |
| `training/olmo-core/src/olmo_core/train/trainer.py:838` | `if state_dict["world_size"] == get_world_size():` â€” **RNG is restored only when the topology matches.** Otherwise the restore is skipped and a `log.warning` is emitted. This is the single most instructive line in the file: resume-equivalence is *conditional*, and when the condition fails the system degrades to a log line rather than an error. Decide for Proteus whether that is acceptable; for a single-device lab the condition always holds, so make it an assertion. |
| `training/olmo-core/src/olmo_core/train/utils.py:37` | `class EnvRngStates` â€” four streams: `python`, `numpy`, `torch`, and optionally `cuda`. This is `Ï_t` made concrete, and Â§3.4's measurement is the reason the fourth field is not redundant: restoring the first three is bit-exact on CPU and leaves 5e-3 of error across 99.998% of parameters on gfx1151 `[M]`. |
| `training/olmo-core/src/olmo_core/train/utils.py:43` | `restore()` â€” each stream is restored **only if the library version matches**, and the method returns a bool that the caller turns into a warning. A NumPy minor-version bump between save and resume silently costs you one RNG stream. |
| `training/olmo-core/src/olmo_core/data/data_loader.py:441` | The dataloader's `state_dict` â€” six fields, and only one of them is a position: `dataset_fingerprint_version`, `dataset_fingerprint`, `batches_processed`, `tokens_processed`, `seed`, `epoch`. **It stores a cursor, not a log.** |
| `training/olmo-core/src/olmo_core/data/data_loader.py:451` | `load_state_dict` â€” the fingerprint check. A dataset fingerprint mismatch **raises** unless you pass `ignore_fingerprint_mismatch=True`. This is the one place in the whole stack where a resume refuses to proceed on a correctness ground, and it is worth copying. |
| `training/olmo-core/src/olmo_core/data/data_loader.py:667` | `_build_global_indices` â€” the epoch's entire instance permutation is a pure function of `(seed, epoch, dataset length, chunk size)`. Nothing about data order is persisted, because it is rebuildable. The permutation is memmapped into the work dir as a derived cache. |
| `training/olmo-core/src/olmo_core/data/data_loader.py:720` | `_get_local_instance_indices` â€” the resume itself: reshape into batches, `indices[self.batches_processed:]`, then stride by worker and DP rank. Two lines of slicing replace all per-worker iterator state. |
| `training/olmo-core/src/olmo_core/data/data_loader.py:760` | `self.batches_processed = self.tokens_processed // self.global_batch_size` â€” the saved batch count is **thrown away and recomputed from tokens**. The cursor is really denominated in tokens, which is what lets you resume the same run with a different batch size or world size and land on the right token boundary. |
| `training/olmo-core/src/olmo_core/train/checkpoint.py:498` | `_temporary_wd` â€” the commit protocol. Write into `<dir>-tmp`, barrier, rename over the target (`:463`, `tmp_dir.replace(str(dir))`). **Atomicity is rename-granularity**, so a torn save loses the whole checkpoint rather than a tail â€” which is the right trade, because a tail is not a recovery unit here. |
| `training/olmo-core/src/olmo_core/train/checkpoint.py:134` | `self._save_metadata(...)` â€” called *after* `_temporary_wd` exits, i.e. after the rename. Combined with `dir_is_checkpoint` at `:295`, which requires `.metadata.json` to exist, this makes the metadata file a **commit record**: present means complete. |
| `training/olmo-core/src/olmo_core/train/callbacks/checkpointer.py:179` | `_remove_checkpoint` â€” "Remove metadata file first to invalidate the checkpoint," then clear the directory. Invalidate the catalog entry before destroying the data. You have written this exact code for a backup catalog. |
| `training/olmo-core/src/olmo_core/train/checkpoint.py:270` | `if hasattr(os, "fdatasync"):  # only available on linux` â€” **read this one carefully on our platform.** The commit-record file is flushed and `fdatasync`'d before the rename on Linux; on Windows the `fdatasync` is skipped and only `flush()` runs. The rename is still atomic on NTFS, but the *data* behind the newly-visible name is not guaranteed durable across a power loss. Native Windows is our primary target (`CLAUDE.md`), so this is a real gap in the reference implementation for us, not a curiosity. |
| `training/olmo-core/src/olmo_core/train/checkpoint.py:136` | `save_async` â€” note the scope of `with self._temporary_wd(dir) as wd:`. It wraps **only** `_save_train_state`; the model and optimizer shards are written directly into the live directory afterwards by a background future, and the commit record is written in the done-callback (`:174`). The consistency group is published in two phases, and the only thing making that safe is that `dir_is_checkpoint` refuses a directory without the commit record. |
| `training/olmo-core/src/olmo_core/train/callbacks/checkpointer.py:63`, `:87`, `:102` | `save_interval = 250`, `remove = ephemeral_only`, `max_checkpoints = 3` â€” the retention policy, as three defaults. Grandfather-father-son with one rolling ephemeral (saved at `:310`, trimmed at `:313` by `while len(self._ephemeral_checkpoints) > 1`). |
| `training/olmo-core/src/olmo_core/distributed/checkpoint/__init__.py:702` | `_prepare_state_dict` â€” five lines that decide the whole sharding story: `full_state_dict=False`, `cpu_offload=True`, optimizer state keyed by parameter FQN. This is Â§3.5's criterion, implemented. |
| `training/olmo-core/src/olmo_core/distributed/checkpoint/__init__.py:331` | The docstring warning: *"if you have keys in the checkpoint dict that are not present in the current state of the model or optimizer, those keys won't be loaded."* Silent partial restore, documented as a warning in prose rather than enforced by a check. |
| `training/olmo-core/src/olmo_core/train/train_module/transformer/train_module.py:318` | `# NOTE: dist_cp_sd.set_(model|optimizer)_state_dict() doesn't respect strict=False option with missing keys, so we have to handle that on our own.` â€” a comment recording that the upstream strictness flag does not do what it says, and the workaround. Schema drift is being handled by hand, in a comment, in the one place someone noticed. |
| `training/olmo-core/src/olmo_core/distributed/checkpoint/filesystem.py:371` | `_get_bytes(relative_path, offset, length)` â€” the entire load data path, as byte-range reads. This is what makes the metadata file a page table. |
| `training/olmo-core/src/olmo_core/distributed/checkpoint/__init__.py:465` | `unshard_checkpoint` â€” the conversion back to a consolidated artifact, with three strategies (`one_file`, `one_file_per_tensor`, `chunks`) because the one-file version's memory cost is linear in model size. Note the warning that safetensors cannot hold optimizer state, since optimizer state contains arbitrary pickled Python objects. |

---

## 6. Exercises

All three run on the Z13 (gfx1151, native Windows, one GPU) and all three have a CPU fallback
that is not a degraded version â€” the code is device-parameterized and both sets of numbers below
were taken. **Exercise A must be run on both devices**, because on this stack the CPU and GPU
answers differ and the difference is the lesson; B and C are device-independent apart from the
two GPU-only steps that are marked as such. Total wall clock for the three, on either device,
is under five minutes of compute â€” the time cost is writing the harness, not running it.

Activate with `. .\scripts\activate-lab.ps1` from the repo root. Standing caveats from
`the-training-loop.md` Â§6 apply: pass `--compile=False`; if a run dies with an access
violation and no Python traceback, clear `TORCH_BLAS_PREFER_HIPBLASLT` and
`HIPBLASLT_TENSILE_LIBPATH` and retry; keep every single tensor under 32 GiB.

**One extra caveat specific to this module.** Everything you measure about resume-equivalence
depends on the environment variables in force, because `ASSUMPTIONS.md: hipblaslt-config` and
`sdpa-is-memory-efficient` are both numerics controls. **Record the environment with every
number**, and do not compare a run taken with AOTriton enabled against one taken without.

### Exercise A â€” Build the resume-equivalence harness, and measure the noise floor first

**Difficulty 3/5. ~60â€“90 min. This is the exercise that becomes rig code.**

The deliverable is a test that can fail. Four processes per seed, three seeds:

1. **Reference.** Fresh process: seed, build, train `2n` steps, dump a flat parameter vector.
2. **Noise floor.** A *second* fresh process doing exactly the same thing. `Îµ_noise` is the max
   absolute difference between (1) and (2). **Do this before anything else.** Without it you
   have no pass criterion.
3. **Split, minimal checkpoint.** Fresh process: train `n` steps, save `{model, optim, step}`
   (nanoGPT's scope). Then *another* fresh process: reseed with the same nominal seed, build,
   load, train `n` more steps, dump.
4. **Split, three-stream checkpoint.** Same, but the saved dict also carries
   `random.getstate()`, `np.random.get_state()`, and `torch.random.get_rng_state()`.
5. **Split, four-stream checkpoint.** Same again, plus `torch.cuda.get_rng_state()`.

Run every arm on both devices. Arms 4 and 5 are identical on CPU and are *not* identical on
GPU, and finding out why is the point of the exercise.

The subprocess boundary is not optional. A single-process test shares the allocator, the
autotune cache, and every module-level global, and it will pass for the wrong reason.

```python
# the load half, which is the only part that is subtle
import pickle, random, numpy as np, torch
ck = torch.load(path, weights_only=False)
model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optim"])
random.setstate(pickle.loads(ck["rng"]["python"]))
np.random.set_state(pickle.loads(ck["rng"]["numpy"]))
torch.random.set_rng_state(ck["rng"]["torch"])          # a uint8 ByteTensor, not a blob
if "cuda" in ck["rng"]:
    torch.cuda.set_rng_state(ck["rng"]["cuda"])         # a SEPARATE generator. this is arm 5.
```

**What to check.** `[M]` Measured 2026-07-26, `torch 2.12.0a0+rocm7.13.0a20260313`, Python
3.12.10, three seeds (1337/1338/1339), every arm in its own subprocess, `n = 10` steps each
side of the split, a 5-tensor toy (33 K parameters) with `Dropout(0.2)` and an RNG-driven data
sampler. The whole CPU set was run twice on different occasions and reproduced.

| Arm | CPU â€” `max \|Î”Î¸\|` (3 seeds) | gfx1151 â€” `max \|Î”Î¸\|` (3 seeds) | frac. differing (GPU) |
|---|---|---|---|
| **noise floor** (two uninterrupted runs) | **0.0, 0.0, 0.0 â€” bit-exact** | **0.0, 0.0, 0.0 â€” bit-exact** | 0.0 |
| model + optimizer + step | 1.03e-2, 1.03e-2, 1.00e-2 | 1.01e-2, 0.94e-2, 1.02e-2 | **1.000** |
| + python / numpy / torch-CPU RNG | **0.0 â€” bit-exact** | **5.20e-3, 6.47e-3, 5.49e-3** | **0.99998** |
| + `torch.cuda` RNG as well | 0.0 â€” bit-exact | **0.0 â€” bit-exact** | 0.0 |

Four things to take from that table, in order of how much they should change your behaviour.

1. **The noise floor is exactly zero on both devices**, so the criterion is bit-exactness and
   the test is maximally sharp. That is a better result than expected on gfx1151 and it is a
   direct contribution to the Hardware Validation Gate's determinism item. Do not generalize
   it: 33 K fp32 parameters with no attention kernel is the easiest possible case. Re-measure
   at 20 M in bf16 with SDPA in the graph before believing it there.
2. **A model+optimizer checkpoint changes every single parameter.** Not "a few in the last
   decimal" â€” `frac_differ = 1.000`, at a magnitude of 1e-2, which is infinitely above a zero
   noise floor. And the loss curve over those steps is unremarkable. If you take one number
   from this module, take that one.
3. **The three-stream fix is complete on CPU and incomplete on GPU.** This is the trap. On CPU
   the residual goes to exactly zero and you would ship it. On GPU the residual only *halves*,
   to 5e-3, with 99.998% of parameters still different â€” because `nn.Dropout` on a device
   tensor draws from `torch.cuda`'s generator, which is a fourth stream that neither
   `torch.manual_seed` nor `torch.random.set_rng_state` touches. **A partially restored RNG is
   the worst state to be in**: the error shrank, so it looks like the fix worked.
4. **The fourth call closes it.** `torch.cuda.set_rng_state` takes it to bit-exact on all three
   seeds. Note *why* the whole gap closes and not just the dropout part: in this design (and
   nanoGPT's) the data position is drawn from the torch **CPU** generator, so restoring the RNG
   restores the cursor as a side effect. If your loader keeps an explicit cursor â€” OLMo-core's
   design (`data_loader.py:441`) â€” you must save that separately and no amount of RNG restoring
   will save you. **Know which of the two designs you are running.**

**Then make it a test.** Move it into `tests/` as
`test_resume_produces_bit_identical_parameters.py`, parameterized over device, with the noise
floor as an explicit fixture rather than a hardcoded zero. Red-then-green: delete the CUDA RNG
restore, watch it go green on CPU and red on GPU, and note that a CPU-only CI would have
missed it.

**Bonus, five minutes:** add a sixth arm that restores everything *except* the optimizer state
and report `max |Î”Î¸|`. That is the difference between "a different trajectory" and "a
different optimizer."

### Exercise B â€” Corrupt a checkpoint three ways and see which ones anyone notices

**Difficulty 2/5. ~45 min. Produces three yes/no answers and one cost number.**

**Part 1 â€” the torn write.** Wrap the output file in an object whose `write` raises after
`limit` bytes, so the failure is deterministic instead of requiring a real kill:

```python
class CrashingWriter:
    def __init__(self, path, limit):
        self.f, self.n, self.limit = open(path, "wb"), 0, limit
    def write(self, b):
        b = bytes(b)
        if self.n + len(b) > self.limit:
            self.f.write(b[:max(0, self.limit - self.n)]); self.f.flush(); self.f.close()
            raise IOError("simulated crash mid-save")
        self.f.write(b); self.n += len(b); return len(b)
    def flush(self): self.f.flush()
```

Write a good checkpoint to `ckpt.pt`, confirm it loads, then attempt a torn save **to the same
path** and report: does the good checkpoint still exist? does the torn file load?

`[M]` Measured, same environment: the good file was 201,258 bytes; the torn write left 110,691
bytes at that path, so **the previous good checkpoint was destroyed** â€” the classic in-place
overwrite. `torch.save` itself raised `RuntimeError: [enforce fail at
inline_container.cc:672] . unexpected pos 100864 vs 100816`, and loading the truncated file
raised `RuntimeError`. **So truncation is loud** â€” the zip central directory is written last,
so a short file is structurally invalid â€” **but you have lost both copies.** Now repeat with a
temp-file-plus-rename and confirm the good checkpoint survives. That is `checkpoint.py:498` in
eight lines, and it is the single highest-value change to make to any script you write.

**Part 2 â€” the bit flip, which is the one that matters.** Save a 4 MiB tensor, flip one byte at
50% of the file offset, then ask three separate questions: does `zipfile.testzip()` see it,
does `torch.load` see it, and would a norm check see it?

`[M]` Measured with a control (an unmodified file first, to prove the CRC is real):

| Question | Answer |
|---|---|
| `zipfile.testzip()` on the unmodified file | `None` â€” no bad entries. **The CRC-32 is present and correct.** |
| `zipfile.testzip()` after the flip | `'crc_control/data/0'` â€” the corrupt entry, named. |
| `torch.load(..., weights_only=True)` after the flip | **`LOADED WITHOUT ERROR`** |
| elements changed | 1 |
| max absolute error | 0.253 |
| all values finite | `True` |
| Frobenius norm ratio to the original | **0.99999994** |

The checksum is in the file, written by PyTorch, and PyTorch's reader ignores it. Now measure
the cost of not ignoring it: time `zipfile.ZipFile(p).testzip()` against `torch.load(p)` on the
same file, and report the ratio. Decide from your own number whether `verify_crc_on_load`
should default to `true` in Proteus. (It reads the whole file and CRC-32s it, so expect the
same order of magnitude as the load itself â€” meaning the honest framing is "one extra read,"
not "free.")

**Part 3 â€” is there really no partial restore?** Snapshot the same run at two different steps,
then build a hybrid model: most layers from the late checkpoint, one or two blocks from the
early one. Evaluate all three on identical batches.

State the prediction first so it can fail: *the spliced model is worse than either parent,
because the layers were co-adapted.*

`[M]` Measured, single run (an anecdote by the house standard), 4-block toy, 800 steps, 50
identical eval batches:

| Model | eval loss |
|---|---|
| all layers from step 100 | 3.4061 |
| all layers from step 800 | 3.2814 |
| step 800, blocks 0â€“1 replaced from step 100 | 3.3046 |
| step 800, block 0 replaced from step 100 | 3.2836 |

**The prediction is refuted at this scale.** Splicing degraded gracefully and monotonically â€”
one stale block cost 0.0022 nats, two cost 0.023, and neither was worse than the early parent.
Two honest caveats before you generalize: the whole run only moved the loss 0.125 nats, so
there was little co-adaptation to break; and this is one run per arm.

Interestingly this *agrees* with the frontier: `[C]` "All is Not Lost: LLM Recovery without
Checkpoints" (arXiv 2506.15461, Jun 2025) recovers a lost pipeline stage by weighted-averaging
its neighbours and beats redundant computation by up to 12% in convergence wall-clock at 5â€“10%
failure rates. So "no partial restore" is not a statement about *eval loss* â€” that part is
contested and the evidence leans against the folklore. **It is a statement about the resume
being coherent:** splice in stale parameters and AdamW's `m` and `v` for those tensors are EMAs
of gradients that were computed for *different* parameters. The trajectory from that point is
not the trajectory of either parent, and nothing reports it. Rewrite the claim that way in your
notes, because the loss-based version does not survive measurement.

### Exercise C â€” Price your own DR, with the formula from Â§3.3

**Difficulty 2/5. ~40 min. Produces one table you will actually use.**

Measure `Î´` and `T_load` for a size sweep, then invert Young's formula.

1. Build models at three sizes, take one optimizer step so AdamW state exists, and
   `torch.save({"model": ..., "optim": ...})` three times each, taking the median. Report bytes,
   bytes/param, save seconds, load seconds, and MB/s.
2. `[M]` The CPU reference to check yourself against: 12.0027 / 12.0007 / 12.0002 bytes per
   parameter at 2,099,712 / 8,393,728 / 33,564,672 parameters, with save throughput 1675 â†’
   1554 â†’ 992 MB/s and load 0.0060 / 0.0167 / 0.0665 s. If your bytes/param is not 12.00, you
   are either not saving optimizer state or not using fp32 masters, and you should find out
   which before continuing.
3. Extrapolate `Î´` to 300 M parameters (3.6 GB) and build the inversion table from Â§3.3:
   `M = Ï„*Â²/(2Î´)` for `Ï„ âˆˆ {60, 300, 900, 3600}` seconds. Then pick an interval and write down
   the implied bet on `M` in one sentence.
4. **On GPU, add the term the CPU run cannot see:** time `state_dict()` with the model on
   `cuda` and the tensors being pulled to host. On unified memory that copy is a memcpy inside
   the same physical DRAM rather than a PCIe transfer, so it may be far cheaper than the
   literature assumes â€” or it may not, since nothing about the D2H path on gfx1151 has been
   measured here. Either answer is a `notebook/` entry, and it is the cheapest unclaimed
   measurement in this module.
5. **While the model is on the GPU, close the Hardware Validation Gate's round-trip item.**
   Save model + optimizer from a stepped model on `cuda`, load into a *differently* initialized
   model in the same process, and compare every parameter and every optimizer state tensor with
   `torch.equal`. `[M]` Measured: **bit-exact on both counts on gfx1151**, with AdamW's `step`
   surviving as `5.0`. But note the one thing that does *not* round-trip identically â€” the
   devices. `[M]` The running optimizer's state tensors were spread across `{cpu, cuda:0}`
   (AdamW keeps `step` as a CPU scalar), and after `load_state_dict` every state tensor was on
   `cuda:0`, because `load_state_dict` casts optimizer state to the device of its parameter.
   Values identical, placement changed. Harmless here; worth knowing before you debug why a
   resumed run has a different per-step host-device sync pattern than the original.

**Then the free part.** While you are timing, add a step-progress heartbeat: append
`{step, wall_clock, tokens}` to a JSONL file every step and write a five-line watchdog that
alarms if the newest record is older than `3 Ã— median_step_time`. Â§3.2 argues `T_detect`
dominates your RTO and `ASSUMPTIONS.md: large-tensor-fault-32gib` shows the failure mode this
catches is real and silent. It is twenty minutes of work against the largest term in the
equation.

### Not reproducible on this hardware

Stated plainly, because a module that teaches sharding as if we can run it is worse than one
that says we cannot:

- **Sharded save/load, and resharding across world sizes.** Needs â‰¥2 ranks.
  `ASSUMPTIONS.md: single-device-only`; `[C]` `torch._C._distributed_c10d` is incomplete on
  gfx1151. Â§3.5 is design-only. The acceptance test belongs on rented hardware.
- **Asynchronous checkpointing as OLMo-core implements it.** `save_async` requires a separate
  CPU-only process group (`callbacks/checkpointer.py:81`, `checkpoint.py:146`). You can emulate
  the *shape* with a thread, but you cannot exercise the code path.
- **The rename-commit protocol under a real crash.** `_temporary_wd` can be read and reasoned
  about; testing it properly means killing the process at a chosen point in the write and
  checking the directory state, which is doable but is a systems test rather than a
  curriculum exercise. The `CrashingWriter` above is the honest substitute and it is testing a
  simulation, not the real failure.
- **Anything about `fdatasync` durability.** `checkpoint.py:270` skips it on Windows. Verifying
  what that costs would require an actual power-loss test rig.
- **Multi-node checkpoint I/O amplification** (Â§3.5, `[C]` arXiv 2512.24511). Single machine,
  single writer.
- **The economic regime the checkpointing literature is about.** Â§3.1 puts a Laguna-scale
  checkpoint at 1.42 TB and Â§3.3 puts frontier `Î´` in the tens of seconds against an `M` of
  hours. At our scale `Î´` is seconds against an `M` of (probably) many hours, so checkpoint
  overhead never becomes the binding constraint and we cannot generate evidence about the
  regime where it does. Read the compression and async papers to understand the shape of the
  problem, not to adopt their conclusions â€” and do not let a 16Ã— compression number justify
  complexity that buys us nothing.
- **MoE-specific checkpoint sparsity.** Every proposal in that line exploits the fact that only
  a fraction of experts is touched per batch. At 20 Mâ€“300 M with a small expert count the effect
  is not measurable above noise, and `ASSUMPTIONS.md: ablation-scale-sufficient` is `untested`
  on exactly this kind of question.

---

## 7. Self-check

1. A colleague shows you a resume test: "we train 100 steps, checkpoint, restart, train 100
   more, and the final loss matches the uninterrupted run to three decimals." Name two distinct
   reasons this test cannot detect an incomplete checkpoint, and state what you would measure
   instead â€” including the one measurement that must be taken *before* the test is meaningful.

2. Write down `Ïƒ_t`, the full run state, from memory. For each component, say what breaks if it
   is omitted from the checkpoint, and say which component nanoGPT omits that would be inert
   under bf16 but harmful under fp16.

3. Your checkpoint takes 5 seconds to write. You checkpoint every 15 minutes. What mean time
   between failures does that interval assume is optimal, what overhead does it imply, and how
   much worse off are you if the true MTBF is 4Ã— smaller than you assumed?

4. A single bit flips in a saved bf16 weight. Under what circumstances would (a) `torch.load`,
   (b) a NaN/inf check on the loaded weights, (c) a weight-norm monitor, or (d) the training
   loss curve reveal it? Give the fraction of bit positions that any of them catch, and name
   the check that is already in the file and is not being read.

5. You save a checkpoint from a run using one AdamW param group and load it into the same model
   after refactoring into decay/no-decay groups. What happens, is it loud or silent, and what
   property of the on-disk key would have made this safe? State that property as a one-sentence
   criterion.

6. `save_async` in OLMo-core renames the temp directory into place before the model shards have
   finished writing. Why is that not a torn-checkpoint bug, and what single artifact makes it
   safe? What would break the safety property?

*(Answers at the end of this file.)*

---

## 8. What is still unsolved here

**Nobody verifies checkpoint integrity, and the check is already in the file.** `[M]` PyTorch
writes a correct CRC-32 per zip entry and its own loader does not read it; a flipped bit
restores as a finite, plausible weight with a norm ratio of 0.99999994. `[C]` "Exploring Silent
Data Corruption as a Reliability Challenge in LLM Training" (arXiv 2604.00726, Apr 2026)
establishes that SDC is a real and recurring phenomenon at scale, and secondary reporting on
Gemini describes events every one to two weeks. What is *not* established anywhere this survey
could find: what fraction of real SDC events would be caught by a CRC on the checkpoint (versus
occurring in compute, between checkpoints, where no checksum can help), and therefore whether
end-to-end checkpoint verification is worth its one extra read. That is a genuinely open
question and a cheap one to start on.

**Incremental and differential checkpointing exist in papers and in no mainstream trainer.**
`[C]` BitSnap (arXiv 2511.12376, Nov 2025) reports 16Ã— via bitmask sparsification without
accuracy loss and 2Ã— via cluster-based quantization; `[C]` DataStates-LLM (arXiv 2406.10707,
Jun 2024) reports up to 48Ã— faster checkpointing from lazy asynchronous multi-level staging;
`[C]` TierCheck (arXiv 2605.17821, May 2026) surveys the three axes the field optimizes
(asynchronous persistence, compression, host-memory-assisted recovery). None of it is in
PyTorch DCP, OLMo-core, or nanoGPT. The gap between "solved in the literature" and "correct in
the code you are about to run" is the same gap `the-training-loop.md` Â§8 identifies for
gradient-accumulation normalization, and it is the recurring lesson of reading source.

**Resume-equivalence has no standard test and nobody publishes a noise floor.** Every trainer
has a checkpoint; no trainer this lab has read ships a test asserting that a resumed run equals
an uninterrupted one, and no paper reports `Îµ_noise` for its hardware. That makes the whole
literature's implicit claim â€” that interruption is invisible â€” unverified rather than false.
It is also directly load-bearing for us: if resume is not equivalent, interruption history is
an uncontrolled variable in every Themis comparison, and on a laptop that is not a rare event.
`[M]` We now have a floor of exactly zero on gfx1151 at 33 K fp32 parameters with no attention
kernel in the graph, which is the easiest case that exists. **The open question is where it
stops being zero** â€” the candidates are bf16 (`[C]` arXiv 2506.09501 reports bf16 as the most
variance-prone of the three formats), a real reduction over a long axis (where
`ASSUMPTIONS.md: hipblaslt-config` `[M]` already shows a 2.8Ã— error swing from an environment
variable), and SDPA, whose backend on this stack is selected by an environment variable that
does not appear in any artifact. Each of those is a one-hour measurement and none has been
done. Until they are, the bit-exactness criterion is validated only for the toy.

**Whether partial restore is a real category is contested, and our own measurement leans against
the folklore.** `[M]` Splicing blocks between two checkpoints of the same run degraded loss
gracefully (0.0022 nats for one stale block), not catastrophically, in a single underpowered
run. `[C]` CheckFree (arXiv 2506.15461, Jun 2025) goes further and reconstructs a lost pipeline
stage from its neighbours well enough to beat redundant computation. Against that, the
optimizer-coherence argument in Â§6 Exercise B part 3 is unmeasured: nobody has reported what
happens to the *trajectory* after a splice, as opposed to the instantaneous loss. Keep the claim
in its coherence form, mark the loss form as contested, and note that a stronger toy would
settle it in an afternoon.

**Atomicity has been argued to be formally unattainable, and the argument is unreplicated.**
`[C]` "Why Atomicity Matters to AI/ML Infrastructure: Snapshots, Firmware Updates, and the Cost
of the Forward-In-Time-Only Category Mistake" (arXiv 2603.02603, Mar 2026) uses process algebra
to argue that no temporal instant can serve as an atomicity boundary under asynchrony with
crashes, and that checkpoint atomicity is a measure-zero event. Single author, formal-methods
framing, no empirical component, not replicated. Presented as contested rather than as fact,
and flagged here mainly because it is the only work found that treats the consistency-group
question the way a distributed-systems person would.

**Nobody has priced checkpointing on unified memory.** The entire asynchronous-checkpointing
literature is built on the premise that device-to-host staging over PCIe is the bottleneck and
host memory is the abundant tier. On the Z13 both premises invert: the staging copy is a memcpy
inside one physical DRAM pool, and `[M]` the host side is the *small* pool â€” 32 GB, because our
own BIOS carve-out gave 96 GB to the GPU (`ENVIRONMENT.md`, `notebook/uma-carveout-controls-fast-tier.md`).
So async checkpointing may be nearly free here and may also be capacity-blocked at a smaller
model size than on a discrete card. Neither has been measured. Exercise C part 4 is the twenty
minutes that starts it.

**And the one that undercuts everything above.** A bit-exact checkpoint round trip certifies the
*serialization*, not the *run*. `ASSUMPTIONS.md: bf16-numerics-unproven` is still `untested`
against `[C]` five documented bf16 bugs on gfx1151, and `hipblaslt-config` shows a 2.8Ã— swing in
long-reduction error from an environment variable. A perfectly reproducible run of arithmetic
you have not validated is a perfectly reproducible wrong answer â€” which is the whole reason Â§4
argues the environment fingerprint belongs *in* the checkpoint.

---

## Answers to the self-check

**1.** Two reasons, and they are independent. **(a) The loss is a scalar summary of ten million
numbers and is dominated by the data, not by the parameters.** `[M]` A checkpoint missing the
RNG state produced a resumed run in which 100% of parameters differed by up to 1.03e-2, across
three seeds, while the loss stayed in the normal range â€” so "matches to three decimals" is
consistent with every parameter being wrong. **(b) "Three decimals" has no denominator.** Without
knowing what two *uninterrupted* runs do to each other, three decimals might be perfect agreement
or might be four orders of magnitude worse than the floor. What to measure instead: a flat
parameter vector compared elementwise, with the pass criterion `Îµ_resume â‰¤ Îµ_noise`, where
`Îµ_noise` comes first â€” two uninterrupted, identically seeded runs **in separate processes**.
`[M]` On this stack that floor is exactly 0.0 on both CPU and gfx1151 at toy scale, which makes
the criterion bit-exactness â€” but the floor is a property of the model, dtype and environment,
not of the machine, so it is a fixture to be re-measured, not a constant to be hardcoded.

**2.** `Ïƒ_t = (Î¸_t, m_t, v_t, t, Ï_t, c_t, s_t)`.
- `Î¸` parameters â€” omit and there is no resume at all.
- `m`, `v` AdamW moments â€” omit and momentum restarts from zero; the effective optimizer for the
  next ~1/(1âˆ’Î²â‚‚) â‰ˆ 20 steps is not AdamW-as-configured, and the bias correction makes the first
  step large.
- `t` step count â€” omit and both the LR schedule and the bias-correction divisors `1 âˆ’ Î²^t` are
  wrong; at `t=1` the divisor is 0.1, a 10Ã— amplification.
- `Ï` RNG, **four** streams (`random`, `numpy`, `torch` CPU, `torch` CUDA) â€” omit and dropout
  masks and, in a nanoGPT-style loader, the data order diverge from the first step. This is the
  one nanoGPT omits and `[M]` it costs `max|Î”Î¸| â‰ˆ 1e-2` with 100% of parameters differing.
  Restoring only the first three is `[M]` **not enough on GPU**: dropout on a device tensor
  draws from the CUDA generator, leaving 5e-3 across 99.998% of parameters. Four, not three.
- `c` dataloader cursor â€” omit and you re-see data, unless (as in nanoGPT) the cursor is a pure
  function of the RNG, in which case `Ï` covers it. Know which design you have.
- `s` everything else stateful â€” the `GradScaler`'s scale and growth counter, EMA shadows,
  scheduler internals, callback counters, and for us Mnemosyne policy state.
**The component that is inert under bf16 and harmful under fp16 is the `GradScaler`**
(`train.py:196`, constructed fresh and never saved). bf16 does not need loss scaling so the
scaler is disabled; under fp16 a resume restarts the multiplicative search for the scale factor
and throws away optimizer steps on overflow until it converges again.

**3.** `M = Ï„*Â²/(2Î´) = 900Â²/(2Â·5) = 810000/10 = 81,000 s â‰ˆ 22.5 hours.` So a 15-minute interval is
a bet that the machine runs about a day between interruptions. Overhead at the optimum is
`âˆš(2Î´/M) = âˆš(10/81000) = âˆš1.235e-4 = 1.11%`, which splits evenly: `Î´/Ï„ = 5/900 = 0.56%` of
write cost and `Ï„/2M = 900/162000 = 0.56%` of expected redone work, plus the `R/M` restart
floor on top. If the true MTBF is 4Ã— smaller (5.6 h), then the true optimum is
`Ï„* = âˆš(2Â·5Â·20250) = 450 s`, so your 900 s interval is `k = 2`, and the relative penalty is
`(k + 1/k)/2 = (2 + 0.5)/2 = 1.25`. **You are 25% worse than optimal on a quantity that is
already ~2% of runtime.** Which is the practical lesson: being 4Ã— wrong about MTBF costs half a
percent of throughput, so stop tuning it.

**4.** **(a) `torch.load` never reveals it** â€” `[M]` measured, `LOADED WITHOUT ERROR`, because
the reader does not check the CRC-32 that the writer stored. **(b) A NaN/inf check** catches only
flips that push the exponent to the top of its range â€” realistically the exponent MSB, and
sometimes the next two or three bits via a max-magnitude threshold. **(c) A weight-norm monitor**
catches strictly less than (b): it is blind to the sign bit by construction, and `[M]` the
measured norm ratio after a payload flip was 0.99999994. **(d) The loss curve** catches only
flips large enough to move a scalar average over millions of parameters â€” i.e. the same
catastrophic ones as (b), and only if the flip lands somewhere the loss is sensitive to. Of
bf16's 16 bit positions, roughly **4** are catchable by any of these: the exponent MSB and the
next three. The other 12 â€” the whole mantissa, the sign, and the low exponent bits â€” restore as a
plausible weight. **The check that is already in the file and not being read is the per-entry
CRC-32 in the zip container**; `zipfile.ZipFile(p).testzip()` finds it in one line, `[M]`
verified against an unmodified control.

**5.** `[M]` It raises `ValueError: loaded state dict has a different number of parameter
groups` â€” **loud, and loud for the wrong reason**: it noticed the group *count* changed, not that
the parameter-to-state mapping changed. Reorder parameters within a single group and the count is
identical, the load succeeds, and AdamW's momentum for tensor A is applied to tensor B, silently.
The root cause is that vanilla PyTorch keys optimizer state by **integer index into the flattened
param groups**, i.e. by construction order. The criterion that makes it safe: *a checkpoint is
topology- and refactor-agnostic iff every stored extent's key is a function of the logical tensor
identity and index range alone â€” never of rank id, world size, or construction order.* DCP
satisfies it by re-keying optimizer state on parameter FQN
(`training/olmo-core/src/olmo_core/distributed/checkpoint/__init__.py:702`).

**6.** It is safe because **the commit record is a separate file written last, and the reader
requires it**. `_save_metadata` runs in the future's done-callback (`checkpoint.py:174`), after
all shards have landed, and `dir_is_checkpoint` (`checkpoint.py:295`) returns `False` for any
directory lacking `.metadata.json` â€” so `find_checkpoints` will not offer a half-written
directory as resumable. It is the same pattern as invalidating a backup catalog entry before
destroying the data, which `_remove_checkpoint` (`callbacks/checkpointer.py:179`) does in the
other direction. What would break it: any reader that scans for directories by name pattern
instead of calling `dir_is_checkpoint`; a filesystem or object store where the metadata file can
become visible before the shard writes it was ordered after (rename is atomic on NTFS, but
`checkpoint.py:270` skips `fdatasync` on Windows, so *durability* ordering across a power loss is
not guaranteed even though *visibility* ordering is); and, most likely in practice, someone
copying the directory with a tool that does not preserve write order.

---

## Sources

`[C]` arXiv titles, authors and dates were resolved against arxiv.org on 2026-07-26; resolution
proves the paper exists, not that it supports the claim beside it.

**Checkpointing systems and fault tolerance.** J. W. Young, "A first order approximation to the
optimum checkpoint interval," *Communications of the ACM* 17(9), 1974 â€” the `Ï„* = âˆš(2Î´M)` result;
not on arXiv. J. T. Daly, "A higher order estimate of the optimum checkpoint interval for restart
dumps," *Future Generation Computer Systems* 22(3), 2006 â€” the second-order refinement; not on
arXiv. arXiv 2406.18820 â€” Universal Checkpointing: A Flexible and Efficient Distributed
Checkpointing System for Large-Scale DNN Training with Reconfigurable Parallelism (Jun 2024).
arXiv 2406.10707 â€” DataStates-LLM: Lazy Asynchronous Checkpointing for Large Language Models
(Jun 2024). arXiv 2511.12376 â€” BitSnap: Checkpoint Sparsification and Quantization in LLM
Training (Nov 2025). arXiv 2605.17821 â€” TierCheck: Tiered Checkpointing for Fault Tolerance in
Large Language Model Training (May 2026). arXiv 2512.24511 â€” Understanding LLM Checkpoint/Restore
I/O Strategies and Patterns (Dec 2025). arXiv 2506.15461 â€” All is Not Lost: LLM Recovery without
Checkpoints (Jun 2025).

**Reliability and corruption.** arXiv 2604.00726 â€” Exploring Silent Data Corruption as a
Reliability Challenge in LLM Training (Apr 2026). arXiv 2509.16293 â€” Robust LLM Training
Infrastructure at ByteDance (Sep 2025); cited for the existence of a production fault-tolerance
stack, not for any specific checkpoint number, which the abstract does not give. arXiv 2603.02603
â€” Why Atomicity Matters to AI/ML Infrastructure (Mar 2026); single-author formal-methods
argument, presented as contested. arXiv 2407.21783 â€” The Llama 3 Herd of Models (Jul 2024); the
466-interruptions-in-54-days figure used for the MTBF worked example.

**Local.** `research/reference/CODE_MAP.md` â€” "Checkpoint sharding and mid-epoch dataloader
resume" (olmo-core) and "nanoGPT: the known-good run for the Hardware Validation Gate".
`ASSUMPTIONS.md` â€” `single-device-only`, `bf16-numerics-unproven`, `hipblaslt-config`,
`large-tensor-fault-32gib`, `gpu-fast-tier-size`, `sdpa-is-memory-efficient`.
`ENVIRONMENT.md` (32 GB host RAM after the 96 GB UMA carve-out; wheel and driver pins).
`notebook/uma-carveout-controls-fast-tier.md`. `research/notes/pretraining-recipes.md` Â§2 (WSD
trunk-and-branch, which is a checkpointing strategy in disguise) and its checkpoint/resume
sequence diagram. `curriculum/the-training-loop.md` Â§3.4 (the byte inventory this module
subtracts gradients from), Â§3.6 (the `GradScaler` as stateful control loop), Â§3.7 (determinism,
which this module depends on and does not restate).

**Measurements.** All `[M]` numbers in this module were taken on 2026-07-26 with
`torch 2.12.0a0+rocm7.13.0a20260313` (HIP 7.2.0), Python 3.12.10, on native Windows, `cpu` and
`cuda` (gfx1151, AMD Radeon 8060S), **each arm in a fresh subprocess**. Resume-equivalence and
noise-floor results are three seeds (1337/1338/1339) and the full CPU set was run twice on
separate occasions with identical results; the GPU set was run once at three seeds per arm. The
bit-flip result was taken with a control (an unmodified file checked first). The splice result
in Exercise B part 3 is a single run per arm on each device and is labelled an anecdote. The
save/load throughput figures are medians of three on a warm page cache and are therefore an
upper bound on cold-start `T_load`. Nothing here was measured in bf16, at ablation scale, or
with `scaled_dot_product_attention` in the graph, and none of it should be extrapolated to
those conditions without re-measuring.

---

## Decision / Riskiest assumption / Next test

**Decision.** Proteus checkpoints save the full run state â€” parameters, optimizer, step, all
four RNG streams, dataloader cursor, and every stateful component's `state_dict` â€” using a
temp-directory-plus-rename commit with a metadata file as the commit record, CRC verification on
load defaulting to on, GFS retention (3 permanent at a fixed step interval, 1 rolling ephemeral),
and an environment fingerprint recording wheel, arch, and the two numerics-relevant environment
variables. Resume-equivalence becomes a rig invariant in `tests/`, with the noise floor as a
measured fixture rather than an assumed zero. Sharded formats are read and understood, not
built, until there is hardware to test them on.

**Riskiest assumption.** That a noise floor of exactly 0.0 â€” `[M]` measured on gfx1151, but on
33 K fp32 parameters with no attention kernel and no long reduction â€” survives contact with a
real arm. Every resume-equivalence claim here is stated as bit-exactness, and bit-exactness is
only the right bar while the floor is zero. If it goes nonzero at 20 M parameters in bf16, the
criterion becomes a threshold, thresholds drift, and the test loses most of its power. Three
things make this genuinely uncertain rather than pro-forma: `ASSUMPTIONS.md:
bf16-numerics-unproven` is still `untested`, `hipblaslt-config` `[M]` shows a 2.8Ã— error swing
in a long bf16 reduction from an environment variable, and `sdpa-is-memory-efficient` `[M]`
shows the attention backend itself is selected by an environment variable.

**Next test.** Re-run Exercise A's noise-floor arm at ablation scale: the nanoGPT
shakespeare_char config (10.7 M parameters) under bf16 autocast, three seeds, two fresh
processes each, with and without `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`. Roughly thirty
minutes of wall clock. It converts the riskiest assumption above into a measurement, it
produces the first determinism number this lab has at a realistic scale, and whichever way it
comes out it changes the pass criterion for every resume test in `tests/`.
