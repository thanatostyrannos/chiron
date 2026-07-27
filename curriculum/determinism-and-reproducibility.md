---
title: Determinism and reproducibility — what a seed buys, what the environment buys, and which one an ablation actually needs
version: 1.0.0
date: 2026-07-26
owner: curriculum-author
track: D — Training systems
prereqs: the-training-loop, tensors-and-autograd, loss-and-optimization
recommended: long-context-and-effective-context (its Exercise B is the measurement this module is built on)
difficulty: 3/5 conceptually, 4/5 to actually pin down on this hardware
time: ~2 h reading, ~3 h exercises (Exercise C is the long one and is worth a whole evening)
---

# Determinism and reproducibility

## 1. What this module settles

A seed fixes the pseudo-random *streams* a run consumes; it fixes nothing about the *order*
in which floating-point numbers are added, and reduction order is decided by kernel
selection, tile geometry, batch composition and library configuration — none of which is in
the seed. This module separates two properties that the word "reproducible" is used for
interchangeably — **bitwise determinism** (same bits) and **statistical reproducibility**
(same conclusion within measured seed-to-seed spread) — shows with arithmetic why an
ablation needs the second and almost never the first, and shows why the one place bitwise
determinism *does* pay is paired designs, checkpoint round-trip, and regression tests. Two
local facts carry it: `[M]` setting `HIPBLASLT_TENSILE_LIBPATH` and
`TORCH_BLAS_PREFER_HIPBLASLT` changes the error of a length-1,048,576 bf16 reduction by
**~2.8×** with seed, inputs, wheel and device all fixed — so **the environment is part of
the seed** — and `[M]` five fresh-process runs of an identically-seeded 20-step fp32
training loop on this GPU produced **five different parameter digests while printing the
same loss to ten decimal places**, which is what "the metric you watch cannot see the
nondeterminism" looks like when you actually measure it (§3.4).

> This module completes `the-training-loop.md` §3.7, which sketches the same territory in
> two pages as part of a tour of the loop. Read that first; nothing here repeats it. Where
> §3.7 lists five sources of nondeterminism, this module derives the cost of each, prices
> what removing it buys, and states which ones are unremovable on gfx1151.
>
> **Deferred to Track D siblings, deliberately.** Checkpoint format, sharding, atomic commit
> and mid-epoch resume belong to `checkpointing-and-resumption.md`; this module only states
> the determinism *requirement* on a round-trip and why it must be checked on weights. Metric
> buffering, host-device sync cost and the JSONL schema belong to
> `training-telemetry-as-observability.md`; this module only adds the fields the run record
> needs to identify an experimental condition (§4.1). Collective ordering, NCCL algorithm
> pinning and tensor-parallel invariance belong to `distributed-training-strategies.md` and
> are `ASSUMPTIONS.md: single-device-only` design-only for us either way (§4.4).

---

## 2. Theory in plain language

### 2.1 One word, two properties, and the conflation is expensive

**Bitwise determinism**: run the same program twice and every output byte matches. This is
what `torch.equal` tests, what a SHA-256 of a tensor's bytes tests, and what vLLM's
`test_rms_norm_determinism` tests with `rtol=0.0, atol=0.0`
(`memory/vllm/tests/v1/determinism/test_rms_norm_batch_invariant.py:338`).

**Statistical reproducibility**: run the experiment again — new seeds, possibly a different
machine — and reach the same *conclusion*, where "same conclusion" means the effect you
measured is larger than the spread you would have seen from seed noise alone.

These are not two strengths of the same property. They are orthogonal:

| | statistically reproducible | not statistically reproducible |
|---|---|---|
| **bitwise deterministic** | the ideal, and rare | **the dangerous quadrant**: a perfectly repeatable result that is an artifact of one seed. Every run agrees to the last bit and the finding is still false. |
| **not bitwise deterministic** | **the normal, healthy state** of a well-run ablation. Bits differ, conclusions hold. | a broken rig, or a real effect too small to see at this scale |

The dangerous quadrant is the one to internalize. A single-seed comparison that reproduces
bit-for-bit feels *more* trustworthy than one that wobbles in the seventh decimal, and it is
not — it is the same anecdote, told twice, with more confidence. `CLAUDE.md`'s ≥3-seed rule
exists because bitwise agreement is not evidence.

There is a third property, sitting between them, that turns out to matter most for this
lab's actual subject:

**Batch invariance**: the output for a given input row is independent of what *other* rows
are in the batch. vLLM tests this separately from determinism, in the adjacent function
(`memory/vllm/tests/v1/determinism/test_rms_norm_batch_invariant.py:375`), and the
assertion is the whole idea in one line — compute a row alone, compute it again as row 4 of
an 8-row batch, and `torch.equal` the two
(`memory/vllm/tests/v1/determinism/test_rms_norm_batch_invariant.py:397`). A kernel can be
perfectly deterministic (same call, same bits, every time) and not batch-invariant (same
row, different neighbours, different bits). Serving systems are full of exactly that shape,
because the batch is assembled from whatever requests happened to arrive.

### 2.2 What a seed actually is

A seed is the initial state of a pseudo-random number generator. Given it, the sequence of
draws is a deterministic function of *how many draws have been requested so far*. Three
things follow, and the third is the one people get wrong.

**First: there is more than one stream.** Python's `random`, NumPy's global generator,
torch's CPU generator, and torch's per-device generator are four independent objects.
OLMo-core seeds all four in five lines — `random.seed`, `np.random.seed`,
`torch.manual_seed`, `torch.cuda.manual_seed_all`
(`training/olmo-core/src/olmo_core/utils.py:174`–`:179`) — with a comment saying the last
call is deliberately redundant. nanoGPT seeds exactly one
(`training/nanogpt/train.py:106`), which is correct there only because it uses no other RNG.
A library you add later (an augmentation, a sampler, a tokenizer with a random dropout)
brings its own stream, and it will not be in your `seed_all`.

**Second: the streams are consumed in program order, so the *program* is part of the seed.**
Insert one extra `torch.randn` for a debug print, and every subsequent draw shifts. This is
why "same seed" is only meaningful relative to "same code path", and why an `if
config.debug:` branch that draws random numbers is a reproducibility bug.

**Third, and this is the non-obvious one: on GPU the draw sequence is a function of the
launch geometry, not just of the draw count.** The device RNG is counter-based — Philox,
keyed by `(seed, offset)` — and each op that consumes randomness advances the offset by an
amount derived from *how the work was tiled across threads*, rounded up, rather than by the
number of random values it logically needed. `[A]` High confidence; it is the documented
design of PyTorch's `philox_cuda_state` / `calc_execution_policy` path, and it means a
dropout mask on a `[B, T, d]` tensor is not the same mask you get on `[B·T, d]` with the
same seed, and the *following* op's draws move too. **Cheapest test, two lines:**
`torch.manual_seed(0); torch.rand(N, device="cuda"); print(torch.rand(4, device="cuda"))`
for `N ∈ {1, 1000, 1_000_000}` — if the trailing four values are not a pure function of the
count of preceding draws, the offset accounting is launch-dependent. Exercise C runs it.

So the honest statement of what a seed buys: **given identical code, identical shapes,
identical device, identical library versions and identical kernel selection, a seed makes
the random draws identical.** It buys nothing outside that conjunction, the conjunction is
longer than the promise — and `[M]` even *inside* it, on this machine, the draws being
identical did not make the result identical: five fp32 training runs with provably identical
input bytes produced five different models (§3.4e). The seed does its job. Its job is
smaller than the word "reproducible" implies.

### 2.3 The five sources, ordered by what it costs to remove them

| # | Source | Mechanism | Cost to remove | Removable here? |
|---|---|---|---|---|
| 1 | **Unseeded RNG streams** | someone's generator was never seeded | one function, `seed_all` | yes, free |
| 2 | **Reduction order** | float addition is not associative; GEMM and softmax split the sum across threads/tiles/split-k and combine partials in whatever geometry the kernel chose | pin the kernel and the split; lose throughput | partly — see §5, and no batch-invariant library covers ROCm |
| 3 | **Atomics** | `atomicAdd` completion order is genuinely nondeterministic across runs; used in embedding backward, `scatter_add`, `index_add`, and fused-MoE finalize | use a two-pass deterministic path; lose throughput | `[M]` **yes for the ops we use** — §3.4(d) measured the gfx1151 coverage map and embedding backward, `scatter_add` and `index_add` all have deterministic implementations available here |
| 4 | **Kernel selection and library configuration** | heuristic algorithm choice, autotuning caches, and *environment variables that switch which library implements a GEMM* | pin the environment and record it | yes — and this is the one nobody records |
| 5 | **Dataloader** | worker completion order, prefetch depth, resume cursor, epoch/seed derivation | derive order arithmetically instead of consuming it from a queue | yes, by design — OLMo-core shows how |

Sources 1 and 5 are engineering. Source 4 is bookkeeping, and it is the one this lab got
wrong once (§4.3). Sources 2 and 3 are physics-adjacent: they exist because the hardware
reorders reductions in order to be fast, and you cannot seed your way out of them.

**And yet.** `[M]` With all five nominally under control — one seed, one environment, one
device, provably identical inputs — twenty steps of fp32 training on this machine produced a
different model in every one of five fresh processes (§3.4e). So the list above is the list
of *mechanisms*, not a checklist that, once completed, delivers determinism. Read it as a
differential diagnosis.

### 2.4 The bridges, and where each one breaks

| You already own | Its counterpart here | Where the analogy breaks |
|---|---|---|
| **Idempotent replay / exactly-once delivery.** Make the operation commutative, replay freely. | A seeded rerun | Float addition is *not* associative and the hardware reorders deliberately. You cannot make the op commutative; you can only pin the order and pay, or accept drift. There is no "at-least-once plus dedup" escape hatch. |
| **Reproducible builds.** Same source hash + same toolchain hash → same binary. The toolchain is *in* the hash. | A reproducible run | The toolchain is **not** in the seed, and no framework puts it there. `[M]` Two runs on this machine, same seed, same wheel, same device, differing only in whether two environment variables were exported, produce a length-1M bf16 reduction whose error differs by 2.8× (§3.3). A build system would call that a different build. Your run record calls it the same run. |
| **A checksum on a replica.** Compare the bytes, know the copy is good. | Checkpoint round-trip verification | Works, and is required — the Hardware Validation Gate demands bit-exact save/load. But you can only checksum a *state*, never a *trajectory*: there is no invariant a training run maintains that a checker could verify, so a corrupted step is undetectable in principle, not just in practice (`the-training-loop.md` §2.1). |
| **Consistent hashing / stable sharding.** Derive placement from a key so no map has to be stored. | Data order derived from `(seed, epoch)` — `training/olmo-core/src/olmo_core/data/data_loader.py:673` | Same idea, executed well: order is `PCG64(seed + epoch)`, never persisted, and a resume recomputes it. The break is in the arithmetic — it is `seed + epoch`, an **addition**, so `(seed=1000, epoch=2)` and `(seed=1001, epoch=1)` are the *same permutation*. A collision in your seed space that you would never see because you never ran both. |
| **Flaky-test quarantine.** Nondeterminism in a test is a defect; find it and kill it. | Seed-to-seed variance in a run | Inverted. Seed variance is not a defect to be eliminated — it is the **measurement error term**, and you need an estimate of it to size any effect at all. A lab that suppressed seed variance would have destroyed its own error bars. Kill *within-seed* nondeterminism if you like; never kill *between-seed*. |
| **Blue/green with identical container images.** Identical image → identical behaviour. | Two matched ablation arms | Identical image is necessary and not sufficient — but for a subtler reason than the textbook one. The textbook break is batch composition: sequence lengths or expert routing differ between arms, the reduction split changes, the numbers change. `[M]` §3.4(b) did *not* find that for a plain GEMM here, so on our hardware the live break is different and worse — `[M]` §3.4(e), where five processes with an identical image, identical seed and identical inputs produced five different models. Two arms can be byte-identical in every input and still not be comparable at the last decimal places. |
| **A hot standby that must be bit-identical to fail over.** | Prefix-cache reuse vs recompute | The KV bytes a prefix cache serves were produced by a prefill of one length; a recompute of the same prefix inside a longer request may tile differently and produce *different bits*. §4.2. A storage tier that returns a slightly different value than the backing store is a corruption bug. Here it is the documented design, and nobody checks. |

---

## 3. The math that actually matters

### 3.1 Non-associativity, and why the worst-case bound is useless

Every floating-point addition is exact-then-rounded:

```
fl(a + b) = (a + b)(1 + δ),      |δ| ≤ u
```

- `fl(·)` — the value actually stored after the hardware rounds
- `δ` — the relative rounding error introduced by this one operation
- `u` — **unit roundoff**, the largest relative error a single round-to-nearest can make.
  For a format with `m` *stored* mantissa bits the significand carries `m+1` bits of
  precision (the leading one is implicit), values near 1.0 are spaced `eps = 2^−m` apart, and
  round-to-nearest can be off by at most half a spacing: `u = eps/2 = 2^−(m+1)`.

| format | `m` | spacing at 1.0 (`eps = 2^−m`) | `u = eps/2` |
|---|---|---|---|
| fp32 | 23 | 1.19e−7 | 5.96e−8 |
| fp16 | 10 | 9.77e−4 | 4.88e−4 |
| bf16 | 7 | 7.81e−3 | **3.91e−3** |

Now sum `n` numbers `x_1 … x_n`. Two orders:

**Sequential** (accumulate into one register, left to right). The classical bound:

```
|Ŝ − S|  ≤  (n − 1) · u · Σ|x_i|  +  O(u²)
```

- `S` — the exact sum, `Ŝ` — the computed one
- `Σ|x_i|` — sum of magnitudes, which is ≥ `|S|` and can be enormously larger

**Pairwise / tree** (split in half, recurse, add the two halves — what every GPU reduction
does):

```
|Ŝ − S|  ≤  (log₂ n) · u · Σ|x_i|  +  O(u²)
```

Divide the two. **Reduction order alone changes the worst-case error bound by a factor of
`(n − 1) / log₂ n`.** At `n = 2²⁰ = 1,048,576` that is `1,048,575 / 20 ≈ 52,000×`. Same
inputs, same format, same mathematics; five orders of magnitude of difference in how wrong
you are allowed to be.

Introduce the **condition number of the sum**:

```
κ  =  Σ|x_i| / |Σ x_i|
```

- `κ = 1` when all terms have the same sign — a benign sum
- `κ → ∞` under cancellation, when large terms nearly cancel

Then the relative error bounds are `(n−1)·u·κ` and `(log₂ n)·u·κ`.

**And here is why you should not trust the bound.** Put our numbers in: `n = 2²⁰`, bf16, and
suppose the accumulator is also bf16. Sequential: `(n−1)·u = 1,048,575 × 3.91e−3 = 4,096`.
A relative error bound of 4,096 is not a bound, it is a shrug. Even the tree bound gives
`20 × 3.91e−3 = 0.078`, i.e. "up to 8% wrong."

**The practical conclusion is the important one: worst-case error analysis cannot tell you
whether your kernel is fine. Only measurement can.** That is not a rhetorical flourish; it
is why the Hardware Validation Gate is a measurement suite rather than a spec review, and
why every claim in this module about *our* arithmetic is `[M]` or nothing.

What analysis *is* good for is predicting the **floor**. Rounding `x` to bf16 gives a
relative error roughly uniform on `[−u, u]`, whose RMS is `u/√3 = 2.26e−3`. So a bf16
matmul whose inputs were rounded from higher precision has a relative-error floor of a few
times `1e−3` **no matter how good the accumulator is**. Any measurement near `2e−3` is at
the floor; anything well above it is telling you about the accumulation.

### 3.2 Where the reduction split actually comes from

A GEMM `C = A·B` with inner dimension `K` computes each output element as a length-`K` sum.
The kernel tiles that sum:

```
K  =  k_split · K_tile
```

- `k_split` — how many independent partial sums the kernel splits the reduction into
  ("split-k"), each computed by a different threadblock
- `K_tile` — the length each threadblock accumulates sequentially in registers
- the `k_split` partials are then combined, either by a second pass or by atomics

Three consequences, and every one of them is a determinism problem:

1. **`k_split` is chosen by a heuristic** from `M`, `N`, `K`, dtype and occupancy targets.
   Change the batch size and `M` changes and `k_split` can change. That is the mechanism of
   batch non-invariance, stated concretely. FlashInfer exposes the fix as an API parameter:
   `fixed_split_size`, documented as leading to "deterministic softmax score reduction in
   the merge_states kernel, and therefore batch-size invariant outputs"
   (`memory/flashinfer/flashinfer/decode.py:1316`) — and the docstring cites the Thinking
   Machines write-up by URL, which is as close to an admission as upstream source gets.
2. **The combine step may use atomics.** FlashInfer's fused-MoE finalize says so in the
   docstring: "The fused epilogue reduces expert outputs via non-associative atomics, so
   results are not deterministic run-to-run. Set to False to use the non-fused,
   deterministic finalize path" (`memory/flashinfer/flashinfer/fused_moe/core.py:1038`).
   That is a throughput-versus-determinism switch, exposed as a boolean, with the cost
   unstated.
3. **Which library provides the GEMM at all is an environment decision.** On ROCm,
   `TORCH_BLAS_PREFER_HIPBLASLT` and `HIPBLASLT_TENSILE_LIBPATH` decide whether the
   hipBLASLt path with its Tensile-generated kernels is used or a fallback is. Different
   library, different tiling, different `k_split`, different reduction tree — from two
   environment variables that appear in no run record anywhere.

### 3.3 Our own measurement, and what it licenses you to say

`[M]` 2026-07-26, `scripts/measure_bf16_reduction_error.py` plus an environment matrix,
`torch 2.12.0a0+rocm7.13.0a20260313` (HIP 7.2.0), gfx1151, native Windows. Shape:
`p ∈ R^N` a softmax weight vector with 50% of its mass on index 0, `V ∈ R^{N×128}`,
`N = 1,048,576`, both cast to bf16 from a float64 draw made **once on CPU and shared
bit-identically with the device**. Reference is the float64 product. Three seeds
(1337/1338/1339); the `N = 1M` rows were re-run in a fresh process and reproduced to every
printed digit; the same-seed GPU arm run twice gave `torch.equal == True` in all six cases.

| Path | relative L2 error vs fp64 | vs the same CPU reference |
|---|---|---|
| CPU | 1.85e−3 | 1.00× |
| GPU, hipBLASLt **configured** | 2.01e−3 | **1.09×** |
| GPU, hipBLASLt **not configured** | 5.60e−3 | **3.03×** |

Read it against §3.1. The bf16 input-rounding floor is `u/√3 = 2.26e−3`. CPU (1.85e−3) and
configured GPU (2.01e−3) are **at the floor** — consistent with an fp32 accumulator, whose
own contribution at depth 20 would be `20 × 5.96e−8 ≈ 1.2e−6`, four orders below the floor
and invisible. The unconfigured GPU path is **2.79× above the configured one**, which is
above the floor and therefore is telling you about the accumulation, not the inputs.

Three things this licenses, and one it does not.

**Licensed 1 — the environment is part of the seed, and this is measured, not argued.**
Seed fixed, inputs bit-identical, wheel fixed, device fixed, code fixed. Two environment
variables. 2.8× in accuracy. Any run whose conclusion depends on the third significant
figure of a long reduction is a different experiment depending on how the shell was set up.

**Licensed 2 — hipBLASLt is a numerics control on this machine, not a throughput tweak.**
`ASSUMPTIONS.md: hipblaslt-config` originally recorded it as a `[C]` 5× throughput cliff;
`[M]` measurement found +12% throughput (18.6 → 20.9 TFLOPS bf16 at 8192³) and 2.8×
accuracy. The row is now "refuted as a throughput cliff; upgraded to a NUMERICS control."
Every run must record whether it was configured.

**Licensed 3 — a knob that reports a capability can be inert.**
`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction` — the obvious suspect —
changes **exactly zero bits** here, in both hipBLASLt configurations
(`ASSUMPTIONS.md: bf16-reduced-precision-knob-works`, refuted). It is present in the API and
does nothing on this stack. This is the same failure shape as
`torch.backends.cuda.flash_sdp_enabled()` returning `True` while SDPA dispatches to the math
backend (`ASSUMPTIONS.md: sdpa-is-memory-efficient`): **the API reports what is permitted,
not what ran.** On an unvalidated stack, a flag is a hypothesis.

**Not licensed — a mechanism.** We do not know *what* changes. Not the accumulator dtype
(untested directly), not the split factor, not the tile shape. We know the environment
selects a different reduction and we know the size of the effect. Anyone who tells you
which, from this data, is guessing.

**And the methodology is the transferable part.** This measurement was first published in
`long-context-and-effective-context.md` as "GPU is 3× worse than CPU at 1M keys" — a real,
replicated, three-seed, fresh-process effect, attributed to the wrong cause because the
environment had not been controlled. It was corrected the same day, by re-running under the
matrix. What made it findable in one attempt was that the original write-up ended with an
explicit list of what had *not* been tested, and the missing variable was sitting in it.
Write that list. Every time.

### 3.4 What this module measured, including two predictions it refuted

`[M]` 2026-07-26, `torch 2.12.0a0+rocm7.13.0a20260313` (HIP 7.2.0), AMD Radeon 8060S /
gfx1151, native Windows, driver 32.0.23033.5002, hipBLASLt configured unless stated.
**Every cell ran in a fresh subprocess** — a driver launched the child, read one JSON line,
and compared SHA-256 digests of raw tensor bytes; no process was ever reused. Repeat counts
are given per row because they are the evidence, and the asymmetry matters: **one
disagreement proves nondeterminism; two agreements do not prove determinism.**

**(a) The environment is part of the seed at the bit level — for some shapes and dtypes, and
not others.** Two GEMMs, `torch.manual_seed(1337)`, operands generated on device, input
digests identical in every run (the control).

| GEMM | dtype | fresh runs, hipBLASLt **on** | fresh runs, hipBLASLt **off** | within-config | across-config |
|---|---|---|---|---|---|
| `[4096,4096]@[4096,4096]` | bf16 | 2 → `cbc45a30805eace6` | 2 → `cbc45a30805eace6` | identical | **identical** |
| `[2048,2048]@[2048,2048]` | fp32 | 3 → `a5f336dc8313619d` | 3 → `bbeccbb74eac04fc` | identical | **different** |

(The bf16 rows also agree on `Σ C` computed in fp64: −36480.65454244614 in all four
processes.)

Read both rows together, because either alone misleads.

- **Every one of the ten processes was internally reproducible.** Within a fixed
  environment, a fresh-process repeat of a GEMM gave identical bits, 2/2 and 3/3. So a GEMM
  on this stack is not randomly nondeterministic.
- **The fp32 pair proves the environment claim bitwise.** Same seed, same inputs (proved by
  the identical input digests), same wheel, same device, same code. Two environment
  variables. Different output bits, replicated three times on each side. There is no
  softer way to state it: **`seed: 1337` does not identify an experimental condition on this
  machine.**
- **And the bf16 pair refutes the prediction the first draft of Exercise A made.** "High
  confidence the bits differ" was wrong for bf16 at 4096³.

**The reconciliation is the finding, and stating it carefully is the exercise.** Three cells
are now known:

| cell | environment matters? |
|---|---|
| bf16, `[1, 2²⁰] @ [2²⁰, 128]` (§3.3, accuracy) | **yes**, 2.8× |
| fp32, `[2048,2048] @ [2048,2048]` (bitwise) | **yes** |
| bf16, `[4096,4096] @ [4096,4096]` (bitwise) | **no** |

Three cells cannot separate two candidate axes. A tempting story — "it bites on
deep-reduction, narrow-output shapes, i.e. decode attention over long context, i.e. exactly
this lab's subject" — explains cells 1 and 3 and is contradicted by cell 2, where a fat
square fp32 GEMM changed its bits. A second story — "it is a dtype effect; fp32 has more
implementation choice than bf16, which goes down one matrix-core path" — explains cells 2 and
3 and says nothing about cell 1. **Both stories are live and this module is not going to
pick one**, because picking one from three cells is precisely the error §3.3 documents this
lab already making once. What is established is the safe operational conclusion, which does
not need the mechanism: **the environment changes results for some real shapes on this
machine, you cannot predict which from first principles, therefore pin it and record it for
everything.** Exercise A maps the boundary properly: 16 cells, about 50 minutes unattended,
and almost all of that is torch imports.

The one thing worth carrying anyway: do not benchmark determinism on a fat square GEMM and
generalize to your workload. Cell 3 is the shape a vendor benchmark uses and it is the one
cell where nothing happened.

**(b) A bf16 GEMM is batch-invariant here, across three orders of magnitude of batch size.**
`X[:bs] @ W` with `W` `[4096,4096]` bf16 and `X` `[1024,4096]` bf16; digest of **row 0
only**, whose input bytes are identical in every case.

`bs ∈ {1, 2, 4, 16, 64, 256, 1024}` → **all seven digests identical**, max abs difference
exactly 0.0 against `bs=1`, row magnitude `|row0|_max = 210.0`. Two fresh processes, same
result. **This refutes the second prediction in Exercise B.**

The batch-variance effect that `[C]` Thinking Machines documented and that FlashInfer
exposes a knob to suppress does **not** appear for this op at these shapes on this
hardware. Three caveats keep it from being a general claim: this is a plain GEMM, not
attention with ragged KV lengths; only `M` varies while `K` is fixed; and nothing here goes
through a split-kv merge kernel. The serving-shaped case — which is where
`memory/flashinfer/flashinfer/decode.py:1316` puts its fix — remains untested.

**(c) Writing the same sum two ways changes the answer by 15×, with no nondeterminism
involved.** `v` `[4,194,304]` fp32 on GPU, seed 1337, two fresh processes, identical digits
in both.

| expression | value | relative error vs fp64 |
|---|---|---|
| `v.sum()` | −2.5444125977e+03 | 1.2e−8 |
| `v.view(2048,2048).sum()` | −2.5444125977e+03 — **bitwise equal to the flat sum** | 1.2e−8 |
| `v.view(2048,2048).sum(dim=1).sum()` | −2.5444130859e+03 | **1.8e−7** |
| fp64 reference | −2.5444126282e+03 | — |

Same numbers, same seed, same environment, same process. Reshaping does not change the
reduction (the flat and 2-D sums are bitwise equal, so the kernel flattens anyway), but
expressing it as row-sums-then-sum picks a different tree and lands **15× further from the
truth**. §3.1's `(n−1)` versus `log₂ n` is not an abstraction; this is it, in four lines of
your own code, and no seed protects you from it.

**(d) The gfx1151 deterministic-op coverage map** — as far as this lab can tell, the first
time this has been written down for this architecture. `torch.use_deterministic_algorithms(True,
warn_only=False)`, one fresh process, each op in its own `try/except`:

| op | status |
|---|---|
| `embedding_dense_backward` | **ok** |
| `scatter_add_` | ok |
| `index_add_` | ok |
| `index_put_(accumulate=True)` | ok |
| `bincount` | ok |
| `index_select` backward | ok |
| `interpolate` bilinear backward | ok |
| `kthvalue`, `median`, `cumsum`, `scatter_reduce_(reduce="sum")` | ok |
| `grid_sampler_2d_backward` | **RuntimeError** — "grid_sampler_2d_backward_cuda does not have a deterministic implementation" |
| `max_pool3d` backward | **RuntimeError** — "max_pool3d_with_indices_backward_cuda does not have a deterministic implementation" |

Read "ok" precisely: it means PyTorch **has** a deterministic implementation for that op on
this backend and selected it. It does not mean the default (non-deterministic-mode) kernel
is deterministic. The good news is specific and load-bearing: **the three ops that appear in
a transformer backward pass — embedding backward, `scatter_add`, `index_add` — all have a
deterministic path available on ROCm here.** The two that raise appear in vision models, not
in ours. So bitwise-deterministic training is not blocked at the op level.

One gap, stated because it is the obvious next cell: **matmul under the flag was not
exercised.** On CUDA, `use_deterministic_algorithms(True)` plus a cuBLAS GEMM raises unless
`CUBLAS_WORKSPACE_CONFIG` is set; nothing in this run triggered that path, so whether ROCm
has an analogous requirement is still unknown.

**(e) The headline. GPU fp32 training is not bitwise reproducible across fresh processes,
and the loss curve does not show it.**

Model: 2-layer decoder, `d=128`, 4 heads, block 64, vocab 256, dropout 0, **468,224
parameters**. AdamW `lr=1e-3`, `betas=(0.9,0.95)`, 20 steps, batch `8×64`.
`torch.manual_seed(1337)` for init; input tokens drawn from a **separate CPU generator
seeded 4242** and copied to the device, so the input bytes are provably identical across
runs. Digest = SHA-256 over all parameters concatenated after step 20.

| condition | fresh processes | parameter digests | loss step 0 | loss step 19 | bitwise reproducible? |
|---|---|---|---|---|---|
| **GPU fp32** | 5 (four configured, one not) | `5ae0d0e6343f7323`, `bb05aa2011d3850e`, `b7e909b0c0c3e06a`, `f0034b60c58376bf`, `bea9e5ebba106e11` — **five for five, all different** | 5.7192234993 (all five) | 5.6602668762 (all five) | **NO** |
| **GPU bf16 autocast** | 4 | `52473c4ff264434e` ×4 | 5.7190856934 | 5.6598205566 | yes, 4/4 |
| **CPU fp32** | 2 | `06c2e7bf0073f599` ×2 | 5.7192239761 | 5.6602678299 | yes, 2/2 |

Four readings.

1. **Five GPU fp32 runs, five distinct parameter digests, and all five printed the same loss
   to ten decimal places.** The nondeterminism is real, it is in the weights, and the metric
   you would have been watching cannot see it. This is the module's central claim, measured,
   on our machine.
2. Four of those five shared one environment and still differed from each other, so the
   fifth (unconfigured) run's distinct digest **cannot be attributed to the environment** —
   run-to-run variation alone produces a fresh digest every time inside one configuration.
   Do not read this table as an environment result; (a) is the environment result. Note also
   how cleanly this separates from (a): a *single* fp32 GEMM is perfectly reproducible 3/3,
   but *twenty training steps* in fp32 are not reproducible 0/5. The nondeterminism is
   therefore not in the forward GEMM; it is somewhere in the backward pass or the optimizer,
   and identifying where is a bisection this module does not have time for.
3. **bf16 reproduced where fp32 did not**, which is the opposite of the naive import of
   `[C]` arXiv 2506.09501 ("bf16 reproduces worst"). That paper is about inference across
   batch sizes and frameworks, not run-to-run training on one device, so it is not
   contradicted — but its conclusion does not transfer, and 4/4 is four, not a property.
   `[A]` Medium confidence on the mechanism: the autocast path likely lands on a fixed
   hipBLASLt bf16 kernel while fp32 lands on a heuristic or autotuned selection that can
   vary per process. Untested; the cheapest probes are
   `torch.backends.cuda.preferred_blas_library()` on both arms, and re-running the fp32 arm
   with `TORCH_BLAS_PREFER_HIPBLASLT=0`.
4. **20 steps is a smoke test, not a trajectory.** The fp32 divergence is already below the
   loss's tenth decimal place at step 20; over 20,000 steps it compounds through 20,000
   nonlinear updates, and the fp32-versus-bf16 ordering could easily invert. Nothing here
   licenses a claim about a real run.

**How big is the divergence, and is it alarming?** This run digested the parameters but did
not record a max-absolute-difference, which is a gap in the harness worth closing. The best
bound available comes from a different measurement: `[M]` `tensors-and-autograd.md` §2.4
found **fp32 gradients on gfx1151 agree with CPU to 3.9e−8 absolute, ~1.1e−6 relative to
`|grad|_max`** (one seed, a 2-layer 64-wide GPT — an anecdote by the house standard, and the
only gradient-correctness evidence this machine has). `3.9e−8` is *below* fp32's unit
roundoff `u = 5.96e−8`, i.e. the GPU fp32 gradient path is at the rounding floor and there
is no correctness problem hiding here. Put the two together and the picture is
unambiguous: **the per-step perturbation is a last-bit rounding difference, not a bug — and
it still produces a different model in five out of five runs.** That is the whole reason
§3.7's discontinuity argument matters. A last-bit difference is harmless everywhere it stays
continuous, and decisive at every `argmax`, `top-k`, and threshold in the system.

**What this does to the Hardware Validation Gate.** `CLAUDE.md` requires "determinism across
repeated runs with a fixed seed." `[M]` **That check fails today for GPU fp32 training on
this machine**, passes 4/4 for GPU bf16 autocast, and passes 2/2 on CPU. It is not a blocker
for the ablation programme — §3.5 and §3.6 argue that statistical reproducibility is what an
ablation needs — but it *is* a blocker for the paired-design strategy in §3.6 and for any
regression test that compares weights rather than metrics. Record it, decide deliberately,
and note that the gate item as written does not say which dtype it means.

### 3.5 The estimator an ablation actually needs

Two arms, `A` and `B`, run at `n` seeds each. Let `X_i` be the metric on seed `i`
(validation loss, recall@k, whatever the pre-registration named).

```
Δ̂  =  X̄_A − X̄_B                          the estimated effect
SE  =  σ · √(2/n)                          its standard error, σ = per-seed sd
```

- `X̄_A` — mean over arm A's seeds
- `σ` — the standard deviation of the metric *across seeds within an arm*
- `√(2/n)` — because two independent means are being differenced

The **minimum detectable effect** — the smallest true difference you have an 80% chance of
calling significant at α = 0.05 two-sided:

```
MDE  ≈  ( t_{0.975, 2n−2} + t_{0.80, 2n−2} ) · σ · √(2/n)
```

- `t_{0.975, df}` — the critical value; how far out you must be to reject
- `t_{0.80, df}` — the power term; how far out you must *typically* land to reject 80% of
  the time
- `df = 2n − 2` — degrees of freedom, and at small `n` this term hurts badly

Worked, because the numbers are the point:

| seeds per arm | df | `t_{0.975}` | `t_{0.80}` | `√(2/n)` | **MDE** |
|---|---|---|---|---|---|
| 3 | 4 | 2.776 | 0.941 | 0.816 | **3.04 σ** |
| 5 | 8 | 2.306 | 0.889 | 0.632 | **2.02 σ** |
| 10 | 18 | 2.101 | 0.862 | 0.447 | **1.33 σ** |
| 20 | 38 | 2.024 | 0.851 | 0.316 | **0.91 σ** |

(Normal-approximation-with-t form; the exact noncentral-t calculation gives ~3.1 σ at
`n = 3` rather than 3.04, so this table is very slightly optimistic at the small end.)

**The house rule of ≥3 seeds buys you the ability to detect an effect roughly three times
the seed-to-seed standard deviation. Nothing smaller.** That is not a criticism of the rule
— three seeds is the right floor for a one-person lab, and it is infinitely better than one
— but it is the honest statement of what it purchases, and it should be quoted in every
pre-registration's SUCCESS threshold. If your predicted effect is 1σ, three seeds cannot
find it and running them is a waste of the wall-clock budget in
`research/notes/pretraining-recipes.md` §5.

Now decompose the variance:

```
σ²  =  σ²_init  +  σ²_data  +  σ²_numeric
```

- `σ²_init` — variance from weight initialization
- `σ²_data` — variance from data order / sampling
- `σ²_numeric` — variance from nondeterministic reductions and atomics, at fixed init and
  data order

**Making the run bitwise deterministic sets `σ²_numeric = 0` and leaves the other two
untouched.** `[A]` Medium-high confidence that `σ²_numeric` is also the *smallest* of the
three by a wide margin — that is the consistent finding in the fine-tuning literature
(`[C]` arXiv 2002.06305, arXiv 2503.07329) and it is what §3.4(e) suggests locally, where
the fp32 per-step perturbation sits at the fp32 rounding floor. It has **never been measured
for pretraining at 20M–300M**, which is §8's nine-run gap. Taking it as given: bitwise
determinism moves the MDE by approximately nothing. **This is the mathematical statement of
why an ablation needs statistical reproducibility and not bitwise determinism**, and it is
worth being able to write down rather than assert — including writing down that it rests on
an unmeasured ordering.

### 3.6 The one place bitwise determinism does buy statistical power

Pair the arms. Run arm A and arm B at the *same* init seed and the *same* data order, and
analyse the differences `D_i = X_{A,i} − X_{B,i}`:

```
MDE_paired  ≈  ( t_{0.975, n−1} + t_{0.80, n−1} ) · σ_D / √n
```

- `σ_D` — standard deviation of the **per-seed difference**, not of the metric

At `n = 3`: `df = 2`, `t_{0.975,2} = 4.303`, `t_{0.80,2} = 1.061`, `√n = 1.732`, so
`MDE = 3.10 σ_D`. Note the coefficient is *worse* than unpaired (3.10 vs 3.04) because you
gave up degrees of freedom. **The entire win is that `σ_D` can be far smaller than `σ`.**

Write the difference out. If the arm effect `Δ` is the same on every seed:

```
D_i  =  Δ  +  (numeric noise only)        ⟹  σ_D² = 2σ²_numeric  ≈  0
```

and the test becomes arbitrarily powerful. If the effect interacts with the seed —
`Δ_i = Δ + ε_i` — then `σ_D² = σ²_ε + 2σ²_numeric` and you learn something real: the size of
`σ_D` relative to `σ` *is* the measurement of how seed-dependent your effect is.

**So the payoff of controlling numerics is not "the numbers match." It is that pairing
becomes valid, and pairing is what makes a 3-seed budget capable of resolving a 1σ effect.**
This is the argument for the determinism work in this module, and as far as this survey
found, nobody in the LLM-ablation literature states it.

**The honest caveat, and it is a large one.** Pairing requires the two arms to share an init
seed *meaningfully*. That works for a policy ablation where the model is identical and only
the memory policy differs — which is exactly Mnemosyne's shape, `mnemosyne-h2o` vs
`mnemosyne-window` on the same checkpoint. It works only partially for an architecture
ablation like `proteus-swa-4to1` vs `proteus-dense`, because the parameter tensors have
different shapes and "the same seed" does not produce matched initializations. **Design
implication: memory-policy ablations should be paired and can be; architecture ablations
cannot be fully paired and need more seeds.** Those are two different seed budgets and the
Themis rig should know the difference.

### 3.7 Top-k under numerical error: the discontinuity that matters for Mnemosyne

Everything above is about continuous outputs, where an `ε` perturbation produces an `ε`
change. Eviction is not continuous. `top-k` is a *discontinuous* function of its input: an
arbitrarily small perturbation flips the selected set whenever two scores straddle the
boundary within that perturbation.

Let `s_(1) ≥ s_(2) ≥ … ≥ s_(N)` be the sorted attention-mass scores and let
`g = s_(k) − s_(k+1)` be the **boundary gap** — the margin between the last token retained
and the first token evicted. If the numeric error in each score is bounded by `ε`, then:

```
g  >  2ε   ⟹  the retained set is unchanged
g  <  2ε   ⟹  the retained set is arbitrary at that boundary
```

For scores drawn from a smooth distribution with density `f` near the boundary, the
probability that any given boundary is within error is approximately `2 ε f`, and with a
cache of `N` tokens evicted to `k` there are many near-boundary pairs, so the expected
number of flipped slots grows with `N` and with `ε`.

Put our number in. `[M]` The relative error of a long bf16 reduction on this machine is
`2.0e−3` configured and `5.6e−3` unconfigured. **Any two tokens whose accumulated attention
mass differs by less than about 0.2% of the score scale are, for eviction purposes,
tied — and which one survives is decided by the environment, the batch composition, and the
kernel's tile geometry.** `[A]` High confidence in the reasoning; the effect size on an
actual eviction policy is **unmeasured** and Exercise B measures it.

Three consequences for Mnemosyne, and they are design consequences, not commentary:

1. **The policy needs an explicit, deterministic tie-break.** Break ties by token position
   (oldest wins, or newest wins — pick one and write it down). A policy whose tie-break is
   "whatever `torch.topk` returned" has made the eviction decision a function of the kernel.
2. **The policy needs a boundary-margin telemetry channel.** Emit, per eviction event, the
   gap `g` at the cut and the fraction of retained tokens whose score was within `2ε` of an
   evicted one. That number is an *attribution* instrument: if a recall regression correlates
   with a shrinking margin, the cause is arithmetic, not policy. `CLAUDE.md` asks for
   attribution rather than outcome; this is a concrete instance nobody publishes.
3. **Two identical requests can get different caches** — *if* the score computation is
   batch-dependent. Because then the batch a request lands in determines the reduction split,
   the split determines the score bits, and the bits decide the boundary; in a serving system
   that is a user-visible nondeterminism with a *policy* cause rather than a sampling cause,
   and it will not show up in any temperature-0 determinism test that only checks logits.
   **Honesty about the antecedent:** `[M]` §3.4(b) found a bf16 GEMM batch-invariant 7/7 on
   this machine, so this leg is *not* currently supported here and may be imported from
   NVIDIA-shaped worry. It is untested for the shape that would actually carry it — attention
   with ragged KV lengths through a split-kv merge. Consequences 1 and 2 do **not** depend on
   this; they follow from run-to-run and environment-to-environment variation alone, both of
   which are `[M]` established (§3.4a, §3.4e).

---

## 4. Why it matters for Proteus and Mnemosyne

### 4.1 Themis must record the environment, or the arms are not matched

`CLAUDE.md` requires matched param counts and token budgets across arms. Matched
*arithmetic* is not in the list and needs to be. Concretely, every run's JSONL header
(schema in `research/notes/pretraining-recipes.md` §9) needs a numerics fingerprint
alongside `seed`:

```jsonc
{ "seed_init": 1337, "seed_data": 34521, "seed_dropout": 90210,
  "torch": "2.12.0a0+rocm7.13.0a20260313", "hip": "7.2.0",
  "gpu_driver": "32.0.23033.5002", "device": "gfx1151",
  "hipblaslt_configured": true,                 // [M] worth 2.8x in long-reduction accuracy
  "hipblaslt_tensile_libpath": "...",
  "aotriton_experimental": false,               // [M] changes SDPA backend AND numerics
  "autocast_dtype": "bfloat16",
  "deterministic_algorithms": false,
  "matmul_fp32_precision": "ieee",
  "env_fingerprint_sha256": "..."               // hash of the sorted (name,value) pairs above
}
```

Three seeds, not one, per `the-training-loop.md` §4: `seed_init`, `seed_data`,
`seed_dropout`. They answer different questions and must be independently settable, because
"does this survive a different data order?" and "does it survive a different
initialization?" are different experiments.

The `env_fingerprint_sha256` is the reproducible-builds move: if two runs' fingerprints
differ, they are not the same experiment, and the rig should say so loudly rather than let
you average them. `[A]` High confidence this is worth building; it costs an afternoon and it
is the single control that would have caught the mis-attribution in §3.3 before it was
written down.

### 4.2 Prefix reuse has a correctness hazard that determinism analysis exposes

`paged-attention-and-prefix-reuse.md` treats the prefix cache as a shared read-only tier
with invalidation hazards. Determinism adds a hazard that module does not cover.

The KV entries for a prefix of length `P` are produced by a prefill whose kernel tiling
depends on the *total* prefill length. Under chunked prefill — where a long prompt is
processed in pieces across scheduler steps — the same prefix is computed with different tile
boundaries than it would have been in a single-shot prefill. So:

```
KV(prefix, computed as part of a length-P prefill)
    ≠ (bitwise)
KV(prefix, computed as part of a length-P' prefill),   P' > P
```

`[A]` High confidence in the mechanism, **untested here**. vLLM's cache is keyed by a chain
hash over *token ids* (`memory/vllm/vllm/v1/core/kv_cache_utils.py:596`), not over the KV
bytes, so a hit returns whichever version happened to be cached first. That is a storage
tier that returns a *different value* than a recompute of the same key would — which in any
storage system you have run is a corruption bug, and here is the documented design.

Whether it matters is an empirical question with a sharp form: **does a prefix-cache hit
change a downstream eviction decision?** If the KV bytes differ by `~1e−3` relative and the
eviction boundary gap is smaller than that (§3.7), then yes, and the cache hit rate becomes
an experimental confound in every Mnemosyne arm. This is a Mnemosyne experiment, it is
cheap, and this lab has not run it. It goes on the backlog.

### 4.3 The failure this lab already had, kept because it is the lesson

`curriculum/README.md` records three exercise-derived findings, one of which did not survive
retest, and this module's §3.3 records a fourth: a real, replicated, three-seed effect
attributed to the wrong cause. The pattern in both:

- **The crash that was not repeatable**: a hipBLASLt segfault on skinny-K GEMMs, tagged `[M]`
  from a single observation, refuted on retest across four shapes in isolated subprocesses,
  all eight runs exit 0. A single non-deterministic observation is an anecdote.
- **The effect attributed to the wrong cause**: "GPU 3× worse than CPU at 1M keys" —
  correct measurement, wrong variable, because the environment was uncontrolled.

The two failure modes are opposites and both are determinism failures. One tagged noise as
signal; the other tagged signal with the wrong label. The controls that would have prevented
each are different: **fresh-process replication** for the first, **an environment matrix**
for the second. Run both. Neither is expensive.

### 4.4 What we cannot do here, stated plainly

Track D is about training systems, and most of training-systems determinism is about
collectives. On this machine:

- **No distributed determinism work is possible.** `ASSUMPTIONS.md: single-device-only` —
  `torch._C._distributed_c10d` is incomplete on gfx1151. Every all-reduce ordering question,
  every NCCL algorithm pin (`memory/vllm/vllm/model_executor/layers/batch_invariant.py:976`
  sets `NCCL_ALGO=allreduce:tree` for exactly this reason), every tensor-parallel-size
  invariance result `[C]` (arXiv 2511.17826) is **design-only** for us. Do not write
  exercises that pretend otherwise.
- **vLLM's batch-invariant mode does not cover us.** `envs.py:601`–`:603` documents
  `VLLM_BATCH_INVARIANT` as "Requires NVIDIA GPU with compute capability >= 9.0", and
  reading `enable_batch_invariant_mode`
  (`memory/vllm/vllm/model_executor/layers/batch_invariant.py:902`) confirms the branches
  are `is_cuda()` and `is_xpu()` — there is no ROCm branch, so on our platform the
  `aten::mm` / `aten::addmm` / `aten::matmul` / `aten::linear` overrides are simply not
  installed. The softmax/mean/bmm overrides would be, but the GEMM — the actual source of
  batch variance — would not.
- **The determinism escape hatch that does exist on ROCm is at the wrong level.** `[C]`
  ROCm 6.3.0 added `rocprim::deterministic_inclusive_scan` /
  `deterministic_reduce_by_key` and the `thrust::hip::par_det` execution policy, explicitly
  for bitwise reproducibility with non-associative operators, at a stated performance cost
  (ROCm 6.3.0 release notes). Those are C++ primitives; nothing in PyTorch's ROCm backend
  routes to them today as far as this survey can tell, and we are on a 7.13-series nightly,
  not 6.3. Whether any of it is reachable from Python here is unknown.
- **What we *can* do, and it is more than expected.** `[M]` §3.4(d) ran
  `torch.use_deterministic_algorithms(True)` on this stack and found deterministic
  implementations available for every op a transformer backward pass touches; only
  `grid_sampler_2d_backward` and `max_pool3d` backward raise, and neither appears in our
  models. So the op-level blocker most people assume is not present. What *is* present is an
  unexplained failure at the level above: `[M]` 0/5 fresh-process reproducibility for twenty
  fp32 training steps (§3.4e) despite 3/3 for a single fp32 GEMM (§3.4a). We know more than
  we did and less than we need.

---

## 5. Read the code

Paths are relative to `research/reference/`; run `scripts/fetch_reference.sh` if the clones
are not materialized. Read in the order given — the first group is what determinism looks
like when someone engineered it, the second is what it looks like when someone shipped a fix
for it, the third is where it leaks.

### Group 1 — OLMo-core: determinism as an engineering discipline

| Where | What to look at, and why |
|---|---|
| `training/olmo-core/src/olmo_core/utils.py:164` | `def seed_all(seed: int)` — the whole function is five lines and four RNG streams. Read it, then grep your own code for every library that might own a fifth. |
| `training/olmo-core/src/olmo_core/utils.py:179` | `torch.cuda.manual_seed_all(seed)` with a comment saying `torch.manual_seed` probably already did it. Deliberate redundancy in a seeding function is a good instinct, not sloppiness. |
| `training/olmo-core/src/olmo_core/internal/experiment.py:462` | `seed_all(config.init_seed)` — the *init* seed. |
| `training/olmo-core/src/olmo_core/internal/experiment.py:252` | `seed=34521` passed to the data loader — a **different, independent** seed for data order, hardcoded in the experiment builder. Hold these two pointers side by side: this is `the-training-loop.md` §4's "two seeds, different questions" as shipped code. |
| `training/olmo-core/src/olmo_core/data/utils.py:490` | `def get_rng` → `np.random.Generator(np.random.PCG64(seed=seed))`. A named, versioned bit generator rather than the legacy global — which is what makes the permutation portable across NumPy versions. |
| `training/olmo-core/src/olmo_core/data/data_loader.py:667` | `_build_global_indices` — the epoch's entire instance permutation as a pure function. **Nothing about data order is ever persisted.** |
| `training/olmo-core/src/olmo_core/data/data_loader.py:673` | `rng = get_rng(self.seed + self.epoch)`. The whole determinism story, and its one flaw: `seed + epoch` is an addition, so `(1000, 2)` and `(1001, 1)` are the same stream. Would you have caught this in review? |
| `training/olmo-core/src/olmo_core/data/data_loader.py:663` | `v=1,  # tick if logic changes` — the derived-index memmap's filename carries a *logic version* alongside seed/epoch/size/chunk. A cache key that includes the code version. Steal this. |
| `training/olmo-core/src/olmo_core/data/data_loader.py:720` | `_get_local_instance_indices` — mid-epoch resume as arithmetic: reshape into batches, drop `batches_processed` rows, stride. No iterator state to persist. |
| `training/olmo-core/src/olmo_core/data/data_loader.py:742` | `indices = indices[worker_info.id :: worker_info.num_workers]` — **the worker count does not change the data order.** Each worker derives its own slice from the same global permutation instead of pulling from a shared queue, so completion order is irrelevant. This is the fix for source #5 in §2.3, in one line. |
| `training/olmo-core/src/olmo_core/data/data_loader.py:584` | `persistent_workers=False` — workers are torn down and rebuilt when the batch size changes (`:597`), so worker state can never silently survive a config change. |
| `training/olmo-core/src/olmo_core/data/data_loader.py:441` | `state_dict` — what a dataloader checkpoint actually contains: dataset fingerprint (+version), `batches_processed`, `tokens_processed`, `seed`, `epoch`. Six fields, no iterator. |
| `training/olmo-core/src/olmo_core/data/data_loader.py:457` | The fingerprint mismatch path: **`raise RuntimeError`**, not a warning, with the message "This will probably result in a different data order!" A hard fail on a silent-corruption path. |
| `training/olmo-core/src/olmo_core/data/data_loader.py:469` | The subtle one: on resume with a *different* seed in the config, the **checkpoint's seed wins** and the config's is discarded with a warning. So "change the seed and resume" quietly does not do what you asked. Decide whether you agree with that choice before you copy it. |
| `training/olmo-core/src/olmo_core/data/data_loader.py:760` | `self.batches_processed = self.tokens_processed // self.global_batch_size` — the cursor is really denominated in tokens, so a resume with a different batch size still lands on the right token boundary. Reproducibility that survives a config change. |
| `training/olmo-core/src/olmo_core/train/checkpoint.py:498` | `_temporary_wd` — write to `<dir>-tmp`, barrier, rename. Atomicity at rename granularity. The DR bridge, and the break: a torn save loses the whole checkpoint, not a tail. |
| `training/olmo-core/src/olmo_core/train/trainer.py:1517` | `torch.cuda.set_sync_debug_mode("warn")` — a tripwire armed after every metrics flush so an *accidental* host-device sync gets reported. |
| `training/olmo-core/src/olmo_core/train/utils.py:137` | `with cuda_sync_debug_mode(0):` — the one place the tripwire is disarmed, because that sync is intentional. Read as a pair with the line above: this is what "we know exactly where our nondeterminism-adjacent stalls are" looks like. |

### Group 2 — vLLM: batch invariance as a shipped feature, and its price list

| Where | What to look at, and why |
|---|---|
| `memory/vllm/vllm/model_executor/layers/batch_invariant.py:902` | `enable_batch_invariant_mode` — the dispatcher-level override. Note it replaces `aten::mm`, `aten::addmm`, `aten::matmul`, `aten::linear`, `aten::_log_softmax`, `aten::softmax`, `aten::mean.dim`, `aten::bmm` with Triton kernels. That is the full list of ops whose reduction order they had to take control of. |
| `memory/vllm/vllm/model_executor/layers/batch_invariant.py:914` | `if current_platform.is_cuda():` … `elif current_platform.is_xpu():`. **No ROCm branch.** This is the pointer that tells you the feature does not exist for us. |
| `memory/vllm/vllm/model_executor/layers/batch_invariant.py:926` | On Hopper/Blackwell they do not override the matmul at all — they set `CUBLAS_WORKSPACE_CONFIG=":16:8"`, because the comment says "the only source of batch variance is split-k, which we disable via the cuBLAS workspace config." Determinism achieved by starving the allocator so split-k cannot be chosen. |
| `memory/vllm/vllm/model_executor/layers/batch_invariant.py:956` | They also force `allow_bf16_reduced_precision_reduction = False`. `[M]` On our stack that knob is inert (`ASSUMPTIONS.md: bf16-reduced-precision-knob-works`). Same line of code, different meaning per platform. |
| `memory/vllm/vllm/model_executor/layers/batch_invariant.py:963` | `override_envs_for_invariance` — **read this function slowly.** Twelve environment variables: a BLAS workspace config, eight NCCL settings pinning the algorithm to a single-threaded tree all-reduce, symmetric-memory off, AOT compile off. This is the complete, honest answer to "what does the environment have to be, for the bits to be the bits." Nobody's run record has twelve environment variables in it. |
| `memory/vllm/vllm/model_executor/layers/batch_invariant.py:990` | `torch.backends.cuda.matmul.fp32_precision = "ieee"` — TF32 off, with the comment "it causes non-deterministic rounding." Compare `training/nanogpt/train.py:107`, which turns TF32 *on*, unconditionally, three lines after seeding. |
| `memory/vllm/vllm/envs.py:603` | `VLLM_BATCH_INVARIANT` default `False`, documented at `:601`–`:602` as "Requires NVIDIA GPU with compute capability >= 9.0." A feature flag whose docstring is a hardware exclusion list. |
| `memory/vllm/tests/v1/determinism/test_rms_norm_batch_invariant.py:338` | `test_rms_norm_determinism` — same input five times, `assert_close(rtol=0.0, atol=0.0)`. This is the bitwise-determinism test. |
| `memory/vllm/tests/v1/determinism/test_rms_norm_batch_invariant.py:375` | `test_rms_norm_batch_invariance` — same row alone vs as row 4 of an 8-row batch. This is the batch-invariance test. **Two adjacent functions testing two different properties, and the difference between them is the whole of §2.1.** |
| `memory/vllm/tests/v1/determinism/test_rms_norm_batch_invariant.py:397` | `assert torch.equal(out_single[0], out_batch[4])` — the one line worth memorising as the definition. |

### Group 3 — where it leaks: kernels that document their own nondeterminism

| Where | What to look at, and why |
|---|---|
| `memory/flashinfer/flashinfer/decode.py:1316` | The `fixed_split_size` docstring: fixing the split "will lead to deterministic softmax score reduction in the merge_states kernel, and therefore batch-size invariant outputs" — and then, two lines later, warns that CUDA-graph compatibility is not guaranteed because the CTA count still varies with sequence length. Determinism traded against a different optimization. |
| `memory/flashinfer/flashinfer/decode.py:1321` | `disable_split_kv` — "Whether to disable the split-kv for determinism in CUDA Graph." A second, coarser knob for the same problem. Two APIs for determinism in one function signature tells you how live the issue is. |
| `memory/flashinfer/flashinfer/fused_moe/core.py:1038` | "The fused epilogue reduces expert outputs via non-associative atomics, so results are not deterministic run-to-run. Set to False to use the non-fused, deterministic finalize path." Directly relevant to `proteus-moe-sigmoid`: MoE combination is an atomic reduction by default. |
| `memory/flashinfer/flashinfer/gdn_prefill.py:221` | Gated-DeltaNet prefill: state slot ids "**must be unique**… two sequences sharing a slot id would concurrently write the same pool row across work tiles, leaving that row's final state nondeterministic," and uniqueness is "a caller precondition (not checked at launch, to avoid a per-call host sync)." An unchecked precondition whose violation is silent nondeterministic state corruption. Hold this next to `constant-state-memory.md`: the fixed-size state has no redundancy, so a corrupted row is unrecoverable. |
| `architecture/mamba/mamba_ssm/ops/triton/ssd_combined.py:375` | `states_in_fp32=True` — the SSD chunk-boundary pass runs in fp32 regardless of model dtype. The most numerically fragile part of the layer got a precision exception; nothing else did. Evidence that kernel authors know exactly where their reductions are dangerous. |
| `training/nanogpt/train.py:106` | `torch.manual_seed(1337 + seed_offset)` — one stream. |
| `training/nanogpt/train.py:107` | `torch.backends.cuda.matmul.allow_tf32 = True`, one line after the seed. On NVIDIA this is a numerics change made silently for speed; `[A]` medium confidence it is inert on gfx1151 since RDNA has no TF32 datapath — cheapest test is timing an fp32 matmul with it on and off. Either way, note the shape: a precision decision three characters from a determinism decision, with no comment connecting them. |
| `training/nanogpt/train.py:116` | `get_batch` — `np.memmap` plus `torch.randint` offsets sampled **with replacement from the global RNG**, no cursor, no epoch. Data position is not state, so it cannot be checkpointed. Compare the OLMo-core group above and count what is missing. |
| `training/nanogpt/train.py:179` | `iter_num = checkpoint['iter_num']` — reading upward, the resume restores model, optimizer, `iter_num`, `best_val_loss`, and nothing else. **No RNG state, no data cursor.** A resumed run therefore diverges from an uninterrupted one *by construction*, which is why the Hardware Validation Gate's checkpoint round-trip must compare **weights**, not loss curves. |

---

## 6. Exercises

All three run on the Z13 (gfx1151, native Windows, one GPU) and all three have a real CPU
fallback. Activate with `. .\scripts\activate-lab.ps1` from the repo root — **except in
Exercise A, where not activating is half the experiment.**

**Caveats that apply to all three:**

- Keep every single tensor under 32 GiB. `[M]` `ASSUMPTIONS.md: large-tensor-fault-32gib` —
  a 32 GiB buffer hangs at 0% CPU with no error. Nothing below comes close, but a careless
  sweep parameter can.
- Do not use `F.scaled_dot_product_attention` for the score-matrix work in Exercise B unless
  you set `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`. `[M]`
  `ASSUMPTIONS.md: sdpa-is-memory-efficient` — by default it retains the `B·nh·T²` score
  matrix (147.2 bytes/T² vs 6.6 with the flag). Exercise B computes scores explicitly
  anyway, which is what you want: it isolates the reduction.
- **Every comparison must be between fresh processes.** A repeat inside one process shares
  the autotune cache, the allocator state and the loaded kernels, and will look far more
  deterministic than reality. The house standard exists because this lab has already tagged
  a within-process observation as `[M]` and had it refuted.
- **Budget on process count, not on compute.** `[M]` On this machine a fresh subprocess that
  imports torch, touches the GPU and does a few seconds of work costs roughly **1.5 minutes
  wall clock, almost all of it import and HIP context creation**. The 26 subprocesses behind
  §3.4 took about 35 minutes and perhaps 90 seconds of that was arithmetic. Plan sweeps by
  counting launches; and never "optimize" by moving repeats inside one process, which is the
  one thing that would invalidate them.
- Report the wheel and driver with any number you keep:
  `torch 2.12.0a0+rocm7.13.0a20260313`, HIP 7.2.0, driver 32.0.23033.5002.

### Exercise A — Is the environment part of the seed? Prove it bitwise.

**Difficulty 2/5. Writing: 30–45 min. Runtime: `[M]` ~1.5 min per subprocess launch on this
machine, nearly all of it torch import, so the full 2 dtypes × 4 sizes × 2 environments × 2
reps = 32 launches is ~50 min unattended. Halve it by dropping to one rep per cell once
you have confirmed within-cell determinism in the first four. CPU fallback: ~5 min.**

§3.3 established a 2.8× *accuracy* difference from two environment variables. §3.4(a) turned
it into a bitwise statement and found it **shape- and dtype-dependent**: fp32 at 2048³
differs between the two environments (3 runs each side, perfectly reproducible within each);
bf16 at 4096³ does not (2 runs each side, all four identical). **The exercise is not to
re-derive that. It is to find the boundary**, which nobody has mapped, and which decides
which of our workloads the environment fingerprint actually protects.

**Build.** One script that takes a mode on `argv` and prints one JSON line; a driver that
launches it in fresh subprocesses under different environments and compares SHA-256 digests
of the raw tensor bytes.

```python
# scratch/env_is_part_of_the_seed.py
import hashlib, json, os, subprocess, sys

LAB = r"C:\venvs\lab"
TENSILE = LAB + r"\Lib\site-packages\_rocm_sdk_libraries_gfx1151\bin\hipblaslt\library"

def digest(t):
    import torch
    x = t.detach().to("cpu").contiguous()
    as_int = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[x.element_size()]
    return hashlib.sha256(x.view(as_int).numpy().tobytes()).hexdigest()[:16]

def probe(dev, dtype, n):
    import torch
    torch.manual_seed(1337)
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    c = a @ b
    if dev == "cuda": torch.cuda.synchronize()
    return {"in_a": digest(a), "in_b": digest(b), "out_c": digest(c),
            "c_sum_f64": float(c.double().sum().item()),
            "hipblaslt": bool(os.environ.get("HIPBLASLT_TENSILE_LIBPATH")),
            "threads": torch.get_num_threads()}

def run(hipblaslt, threads=None):
    env = dict(os.environ)
    env.pop("HIPBLASLT_TENSILE_LIBPATH", None); env.pop("TORCH_BLAS_PREFER_HIPBLASLT", None)
    if hipblaslt:
        env["HIPBLASLT_TENSILE_LIBPATH"] = TENSILE
        env["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"
    if threads: env["OMP_NUM_THREADS"] = str(threads)
    out = subprocess.run([sys.executable, __file__, "child"], env=env,
                         capture_output=True, text=True, timeout=1800)
    return json.loads([l for l in out.stdout.splitlines() if l.startswith("{")][-1])

if __name__ == "__main__":
    if len(sys.argv) > 1:                     # child: shape/dtype from argv
        import torch
        n, dt = int(sys.argv[2]), {"bf16": torch.bfloat16, "fp32": torch.float32}[sys.argv[3]]
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(json.dumps(probe(dev, dt, n)))
    else:                                     # driver: sweep, 2 reps per cell
        for dt in ("fp32", "bf16"):
            for n in (512, 1024, 2048, 4096):
                for hip in (True, False):
                    for rep in (1, 2):
                        print(f"{dt} n={n:5d} hipblaslt={int(hip)} rep{rep} "
                              f"{json.dumps(run(hip, n, dt))}")
```

(You will need to thread `n` and `dt` through `run()` into the child's `argv`; that is three
lines and is left as the only implementation work.)

**What to check.**

1. **The control, first, always.** `in_a` and `in_b` must be identical across every run of a
   given `(n, dtype)`. If they are not, the seed is not doing its job and nothing else in
   this module applies.
2. **Within-configuration determinism:** rep1 vs rep2 at fixed `(n, dtype, hipblaslt)`. `[M]`
   Measured identical in all ten processes we ran (§3.4a). If any cell of yours comes out
   `False`, **that is the most important result in the exercise** and it invalidates the
   "pin the environment" strategy in §4.1 by itself — write it up immediately.
3. **Across-configuration difference:** `ON` vs `OFF` at fixed `(n, dtype)`. This is the
   boundary you are mapping. `[M]` fp32 @ 2048 → **different**. bf16 @ 4096 → **identical**.
   The other six cells are open.
4. **Magnitude, where they differ:** `c_sum_f64` relative difference. §3.4(a) reports the
   *existence* of the fp32 difference but never measured its *size*; that is a gap in our own
   data and you can close it in the same run.

**Two predictions, stated so they can fail.** `[A]` Medium confidence that the fp32/bf16
split is the dominant axis (fp32 differs at every `n`, bf16 at none), because bf16 GEMMs on
this part go through the matrix cores by one path while fp32 has more implementation choice.
`[A]` Low-medium confidence in the alternative — that it is a *size* effect and small `n`
differs in both dtypes because small GEMMs are the ones that need split-k to fill the
machine. **These two predictions disagree about the bf16 @ 512 cell**, which makes that one
cell the cheapest discriminating measurement in the exercise. Run it first.

**CPU fallback, and it is not a consolation prize.** Drop the `cuda` branch and vary
`OMP_NUM_THREADS ∈ {1, 4, 16}` instead of the hipBLASLt variables. Thread count changes the
partitioning of the reduction exactly the way `k_split` does on the GPU, and it is an
environment variable that appears in no run record either. Same lesson, any machine, three
minutes.

**Deliverable.** A 16-cell table (2 dtypes × 4 sizes × ON/OFF) of digests with a
within-cell repeat, the boundary stated in one sentence, and the relative magnitude wherever
`ON ≠ OFF`. If the boundary is not cleanly explained by dtype or by size, that is a
`notebook/` entry — it would mean the environment's numerics effect is governed by the
library's internal heuristic table, which is neither predictable nor documented, and the
only safe policy is to pin the environment for everything.

### Exercise B — Batch invariance, and does it move an eviction decision?

**Difficulty 3/5. Writing: 45–60 min. Runtime: ~6 min GPU (three or four process launches at
`[M]` ~1.5 min each), ~5 min CPU.**

Part one is the standard batch-invariance probe. Part two is the one that matters for this
lab and, as far as this survey found, has not been published: **does batch-dependent numeric
error change which tokens a KV-eviction policy keeps?**

**Read the expected outcome before you start, because it changed.** `[M]` §3.4(b) found a
bf16 GEMM batch-invariant 7/7 here, so **the most likely result of part two as written is
zero flips**, and zero flips is a real answer that closes a real question — it would mean
§3.7's batch-composition leg does not apply on this hardware and the Mnemosyne tie-break
requirement rests on run-to-run and environment variation alone. If you want the version
most likely to find something, skip to the ragged-KV extension at the end of part one; that
is the shape the published effect lives in.

**Part one — the probe, which we have already run, so treat it as a calibration.** One
weight matrix `W ∈ R^{4096×4096}` bf16, one input stack `X ∈ R^{1024×4096}` bf16, both from
a fixed seed. For `bs ∈ {1, 2, 4, 16, 64, 256, 1024}` compute `Y = X[:bs] @ W` and digest
**row 0 only**. Row 0's input is byte-identical in every case; only its neighbours change.

```python
rows, maxdiff, ref = {}, {}, None
for bs in (1, 2, 4, 16, 64, 256, 1024):
    r = (x[:bs] @ w)[0]
    torch.cuda.synchronize()
    rows[bs] = digest(r)
    if ref is None: ref = r.clone()
    maxdiff[bs] = float((r.float() - ref.float()).abs().max())
```

**Checkable numbers:** how many of the seven batch sizes give a digest equal to `bs=1`, and
the largest `maxdiff` relative to `ref.abs().max()`.

`[M]` **We got seven out of seven identical, `maxdiff` exactly 0.0 at every batch size, in
two fresh processes** (§3.4b). The prediction this exercise originally carried —
"medium-high confidence the count is less than seven" — was wrong. If you reproduce 7/7, you
have calibrated your harness against a known answer, which is what part one is now for; if
you get anything else, one of us has a bug and finding out which is worth an hour.

Then change one thing at a time and look for the boundary, because a plain GEMM is the
*easiest* case: vary `K` (the reduction depth) down to 128 and up to 32768 at fixed `bs`;
switch to fp32; and — the case that actually matters — replace the GEMM with a real attention
call over **ragged** KV lengths, which is the shape FlashInfer's `fixed_split_size` knob
exists for (`memory/flashinfer/flashinfer/decode.py:1316`). A uniform batch of equal-length
sequences cannot exhibit the effect the knob suppresses.

**Part two — the eviction-flip measurement.** Build a synthetic attention step, no model
required:

1. `Q ∈ R^{1×128}`, `K ∈ R^{N×128}` with `N = 8192`, bf16, fixed seed.
2. Scores `s = softmax(Q Kᵀ / √128)`, computed **twice**: once with `Q` alone, once with `Q`
   embedded as row `j` of a `[B, 128]` query batch for `B ∈ {8, 64, 256}` (other rows random,
   drawn from a *separate* generator so `Q` itself is bit-identical).
3. For `k ∈ {N/2, N/4, N/8}` take `torch.topk(s, k)` on each version and report:
   - `|selected_alone Δ selected_batched|` — the symmetric difference, i.e. **how many
     tokens a batch-composition change alone caused to be evicted or retained**;
   - the boundary gap `g = s_(k) − s_(k+1)` and the relative numeric difference between the
     two score vectors, so you can check §3.7's `g < 2ε` prediction directly;
   - the total attention mass of the flipped set — because 3 flipped tokens out of 4096
     carrying 0.001% of the mass is a curiosity, and 3 carrying 5% is a finding.

**How this can fail, which is the point.** If the symmetric difference is 0 at every `k` and
every `B`, then either your kernel is batch-invariant (check part one — the two results must
agree) or your synthetic score distribution has boundary gaps far larger than any realistic
attention distribution, which is itself worth knowing: real attention scores are heavily
concentrated, so most of the mass sits above the cut and the boundary region is a long flat
tail of near-ties. That flat tail is precisely where §3.7 says the trouble is, so if your
synthetic scores do not reproduce it, generate them from a power-law instead of a Gaussian
and rerun.

**CPU fallback.** Everything runs on CPU unchanged at `N = 8192`. CPU GEMM is *usually*
batch-invariant for these shapes, which makes the CPU arm a useful control rather than a
substitute: if the flip count is 0 on CPU and nonzero on GPU, you have isolated the cause.

**Deliverable.** A `(B, k)` grid of symmetric-difference counts and flipped-mass fractions,
plus the part-one digest table. If any cell is nonzero, this is a `notebook/` entry with a
pre-registered card, because it establishes that **KV eviction is not a pure function of the
sequence** on this hardware — and that is a constraint on every Mnemosyne experiment, not a
curiosity.

### Exercise C — The gfx1151 determinism coverage map

**Difficulty 4/5. Writing: 60–90 min. Runtime: ~20 min GPU (each fresh subprocess costs
~1.5 min, dominated by torch import). Closes a Hardware Validation Gate item.**

§3.4(d) and §3.4(e) are the first pass at this and they leave three specific holes. This
exercise is not "reproduce the table" — it is **find where the nondeterminism in §3.4(e)
lives**, which is the question the table raises and does not answer.

**C1 — bisect the fp32 training nondeterminism.** `[M]` A single fp32 GEMM is bitwise
reproducible 3/3 (§3.4a). `[M]` Twenty fp32 training steps are reproducible 0/5 (§3.4e). The
nondeterminism is therefore *not* in the forward GEMM. Bisect it by digesting after each
stage of step 1 only, across two fresh processes:

1. after `model.__init__` (initialization) — expect identical; if not, stop, the RNG is the
   problem
2. after the forward pass — digest `loss` and the logits
3. after `loss.backward()` — digest each `p.grad` **separately**, not concatenated, so a
   mismatch names the parameter
4. after `opt.step()` — digest each parameter

**The checkable output is the name of the first tensor that differs.** `[A]` Medium
confidence it is the embedding gradient (`wte.weight.grad`), because embedding backward is
an `index_add` over the vocabulary and is the canonical atomics-based backward op — but
§3.4(d) says a deterministic implementation *exists* for it here, so if that is the culprit
then the interesting follow-up is that the deterministic path is not selected by default.
`[A]` Low-medium confidence on the alternative: `scaled_dot_product_attention`'s backward,
which `ASSUMPTIONS.md: sdpa-is-memory-efficient` shows is already doing something
unexpected on this stack. **The two candidates are distinguishable by this one measurement**
and neither has been checked.

**C2 — does `use_deterministic_algorithms(True)` actually fix it, and what does it cost?**
Re-run the §3.4(e) fp32 arm with the flag on, two fresh processes. Three outcomes, all
useful:

- **Bitwise identical** → bitwise-deterministic training is available here. Report the
  throughput cost (steps/s with and without) and compare it to `[C]` DASH's up-to-37.9%
  figure for deterministic attention backward (arXiv 2601.21824). That number would be a
  genuine contribution: nobody has published it for RDNA.
- **Still different** → some op in the path is nondeterministic *and* not registered as such
  in PyTorch's deterministic-algorithms machinery, which is a PyTorch/ROCm bug worth
  reporting upstream with the C1 bisection attached.
- **Raises** → you have found the missing op, and the exception names it. Add it to
  §3.4(d)'s table.

Record separately whether the flag demands `CUBLAS_WORKSPACE_CONFIG` on this stack (on CUDA
it does, for cuBLAS ≥ 10.2). `[A]` Medium confidence it is inert on ROCm because hipBLAS
does not read that variable — **but vLLM's Hopper path achieves determinism with nothing but
that variable** (`memory/vllm/vllm/model_executor/layers/batch_invariant.py:926`), so if
there is a ROCm equivalent, finding it is the highest-value output of this exercise.

**C3 — the Philox offset probe, two lines, and it settles an ablation-design question.**
`torch.manual_seed(0)`, draw `N` values, then draw four more, for `N ∈ {1, 1000, 1_000_000}`
on GPU and CPU. If the trailing four are a pure function of the count of preceding draws,
the offset accounting is count-based; if they are not, it is launch-geometry-based and
§2.2's third point is confirmed on this hardware. **This decides whether dropout masks are
stable under a batch-size change, which decides whether batch size is a clean ablation
axis** — and if it is not, every batch-size sweep in this lab is confounded by a different
dropout realization and must set `dropout = 0`.

**CPU fallback.** C1 runs on CPU as the control arm — `[M]` CPU fp32 was bitwise
reproducible 2/2, so a CPU bisection should find *no* differing tensor, which is how you
validate the harness before trusting it on GPU. C3 runs on CPU unchanged.

**Deliverable, and this one goes in `ASSUMPTIONS.md` as a new row.** The name of the first
divergent tensor, the `use_deterministic_algorithms` verdict with its throughput cost, and
one sentence on Philox. Label the counts honestly: two fresh processes agreeing is *evidence
of* determinism, not proof; a single disagreement is proof of nondeterminism and needs no
replication. The asymmetry is in your favour here — negative results are cheap.

---

## 7. Self-check

1. A colleague reports that their two ablation arms are "fully reproducible — same seed,
   bit-identical loss curves." State the two properties they might mean, say which (if
   either) they have actually demonstrated, and say what it does and does not tell you about
   whether their architecture change works.

2. You sum 2²⁰ bf16 numbers. Give the worst-case relative error bound for sequential
   accumulation and for a pairwise tree, and the ratio between them. Then say why neither
   number is useful, and what quantity you would predict instead.

3. Your ablation's per-seed standard deviation is 0.012 nats and you are running 3 seeds per
   arm. What is the smallest effect you can detect? You then make the runs bitwise
   deterministic. What is the smallest effect you can detect now? What change *would* have
   helped, and under exactly what condition does it work?

4. `HIPBLASLT_TENSILE_LIBPATH` is set in one shell and not in another. Everything else —
   seed, code, wheel, driver, device, input bytes — is identical. What is measured to change,
   by how much, and why is this a stronger statement about experimental design than "GPUs are
   nondeterministic"?

5. A Mnemosyne eviction policy keeps the top 512 tokens by accumulated attention mass. Give
   the condition under which two runs of that policy on identical input select different
   token sets, and name the two mechanisms that make it a *design* problem rather than a
   rounding curiosity.

6. Name three things on this hardware that make bitwise-deterministic *training* impossible
   or unverified today, and for each say whether it is a hardware limit, a software gap, or
   an unmeasured unknown.

7. `[M]` A single fp32 GEMM on this GPU is bitwise reproducible across fresh processes,
   3 for 3. `[M]` Twenty fp32 training steps on the same GPU are bitwise reproducible 0 for
   5, while printing the same loss to ten decimal places. Reconcile those two facts, and say
   what single measurement would localize the cause.

*(Answers at the end of this file.)*

---

## 8. What is still unsolved here

**There is no batch-invariant kernel library for ROCm, and the field's fix does not port.**
`[C]` Thinking Machines Lab ("Defeating Nondeterminism in LLM Inference", 11 Sep 2025) built
batch-invariant RMSNorm, matmul and attention kernels and got 1000 bit-identical runs out of
1000; `[C]` SGLang integrated them with CUDA graphs and reduced the throughput cost from
~61.5% to ~34.35% (LMSYS blog, 22 Sep 2025); vLLM shipped the mode behind
`VLLM_BATCH_INVARIANT`. All of it is CUDA (and, in vLLM, XPU) — the platform branch in
`memory/vllm/vllm/model_executor/layers/batch_invariant.py:914` has no ROCm arm, and the flag
is documented as requiring compute capability ≥ 9.0. `[C]` ROCm 6.3 added deterministic
rocPRIM scans and `thrust::hip::par_det`, but that is a C++ primitive layer with no visible
PyTorch route. **Whether batch invariance is even achievable on gfx1151 from Python is
unknown**, and it is a well-scoped piece of work if this lab ever needs it.

The awkward part is that `[M]` §3.4(b) found a bf16 GEMM already batch-invariant here, 7/7
across `bs ∈ [1, 1024]`. Two readings are open and they have opposite consequences.
Either RDNA's GEMM scheduling happens not to exhibit the effect at these shapes — in which
case the missing ROCm support costs us nothing and the whole concern is imported anxiety —
or the effect lives in the attention split-kv merge and the sequence-length axis, which our
probe never touched, in which case we have measured the easy case and concluded from it. The
discriminating experiment is Exercise B's ragged-KV extension, and until it runs, **do not
quote our 7/7 as "gfx1151 is batch-invariant."** It is "a square bf16 GEMM is, at fixed `K`."

**Where the fp32 training nondeterminism actually lives is unknown, and it is ours to
find.** `[M]` One fp32 GEMM: reproducible 3/3. `[M]` Twenty fp32 training steps: reproducible
0/5. Those two facts bracket the culprit to the backward pass or the optimizer, and
`[M]` §3.4(d) says the usual suspects (embedding backward, `scatter_add`, `index_add`) all
*have* deterministic implementations available on this backend — so if one of them is the
cause, the finding is that PyTorch is not selecting the deterministic path by default, which
is a different and more actionable bug than "atomics are nondeterministic." Exercise C's
bisection costs an evening and names the tensor.

**The training-side story is much thinner than the inference-side story, and both are new.**
Every *published* result cited so far — Thinking Machines, SGLang, vLLM's shipped mode — is
about inference. For training, `[C]` DASH (arXiv 2601.21824, 29 Jan
2026) reports that a deterministic attention backward pass costs **up to 37.9% throughput**
and proposes a DAG-scheduling fix recovering up to 1.28×. That is the only clean price tag
this survey found for deterministic *training*, it is six months old, and it is about
attention alone — nobody has published the end-to-end cost of a bitwise-deterministic
pretraining run at any scale. `[C]` arXiv 2511.17826 extends determinism across tensor-
parallel sizes to kill the training/inference mismatch in RL, and `[C]` arXiv 2601.17768
(LLM-42) argues for paying for determinism only on the fraction of traffic that needs it.
This is an actively contested design space as of mid-2026, not a settled one, and none of it
is reachable on a single device.

**Nobody has priced determinism against ablation validity.** The entire literature argues
for determinism on grounds of debuggability, RL correctness, or compliance. §3.5–3.6 gives
the argument from statistical power — that determinism's real payoff is making a *paired*
design valid, which is what lets a 3-seed budget resolve a 1σ effect — and this survey found
nobody making it. It is also directly testable here and cheap: run the same policy ablation
paired and unpaired at 3 seeds and compare the widths of the confidence intervals. If the
paired interval is not materially narrower, the argument is wrong and should be deleted from
this module.

**The variance decomposition `σ² = σ²_init + σ²_data + σ²_numeric` has never been measured at
our scale.** `[C]` arXiv 2002.06305 (Feb 2020) separated init from data order for fine-tuning
BERT and found both material; `[C]` arXiv 2503.07329 (Mar 2025) revisits seed effects on
LLM fine-tuning at macro and micro level; `[C]` arXiv 2504.07086 (Apr 2025) documents
5–15 point Pass@1 swings across seeds on reasoning benchmarks. All of that is fine-tuning or
evaluation. For 20M–300M **pretraining**, which is where every Chiron ablation lives, the
three components have not been separated and the total is not published either. Measuring it
is three runs varying only `seed_init`, three varying only `seed_data`, and three varying
neither — nine short runs that would calibrate every SUCCESS threshold this lab ever writes.
It is arguably the highest-value nine runs available and it is not on the backlog yet.

**Whether the eviction-flip effect in §3.7 exists at all is unmeasured.** The reasoning is
sound and the numbers are ours, but the chain — batch composition → split-k → score bits →
top-k boundary → retained set → recall — has four untested links. Exercise B tests two of
them. If the answer is "zero flips at every batch size," §3.7 is a nice derivation with no
consequences and should be demoted to a footnote. If it is nonzero, it constrains the design
of every Mnemosyne policy and belongs in the interface spec.

**And a small one that is entirely ours.** `training/olmo-core/src/olmo_core/data/data_loader.py:673`
derives the epoch permutation from `seed + epoch`. That is a collision in the seed space —
`(1000, 2)` and `(1001, 1)` give the same data order — and it is in the most careful
reproducibility code this reference library contains. It is almost certainly harmless in
practice (nobody runs both). It is also exactly the class of defect that a determinism
audit exists to find, and finding it in *their* code should calibrate your expectations
about finding it in yours.

---

## Answers to the self-check

**1.** The two properties are **bitwise determinism** (same bits, run to run) and
**statistical reproducibility** (same conclusion, across seeds, relative to measured
seed-to-seed spread). They have demonstrated neither, and this is the trap. A bit-identical
*loss curve* is not bitwise determinism: `[M]` five fresh-process fp32 runs on this machine
produced five different parameter digests while printing identical losses to ten decimal
places (§3.4e), so a matching loss curve is compatible with a materially different model.
What they have shown is that a scalar summary matched. Even if they had shown true bitwise
determinism, it would tell you their rig is stable and their regression tests will work, and
**nothing whatsoever** about whether the architecture change works — bitwise determinism sets
`σ²_numeric = 0` and leaves `σ²_init` and `σ²_data`, the dominant terms, untouched. Two
bit-identical single-seed arms are one anecdote each. Ask for the seed count, the confidence
interval, and whether the digest was taken on weights or on a printed metric; if the answer
is "one seed, but it's reproducible," they are in the dangerous quadrant of §2.1's table.

**2.** bf16 unit roundoff `u = 2⁻⁸ = 3.91e−3`. Sequential: `(n−1)·u·κ = 1,048,575 × 3.91e−3
= 4,096κ`. Pairwise: `(log₂ n)·u·κ = 20 × 3.91e−3 = 0.078κ`. Ratio `(n−1)/log₂ n ≈ 52,000×`.
Neither is useful because both exceed any tolerance you would set — a bound of "up to 8%
wrong" (let alone 4,096) tells you nothing about a kernel that in fact delivers `2e−3`.
What you should predict instead is the **input-rounding floor**: rounding to bf16 gives
relative error with RMS `u/√3 = 2.26e−3`, so any bf16 matmul reading rounded inputs cannot
do better than a few times `1e−3` regardless of accumulator quality. `[M]` Our configured
path measures `2.01e−3` — at the floor, i.e. the accumulation is contributing essentially
nothing, consistent with fp32 partials. The unconfigured path measures `5.60e−3`, 2.8×
above the floor, so *that* one is telling you about the accumulation. Worst-case analysis
tells you where measurement is required; it does not substitute for it.

**3.** With `n = 3`, `MDE ≈ 3.04 σ = 3.04 × 0.012 = 0.036 nats`. After making the runs
bitwise deterministic: **still ≈ 0.036 nats.** Determinism removes `σ²_numeric`, which is
the smallest of the three variance components and very likely under a percent of the total;
the MDE moves by an amount you could not measure. What *would* have helped: (a) more seeds —
`n = 10` gives `1.33σ = 0.016 nats`, a 2.3× improvement, at 3.3× the compute; or (b) a
**paired design**, running both arms at the same `seed_init` and `seed_data` and testing the
per-seed differences, where `MDE_paired ≈ 3.10 σ_D` and `σ_D` can be far below `σ`. Pairing
works under one condition: the two arms must share a meaningful initialization, which holds
for a memory-policy ablation on a fixed checkpoint (`mnemosyne-h2o` vs `mnemosyne-window`)
and only partially for an architecture ablation (`proteus-swa-4to1` vs `proteus-dense`),
where the parameter tensors have different shapes and the same seed does not produce matched
inits.

**4.** `[M]` The relative L2 error of a length-1,048,576 bf16 weighted sum against an fp64
reference changes from **2.01e−3 (configured) to 5.60e−3 (unconfigured)** — a factor of
**2.79** — with the same CPU reference in both cases, reproduced across three seeds and in
fresh processes (2026-07-26, `scripts/measure_bf16_reduction_error.py`,
`torch 2.12.0a0+rocm7.13.0a20260313`). The variables select a different BLAS implementation,
hence a different tiling and split-k factor, hence a different reduction tree; §3.1 shows the
reduction tree is worth up to five orders of magnitude of error bound, so a 2.8× realized
difference is unremarkable *as arithmetic* and devastating *as methodology*. It is stronger
than "GPUs are nondeterministic" because it is not about nondeterminism at all: both
configurations are internally repeatable. It says the **environment is an experimental
variable**, that it is currently unrecorded in every run record this lab and most published
work produces, and therefore that "same seed" is not a sufficient description of an
experimental condition. Reproducible builds solved this decades ago by putting the toolchain
in the hash; ML has not.

**5.** The sets differ whenever the boundary gap `g = s_(k) − s_(k+1)` is smaller than twice
the numeric error `ε` in the scores. `[M]` `ε ≈ 2e−3` relative with hipBLASLt configured and
`5.6e−3` without, so tokens within roughly 0.2% of each other in accumulated mass are
effectively tied and their ordering is decided by the kernel. It is a design problem rather
than a rounding curiosity for two reasons. **First, `top-k` is discontinuous**: unlike a
logit, where an `ε` perturbation produces an `ε` change downstream, an `ε` perturbation here
produces an `O(1)` change — a token is either in the cache or destroyed, and if it is
destroyed the information is gone, not degraded. **Second, `ε` is not a constant of the
model — it moves with the environment** (`[M]` 2.8× between two hipBLASLt configurations)
**and with run-to-run variation** (`[M]` 0/5 fresh-process reproducibility for fp32 training,
§3.4e), so the retained set is a function of things no experiment record currently captures.
The batch-composition version of this — two identical requests served in different batches
getting different caches — is the sharpest form and is **not** currently supported on our
hardware: `[M]` §3.4(b) found a bf16 GEMM batch-invariant 7/7. Credit for saying so; the
first two mechanisms stand without it. The mitigations are a deterministic tie-break (by
position, stated in the policy spec) and a boundary-margin telemetry channel reporting `g`
and the fraction of retained tokens within `2ε` of an evicted one.

**6.** Three, with honest labels:
- **Collectives are unavailable** — `ASSUMPTIONS.md: single-device-only`, `[C]`
  `torch._C._distributed_c10d` incomplete on gfx1151. **Software gap** (and irrelevant to us
  while we are single-device, but it blocks reproducing any published multi-GPU determinism
  result).
- **No batch-invariant kernel path exists for ROCm** — vLLM's implementation branches on
  `is_cuda()` / `is_xpu()` only
  (`memory/vllm/vllm/model_executor/layers/batch_invariant.py:914`) and the feature flag is
  documented as requiring CC ≥ 9.0 (`memory/vllm/vllm/envs.py:601`). **Software gap**, with a
  possible C++-level foothold in ROCm 6.3's deterministic rocPRIM primitives that nobody has
  wired to PyTorch.
- **GPU fp32 training is measurably nondeterministic and we do not know why** — `[M]` 0/5
  fresh processes reproduced (§3.4e). This is now *measured*, not unknown; what is unknown is
  the **cause**, and §3.4(d) rules out the easy explanation by showing that embedding
  backward, `scatter_add` and `index_add` all have deterministic implementations available on
  this backend. **Software gap of unidentified location**, and Exercise C's bisection names
  the tensor in an evening.
- (Bonus, and it belongs on the list) **bf16 numerics are unproven** —
  `ASSUMPTIONS.md: bf16-numerics-unproven`, still `untested`, against `[C]` five documented
  bf16 bugs on gfx1151. **Unmeasured unknown**, and it undercuts every other item, because
  every measurement above was taken in the arithmetic under suspicion.

Note what is *not* on this list any more. "We have never run
`torch.use_deterministic_algorithms(True)`" was true when this module was drafted; §3.4(d)
ran it, and the two ops that raise (`grid_sampler_2d_backward`, `max_pool3d` backward) appear
in vision models, not in ours. **The op-level blocker people expect does not exist here.**

**7.** They are not in tension; they localize the problem. A single forward GEMM is one
kernel launch with one deterministic algorithm selection, and it reproduced 3/3 in each
environment. Twenty training steps add the backward pass — where the atomics live — and the
optimizer, and reproduced 0/5. So **the nondeterminism is downstream of the forward GEMM**:
in a backward kernel, in an in-place optimizer update, or in the reduction inside grad-norm
or loss accumulation. The loss agreeing to ten decimal places is not counter-evidence; it is
the expected shape, because the loss is a scalar mean over 512 tokens and averages away a
last-bit perturbation that is nevertheless *present and compounding* in 468,224 parameters.
The single localizing measurement is **a per-tensor digest after each stage of step 1 across
two fresh processes** — after init, after forward, after `backward()` (digesting each
`p.grad` separately so a mismatch names the parameter), after `opt.step()`. The first stage
at which any digest differs, and the name of the first tensor that differs, is the answer.
That is Exercise C1, and `[A]` medium confidence the answer is the embedding gradient —
which, given §3.4(d), would mean PyTorch has a deterministic implementation for it and is
not choosing it by default.

---

## Sources

`[C]` arXiv ids below were surfaced by literature search on 2026-07-26; resolution proves the
paper exists, not that it supports the claim beside it. Where an id could not be confirmed it
is cited by title and venue instead.

**Determinism and batch invariance.** Thinking Machines Lab, "Defeating Nondeterminism in LLM
Inference" (thinkingmachines.ai/blog, 11 Sep 2025) — industry blog with released code, not
peer-reviewed; cited as such, and cited *by upstream source* at
`memory/flashinfer/flashinfer/decode.py:1317`. LMSYS Org, "Towards Deterministic Inference in
SGLang and Reproducible RL Training" (lmsys.org/blog, 22 Sep 2025) — the ~61.5% → ~34.35%
overhead figures and the 100%-reproducible RL training claim. arXiv 2506.09501 —
Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference (Jun 2025,
rev. Oct 2025; code at github.com/nanomaoli/llm_reproducibility) — reports bf16 as markedly
worse than fp16/fp32 for reproducibility. arXiv 2601.21824 — DASH: Deterministic Attention
Scheduling for High-throughput Reproducible LLM Training (29 Jan 2026) — the up-to-37.9%
deterministic-backward throughput cost and a 1.28× recovery. arXiv 2511.17826 — Deterministic
Inference across Tensor Parallel Sizes That Eliminates Training–Inference Mismatch (Nov 2025).
arXiv 2601.17768 — LLM-42: Enabling Determinism in LLM Inference with Verified Speculation
(Jan 2026). arXiv 2606.03019 — Reproducibility is the New Copyleft: Defining AGI-oriented
Reproducible Builds (Jun 2026).

**Seed variance and experimental design.** arXiv 2002.06305 — Fine-Tuning Pretrained Language
Models: Weight Initializations, Data Orders, and Early Stopping (Feb 2020). arXiv 2503.07329 —
Assessing the Macro and Micro Effects of Random Seeds on Fine-Tuning Large Language Models
(Mar 2025). arXiv 2504.07086 — A Sober Look at Progress in Language Model Reasoning: Pitfalls
and Paths to Reproducibility (Apr 2025).

**Numerics background.** Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
ch. 4 — the sequential and pairwise summation bounds and the condition number `κ` used in
§3.1; standard textbook material, cited rather than derived. arXiv 1710.03740 — Mixed
Precision Training (Oct 2017), for the fp32-master-weights argument this module assumes.

**Platform.** ROCm 6.3.0 release notes — `rocprim::deterministic_*` scans and
`thrust::hip::par_det`. ROCm issue #6022 (hipBLASLt configuration; WSL2 VRAM mapping) and
#6034 (gfx1151 bf16 bugs), both `[C]` via `CLAUDE.md`.

**Local — this module's own measurements.** §3.4(a)–(e) were produced for this module on
2026-07-26 by the fresh-subprocess probe harness whose two halves appear inline in Exercise A
and Exercise C. That harness is **not yet committed**: per `CLAUDE.md → Engineering
conventions` a one-off analysis script is exempt from TDD only while it stays one-off, and
on reuse it migrates into the rig and acquires tests. It will be reused — Exercise C is
literally "run it again with a bisection" — so the first person to run Exercise C should land
it under `themis/` with tests rather than re-paste it. Twenty-six fresh subprocesses total:
10 GEMM (4 bf16 @ 4096³, 6 fp32 @ 2048³), 4 batch-invariance/reduction-shape, 1
deterministic-algorithms sweep, 11 training (5 GPU fp32, 4 GPU bf16, 2 CPU fp32).

**Local — prior.** `ASSUMPTIONS.md` — `hipblaslt-config` (`[M]` 2.01e−3 vs 5.60e−3, ~2.8×),
`bf16-reduced-precision-knob-works` (refuted, zero-bit change),
`sdpa-is-memory-efficient` (refuted by default; 147.2 vs 6.6 bytes/T²),
`large-tensor-fault-32gib` (`[M]` silent hang at 0% CPU), `gpu-fast-tier-size`
(`[M]` ≥62 GiB at ~200 GB/s), `gemm-throughput-below-reference` (`[M]` 20.9 TFLOPS bf16 at
8192³), `bf16-numerics-unproven` (untested), `single-device-only`.
`scripts/measure_bf16_reduction_error.py` (the N=1,048,576 / d=128 / 3-seed harness).
`scripts/measure_attention_memory_path.py`. `curriculum/the-training-loop.md` §3.6–§3.8.
`curriculum/tensors-and-autograd.md` §2.4 (`[M]` fp32 gradients match CPU to 3.9e−8
absolute, ~1.1e−6 relative — one seed, 2-layer 64-wide GPT, an anecdote by the house
standard but the only gradient-correctness evidence this machine has).
`curriculum/long-context-and-effective-context.md` Exercise B and its same-day correction.
`curriculum/paged-attention-and-prefix-reuse.md`. `research/memory/open-problems-ranked.md`.
`research/notes/pretraining-recipes.md` §5, §9. `ENVIRONMENT.md`.

---

## Decision / Riskiest assumption / Next test

**Decision.** Do not pursue bitwise-deterministic training as a goal. Pursue **environment
determinism** instead: every Themis run record carries a numerics fingerprint
(`hipblaslt_configured`, `aotriton_experimental`, wheel, HIP, driver, autocast dtype,
`matmul_fp32_precision`, plus a SHA-256 over the sorted pairs), the rig refuses to aggregate
runs whose fingerprints differ, and memory-policy ablations are run **paired** — same
`seed_init`, same `seed_data`, arms differing only in the policy — so the 3-seed budget buys
a 3.1·σ_D threshold instead of a 3.0·σ one. Keep bitwise determinism as a requirement in
exactly two places: the checkpoint round-trip (on weights, not loss) and rig regression
tests.

**One amendment forced by this module's own measurements.** `[M]` GPU fp32 training does not
reproduce bitwise (0/5) while GPU bf16 autocast does (4/4). Pairing therefore currently
*works better in bf16 than in fp32* on this machine, which inverts the reflex to "run the
careful comparison in fp32." Until Exercise C localizes the fp32 nondeterminism, paired
memory-policy ablations should run under **bf16 autocast**, and any regression test that
compares weights bit-exactly must be written against the bf16 path or against CPU. Note the
awkwardness honestly: that recommendation rests on 4 samples and on
`ASSUMPTIONS.md: bf16-numerics-unproven` still being `untested`.

**Riskiest assumption.** That `σ²_numeric` really is small relative to `σ²_init + σ²_data`
at 20M–300M on this hardware. Every argument in §3.5–3.6 rests on it, it is `[A]` inherited
from the fine-tuning literature, and nothing in this lab has measured it. If numeric variance
turns out comparable to seed variance at our scale — plausible given `[M]` a 2.8×
environment-dependent error swing and `[C]` five documented bf16 bugs — then paired designs
are invalid and the environment fingerprint is insufficient. The reflexive fix ("move to
fp32") is now itself suspect: `[M]` fp32 is the arm that failed to reproduce here. The
cheapest thing that would move this assumption is the nine-run variance decomposition in §8 —
three runs varying only `seed_init`, three varying only `seed_data`, three varying neither —
which measures all three components directly instead of assuming their ordering.

**Next test.** Exercise C1, the per-tensor bisection, in one evening. §3.4 already converted
the Hardware Validation Gate's determinism item from `untested` to `[M]` and produced the
gfx1151 deterministic-op coverage map; what it did not do is explain why five identically
seeded fp32 runs produce five different models while a single fp32 GEMM produces one. That
question has a one-evening answer, the answer names a tensor, and the tensor decides whether
the paired-design strategy above is implementable in fp32 or only in bf16.
