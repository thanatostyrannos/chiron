# We will not enable AOTriton attention by default, despite an 18x activation-memory win

Status:   Accepted
Date:     2026-07-26
Deciders: Founder (owner), Claude (research staff)

## Context

`F.scaled_dot_product_attention` on gfx1151 does not take a memory-efficient path
unless `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` is set. Measured 2026-07-26 with
`scripts/measure_attention_memory_path.py` on torch `2.12.0a0+rocm7.13.0a20260313`:

| Configuration | Retained activation | Saved at T=512, B=4, nh=8 |
|---|---|---|
| default | **147.2 bytes/T²** | 38.0 MiB |
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` | **6.6 bytes/T²** | 2.1 MiB |

`torch.backends.cuda.flash_sdp_enabled()` returns `True` in both cases — it reports
what is *permitted*, not what ran. The only honest signal is a stderr `UserWarning`.

The consequence is not merely wasted memory. By default the score tensor grows as
`B·nh·T²` and reaches `large-tensor-fault-32gib`, whose failure mode on this machine is
a **silent hang at 0 CPU seconds** rather than an OOM. Every long-context arm in the
prospective backlog was pointed at that.

Against enabling it: the flag is marked experimental by AMD, and an attention kernel
swap is a **numerics change**. `bf16-numerics-unproven` is untested and the Hardware
Validation Gate has not run, so we have no baseline against which to detect a
regression the flag might introduce. Turning it on now would silently alter every
result this machine has produced or will produce, before anything has compared the two.

The 18x also closely matches the ~19x undocumented AOTriton attention speedup `[C]`
flagged in CLAUDE.md as unverified, which is suggestive but not established.

## Decision

We will leave `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` **unset** in
`scripts/activate-lab.ps1`, and document it there as a deliberate non-default with the
measurements and this reasoning inline.

The Hardware Validation Gate will run its **numerics suite in both configurations** and
report them side by side. Until that comparison exists, any experiment that needs the
memory-efficient path sets the variable per-run and **records it in that run's config**,
so no result is ever ambiguous about which attention kernel produced it.

The owner decides whether it becomes a default, on the evidence of that comparison.

## Consequences

**Makes easy.** Every result to date remains interpretable under one kernel. When the
gate runs, we get a real bf16-vs-fp32 comparison across both kernels rather than a
single number of unknown provenance — which is strictly more information than starting
with the flag on.

**Makes hard.** Long-context work is effectively blocked until the gate runs, because
without the flag the score tensor walks into a silent hang. Anyone doing exploratory
long-context work in the meantime must set the variable manually and knows their numbers
are not comparable to the default-configuration baseline.

**Forecloses.** Nothing permanently. If the gate shows the kernels agree numerically,
this ADR is superseded and the flag becomes a default — that is the expected outcome and
the reason the comparison is specified rather than deferred.

**Debt taken.** Two attention configurations must be carried through the gate instead of
one, roughly doubling that suite's runtime. Repayment trigger: the gate's both-ways
comparison lands and a successor ADR fixes the default.
