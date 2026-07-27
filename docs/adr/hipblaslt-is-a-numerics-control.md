# We will treat hipBLASLt configuration as a numerics control, not a throughput setting

Status:   Accepted
Date:     2026-07-26
Deciders: Founder (owner), Claude (research staff)

## Context

`HIPBLASLT_TENSILE_LIBPATH` and `TORCH_BLAS_PREFER_HIPBLASLT=1` entered this project as
a throughput concern. CLAUDE.md `[C]` cites ROCm #6022 for a ~5x GEMM cliff when the
Tensile library path is wrong. We measured that on 2026-07-26 and **the cliff did not
reproduce**: 18.6 → 20.9 TFLOPS bf16 at 8192³, a 12% gain. The row in `ASSUMPTIONS.md`
was updated to say the setting "still helps" and is "free — set it anyway".

That framing was wrong, and the correction came from an unexpected direction. A
curriculum module measured bf16 attention-weighted-sum error at N=1,048,576 and reported
the GPU 3x worse than the CPU reference. It did not reproduce for the parent agent, who
measured 1.09x — while both parties' **CPU** figures agreed to four significant figures.
Same reference, different GPU result, so the divergence was in the GPU path. An
environment matrix at N=1,048,576, seed 1337, identical fp64 reference (`1.8486e-3`):

| GPU path | relative error | vs CPU |
|---|---|---|
| hipBLASLt configured | `2.0079e-3` | **1.09x** |
| both variables unset | `5.6046e-3` | **3.03x** |

`5.6046e-3` is the module's own figure to five significant figures: it had been measured
without `scripts/activate-lab.ps1`. The effect was real and repeatable across seeds and
fresh processes; the *attribution* was wrong. It is not GPU-versus-CPU. It is
hipBLASLt-configured-versus-not.

Separately, `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction` changes
**exactly zero bits** in either configuration. It is inert on this stack.

This matters because long-reduction accuracy is not a peripheral property for this lab.
A memory-systems result at long context is a claim about what survives a long weighted
sum. A 2.8x error inflation in exactly that operation is confounding before any memory
policy is applied.

## Decision

We will treat hipBLASLt configuration as a **numerics variable of the instrument**, on
the same footing as dtype and kernel selection, and not as a performance option.

Concretely: `scripts/activate-lab.ps1` sets both variables and remains the only
supported way to enter the lab environment. Every recorded run states whether hipBLASLt
was configured. The Hardware Validation Gate pins it and reports numerics with it set.
Any result produced without it is marked confounded and is not comparable to one
produced with it.

`allow_bf16_reduced_precision_reduction` is banned as an experimental axis: it is
recorded as inert, and using it would suggest a control we do not have.

## Consequences

**Makes easy.** One documented environment, one numerics baseline, and a named reason
that survives the people who found it. The `activate-lab.ps1` habit is now load-bearing
rather than a convenience, which makes it likelier to be followed.

**Makes hard.** Results from before this decision must be audited for whether the
environment was active. In practice that is a small set, but it is not zero — and the
finding itself came from a module that had not used the script.

**Forecloses.** Comparing our absolute error figures against any published gfx1151
number that does not state its BLAS configuration. That comparison was never sound; we
now know why.

**Debt taken.** The mechanism is uncharacterised — we know the configuration changes the
result, not which kernel or accumulation order is responsible, and we have not swept the
131K→1M transition where the effect appears. Repayment trigger: the Hardware Validation
Gate's numerics suite, which should localise it or explicitly record it as open.
