# ASSUMPTIONS — running ledger (G2)

A **register** (see CLAUDE.md → DOCUMENT CLASSES): rows are appended and their
**status** is updated; rows are never deleted. Every assumption baked into the
kickoff starts here. Update at every Definition of Done.

Status ∈ `untested` / `supported` / `refuted`. Evidence tag ∈ `[M]` measured
(run ID + seed count or preflight), `[C]` cited (arXiv/URL + date), `[A]` assumed
(state confidence + the cheapest test that would move it). Never state an `[A]` in
the register of an `[M]`.

| # | Assumption | Status | Evidence | Date |
|---|---|---|---|---|
| hardware-capacity-ceiling | The gfx1151 UMA pool yields a usable single-allocation ceiling near the BIOS UMA FB Size (target 96 GB), and that number bounds every long-context / KV-capacity experiment. | **untested** | `[M]` preflight 2026-07-24 could **not** measure it — torch is a CUDA wheel and sees no AMD device. This is THE number to fill. Cheapest test: install gfx1151 torch, re-run `scripts/preflight.ps1` capacity probe. | 2026-07-24 |
| torch-build | The lab's venv will run AMD gfx1151 ROCm nightly torch; the system Python's `torch 2.11.0+cu128` (CUDA) is irrelevant to the lab and will not be used. | untested | `[M]` ENVIRONMENT.md 2026-07-24: system torch is `+cu128`, `torch.cuda.is_available()==False`. `[A]` high confidence the venv path works. Test: `pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch` into a clean space-free venv, confirm `get_device_name`==Radeon 8060S. | 2026-07-24 |
| native-windows-over-wsl2 | Native Windows is the correct ROCm target; WSL2 clamps the ROCm pool to the `.wslconfig` memory value and cannot reach dedicated VRAM. | untested (locally) | `[C]` ROCm issue #6022 (librocdxg VRAM mapping under WSL2). Decisive local test: allocate > `.wslconfig` limit and check whether it reaches the dedicated pool. | 2026-07-24 |
| bf16-numerics-unproven | bf16 on gfx1151 is numerically trustworthy for matmul / softmax / RMSNorm / attention at the shapes we use. | untested | `[C]` five critical bf16 bugs documented on gfx1151 (ROCm #6034 / amdsense). Test: Hardware Validation Gate numerics suite (bf16 vs fp32 reference, max abs/rel error). | 2026-07-24 |
| single-device-only | Distributed collectives are incomplete on gfx1151, so all training is single-device; FSDP/DDP/TP/PP/EP are design-only until rented hardware validates them. | supported (as constraint) | `[C]` `torch._C._distributed_c10d` incomplete on gfx1151 (CLAUDE.md, ROCm docs). | 2026-07-24 |
| hipblaslt-config | GEMM throughput requires `HIPBLASLT_TENSILE_LIBPATH` set and `TORCH_BLAS_PREFER_HIPBLASLT=1`; a bad path drops GEMM ~5x (≈33→≈6 TFLOPS). | untested | `[C]` ROCm #6022. `[M]` preflight 2026-07-24: currently unset. Test: set env, benchmark GEMM at 8192³. | 2026-07-24 |
| z13-is-right-instrument | A capacity/bandwidth-bound research agenda (memory systems) is better served by 128 GB unified memory than by a 20 GB discrete card, despite slower wall-clock. | supported | `[C]` 93-experiment campaign on this silicon measured 25% MFU vs 7.7% on an RTX 4090 on a memory-bound recipe (ROCm #6034, Mar 2026). | 2026-07-24 |
| ablation-scale-sufficient | 20M–300M params on 0.5–5B tokens is enough to answer the memory-systems questions we care about (recall cliffs, eviction/recall tradeoffs, hybrid ratios). | untested | `[A]` medium confidence, from kickoff design. Test: the Ablation Backlog's first hypothesis producing a decision-changing result at this scale. | 2026-07-24 |
| cloud-budget-zero | Starting `CLOUD_BUDGET = $0`; all work runs on the Z13, renting H100 hours only when a specific result justifies it. | supported | `[A]` user-set 2026-07-24. Contribution-margin (info/$) ranks any future spend (see G3 translation). | 2026-07-24 |
| mnemosyne-separable | Mnemosyne (memory subsystem) is mechanically separable from Proteus — it never imports proteus/themis — which is what makes it a contribution rather than an implementation detail. | supported (mechanism in place) | `[C]` enforced by `packages/mnemosyne/pyproject.toml` (no proteus dep) + `tests/test_package_boundaries.py` lint contract, scaffolded 2026-07-24. `[M]` 2026-07-26: contract proved red-then-green — a transient `import proteus` under `packages/mnemosyne/src/mnemosyne/` fails the test naming the offending file; removing it passes. The guard is real, not decorative. Acceptance test (clean-venv wheel) still runs at the mnemosyne-core milestone. | 2026-07-26 |
| reference-model | Poolside Laguna S 2.1 (118B-A8.5B MoE, mixed SWA/global attention, OpenMDW-1.1) is the primary reference architecture under study. | supported | `[C]` model card / configs (to be fetched in Reference Library phase). | 2026-07-24 |
| gfx1151-windows-wheels-exist | AMD ships gfx1151 ROCm nightly PyTorch wheels for native Windows via the rocm.nightlies index. | untested (install not yet run) | `[C]` CLAUDE.md hardware section; AMD TheRock nightly. Test: the torch-build install above. | 2026-07-24 |
| toolchain-present | git, gh (authenticated), Python 3.12, uv are installed and adequate for scaffold/design work. | supported | `[M]` preflight 2026-07-24: all green. | 2026-07-24 |

## Open blockers derived from the above

- **`hardware-capacity-ceiling` is unmeasured.** It gates every long-context experiment and is the flagship `[M]` this project owes. It cannot be filled until `torch-build` is done (gfx1151 nightlies installed into a clean venv). Neither blocks Phases 0–5 (scaffold → rig design), which need no GPU.
