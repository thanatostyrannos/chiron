# Activate the Chiron lab environment for the current PowerShell session.
#
# The lab venv lives OUTSIDE the repo at a space-free path: paths containing spaces
# are known-broken on the gfx1151 pip/ROCm stack, and C:\Users\<name>\... is a coin
# flip on that. Nothing here modifies system or user environment variables -- the
# settings are session-scoped so an experiment always states its own environment.
#
# Usage (dot-source it, so the variables survive the call):
#     . .\scripts\activate-lab.ps1

$LabVenv = 'C:\venvs\lab'

if (-not (Test-Path $LabVenv)) {
    Write-Error "Lab venv not found at $LabVenv. See ENVIRONMENT.md for the install."
    return
}

$env:PATH = "$LabVenv\Scripts;$env:PATH"

# hipBLASLt: a wrong or missing Tensile library path costs GEMM throughput silently
# (ROCm #6022). Validated by scripts/benchmark_gemm.py, not by the path existing.
$RocmLibs = "$LabVenv\Lib\site-packages\_rocm_sdk_libraries_gfx1151"
$env:HIPBLASLT_TENSILE_LIBPATH = "$RocmLibs\bin\hipblaslt\library"
$env:TORCH_BLAS_PREFER_HIPBLASLT = '1'

# DELIBERATELY NOT SET -- read before enabling:
#
#   $env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = '1'
#
# Without it, F.scaled_dot_product_attention on gfx1151 falls back to the math backend
# and RETAINS the B*nh*T*T score matrix: [M] 147.2 bytes/T^2 vs 6.6 with the flag, an
# ~18x reduction in saved activations at T=512 (scripts/measure_attention_memory_path.py,
# 2026-07-26). torch.backends.cuda.flash_sdp_enabled() returns True either way -- it
# reports what is permitted, not what ran.
#
# So long-context work is effectively impossible without it: the score tensor grows as
# B*nh*T^2 straight into the >=32 GiB single-tensor fault, which hangs silently at 0 CPU.
#
# It stays off here anyway, because the flag is marked experimental and therefore changes
# NUMERICS. Turning it on by default would silently alter every result on this machine
# before the Hardware Validation Gate has ever compared the two. The gate must run its
# numerics suite BOTH ways and the owner decides; until then, set it per-experiment and
# record it in the run's config. See ASSUMPTIONS.md -> sdpa-is-memory-efficient.

Write-Host "Chiron lab environment active:"
Write-Host "  python  $((Get-Command python).Source)"
Write-Host "  hipBLASLt Tensile path  $env:HIPBLASLT_TENSILE_LIBPATH"
