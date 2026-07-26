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

Write-Host "Chiron lab environment active:"
Write-Host "  python  $((Get-Command python).Source)"
Write-Host "  hipBLASLt Tensile path  $env:HIPBLASLT_TENSILE_LIBPATH"
